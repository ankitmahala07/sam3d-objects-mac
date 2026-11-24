# Copyright (c) Meta Platforms, Inc. and affiliates.
import math
import os
import torch
import torch.nn.functional as F


def block_diag_attn_mask(q_seqlens, kv_seqlens, device=None, dtype=torch.float32):
    """
    Create an additive attention mask for block-diagonal attention.
    The result is shape [sum_q, sum_kv], with 0.0 in the valid
    region(s) and -inf elsewhere.
    """
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    # Start with everything "masked out"
    attn_mask = torch.full(
        (total_q, total_kv), float("-inf"), device=device, dtype=dtype
    )

    q_start = 0
    kv_start = 0
    for q_len, kv_len in zip(q_seqlens, kv_seqlens):
        attn_mask[q_start : q_start + q_len, kv_start : kv_start + kv_len] = 0
        q_start += q_len
        kv_start += kv_len

    return attn_mask


def mps_should_chunk(q):
    return q.device.type == "mps" and os.environ.get("SAM3D_MPS_USE_SDPA", "0") != "1"


def mps_chunk_size():
    raw = os.environ.get("SAM3D_MPS_ATTN_CHUNK", "128")
    try:
        return max(1, int(raw))
    except ValueError:
        return 128


def chunked_sdpa_bhlc(q, k, v, chunk_size=None):
    """Scaled dot-product attention for [B, H, L, C] tensors without MPS SDPA."""
    chunk_size = mps_chunk_size() if chunk_size is None else chunk_size
    scale_factor = 1 / math.sqrt(q.size(-1))
    out_dtype = v.dtype
    k_t = k.float().transpose(-2, -1).contiguous()
    v = v.float()
    out_chunks = []
    for start in range(0, q.shape[-2], chunk_size):
        q_chunk = q[..., start : start + chunk_size, :].float()
        attn = torch.matmul(q_chunk, k_t) * scale_factor
        attn = torch.softmax(attn, dim=-1)
        out_chunks.append(torch.matmul(attn, v).to(out_dtype))
    return torch.cat(out_chunks, dim=-2)


def chunked_sdpa_lhc(q, k, v, chunk_size=None):
    """Scaled dot-product attention for [L, H, C] tensors without MPS SDPA."""
    out = chunked_sdpa_bhlc(
        q.permute(1, 0, 2).unsqueeze(0),
        k.permute(1, 0, 2).unsqueeze(0),
        v.permute(1, 0, 2).unsqueeze(0),
        chunk_size=chunk_size,
    )
    return out.squeeze(0).permute(1, 0, 2)


def masked_chunked_sdpa(q, k, v, q_seqlen, kv_seqlen):
    q_start = 0
    kv_start = 0
    out_chunks = []
    for q_len, kv_len in zip(q_seqlen, kv_seqlen):
        q_end = q_start + q_len
        kv_end = kv_start + kv_len
        out_chunks.append(
            chunked_sdpa_lhc(
                q[0, q_start:q_end],
                k[0, kv_start:kv_end],
                v[0, kv_start:kv_end],
            )
        )
        q_start = q_end
        kv_start = kv_end
    return torch.cat(out_chunks, dim=0)


# GenIA port guard: on MPS the dense path below hands F.sdpa a block-diag
# mask, and the MPS SDPA math backend materialises (H x sum_q x sum_kv)
# attention weights; past MPS tensor indexing limits (~2^31 elements, and
# well before that in memory) it raises/OOMs. GenIA guided stage-2 GS
# decode hit H=16 x 19350 x 19350 = 5.99e9. The chunked path computes the
# identical per-block math without materialising either tensor.
_MPS_SDPA_WEIGHTS_MAX = 250_000_000


def masked_sdpa(q, k, v, q_seqlen, kv_seqlen):
    """
    Mimic xFormers' memory_efficient_attention using PyTorch 2.0 scaled_dot_product_attention.
    """
    if mps_should_chunk(q):
        return masked_chunked_sdpa(q, k, v, q_seqlen, kv_seqlen)
    if q.device.type == "mps" and \
            q.shape[2] * sum(q_seqlen) * sum(kv_seqlen) > _MPS_SDPA_WEIGHTS_MAX:
        return masked_chunked_sdpa(q, k, v, q_seqlen, kv_seqlen)

    # Build the block-diagonal additive mask
    # shape: [sum_q_len, sum_kv_len] with 0 where allowed, -inf where masked
    attn_mask_2d = block_diag_attn_mask(
        q_seqlen, kv_seqlen, device=q.device, dtype=q.dtype
    )

    # PyTorch’s scaled_dot_product_attention expects a mask broadcastable to
    # [batch_size, n_heads, q_len, kv_len]. For a single batch, single head:
    attn_mask_4d = attn_mask_2d.unsqueeze(0).unsqueeze(0)
    q = q.permute(0, 2, 1, 3)  # [N, H, L, C]
    k = k.permute(0, 2, 1, 3)  # [N, H, L, C]
    v = v.permute(0, 2, 1, 3)  # [N, H, L, C]

    # Now call PyTorch 2.0’s built-in SDPA
    # By default, it will automatically apply the "1/sqrt(dim)" scaling internally.
    out = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask_4d,  # Additive mask
        dropout_p=0.0,  # or whatever dropout you need
        is_causal=False,  # True if you want a causal (triangular) mask
    )
    # out is shape [1, sum_q_len, dim]
    out = out.permute(0, 2, 1, 3)

    return out[0]


def _is_stacked_bias(attn_bias: torch.Tensor, N: int) -> bool:
    """Detect ``(N, H, L_per_view, P_per_view)`` stacked layout vs the
    legacy ``(1, H, L_or_sum_q, P_or_sum_kv)`` packed/template layout.

    The two shapes are indistinguishable when ``N == 1`` (both have
    ``shape[0] == 1``); we conservatively treat that case as packed,
    matching the original behaviour. Callers that build single-view
    stacked biases get the same result either way (one block).
    """
    return attn_bias.shape[0] == N and attn_bias.shape[0] != 1


def _slice_view_bias(
    attn_bias: torch.Tensor,
    i: int,
    q_ofs: int,
    kv_ofs: int,
    q_len: int,
    kv_len: int,
    *,
    stacked: bool,
) -> torch.Tensor:
    """Return the bias slice corresponding to view ``i`` in the per-view
    loop. ``stacked`` selects between the two storage layouts; see
    :func:`_is_stacked_bias`."""
    if stacked:
        return attn_bias[i:i + 1, :, :q_len, :kv_len]
    return attn_bias[:, :, q_ofs:q_ofs + q_len, kv_ofs:kv_ofs + kv_len]


def _sdpa_with_bias_bhlc(q, k, v, bias):
    """SDPA for [B, H, L, C] tensors with an additive [B, H, Lq, Lkv] bias.

    On MPS without SAM3D_MPS_USE_SDPA, mirrors chunked_sdpa_bhlc's fp32 math
    path (chunked over the query dim, bias added to the scores); elsewhere a
    single F.scaled_dot_product_attention with the bias as attn_mask.
    """
    if mps_should_chunk(q):
        chunk_size = mps_chunk_size()
        scale_factor = 1 / math.sqrt(q.size(-1))
        out_dtype = v.dtype
        k_t = k.float().transpose(-2, -1).contiguous()
        v_f = v.float()
        bias_f = bias.float()
        out_chunks = []
        for start in range(0, q.shape[-2], chunk_size):
            q_chunk = q[..., start:start + chunk_size, :].float()
            attn = torch.matmul(q_chunk, k_t) * scale_factor
            attn = attn + bias_f[..., start:start + chunk_size, :]
            attn = torch.softmax(attn, dim=-1)
            out_chunks.append(torch.matmul(attn, v_f).to(out_dtype))
        return torch.cat(out_chunks, dim=-2)
    return F.scaled_dot_product_attention(
        query=q, key=k, value=v, attn_mask=bias, dropout_p=0.0, is_causal=False
    )


def sdpa_blockdiag_with_bias(q, k, v, q_seqlen, kv_seqlen, attn_bias):
    """SDPA per view-block with additive bias (MPS port of GenIA's
    ``xops_blockdiag_with_bias``, which streams the same loop through the
    xformers cutlass kernel).

    Args:
        q, k, v: ``(1, sum_L, H, D)`` packed tensors, the layout the
                 xformers ``BlockDiagonalMask`` path expects.
        q_seqlen / kv_seqlen: per-view query / key lengths.
        attn_bias: additive bias, ``(N, 1 or H, L_per_view, P_per_view)``
            stacked per-view storage, or ``(1, 1 or H, sum_q, sum_kv)``
            packed dense form; see :func:`_is_stacked_bias`.

    Returns:
        ``(sum_q, H, D)`` - same layout as :func:`masked_sdpa`.
    """
    target_dtype = q.dtype
    H = q.shape[-2]
    N = len(q_seqlen)
    stacked = _is_stacked_bias(attn_bias, N)
    outs = []
    q_ofs = 0
    kv_ofs = 0
    for i, (q_len, kv_len) in enumerate(zip(q_seqlen, kv_seqlen)):
        qi = q[:, q_ofs:q_ofs + q_len].permute(0, 2, 1, 3)
        ki = k[:, kv_ofs:kv_ofs + kv_len].permute(0, 2, 1, 3)
        vi = v[:, kv_ofs:kv_ofs + kv_len].permute(0, 2, 1, 3)
        bi = _slice_view_bias(
            attn_bias, i, q_ofs, kv_ofs, q_len, kv_len, stacked=stacked
        ).to(target_dtype)
        if bi.shape[1] == 1 and H > 1:
            bi = bi.expand(-1, H, -1, -1).contiguous()
        out_i = _sdpa_with_bias_bhlc(qi, ki, vi, bi)
        outs.append(out_i.permute(0, 2, 1, 3)[0])
        q_ofs += q_len
        kv_ofs += kv_len
    return torch.cat(outs, dim=0)
