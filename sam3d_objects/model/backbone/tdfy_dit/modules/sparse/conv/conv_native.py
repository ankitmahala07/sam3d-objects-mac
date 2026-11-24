# Copyright (c) Meta Platforms, Inc. and affiliates.
# Pure-PyTorch implementations of SubMConv3d / SparseConv3d / SparseInverseConv3d
# for the "native" backend (CPU / MPS, no spconv / CUDA required).
#
# Weight layout mirrors spconv: [K, K, K, in_channels, out_channels]
# Coordinate layout: [N, 4] — (batch, z, y, x)
#
# Two implementations live side by side:
#   * "orig" — the original reference implementations (Python dict + per-voxel
#     loops). Correct, very slow.
#   * "vec"  — vectorized implementations (added 2026-11-22, alt3d). Neighbor
#     indices are computed with sorted int64 coordinate keys + searchsorted
#     (no per-voxel Python loops), cached by a content hash of the coords, and
#     the conv math is gather + per-offset matmul + index_add_ on the feature
#     device (MPS). Verified output-equivalent to "orig" (see
#     verify_conv_vec.py and ~/alt3d/LOG.md).
#
# Select with env SAM3D_CONV_IMPL=orig|vec (default: vec).

from __future__ import annotations
import hashlib
import os
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .native_backend import NativeSparseData

# Lazy import to avoid circular dependency (sparse.__init__ imports conv.__init__)
def _get_SparseTensor():
    from ..basic import SparseTensor
    return SparseTensor

def _get_DEBUG():
    from .. import DEBUG
    return DEBUG

__all__ = ["SparseConv3d", "SparseInverseConv3d"]


def _conv_impl() -> str:
    v = os.environ.get("SAM3D_CONV_IMPL", "vec").strip().lower()
    return v if v in ("orig", "vec") else "vec"


# ---------------------------------------------------------------------------
# Original helpers (reference implementation, kept verbatim)
# ---------------------------------------------------------------------------

def _build_coord_map(indices: torch.Tensor) -> dict:
    """Build a dict (b, z, y, x) -> row_index (on CPU as plain Python ints)."""
    cpu = indices.cpu().numpy()
    return {(int(r[0]), int(r[1]), int(r[2]), int(r[3])): i for i, r in enumerate(cpu)}


def _subm_forward_orig(
    feats: torch.Tensor,       # [N, Cin]
    indices: torch.Tensor,     # [N, 4] int64
    weight: torch.Tensor,      # [Kz, Ky, Kx, Cin, Cout]
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Submanifold conv: output coords == input coords."""
    Kz, Ky, Kx, Cin, Cout = weight.shape
    N = feats.shape[0]
    out = torch.zeros(N, Cout, dtype=feats.dtype, device=feats.device)

    coord_map = _build_coord_map(indices)

    for kz in range(Kz):
        dz = kz - Kz // 2
        for ky in range(Ky):
            dy = ky - Ky // 2
            for kx in range(Kx):
                dx = kx - Kx // 2
                W = weight[kz, ky, kx]  # [Cin, Cout]

                nb_idx = []
                src_idx = []
                cpu_idx = indices.cpu()
                for i in range(N):
                    b = int(cpu_idx[i, 0])
                    nz = int(cpu_idx[i, 1]) + dz
                    ny = int(cpu_idx[i, 2]) + dy
                    nx = int(cpu_idx[i, 3]) + dx
                    j = coord_map.get((b, nz, ny, nx))
                    if j is not None:
                        nb_idx.append(j)
                        src_idx.append(i)

                if src_idx:
                    src_t = torch.tensor(src_idx, device=feats.device)
                    nb_t = torch.tensor(nb_idx, device=feats.device)
                    gathered = feats[nb_t]         # [M, Cin]
                    contrib = (gathered @ W).to(out.dtype)  # [M, Cout]
                    out.index_add_(0, src_t, contrib)

    if bias is not None:
        out = out + bias
    return out


def _sparse_forward_orig(
    feats: torch.Tensor,       # [N, Cin]
    indices: torch.Tensor,     # [N, 4] int64  (b,z,y,x)
    weight: torch.Tensor,      # [Kz, Ky, Kx, Cin, Cout]
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int, int],
    padding: int,
    spatial_shape: list,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Strided sparse conv.  Returns (out_feats, out_indices, out_spatial_shape).
    out_indices are int64 sorted by batch.
    """
    Kz, Ky, Kx, Cin, Cout = weight.shape
    sz, sy, sx = stride

    # Compute output coords: floor(in_coord / stride), for each voxel × kernel offset
    out_coord_set: set = set()
    cpu_idx = indices.cpu()

    for i in range(len(cpu_idx)):
        b = int(cpu_idx[i, 0])
        z = int(cpu_idx[i, 1])
        y = int(cpu_idx[i, 2])
        x = int(cpu_idx[i, 3])
        for kz in range(Kz):
            dz = kz - Kz // 2
            for ky in range(Ky):
                dy = ky - Ky // 2
                for kx in range(Kx):
                    dx = kx - Kx // 2
                    oz = (z + dz * sz) // sz  # simplified: stride conv output coord
                    oy = (y + dy * sy) // sy
                    ox = (x + dx * sx) // sx
                    Dout = (spatial_shape[0] + padding * 2 - Kz) // sz + 1 if spatial_shape else None
                    Hout = (spatial_shape[1] + padding * 2 - Ky) // sy + 1 if spatial_shape else None
                    Wout = (spatial_shape[2] + padding * 2 - Kx) // sx + 1 if spatial_shape else None
                    if Dout is not None and (oz < 0 or oz >= Dout): continue
                    if Hout is not None and (oy < 0 or oy >= Hout): continue
                    if Wout is not None and (ox < 0 or ox >= Wout): continue
                    out_coord_set.add((b, oz, oy, ox))

    # Sort output coords by (batch, z, y, x) for downstream layout invariant
    out_coords_list = sorted(out_coord_set)
    if not out_coords_list:
        empty_idx = torch.zeros((0, 4), dtype=torch.int64, device=feats.device)
        empty_f = torch.zeros((0, Cout), dtype=feats.dtype, device=feats.device)
        return empty_f, empty_idx, [0, 0, 0]

    out_idx_t = torch.tensor(out_coords_list, dtype=torch.int64, device=feats.device)
    Nout = out_idx_t.shape[0]
    out_feats = torch.zeros(Nout, Cout, dtype=feats.dtype, device=feats.device)

    out_map = {c: i for i, c in enumerate(out_coords_list)}
    coord_map = _build_coord_map(indices)

    for kz in range(Kz):
        dz = kz - Kz // 2
        for ky in range(Ky):
            dy = ky - Ky // 2
            for kx in range(Kx):
                dx = kx - Kx // 2
                W = weight[kz, ky, kx]  # [Cin, Cout]

                src_idx_list = []
                dst_idx_list = []

                for i in range(len(cpu_idx)):
                    b  = int(cpu_idx[i, 0])
                    iz = int(cpu_idx[i, 1])
                    iy = int(cpu_idx[i, 2])
                    ix_ = int(cpu_idx[i, 3])
                    oz = (iz + dz * sz) // sz
                    oy = (iy + dy * sy) // sy
                    ox = (ix_ + dx * sx) // sx
                    dst_key = (b, oz, oy, ox)
                    di = out_map.get(dst_key)
                    si = coord_map.get((b, iz, iy, ix_))
                    if di is not None and si is not None:
                        src_idx_list.append(si)
                        dst_idx_list.append(di)

                if src_idx_list:
                    src_t = torch.tensor(src_idx_list, device=feats.device)
                    dst_t = torch.tensor(dst_idx_list, device=feats.device)
                    contrib = (feats[src_t] @ W).to(out_feats.dtype)   # [M, Cout]
                    out_feats.index_add_(0, dst_t, contrib)

    if bias is not None:
        out_feats = out_feats + bias

    bs, zs, ys, xs = zip(*out_coords_list)
    out_spatial = [max(zs) + 1, max(ys) + 1, max(xs) + 1]

    return out_feats, out_idx_t, out_spatial


def _inverse_scatter_orig(
    feats: torch.Tensor,       # [N, Cin] fp32 (already .float()ed by caller)
    coarse_idx: torch.Tensor,  # [N, 4] int64
    weight: torch.Tensor,      # [Kz, Ky, Kx, Cout, Cin]
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Original Python-loop transposed-conv scatter, extracted verbatim from
    SparseInverseConv3d.forward. Returns (out_feats, fine_idx_t, fine_spatial).
    out_feats has bias applied (matching the original inline code).
    """
    Kz, Ky, Kx, Cout, Cin = weight.shape
    sz, sy, sx = stride

    # Collect fine output coords (all possible fine voxels)
    fine_coord_set: set = set()
    cpu_coarse = coarse_idx.cpu()
    for i in range(len(cpu_coarse)):
        b = int(cpu_coarse[i, 0])
        z = int(cpu_coarse[i, 1])
        y = int(cpu_coarse[i, 2])
        xc = int(cpu_coarse[i, 3])
        for kz in range(Kz):
            dz = kz - Kz // 2
            for ky in range(Ky):
                dy = ky - Ky // 2
                for kx in range(Kx):
                    dx = kx - Kx // 2
                    fz = z * sz + dz
                    fy = y * sy + dy
                    fx = xc * sx + dx
                    if fz >= 0 and fy >= 0 and fx >= 0:
                        fine_coord_set.add((b, fz, fy, fx))

    fine_coords_list = sorted(fine_coord_set)
    if not fine_coords_list:
        empty_f = torch.zeros((0, Cout), dtype=feats.dtype, device=feats.device)
        empty_idx = torch.zeros((0, 4), dtype=torch.int64, device=feats.device)
        return empty_f, empty_idx, [0, 0, 0]

    fine_idx_t = torch.tensor(fine_coords_list, dtype=torch.int64, device=feats.device)
    Nfine = fine_idx_t.shape[0]
    out_feats = torch.zeros(Nfine, Cout, dtype=feats.dtype, device=feats.device)

    coarse_map = _build_coord_map(coarse_idx)

    for kz in range(Kz):
        dz = kz - Kz // 2
        for ky in range(Ky):
            dy = ky - Ky // 2
            for kx in range(Kx):
                dx = kx - Kx // 2
                # weight[kz,ky,kx]: [Cout, Cin] — transpose of forward weight
                W = weight[kz, ky, kx]  # [Cout, Cin]

                src_list = []
                dst_list = []
                for fi, fc in enumerate(fine_coords_list):
                    b, fz, fy, fx = fc
                    cz = (fz - dz) // sz
                    cy = (fy - dy) // sy
                    cx_ = (fx - dx) // sx
                    if (fz - dz) % sz != 0: continue
                    if (fy - dy) % sy != 0: continue
                    if (fx - dx) % sx != 0: continue
                    si = coarse_map.get((b, cz, cy, cx_))
                    if si is not None:
                        src_list.append(si)
                        dst_list.append(fi)

                if src_list:
                    src_t = torch.tensor(src_list, device=feats.device)
                    dst_t = torch.tensor(dst_list, device=feats.device)
                    # feats[src] @ W.T  => [M, Cout]
                    contrib = (feats[src_t] @ W.T).to(out_feats.dtype)
                    out_feats.index_add_(0, dst_t, contrib)

    if bias is not None:
        out_feats = out_feats + bias.float()

    bs_, zs, ys, xs = zip(*fine_coords_list)
    fine_spatial = [max(zs) + 1, max(ys) + 1, max(xs) + 1]

    return out_feats, fine_idx_t, fine_spatial


# ---------------------------------------------------------------------------
# Vectorized implementations ("vec")
#
# All neighbor-index computation happens on CPU with int64 tensor ops
# (sort / searchsorted / unique) — no per-voxel Python loops. Results are
# cached, keyed by a blake2b content hash of the coords tensor plus the conv
# geometry, because stage-2 sampling reuses identical coords for every step.
# The feature math (gather, matmul, index_add_) runs on the feature device.
# ---------------------------------------------------------------------------

_PAIR_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_PAIR_CACHE_MAX = 24


def _cache_get(key: tuple) -> Optional[dict]:
    v = _PAIR_CACHE.get(key)
    if v is not None:
        _PAIR_CACHE.move_to_end(key)
    return v


def _cache_put(key: tuple, value: dict) -> None:
    _PAIR_CACHE[key] = value
    _PAIR_CACHE.move_to_end(key)
    while len(_PAIR_CACHE) > _PAIR_CACHE_MAX:
        _PAIR_CACHE.popitem(last=False)


def _coords_cache_key(coords_cpu: torch.Tensor, extra: tuple) -> tuple:
    h = hashlib.blake2b(
        coords_cpu.numpy().tobytes(), digest_size=16
    ).hexdigest()
    return (h, tuple(coords_cpu.shape)) + tuple(extra)


def _encode(coords: torch.Tensor, Rz: int, Ry: int, Rx: int) -> torch.Tensor:
    """Mixed-radix encode [N,4] int64 coords (b,z,y,x) -> [N] int64 keys.
    Injective iff 0 <= z < Rz, 0 <= y < Ry, 0 <= x < Rx (b is the top digit).
    Key order == lexicographic (b,z,y,x) order."""
    k = coords[:, 0]
    k = k * Rz + coords[:, 1]
    k = k * Ry + coords[:, 2]
    k = k * Rx + coords[:, 3]
    return k


def _decode(keys: torch.Tensor, Rz: int, Ry: int, Rx: int) -> torch.Tensor:
    x = keys % Rx
    keys = keys // Rx
    y = keys % Ry
    keys = keys // Ry
    z = keys % Rz
    b = keys // Rz
    return torch.stack([b, z, y, x], dim=1)


def _offsets(kernel_size: Tuple[int, int, int]):
    Kz, Ky, Kx = kernel_size
    return [
        (kz - Kz // 2, ky - Ky // 2, kx - Kx // 2)
        for kz in range(Kz)
        for ky in range(Ky)
        for kx in range(Kx)
    ]


def _pairs_on_device(entry: dict, device: torch.device):
    """Return (pairs, aux) with index tensors moved to `device` (cached)."""
    dk = str(device)
    got = entry["dev"].get(dk)
    if got is None:
        pairs = [(s.to(device), d.to(device)) for (s, d) in entry["pairs_cpu"]]
        aux = {k: v.to(device) for k, v in entry.get("aux_cpu", {}).items()}
        got = (pairs, aux)
        entry["dev"][dk] = got
    return got


# --- SubM conv --------------------------------------------------------------

def _subm_pairs_build(indices_cpu: torch.Tensor, kernel_size: Tuple[int, int, int]):
    """Build per-offset (src_row, dst_row) pairs for a submanifold conv.
    For output voxel i and offset d, src is the row of the input voxel at
    coord_i + d (if it exists), dst is i. Matches _subm_forward_orig."""
    Kz, Ky, Kx = kernel_size
    N = indices_cpu.shape[0]
    maxs = indices_cpu.max(dim=0).values  # [4]
    # Shifted coords (coord + offset) range per spatial axis:
    # [-(K//2), max + (K-1-K//2)]. Shift by origin so digits are >= 0 and use
    # radix = range size, which makes the mixed-radix key injective over the
    # whole shifted range, so a key equality check is an exact coord match.
    # NOTE: pin to indices_cpu.device - the pipeline runs under a
    # `with torch.device("mps")` context that changes the default device for
    # unpinned tensor creation, which previously made lo/hi land on MPS while
    # maxs stayed on CPU (device-mismatch RuntimeError in stage 2).
    lo = torch.tensor(
        [0, -(Kz // 2), -(Ky // 2), -(Kx // 2)],
        dtype=torch.int64,
        device=indices_cpu.device,
    )
    hi = torch.tensor(
        [0, Kz - 1 - (Kz // 2), Ky - 1 - (Ky // 2), Kx - 1 - (Kx // 2)],
        dtype=torch.int64,
        device=indices_cpu.device,
    )
    sizes = (maxs + hi - lo + 1).tolist()
    Rz, Ry, Rx = int(sizes[1]), int(sizes[2]), int(sizes[3])
    shifted = indices_cpu.clone()
    shifted[:, 1] -= lo[1]
    shifted[:, 2] -= lo[2]
    shifted[:, 3] -= lo[3]
    keys = _encode(shifted, Rz, Ry, Rx)
    sorted_keys, order = torch.sort(keys)

    pairs = []
    for dz, dy, dx in _offsets(kernel_size):
        off = dz * Ry * Rx + dy * Rx + dx
        target = keys + off
        pos = torch.searchsorted(sorted_keys, target)
        pos_c = pos.clamp(max=N - 1)
        valid = (pos < N) & (sorted_keys[pos_c] == target)
        dst = valid.nonzero(as_tuple=True)[0]
        src = order[pos[valid]]
        pairs.append((src, dst))
    return pairs


def _subm_forward_vec(
    feats: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    Kz, Ky, Kx, Cin, Cout = weight.shape
    N = feats.shape[0]
    out = torch.zeros(N, Cout, dtype=feats.dtype, device=feats.device)
    if N > 0:
        idx_cpu = indices.detach().to("cpu", torch.int64).contiguous()
        key = _coords_cache_key(idx_cpu, ("subm", Kz, Ky, Kx))
        entry = _cache_get(key)
        if entry is None:
            entry = {
                "pairs_cpu": _subm_pairs_build(idx_cpu, (Kz, Ky, Kx)),
                "aux_cpu": {},
                "dev": {},
            }
            _cache_put(key, entry)
        pairs, _ = _pairs_on_device(entry, feats.device)
        k = 0
        for kz in range(Kz):
            for ky in range(Ky):
                for kx in range(Kx):
                    src, dst = pairs[k]
                    k += 1
                    if src.numel():
                        contrib = (feats[src] @ weight[kz, ky, kx]).to(out.dtype)
                        out.index_add_(0, dst, contrib)
    if bias is not None:
        out = out + bias
    return out


def _subm_forward(feats, indices, weight, bias):
    if _conv_impl() == "orig":
        return _subm_forward_orig(feats, indices, weight, bias)
    return _subm_forward_vec(feats, indices, weight, bias)


# --- Strided sparse conv ------------------------------------------------------

def _sparse_pairs_build(
    indices_cpu: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: int,
    spatial_shape: list,
):
    """Build (out_coords_cpu, per-offset (src, dst) pairs, out_spatial) for the
    strided conv, replicating _sparse_forward_orig semantics exactly:
    output coord for input voxel i and offset d is (coord_i // stride) + d,
    bounds-checked against Dout = (D + 2p - K)//s + 1 when spatial_shape is
    given; each input voxel scatters feats[i] @ W[d] into that output voxel."""
    Kz, Ky, Kx = kernel_size
    sz, sy, sx = stride
    N = indices_cpu.shape[0]
    b = indices_cpu[:, 0]
    base = torch.stack(
        [indices_cpu[:, 1] // sz, indices_cpu[:, 2] // sy, indices_cpu[:, 3] // sx],
        dim=1,
    )
    if spatial_shape:
        Dout = [
            (int(spatial_shape[0]) + padding * 2 - Kz) // sz + 1,
            (int(spatial_shape[1]) + padding * 2 - Ky) // sy + 1,
            (int(spatial_shape[2]) + padding * 2 - Kx) // sx + 1,
        ]
    else:
        Dout = None

    arangeN = torch.arange(N, dtype=torch.int64, device=indices_cpu.device)
    cand_list, src_list, off_list = [], [], []
    for k, (dz, dy, dx) in enumerate(_offsets(kernel_size)):
        cand = torch.stack([b, base[:, 0] + dz, base[:, 1] + dy, base[:, 2] + dx], dim=1)
        if Dout is not None:
            valid = (
                (cand[:, 1] >= 0) & (cand[:, 1] < Dout[0])
                & (cand[:, 2] >= 0) & (cand[:, 2] < Dout[1])
                & (cand[:, 3] >= 0) & (cand[:, 3] < Dout[2])
            )
        else:
            valid = torch.ones(N, dtype=torch.bool, device=indices_cpu.device)
        cand_list.append(cand[valid])
        src_list.append(arangeN[valid])
        off_list.append(
            torch.full((int(valid.sum()),), k, dtype=torch.int64, device=indices_cpu.device)
        )

    cand_all = torch.cat(cand_list, dim=0)
    if cand_all.shape[0] == 0:
        empty_c = torch.zeros((0, 4), dtype=torch.int64, device=indices_cpu.device)
        return empty_c, [(torch.zeros(0, dtype=torch.int64, device=indices_cpu.device), torch.zeros(0, dtype=torch.int64, device=indices_cpu.device)) for _ in _offsets(kernel_size)], [0, 0, 0]
    src_all = torch.cat(src_list, dim=0)
    off_all = torch.cat(off_list, dim=0)

    mins = cand_all.min(dim=0).values
    shifted = cand_all - mins
    sizes = (shifted.max(dim=0).values + 1).tolist()
    Rz, Ry, Rx = int(sizes[1]), int(sizes[2]), int(sizes[3])
    keys = _encode(shifted, Rz, Ry, Rx)
    uniq_keys, inverse = torch.unique(keys, return_inverse=True)
    out_coords = _decode(uniq_keys, Rz, Ry, Rx) + mins

    pairs = []
    for k in range(len(_offsets(kernel_size))):
        sel = off_all == k
        pairs.append((src_all[sel], inverse[sel]))

    out_spatial = [
        int(out_coords[:, 1].max()) + 1,
        int(out_coords[:, 2].max()) + 1,
        int(out_coords[:, 3].max()) + 1,
    ]
    return out_coords, pairs, out_spatial


def _sparse_forward_vec(
    feats: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int, int],
    padding: int,
    spatial_shape: list,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    Kz, Ky, Kx, Cin, Cout = weight.shape
    idx_cpu = indices.detach().to("cpu", torch.int64).contiguous()
    key = _coords_cache_key(
        idx_cpu,
        ("sparse", Kz, Ky, Kx, tuple(stride), int(padding),
         tuple(int(s) for s in spatial_shape) if spatial_shape else None),
    )
    entry = _cache_get(key)
    if entry is None:
        out_coords, pairs, out_spatial = _sparse_pairs_build(
            idx_cpu, (Kz, Ky, Kx), stride, padding, spatial_shape
        )
        entry = {
            "pairs_cpu": pairs,
            "aux_cpu": {"out_coords": out_coords},
            "out_spatial": out_spatial,
            "dev": {},
        }
        _cache_put(key, entry)
    pairs, aux = _pairs_on_device(entry, feats.device)
    out_coords = aux["out_coords"]
    out_spatial = entry["out_spatial"]
    Nout = out_coords.shape[0]
    out_feats = torch.zeros(Nout, Cout, dtype=feats.dtype, device=feats.device)
    k = 0
    for kz in range(Kz):
        for ky in range(Ky):
            for kx in range(Kx):
                src, dst = pairs[k]
                k += 1
                if src.numel():
                    contrib = (feats[src] @ weight[kz, ky, kx]).to(out_feats.dtype)
                    out_feats.index_add_(0, dst, contrib)
    if bias is not None:
        out_feats = out_feats + bias
    return out_feats, out_coords, out_spatial


def _sparse_forward(feats, indices, weight, bias, stride, padding, spatial_shape, batch_size):
    if _conv_impl() == "orig":
        return _sparse_forward_orig(
            feats, indices, weight, bias, stride, padding, spatial_shape, batch_size
        )
    return _sparse_forward_vec(
        feats, indices, weight, bias, stride, padding, spatial_shape, batch_size
    )


# --- Inverse (transposed) sparse conv -----------------------------------------

def _inverse_pairs_build(
    coarse_cpu: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
):
    """Build (fine_coords_cpu, per-offset (src=coarse row, dst=fine row) pairs)
    replicating _inverse_scatter_orig: fine coords are {coarse*s + d >= 0};
    for offset d, fine voxel f takes coarse voxel (f - d)/s when divisible
    and present."""
    Kz, Ky, Kx = kernel_size
    sz, sy, sx = stride
    N = coarse_cpu.shape[0]
    b = coarse_cpu[:, 0]
    offsets = _offsets(kernel_size)

    cand_list = []
    for dz, dy, dx in offsets:
        cand = torch.stack(
            [b, coarse_cpu[:, 1] * sz + dz, coarse_cpu[:, 2] * sy + dy, coarse_cpu[:, 3] * sx + dx],
            dim=1,
        )
        valid = (cand[:, 1] >= 0) & (cand[:, 2] >= 0) & (cand[:, 3] >= 0)
        cand_list.append(cand[valid])
    cand_all = torch.cat(cand_list, dim=0) if cand_list else torch.zeros((0, 4), dtype=torch.int64, device=coarse_cpu.device)
    if cand_all.shape[0] == 0:
        empty_c = torch.zeros((0, 4), dtype=torch.int64, device=coarse_cpu.device)
        return empty_c, [(torch.zeros(0, dtype=torch.int64, device=coarse_cpu.device), torch.zeros(0, dtype=torch.int64, device=coarse_cpu.device)) for _ in offsets]

    mins = cand_all.min(dim=0).values
    shifted = cand_all - mins
    sizes = (shifted.max(dim=0).values + 1).tolist()
    Rz, Ry, Rx = int(sizes[1]), int(sizes[2]), int(sizes[3])
    keys = _encode(shifted, Rz, Ry, Rx)
    uniq_keys, _ = torch.unique(keys, return_inverse=True)
    fine_coords = _decode(uniq_keys, Rz, Ry, Rx) + mins

    cmax = coarse_cpu.max(dim=0).values
    cRz, cRy, cRx = int(cmax[1]) + 1, int(cmax[2]) + 1, int(cmax[3]) + 1
    ckeys = _encode(coarse_cpu, cRz, cRy, cRx)
    sorted_ck, order = torch.sort(ckeys)

    pairs = []
    for dz, dy, dx in offsets:
        num_z = fine_coords[:, 1] - dz
        num_y = fine_coords[:, 2] - dy
        num_x = fine_coords[:, 3] - dx
        cz = num_z // sz
        cy = num_y // sy
        cx = num_x // sx
        ok = (
            (num_z % sz == 0) & (num_y % sy == 0) & (num_x % sx == 0)
            & (num_z >= 0) & (num_y >= 0) & (num_x >= 0)
            & (cz < cRz) & (cy < cRy) & (cx < cRx)
        )
        cand_coarse = torch.stack([fine_coords[:, 0], cz, cy, cx], dim=1)
        k2 = _encode(cand_coarse, cRz, cRy, cRx)
        pos = torch.searchsorted(sorted_ck, k2)
        pos_c = pos.clamp(max=N - 1)
        found = (pos < N) & (sorted_ck[pos_c] == k2)
        valid = ok & found
        dst = valid.nonzero(as_tuple=True)[0]
        src = order[pos[valid]]
        pairs.append((src, dst))
    return fine_coords, pairs


def _inverse_scatter_vec(
    feats: torch.Tensor,
    coarse_idx: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    Kz, Ky, Kx, Cout, Cin = weight.shape
    idx_cpu = coarse_idx.detach().to("cpu", torch.int64).contiguous()
    if idx_cpu.shape[0] == 0:
        empty_f = torch.zeros((0, Cout), dtype=feats.dtype, device=feats.device)
        empty_i = torch.zeros((0, 4), dtype=torch.int64, device=feats.device)
        return empty_f, empty_i, [0, 0, 0]
    key = _coords_cache_key(idx_cpu, ("inv", Kz, Ky, Kx, tuple(stride)))
    entry = _cache_get(key)
    if entry is None:
        fine_coords, pairs = _inverse_pairs_build(idx_cpu, (Kz, Ky, Kx), stride)
        entry = {
            "pairs_cpu": pairs,
            "aux_cpu": {"fine_coords": fine_coords},
            "dev": {},
        }
        _cache_put(key, entry)
    pairs, aux = _pairs_on_device(entry, feats.device)
    fine_coords = aux["fine_coords"]
    Nfine = fine_coords.shape[0]
    out_feats = torch.zeros(Nfine, Cout, dtype=feats.dtype, device=feats.device)
    if Nfine > 0:
        k = 0
        for kz in range(Kz):
            for ky in range(Ky):
                for kx in range(Kx):
                    src, dst = pairs[k]
                    k += 1
                    if src.numel():
                        W = weight[kz, ky, kx]  # [Cout, Cin]
                        contrib = (feats[src] @ W.T).to(out_feats.dtype)
                        out_feats.index_add_(0, dst, contrib)
        if bias is not None:
            out_feats = out_feats + bias.float()
        fine_spatial = [
            int(fine_coords[:, 1].max()) + 1,
            int(fine_coords[:, 2].max()) + 1,
            int(fine_coords[:, 3].max()) + 1,
        ]
    else:
        fine_spatial = [0, 0, 0]
    return out_feats, fine_coords, fine_spatial


# ---------------------------------------------------------------------------
# Module wrappers (mirror conv_spconv.py interface)
# ---------------------------------------------------------------------------

class SparseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        dilation: int = 1,
        padding=None,
        bias: bool = True,
        indice_key=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.stride = (
            tuple(stride) if isinstance(stride, (list, tuple))
            else (stride, stride, stride)
        )
        self.padding = padding if padding is not None else 0
        self._is_subm = all(s == 1 for s in self.stride) and padding is None

        K = kernel_size
        # Weight: [K, K, K, in_ch, out_ch]  (matches spconv checkpoint layout)
        self.weight = nn.Parameter(torch.empty(K, K, K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight.view(-1, in_channels, out_channels).permute(2, 1, 0))

    def forward(self, x: "SparseTensor") -> "SparseTensor":
        feats = x.feats.float()          # keep fp32 for stability; re-cast after
        indices = x.coords.long()        # [N,4] int64 for indexing

        if self._is_subm:
            out_feats = _subm_forward(feats, indices, self.weight.float(),
                                       self.bias.float() if self.bias is not None else None)
            out_feats = out_feats.to(x.feats.dtype)
            new_data = NativeSparseData(out_feats, x.coords, x.data.spatial_shape, x.data.batch_size)
            new_data.indice_dict = x.data.indice_dict
            out = _get_SparseTensor()(new_data, shape=torch.Size([x.shape[0], self.out_channels]),
                               layout=x.layout, scale=x._scale,
                               spatial_cache=x._spatial_cache)
            return out

        # Strided conv
        pad = self.padding if isinstance(self.padding, int) else self.padding[0]
        out_feats, out_idx, out_spatial = _sparse_forward(
            feats, indices, self.weight.float(),
            self.bias.float() if self.bias is not None else None,
            self.stride, pad,
            x.data.spatial_shape, x.data.batch_size,
        )
        out_feats = out_feats.to(x.feats.dtype)
        out_idx_i32 = out_idx.to(torch.int32)

        # Sort by batch (match spconv output layout)
        sort_fwd = out_idx_i32[:, 0].argsort()
        sort_bwd = torch.zeros_like(sort_fwd).scatter_(
            0, sort_fwd, torch.arange(sort_fwd.shape[0], device=sort_fwd.device))
        sorted_feats = out_feats[sort_fwd]
        sorted_idx = out_idx_i32[sort_fwd]

        unsorted_data = NativeSparseData(out_feats, out_idx_i32, out_spatial, x.data.batch_size)
        new_data = NativeSparseData(sorted_feats, sorted_idx, out_spatial, x.data.batch_size)
        new_data.indice_dict = x.data.indice_dict

        out = _get_SparseTensor()(
            new_data,
            shape=torch.Size([x.shape[0], self.out_channels]),
            layout=None,
            scale=tuple(s * stride for s, stride in zip(x._scale, self.stride)),
            spatial_cache=x._spatial_cache,
        )
        out.register_spatial_cache(f"conv_{self.stride}_unsorted_data", unsorted_data)
        out.register_spatial_cache(f"conv_{self.stride}_sort_bwd", sort_bwd)
        return out


class SparseInverseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        dilation: int = 1,
        bias: bool = True,
        indice_key=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.stride = (
            tuple(stride) if isinstance(stride, (list, tuple))
            else (stride, stride, stride)
        )

        K = kernel_size
        # Transposed weight reuses the same layout [K, K, K, out_ch, in_ch]
        # (inverse: out_ch of forward becomes in_ch of inverse)
        self.weight = nn.Parameter(torch.empty(K, K, K, out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight.view(-1, out_channels, in_channels).permute(2, 1, 0))

    def forward(self, x: "SparseTensor") -> "SparseTensor":
        spatial_changed = any(s != 1 for s in self.stride)

        if spatial_changed:
            unsorted_data: NativeSparseData = x.get_spatial_cache(
                f"conv_{self.stride}_unsorted_data")
            sort_bwd = x.get_spatial_cache(f"conv_{self.stride}_sort_bwd")
            # Recover original (unsorted) order, then apply features
            data = unsorted_data.replace_feature(x.feats[sort_bwd])
            if _get_DEBUG():
                assert torch.equal(data.indices, x.coords[sort_bwd]), \
                    "SparseInverseConv3d: coord recovery mismatch"
        else:
            data = x.data

        # data.indices: [N,4] int32 — these are the *downsampled* coords.
        # We scatter back to the original (finer) resolution: the scatter
        # itself lives in _inverse_scatter_orig / _inverse_scatter_vec
        # (selected by SAM3D_CONV_IMPL), which share identical semantics.
        coarse_idx = data.indices.long()   # [N, 4]
        feats = data.features.float()      # [N, Cin]

        if _conv_impl() == "orig":
            out_feats, fine_idx_t, fine_spatial = _inverse_scatter_orig(
                feats, coarse_idx, self.weight, self.bias, self.stride)
        else:
            out_feats, fine_idx_t, fine_spatial = _inverse_scatter_vec(
                feats, coarse_idx, self.weight, self.bias, self.stride)

        if fine_idx_t.shape[0] == 0:
            empty_idx = torch.zeros((0, 4), dtype=torch.int32, device=x.feats.device)
            empty_f = torch.zeros((0, self.out_channels), dtype=x.feats.dtype, device=x.feats.device)
            new_data = NativeSparseData(empty_f, empty_idx, [0, 0, 0], x.data.batch_size)
            return _get_SparseTensor()(new_data,
                                shape=torch.Size([x.shape[0], self.out_channels]),
                                layout=None,
                                scale=tuple(s // stride for s, stride in zip(x._scale, self.stride)),
                                spatial_cache=x._spatial_cache)

        out_feats = out_feats.to(x.feats.dtype)
        fine_idx_i32 = fine_idx_t.to(torch.int32)

        new_data = NativeSparseData(out_feats, fine_idx_i32, fine_spatial, x.data.batch_size)
        out = _get_SparseTensor()(
            new_data,
            shape=torch.Size([x.shape[0], self.out_channels]),
            layout=None,
            scale=tuple(s // stride for s, stride in zip(x._scale, self.stride)),
            spatial_cache=x._spatial_cache,
        )
        return out
