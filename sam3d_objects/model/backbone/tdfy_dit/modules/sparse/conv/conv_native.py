# Copyright (c) Meta Platforms, Inc. and affiliates.
# Pure-PyTorch implementations of SubMConv3d / SparseConv3d / SparseInverseConv3d
# for the "native" backend (CPU / MPS, no spconv / CUDA required).
#
# Weight layout mirrors spconv: [K, K, K, in_channels, out_channels]
# Coordinate layout: [N, 4] — (batch, z, y, x)
#
# Performance note: these are correct reference implementations. They are not
# optimised for speed; correctness comes first.

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .native_backend import NativeSparseData

# Lazy import to avoid circular dependency (sparse.__init__ imports conv.__init__)
def _get_SparseTensor():
    from ..basic import SparseTensor
    return SparseTensor

def _get_DEBUG():
    from .. import DEBUG
    return DEBUG

__all__ = ["SparseConv3d", "SparseInverseConv3d"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_coord_map(indices: torch.Tensor) -> dict:
    """Build a dict (b, z, y, x) -> row_index (on CPU as plain Python ints)."""
    cpu = indices.cpu().numpy()
    return {(int(r[0]), int(r[1]), int(r[2]), int(r[3])): i for i, r in enumerate(cpu)}


def _subm_forward(
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
                    contrib = gathered @ W          # [M, Cout]
                    out.index_add_(0, src_t, contrib)

    if bias is not None:
        out = out + bias
    return out


def _sparse_forward(
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
                    contrib = feats[src_t] @ W   # [M, Cout]
                    out_feats.index_add_(0, dst_t, contrib)

    if bias is not None:
        out_feats = out_feats + bias

    bs, zs, ys, xs = zip(*out_coords_list)
    out_spatial = [max(zs) + 1, max(ys) + 1, max(xs) + 1]

    return out_feats, out_idx_t, out_spatial


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
        # We need to scatter back to the original (finer) resolution.
        # Recover original fine coords from unsorted_data's indice_dict is not
        # available in the native backend; instead we reconstruct by upsizing:
        # fine_coord = coarse_coord * stride + kernel_offset
        # and accumulating into a dense output of the original spatial shape.
        #
        # For the conv transpose semantics expected here, we use a simplified
        # scatter approach: for each coarse voxel, iterate kernel offsets,
        # compute fine voxel, and scatter-add weight contribution.

        coarse_idx = data.indices.long()   # [N, 4]
        feats = data.features.float()      # [N, Cin]
        Kz, Ky, Kx, Cout, Cin = self.weight.shape
        sz, sy, sx = self.stride

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
            empty_idx = torch.zeros((0, 4), dtype=torch.int32, device=x.feats.device)
            empty_f = torch.zeros((0, self.out_channels), dtype=x.feats.dtype, device=x.feats.device)
            new_data = NativeSparseData(empty_f, empty_idx, [0, 0, 0], x.data.batch_size)
            return _get_SparseTensor()(new_data,
                                shape=torch.Size([x.shape[0], self.out_channels]),
                                layout=None,
                                scale=tuple(s // stride for s, stride in zip(x._scale, self.stride)),
                                spatial_cache=x._spatial_cache)

        fine_map = {c: i for i, c in enumerate(fine_coords_list)}
        fine_idx_t = torch.tensor(fine_coords_list, dtype=torch.int64, device=x.feats.device)
        Nfine = fine_idx_t.shape[0]
        out_feats = torch.zeros(Nfine, self.out_channels, dtype=feats.dtype, device=feats.device)

        coarse_map = _build_coord_map(coarse_idx)

        for kz in range(Kz):
            dz = kz - Kz // 2
            for ky in range(Ky):
                dy = ky - Ky // 2
                for kx in range(Kx):
                    dx = kx - Kx // 2
                    # weight[kz,ky,kx]: [Cout, Cin] — transpose of forward weight
                    W = self.weight[kz, ky, kx]  # [Cout, Cin]

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
                        contrib = feats[src_t] @ W.T
                        out_feats.index_add_(0, dst_t, contrib)

        if self.bias is not None:
            out_feats = out_feats + self.bias.float()

        out_feats = out_feats.to(x.feats.dtype)
        fine_idx_i32 = fine_idx_t.to(torch.int32)

        bs_, zs, ys, xs = zip(*fine_coords_list)
        fine_spatial = [max(zs) + 1, max(ys) + 1, max(xs) + 1]

        new_data = NativeSparseData(out_feats, fine_idx_i32, fine_spatial, x.data.batch_size)
        out = _get_SparseTensor()(
            new_data,
            shape=torch.Size([x.shape[0], self.out_channels]),
            layout=None,
            scale=tuple(s // stride for s, stride in zip(x._scale, self.stride)),
            spatial_cache=x._spatial_cache,
        )
        return out
