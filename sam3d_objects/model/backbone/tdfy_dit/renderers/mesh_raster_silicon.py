# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
mesh_raster_silicon — a pure-PyTorch z-buffered triangle rasterizer.

A CUDA-free stand-in for nvdiffrast / utils3d's ``RastContext(backend="cuda")`` for the
two mesh-rasterization needs in this repo's texture-baking / mesh-cleanup paths:

  * per-pixel face id + barycentric coords (for perspective-correct UV interpolation)
  * per-pixel visibility (which faces are seen from a given view)

Runs on MPS / CPU / CUDA. Tile-based with a proper depth test, so occluded faces no
longer overwrite visible ones (the old 9-point scatter had no z-buffer, which dirtied
baked textures). Forward-only.
"""

from typing import Optional
import math
import torch

__all__ = ["rasterize_mesh"]


@torch.no_grad()
def rasterize_mesh(
    verts: torch.Tensor,   # [V, 3] world-space
    faces: torch.Tensor,   # [F, 3] long
    mvp: torch.Tensor,     # [4, 4] projection @ view  (clip = mvp @ [v,1])
    height: int,
    width: int,
    tile_size: int = 32,
    chunk: int = 4096,
):
    """Rasterize a triangle mesh with a z-buffer.

    Returns a dict with:
        face_id : [H, W] long, index of the nearest covering face, -1 where empty
        bary    : [H, W, 3] perspective-correct barycentric weights (sum to 1 where hit)
        mask    : [H, W] bool, True where a face was hit
        depth   : [H, W] NDC depth of the nearest face (1e10 where empty)
    """
    device = verts.device
    dtype = torch.float32
    verts = verts.to(dtype)
    V = verts.shape[0]
    F = faces.shape[0]

    face_id = torch.full((height, width), -1, dtype=torch.long, device=device)
    bary = torch.zeros(height, width, 3, device=device, dtype=dtype)
    zbuf = torch.full((height, width), 1.0e10, device=device, dtype=dtype)
    if F == 0:
        return {"face_id": face_id, "bary": bary, "mask": face_id >= 0, "depth": zbuf}

    # --- vertices -> clip -> ndc -> screen ----------------------------------
    ones = torch.ones(V, 1, device=device, dtype=dtype)
    clip = torch.cat([verts, ones], dim=-1) @ mvp.to(dtype).T   # [V, 4]
    w = clip[:, 3]
    w_safe = torch.where(w.abs() < 1e-6, torch.full_like(w, 1e-6), w)
    ndc = clip[:, :3] / w_safe[:, None]
    sx = (ndc[:, 0] + 1.0) * 0.5 * width
    sy = (1.0 - ndc[:, 1]) * 0.5 * height
    sz = ndc[:, 2]                                              # depth for z-test
    screen = torch.stack([sx, sy], dim=-1)                     # [V, 2]

    fv = faces.long()
    i0, i1, i2 = fv[:, 0], fv[:, 1], fv[:, 2]
    p0, p1, p2 = screen[i0], screen[i1], screen[i2]            # [F, 2]
    z0, z1, z2 = sz[i0], sz[i1], sz[i2]
    w0, w1, w2 = w_safe[i0], w_safe[i1], w_safe[i2]

    # signed area (x2) of each screen triangle
    area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - \
           (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])

    in_front = (w[i0] > 1e-6) & (w[i1] > 1e-6) & (w[i2] > 1e-6)
    minx = torch.minimum(torch.minimum(p0[:, 0], p1[:, 0]), p2[:, 0])
    maxx = torch.maximum(torch.maximum(p0[:, 0], p1[:, 0]), p2[:, 0])
    miny = torch.minimum(torch.minimum(p0[:, 1], p1[:, 1]), p2[:, 1])
    maxy = torch.maximum(torch.maximum(p0[:, 1], p1[:, 1]), p2[:, 1])
    keep = in_front & (area.abs() > 1e-9) & (maxx >= 0) & (minx < width) & (maxy >= 0) & (miny < height)
    if not bool(keep.any()):
        return {"face_id": face_id, "bary": bary, "mask": face_id >= 0, "depth": zbuf}

    # --- build (tile, face) pairs from each face's tile bbox -----------------
    ntx = (width + tile_size - 1) // tile_size
    nty = (height + tile_size - 1) // tile_size

    fidx = torch.nonzero(keep, as_tuple=False).flatten()       # [M]
    fminx = minx[fidx].clamp(0, width - 1)
    fmaxx = maxx[fidx].clamp(0, width - 1)
    fminy = miny[fidx].clamp(0, height - 1)
    fmaxy = maxy[fidx].clamp(0, height - 1)
    tx0 = torch.floor(fminx / tile_size).long()
    tx1 = torch.floor(fmaxx / tile_size).long()
    ty0 = torch.floor(fminy / tile_size).long()
    ty1 = torch.floor(fmaxy / tile_size).long()

    span = int(torch.maximum((tx1 - tx0), (ty1 - ty0)).max().item()) + 1
    span = max(1, min(span, 16))                               # cap memory for huge tris
    offs = torch.arange(span, device=device)
    gx = tx0[:, None, None] + offs[None, :, None]              # [M, span, span]
    gy = ty0[:, None, None] + offs[None, None, :]
    in_range = (gx <= tx1[:, None, None]) & (gy <= ty1[:, None, None]) & \
               (gx >= 0) & (gx < ntx) & (gy >= 0) & (gy < nty)

    f_local = torch.arange(fidx.shape[0], device=device)[:, None, None].expand(-1, span, span)
    f_pair = f_local[in_range]                                 # idx into fidx
    tile_pair = (gy * ntx + gx)[in_range]
    if f_pair.numel() == 0:
        return {"face_id": face_id, "bary": bary, "mask": face_id >= 0, "depth": zbuf}

    # group faces by tile (order within tile does not matter — z-test handles depth)
    order = torch.argsort(tile_pair)
    tile_pair = tile_pair[order]
    f_global = fidx[f_pair[order]]                             # face ids into full arrays

    change = torch.ones_like(tile_pair, dtype=torch.bool)
    change[1:] = tile_pair[1:] != tile_pair[:-1]
    starts = torch.nonzero(change, as_tuple=False).flatten()
    ends = torch.cat([starts[1:], torch.tensor([tile_pair.shape[0]], device=device)])
    tiles_l = tile_pair[starts].tolist()
    starts_l = starts.tolist()
    ends_l = ends.tolist()

    # gather per-face screen attributes once, in tile-sorted order
    P0 = p0[f_global]; P1 = p1[f_global]; P2 = p2[f_global]    # [K, 2]
    Z0 = z0[f_global]; Z1 = z1[f_global]; Z2 = z2[f_global]    # [K]
    W0 = w0[f_global]; W1 = w1[f_global]; W2 = w2[f_global]
    AREA = area[f_global]                                      # [K]

    eps = 1e-9
    for s, e, tid in zip(starts_l, ends_l, tiles_l):
        ti = tid % ntx
        tj = tid // ntx
        x0 = ti * tile_size
        y0 = tj * tile_size
        x1 = min(x0 + tile_size, width)
        y1 = min(y0 + tile_size, height)
        tw, th = x1 - x0, y1 - y0

        xs = torch.arange(x0, x1, device=device, dtype=dtype) + 0.5
        ys = torch.arange(y0, y1, device=device, dtype=dtype) + 0.5
        gyy, gxx = torch.meshgrid(ys, xs, indexing="ij")
        px = gxx.reshape(-1)        # [Pn]
        py = gyy.reshape(-1)
        Pn = px.shape[0]

        best_z = torch.full((Pn,), 1.0e10, device=device, dtype=dtype)
        best_f = torch.full((Pn,), -1, dtype=torch.long, device=device)
        best_b = torch.zeros(Pn, 3, device=device, dtype=dtype)

        for cs in range(s, e, chunk):
            ce = min(cs + chunk, e)
            a0x, a0y = P0[cs:ce, 0], P0[cs:ce, 1]
            a1x, a1y = P1[cs:ce, 0], P1[cs:ce, 1]
            a2x, a2y = P2[cs:ce, 0], P2[cs:ce, 1]
            ar = AREA[cs:ce]                                   # [g]
            inv_area = 1.0 / ar

            # edge functions -> screen-space barycentric [Pn, g]
            dxp = px[:, None]
            dyp = py[:, None]
            l0 = ((a1x - dxp) * (a2y - dyp) - (a2x - dxp) * (a1y - dyp)) * inv_area[None, :]
            l1 = ((a2x - dxp) * (a0y - dyp) - (a0x - dxp) * (a2y - dyp)) * inv_area[None, :]
            l2 = 1.0 - l0 - l1
            inside = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)

            zc = l0 * Z0[None, cs:ce] + l1 * Z1[None, cs:ce] + l2 * Z2[None, cs:ce]  # [Pn, g]
            zc = torch.where(inside, zc, torch.full_like(zc, 1.0e10))

            zmin, amin = zc.min(dim=1)                         # nearest face per pixel in chunk
            better = zmin < best_z
            if bool(better.any()):
                rows = torch.nonzero(better, as_tuple=False).flatten()
                sel = amin[rows]
                best_z[rows] = zmin[rows]
                best_f[rows] = (cs + sel)
                # perspective-correct barycentric for the winning face
                bl0 = l0[rows, sel]; bl1 = l1[rows, sel]; bl2 = l2[rows, sel]
                pw0 = 1.0 / W0[cs:ce][sel]
                pw1 = 1.0 / W1[cs:ce][sel]
                pw2 = 1.0 / W2[cs:ce][sel]
                b0 = bl0 * pw0; b1 = bl1 * pw1; b2 = bl2 * pw2
                bsum = (b0 + b1 + b2).clamp_min(1e-12)
                best_b[rows, 0] = b0 / bsum
                best_b[rows, 1] = b1 / bsum
                best_b[rows, 2] = b2 / bsum

        hit = best_f >= 0
        # map local tile-sorted face index back to global face id
        gf = torch.where(hit, f_global[best_f.clamp_min(0)], torch.full_like(best_f, -1))
        face_id[y0:y1, x0:x1] = gf.reshape(th, tw)
        zbuf[y0:y1, x0:x1] = best_z.reshape(th, tw)
        bary[y0:y1, x0:x1, :] = best_b.reshape(th, tw, 3)

    return {"face_id": face_id, "bary": bary, "mask": face_id >= 0, "depth": zbuf}
