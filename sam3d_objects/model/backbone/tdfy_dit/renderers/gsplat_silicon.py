# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
gsplat_silicon — a pure-PyTorch 3D Gaussian Splatting rasterizer.

Drop-in replacement for ``gsplat.rasterization`` that runs on Apple Silicon (MPS),
CPU, or CUDA, since the real ``gsplat`` ships CUDA-only kernels.

Implements EWA volume splatting (Zwicker et al. 2001) the same way gsplat does:
  * project 3D gaussians to 2D, build the 2D covariance via the projection Jacobian
  * apply the 0.3px low-pass (dilation) filter
  * tile the image, depth-sort the gaussians per tile, alpha-composite front-to-back

It is forward-only (no autograd through the rasterizer), which is all the texture
baking / multi-view rendering paths in this repo need.

The public ``rasterization`` function matches the subset of the gsplat API used in
``gaussian_render.py``:
    means, quats, scales, opacities, colors, viewmats, Ks, width, height,
    sh_degree=None, backgrounds=None
and returns ``(render_colors [C,H,W,3], render_alphas [C,H,W,1], meta)``.
"""

from typing import Optional, Tuple
import math
import torch

from .sh_utils import eval_sh, C0

__all__ = ["rasterization"]


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) quaternions [N, 4] -> rotation matrices [N, 3, 3]."""
    quats = torch.nn.functional.normalize(quats, dim=-1)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
            2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    return R


def _compute_colors(
    means: torch.Tensor,
    colors: torch.Tensor,
    sh_degree: Optional[int],
    viewmat: torch.Tensor,
) -> torch.Tensor:
    """Resolve per-gaussian RGB [N, 3] from either precomputed colors or SH coeffs."""
    if sh_degree is None:
        # colors are already RGB [N, 3]
        return colors
    # colors are SH coefficients in gsplat layout [N, K, 3]; eval_sh wants [N, 3, K]
    sh = colors.transpose(1, 2)
    if sh_degree == 0:
        rgb = C0 * sh[..., 0] + 0.5  # view-independent
    else:
        campos = torch.inverse(viewmat)[:3, 3]
        dirs = torch.nn.functional.normalize(means - campos[None, :], dim=-1)
        rgb = eval_sh(sh_degree, sh, dirs) + 0.5
    return rgb.clamp_min(0.0)


def _rasterize_one_view(
    means: torch.Tensor,      # [N, 3] world
    quats: torch.Tensor,      # [N, 4] (w,x,y,z)
    scales: torch.Tensor,     # [N, 3]
    opacities: torch.Tensor,  # [N]
    rgb: torch.Tensor,        # [N, 3]
    viewmat: torch.Tensor,    # [4, 4] world->camera
    K: torch.Tensor,          # [3, 3] pixel intrinsics
    width: int,
    height: int,
    bg: torch.Tensor,         # [3]
    near_plane: float = 0.01,
    eps2d: float = 0.3,
    tile_size: int = 32,
    chunk: int = 8192,
    min_transmittance: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = means.device
    dtype = torch.float32
    means = means.to(dtype)
    N = means.shape[0]

    out_color = bg[None, None, :].expand(height, width, 3).clone()
    out_alpha = torch.zeros(height, width, 1, device=device, dtype=dtype)
    if N == 0:
        return out_color, out_alpha

    Rv = viewmat[:3, :3].to(dtype)
    tv = viewmat[:3, 3].to(dtype)
    fx, fy = K[0, 0].to(dtype), K[1, 1].to(dtype)
    cx, cy = K[0, 2].to(dtype), K[1, 2].to(dtype)

    # --- world -> camera, project to 2D --------------------------------------
    p_cam = means @ Rv.T + tv[None, :]          # [N, 3]
    xc, yc, zc = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]
    valid = zc > near_plane

    zc_safe = torch.where(valid, zc, torch.ones_like(zc))
    u = fx * xc / zc_safe + cx
    v = fy * yc / zc_safe + cy
    mean2d = torch.stack([u, v], dim=-1)        # [N, 2]

    # --- 3D covariance (world), rotate to camera -----------------------------
    Rq = quat_to_rotmat(quats).to(dtype)        # [N, 3, 3]
    M = Rq * scales[:, None, :].to(dtype)       # scale the columns -> [N, 3, 3]
    Sigma = M @ M.transpose(1, 2)               # [N, 3, 3] world-space cov
    Sigma_cam = Rv[None] @ Sigma @ Rv.T[None]   # [N, 3, 3]

    # --- projection Jacobian -> 2D covariance (EWA) --------------------------
    J = torch.zeros(N, 2, 3, device=device, dtype=dtype)
    inv_z = 1.0 / zc_safe
    J[:, 0, 0] = fx * inv_z
    J[:, 0, 2] = -fx * xc * inv_z * inv_z
    J[:, 1, 1] = fy * inv_z
    J[:, 1, 2] = -fy * yc * inv_z * inv_z
    cov2d = J @ Sigma_cam @ J.transpose(1, 2)   # [N, 2, 2]
    a = cov2d[:, 0, 0] + eps2d
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + eps2d

    det = a * c - b * b
    valid = valid & (det > 1e-12)
    det_safe = torch.where(det > 1e-12, det, torch.ones_like(det))
    inv_det = 1.0 / det_safe
    # conic = inverse of [[a,b],[b,c]] = (1/det)[[c,-b],[-b,a]]
    conic = torch.stack([c * inv_det, -b * inv_det, a * inv_det], dim=-1)  # [N, 3]

    # 3-sigma radius from the larger eigenvalue of the 2D covariance
    mid = 0.5 * (a + c)
    disc = torch.sqrt(torch.clamp(mid * mid - det, min=0.0))
    lambda_max = mid + disc
    radius = 3.0 * torch.sqrt(torch.clamp(lambda_max, min=0.0))           # [N]

    # cull gaussians whose footprint lies fully outside the image
    in_view = (
        valid
        & (u + radius >= 0) & (u - radius < width)
        & (v + radius >= 0) & (v - radius < height)
    )
    if not bool(in_view.any()):
        return out_color, out_alpha

    # --- build (tile, gaussian) pairs via a clamped neighbourhood ------------
    ntx = (width + tile_size - 1) // tile_size
    nty = (height + tile_size - 1) // tile_size

    # how far (in tiles) the largest splat reaches, capped so memory stays bounded
    rmax = float(torch.clamp(radius[in_view].max(), min=1.0).item())
    h = max(1, min(3, math.ceil(rmax / tile_size)))
    D = 2 * h + 1

    idx_all = torch.nonzero(in_view, as_tuple=False).flatten()            # [M]
    gx = u[idx_all]
    gy = v[idx_all]
    gr = radius[idx_all]
    ctx = torch.floor(gx / tile_size).long()
    cty = torch.floor(gy / tile_size).long()

    offs = torch.arange(-h, h + 1, device=device)
    tx = ctx[:, None, None] + offs[None, :, None]      # [M, D, D]
    ty = cty[:, None, None] + offs[None, None, :]      # [M, D, D]

    tile_x0 = (tx * tile_size).to(dtype)
    tile_y0 = (ty * tile_size).to(dtype)
    tile_x1 = tile_x0 + tile_size
    tile_y1 = tile_y0 + tile_size
    gxb = gx[:, None, None]
    gyb = gy[:, None, None]
    grb = gr[:, None, None]
    overlap = (
        (gxb + grb >= tile_x0) & (gxb - grb <= tile_x1)
        & (gyb + grb >= tile_y0) & (gyb - grb <= tile_y1)
        & (tx >= 0) & (tx < ntx) & (ty >= 0) & (ty < nty)
    )

    g_local = torch.arange(idx_all.shape[0], device=device)[:, None, None].expand(-1, D, D)
    g_pair = g_local[overlap]                          # indices into idx_all
    tile_pair = (ty * ntx + tx)[overlap]               # flat tile id
    if g_pair.numel() == 0:
        return out_color, out_alpha

    depth_pair = zc[idx_all][g_pair]    # camera-space depth

    # Sort by (tile, depth): tile primary, depth secondary, so each tile's gaussians are
    # contiguous and ordered front-to-back. Use the float64 combined key tile*1e6 + depth
    # (52-bit mantissa keeps tile and depth cleanly separated). MPS has no float64, so the
    # key + argsort are computed on the CPU and only the resulting order is moved back.
    key = tile_pair.detach().cpu().to(torch.float64) * 1.0e6 + depth_pair.detach().cpu().to(torch.float64)
    order = torch.argsort(key).to(device)
    g_pair = g_pair[order]
    tile_pair = tile_pair[order]
    g_global = idx_all[g_pair]                         # gaussian ids into full arrays

    # per-tile segment boundaries
    change = torch.ones_like(tile_pair, dtype=torch.bool)
    change[1:] = tile_pair[1:] != tile_pair[:-1]
    starts = torch.nonzero(change, as_tuple=False).flatten()
    ends = torch.cat([starts[1:], torch.tensor([tile_pair.shape[0]], device=device)])
    tile_of_seg = tile_pair[starts]

    starts_l = starts.tolist()
    ends_l = ends.tolist()
    tiles_l = tile_of_seg.tolist()

    # gather per-gaussian attributes once, in sorted order
    m2_all = mean2d[g_global]      # [P, 2]
    cn_all = conic[g_global]       # [P, 3]
    op_all = opacities[g_global].to(dtype).clamp(0.0, 1.0)  # [P]
    col_all = rgb[g_global].to(dtype)                       # [P, 3]

    # --- composite each tile --------------------------------------------------
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
        gridy, gridx = torch.meshgrid(ys, xs, indexing="ij")
        px = gridx.reshape(-1)      # [P]
        py = gridy.reshape(-1)
        P = px.shape[0]

        m2 = m2_all[s:e]            # [G, 2]
        cn = cn_all[s:e]            # [G, 3]
        op = op_all[s:e]            # [G]
        col = col_all[s:e]         # [G, 3]
        G = m2.shape[0]

        T = torch.ones(P, device=device, dtype=dtype)
        color = torch.zeros(P, 3, device=device, dtype=dtype)

        for cs in range(0, G, chunk):
            ce = min(cs + chunk, G)
            mm = m2[cs:ce]                       # [g, 2]
            dx = px[:, None] - mm[None, :, 0]    # [P, g]
            dy = py[:, None] - mm[None, :, 1]
            cc = cn[cs:ce]                       # [g, 3] = (A, B, C)
            power = -0.5 * (
                cc[None, :, 0] * dx * dx
                + 2.0 * cc[None, :, 1] * dx * dy
                + cc[None, :, 2] * dy * dy
            )                                    # [P, g]
            alpha = op[None, cs:ce] * torch.exp(power)
            alpha = torch.where(power > 0, torch.zeros_like(alpha), alpha)
            alpha = alpha.clamp(max=0.999)

            one_minus = 1.0 - alpha
            cp = torch.cumprod(one_minus, dim=1)                 # inclusive
            excl = torch.cat([torch.ones(P, 1, device=device, dtype=dtype), cp[:, :-1]], dim=1)
            w = alpha * excl * T[:, None]                        # [P, g]
            color = color + (w[:, :, None] * col[None, cs:ce, :]).sum(dim=1)
            T = T * cp[:, -1]
            if float(T.max().item()) < min_transmittance:
                break

        color = color + T[:, None] * bg[None, :]
        out_color[y0:y1, x0:x1, :] = color.reshape(th, tw, 3)
        out_alpha[y0:y1, x0:x1, 0] = (1.0 - T).reshape(th, tw)

    return out_color, out_alpha


@torch.no_grad()
def rasterization(
    means: torch.Tensor,         # [N, 3]
    quats: torch.Tensor,         # [N, 4] (w, x, y, z)
    scales: torch.Tensor,        # [N, 3]
    opacities: torch.Tensor,     # [N]
    colors: torch.Tensor,        # [N, 3] (RGB) or [N, K, 3] (SH coeffs)
    viewmats: torch.Tensor,      # [C, 4, 4] world->camera
    Ks: torch.Tensor,            # [C, 3, 3] pixel intrinsics
    width: int,
    height: int,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[torch.Tensor] = None,
    near_plane: float = 0.01,
    eps2d: float = 0.3,
    tile_size: int = 32,
    **kwargs,
):
    """Pure-PyTorch gsplat-compatible rasterization (forward only).

    Returns:
        render_colors: [C, H, W, 3]
        render_alphas: [C, H, W, 1]
        meta: dict (empty; kept for API compatibility)
    """
    device = means.device
    C = viewmats.shape[0]

    if backgrounds is None:
        bg_all = torch.zeros(C, 3, device=device, dtype=torch.float32)
    else:
        bg = backgrounds.to(device=device, dtype=torch.float32)
        bg_all = bg[None, :].expand(C, 3) if bg.dim() == 1 else bg

    colors_out = []
    alphas_out = []
    for ci in range(C):
        rgb = _compute_colors(means, colors, sh_degree, viewmats[ci])
        col, alpha = _rasterize_one_view(
            means, quats, scales, opacities, rgb,
            viewmats[ci], Ks[ci], int(width), int(height), bg_all[ci],
            near_plane=near_plane, eps2d=eps2d, tile_size=tile_size,
        )
        colors_out.append(col)
        alphas_out.append(alpha)

    render_colors = torch.stack(colors_out, dim=0)   # [C, H, W, 3]
    render_alphas = torch.stack(alphas_out, dim=0)   # [C, H, W, 1]
    return render_colors, render_alphas, {}
