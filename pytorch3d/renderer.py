# Minimal pytorch3d.renderer subset (alt3d shim): look_at_view_transform
# implemented faithfully; rasterizer/camera classes are import-only stubs
# that raise if actually used (the Mac port replaces rendering itself).
import torch
import torch.nn.functional as F


def look_at_rotation(camera_position, at, up, device="cpu"):
    z_axis = F.normalize(camera_position - at, eps=1e-5, dim=1)
    x_axis = torch.cross(up, z_axis, dim=1)
    x_axis = F.normalize(x_axis, eps=1e-5, dim=1)
    y_axis = torch.cross(z_axis, x_axis, dim=1)
    R = torch.cat(
        (x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1
    )
    return R.transpose(1, 2)


def look_at_view_transform(dist=1.0, elev=0.0, azim=0.0, at=((0, 0, 0),), up=((0, 1, 0),), eye=None, device="cpu"):
    if eye is None:
        # GenIA port addition: pytorch3d's spherical form -
        # eye = at + dist * (cos(elev) sin(azim), sin(elev), cos(elev) cos(azim))
        dist_t = torch.as_tensor(dist, dtype=torch.float32, device=device).reshape(-1)
        elev_t = torch.deg2rad(
            torch.as_tensor(elev, dtype=torch.float32, device=device).reshape(-1))
        azim_t = torch.deg2rad(
            torch.as_tensor(azim, dtype=torch.float32, device=device).reshape(-1))
        at_t0 = torch.as_tensor(at, dtype=torch.float32, device=device).reshape(-1, 3)
        eye = torch.stack(
            [
                dist_t * torch.cos(elev_t) * torch.sin(azim_t),
                dist_t * torch.sin(elev_t),
                dist_t * torch.cos(elev_t) * torch.cos(azim_t),
            ],
            dim=-1,
        ) + at_t0
    camera_position = torch.as_tensor(eye, dtype=torch.float32, device=device)
    if camera_position.dim() == 1:
        camera_position = camera_position.unsqueeze(0)
    at_t = torch.as_tensor(at, dtype=torch.float32, device=device)
    if at_t.dim() == 1:
        at_t = at_t.unsqueeze(0)
    up_t = torch.as_tensor(up, dtype=torch.float32, device=device)
    if up_t.dim() == 1:
        up_t = up_t.unsqueeze(0)
    R = look_at_rotation(camera_position, at_t, up_t, device=device)
    T = -torch.bmm(R.transpose(1, 2), camera_position[:, :, None])[:, :, 0]
    return R, T


class _ShimStub:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is an import-only stub in the alt3d pytorch3d shim; "
            "the Mac port is expected to use its own rasterizer on this path."
        )


class PerspectiveCameras(_ShimStub):
    pass


class RasterizationSettings(_ShimStub):
    pass


class MeshRenderer(_ShimStub):
    pass


class MeshRasterizer(_ShimStub):
    pass


class SoftSilhouetteShader(_ShimStub):
    pass


class BlendParams:
    def __init__(self, *args, **kwargs):
        pass


class TexturesVertex(_ShimStub):
    pass
