# Import-only stub (alt3d shim). The Mac port does not use pytorch3d Meshes
# on the CLI path; raising on construction keeps any surprise use loud.


class Meshes:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "pytorch3d.structures.Meshes is an import-only stub in the alt3d shim."
        )


class Pointclouds:
    """Minimal functional Pointclouds (GenIA port addition).

    GenIA's final evaluation builds point clouds through the fork's
    SceneVisualizer.object_pointcloud and reads them back via
    points_list(); the import-only stub made that path unusable. This
    implements the small container surface those call sites use: padded
    tensor or list input, points/features/normals as list or padded
    tensors, num_points_per_cloud, device, and .to(). Meshes above stays
    an import-only stub on purpose: GenIA's visibility code detects it
    via NotImplementedError and routes to its silicon rasterizer.
    """

    def __init__(self, points=None, features=None, normals=None):
        import torch

        self._points_list = self._as_list(points, torch)
        self._features_list = self._as_list(features, torch)
        self._normals_list = self._as_list(normals, torch)
        if self._points_list is None:
            raise ValueError("Pointclouds requires points")

    @staticmethod
    def _as_list(x, torch):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return [t for t in x]
        if torch.is_tensor(x):
            if x.dim() == 2:
                return [x]
            return [x[i] for i in range(x.shape[0])]
        raise TypeError(f"unsupported points type: {type(x)}")

    def __len__(self):
        return len(self._points_list)

    @property
    def device(self):
        return self._points_list[0].device

    def num_points_per_cloud(self):
        import torch

        return torch.tensor(
            [p.shape[0] for p in self._points_list],
            device=self.device, dtype=torch.int64)

    def points_list(self):
        return self._points_list

    def features_list(self):
        return self._features_list

    def normals_list(self):
        return self._normals_list

    def _padded(self, lst):
        import torch
        import torch.nn.functional as F

        if lst is None:
            return None
        max_n = max(t.shape[0] for t in lst)
        out = []
        for t in lst:
            if t.shape[0] < max_n:
                t = F.pad(t, (0, 0, 0, max_n - t.shape[0]))
            out.append(t)
        return torch.stack(out, dim=0)

    def points_padded(self):
        return self._padded(self._points_list)

    def features_padded(self):
        return self._padded(self._features_list)

    def normals_padded(self):
        return self._padded(self._normals_list)

    def to(self, device):
        def mv(lst):
            return None if lst is None else [t.to(device) for t in lst]

        return Pointclouds(points=mv(self._points_list),
                           features=mv(self._features_list),
                           normals=mv(self._normals_list))
