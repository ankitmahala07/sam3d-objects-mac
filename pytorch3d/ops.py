# alt3d shim: pure-PyTorch stand-in for the pytorch3d.ops pieces GenIA's
# call sites use (knn_points, estimate_pointcloud_normals). Selection is
# computed with chunked cdist under no_grad (discrete, as in pytorch3d);
# returned dists are recomputed from the gathered neighbours so gradients
# flow to both point sets through the selected pairs, matching pytorch3d's
# backward semantics.
from collections import namedtuple

import torch

KNN = namedtuple("KNN", ["dists", "idx", "knn"])


def knn_points(p1, p2, lengths1=None, lengths2=None, norm=2, K=1,
               version=-1, return_nn=False, return_sorted=True):
    """p1 (B, N, 3), p2 (B, M, 3) -> KNN(dists (B, N, K) squared, idx, knn).

    Only the full-cloud case (no lengths) and L2 norm are supported - that
    is all GenIA's call sites use.
    """
    if lengths1 is not None or lengths2 is not None:
        raise NotImplementedError("shim knn_points: lengths not supported")
    if norm != 2:
        raise NotImplementedError("shim knn_points: only norm=2")
    B, N, _ = p1.shape
    M = p2.shape[1]
    K = min(K, M)
    device = p1.device
    idx_out = torch.empty(B, N, K, dtype=torch.long, device=device)
    with torch.no_grad():
        chunk = 1024
        for b in range(B):
            for s in range(0, N, chunk):
                d = torch.cdist(p1[b, s:s + chunk].float(),
                                p2[b].float())  # (n, M)
                top = torch.topk(d, K, dim=1, largest=False,
                                 sorted=return_sorted)
                idx_out[b, s:s + chunk] = top.indices.long()
    # Gather neighbours differentiably; dists from the gathered pairs.
    p2_exp = p2[:, None, :, :].expand(B, N, M, 3)
    idx_exp = idx_out[..., None].expand(B, N, K, 3)
    knn = torch.gather(p2_exp, 2, idx_exp)  # (B, N, K, 3)
    diff = p1[:, :, None, :] - knn
    dists = (diff * diff).sum(-1)
    return KNN(dists=dists, idx=idx_out, knn=knn)


def knn_gather(x, idx):
    """x (B, M, C), idx (B, N, K) -> (B, N, K, C)."""
    B, M, C = x.shape
    N, K = idx.shape[1], idx.shape[2]
    x_exp = x[:, None, :, :].expand(B, N, M, C)
    return torch.gather(x_exp, 2, idx[..., None].expand(B, N, K, C))


def estimate_pointcloud_normals(points, neighborhood_size=50,
                                disambiguate_directions=True):
    """Local-PCA normals: smallest-eigenvalue eigenvector of each point's
    neighbourhood covariance (neighbourhood includes the point itself,
    as in pytorch3d). points (B, N, 3) -> normals (B, N, 3).

    With disambiguate_directions, normals are oriented away from the cloud
    centroid; GenIA's caller passes False and re-orients itself.
    """
    B, N, _ = points.shape
    k = min(neighborhood_size, N)
    knn = knn_points(points, points, K=k, return_nn=True)
    neigh = knn.knn  # (B, N, k, 3)
    centroid = neigh.mean(dim=2, keepdim=True)
    centered = neigh - centroid
    cov = centered.transpose(-1, -2) @ centered  # (B, N, 3, 3)
    _, eigvecs = torch.linalg.eigh(cov)
    normals = eigvecs[..., :, 0]  # smallest eigenvalue first
    normals = torch.nn.functional.normalize(normals, dim=-1)
    if disambiguate_directions:
        cloud_centroid = points.mean(dim=1, keepdim=True)  # (B, 1, 3)
        away = points - cloud_centroid
        flip = (normals * away).sum(-1, keepdim=True) < 0
        normals = torch.where(flip, -normals, normals)
    return normals
