from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np
import open3d as o3d
import trimesh


@dataclass
class RetopoResult:
    vertices: np.ndarray
    faces: np.ndarray
    quads: np.ndarray
    stats: dict


_CORNERS = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int32,
)

_EDGES = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _auto_target(face_count: int, target_faces: Optional[int]) -> int:
    if target_faces is not None:
        return max(500, min(int(target_faces), int(face_count)))
    if face_count <= 2500:
        return int(face_count)
    if face_count <= 20000:
        return 2000
    return 10000


def _principal_frame(vertices: np.ndarray):
    center = vertices.mean(axis=0)
    centered = vertices - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = centered @ axes
    return local.astype(np.float32), center.astype(np.float32), axes.astype(np.float32)


def _clean_mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    return mesh


def _make_scene(vertices: np.ndarray, faces: np.ndarray):
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    legacy.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    legacy.compute_vertex_normals()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene


def _signed_distance(scene, points: np.ndarray, chunk: int = 250000) -> np.ndarray:
    out = []
    for start in range(0, points.shape[0], chunk):
        part = points[start : start + chunk].astype(np.float32, copy=False)
        query = o3d.core.Tensor(part, dtype=o3d.core.Dtype.Float32)
        out.append(scene.compute_signed_distance(query).numpy())
    return np.concatenate(out, axis=0)


def _closest_points(scene, points: np.ndarray, chunk: int = 250000) -> np.ndarray:
    out = []
    for start in range(0, points.shape[0], chunk):
        part = points[start : start + chunk].astype(np.float32, copy=False)
        query = o3d.core.Tensor(part, dtype=o3d.core.Dtype.Float32)
        out.append(scene.compute_closest_points(query)["points"].numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def _grid_for_target(
    vertices: np.ndarray,
    area: float,
    target_triangles: int,
    max_cells: int,
):
    target_quads = max(250, int(math.ceil(target_triangles / 2)))
    base_quads = max(64, int(target_quads * 0.9))
    extents = np.ptp(vertices, axis=0)
    longest = max(float(extents.max()), 1e-6)
    extents = np.maximum(extents, longest * 0.02)
    cell = math.sqrt(max(float(area), 1e-8) / float(base_quads))
    dims = np.ceil(extents / max(cell, 1e-6)).astype(np.int32)
    dims = np.clip(dims, 6, max_cells)

    max_total_cells = max(120000, min(900000, max_cells**3))
    total = int(np.prod(dims))
    if total > max_total_cells:
        scale = (max_total_cells / total) ** (1.0 / 3.0)
        dims = np.maximum(6, np.floor(dims * scale).astype(np.int32))
    return dims.astype(np.int32), target_quads


def _sample_sdf_grid(scene, bounds_min, bounds_max, dims):
    axes = [
        np.linspace(bounds_min[i], bounds_max[i], int(dims[i]) + 1, dtype=np.float32)
        for i in range(3)
    ]
    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    values = _signed_distance(scene, points).reshape(
        int(dims[0]) + 1, int(dims[1]) + 1, int(dims[2]) + 1
    )
    spacing = np.asarray(
        [
            axes[0][1] - axes[0][0],
            axes[1][1] - axes[1][0],
            axes[2][1] - axes[2][0],
        ],
        dtype=np.float32,
    )
    return values.astype(np.float32), axes, spacing


def _cell_corner_values(values, i, j, k):
    return np.asarray(
        [
            values[i, j, k],
            values[i + 1, j, k],
            values[i, j + 1, k],
            values[i + 1, j + 1, k],
            values[i, j, k + 1],
            values[i + 1, j, k + 1],
            values[i, j + 1, k + 1],
            values[i + 1, j + 1, k + 1],
        ],
        dtype=np.float32,
    )


def _extract_surface_net(values, bounds_min, spacing):
    nx, ny, nz = np.asarray(values.shape, dtype=np.int32) - 1
    cell_index = np.full((nx, ny, nz), -1, dtype=np.int32)
    vertices = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                vals = _cell_corner_values(values, i, j, k)
                if vals.min() > 0.0 or vals.max() < 0.0:
                    continue

                base = bounds_min + np.asarray([i, j, k], dtype=np.float32) * spacing
                intersections = []
                for a, b in _EDGES:
                    va = vals[a]
                    vb = vals[b]
                    if (va < 0.0 and vb < 0.0) or (va > 0.0 and vb > 0.0):
                        continue
                    denom = va - vb
                    if abs(float(denom)) < 1e-8:
                        t = 0.5
                    else:
                        t = float(va / denom)
                    pa = base + _CORNERS[a].astype(np.float32) * spacing
                    pb = base + _CORNERS[b].astype(np.float32) * spacing
                    intersections.append(pa + np.clip(t, 0.0, 1.0) * (pb - pa))

                if not intersections:
                    continue
                cell_index[i, j, k] = len(vertices)
                vertices.append(np.mean(intersections, axis=0))

    if not vertices:
        raise RuntimeError("Native retopo found no surface cells")

    quads = []

    def add_quad(cells, flip):
        ids = []
        for ci, cj, ck in cells:
            if ci < 0 or cj < 0 or ck < 0 or ci >= nx or cj >= ny or ck >= nz:
                return
            idx = cell_index[ci, cj, ck]
            if idx < 0:
                return
            ids.append(int(idx))
        if len(set(ids)) != 4:
            return
        if flip:
            ids = [ids[0], ids[3], ids[2], ids[1]]
        quads.append(ids)

    for i in range(nx):
        for j in range(1, ny):
            for k in range(1, nz):
                if values[i, j, k] * values[i + 1, j, k] < 0.0:
                    add_quad(
                        [(i, j - 1, k - 1), (i, j, k - 1), (i, j, k), (i, j - 1, k)],
                        values[i, j, k] < values[i + 1, j, k],
                    )

    for i in range(1, nx):
        for j in range(ny):
            for k in range(1, nz):
                if values[i, j, k] * values[i, j + 1, k] < 0.0:
                    add_quad(
                        [(i - 1, j, k - 1), (i, j, k - 1), (i, j, k), (i - 1, j, k)],
                        values[i, j, k] > values[i, j + 1, k],
                    )

    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(nz):
                if values[i, j, k] * values[i, j, k + 1] < 0.0:
                    add_quad(
                        [(i - 1, j - 1, k), (i, j - 1, k), (i, j, k), (i - 1, j, k)],
                        values[i, j, k] < values[i, j, k + 1],
                    )

    if not quads:
        raise RuntimeError("Native retopo found surface cells but produced no quads")

    return np.asarray(vertices, dtype=np.float32), np.asarray(quads, dtype=np.int64)


def _orient_quads(vertices: np.ndarray, quads: np.ndarray, center: np.ndarray):
    oriented = quads.copy()
    for idx, quad in enumerate(oriented):
        pts = vertices[quad]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if np.dot(normal, pts.mean(axis=0) - center) < 0.0:
            oriented[idx] = quad[::-1]
    return oriented


def _triangulate_quads(vertices: np.ndarray, quads: np.ndarray):
    tris = np.empty((quads.shape[0] * 2, 3), dtype=np.int64)
    tris[0::2] = quads[:, [0, 1, 2]]
    tris[1::2] = quads[:, [0, 2, 3]]
    return np.asarray(vertices, dtype=np.float32), tris


def retopologize(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: Optional[int] = None,
    max_cells: int = 96,
    project_to_source: bool = True,
    verbose: bool = False,
) -> RetopoResult:
    source = _clean_mesh(vertices, faces)
    if source.faces.size == 0:
        raise RuntimeError("Native retopo received an empty mesh")

    target_triangles = _auto_target(len(source.faces), target_faces)
    local_vertices, center, axes = _principal_frame(np.asarray(source.vertices))
    local_faces = np.asarray(source.faces, dtype=np.int64)
    local_mesh = _clean_mesh(local_vertices, local_faces)
    area = float(max(local_mesh.area, 1e-8))

    dims, target_quads = _grid_for_target(
        np.asarray(local_mesh.vertices), area, target_triangles, max_cells=max_cells
    )
    bounds_min = np.asarray(local_mesh.vertices).min(axis=0)
    bounds_max = np.asarray(local_mesh.vertices).max(axis=0)
    pad = np.maximum((bounds_max - bounds_min) * 0.08, 1e-3)
    bounds_min = bounds_min - pad
    bounds_max = bounds_max + pad

    if verbose:
        print(
            "Native retopo surface net: "
            f"target={target_triangles} triangles, grid={tuple(int(v) for v in dims)}"
        )

    scene = _make_scene(np.asarray(local_mesh.vertices), local_faces)
    values, _axes, spacing = _sample_sdf_grid(scene, bounds_min, bounds_max, dims)
    net_vertices, quads = _extract_surface_net(values, bounds_min, spacing)

    if project_to_source:
        net_vertices = _closest_points(scene, net_vertices)

    quads = _orient_quads(net_vertices, quads, np.asarray(local_mesh.vertices).mean(axis=0))
    world_vertices = net_vertices @ axes.T + center
    tri_vertices, tri_faces = _triangulate_quads(world_vertices, quads)

    stats = {
        "target_faces": int(target_triangles),
        "target_quads": int(target_quads),
        "grid": tuple(int(v) for v in dims),
        "source_faces": int(len(source.faces)),
        "quad_faces": int(len(quads)),
        "tri_faces": int(len(tri_faces)),
        "vertices": int(len(tri_vertices)),
    }
    return RetopoResult(
        vertices=tri_vertices,
        faces=tri_faces,
        quads=quads.astype(np.int64),
        stats=stats,
    )


def write_quad_obj(path: str, vertices: np.ndarray, quads: np.ndarray):
    with open(path, "w") as f:
        f.write("# SAM-3D native surface-net retopo source\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for q in quads:
            f.write(f"f {int(q[0]) + 1} {int(q[1]) + 1} {int(q[2]) + 1} {int(q[3]) + 1}\n")
