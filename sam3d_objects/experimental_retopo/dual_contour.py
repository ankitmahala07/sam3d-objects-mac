from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import os
from typing import Optional

import numpy as np
import open3d as o3d
import trimesh


@dataclass
class ExperimentalRetopoResult:
    vertices: np.ndarray
    faces: np.ndarray
    quads: np.ndarray
    repair_faces: np.ndarray
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

_CELL_EDGES = (
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

_CUBE_TETRAHEDRA = (
    (0, 1, 3, 7),
    (0, 3, 2, 7),
    (0, 2, 6, 7),
    (0, 6, 4, 7),
    (0, 4, 5, 7),
    (0, 5, 1, 7),
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _auto_target(face_count: int, target_faces: Optional[int]) -> int:
    if target_faces is not None:
        return max(500, int(target_faces))
    if face_count <= 20_000:
        return min(face_count, 2_000)
    return 10_000


def _clean_source(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    finite_vertices = np.isfinite(vertices).all(axis=1)
    valid_faces = (
        (faces >= 0).all(axis=1)
        & (faces < vertices.shape[0]).all(axis=1)
        & finite_vertices[faces].all(axis=1)
    )
    faces = faces[valid_faces]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits_vertex=8)
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise RuntimeError("Experimental retopo received an empty source mesh")
    return mesh


def _largest_source_component(mesh: trimesh.Trimesh):
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh, 0
    largest = max(components, key=lambda component: len(component.faces))
    return largest, len(components) - 1


def _close_boundary_loops(mesh: trimesh.Trimesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edge_count = {}
    edge_direction = {}
    for triangle in faces:
        for a, b in zip(triangle, np.roll(triangle, -1)):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            edge_count[edge] = edge_count.get(edge, 0) + 1
            edge_direction.setdefault(edge, (int(a), int(b)))
    if any(count > 2 for count in edge_count.values()):
        return mesh, 0

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    if not boundary_edges:
        return mesh, 0
    adjacency = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return mesh, 0

    unvisited = set(boundary_edges)
    loops = []
    while unvisited:
        start, current = next(iter(unvisited))
        previous = start
        loop = [start, current]
        unvisited.discard((min(start, current), max(start, current)))
        while current != start:
            candidates = [value for value in adjacency[current] if value != previous]
            if len(candidates) != 1:
                return mesh, 0
            next_vertex = candidates[0]
            edge = (min(current, next_vertex), max(current, next_vertex))
            if next_vertex != start and edge not in unvisited:
                return mesh, 0
            unvisited.discard(edge)
            previous, current = current, next_vertex
            if current != start:
                loop.append(current)
        loops.append(loop)

    centers = np.asarray([vertices[np.asarray(loop)].mean(axis=0) for loop in loops])
    output_vertices = np.concatenate([vertices, centers], axis=0)
    caps = []
    for loop_index, loop in enumerate(loops):
        center_index = vertices.shape[0] + loop_index
        for a, b in zip(loop, np.roll(loop, -1)):
            direction = edge_direction[(min(int(a), int(b)), max(int(a), int(b)))]
            if direction == (int(a), int(b)):
                caps.append((int(b), int(a), center_index))
            else:
                caps.append((int(a), int(b), center_index))
    closed = trimesh.Trimesh(
        vertices=output_vertices,
        faces=np.concatenate([faces, np.asarray(caps, dtype=np.int64)], axis=0),
        process=False,
    )
    return closed, len(loops)


def _principal_frame(vertices: np.ndarray):
    center = vertices.mean(axis=0)
    centered = vertices - center
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    axes = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = centered @ axes
    return local.astype(np.float32), center.astype(np.float32), axes.astype(np.float32)


def _make_scene(vertices: np.ndarray, faces: np.ndarray):
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64, copy=False))
    legacy.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32, copy=False))
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene


def _query_signed_distance(scene, points: np.ndarray, chunk_size: int) -> np.ndarray:
    output = []
    for start in range(0, points.shape[0], chunk_size):
        query = o3d.core.Tensor(
            points[start : start + chunk_size].astype(np.float32, copy=False),
            dtype=o3d.core.Dtype.Float32,
        )
        output.append(scene.compute_signed_distance(query).numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def _query_unsigned_distance(scene, points: np.ndarray, chunk_size: int) -> np.ndarray:
    output = []
    for start in range(0, points.shape[0], chunk_size):
        query = o3d.core.Tensor(
            points[start : start + chunk_size].astype(np.float32, copy=False),
            dtype=o3d.core.Dtype.Float32,
        )
        output.append(scene.compute_distance(query).numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def _query_closest(scene, points: np.ndarray, chunk_size: int = 200_000):
    closest = []
    normals = []
    for start in range(0, points.shape[0], chunk_size):
        query = o3d.core.Tensor(
            points[start : start + chunk_size].astype(np.float32, copy=False),
            dtype=o3d.core.Dtype.Float32,
        )
        result = scene.compute_closest_points(query)
        closest.append(result["points"].numpy())
        normals.append(result["primitive_normals"].numpy())
    return (
        np.concatenate(closest).astype(np.float32, copy=False),
        np.concatenate(normals).astype(np.float32, copy=False),
    )


def _grid_spec(
    vertices: np.ndarray,
    area: float,
    target_faces: int,
    preserve_thin_features: bool = False,
):
    target_quads = max(250, int(math.ceil(target_faces / 2)))
    cell_size = math.sqrt(max(area, 1e-8) / target_quads)
    cell_size *= _env_float("SAM3D_EXPERIMENTAL_CELL_SCALE", 1.0)

    source_min = vertices.min(axis=0)
    source_max = vertices.max(axis=0)
    source_extent = np.maximum(source_max - source_min, 1e-6)
    min_thickness_cells = (
        max(4, _env_int("SAM3D_EXPERIMENTAL_MIN_THICKNESS_CELLS", 16))
        if preserve_thin_features
        else 4
    )
    cell_size = min(cell_size, float(source_extent.min()) / min_thickness_cells)
    padding_cells = max(2, _env_int("SAM3D_EXPERIMENTAL_PADDING_CELLS", 2))
    bounds_min = source_min - cell_size * padding_cells
    bounds_max = source_max + cell_size * padding_cells

    max_axis = max(24, _env_int("SAM3D_EXPERIMENTAL_MAX_AXIS", 144))
    max_points = max(200_000, _env_int("SAM3D_EXPERIMENTAL_MAX_GRID_POINTS", 2_500_000))
    extent = bounds_max - bounds_min
    dims = np.maximum(6, np.ceil(extent / max(cell_size, 1e-6)).astype(np.int32))

    scale = max(float(dims.max()) / max_axis, 1.0)
    point_count = int(np.prod(dims.astype(np.int64) + 1))
    if point_count > max_points:
        scale = max(scale, (point_count / max_points) ** (1.0 / 3.0))
    if scale > 1.0:
        cell_size *= scale
        bounds_min = source_min - cell_size * padding_cells
        bounds_max = source_max + cell_size * padding_cells
        extent = bounds_max - bounds_min
        dims = np.maximum(6, np.ceil(extent / cell_size).astype(np.int32))

    spacing = extent / dims
    return bounds_min.astype(np.float32), bounds_max.astype(np.float32), dims, spacing.astype(np.float32)


def _exterior_flood(free_space: np.ndarray):
    nx, ny, nz = free_space.shape
    outside = np.zeros_like(free_space, dtype=bool)
    boundary = np.zeros_like(free_space, dtype=bool)
    boundary[0] = True
    boundary[-1] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    seeds = np.argwhere(boundary & free_space)
    queue = deque((int(i), int(j), int(k)) for i, j, k in seeds)
    outside[boundary & free_space] = True
    while queue:
        i, j, k = queue.popleft()
        for ni, nj, nk in (
            (i - 1, j, k),
            (i + 1, j, k),
            (i, j - 1, k),
            (i, j + 1, k),
            (i, j, k - 1),
            (i, j, k + 1),
        ):
            if (
                0 <= ni < nx
                and 0 <= nj < ny
                and 0 <= nk < nz
                and free_space[ni, nj, nk]
                and not outside[ni, nj, nk]
            ):
                outside[ni, nj, nk] = True
                queue.append((ni, nj, nk))
    return outside


def _signed_from_unsigned(unsigned: np.ndarray, spacing: np.ndarray):
    barrier_scale = _env_float("SAM3D_EXPERIMENTAL_SHELL_BAND", 0.55)
    max_barrier_scale = max(
        barrier_scale,
        _env_float("SAM3D_EXPERIMENTAL_MAX_SHELL_BAND", 2.25),
    )
    labels = None
    while barrier_scale <= max_barrier_scale + 1e-6:
        barrier = unsigned <= float(np.max(spacing)) * barrier_scale
        free_space = ~barrier
        outside = _exterior_flood(free_space)
        inside_free = free_space & ~outside
        if bool(inside_free.any()):
            labels = np.zeros(unsigned.shape, dtype=np.int8)
            labels[outside] = 1
            labels[inside_free] = -1
            break
        barrier_scale += 0.4
    if labels is None:
        raise RuntimeError(
            "Experimental retopo could not infer an enclosed interior from the decoded surface"
        )

    unknown = labels == 0
    propagation_limit = int(sum(unsigned.shape))
    for _ in range(propagation_limit):
        if not bool(unknown.any()):
            break
        score = np.zeros(labels.shape, dtype=np.int16)
        score[1:] += labels[:-1]
        score[:-1] += labels[1:]
        score[:, 1:] += labels[:, :-1]
        score[:, :-1] += labels[:, 1:]
        score[:, :, 1:] += labels[:, :, :-1]
        score[:, :, :-1] += labels[:, :, 1:]
        fill = unknown & (score != 0)
        if not bool(fill.any()):
            break
        labels[fill] = np.sign(score[fill]).astype(np.int8)
        unknown = labels == 0
    labels[unknown] = 1
    return unsigned * labels.astype(np.float32)


def _sample_grid(scene, bounds_min, bounds_max, dims, robust_sign=False):
    axes = [
        np.linspace(bounds_min[i], bounds_max[i], int(dims[i]) + 1, dtype=np.float32)
        for i in range(3)
    ]
    chunk_size = max(10_000, _env_int("SAM3D_EXPERIMENTAL_QUERY_CHUNK", 200_000))
    values = np.empty(
        (int(dims[0]) + 1, int(dims[1]) + 1, int(dims[2]) + 1),
        dtype=np.float32,
    )
    yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), axis=-1).reshape(-1, 2)
    for x_index, x_value in enumerate(axes[0]):
        points = np.empty((yz.shape[0], 3), dtype=np.float32)
        points[:, 0] = x_value
        points[:, 1:] = yz
        query = _query_unsigned_distance if robust_sign else _query_signed_distance
        values[x_index] = query(scene, points, chunk_size).reshape(len(axes[1]), len(axes[2]))
    spacing = np.asarray([axis[1] - axis[0] for axis in axes], dtype=np.float32)
    if robust_sign:
        values = _signed_from_unsigned(values, spacing)
    gradients = np.stack(np.gradient(values, *spacing, edge_order=1), axis=-1).astype(np.float32)
    return values, gradients, axes, spacing


def _cell_corner_values(values: np.ndarray, i: int, j: int, k: int):
    return values[
        i + _CORNERS[:, 0],
        j + _CORNERS[:, 1],
        k + _CORNERS[:, 2],
    ]


def _cell_corner_gradients(gradients: np.ndarray, i: int, j: int, k: int):
    return gradients[
        i + _CORNERS[:, 0],
        j + _CORNERS[:, 1],
        k + _CORNERS[:, 2],
    ]


def _solve_qef(points: np.ndarray, normals: np.ndarray, cell_min: np.ndarray, spacing: np.ndarray):
    lengths = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(normals).all(axis=1) & (lengths > 1e-8)
    points = points[valid]
    normals = normals[valid] / lengths[valid, None]
    center = cell_min + spacing * 0.5
    if points.shape[0] < 3:
        return points.mean(axis=0) if points.shape[0] else center, 0.0

    regularization = max(1e-8, _env_float("SAM3D_EXPERIMENTAL_QEF_REGULARIZATION", 4.0))
    root_regularization = math.sqrt(regularization)
    matrix = np.concatenate([normals, np.eye(3, dtype=np.float32) * root_regularization])
    target = np.concatenate(
        [np.einsum("ij,ij->i", normals, points), center * root_regularization]
    )
    try:
        position = np.linalg.lstsq(matrix, target, rcond=1e-5)[0]
        singular = np.linalg.svd(normals, compute_uv=False)
    except np.linalg.LinAlgError:
        return points.mean(axis=0), 0.0

    margin = spacing * 0.08
    position = np.clip(position, cell_min - margin, cell_min + spacing + margin)
    feature_strength = 0.0
    if singular.shape[0] >= 2 and singular[0] > 1e-8:
        feature_strength = float(np.clip(singular[1] / singular[0], 0.0, 1.0))
    return position.astype(np.float32), feature_strength


def _extract_vertices(values, gradients, bounds_min, spacing):
    v000 = values[:-1, :-1, :-1]
    corner_stack = np.stack(
        [
            v000,
            values[1:, :-1, :-1],
            values[:-1, 1:, :-1],
            values[1:, 1:, :-1],
            values[:-1, :-1, 1:],
            values[1:, :-1, 1:],
            values[:-1, 1:, 1:],
            values[1:, 1:, 1:],
        ],
        axis=-1,
    )
    active = (corner_stack.min(axis=-1) <= 0.0) & (corner_stack.max(axis=-1) >= 0.0)
    active &= (corner_stack < 0.0).any(axis=-1) & (corner_stack > 0.0).any(axis=-1)
    active_cells = np.argwhere(active)
    if active_cells.shape[0] == 0:
        raise RuntimeError("Experimental retopo found no signed-distance surface cells")

    cell_index = np.full(active.shape, -1, dtype=np.int32)
    output = np.empty((active_cells.shape[0], 3), dtype=np.float32)
    feature_strength = np.zeros(active_cells.shape[0], dtype=np.float32)

    for vertex_index, (i, j, k) in enumerate(active_cells):
        corner_values = _cell_corner_values(values, int(i), int(j), int(k))
        corner_gradients = _cell_corner_gradients(gradients, int(i), int(j), int(k))
        cell_min = bounds_min + np.asarray([i, j, k], dtype=np.float32) * spacing
        corner_positions = cell_min + _CORNERS.astype(np.float32) * spacing
        intersections = []
        normals = []
        for corner_a, corner_b in _CELL_EDGES:
            value_a = float(corner_values[corner_a])
            value_b = float(corner_values[corner_b])
            if (value_a < 0.0) == (value_b < 0.0):
                continue
            denominator = value_a - value_b
            amount = 0.5 if abs(denominator) < 1e-12 else value_a / denominator
            amount = float(np.clip(amount, 0.0, 1.0))
            intersections.append(
                corner_positions[corner_a]
                + amount * (corner_positions[corner_b] - corner_positions[corner_a])
            )
            normals.append(
                corner_gradients[corner_a]
                + amount * (corner_gradients[corner_b] - corner_gradients[corner_a])
            )
        position, strength = _solve_qef(
            np.asarray(intersections, dtype=np.float32),
            np.asarray(normals, dtype=np.float32),
            cell_min,
            spacing,
        )
        output[vertex_index] = position
        feature_strength[vertex_index] = strength
        cell_index[int(i), int(j), int(k)] = vertex_index

    return output, feature_strength, cell_index


def _extract_quads(values: np.ndarray, cell_index: np.ndarray):
    nx, ny, nz = cell_index.shape
    quads = []

    def append_quad(cells, flip):
        ids = [int(cell_index[cell]) for cell in cells]
        if min(ids) < 0 or len(set(ids)) != 4:
            return
        if flip:
            ids = [ids[0], ids[3], ids[2], ids[1]]
        quads.append(ids)

    crossings = np.argwhere((values[:-1] * values[1:]) < 0.0)
    for i, j, k in crossings:
        if j == 0 or k == 0 or j >= ny or k >= nz:
            continue
        append_quad(
            ((i, j - 1, k - 1), (i, j, k - 1), (i, j, k), (i, j - 1, k)),
            values[i, j, k] < values[i + 1, j, k],
        )

    crossings = np.argwhere((values[:, :-1] * values[:, 1:]) < 0.0)
    for i, j, k in crossings:
        if i == 0 or k == 0 or i >= nx or k >= nz:
            continue
        append_quad(
            ((i - 1, j, k - 1), (i, j, k - 1), (i, j, k), (i - 1, j, k)),
            values[i, j, k] > values[i, j + 1, k],
        )

    crossings = np.argwhere((values[:, :, :-1] * values[:, :, 1:]) < 0.0)
    for i, j, k in crossings:
        if i == 0 or j == 0 or i >= nx or j >= ny:
            continue
        append_quad(
            ((i - 1, j - 1, k), (i, j - 1, k), (i, j, k), (i - 1, j, k)),
            values[i, j, k] < values[i, j, k + 1],
        )

    if not quads:
        raise RuntimeError("Experimental retopo generated cells but no quad faces")
    return np.asarray(quads, dtype=np.int64)


def _quad_components(quads: np.ndarray):
    parent = np.arange(quads.shape[0], dtype=np.int64)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    edge_owner = {}
    for face_index, quad in enumerate(quads):
        for a, b in zip(quad, np.roll(quad, -1)):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            previous = edge_owner.get(edge)
            if previous is None:
                edge_owner[edge] = face_index
            else:
                union(face_index, previous)
    roots = np.asarray([find(i) for i in range(quads.shape[0])], dtype=np.int64)
    _, labels, counts = np.unique(roots, return_inverse=True, return_counts=True)
    return labels, counts


def _keep_largest_component(vertices, quads, feature_strength):
    labels, counts = _quad_components(quads)
    keep_label = int(counts.argmax())
    kept_quads = quads[labels == keep_label]
    used = np.unique(kept_quads.reshape(-1))
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    return vertices[used], remap[kept_quads], feature_strength[used], int(counts.shape[0] - 1)


def _triangulate_quads(vertices: np.ndarray, quads: np.ndarray):
    def split_score(first, second):
        first_points = vertices[first]
        second_points = vertices[second]
        first_area = np.linalg.norm(
            np.cross(first_points[:, 1] - first_points[:, 0], first_points[:, 2] - first_points[:, 0]),
            axis=1,
        )
        second_area = np.linalg.norm(
            np.cross(second_points[:, 1] - second_points[:, 0], second_points[:, 2] - second_points[:, 0]),
            axis=1,
        )
        return np.minimum(first_area, second_area)

    split_ac_a = quads[:, [0, 1, 2]]
    split_ac_b = quads[:, [0, 2, 3]]
    split_bd_a = quads[:, [0, 1, 3]]
    split_bd_b = quads[:, [1, 2, 3]]
    score_ac = split_score(split_ac_a, split_ac_b)
    score_bd = split_score(split_bd_a, split_bd_b)
    prefer_ac = score_ac >= score_bd

    occupied_edges = set()
    for quad in quads:
        for a, b in zip(quad, np.roll(quad, -1)):
            occupied_edges.add((min(int(a), int(b)), max(int(a), int(b))))

    use_ac = np.empty(quads.shape[0], dtype=bool)
    for face_index, quad in enumerate(quads):
        diagonal_ac = (min(int(quad[0]), int(quad[2])), max(int(quad[0]), int(quad[2])))
        diagonal_bd = (min(int(quad[1]), int(quad[3])), max(int(quad[1]), int(quad[3])))
        ac_available = diagonal_ac not in occupied_edges
        bd_available = diagonal_bd not in occupied_edges
        if ac_available and bd_available:
            choose_ac = bool(prefer_ac[face_index])
        elif ac_available:
            choose_ac = True
        elif bd_available:
            choose_ac = False
        else:
            choose_ac = bool(prefer_ac[face_index])
        use_ac[face_index] = choose_ac
        occupied_edges.add(diagonal_ac if choose_ac else diagonal_bd)

    triangles = np.empty((quads.shape[0] * 2, 3), dtype=np.int64)
    triangles[0::2] = np.where(use_ac[:, None], split_ac_a, split_bd_a)
    triangles[1::2] = np.where(use_ac[:, None], split_ac_b, split_bd_b)
    return triangles


def _repair_nonmanifold_quad_patches(vertices, quads, source_scene):
    edge_faces = {}
    for face_index, quad in enumerate(quads):
        for a, b in zip(quad, np.roll(quad, -1)):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            edge_faces.setdefault(edge, []).append(face_index)
    bad_edges = {edge for edge, owners in edge_faces.items() if len(owners) > 2}
    if not bad_edges:
        return vertices, quads, np.empty((0, 3), dtype=np.int64), 0

    remove = np.zeros(quads.shape[0], dtype=bool)
    for edge in bad_edges:
        remove[np.asarray(edge_faces[edge], dtype=np.int64)] = True
    edge_direction = {}
    boundary_edges = []
    adjacency = {}
    max_patch_passes = max(1, _env_int("SAM3D_EXPERIMENTAL_REPAIR_PASSES", 8))
    for _ in range(max_patch_passes):
        kept_quads = quads[~remove]
        if kept_quads.shape[0] == 0:
            return vertices, quads, np.empty((0, 3), dtype=np.int64), 0
        kept_triangles = _triangulate_quads(vertices, kept_quads)
        edge_count = {}
        edge_direction = {}
        for triangle in kept_triangles:
            for a, b in zip(triangle, np.roll(triangle, -1)):
                key = (min(int(a), int(b)), max(int(a), int(b)))
                edge_count[key] = edge_count.get(key, 0) + 1
                edge_direction.setdefault(key, (int(a), int(b)))
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        adjacency = {}
        for a, b in boundary_edges:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        irregular_vertices = {
            vertex for vertex, neighbors in adjacency.items() if len(neighbors) != 2
        }
        if adjacency and not irregular_vertices:
            break
        if not irregular_vertices:
            return vertices, quads, np.empty((0, 3), dtype=np.int64), 0
        grow = (~remove) & np.any(
            np.isin(quads, np.fromiter(irregular_vertices, dtype=np.int64)), axis=1
        )
        if not bool(grow.any()):
            return vertices, quads, np.empty((0, 3), dtype=np.int64), 0
        remove |= grow
    else:
        return vertices, quads, np.empty((0, 3), dtype=np.int64), 0

    unvisited = {tuple(edge) for edge in boundary_edges}
    loops = []
    while unvisited:
        first_edge = next(iter(unvisited))
        start, current = first_edge
        previous = start
        loop = [start, current]
        unvisited.discard((min(start, current), max(start, current)))
        while current != start:
            candidates = [value for value in adjacency[current] if value != previous]
            if len(candidates) != 1:
                return vertices, quads, np.empty((0, 3), dtype=np.int64), 0
            next_vertex = candidates[0]
            edge = (min(current, next_vertex), max(current, next_vertex))
            if next_vertex != start and edge not in unvisited:
                return vertices, quads, np.empty((0, 3), dtype=np.int64), 0
            unvisited.discard(edge)
            previous, current = current, next_vertex
            if current != start:
                loop.append(current)
        loops.append(loop)

    output_vertices = vertices.copy()
    repair_faces = []
    for loop in loops:
        center_index = output_vertices.shape[0]
        center = vertices[np.asarray(loop)].mean(axis=0, keepdims=True)
        output_vertices = np.concatenate([output_vertices, center], axis=0)
        for a, b in zip(loop, np.roll(loop, -1)):
            repair_faces.append((int(a), int(b), center_index))
    repair_faces = np.asarray(repair_faces, dtype=np.int64)
    centers = output_vertices[repair_faces].mean(axis=1)
    _, source_normals = _query_closest(source_scene, centers)
    points = output_vertices[repair_faces]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    flip = np.einsum("ij,ij->i", normals, source_normals) < 0.0
    repair_faces[flip] = repair_faces[flip][:, ::-1]
    return (
        output_vertices,
        kept_quads,
        repair_faces,
        int(remove.sum()),
    )


def _split_nonmanifold_vertex_fans(vertices: np.ndarray, quads: np.ndarray):
    edge_faces = {}
    incident_faces = [set() for _ in range(vertices.shape[0])]
    for face_index, quad in enumerate(quads):
        for vertex in quad:
            incident_faces[int(vertex)].add(face_index)
        for a, b in zip(quad, np.roll(quad, -1)):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            edge_faces.setdefault(edge, []).append(face_index)

    quad_points = vertices[quads]
    quad_normals = np.cross(
        quad_points[:, 1] - quad_points[:, 0],
        quad_points[:, 3] - quad_points[:, 0],
    )
    normal_lengths = np.linalg.norm(quad_normals, axis=1)
    valid_normals = normal_lengths > 1e-12
    quad_normals[valid_normals] /= normal_lengths[valid_normals, None]

    links = [[] for _ in range(vertices.shape[0])]
    for (a, b), owners in edge_faces.items():
        if len(owners) == 2:
            pairs = [(owners[0], owners[1])]
        else:
            remaining = list(owners)
            pairs = []
            while len(remaining) >= 2:
                best = None
                for first_index in range(len(remaining)):
                    for second_index in range(first_index + 1, len(remaining)):
                        face_a = remaining[first_index]
                        face_b = remaining[second_index]
                        score = float(np.dot(quad_normals[face_a], quad_normals[face_b]))
                        if best is None or score > best[0]:
                            best = (score, first_index, second_index)
                _, first_index, second_index = best
                face_a = remaining[first_index]
                face_b = remaining[second_index]
                pairs.append((face_a, face_b))
                del remaining[second_index]
                del remaining[first_index]
        for pair in pairs:
            links[a].append(pair)
            links[b].append(pair)

    output_vertices = [vertex.copy() for vertex in vertices]
    output_quads = quads.copy()
    split_vertices = 0
    for vertex_index, incident in enumerate(incident_faces):
        if len(incident) <= 1:
            continue
        parent = {face_index: face_index for face_index in incident}

        def find(face_index):
            while parent[face_index] != face_index:
                parent[face_index] = parent[parent[face_index]]
                face_index = parent[face_index]
            return face_index

        for face_a, face_b in links[vertex_index]:
            root_a = find(face_a)
            root_b = find(face_b)
            if root_a != root_b:
                parent[root_b] = root_a

        components = {}
        for face_index in incident:
            components.setdefault(find(face_index), []).append(face_index)
        if len(components) <= 1:
            continue
        ordered = sorted(components.values(), key=len, reverse=True)
        for face_group in ordered[1:]:
            new_vertex = len(output_vertices)
            output_vertices.append(vertices[vertex_index].copy())
            for face_index in face_group:
                locations = output_quads[face_index] == vertex_index
                output_quads[face_index, locations] = new_vertex
            split_vertices += 1

    return np.asarray(output_vertices, dtype=np.float32), output_quads, split_vertices


def _refine_repair_slivers(vertices, quads, repair_faces):
    if repair_faces.shape[0] == 0:
        return vertices, repair_faces, 0
    aspect_limit = _env_float("SAM3D_EXPERIMENTAL_REPAIR_ASPECT", 30.0)
    max_splits = max(0, _env_int("SAM3D_EXPERIMENTAL_REPAIR_SPLITS", 600))
    if max_splits == 0:
        return vertices, repair_faces, 0

    protected_edges = set()
    for quad in quads:
        for a, b in zip(quad, np.roll(quad, -1)):
            protected_edges.add((min(int(a), int(b)), max(int(a), int(b))))

    output_vertices = [vertex.copy() for vertex in vertices]
    output_faces = [list(map(int, face)) for face in repair_faces]
    split_count = 0
    for _ in range(max_splits):
        face_array = np.asarray(output_faces, dtype=np.int64)
        points = np.asarray(output_vertices, dtype=np.float32)[face_array]
        edge_lengths = np.stack(
            [
                np.linalg.norm(points[:, 1] - points[:, 0], axis=1),
                np.linalg.norm(points[:, 2] - points[:, 1], axis=1),
                np.linalg.norm(points[:, 0] - points[:, 2], axis=1),
            ],
            axis=1,
        )
        area_twice = np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            axis=1,
        )
        longest = edge_lengths.max(axis=1)
        height = np.divide(area_twice, longest, out=np.zeros_like(longest), where=longest > 0)
        aspect = np.divide(
            longest,
            height,
            out=np.full_like(longest, np.inf),
            where=height > 1e-12,
        )
        skinny = np.flatnonzero(aspect > aspect_limit)
        if skinny.size == 0:
            break

        edge_owners = {}
        for face_index, face in enumerate(face_array):
            for a, b in zip(face, np.roll(face, -1)):
                edge = (min(int(a), int(b)), max(int(a), int(b)))
                edge_owners.setdefault(edge, []).append(face_index)

        chosen_edge = None
        for face_index in skinny[np.argsort(aspect[skinny])[::-1]]:
            face = face_array[face_index]
            order = np.argsort(edge_lengths[face_index])[::-1]
            face_edges = ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
            for edge_index in order:
                a, b = face_edges[int(edge_index)]
                edge = (min(int(a), int(b)), max(int(a), int(b)))
                if edge not in protected_edges and len(edge_owners.get(edge, ())) == 2:
                    chosen_edge = edge
                    break
            if chosen_edge is not None:
                break
        if chosen_edge is None:
            break

        a, b = chosen_edge
        midpoint_index = len(output_vertices)
        output_vertices.append(
            (np.asarray(output_vertices[a]) + np.asarray(output_vertices[b])) * 0.5
        )
        owner_set = set(edge_owners[chosen_edge])
        updated_faces = []
        for face_index, face in enumerate(output_faces):
            if face_index not in owner_set:
                updated_faces.append(face)
                continue
            a_position = face.index(a)
            b_position = face.index(b)
            third = next(vertex for vertex in face if vertex not in (a, b))
            if (a_position + 1) % 3 == b_position:
                updated_faces.append([a, midpoint_index, third])
                updated_faces.append([midpoint_index, b, third])
            else:
                updated_faces.append([b, midpoint_index, third])
                updated_faces.append([midpoint_index, a, third])
        output_faces = updated_faces
        split_count += 1

    return (
        np.asarray(output_vertices, dtype=np.float32),
        np.asarray(output_faces, dtype=np.int64),
        split_count,
    )


def _orient_quads(vertices: np.ndarray, quads: np.ndarray, source_scene):
    centers = vertices[quads].mean(axis=1)
    _, source_normals = _query_closest(source_scene, centers)
    points = vertices[quads]
    quad_normals = np.cross(points[:, 1] - points[:, 0], points[:, 3] - points[:, 0])
    flip = np.einsum("ij,ij->i", quad_normals, source_normals) < 0.0
    oriented = quads.copy()
    oriented[flip] = oriented[flip][:, ::-1]
    return oriented


def _orient_triangles(vertices: np.ndarray, triangles: np.ndarray, source_scene):
    if triangles.shape[0] == 0:
        return triangles
    centers = vertices[triangles].mean(axis=1)
    _, source_normals = _query_closest(source_scene, centers)
    points = vertices[triangles]
    triangle_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    flip = np.einsum("ij,ij->i", triangle_normals, source_normals) < 0.0
    oriented = triangles.copy()
    oriented[flip] = oriented[flip][:, ::-1]
    return oriented


def _orient_connected_polygons(vertices, quads, triangles, source_scene):
    polygons = [list(map(int, polygon)) for polygon in quads]
    polygons.extend(list(map(int, polygon)) for polygon in triangles)
    if not polygons:
        return quads, triangles

    edge_owners = {}
    for polygon_index, polygon in enumerate(polygons):
        for a, b in zip(polygon, np.roll(polygon, -1)):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            edge_owners.setdefault(edge, []).append((polygon_index, int(a), int(b)))
    adjacency = [[] for _ in polygons]
    for owners in edge_owners.values():
        if len(owners) != 2:
            continue
        face_a, a0, a1 = owners[0]
        face_b, b0, b1 = owners[1]
        same_direction = a0 == b0 and a1 == b1
        adjacency[face_a].append((face_b, same_direction))
        adjacency[face_b].append((face_a, same_direction))

    flip = np.zeros(len(polygons), dtype=bool)
    visited = np.zeros(len(polygons), dtype=bool)
    for start in range(len(polygons)):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor, same_direction in adjacency[current]:
                required_flip = bool(flip[current] ^ same_direction)
                if not visited[neighbor]:
                    flip[neighbor] = required_flip
                    visited[neighbor] = True
                    queue.append(neighbor)

    for index in np.flatnonzero(flip):
        polygons[int(index)].reverse()

    centers = []
    normals = []
    for polygon in polygons:
        points = vertices[np.asarray(polygon)]
        centers.append(points.mean(axis=0))
        normals.append(np.cross(points[1] - points[0], points[-1] - points[0]))
    _, source_normals = _query_closest(source_scene, np.asarray(centers, dtype=np.float32))
    agreement = np.einsum("ij,ij->i", np.asarray(normals), source_normals)
    if float(np.median(agreement)) < 0.0:
        for polygon in polygons:
            polygon.reverse()

    quad_count = quads.shape[0]
    oriented_quads = np.asarray(polygons[:quad_count], dtype=np.int64).reshape(-1, 4)
    oriented_triangles = np.asarray(polygons[quad_count:], dtype=np.int64).reshape(-1, 3)
    return oriented_quads, oriented_triangles


def _keep_largest_polygon_component(vertices, quads, triangles):
    polygons = [tuple(map(int, polygon)) for polygon in quads]
    polygons.extend(tuple(map(int, polygon)) for polygon in triangles)
    if not polygons:
        raise RuntimeError("Experimental retopo produced no polygons")
    parent = np.arange(len(polygons), dtype=np.int64)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    edge_owner = {}
    for polygon_index, polygon in enumerate(polygons):
        for a, b in zip(polygon, np.roll(polygon, -1)):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            previous = edge_owner.get(edge)
            if previous is None:
                edge_owner[edge] = polygon_index
            else:
                union(polygon_index, previous)
    roots = np.asarray([find(index) for index in range(len(polygons))])
    unique, labels = np.unique(roots, return_inverse=True)
    weights = np.asarray([2 if len(polygon) == 4 else 1 for polygon in polygons])
    counts = np.bincount(labels, weights=weights)
    keep_label = int(counts.argmax())
    quad_labels = labels[: quads.shape[0]]
    triangle_labels = labels[quads.shape[0] :]
    kept_quads = quads[quad_labels == keep_label]
    kept_triangles = triangles[triangle_labels == keep_label]
    used_parts = []
    if kept_quads.shape[0]:
        used_parts.append(kept_quads.reshape(-1))
    if kept_triangles.shape[0]:
        used_parts.append(kept_triangles.reshape(-1))
    used = np.unique(np.concatenate(used_parts))
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    return (
        vertices[used],
        remap[kept_quads],
        remap[kept_triangles],
        int(unique.shape[0] - 1),
    )


def _marching_tetrahedra(values, bounds_min, spacing, source_scene):
    nx, ny, nz = np.asarray(values.shape, dtype=np.int32) - 1
    vertices = []
    quads = []
    triangles = []
    edge_vertices = {}
    grid_shape = (ny + 1, nz + 1)

    def grid_id(i, j, k):
        return (int(i) * grid_shape[0] + int(j)) * grid_shape[1] + int(k)

    def edge_vertex(corner_a, corner_b, corner_positions, corner_values, corner_grid_ids):
        id_a = int(corner_grid_ids[corner_a])
        id_b = int(corner_grid_ids[corner_b])
        edge = (min(id_a, id_b), max(id_a, id_b))
        cached = edge_vertices.get(edge)
        if cached is not None:
            return cached
        value_a = float(corner_values[corner_a])
        value_b = float(corner_values[corner_b])
        denominator = value_a - value_b
        amount = 0.5 if abs(denominator) < 1e-12 else value_a / denominator
        amount = float(np.clip(amount, 0.0, 1.0))
        position = corner_positions[corner_a] + amount * (
            corner_positions[corner_b] - corner_positions[corner_a]
        )
        index = len(vertices)
        vertices.append(position.astype(np.float32))
        edge_vertices[edge] = index
        return index

    v000 = values[:-1, :-1, :-1]
    corner_stack = np.stack(
        [
            v000,
            values[1:, :-1, :-1],
            values[:-1, 1:, :-1],
            values[1:, 1:, :-1],
            values[:-1, :-1, 1:],
            values[1:, :-1, 1:],
            values[:-1, 1:, 1:],
            values[1:, 1:, 1:],
        ],
        axis=-1,
    )
    active = (corner_stack < 0.0).any(axis=-1) & (corner_stack > 0.0).any(axis=-1)
    for i, j, k in np.argwhere(active):
        base = bounds_min + np.asarray([i, j, k], dtype=np.float32) * spacing
        corner_positions = base + _CORNERS.astype(np.float32) * spacing
        corner_values = _cell_corner_values(values, int(i), int(j), int(k))
        corner_grid_ids = np.asarray(
            [grid_id(i + di, j + dj, k + dk) for di, dj, dk in _CORNERS],
            dtype=np.int64,
        )
        for tetrahedron in _CUBE_TETRAHEDRA:
            tetrahedron = np.asarray(tetrahedron, dtype=np.int64)
            inside = [int(index) for index in tetrahedron if corner_values[index] < 0.0]
            outside = [int(index) for index in tetrahedron if corner_values[index] >= 0.0]
            if len(inside) == 0 or len(outside) == 0:
                continue
            if len(inside) == 1 or len(outside) == 1:
                pivot = inside[0] if len(inside) == 1 else outside[0]
                others = outside if len(inside) == 1 else inside
                triangle = [
                    edge_vertex(
                        pivot,
                        other,
                        corner_positions,
                        corner_values,
                        corner_grid_ids,
                    )
                    for other in others
                ]
                if len(set(triangle)) == 3:
                    triangles.append(triangle)
            else:
                inside_a, inside_b = inside
                outside_a, outside_b = outside
                quad = [
                    edge_vertex(inside_a, outside_a, corner_positions, corner_values, corner_grid_ids),
                    edge_vertex(inside_a, outside_b, corner_positions, corner_values, corner_grid_ids),
                    edge_vertex(inside_b, outside_b, corner_positions, corner_values, corner_grid_ids),
                    edge_vertex(inside_b, outside_a, corner_positions, corner_values, corner_grid_ids),
                ]
                if len(set(quad)) == 4:
                    quads.append(quad)

    if not vertices:
        raise RuntimeError("Experimental marching tetrahedra found no surface")
    vertices = np.asarray(vertices, dtype=np.float32)
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    quads = _orient_quads(vertices, quads, source_scene)
    triangles = _orient_triangles(vertices, triangles, source_scene)
    return _keep_largest_polygon_component(vertices, quads, triangles)


def _vertex_adjacency(vertex_count: int, quads: np.ndarray):
    adjacency = [set() for _ in range(vertex_count)]
    for quad in quads:
        for a, b in zip(quad, np.roll(quad, -1)):
            adjacency[int(a)].add(int(b))
            adjacency[int(b)].add(int(a))
    return adjacency


def _vertex_normals(vertices: np.ndarray, triangles: np.ndarray):
    points = vertices[triangles]
    face_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float32)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    return normals


def _relax_and_project(vertices, quads, feature_strength, source_scene, spacing):
    iterations = max(0, _env_int("SAM3D_EXPERIMENTAL_RELAX_ITERS", 2))
    if iterations == 0:
        return vertices
    adjacency = _vertex_adjacency(vertices.shape[0], quads)
    base_weight = np.clip(_env_float("SAM3D_EXPERIMENTAL_RELAX_WEIGHT", 0.18), 0.0, 0.5)
    projection_limit = float(np.linalg.norm(spacing)) * _env_float(
        "SAM3D_EXPERIMENTAL_PROJECTION_LIMIT", 0.35
    )
    output = vertices.copy()
    for _ in range(iterations):
        triangles = _triangulate_quads(output, quads)
        normals = _vertex_normals(output, triangles)
        candidate = output.copy()
        for index, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            neighbor_mean = output[np.fromiter(neighbors, dtype=np.int64)].mean(axis=0)
            laplacian = neighbor_mean - output[index]
            tangent = laplacian - normals[index] * np.dot(laplacian, normals[index])
            feature_lock = float(np.clip(feature_strength[index] * 1.5, 0.0, 0.9))
            candidate[index] += tangent * base_weight * (1.0 - feature_lock)

        closest, _ = _query_closest(source_scene, candidate)
        correction = closest - candidate
        lengths = np.linalg.norm(correction, axis=1)
        scale = np.minimum(1.0, projection_limit / np.maximum(lengths, 1e-12))
        feature_blend = 1.0 - np.clip(feature_strength * 0.65, 0.0, 0.65)
        output = candidate + correction * scale[:, None] * feature_blend[:, None]
    return output.astype(np.float32)


def _mesh_metrics(vertices, triangles, source_vertices, source_scene, scale):
    edges = np.sort(
        np.concatenate(
            [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
            axis=0,
        ),
        axis=1,
    )
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int((edge_counts == 1).sum())
    nonmanifold_edges = int((edge_counts > 2).sum())

    triangle_points = vertices[triangles]
    lengths = np.stack(
        [
            np.linalg.norm(triangle_points[:, 1] - triangle_points[:, 0], axis=1),
            np.linalg.norm(triangle_points[:, 2] - triangle_points[:, 1], axis=1),
            np.linalg.norm(triangle_points[:, 0] - triangle_points[:, 2], axis=1),
        ],
        axis=1,
    )
    area_twice = np.linalg.norm(
        np.cross(
            triangle_points[:, 1] - triangle_points[:, 0],
            triangle_points[:, 2] - triangle_points[:, 0],
        ),
        axis=1,
    )
    longest = lengths.max(axis=1)
    height = np.divide(area_twice, longest, out=np.zeros_like(longest), where=longest > 0)
    aspect = np.divide(
        longest,
        height,
        out=np.full_like(longest, np.inf),
        where=height > 1e-12,
    )

    output_closest, _ = _query_closest(source_scene, vertices)
    output_distance = np.linalg.norm(vertices - output_closest, axis=1)

    sample_count = min(source_vertices.shape[0], _env_int("SAM3D_EXPERIMENTAL_ERROR_SAMPLES", 20_000))
    sample_ids = np.linspace(0, source_vertices.shape[0] - 1, sample_count).astype(np.int64)
    output_scene = _make_scene(vertices, triangles)
    source_closest, _ = _query_closest(output_scene, source_vertices[sample_ids])
    source_distance = np.linalg.norm(source_vertices[sample_ids] - source_closest, axis=1)
    combined = np.concatenate([output_distance, source_distance]) / max(scale, 1e-8)
    runtime_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    component_count = len(runtime_mesh.split(only_watertight=False))

    return {
        "vertices": int(vertices.shape[0]),
        "triangles": int(triangles.shape[0]),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "components": int(component_count),
        "winding_consistent": bool(runtime_mesh.is_winding_consistent),
        "aspect_p95": float(np.percentile(aspect, 95)),
        "aspect_max": float(np.max(aspect)),
        "surface_error_p95": float(np.percentile(combined, 95)),
        "surface_error_max": float(np.max(combined)),
    }, triangles


def _build_once(
    local_vertices,
    faces,
    source_scene,
    sign_scene,
    area,
    target_faces,
    robust_sign,
    verbose,
):
    bounds_min, bounds_max, dims, spacing = _grid_spec(
        local_vertices,
        area,
        target_faces,
        preserve_thin_features=robust_sign,
    )
    if verbose:
        print(
            "Experimental dual contour: "
            f"target={target_faces:,} triangles, grid={tuple(int(value) for value in dims)}"
        )
    values, gradients, _axes, sampled_spacing = _sample_grid(
        sign_scene, bounds_min, bounds_max, dims, robust_sign=robust_sign
    )
    vertices, feature_strength, cell_index = _extract_vertices(
        values, gradients, bounds_min, sampled_spacing
    )
    quads = _extract_quads(values, cell_index)
    vertices, quads, feature_strength, removed_components = _keep_largest_component(
        vertices, quads, feature_strength
    )
    quads = _orient_quads(vertices, quads, source_scene)
    vertices = _relax_and_project(
        vertices, quads, feature_strength, source_scene, sampled_spacing
    )
    quads = _orient_quads(vertices, quads, source_scene)
    unsplit_vertices = vertices
    unsplit_quads = quads
    vertices, quads, split_vertices = _split_nonmanifold_vertex_fans(vertices, quads)
    _, split_component_counts = _quad_components(quads)
    split_edges = np.sort(
        np.concatenate(
            [
                quads[:, [0, 1]],
                quads[:, [1, 2]],
                quads[:, [2, 3]],
                quads[:, [3, 0]],
            ],
            axis=0,
        ),
        axis=1,
    )
    _, split_edge_counts = np.unique(split_edges, axis=0, return_counts=True)
    if split_component_counts.shape[0] > 1 or bool((split_edge_counts != 2).any()):
        vertices, quads, split_vertices = unsplit_vertices, unsplit_quads, 0
    vertices, quads, repair_faces, repaired_quads = _repair_nonmanifold_quad_patches(
        vertices, quads, source_scene
    )
    vertices, repair_faces, repair_splits = _refine_repair_slivers(
        vertices, quads, repair_faces
    )
    dual_faces = _triangulate_quads(vertices, quads)
    if repair_faces.shape[0]:
        dual_faces = np.concatenate([dual_faces, repair_faces], axis=0)
    dual_mesh = trimesh.Trimesh(vertices=vertices, faces=dual_faces, process=False)
    mode = "qef-dual-contour"
    dual_edges = np.sort(
        np.concatenate(
            [
                dual_faces[:, [0, 1]],
                dual_faces[:, [1, 2]],
                dual_faces[:, [2, 0]],
            ],
            axis=0,
        ),
        axis=1,
    )
    _, dual_edge_counts = np.unique(dual_edges, axis=0, return_counts=True)
    if (
        len(dual_mesh.split(only_watertight=False)) > 1
        or bool((dual_edge_counts != 2).any())
    ):
        vertices, quads, repair_faces, marching_removed = _marching_tetrahedra(
            values,
            bounds_min,
            sampled_spacing,
            source_scene,
        )
        removed_components += marching_removed
        repaired_quads = 0
        split_vertices = 0
        repair_splits = 0
        mode = "marching-tetrahedra"
    quads, repair_faces = _orient_connected_polygons(
        vertices,
        quads,
        repair_faces,
        source_scene,
    )
    return (
        vertices,
        quads,
        repair_faces,
        removed_components,
        repaired_quads,
        split_vertices,
        repair_splits,
        mode,
        tuple(int(value) for value in dims),
    )


def retopologize(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: Optional[int] = None,
    verbose: bool = False,
) -> ExperimentalRetopoResult:
    source = _clean_source(vertices, faces)
    source, removed_source_components = _largest_source_component(source)
    sign_source, closed_source_loops = _close_boundary_loops(source)
    source_needs_robust_sign = not bool(source.is_watertight and source.is_winding_consistent)
    if source_needs_robust_sign:
        sign_source = source
    source_vertices_world = np.asarray(source.vertices, dtype=np.float32)
    source_faces = np.asarray(source.faces, dtype=np.int64)
    local_vertices, center, axes = _principal_frame(source_vertices_world)
    local_source = trimesh.Trimesh(local_vertices, source_faces, process=False)
    area = float(max(local_source.area, 1e-8))
    scale = float(max(np.linalg.norm(np.ptp(local_vertices, axis=0)), 1e-8))
    source_scene = _make_scene(local_vertices, source_faces)
    sign_vertices_world = np.asarray(sign_source.vertices, dtype=np.float32)
    sign_vertices_local = (sign_vertices_world - center) @ axes
    sign_faces = np.asarray(sign_source.faces, dtype=np.int64)
    sign_scene = _make_scene(sign_vertices_local, sign_faces)
    robust_sign = source_needs_robust_sign
    if verbose and robust_sign:
        print("Experimental source is open or multi-shell; using unsigned shell flood fill")
    elif verbose and closed_source_loops:
        print(
            "Experimental source sign repair: "
            f"closed {closed_source_loops} boundary loop(s) on the main body"
        )

    requested_target = _auto_target(source_faces.shape[0], target_faces)
    effective_target = requested_target
    max_multiplier = max(1.0, _env_float("SAM3D_EXPERIMENTAL_MAX_TARGET_MULT", 4.0))
    max_target = min(
        max(requested_target, int(requested_target * max_multiplier)),
        _env_int("SAM3D_EXPERIMENTAL_MAX_FACES", 40_000),
    )
    max_attempts = max(1, _env_int("SAM3D_EXPERIMENTAL_QUALITY_ATTEMPTS", 2))
    error_limit = _env_float("SAM3D_EXPERIMENTAL_ERROR_P95", 0.015)
    attempt_stats = []
    final = None

    for attempt in range(max_attempts):
        (
            local_output,
            quads,
            repair_faces,
            removed_components,
            repaired_quads,
            split_vertices,
            repair_splits,
            mode,
            grid,
        ) = _build_once(
            local_vertices,
            source_faces,
            source_scene,
            sign_scene,
            area,
            effective_target,
            robust_sign,
            verbose,
        )
        triangles = _triangulate_quads(local_output, quads)
        if repair_faces.shape[0]:
            triangles = np.concatenate([triangles, repair_faces], axis=0)
        metrics, triangles = _mesh_metrics(
            local_output, triangles, local_vertices, source_scene, scale
        )
        metrics.update(
            {
                "quads": int(quads.shape[0]),
                "repair_triangles": int(repair_faces.shape[0]),
                "repaired_quads": int(repaired_quads),
                "split_vertices": int(split_vertices),
                "repair_splits": int(repair_splits),
                "mode": mode,
                "attempt": attempt + 1,
                "target_faces": int(effective_target),
                "grid": grid,
                "removed_components": int(removed_components),
            }
        )
        attempt_stats.append(metrics)
        final = (local_output, quads, repair_faces, triangles, metrics)

        topology_ok = (
            metrics["boundary_edges"] == 0
            and metrics["nonmanifold_edges"] == 0
            and metrics["components"] == 1
            and metrics["winding_consistent"]
        )
        quality_ok = metrics["surface_error_p95"] <= error_limit
        quality_ok = quality_ok and metrics["aspect_p95"] <= _env_float(
            "SAM3D_EXPERIMENTAL_ASPECT_P95", 12.0
        )
        if verbose:
            print(
                "Experimental quality: "
                f"{metrics['quads']:,} quads / {metrics['triangles']:,} triangles "
                f"({metrics['mode']}), "
                f"p95 error={metrics['surface_error_p95'] * 100:.3f}% of bounds, "
                f"components={metrics['components']}, boundary={metrics['boundary_edges']}, "
                f"nonmanifold={metrics['nonmanifold_edges']}"
            )
        if topology_ok and quality_ok:
            break
        if attempt + 1 >= max_attempts or effective_target >= max_target:
            break
        effective_target = min(
            max_target,
            max(int(effective_target * 1.8), int(metrics["triangles"] * 1.5)),
        )
        if verbose:
            print(f"Experimental quality gate raised the target to {effective_target:,} triangles")

    local_output, quads, repair_faces, triangles, metrics = final
    if (
        metrics["boundary_edges"] != 0
        or metrics["nonmanifold_edges"] != 0
        or metrics["components"] != 1
        or not metrics["winding_consistent"]
    ):
        raise RuntimeError(
            "Experimental retopo rejected a non-manifold result "
            f"(components={metrics['components']}, boundary={metrics['boundary_edges']}, "
            f"nonmanifold={metrics['nonmanifold_edges']}, "
            f"winding={metrics['winding_consistent']})"
        )
    reject_error_limit = _env_float("SAM3D_EXPERIMENTAL_REJECT_ERROR_P95", 0.03)
    if metrics["surface_error_p95"] > reject_error_limit:
        raise RuntimeError(
            "Experimental retopo rejected a low-fidelity result "
            f"(p95 surface error={metrics['surface_error_p95'] * 100:.3f}% of bounds)"
        )
    reject_aspect = _env_float("SAM3D_EXPERIMENTAL_REJECT_ASPECT_MAX", 100.0)
    if metrics["aspect_max"] > reject_aspect:
        raise RuntimeError(
            "Experimental retopo rejected skinny topology "
            f"(maximum triangle aspect={metrics['aspect_max']:.3f})"
        )

    world_vertices = local_output @ axes.T + center
    stats = {
        "source_vertices": int(source_vertices_world.shape[0]),
        "source_faces": int(source_faces.shape[0]),
        "removed_source_components": int(removed_source_components),
        "closed_source_loops": int(closed_source_loops),
        "requested_faces": int(requested_target),
        "effective_target_faces": int(metrics["target_faces"]),
        "attempts": attempt_stats,
        **metrics,
    }
    return ExperimentalRetopoResult(
        vertices=world_vertices.astype(np.float32),
        faces=triangles.astype(np.int64),
        quads=quads.astype(np.int64),
        repair_faces=repair_faces.astype(np.int64),
        stats=stats,
    )


def write_quad_obj(
    path: str,
    vertices: np.ndarray,
    quads: np.ndarray,
    repair_faces: Optional[np.ndarray] = None,
):
    with open(path, "w", encoding="ascii") as output:
        output.write("# Experimental in-repo quad retopology\n")
        for vertex in vertices:
            output.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for quad in quads:
            output.write(
                "f " + " ".join(str(int(index) + 1) for index in quad) + "\n"
            )
        if repair_faces is not None:
            for triangle in repair_faces:
                output.write(
                    "f " + " ".join(str(int(index) + 1) for index in triangle) + "\n"
                )
