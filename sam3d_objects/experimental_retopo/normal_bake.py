from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh
from PIL import Image


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def smooth_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    points = vertices[faces]
    face_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float32)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid, 2] = 1.0
    return normals


def seamless_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    weld_digits: int = 7,
) -> np.ndarray:
    """Calculate one smooth normal per position and map it across UV duplicates."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    _, inverse = np.unique(
        np.round(vertices, decimals=weld_digits),
        axis=0,
        return_inverse=True,
    )
    points = vertices[faces]
    face_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    welded = np.zeros((int(inverse.max(initial=-1)) + 1, 3), dtype=np.float32)
    for corner in range(3):
        np.add.at(welded, inverse[faces[:, corner]], face_normals)
    lengths = np.linalg.norm(welded, axis=1)
    valid = lengths > 1e-12
    welded[valid] /= lengths[valid, None]
    welded[~valid, 2] = 1.0
    return welded[inverse]


def transfer_surface_normals(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
) -> np.ndarray:
    """Transfer interpolated smooth normals from a dense guide to a retopo mesh."""
    source_vertices = np.asarray(source_vertices, dtype=np.float32)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    target_vertices = np.asarray(target_vertices, dtype=np.float32)
    source_normals = smooth_vertex_normals(source_vertices, source_faces)
    scene = _make_scene(source_vertices, source_faces)
    transferred = _closest_smooth_normals(
        scene,
        target_vertices,
        source_faces,
        source_normals,
    )
    target_normals = smooth_vertex_normals(target_vertices, target_faces)
    opposite = np.einsum("ij,ij->i", transferred, target_normals) < 0.0
    transferred[opposite] *= -1.0
    return _normalize(transferred)


def _clean_reference(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    finite = np.isfinite(vertices).all(axis=1)
    valid = (
        (faces >= 0).all(axis=1)
        & (faces < vertices.shape[0]).all(axis=1)
        & finite[faces].all(axis=1)
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces[valid], process=False)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits_vertex=8)
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    components = mesh.split(only_watertight=False)
    if components:
        mesh = max(components, key=lambda component: len(component.faces))
    if len(mesh.faces) == 0:
        raise RuntimeError("Experimental normal bake received an empty reference mesh")
    return mesh


def _make_scene(vertices: np.ndarray, faces: np.ndarray):
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64, copy=False))
    legacy.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32, copy=False))
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene


def _normalize(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    output = vectors.copy()
    valid = lengths > 1e-12
    output[valid] /= lengths[valid, None]
    output[~valid] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return output.astype(np.float32, copy=False)


def _gaussian_surface_normals(scales: np.ndarray, rotations: np.ndarray):
    rotations = np.asarray(rotations, dtype=np.float32)
    rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-12)
    w, x, y, z = rotations.T
    matrices = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    minor = np.argmin(np.asarray(scales), axis=1)
    return matrices[np.arange(matrices.shape[0]), :, minor]


def _rasterize_uv_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    vertex_normals: np.ndarray,
    size: int,
):
    pixel_count = size * size
    positions = np.zeros((pixel_count, 3), dtype=np.float32)
    normals = np.zeros((pixel_count, 3), dtype=np.float32)
    face_ids = np.full(pixel_count, -1, dtype=np.int32)
    uv_pixels = np.asarray(uvs, dtype=np.float32) * float(size - 1)
    epsilon = -1e-5

    for face_index, face in enumerate(faces):
        triangle = uv_pixels[face]
        min_x = max(0, int(np.floor(triangle[:, 0].min())))
        max_x = min(size - 1, int(np.ceil(triangle[:, 0].max())))
        min_y = max(0, int(np.floor(triangle[:, 1].min())))
        max_y = min(size - 1, int(np.ceil(triangle[:, 1].max())))
        if min_x > max_x or min_y > max_y:
            continue

        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) < 1e-10:
            continue
        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(xs, ys)
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= epsilon) & (w1 >= epsilon) & (w2 >= epsilon)
        if not bool(inside.any()):
            continue

        local_y, local_x = np.nonzero(inside)
        px = local_x + min_x
        py = local_y + min_y
        flat = py * size + px
        barycentric = np.stack(
            [w0[inside], w1[inside], w2[inside]], axis=1
        ).astype(np.float32, copy=False)
        positions[flat] = barycentric @ vertices[face]
        normals[flat] = barycentric @ vertex_normals[face]
        face_ids[flat] = face_index

    covered = np.flatnonzero(face_ids >= 0)
    normals[covered] = _normalize(normals[covered])
    return positions, normals, face_ids, covered


def bake_gaussian_color_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    vertex_normals: np.ndarray,
    splat_points: np.ndarray,
    splat_colors: np.ndarray,
    splat_opacity: np.ndarray,
    splat_scales: np.ndarray,
    splat_rotations: np.ndarray,
    texture_size: int = 1024,
):
    """Bake continuous color from local Gaussian neighborhoods after retopology."""
    size = max(64, int(texture_size))
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float32)
    vertex_normals = _normalize(np.asarray(vertex_normals, dtype=np.float32))
    positions, low_normals, face_ids, covered = _rasterize_uv_surface(
        vertices, faces, uvs, vertex_normals, size
    )
    if covered.shape[0] == 0:
        raise RuntimeError("Experimental color bake found no covered UV texels")

    points = np.asarray(splat_points, dtype=np.float32)
    colors = np.clip(np.asarray(splat_colors, dtype=np.float32), 0.0, 1.0)
    opacity = np.asarray(splat_opacity, dtype=np.float32).reshape(-1)
    normals = _gaussian_surface_normals(splat_scales, splat_rotations)
    center = points.mean(axis=0)
    outward = np.einsum("ij,ij->i", normals, points - center) < 0.0
    normals[outward] *= -1.0
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & np.isfinite(opacity)
        & (opacity >= _env_float("SAM3D_EXPERIMENTAL_COLOR_OPACITY", 0.05))
    )
    points = points[valid]
    colors = colors[valid]
    opacity = opacity[valid]
    normals = normals[valid]
    tree = cKDTree(points)
    neighbors = max(4, int(_env_float("SAM3D_EXPERIMENTAL_COLOR_NEIGHBORS", 48)))
    neighbors = min(neighbors, points.shape[0])
    chunk_size = 100_000
    baked = np.zeros((covered.shape[0], 3), dtype=np.float32)
    for start in range(0, covered.shape[0], chunk_size):
        stop = min(covered.shape[0], start + chunk_size)
        query = positions[covered[start:stop]]
        query_normals = low_normals[covered[start:stop]]
        distances, indices = tree.query(query, k=neighbors, workers=-1)
        if neighbors == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        bandwidth = np.maximum(distances[:, min(neighbors - 1, neighbors // 2)], 1e-6)
        spatial = np.exp(-0.5 * np.square(distances / bandwidth[:, None]))
        agreement = np.einsum("ijk,ik->ij", normals[indices], query_normals)
        side_weight = np.square(np.clip(agreement, 0.0, 1.0))
        weights = spatial * opacity[indices] * side_weight
        weight_sum = weights.sum(axis=1)
        weak = weight_sum < 1e-8
        if weak.any():
            weights[weak] = spatial[weak] * opacity[indices[weak]]
            weight_sum[weak] = weights[weak].sum(axis=1)
        baked[start:stop] = np.einsum(
            "ij,ijk->ik", weights, colors[indices]
        ) / np.maximum(weight_sum[:, None], 1e-8)

    image = np.zeros((size * size, 3), dtype=np.uint8)
    image[covered] = np.clip(baked * 255.0, 0, 255).astype(np.uint8)
    image = image.reshape(size, size, 3)
    missing = (face_ids < 0).reshape(size, size).astype(np.uint8)
    image = cv2.inpaint(image, missing, 3, cv2.INPAINT_TELEA)
    return np.ascontiguousarray(image[::-1])


def _face_tangent_frames(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray):
    points = vertices[faces]
    texcoords = uvs[faces]
    edge1 = points[:, 1] - points[:, 0]
    edge2 = points[:, 2] - points[:, 0]
    duv1 = texcoords[:, 1] - texcoords[:, 0]
    duv2 = texcoords[:, 2] - texcoords[:, 0]
    determinant = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    safe = np.where(np.abs(determinant) > 1e-12, determinant, 1.0)
    tangents = (
        edge1 * duv2[:, 1, None] - edge2 * duv1[:, 1, None]
    ) / safe[:, None]
    bitangents = (
        edge2 * duv1[:, 0, None] - edge1 * duv2[:, 0, None]
    ) / safe[:, None]
    face_normals = _normalize(np.cross(edge1, edge2))

    invalid = np.abs(determinant) <= 1e-12
    if bool(invalid.any()):
        helper = np.zeros_like(face_normals[invalid])
        helper[:, 0] = 1.0
        parallel = np.abs(face_normals[invalid, 0]) > 0.8
        helper[parallel] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        tangents[invalid] = np.cross(helper, face_normals[invalid])
        bitangents[invalid] = np.cross(face_normals[invalid], tangents[invalid])

    tangents = _normalize(tangents)
    bitangents = _normalize(bitangents)
    handedness = np.sign(
        np.einsum("ij,ij->i", np.cross(face_normals, tangents), bitangents)
    ).astype(np.float32)
    handedness[handedness == 0.0] = 1.0
    return tangents, handedness


def _closest_smooth_normals(
    scene,
    points: np.ndarray,
    source_faces: np.ndarray,
    source_normals: np.ndarray,
    chunk_size: int = 200_000,
):
    output = np.empty_like(points, dtype=np.float32)
    for start in range(0, points.shape[0], chunk_size):
        stop = min(points.shape[0], start + chunk_size)
        query = o3d.core.Tensor(points[start:stop], dtype=o3d.core.Dtype.Float32)
        result = scene.compute_closest_points(query)
        primitive_ids = result["primitive_ids"].numpy().astype(np.int64, copy=False)
        primitive_uvs = result["primitive_uvs"].numpy().astype(np.float32, copy=False)
        weights = np.column_stack(
            [1.0 - primitive_uvs.sum(axis=1), primitive_uvs[:, 0], primitive_uvs[:, 1]]
        )
        output[start:stop] = np.einsum(
            "ij,ijk->ik", weights, source_normals[source_faces[primitive_ids]]
        )
    return _normalize(output)


def _add_albedo_relief(
    normal_map: np.ndarray,
    base_color: Optional[np.ndarray],
    coverage: np.ndarray,
):
    strength = max(0.0, _env_float("SAM3D_EXPERIMENTAL_ALBEDO_RELIEF", 0.08))
    if base_color is None or strength <= 0.0:
        return normal_map
    color = np.asarray(base_color, dtype=np.float32)
    if color.shape[:2] != normal_map.shape[:2]:
        color = cv2.resize(
            color,
            (normal_map.shape[1], normal_map.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    color /= 255.0
    luminance = color[..., 0] * 0.2126 + color[..., 1] * 0.7152 + color[..., 2] * 0.0722
    broad = cv2.GaussianBlur(luminance, (0, 0), sigmaX=2.0, sigmaY=2.0)
    detail = np.clip(luminance - broad, -0.12, 0.12)
    dx = cv2.Sobel(detail, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(detail, cv2.CV_32F, 0, 1, ksize=3)

    vectors = normal_map.astype(np.float32) / 127.5 - 1.0
    vectors[..., 0] -= dx * strength
    vectors[..., 1] -= dy * strength
    lengths = np.linalg.norm(vectors, axis=2, keepdims=True)
    vectors /= np.maximum(lengths, 1e-8)
    encoded = np.clip((vectors * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    encoded[~coverage] = np.asarray([128, 128, 255], dtype=np.uint8)
    return encoded


def bake_tangent_normal_map(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    vertex_normals: Optional[np.ndarray] = None,
    texture_size: int = 1024,
    base_color: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
):
    size = max(64, int(texture_size))
    reference = _clean_reference(source_vertices, source_faces)
    source_vertices = np.asarray(reference.vertices, dtype=np.float32)
    source_faces = np.asarray(reference.faces, dtype=np.int64)
    source_normals = smooth_vertex_normals(source_vertices, source_faces)
    scene = _make_scene(source_vertices, source_faces)

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float32)
    if vertex_normals is None:
        vertex_normals = smooth_vertex_normals(vertices, faces)
    else:
        vertex_normals = _normalize(np.asarray(vertex_normals, dtype=np.float32))

    positions, low_normals, face_ids, covered = _rasterize_uv_surface(
        vertices, faces, uvs, vertex_normals, size
    )
    if covered.shape[0] == 0:
        raise RuntimeError("Experimental normal bake found no covered UV texels")

    detail_normals = _closest_smooth_normals(
        scene,
        positions[covered],
        source_faces,
        source_normals,
    )
    low = low_normals[covered]
    opposite = np.einsum("ij,ij->i", detail_normals, low) < 0.0
    detail_normals[opposite] *= -1.0

    tangents, handedness = _face_tangent_frames(vertices, faces, uvs)
    tangent = tangents[face_ids[covered]]
    tangent -= low * np.einsum("ij,ij->i", tangent, low)[:, None]
    tangent = _normalize(tangent)
    bitangent = np.cross(low, tangent)
    bitangent *= handedness[face_ids[covered], None]
    bitangent = _normalize(bitangent)

    tangent_normal = np.stack(
        [
            np.einsum("ij,ij->i", detail_normals, tangent),
            np.einsum("ij,ij->i", detail_normals, bitangent),
            np.einsum("ij,ij->i", detail_normals, low),
        ],
        axis=1,
    )
    tangent_normal[:, 2] = np.maximum(tangent_normal[:, 2], 0.05)
    tangent_normal = _normalize(tangent_normal)
    normal_strength = float(
        np.clip(_env_float("SAM3D_EXPERIMENTAL_NORMAL_STRENGTH", 0.35), 0.0, 1.0)
    )
    tangent_normal *= normal_strength
    tangent_normal[:, 2] += 1.0 - normal_strength
    tangent_normal = _normalize(tangent_normal)

    image = np.empty((size * size, 3), dtype=np.uint8)
    image[:] = np.asarray([128, 128, 255], dtype=np.uint8)
    image[covered] = np.clip((tangent_normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    image = image.reshape(size, size, 3)
    coverage = (face_ids >= 0).reshape(size, size)

    # Trimesh exports v as 1-v. Mirror the image to preserve texel lookup and invert
    # tangent-space Y because dp/dv changes sign under that coordinate transform.
    image = np.ascontiguousarray(image[::-1])
    coverage = np.ascontiguousarray(coverage[::-1])
    image[..., 1] = 255 - image[..., 1]
    image = _add_albedo_relief(image, base_color, coverage)

    if output_path:
        Image.fromarray(image, mode="RGB").save(output_path)

    angular_delta = np.degrees(
        np.arccos(np.clip(tangent_normal[:, 2], -1.0, 1.0))
    )
    stats = {
        "size": size,
        "covered_texels": int(covered.shape[0]),
        "coverage": float(covered.shape[0] / float(size * size)),
        "detail_angle_p95": float(np.percentile(angular_delta, 95)),
        "strength": normal_strength,
        "albedo_relief": float(
            max(0.0, _env_float("SAM3D_EXPERIMENTAL_ALBEDO_RELIEF", 0.08))
        ),
    }
    return image, stats
