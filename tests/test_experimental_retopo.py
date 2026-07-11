import os

import numpy as np
import trimesh

from sam3d_objects.experimental_retopo import (
    bake_gaussian_color_texture,
    bake_tangent_normal_map,
    retopologize,
    write_quad_obj,
)


def _assert_manifold_result(result):
    assert np.isfinite(result.vertices).all()
    assert result.faces.shape[0] > 0
    assert result.quads.shape[0] > 0
    assert result.stats["boundary_edges"] == 0
    assert result.stats["nonmanifold_edges"] == 0
    assert result.stats["components"] == 1
    assert result.stats["winding_consistent"]
    assert result.stats["surface_error_p95"] < 0.01
    assert result.stats["aspect_max"] < 100.0
    runtime = trimesh.Trimesh(result.vertices, result.faces, process=False)
    assert runtime.volume > 0.0


def test_hard_surface_grid_preserves_planes(tmp_path):
    source = trimesh.creation.box(extents=[2.0, 1.0, 0.7])
    result = retopologize(
        np.asarray(source.vertices),
        np.asarray(source.faces),
        target_faces=800,
    )

    _assert_manifold_result(result)
    assert result.repair_faces.shape[0] == 0
    assert result.stats["aspect_p95"] < 3.0
    assert result.stats["dihedral_p50"] < 1.0
    assert result.stats["adaptive_rejected"] == "transition triangle ratio"

    path = tmp_path / "grid.obj"
    write_quad_obj(path, result.vertices, result.quads, result.repair_faces)
    face_lines = [line for line in path.read_text("ascii").splitlines() if line.startswith("f ")]
    assert len(face_lines) == result.quads.shape[0] + result.repair_faces.shape[0]
    assert sum(len(line.split()) == 5 for line in face_lines) == result.quads.shape[0]


def test_curved_surface_is_watertight():
    source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    result = retopologize(
        np.asarray(source.vertices),
        np.asarray(source.faces),
        target_faces=800,
    )

    _assert_manifold_result(result)
    assert "adaptive_rejected" in result.stats
    assert result.stats["dihedral_p95"] < 15.0
    runtime = trimesh.Trimesh(result.vertices, result.faces, process=False)
    assert runtime.is_watertight


def test_adaptive_curved_patch_stays_manifold():
    names = {
        "SAM3D_EXPERIMENTAL_ADAPTIVE_ERROR": "0.03",
        "SAM3D_EXPERIMENTAL_ADAPTIVE_MAX_FRACTION": "0.05",
        "SAM3D_EXPERIMENTAL_ADAPTIVE_DIHEDRAL_INCREASE": "2",
        "SAM3D_EXPERIMENTAL_ADAPTIVE_MAX_TRANSITION_RATIO": "0.5",
    }
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(names)
    try:
        source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        result = retopologize(
            np.asarray(source.vertices),
            np.asarray(source.faces),
            target_faces=800,
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    _assert_manifold_result(result)
    assert result.stats["adaptive_refined_quads"] > 0
    assert result.stats["adaptive_transition_faces"] > 0
    assert result.stats["adaptive_transition_ratio"] <= 0.5
    assert result.stats["adaptive_rejected"] is None


def test_v2_keeps_clean_hard_surface_unchanged():
    source = trimesh.creation.box(extents=[2.0, 1.0, 0.7])
    source_vertices = np.asarray(source.vertices)
    source_faces = np.asarray(source.faces)
    baseline = retopologize(source_vertices, source_faces, target_faces=800)
    result = retopologize(
        source_vertices,
        source_faces,
        target_faces=800,
        smooth=True,
    )

    _assert_manifold_result(result)
    assert result.stats["surface_style"] == "v2-safe-fallback"
    assert not result.stats["v2_accepted"]
    assert np.array_equal(result.quads, baseline.quads)
    assert result.faces.shape == baseline.faces.shape
    assert np.allclose(result.vertices, baseline.vertices)


def test_v2_improves_curved_surface_without_changing_topology():
    source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    source_vertices = np.asarray(source.vertices)
    source_faces = np.asarray(source.faces)
    baseline = retopologize(source_vertices, source_faces, target_faces=800)
    result = retopologize(
        source_vertices,
        source_faces,
        target_faces=800,
        smooth=True,
    )

    _assert_manifold_result(result)
    assert result.stats["surface_style"] == "smooth-v2"
    assert result.stats["v2_accepted"]
    assert result.stats["v2_profile"] in ("gentle", "balanced", "strong")
    assert np.array_equal(result.quads, baseline.quads)
    assert result.faces.shape == baseline.faces.shape
    assert result.stats["dihedral_p50"] < baseline.stats["dihedral_p50"]
    assert result.stats["surface_error_p95"] <= baseline.stats["surface_error_p95"] * 1.1


def test_open_multishell_source_becomes_one_body():
    body = trimesh.creation.box(extents=[2.0, 1.0, 0.7])
    body.update_faces(np.arange(len(body.faces) - 2))
    detail = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
    detail.apply_translation([3.0, 3.0, 3.0])
    source = trimesh.util.concatenate([body, detail])

    result = retopologize(
        np.asarray(source.vertices),
        np.asarray(source.faces),
        target_faces=800,
    )

    _assert_manifold_result(result)
    assert result.stats["removed_source_components"] == 1
    runtime = trimesh.Trimesh(result.vertices, result.faces, process=False)
    assert len(runtime.split(only_watertight=False)) == 1


def test_normal_bake_transfers_reference_curvature(tmp_path):
    axis = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    zz = 0.12 * np.exp(-5.0 * (xx * xx + yy * yy))
    source_vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    source_faces = []
    width = axis.shape[0]
    for y in range(width - 1):
        for x in range(width - 1):
            a = y * width + x
            b = a + 1
            c = a + width
            d = c + 1
            source_faces.extend([(a, b, d), (a, d, c)])

    vertices = np.asarray(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    output_path = tmp_path / "normal.png"
    image, stats = bake_tangent_normal_map(
        source_vertices,
        np.asarray(source_faces, dtype=np.int64),
        vertices,
        faces,
        uvs,
        texture_size=96,
        output_path=output_path,
    )

    assert output_path.is_file()
    assert image.shape == (96, 96, 3)
    assert stats["coverage"] > 0.95
    assert stats["detail_angle_p95"] > 0.5
    assert np.std(image[..., :2].astype(np.float32)) > 1.0


def test_gaussian_color_bake_runs_after_final_surface():
    vertices = np.asarray(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    normals = np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (4, 1))
    axis = np.linspace(-1.0, 1.0, 15, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    colors = np.column_stack(
        [(points[:, 0] + 1.0) * 0.5, (points[:, 1] + 1.0) * 0.5, np.full(points.shape[0], 0.25)]
    ).astype(np.float32)
    opacity = np.ones(points.shape[0], dtype=np.float32)
    scales = np.tile(np.asarray([[0.08, 0.08, 0.01]], dtype=np.float32), (points.shape[0], 1))
    rotations = np.tile(np.asarray([[1, 0, 0, 0]], dtype=np.float32), (points.shape[0], 1))

    texture = bake_gaussian_color_texture(
        vertices,
        faces,
        uvs,
        normals,
        points,
        colors,
        opacity,
        scales,
        rotations,
        texture_size=64,
    )

    assert texture.shape == (64, 64, 3)
    assert texture[..., 0].std() > 40
    assert texture[..., 1].std() > 40
    assert 45 < float(texture[..., 2].mean()) < 80
