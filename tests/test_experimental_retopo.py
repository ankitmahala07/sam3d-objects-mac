import numpy as np
import trimesh

from sam3d_objects.experimental_retopo import retopologize, write_quad_obj


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

    path = tmp_path / "grid.obj"
    write_quad_obj(path, result.vertices, result.quads, result.repair_faces)
    face_lines = [line for line in path.read_text("ascii").splitlines() if line.startswith("f ")]
    assert len(face_lines) == result.quads.shape[0]
    assert all(len(line.split()) == 5 for line in face_lines)


def test_curved_surface_is_watertight():
    source = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    result = retopologize(
        np.asarray(source.vertices),
        np.asarray(source.faces),
        target_faces=800,
    )

    _assert_manifold_result(result)
    runtime = trimesh.Trimesh(result.vertices, result.faces, process=False)
    assert runtime.is_watertight


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
