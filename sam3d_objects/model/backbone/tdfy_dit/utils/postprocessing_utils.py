# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
import numpy as np
import torch
import utils3d
from PIL import Image
from tqdm import tqdm
import trimesh
import trimesh.visual
import xatlas
import pyvista as pv
from pymeshfix import _meshfix
import igraph
import cv2
from PIL import Image
from .random_utils import sphere_hammersley_sequence
from .render_utils import render_multiview
from ..renderers import GaussianRenderer
from ..representations import Strivec, Gaussian, MeshExtractResult
from loguru import logger

@torch.no_grad()
def _fill_holes(
    verts,
    faces,
    max_hole_size=0.04,
    max_hole_nbe=32,
    resolution=128,
    num_views=500,
    debug=False,
    verbose=False,
):
    """
    Rasterize a mesh from multiple views and remove invisible faces.
    Also includes postprocessing to:
        1. Remove connected components that are have low visibility.
        2. Mincut to remove faces at the inner side of the mesh connected to the outer side with a small hole.

    Args:
        verts (torch.Tensor): Vertices of the mesh. Shape (V, 3).
        faces (torch.Tensor): Faces of the mesh. Shape (F, 3).
        max_hole_size (float): Maximum area of a hole to fill.
        resolution (int): Resolution of the rasterization.
        num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """
    device = verts.device
    # Construct cameras on CPU (utils3d uses float64 internally, which MPS rejects), then
    # move the small [4,4] matrices to the mesh's device for rasterization.
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views)
        yaws.append(y)
        pitchs.append(p)
    yaws = torch.tensor(yaws)
    pitchs = torch.tensor(pitchs)
    radius = 2.0
    fov = torch.deg2rad(torch.tensor(40))
    projection = utils3d.torch.perspective_from_fov_xy(fov, fov, 1, 3).to(device).float()
    views = []
    for yaw, pitch in zip(yaws, pitchs):
        orig = torch.stack(
            [
                torch.sin(yaw) * torch.cos(pitch),
                torch.cos(yaw) * torch.cos(pitch),
                torch.sin(pitch),
            ]
        ).float() * radius
        view = utils3d.torch.view_look_at(
            orig,
            torch.tensor([0, 0, 0]).float(),
            torch.tensor([0, 0, 1]).float(),
        )
        views.append(view)
    views = torch.stack(views, dim=0).to(device).float()

    # Rasterize with the pure-PyTorch z-buffered rasterizer (CUDA-free).
    from ..renderers.mesh_raster_silicon import rasterize_mesh
    visblity = torch.zeros(faces.shape[0], dtype=torch.int32, device=device)
    for i in tqdm(
        range(views.shape[0]),
        total=views.shape[0],
        disable=not verbose,
        desc="Rasterizing",
    ):
        mvp = projection @ views[i]
        buffers = rasterize_mesh(verts, faces, mvp, resolution, resolution)
        face_id = buffers["face_id"][buffers["mask"]]   # already 0-based, -1 = empty
        face_id = torch.unique(face_id)
        face_id = face_id[face_id >= 0].long()
        visblity[face_id] += 1
    visblity = visblity.float() / num_views

    # Mincut
    ## construct outer faces
    edges, face2edge, edge_degrees = utils3d.torch.compute_edges(faces)
    boundary_edge_indices = torch.nonzero(edge_degrees == 1).reshape(-1)
    connected_components = utils3d.torch.compute_connected_components(
        faces, edges, face2edge
    )
    outer_face_indices = torch.zeros(
        faces.shape[0], dtype=torch.bool, device=faces.device
    )
    for i in range(len(connected_components)):
        outer_face_indices[connected_components[i]] = visblity[
            connected_components[i]
        ] > min(max(visblity[connected_components[i]].quantile(0.75).item(), 0.25), 0.5)
    outer_face_indices = outer_face_indices.nonzero().reshape(-1)

    ## construct inner faces
    inner_face_indices = torch.nonzero(visblity == 0).reshape(-1)
    if verbose:
        tqdm.write(f"Found {inner_face_indices.shape[0]} invisible faces")
    if inner_face_indices.shape[0] == 0:
        return verts, faces

    ## Construct dual graph (faces as nodes, edges as edges)
    dual_edges, dual_edge2edge = utils3d.torch.compute_dual_graph(face2edge)
    dual_edge2edge = edges[dual_edge2edge]
    dual_edges_weights = torch.norm(
        verts[dual_edge2edge[:, 0]] - verts[dual_edge2edge[:, 1]], dim=1
    )
    if verbose:
        tqdm.write(f"Dual graph: {dual_edges.shape[0]} edges")

    ## solve mincut problem
    ### construct main graph
    g = igraph.Graph()
    g.add_vertices(faces.shape[0])
    g.add_edges(dual_edges.cpu().numpy())
    g.es["weight"] = dual_edges_weights.cpu().numpy()

    ### source and target
    g.add_vertex("s")
    g.add_vertex("t")

    ### connect invisible faces to source
    g.add_edges(
        [(f, "s") for f in inner_face_indices],
        attributes={
            "weight": torch.ones(inner_face_indices.shape[0], dtype=torch.float32)
            .cpu()
            .numpy()
        },
    )

    ### connect outer faces to target
    g.add_edges(
        [(f, "t") for f in outer_face_indices],
        attributes={
            "weight": torch.ones(outer_face_indices.shape[0], dtype=torch.float32)
            .cpu()
            .numpy()
        },
    )

    ### solve mincut
    cut = g.mincut("s", "t", (np.array(g.es["weight"]) * 1000).tolist())
    remove_face_indices = torch.tensor(
        [v for v in cut.partition[0] if v < faces.shape[0]],
        dtype=torch.long,
        device=faces.device,
    )
    if verbose:
        tqdm.write(f"Mincut solved, start checking the cut")

    ### check if the cut is valid with each connected component
    to_remove_cc = utils3d.torch.compute_connected_components(
        faces[remove_face_indices]
    )
    if debug:
        tqdm.write(f"Number of connected components of the cut: {len(to_remove_cc)}")
    valid_remove_cc = []
    cutting_edges = []
    for cc in to_remove_cc:
        #### check if the connected component has low visibility
        visblity_median = visblity[remove_face_indices[cc]].median()
        if debug:
            tqdm.write(f"visblity_median: {visblity_median}")
        if visblity_median > 0.25:
            continue

        #### check if the cuting loop is small enough
        cc_edge_indices, cc_edges_degree = torch.unique(
            face2edge[remove_face_indices[cc]], return_counts=True
        )
        cc_boundary_edge_indices = cc_edge_indices[cc_edges_degree == 1]
        cc_new_boundary_edge_indices = cc_boundary_edge_indices[
            ~torch.isin(cc_boundary_edge_indices, boundary_edge_indices)
        ]
        if len(cc_new_boundary_edge_indices) > 0:
            cc_new_boundary_edge_cc = utils3d.torch.compute_edge_connected_components(
                edges[cc_new_boundary_edge_indices]
            )
            cc_new_boundary_edges_cc_center = [
                verts[edges[cc_new_boundary_edge_indices[edge_cc]]]
                .mean(dim=1)
                .mean(dim=0)
                for edge_cc in cc_new_boundary_edge_cc
            ]
            cc_new_boundary_edges_cc_area = []
            for i, edge_cc in enumerate(cc_new_boundary_edge_cc):
                _e1 = (
                    verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 0]]
                    - cc_new_boundary_edges_cc_center[i]
                )
                _e2 = (
                    verts[edges[cc_new_boundary_edge_indices[edge_cc]][:, 1]]
                    - cc_new_boundary_edges_cc_center[i]
                )
                cc_new_boundary_edges_cc_area.append(
                    torch.norm(torch.cross(_e1, _e2, dim=-1), dim=1).sum() * 0.5
                )
            if debug:
                cutting_edges.append(cc_new_boundary_edge_indices)
                tqdm.write(f"Area of the cutting loop: {cc_new_boundary_edges_cc_area}")
            if any([l > max_hole_size for l in cc_new_boundary_edges_cc_area]):
                continue

        valid_remove_cc.append(cc)

    if debug:
        face_v = verts[faces].mean(dim=1).cpu().numpy()
        vis_dual_edges = dual_edges.cpu().numpy()
        vis_colors = np.zeros((faces.shape[0], 3), dtype=np.uint8)
        vis_colors[inner_face_indices.cpu().numpy()] = [0, 0, 255]
        vis_colors[outer_face_indices.cpu().numpy()] = [0, 255, 0]
        vis_colors[remove_face_indices.cpu().numpy()] = [255, 0, 255]
        if len(valid_remove_cc) > 0:
            vis_colors[
                remove_face_indices[torch.cat(valid_remove_cc)].cpu().numpy()
            ] = [255, 0, 0]
        utils3d.io.write_ply(
            "dbg_dual.ply", face_v, edges=vis_dual_edges, vertex_colors=vis_colors
        )

        vis_verts = verts.cpu().numpy()
        vis_edges = edges[torch.cat(cutting_edges)].cpu().numpy()
        utils3d.io.write_ply("dbg_cut.ply", vis_verts, edges=vis_edges)

    if len(valid_remove_cc) > 0:
        remove_face_indices = remove_face_indices[torch.cat(valid_remove_cc)]
        mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
        mask[remove_face_indices] = 0
        faces = faces[mask]
        faces, verts = utils3d.torch.remove_unreferenced_vertices(faces, verts)
        if verbose:
            tqdm.write(f"Removed {(~mask).sum()} faces by mincut")
    else:
        if verbose:
            tqdm.write(f"Removed 0 faces by mincut")

    mesh = _meshfix.PyTMesh()
    mesh.load_array(verts.cpu().numpy(), faces.cpu().numpy())
    mesh.fill_small_boundaries(nbe=max_hole_nbe, refine=True)
    verts, faces = mesh.return_arrays()
    verts, faces = torch.tensor(
        verts, device=device, dtype=torch.float32
    ), torch.tensor(faces, device=device, dtype=torch.int32)

    return verts, faces


def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = True,
    simplify_ratio: float = 0.9,
    fill_holes: bool = True,
    fill_holes_max_hole_size: float = 0.04,
    fill_holes_max_hole_nbe: int = 32,
    fill_holes_resolution: int = 1024,
    fill_holes_num_views: int = 1000,
    remove_floaters: bool = True,
    floater_frac: float = 0.01,
    debug: bool = False,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying, removing invisible faces, and removing isolated pieces.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_hole_size (float): Maximum area of a hole to fill.
        fill_holes_max_hole_nbe (int): Maximum number of boundary edges of a hole to fill.
        fill_holes_resolution (int): Resolution of the rasterization.
        fill_holes_num_views (int): Number of views to rasterize the mesh.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(
            f"Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
        )

    # Device for the torch-based steps below (MPS / CPU / CUDA).
    _dev = "mps" if torch.backends.mps.is_available() and not torch.cuda.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Simplify
    if simplify and simplify_ratio > 0:
        mesh = pv.PolyData(
            vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1)
        )
        # pyvista.decimate requires an all-triangle mesh; triangulate first to be safe.
        mesh = mesh.triangulate()
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if verbose:
            tqdm.write(
                f"After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
            )

    # Remove invisible faces
    if fill_holes:
        vertices, faces = (
            torch.tensor(vertices).to(_dev),
            torch.tensor(faces.astype(np.int32)).to(_dev),
        )
        vertices, faces = _fill_holes(
            vertices,
            faces,
            max_hole_size=fill_holes_max_hole_size,
            max_hole_nbe=fill_holes_max_hole_nbe,
            resolution=fill_holes_resolution,
            num_views=fill_holes_num_views,
            debug=debug,
            verbose=verbose,
        )
        vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
        if verbose:
            tqdm.write(
                f"After remove invisible faces: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
            )

    # Drop small disconnected pieces ("floaters"): decimation + mincut can leave dozens
    # of tiny islands. Keep only components with at least floater_frac of the largest
    # component's face count (with a small absolute floor).
    if remove_floaters:
        ft = torch.tensor(faces.astype(np.int64), device=_dev)
        vt = torch.tensor(vertices, device=_dev).float()
        ccs = utils3d.torch.compute_connected_components(ft)
        if len(ccs) > 1:
            largest = max(len(cc) for cc in ccs)
            thresh = max(50, int(floater_frac * largest))
            keep = [cc for cc in ccs if len(cc) >= thresh]
            if keep:
                keep_idx = torch.cat(keep)
                ft = ft[keep_idx]
                ft, vt = utils3d.torch.remove_unreferenced_vertices(ft, vt)
                vertices, faces = vt.cpu().numpy(), ft.cpu().numpy()
                if verbose:
                    tqdm.write(
                        f"After floater removal: kept {len(keep)}/{len(ccs)} components, "
                        f"{vertices.shape[0]} vertices, {faces.shape[0]} faces"
                    )

    return vertices, faces


def resolve_game_target_faces(face_count, target_faces=None):
    if target_faces is not None:
        return max(4, min(int(target_faces), int(face_count)))
    if face_count <= 2500:
        return int(face_count)
    if face_count <= 20000:
        return 2000
    if face_count <= 80000:
        return 10000
    return 10000


def normalize_game_remesh_method(method):
    value = (method or "decimate").lower()
    if value in ("decimate", "stable", "existing", "quadric"):
        return "decimate"
    if value in ("experimental", "retopo", "artist", "feature", "feature-aware"):
        return "experimental"
    return "decimate"


def _polydata_from_triangles(vertices: np.ndarray, faces: np.ndarray):
    return pv.PolyData(
        vertices.astype(np.float64),
        np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1),
    ).triangulate().clean()


def _triangle_only_polydata(mesh):
    mesh = mesh.copy()
    mesh.clear_data()
    mesh = mesh.extract_surface(
        pass_pointid=False,
        pass_cellid=False,
        algorithm="dataset_surface",
    ).triangulate().clean()
    raw = np.asarray(mesh.faces)
    if raw.size == 0:
        return pv.PolyData(mesh.points)

    triangles = []
    i = 0
    while i < raw.size:
        n = int(raw[i])
        face = raw[i + 1 : i + 1 + n]
        if n == 3:
            triangles.append(face)
        i += n + 1

    if not triangles:
        return pv.PolyData(mesh.points)

    faces = np.asarray(triangles, dtype=np.int64)
    return _polydata_from_triangles(mesh.points, faces)


def _polydata_to_triangles(mesh):
    mesh = _triangle_only_polydata(mesh)
    remesh_vertices = mesh.points.astype(np.float32)
    remesh_faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    return remesh_vertices, remesh_faces


def _clean_open3d_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def _enforce_face_budget(
    vertices: np.ndarray,
    faces: np.ndarray,
    target: int,
    verbose: bool = False,
    label: str = "Final face budget",
):
    if target >= faces.shape[0]:
        return vertices, faces

    best_vertices, best_faces = vertices, faces
    for _ in range(4):
        if best_faces.shape[0] <= target:
            break
        reduction = 1.0 - (target / max(1, best_faces.shape[0]))
        reduction = min(0.99, max(0.0, reduction))
        if reduction <= 0:
            break
        try:
            mesh = _polydata_from_triangles(best_vertices, best_faces)
            mesh = mesh.decimate(
                reduction,
                volume_preservation=True,
                boundary_constraints=True,
                boundary_weight=20.0,
                progress_bar=False,
            )
            cand_vertices, cand_faces = _polydata_to_triangles(mesh)
        except Exception as exc:
            if verbose:
                tqdm.write(f"{label} failed ({exc}); keeping previous mesh")
            break

        if cand_faces.size == 0 or cand_faces.shape[0] >= best_faces.shape[0]:
            break
        best_vertices, best_faces = cand_vertices, cand_faces

    if verbose and best_faces.shape[0] != faces.shape[0]:
        tqdm.write(
            f"{label}: {vertices.shape[0]} vertices, {faces.shape[0]} faces -> "
            f"{best_vertices.shape[0]} vertices, {best_faces.shape[0]} faces"
        )
    return best_vertices, best_faces


def _quadric_game_decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces=None,
    boundary_weight: float = 10.0,
    verbose: bool = False,
    label: str = "Game remesh",
):
    import open3d as o3d

    target = resolve_game_target_faces(faces.shape[0], target_faces)
    if target >= faces.shape[0]:
        if verbose:
            tqdm.write(
                f"{label} skipped: {faces.shape[0]} faces already <= target {target}"
            )
        return vertices, faces

    if verbose:
        tqdm.write(
            f"{label}: {vertices.shape[0]} vertices, {faces.shape[0]} faces -> "
            f"target {target} faces"
        )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh = _clean_open3d_mesh(mesh)

    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target,
        boundary_weight=boundary_weight,
    )
    mesh = _clean_open3d_mesh(mesh)

    for _ in range(3):
        if len(mesh.triangles) <= target:
            break
        retry_target = max(4, int(target * 0.85))
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=retry_target,
            boundary_weight=boundary_weight,
        )
        mesh = _clean_open3d_mesh(mesh)

    remesh_vertices = np.asarray(mesh.vertices).astype(np.float32)
    remesh_faces = np.asarray(mesh.triangles).astype(np.int64)
    if remesh_vertices.size == 0 or remesh_faces.size == 0:
        if verbose:
            tqdm.write(f"{label} produced an empty mesh; keeping cleaned mesh")
        return vertices, faces
    remesh_vertices, remesh_faces = _enforce_face_budget(
        remesh_vertices,
        remesh_faces,
        target,
        verbose=verbose,
        label=f"{label} hard budget",
    )

    if verbose:
        tqdm.write(
            f"After {label.lower()}: {remesh_vertices.shape[0]} vertices, "
            f"{remesh_faces.shape[0]} faces"
        )
    return remesh_vertices, remesh_faces


def _experimental_game_retopo(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces=None,
    verbose: bool = False,
):
    target = resolve_game_target_faces(faces.shape[0], target_faces)
    if target >= faces.shape[0]:
        if verbose:
            tqdm.write(
                f"Experimental retopo skipped: {faces.shape[0]} faces already <= target {target}"
            )
        return vertices, faces

    if verbose:
        tqdm.write(
            f"Experimental retopo: {vertices.shape[0]} vertices, {faces.shape[0]} faces -> "
            f"target {target} faces"
        )

    try:
        reduction = 1.0 - (target / max(1, faces.shape[0]))
        reduction = min(0.98, max(0.0, reduction))
        mesh = _polydata_from_triangles(vertices, faces)
        mesh = mesh.compute_normals(
            cell_normals=True,
            point_normals=True,
            split_vertices=False,
            auto_orient_normals=True,
            inplace=False,
        )
        mesh = _triangle_only_polydata(mesh)
        mesh = mesh.decimate_pro(
            reduction,
            feature_angle=30.0,
            split_angle=60.0,
            splitting=True,
            pre_split_mesh=True,
            preserve_topology=True,
            boundary_vertex_deletion=False,
            progress_bar=verbose,
        )
        remesh_vertices, remesh_faces = _polydata_to_triangles(mesh)
    except Exception as exc:
        if verbose:
            tqdm.write(
                f"Experimental retopo failed ({exc}); falling back to stable decimation"
            )
        return _quadric_game_decimate(
            vertices,
            faces,
            target_faces=target,
            boundary_weight=20.0,
            verbose=verbose,
            label="Game remesh fallback",
        )

    if remesh_vertices.size == 0 or remesh_faces.size == 0:
        if verbose:
            tqdm.write("Experimental retopo produced an empty mesh; keeping cleaned mesh")
        return vertices, faces

    if remesh_faces.shape[0] > target:
        remesh_vertices, remesh_faces = _quadric_game_decimate(
            remesh_vertices,
            remesh_faces,
            target_faces=target,
            boundary_weight=25.0,
            verbose=verbose,
            label="Experimental final limiter",
        )
    remesh_vertices, remesh_faces = _enforce_face_budget(
        remesh_vertices,
        remesh_faces,
        target,
        verbose=verbose,
        label="Experimental hard budget",
    )

    if verbose:
        tqdm.write(
            f"After experimental retopo: {remesh_vertices.shape[0]} vertices, "
            f"{remesh_faces.shape[0]} faces"
        )
    return remesh_vertices, remesh_faces


def game_remesh_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces=None,
    method: str = "decimate",
    verbose: bool = False,
):
    method = normalize_game_remesh_method(method)
    if method == "experimental":
        return _experimental_game_retopo(
            vertices,
            faces,
            target_faces=target_faces,
            verbose=verbose,
        )

    return _quadric_game_decimate(
        vertices,
        faces,
        target_faces=target_faces,
        verbose=verbose,
    )


def parametrize_mesh(vertices: np.array, faces: np.array):
    """
    Parametrize a mesh to a texture space, using xatlas.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
    """

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    vertices = vertices[vmapping]
    faces = indices

    return vertices, faces, uvs

@torch.inference_mode(False)
@torch.enable_grad()
def bake_texture(
    vertices: np.array,
    faces: np.array,
    uvs: np.array,
    observations: List[np.array],
    masks: List[np.array],
    extrinsics: List[np.array],
    intrinsics: List[np.array],
    texture_size: int = 2048,
    near: float = 0.1,
    far: float = 10.0,
    mode: Literal["fast", "opt", "average"] = "opt",
    lambda_tv: float = 1e-2,
    verbose: bool = False,
    rendering_engine: str = "nvdiffrast",  # nvdiffrast OR "pytorch3d"
    device: str = None,

):
    if device is None:
        import torch as _t
        device = "mps" if _t.backends.mps.is_available() and not _t.cuda.is_available() else "cuda"
    """
    Bake texture to a mesh from multiple observations.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        uvs (np.array): UV coordinates of the mesh. Shape (V, 2).
        observations (List[np.array]): List of observations. Each observation is a 2D image. Shape (H, W, 3).
        masks (List[np.array]): List of masks. Each mask is a 2D image. Shape (H, W).
        extrinsics (List[np.array]): List of extrinsics. Shape (4, 4).
        intrinsics (List[np.array]): List of intrinsics. Shape (3, 3).
        texture_size (int): Size of the texture.
        near (float): Near plane of the camera.
        far (float): Far plane of the camera.
        mode (Literal['fast', 'opt']): Mode of texture baking.
        lambda_tv (float): Weight of total variation loss in optimization.
        verbose (bool): Whether to print progress.
    """


    vertices = torch.tensor(vertices).to(device)
    faces = torch.tensor(faces.astype(np.int32)).to(device)
    uvs = torch.tensor(uvs).to(device)
    observations = [torch.tensor(obs / 255.0).float().to(device) for obs in observations]
    masks = [torch.tensor(m > 0).bool().to(device) for m in masks]
    views = [
        utils3d.torch.extrinsics_to_view(torch.tensor(extr).to(device))
        for extr in extrinsics
    ]
    projections = [
        utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).to(device), near, far)
        for intr in intrinsics
    ]

    if mode == "fast":
        texture = torch.zeros(
            (texture_size * texture_size, 3), dtype=torch.float32
        ).to(device)
        texture_weights = torch.zeros(
            (texture_size * texture_size), dtype=torch.float32
        ).to(device)
        rastctx = utils3d.torch.RastContext(backend=device if device.startswith("cuda") else "cuda")
        for observation, view, projection in tqdm(
            zip(observations, views, projections),
            total=len(observations),
            disable=not verbose,
            desc="Texture baking (fast)",
        ):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx,
                    vertices[None],
                    faces,
                    observation.shape[1],
                    observation.shape[0],
                    uv=uvs[None],
                    view=view,
                    projection=projection,
                )
                uv_map = rast["uv"][0].detach().flip(0)
                mask = rast["mask"][0].detach().bool() & masks[0]

            # nearest neighbor interpolation
            uv_map = (uv_map * texture_size).floor().long()
            obs = observation[mask]
            uv_map = uv_map[mask]
            idx = uv_map[:, 0] + (texture_size - uv_map[:, 1] - 1) * texture_size
            texture = texture.scatter_add(0, idx.view(-1, 1).expand(-1, 3), obs)
            texture_weights = texture_weights.scatter_add(
                0,
                idx,
                torch.ones((obs.shape[0]), dtype=torch.float32, device=texture.device),
            )

        mask = texture_weights > 0
        texture[mask] /= texture_weights[mask][:, None]
        texture = np.clip(
            texture.reshape(texture_size, texture_size, 3).cpu().numpy() * 255, 0, 255
        ).astype(np.uint8)

        # inpaint
        mask = (
            (texture_weights == 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
            .reshape(texture_size, texture_size)
        )
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)

    elif mode == "opt":
        # NOTE: do NOT flip observations/masks vertically here. The z-buffered
        # _software_rasterize_uv below produces UV maps in the same top-left-origin
        # orientation as the rendered observations (verified: the pinhole projection
        # matches the renderer exactly). Flipping would vertically mirror each
        # observation against its UV map, so 100 conflicting views average to mud.
        _uv = []
        _uv_dr = []

        def _software_rasterize_uv(vertices, faces, uvs, H, W, view, projection):
            """Z-buffered CPU/MPS UV rasterization (pure PyTorch, no CUDA).

            Uses a proper depth test so occluded back-faces no longer overwrite visible
            ones, then interpolates per-pixel UV with perspective-correct barycentrics.
            """
            from ..renderers.mesh_raster_silicon import rasterize_mesh
            verts = vertices[0]                       # [V, 3]
            uvs_ = uvs[0] if uvs.ndim == 3 else uvs   # [V, 2]
            mvp = projection @ view                   # [4, 4]

            out = rasterize_mesh(verts, faces, mvp, H, W)
            fid = out["face_id"]                      # [H, W]  (-1 where empty)
            bary = out["bary"]                        # [H, W, 3]
            mask_map = out["mask"]                    # [H, W]

            face_vidx = faces[fid.clamp_min(0)]       # [H, W, 3] vertex ids per pixel
            face_uv = uvs_[face_vidx]                 # [H, W, 3, 2]
            uv_map = (bary.unsqueeze(-1) * face_uv).sum(dim=2)        # [H, W, 2]
            uv_map = torch.where(mask_map.unsqueeze(-1), uv_map, torch.zeros_like(uv_map))
            return {"uv": uv_map.unsqueeze(0), "mask": mask_map.unsqueeze(0)}

        for observation, view, projection in tqdm(
            zip(observations, views, projections),
            total=len(views),
            disable=not verbose,
            desc="Texture baking (opt): UV",
        ):
            with torch.no_grad():
                rast = _software_rasterize_uv(
                    vertices[None],
                    faces,
                    uvs[None],
                    observation.shape[0],
                    observation.shape[1],
                    view,
                    projection,
                )
                _uv.append(rast["uv"].detach())
                _uv_dr.append(rast["uv"].detach())  # uv_dr unused with pytorch3d engine

        texture = torch.nn.Parameter(
            torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).to(device)
        )
        optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

        def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return start_lr * (end_lr / start_lr) ** (step / total_steps)

        def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
            return end_lr + 0.5 * (start_lr - end_lr) * (
                1 + np.cos(np.pi * step / total_steps)
            )

        def tv_loss(texture):
            return torch.nn.functional.l1_loss(
                texture[:, :-1, :, :], texture[:, 1:, :, :]
            ) + torch.nn.functional.l1_loss(texture[:, :, :-1, :], texture[:, :, 1:, :])



        def render_pt3d_texture(texture, uv, uv_dr=None):
            import torch.nn.functional as F
            texture_perm = texture.permute(0, 3, 1, 2)
            grid = uv * 2 - 1
            if grid.dim() == 3:
                grid = grid.unsqueeze(0)  # (1, H, W, 2)
            elif grid.dim() == 4 and grid.shape[0] == 1:
                pass  
            elif grid.dim() == 4 and grid.shape[1] == 1:
                grid = grid.squeeze(1)  # remove extra batch dimension if necessary
            else:
                raise ValueError(f"Unexpected grid shape: {grid.shape}")
            render = F.grid_sample(
                texture_perm, grid, mode='bilinear', padding_mode='border', align_corners=True
            )
            render = render.permute(0, 2, 3, 1)[0]  # (H_out, W_out, 3)
            return render
        
        
        total_steps = 2500
        
        with tqdm(
            total=total_steps,
            disable=not verbose,
            desc="Texture baking (opt): optimizing",
            ) as pbar:
            for step in range(total_steps):
                optimizer.zero_grad()
                selected = np.random.randint(0, len(views))
                uv, uv_dr, observation, mask = (
                    _uv[selected],
                    _uv_dr[selected],
                    observations[selected],
                    masks[selected],
                )
                
                if rendering_engine == "nvdiffrast":
                    import nvdiffrast.torch as dr
                    render = dr.texture(texture, uv, uv_dr)[0]

                if rendering_engine == "pytorch3d":
                    render = render_pt3d_texture(texture, uv)
                    
                loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
                if lambda_tv > 0:
                    loss += lambda_tv * tv_loss(texture)
                loss.backward()
                optimizer.step()
                # annealing
                optimizer.param_groups[0]["lr"] = cosine_anealing(
                    optimizer, step, total_steps, 1e-2, 1e-5
                    )
                pbar.set_postfix({"loss": loss.item()})
                pbar.update()
        # Texture is baked in the natural UV convention (row = v, col = u, origin
        # top-left) — matching the unflipped observations. Do inpainting in this same
        # space, then apply ONE vertical flip at the very end (below) to compensate for
        # trimesh's V-flip on GLB export, so standard glTF viewers sample it correctly.
        texture = np.clip(
            texture[0].detach().cpu().numpy() * 255, 0, 255
        ).astype(np.uint8)
        # Build inpaint mask: pixels not covered by any UV triangle
        uv_coverage = torch.zeros(texture_size, texture_size, device=vertices.device, dtype=torch.bool)
        uvs_px = (uvs * texture_size).long().clamp(0, texture_size - 1)
        tri_uvs_px = uvs_px[faces]  # [F, 3, 2]
        for (a, b, c) in [(0.5,0.25,0.25),(0.25,0.5,0.25),(0.25,0.25,0.5),(1/3,1/3,1/3)]:
            px = (a*tri_uvs_px[:,0]+b*tri_uvs_px[:,1]+c*tri_uvs_px[:,2]).long()
            xi = px[:,0].clamp(0,texture_size-1)
            yi = px[:,1].clamp(0,texture_size-1)
            uv_coverage[yi, xi] = True
        mask = (1 - uv_coverage.cpu().numpy().astype(np.uint8)).astype(np.uint8)
        texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
        # Compensate trimesh's GLB-export V-flip (it stores uv.v -> 1-uv.v): flip the
        # texture vertically so a standard glTF viewer samples the correct texels.
        texture = np.ascontiguousarray(texture[::-1])

    elif mode == "average":
        # Deterministic angle-weighted multi-view average using the z-buffered software
        # rasterizer. Produces a smooth, continuous "proper finish" (no stochastic Adam
        # patchiness / dabs), and is faster than opt. Best for the MPS / pytorch3d path.
        from ..renderers.mesh_raster_silicon import rasterize_mesh
        faces_l = faces.long()
        v0 = vertices[faces_l[:, 0]]; v1 = vertices[faces_l[:, 1]]; v2 = vertices[faces_l[:, 2]]
        face_n = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)  # [F,3]
        H = observations[0].shape[0]; W = observations[0].shape[1]
        tex_sum = torch.zeros(texture_size * texture_size, 3, device=device)
        tex_w = torch.zeros(texture_size * texture_size, device=device)
        for i in range(len(observations)):
            mvp = projections[i] @ views[i]
            out = rasterize_mesh(vertices, faces_l, mvp, H, W)
            fid = out["face_id"]; m = out["mask"]
            if not bool(m.any()):
                continue
            uvp = (out["bary"].unsqueeze(-1) * uvs[faces_l[fid.clamp_min(0)]]).sum(2)  # [H,W,2]
            cam_fwd = torch.tensor(extrinsics[i], device=device, dtype=torch.float32)[2, :3]
            facing = (-(face_n[fid.clamp_min(0)] * cam_fwd[None, None, :]).sum(-1)).clamp(min=0.0) ** 2
            sel = m
            col = observations[i][sel]            # already /255 float
            wsel = facing[sel]
            uvpx = (uvp[sel].clamp(0, 1) * (texture_size - 1)).long()
            lin = uvpx[:, 1] * texture_size + uvpx[:, 0]
            tex_sum.index_add_(0, lin, col * wsel[:, None])
            tex_w.index_add_(0, lin, wsel)
        cov = tex_w > 0
        tex = torch.zeros(texture_size * texture_size, 3, device=device)
        tex[cov] = tex_sum[cov] / tex_w[cov][:, None]
        texture = (tex.reshape(texture_size, texture_size, 3).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        inpaint_mask = (~cov).reshape(texture_size, texture_size).cpu().numpy().astype(np.uint8)
        texture = cv2.inpaint(texture, inpaint_mask, 3, cv2.INPAINT_TELEA)
        # Same trimesh export V-flip compensation as the opt path.
        texture = np.ascontiguousarray(texture[::-1])

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return texture


def _empty_device_cache(device):
    device_type = torch.device(device).type
    if device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _emit_progress(progress_callback, label, amount=1):
    if progress_callback is not None and amount > 0:
        progress_callback("advance", label=label, amount=amount)


def bake_texture_average_streaming(
    app_rep: Gaussian,
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_size: int = 2048,
    nviews: int = 100,
    render_resolution: int = 1024,
    near: float = 1,
    far: float = 3,
    verbose: bool = True,
    device: str = None,
    progress_callback=None,
    progress_units: int = 0,
):
    """Angle-weighted texture baking without storing all rendered views at once."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() and not torch.cuda.is_available() else "cuda"

    from ..renderers.mesh_raster_silicon import rasterize_mesh
    from .render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces_l = torch.tensor(faces.astype(np.int32), device=device).long()
    uvs = torch.tensor(uvs, dtype=torch.float32, device=device)

    v0 = vertices[faces_l[:, 0]]
    v1 = vertices[faces_l[:, 1]]
    v2 = vertices[faces_l[:, 2]]
    face_n = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)

    tex_sum = torch.zeros(texture_size * texture_size, 3, device=device)
    tex_w = torch.zeros(texture_size * texture_size, device=device)

    renderer = GaussianRenderer()
    renderer.rendering_options.resolution = render_resolution
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 1
    renderer.rendering_options.backend = (
        "gsplat" if torch.device(device).type == "mps" else "inria"
    )
    renderer.pipe.kernel_size = 0.1
    renderer.pipe.use_mip_gaussian = True

    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, 2, 40
    )

    progressed = 0
    for i in tqdm(
        range(nviews),
        total=nviews,
        disable=not verbose,
        desc="Texture baking (average)",
    ):
        extr = extrinsics[i].to(device)
        intr = intrinsics[i].to(device)
        with torch.no_grad():
            rendered = renderer.render(app_rep, extr, intr)["color"]
            observation = rendered.detach().permute(1, 2, 0).float().clamp(0, 1)
            H, W = observation.shape[:2]

            view = utils3d.torch.extrinsics_to_view(extr)
            projection = utils3d.torch.intrinsics_to_perspective(intr, near, far)
            out = rasterize_mesh(vertices, faces_l, projection @ view, H, W)
            fid = out["face_id"]
            sel = out["mask"]
            if bool(sel.any()):
                uvp = (
                    out["bary"].unsqueeze(-1)
                    * uvs[faces_l[fid.clamp_min(0)]]
                ).sum(2)
                cam_fwd = extr.to(dtype=torch.float32)[2, :3]
                facing = (
                    -(face_n[fid.clamp_min(0)] * cam_fwd[None, None, :]).sum(-1)
                ).clamp(min=0.0) ** 2
                col = observation[sel]
                wsel = facing[sel]
                uvpx = (uvp[sel].clamp(0, 1) * (texture_size - 1)).long()
                lin = uvpx[:, 1] * texture_size + uvpx[:, 0]
                tex_sum.index_add_(0, lin, col * wsel[:, None])
                tex_w.index_add_(0, lin, wsel)
                del uvp, cam_fwd, facing, col, wsel, uvpx, lin

        del extr, intr, rendered, observation, view, projection, out, fid, sel
        _empty_device_cache(device)

        target_progress = ((i + 1) * progress_units) // max(1, nviews)
        if target_progress > progressed:
            _emit_progress(
                progress_callback,
                f"Texture bake {i + 1}/{nviews}",
                target_progress - progressed,
            )
            progressed = target_progress

    cov = tex_w > 0
    tex_sum[cov] = tex_sum[cov] / tex_w[cov][:, None]
    texture = (tex_sum.reshape(texture_size, texture_size, 3).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    inpaint_mask = (~cov).reshape(texture_size, texture_size).cpu().numpy().astype(np.uint8)
    texture = cv2.inpaint(texture, inpaint_mask, 3, cv2.INPAINT_TELEA)
    texture = np.ascontiguousarray(texture[::-1])
    _empty_device_cache(device)
    return texture


def to_glb(
    app_rep: Union[Strivec, Gaussian],
    mesh: MeshExtractResult,
    simplify: float = 0.95,
    fill_holes: bool = True,
    fill_holes_max_size: float = 0.04,
    fill_holes_resolution: int = 1024,
    fill_holes_num_views: int = 1000,
    texture_size: int = 1024,
    lambda_tv: float = 0.01,
    texture_mode: str = "opt",   # "opt" (Adam) | "average" (smooth angle-weighted) | "fast"
    texture_views: int = 100,
    texture_render_resolution: int = 1024,
    game_remesh: bool = False,
    game_target_faces: Optional[int] = None,
    game_remesh_method: str = "decimate",
    debug: bool = False,
    verbose: bool = True,
    with_mesh_postprocess=True,
    with_texture_baking=True,
    use_vertex_color=False,
    rendering_engine: str = "nvdiffrast",  # nvdiffrast OR "pytorch3d"
    progress_callback=None,
) -> trimesh.Trimesh:
    """
    Convert a generated asset to a glb file.

    Args:
        app_rep (Union[Strivec, Gaussian]): Appearance representation.
        mesh (MeshExtractResult): Extracted mesh.
        simplify (float): Ratio of faces to remove in simplification.
        fill_holes (bool): Whether to fill holes in the mesh.
        fill_holes_max_size (float): Maximum area of a hole to fill.
        texture_size (int): Size of the texture.
        debug (bool): Whether to print debug information.
        verbose (bool): Whether to print progress.
    """
    vertices = mesh.vertices.float().cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    vert_colors = mesh.vertex_attrs[:, :3].cpu().numpy()

    if with_mesh_postprocess:
        # mesh postprocess
        vertices, faces = postprocess_mesh(
            vertices,
            faces,
            simplify=simplify > 0,
            simplify_ratio=simplify,
            fill_holes=fill_holes,
            fill_holes_max_hole_size=fill_holes_max_size,
            fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify)),
            fill_holes_resolution=fill_holes_resolution,
            fill_holes_num_views=fill_holes_num_views,
            debug=debug,
            verbose=verbose,
        )
        _emit_progress(progress_callback, "Mesh cleanup", 1)

    if game_remesh:
        vertices, faces = game_remesh_mesh(
            vertices,
            faces,
            target_faces=game_target_faces,
            method=game_remesh_method,
            verbose=verbose,
        )
        _emit_progress(progress_callback, "Game remesh", 1)

    if with_texture_baking:
        # parametrize mesh
        vertices, faces, uvs = parametrize_mesh(vertices, faces)
        logger.info("Baking texture ...")
        _emit_progress(progress_callback, "UV unwrap", 1)

        if texture_mode == "average" and isinstance(app_rep, Gaussian):
            texture = bake_texture_average_streaming(
                app_rep,
                vertices,
                faces,
                uvs,
                texture_size=texture_size,
                nviews=texture_views,
                render_resolution=texture_render_resolution,
                verbose=verbose,
                progress_callback=progress_callback,
                progress_units=3,
            )
        else:
            # bake texture
            observations, extrinsics, intrinsics = render_multiview(
                app_rep, resolution=texture_render_resolution, nviews=texture_views
            )
            masks = [np.any(observation > 0, axis=-1) for observation in observations]
            extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
            intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
            texture = bake_texture(
                vertices,
                faces,
                uvs,
                observations,
                masks,
                extrinsics,
                intrinsics,
                texture_size=texture_size,
                mode=texture_mode,
                lambda_tv=lambda_tv,
                verbose=verbose,
                rendering_engine=rendering_engine
            )
            _emit_progress(progress_callback, "Texture bake", 3)
        texture = Image.fromarray(texture)
        material = trimesh.visual.material.PBRMaterial(
            roughnessFactor=1.0,
            baseColorTexture=texture,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        )

    # rotate mesh (from z-up to y-up)
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    if not with_mesh_postprocess and not with_texture_baking and use_vertex_color:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = vert_colors
    else:
        mesh = trimesh.Trimesh(
            vertices,
            faces,
            visual=(
                trimesh.visual.TextureVisuals(uv=uvs, material=material)
                if with_texture_baking
                else None
            ),
        )

    return mesh


def simplify_gs(
    gs: Gaussian,
    simplify: float = 0.95,
    verbose: bool = True,
):
    """
    Simplify 3D Gaussians
    NOTE: this function is not used in the current implementation for the unsatisfactory performance.

    Args:
        gs (Gaussian): 3D Gaussian.
        simplify (float): Ratio of Gaussians to remove in simplification.
    """
    if simplify <= 0:
        return gs

    # simplify
    observations, extrinsics, intrinsics = render_multiview(
        gs, resolution=1024, nviews=100
    )
    observations = [
        torch.tensor(obs / 255.0).float().cuda().permute(2, 0, 1)
        for obs in observations
    ]

    # Following https://arxiv.org/pdf/2411.06019
    renderer = GaussianRenderer(
        {
            "resolution": 1024,
            "near": 0.8,
            "far": 1.6,
            "ssaa": 1,
            "bg_color": (0, 0, 0),
        }
    )
    new_gs = Gaussian(**gs.init_params)
    new_gs._features_dc = gs._features_dc.clone()
    new_gs._features_rest = (
        gs._features_rest.clone() if gs._features_rest is not None else None
    )
    new_gs._opacity = torch.nn.Parameter(gs._opacity.clone())
    new_gs._rotation = torch.nn.Parameter(gs._rotation.clone())
    new_gs._scaling = torch.nn.Parameter(gs._scaling.clone())
    new_gs._xyz = torch.nn.Parameter(gs._xyz.clone())

    start_lr = [1e-4, 1e-3, 5e-3, 0.025]
    end_lr = [1e-6, 1e-5, 5e-5, 0.00025]
    optimizer = torch.optim.Adam(
        [
            {"params": new_gs._xyz, "lr": start_lr[0]},
            {"params": new_gs._rotation, "lr": start_lr[1]},
            {"params": new_gs._scaling, "lr": start_lr[2]},
            {"params": new_gs._opacity, "lr": start_lr[3]},
        ],
        lr=start_lr[0],
    )

    def exp_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return start_lr * (end_lr / start_lr) ** (step / total_steps)

    def cosine_anealing(optimizer, step, total_steps, start_lr, end_lr):
        return end_lr + 0.5 * (start_lr - end_lr) * (
            1 + np.cos(np.pi * step / total_steps)
        )

    _zeta = new_gs.get_opacity.clone().detach().squeeze()
    _lambda = torch.zeros_like(_zeta)
    _delta = 1e-7
    _interval = 10
    num_target = int((1 - simplify) * _zeta.shape[0])

    with tqdm(total=2500, disable=not verbose, desc="Simplifying Gaussian") as pbar:
        for i in range(2500):
            # prune
            if i % 100 == 0:
                mask = new_gs.get_opacity.squeeze() > 0.05
                mask = torch.nonzero(mask).squeeze()
                new_gs._xyz = torch.nn.Parameter(new_gs._xyz[mask])
                new_gs._rotation = torch.nn.Parameter(new_gs._rotation[mask])
                new_gs._scaling = torch.nn.Parameter(new_gs._scaling[mask])
                new_gs._opacity = torch.nn.Parameter(new_gs._opacity[mask])
                new_gs._features_dc = new_gs._features_dc[mask]
                new_gs._features_rest = (
                    new_gs._features_rest[mask]
                    if new_gs._features_rest is not None
                    else None
                )
                _zeta = _zeta[mask]
                _lambda = _lambda[mask]
                # update optimizer state
                for param_group, new_param in zip(
                    optimizer.param_groups,
                    [new_gs._xyz, new_gs._rotation, new_gs._scaling, new_gs._opacity],
                ):
                    stored_state = optimizer.state[param_group["params"][0]]
                    if "exp_avg" in stored_state:
                        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    del optimizer.state[param_group["params"][0]]
                    param_group["params"][0] = new_param
                    optimizer.state[param_group["params"][0]] = stored_state

            opacity = new_gs.get_opacity.squeeze()

            # sparisfy
            if i % _interval == 0:
                _zeta = _lambda + opacity.detach()
                if opacity.shape[0] > num_target:
                    index = _zeta.topk(num_target)[1]
                    _m = torch.ones_like(_zeta, dtype=torch.bool)
                    _m[index] = 0
                    _zeta[_m] = 0
                _lambda = _lambda + opacity.detach() - _zeta

            # sample a random view
            view_idx = np.random.randint(len(observations))
            observation = observations[view_idx]
            extrinsic = extrinsics[view_idx]
            intrinsic = intrinsics[view_idx]

            color = renderer.render(new_gs, extrinsic, intrinsic)["color"]
            rgb_loss = torch.nn.functional.l1_loss(color, observation)
            loss = rgb_loss + _delta * torch.sum(
                torch.pow(_lambda + opacity - _zeta, 2)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update lr
            for j in range(len(optimizer.param_groups)):
                optimizer.param_groups[j]["lr"] = cosine_anealing(
                    optimizer, i, 2500, start_lr[j], end_lr[j]
                )

            pbar.set_postfix(
                {
                    "loss": rgb_loss.item(),
                    "num": opacity.shape[0],
                    "lambda": _lambda.mean().item(),
                }
            )
            pbar.update()

    new_gs._xyz = new_gs._xyz.data
    new_gs._rotation = new_gs._rotation.data
    new_gs._scaling = new_gs._scaling.data
    new_gs._opacity = new_gs._opacity.data

    return new_gs
