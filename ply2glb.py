#!/usr/bin/env python3
"""
ply2glb — Convert a SAM-3D gaussian splat to a textured GLB mesh.

Usage:
    python ply2glb.py <output_folder>
    ./run.sh glb <output_folder>

Loads only the mesh decoder (~500 MB), not the full inference pipeline.
Requires splat.ply in <output_folder>. slat.pt is required only when mesh.glb
does not exist yet or SAM3D_GAME_REBUILD_SOURCE_GLB=1 is set.
"""

import sys, os, time, argparse, gc, shutil, tempfile
from types import SimpleNamespace
import torch
import numpy as np
from PIL import Image as PILImage
from sam3d_progress import CliProgress

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "notebook"))
os.environ.setdefault("SPARSE_BACKEND", "native")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["LIDRA_SKIP_INIT"] = "true"

R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"
C = "\033[96m"; W = "\033[97m"; DIM = "\033[2m"; BOLD = "\033[1m"; RST = "\033[0m"
EXPORT_ROTATION = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)

def hdr(msg):  print(f"\n{BOLD}{C}{'─'*58}{RST}\n{BOLD}{W}  {msg}{RST}\n{DIM}{'─'*58}{RST}")
def step(msg): print(f"  {C}›{RST}  {msg}", flush=True)
def ok(msg):   print(f"  {G}✓{RST}  {msg}", flush=True)
def err(msg):  print(f"  {R}✗{RST}   {msg}"); sys.exit(1)
def saved(label, path): print(f"  {G}▶ SAVED{RST}  {BOLD}{label:<12}{RST}  {path}")


def int_env(name, default=0):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def bool_env(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def choose_mesh_decode_device(render_device, active_voxels):
    requested = os.environ.get("SAM3D_MESH_DECODE_DEVICE", "auto").strip().lower()
    if requested in ("cpu", "mps", "cuda"):
        if requested == "mps" and not torch.backends.mps.is_available():
            return render_device, "requested mps is unavailable"
        if requested == "cuda" and not torch.cuda.is_available():
            return render_device, "requested cuda is unavailable"
        return requested, f"forced by SAM3D_MESH_DECODE_DEVICE={requested}"

    cpu_threshold = int_env("SAM3D_CPU_DECODE_VOXELS", 30000)
    if torch.device(render_device).type == "mps" and active_voxels >= cpu_threshold:
        return "cpu", f"large SLAT ({active_voxels:,} voxels >= {cpu_threshold:,})"
    return render_device, "auto"


def make_progress(extra_units=0):
    initial = max(0, int_env("SAM3D_PROGRESS_DONE", 0))
    total = int_env("SAM3D_PROGRESS_TOTAL", 0)
    if total <= initial:
        total = initial + 10 + int(extra_units)
    return CliProgress(total=total, initial=initial)


def parse_target_faces(raw):
    if raw is None or str(raw).lower() == "auto":
        return None
    try:
        value = int(raw)
    except ValueError:
        err(f"Invalid target face count: {raw}")
    if value < 500:
        err("Target face count must be at least 500 for game-quality exports.")
    return value


def parse_target_vertices(raw):
    value = (raw or "auto").strip().lower()
    if value == "auto":
        return None
    try:
        parsed = int(value)
    except ValueError:
        err(f"Invalid target vertex count: {raw}")
    if not 2000 <= parsed <= 10000:
        err("Game vertex target must be between 2,000 and 10,000.")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SAM-3D splat/slat outputs to a textured GLB."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Output folder containing splat.ply and slat.pt",
    )
    parser.add_argument(
        "--target-faces",
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--target-vertices",
        default=os.environ.get("SAM3D_GAME_TARGET_VERTICES", "auto"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def load_slat(slat_path):
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
    data = torch.load(slat_path, map_location="cpu", weights_only=False)
    slat = sp.SparseTensor(feats=data["feats"], coords=data["coords"])
    return slat


def load_gaussian(ply_path, device):
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
    gs = Gaussian(aabb=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], sh_degree=0, device=device)
    gs.load_ply(ply_path)
    return gs


def load_mesh_decoder(device):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam3d_objects.model.io import load_model_from_checkpoint

    ckpt_dir = os.path.join(ROOT, "checkpoints", "hf", "checkpoints")
    config = OmegaConf.load(os.path.join(ckpt_dir, "slat_decoder_mesh.yaml"))
    from omegaconf import OmegaConf as _OC
    # force fp32 — fp16 causes NaN in the swin-attention blocks on MPS
    config = _OC.merge(config, _OC.create({"device": device, "use_fp16": False}))
    model = instantiate(config)          # build the nn.Module from config
    load_model_from_checkpoint(
        model,
        os.path.join(ckpt_dir, "slat_decoder_mesh.ckpt"),
        device=device,
        state_dict_key=None,
    )
    model = model.to(device)
    model.eval()
    return model


def empty_device_cache(device):
    device_type = torch.device(device).type
    if device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_mesh_result_to_cpu(mesh_result):
    for name in ("vertices", "faces", "vertex_attrs", "face_normal"):
        value = getattr(mesh_result, name, None)
        if isinstance(value, torch.Tensor):
            setattr(mesh_result, name, value.detach().cpu())
    return mesh_result


def _mesh_from_scene_or_mesh(asset):
    import trimesh

    if isinstance(asset, trimesh.Trimesh):
        return asset
    if isinstance(asset, trimesh.Scene):
        if hasattr(asset, "to_geometry"):
            mesh = asset.to_geometry()
            if isinstance(mesh, trimesh.Trimesh):
                return mesh
        geometries = list(asset.geometry.values())
        if not geometries:
            raise RuntimeError("No geometry found while exporting OBJ.")
        if len(geometries) == 1:
            return geometries[0]
        return trimesh.util.concatenate(geometries)
    raise RuntimeError(f"Unsupported mesh asset type: {type(asset)!r}")


def load_glb_mesh(glb_path):
    import trimesh

    return _mesh_from_scene_or_mesh(trimesh.load(glb_path, force="scene"))


def validate_glb_mesh(glb_path):
    if not os.path.isfile(glb_path) or os.path.getsize(glb_path) <= 0:
        return False
    try:
        mesh = load_glb_mesh(glb_path)
        return len(mesh.vertices) > 0 and len(mesh.faces) > 0
    except Exception:
        return False


def source_glb_to_mesh_result(glb_path):
    mesh = load_glb_mesh(glb_path).copy()
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    # mesh.glb is exported y-up; to_glb expects the original z-up decoder space.
    vertices = np.asarray(mesh.vertices, dtype=np.float32) @ EXPORT_ROTATION.T
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError(f"No usable geometry found in {glb_path}")
    return SimpleNamespace(
        vertices=torch.from_numpy(vertices).float(),
        faces=torch.from_numpy(faces).long(),
        vertex_attrs=torch.ones((vertices.shape[0], 3), dtype=torch.float32),
    )


def _material_texture(mesh):
    material = getattr(getattr(mesh, "visual", None), "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        return PILImage.new("RGB", (4, 4), (255, 255, 255))
    if not isinstance(image, PILImage.Image):
        image = PILImage.fromarray(np.asarray(image))
    return image.convert("RGB")


def export_textured_obj(mesh, obj_path, texture_path, normal_path=None):
    mesh = _mesh_from_scene_or_mesh(mesh)
    mesh = mesh.copy()
    if not mesh.is_watertight:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()

    texture = _material_texture(mesh)
    texture.save(texture_path)

    obj_dir = os.path.dirname(obj_path)
    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
    material_name = "sam3d_material"
    texture_name = os.path.basename(texture_path)
    normal_name = os.path.basename(normal_path) if normal_path and os.path.isfile(normal_path) else None
    with open(mtl_path, "w", encoding="ascii") as mtl:
        mtl.write(f"newmtl {material_name}\n")
        mtl.write("Ka 1.000000 1.000000 1.000000\n")
        mtl.write("Kd 1.000000 1.000000 1.000000\n")
        mtl.write("Ks 0.000000 0.000000 0.000000\n")
        mtl.write("d 1.000000\n")
        mtl.write("illum 2\n")
        mtl.write(f"map_Kd {texture_name}\n")
        if normal_name:
            mtl.write(f"map_Bump {normal_name}\n")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    uvs = getattr(getattr(mesh, "visual", None), "uv", None)
    if uvs is None or len(uvs) != len(vertices):
        uvs = np.zeros((len(vertices), 2), dtype=np.float64)
    else:
        uvs = np.asarray(uvs, dtype=np.float64)

    with open(obj_path, "w", encoding="ascii") as obj:
        obj.write(f"mtllib {os.path.basename(mtl_path)}\n")
        for vertex in vertices:
            obj.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for uv in uvs:
            obj.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")
        for normal in normals:
            obj.write(f"vn {normal[0]:.8f} {normal[1]:.8f} {normal[2]:.8f}\n")
        obj.write(f"usemtl {material_name}\n")
        for face in faces:
            a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
            obj.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
    return obj_path, mtl_path, texture_path


def export_glb_to_obj_family(glb_path, folder, stem, normal_path=None):
    import trimesh

    asset = trimesh.load(glb_path, force="scene")
    return export_textured_obj(
        asset,
        os.path.join(folder, f"{stem}.obj"),
        os.path.join(folder, f"{stem}_texture.png"),
        normal_path=normal_path,
    )


def auto_game_target_faces(source_glb_path, target_vertices):
    if target_vertices is None:
        return None
    return int(max(4000, min(20000, target_vertices * 2)))


def main():
    print(f"\n{BOLD}{W}  ply2glb  ·  Gaussian splat → Textured GLB{RST}")
    args = parse_args()

    if not args.folder:
        print(
            f"  Usage: {sys.argv[0]} <output_folder>"
        )
        sys.exit(1)

    folder = os.path.abspath(args.folder.strip().strip("'\""))
    ply_path  = os.path.join(folder, "splat.ply")
    slat_path = os.path.join(folder, "slat.pt")
    game_sidecar_base = "mesh_game"
    source_glb_path = os.path.join(folder, "mesh.glb")
    default_name = "mesh_game.glb"
    out_name = args.output or default_name
    glb_path  = os.path.join(folder, out_name)
    target_vertices = parse_target_vertices(args.target_vertices)
    target_faces = auto_game_target_faces(source_glb_path, target_vertices)

    if not os.path.isfile(ply_path):
        err(f"splat.ply not found in {folder}")

    rebuild_source_glb = bool_env("SAM3D_GAME_REBUILD_SOURCE_GLB", False)
    source_glb_valid = validate_glb_mesh(source_glb_path)
    if os.path.isfile(source_glb_path) and not source_glb_valid:
        print(f"  {Y}⚠{RST}   Existing mesh.glb is not loadable; rebuilding it: {source_glb_path}")
    should_write_source_glb = rebuild_source_glb or not source_glb_valid
    if should_write_source_glb and not os.path.isfile(slat_path):
        err(f"slat.pt not found in {folder} — re-run the CLI to regenerate (it now saves slat.pt)")

    render_device = (
        "mps" if torch.backends.mps.is_available() and not torch.cuda.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    ok(f"Device: {render_device}")
    ok("Pipeline: mesh.glb → mesh.obj → mesh_game.glb/.obj + texture/normal")
    ok(f"Game vertex target: {target_vertices or 'auto (2k-10k)'}")
    progress = make_progress(
        extra_units=8 if should_write_source_glb else 2
    )

    hdr("LOADING ASSETS")
    progress("phase", label="Load GLB assets")
    mesh_result = None
    if should_write_source_glb:
        step("Loading sparse latent (slat.pt) on CPU…")
        slat = load_slat(slat_path)
        ok(f"SLAT: {slat.feats.shape[0]:,} active voxels")

        # Guard: a non-finite latent would produce garbage geometry.
        if not bool(torch.isfinite(slat.feats).all().detach().cpu().item()):
            progress.close()
            err("slat.pt contains NaN/Inf (failed generation). "
                "Re-run the CLI to regenerate this object before converting to GLB.")
        progress.advance("Load GLB assets", 1)

        decode_device, decode_reason = choose_mesh_decode_device(render_device, slat.feats.shape[0])
        ok(f"Mesh decode device: {decode_device} ({decode_reason})")

        hdr("LOADING MESH DECODER")
        progress("phase", label="Load mesh decoder")
        step("Loading slat_decoder_mesh (~500 MB)…")
        t0 = time.time()
        decoder = load_mesh_decoder(decode_device)
        progress.advance("Load mesh decoder", 1)
        ok(f"Mesh decoder ready  ({time.time()-t0:.1f}s)")

        hdr("DECODING MESH")
        progress("phase", label="Decode mesh")
        step("Running mesh decoder…")
        t0 = time.time()
        slat_decode = slat.to(decode_device)
        with torch.no_grad():
            mesh_result = decoder(slat_decode)[0]
        mesh_result = move_mesh_result_to_cpu(mesh_result)
        progress.advance("Decode mesh", 2)
        ok(f"Mesh decoded  ({time.time()-t0:.1f}s)  — {mesh_result.vertices.shape[0]:,} verts / {mesh_result.faces.shape[0]:,} faces")

        del decoder, slat, slat_decode
        gc.collect()
        empty_device_cache(decode_device)
        ok("Released mesh decoder memory before loading gaussian splats")
    else:
        ok("Existing mesh.glb is loadable; skipping the mesh decoder")
        progress.advance("Load GLB assets", 1)

    hdr("LOADING GAUSSIAN")
    progress("phase", label="Load gaussian")
    step("Loading gaussian (splat.ply)…")
    gs = load_gaussian(ply_path, render_device)
    ok(f"Gaussian: {gs.get_xyz.shape[0]:,} splats")

    if not bool(torch.isfinite(gs.get_xyz).all().detach().cpu().item()):
        progress.close()
        err("splat.ply contains NaN/Inf (failed generation). "
            "Re-run the CLI to regenerate this object before converting to GLB.")

    hdr("CLEANUP MESH + BAKE TEXTURE + EXPORT GLB")
    progress("phase", label="Source mesh + game mesh")
    t0 = time.time()
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb
    import torch as _t
    on_mps = torch.device(render_device).type == "mps"
    texture_views = int_env("SAM3D_TEXTURE_VIEWS", 100)
    texture_render_resolution = int_env("SAM3D_TEXTURE_RENDER_RES", 1024)
    texture_size = int_env("SAM3D_TEXTURE_SIZE", 2048)
    game_temp_dir = tempfile.mkdtemp(prefix=".mesh-game-", dir=folder)
    game_quad_path = None
    game_normal_path = None
    export_glb_path = glb_path
    game_quad_path = os.path.join(
        game_temp_dir,
        f"{game_sidecar_base}_quads.obj",
    )
    game_normal_path = os.path.join(
        game_temp_dir,
        f"{game_sidecar_base}_normal.png",
    )
    export_glb_path = os.path.join(game_temp_dir, out_name)
    if on_mps:
        step(
            "Clean source mesh → game mesh → streamed texture bake "
            + f"({texture_views} views @ {texture_render_resolution}px, {texture_size}px atlas)…"
        )
    else:
        step("Running to_glb (texture baking ~1 min)…")
    # Full pipeline now runs on MPS via pure-PyTorch rasterizers:
    #   - mesh postprocess: mesh simplification (triangulated first) + _fill_holes (z-buffered
    #     software mesh raster instead of nvdiffrast)
    #   - texture baking: gsplat_silicon multi-view render + z-buffered UV raster + grid_sample
    # fill_holes views/resolution are reduced on MPS so the software rasterizer stays tractable.
    try:
        if should_write_source_glb:
            if mesh_result is None:
                raise RuntimeError("Internal error: source GLB rebuild requested without decoded mesh.")
            progress("phase", label="Source mesh + texture bake")
            step("Building cleaned high-detail source mesh.glb before game retopo…")
            source_export_path = os.path.join(game_temp_dir, "mesh.glb")
            source_glb = to_glb(
                gs,
                mesh_result,
                simplify=0.90,
                fill_holes=True,
                fill_holes_resolution=512 if on_mps else 1024,
                fill_holes_num_views=100 if on_mps else 1000,
                texture_size=texture_size,
                texture_views=texture_views,
                texture_render_resolution=texture_render_resolution,
                game_remesh=False,
                game_target_faces=None,
                game_remesh_method="quality",
                game_retopo_sidecar_path=None,
                experimental_retopo=False,
                experimental_target_faces=None,
                experimental_quad_path=None,
                experimental_normal_path=None,
                texture_mode="average",
                with_mesh_postprocess=True,
                with_texture_baking=True,
                use_vertex_color=False,
                rendering_engine="pytorch3d",
                progress_callback=progress,
            )
            source_glb.export(source_export_path)
            os.replace(source_export_path, source_glb_path)
            progress.advance("Export source GLB", 1)
            saved("mesh.glb", source_glb_path)
            del source_glb
            gc.collect()
            empty_device_cache(render_device)
        else:
            ok("Using existing mesh.glb as the high-detail comparison source")
        source_obj, source_mtl, source_texture = export_glb_to_obj_family(source_glb_path, folder, "mesh")
        saved("mesh.obj", source_obj)
        saved("mesh.mtl", source_mtl)
        saved("mesh_texture.png", source_texture)

        progress("phase", label="Load source mesh.glb")
        step("Loading mesh.glb geometry as the game optimization source…")
        game_mesh_result = source_glb_to_mesh_result(source_glb_path)
        ok(
            "Game source loaded from mesh.glb "
            f"({game_mesh_result.vertices.shape[0]:,} verts / "
            f"{game_mesh_result.faces.shape[0]:,} faces)"
        )
        del mesh_result
        gc.collect()
        empty_device_cache(render_device)

        progress("phase", label="Game mesh + texture bake")
        glb = to_glb(
            gs,
            game_mesh_result,
            simplify=0.0,
            fill_holes=False,
            fill_holes_resolution=512 if on_mps else 1024,
            fill_holes_num_views=100 if on_mps else 1000,
            texture_size=texture_size,
            texture_views=texture_views,
            texture_render_resolution=texture_render_resolution,
            game_remesh=False,
            game_target_faces=target_faces,
            game_remesh_method="quality",
            game_retopo_sidecar_path=None,
            experimental_retopo=True,
            experimental_target_faces=target_faces,
            experimental_quad_path=game_quad_path,
            experimental_normal_path=game_normal_path,
            experimental_source_is_clean=True,
            texture_mode="average",  # smooth angle-weighted multi-view average (no Adam patchiness)
            with_mesh_postprocess=True,   # includes floater removal (remove_floaters default on)
            with_texture_baking=True,
            use_vertex_color=False,
            rendering_engine="pytorch3d",
            progress_callback=progress,
        )
        ok(f"GLB ready  ({time.time()-t0:.1f}s)")
        glb.export(export_glb_path)
        os.replace(
            game_quad_path,
            os.path.join(folder, f"{game_sidecar_base}_quads.obj"),
        )
        os.replace(
            game_quad_path.replace("_quads.obj", "_report.json"),
            os.path.join(folder, f"{game_sidecar_base}_report.json"),
        )
        final_normal_path = os.path.join(folder, f"{game_sidecar_base}_normal.png")
        os.replace(game_normal_path, final_normal_path)
        os.replace(export_glb_path, glb_path)
        export_textured_obj(
            glb,
            os.path.join(folder, f"{game_sidecar_base}.obj"),
            os.path.join(folder, f"{game_sidecar_base}_texture.png"),
            normal_path=final_normal_path,
        )
    finally:
        if game_temp_dir:
            shutil.rmtree(game_temp_dir, ignore_errors=True)
    progress.advance("Export GLB", 1)
    if bool_env("SAM3D_PROGRESS_FINISH", True):
        progress.finish("Complete")
    else:
        progress.close()
    saved(out_name, glb_path)
    for filename in (
        f"{game_sidecar_base}.obj",
        f"{game_sidecar_base}.mtl",
        f"{game_sidecar_base}_texture.png",
        f"{game_sidecar_base}_normal.png",
        f"{game_sidecar_base}_quads.obj",
        f"{game_sidecar_base}_report.json",
    ):
        saved(filename, os.path.join(folder, filename))

    hdr("DONE")


if __name__ == "__main__":
    main()
