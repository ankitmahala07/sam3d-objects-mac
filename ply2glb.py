#!/usr/bin/env python3
"""
ply2glb — Convert a SAM-3D gaussian splat to a textured GLB mesh.

Usage:
    python ply2glb.py <output_folder>
    python ply2glb.py --game-ready --target-faces 2000 <output_folder>
    ./run.sh glb <output_folder>
    ./run.sh game <output_folder> [target_faces]

Loads only the mesh decoder (~500 MB), not the full inference pipeline.
Requires slat.pt and splat.ply to exist in <output_folder>.
"""

import sys, os, time, argparse
import torch
import numpy as np
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
    if value < 4:
        err("Target face count must be at least 4.")
    return value


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
        "--game-ready",
        "--remesh",
        action="store_true",
        help="Optimize the mesh before UV unwrap and texture baking, writing mesh_game.glb.",
    )
    parser.add_argument(
        "--target-faces",
        default="auto",
        help="Target triangle count for --game-ready. Use 'auto' or an integer.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output GLB filename inside the folder.",
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


def main():
    print(f"\n{BOLD}{W}  ply2glb  ·  Gaussian splat → Textured GLB{RST}")
    args = parse_args()

    if not args.folder:
        print(f"  Usage: {sys.argv[0]} [--game-ready --target-faces auto|N] <output_folder>")
        sys.exit(1)

    folder = os.path.abspath(args.folder.strip().strip("'\""))
    ply_path  = os.path.join(folder, "splat.ply")
    slat_path = os.path.join(folder, "slat.pt")
    out_name = args.output or ("mesh_game.glb" if args.game_ready else "mesh.glb")
    glb_path  = os.path.join(folder, out_name)
    target_faces = parse_target_faces(args.target_faces)

    if not os.path.isfile(ply_path):
        err(f"splat.ply not found in {folder}")
    if not os.path.isfile(slat_path):
        err(f"slat.pt not found in {folder} — re-run the CLI to regenerate (it now saves slat.pt)")

    device = "mps" if torch.backends.mps.is_available() and not torch.cuda.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    ok(f"Device: {device}")
    if args.game_ready:
        ok(f"Game-ready remesh: target={target_faces or 'auto'}")
    progress = make_progress(extra_units=1 if args.game_ready else 0)

    hdr("LOADING MESH DECODER")
    progress("phase", label="Load mesh decoder")
    step("Loading slat_decoder_mesh (~500 MB)…")
    t0 = time.time()
    decoder = load_mesh_decoder(device)
    progress.advance("Load mesh decoder", 1)
    ok(f"Mesh decoder ready  ({time.time()-t0:.1f}s)")

    hdr("LOADING ASSETS")
    progress("phase", label="Load GLB assets")
    step("Loading sparse latent (slat.pt)…")
    slat = load_slat(slat_path).to(device)
    ok(f"SLAT: {slat.feats.shape[0]:,} active voxels")

    step("Loading gaussian (splat.ply)…")
    gs = load_gaussian(ply_path, device)
    ok(f"Gaussian: {gs.get_xyz.shape[0]:,} splats")

    # Guard: a non-finite splat would produce garbage geometry.
    if (
        not bool(torch.isfinite(gs.get_xyz).all().detach().cpu().item())
        or not bool(torch.isfinite(slat.feats).all().detach().cpu().item())
    ):
        progress.close()
        err("splat.ply / slat.pt contain NaN/Inf (failed generation). "
            "Re-run the CLI to regenerate this object before converting to GLB.")
    progress.advance("Load GLB assets", 1)

    hdr("DECODING MESH")
    progress("phase", label="Decode mesh")
    step("Running mesh decoder…")
    t0 = time.time()
    with torch.no_grad():
        mesh_result = decoder(slat)[0]
    progress.advance("Decode mesh", 2)
    ok(f"Mesh decoded  ({time.time()-t0:.1f}s)  — {mesh_result.vertices.shape[0]:,} verts / {mesh_result.faces.shape[0]:,} faces")

    hdr("CLEANUP MESH + BAKE TEXTURE + EXPORT GLB")
    progress("phase", label="Cleanup + texture bake")
    t0 = time.time()
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb
    import torch as _t
    on_mps = _t.backends.mps.is_available() and not _t.cuda.is_available()
    texture_views = int_env("SAM3D_TEXTURE_VIEWS", 100)
    texture_render_resolution = int_env("SAM3D_TEXTURE_RENDER_RES", 1024)
    texture_size = int_env("SAM3D_TEXTURE_SIZE", 2048)
    if on_mps:
        step(
            "Mesh cleanup → streamed texture bake "
            f"({texture_views} views @ {texture_render_resolution}px, {texture_size}px atlas)…"
        )
    else:
        step("Running to_glb (texture baking ~1 min)…")
    # Full pipeline now runs on MPS via pure-PyTorch rasterizers:
    #   - mesh postprocess: pyvista decimation (triangulated first) + _fill_holes (z-buffered
    #     software mesh raster instead of nvdiffrast)
    #   - texture baking: gsplat_silicon multi-view render + z-buffered UV raster + grid_sample
    # fill_holes views/resolution are reduced on MPS so the software rasterizer stays tractable.
    glb = to_glb(
        gs,
        mesh_result,
        simplify=0.90,           # keep ~10% of faces (smoother geometry than 0.95)
        fill_holes=True,
        fill_holes_resolution=512 if on_mps else 1024,
        fill_holes_num_views=100 if on_mps else 1000,
        texture_size=texture_size,
        texture_views=texture_views,
        texture_render_resolution=texture_render_resolution,
        game_remesh=args.game_ready,
        game_target_faces=target_faces,
        texture_mode="average",  # smooth angle-weighted multi-view average (no Adam patchiness)
        with_mesh_postprocess=True,   # includes floater removal (remove_floaters default on)
        with_texture_baking=True,
        use_vertex_color=False,
        rendering_engine="pytorch3d",
        progress_callback=progress,
    )
    ok(f"GLB ready  ({time.time()-t0:.1f}s)")

    glb.export(glb_path)
    progress.advance("Export GLB", 1)
    if bool_env("SAM3D_PROGRESS_FINISH", True):
        progress.finish("Complete")
    else:
        progress.close()
    saved(out_name, glb_path)

    hdr("DONE")


if __name__ == "__main__":
    main()
