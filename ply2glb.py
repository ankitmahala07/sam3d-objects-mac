#!/usr/bin/env python3
"""
ply2glb — Convert a SAM-3D gaussian splat to a textured GLB mesh.

Usage:
    python ply2glb.py <output_folder>
    ./run.sh glb <output_folder>

Loads only the mesh decoder (~500 MB), not the full inference pipeline.
Requires slat.pt and splat.ply to exist in <output_folder>.
"""

import sys, os, time
import torch
import numpy as np

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

    if len(sys.argv) < 2:
        print(f"  Usage: {sys.argv[0]} <output_folder_containing_splat.ply_and_slat.pt>")
        sys.exit(1)

    folder = os.path.abspath(sys.argv[1].strip().strip("'\""))
    ply_path  = os.path.join(folder, "splat.ply")
    slat_path = os.path.join(folder, "slat.pt")
    glb_path  = os.path.join(folder, "mesh.glb")

    if not os.path.isfile(ply_path):
        err(f"splat.ply not found in {folder}")
    if not os.path.isfile(slat_path):
        err(f"slat.pt not found in {folder} — re-run the CLI to regenerate (it now saves slat.pt)")

    device = "mps" if torch.backends.mps.is_available() and not torch.cuda.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    ok(f"Device: {device}")

    hdr("LOADING MESH DECODER")
    step("Loading slat_decoder_mesh (~500 MB)…")
    t0 = time.time()
    decoder = load_mesh_decoder(device)
    ok(f"Mesh decoder ready  ({time.time()-t0:.1f}s)")

    hdr("LOADING ASSETS")
    step("Loading sparse latent (slat.pt)…")
    slat = load_slat(slat_path).to(device)
    ok(f"SLAT: {slat.feats.shape[0]:,} active voxels")

    step("Loading gaussian (splat.ply)…")
    gs = load_gaussian(ply_path, device)
    ok(f"Gaussian: {gs.get_xyz.shape[0]:,} splats")

    # Guard: a NaN splat (fp16 overflow during generation) would produce garbage geometry.
    if bool(torch.isnan(gs.get_xyz).any()) or bool(torch.isnan(slat.feats).any()):
        err("splat.ply / slat.pt contain NaN (failed fp16 generation). "
            "Re-run the CLI to regenerate this object before converting to GLB.")

    hdr("DECODING MESH")
    step("Running mesh decoder…")
    t0 = time.time()
    with torch.no_grad():
        mesh_result = decoder(slat)[0]
    ok(f"Mesh decoded  ({time.time()-t0:.1f}s)  — {mesh_result.vertices.shape[0]:,} verts / {mesh_result.faces.shape[0]:,} faces")

    hdr("CLEANUP MESH + BAKE TEXTURE + EXPORT GLB")
    t0 = time.time()
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import to_glb
    import torch as _t
    on_mps = _t.backends.mps.is_available() and not _t.cuda.is_available()
    if on_mps:
        step("Mesh cleanup (decimate + fill holes) → 100-view render → texture optimize — several minutes…")
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
        texture_size=2048,       # higher-res atlas -> more colour detail
        texture_mode="average",  # smooth angle-weighted multi-view average (no Adam patchiness)
        with_mesh_postprocess=True,   # includes floater removal (remove_floaters default on)
        with_texture_baking=True,
        use_vertex_color=False,
        rendering_engine="pytorch3d",
    )
    ok(f"GLB ready  ({time.time()-t0:.1f}s)")

    glb.export(glb_path)
    saved("mesh.glb", glb_path)

    hdr("DONE")


if __name__ == "__main__":
    main()
