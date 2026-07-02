#!/usr/bin/env python3
"""
SAM-3D Objects — CLI  (Apple Silicon / MPS)

Streamlined full pipeline:
    1. Ask for an image  (always treated as a raw photo — the background is
       removed here with rembg, so you never need to pre-extract it)
    2. Ask for an output folder name  (created as  ./outputs/<name>/ )
    3. Run the full 3D pipeline immediately  (voxel → gaussian splat)

Writes  extracted.png, splat.ply, slat.pt  into  outputs/<name>/.
The textured GLB is produced afterwards by ply2glb.py (see  ./run.sh  →  full flow).
"""

import sys, os, subprocess, time, random
import numpy as np
from PIL import Image as PILImage

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "notebook"))
os.environ.setdefault("SPARSE_BACKEND", "native")
# Disable MPS memory cap so the pipeline isn't silently killed mid-inference
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Diffusion steps for both stages (STANDARD quality ≈ 12 min on Apple Silicon).
STEPS = 25

# ── colour helpers ────────────────────────────────────────────────────────────
R    = "\033[91m"; G  = "\033[92m"; Y  = "\033[93m"
B    = "\033[94m"; M  = "\033[95m"; C  = "\033[96m"
W    = "\033[97m"; DIM = "\033[2m"; BOLD = "\033[1m"; RST = "\033[0m"

def hdr(msg):   print(f"\n{BOLD}{C}{'─'*58}{RST}\n{BOLD}{W}  {msg}{RST}\n{DIM}{'─'*58}{RST}")
def saved(label, path): print(f"  {G}▶ SAVED{RST}  {BOLD}{label:<16}{RST}  {path}")
def step(msg):  print(f"  {C}›{RST}  {msg}", flush=True)
def ok(msg):    print(f"  {G}✓{RST}  {msg}", flush=True)
def warn(msg):  print(f"  {Y}⚠{RST}   {msg}")
def err(msg):   print(f"  {R}✗{RST}   {msg}")

def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    try:
        val = input(f"  {B}?{RST}  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)
    return val if val else (str(default) if default is not None else "")


# ── prompts ───────────────────────────────────────────────────────────────────
def get_image():
    """Ask for an image path. The image is always treated as a raw photo; the
    background is removed automatically further down (rembg)."""
    hdr("STEP 1 — IMAGE")
    while True:
        path = ask("Image path (drag file here)").strip().strip("'\"")
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            err(f"File not found: {path}")
            continue
        try:
            img = PILImage.open(path)
            rgb = np.array(img.convert("RGB"))
            ok(f"{img.width}×{img.height}  ←  {path}")
            return rgb
        except Exception as e:
            err(f"Could not open: {e}")


def get_output_folder():
    """Ask for a name only; the folder is always created under ./outputs/<name>/."""
    hdr("STEP 2 — OUTPUT NAME")
    outputs_root = os.path.join(ROOT, "outputs")
    os.makedirs(outputs_root, exist_ok=True)
    while True:
        name = ask("Output folder name").strip().strip("'\"")
        # Only a name is allowed — strip any path the user may have pasted.
        name = os.path.basename(name.rstrip("/"))
        if not name:
            err("Please enter a name.")
            continue
        folder = os.path.join(outputs_root, name)
        os.makedirs(folder, exist_ok=True)
        ok(f"→ {folder}")
        return folder


# ── full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(img_rgb, mask, obj_dir):
    import torch as _t

    hdr("GPU / MEMORY CHECK")
    try:
        gpu_procs = subprocess.check_output(
            ["pgrep", "-lf", "Claude Helper|ollama"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()
        if gpu_procs:
            warn("These processes are also using GPU memory (may cause a watchdog crash):")
            for p in gpu_procs:
                print(f"      {DIM}{p}{RST}")
            warn("Quit them if the run fails with NaN / an allocation error.")
    except subprocess.CalledProcessError:
        ok("No competing GPU processes detected")

    hdr("LOADING PIPELINE")
    step("Loading pipeline…  (may take ~30s on first run)")
    CONFIG = os.path.join(ROOT, "checkpoints", "hf", "checkpoints", "pipeline.yaml")
    from inference import Inference
    pipeline = Inference(CONFIG, compile=False)
    ok("Pipeline ready")

    seed = random.randint(0, 41)   # capped 0-41: 42 was observed to overflow fp16
    ok(f"Using seed {seed}")

    hdr("OBJECT")
    rgba = pipeline.merge_mask_to_rgba(img_rgb, mask)
    p_ext = os.path.join(obj_dir, "extracted.png")
    PILImage.fromarray(rgba).save(p_ext)
    saved("extracted.png", p_ext)

    # Single attempt only. fp16 diffusion on MPS can overflow to NaN under memory
    # pressure; we detect it and refuse to write a dead splat, but we do NOT retry
    # in-process — the prior attempt's ~15 GB stays allocated and a second run hits a
    # hard Metal allocation crash. Retry by re-running as a fresh process.
    step(f"Running  (steps={STEPS}  seed={seed})…")
    t0 = time.time()
    output = pipeline._pipeline.run(
        PILImage.fromarray(rgba),
        seed=seed,
        stage1_inference_steps=STEPS,
        stage2_inference_steps=STEPS,
        decode_formats=["gaussian"],   # skip mesh decoder entirely (ply2glb does it)
        with_mesh_postprocess=False,
        with_texture_baking=False,
        use_vertex_color=False,
    )
    ok(f"Done  ({time.time()-t0:.1f}s)")

    gs = output.get("gs") or output["gaussian"][0]
    if bool(_t.isnan(gs.get_xyz).any()) or bool(_t.isnan(gs._features_dc).any()):
        err("Output is NaN (fp16 overflow — usually memory pressure). "
            "Free memory (quit other GPU apps) and re-run; not writing a dead splat.ply.")
        if _t.backends.mps.is_available():
            _t.mps.empty_cache()
        return False

    p_ply = os.path.join(obj_dir, "splat.ply")
    gs.save_ply(p_ply)
    saved("splat.ply", p_ply)
    ok(f"{gs.get_xyz.shape[0]:,} gaussians")

    # Save the sparse latent so ply2glb.py can convert to a textured mesh later.
    slat = output.get("slat")
    if slat is not None:
        p_slat = os.path.join(obj_dir, "slat.pt")
        _t.save({"feats": slat.feats.cpu(), "coords": slat.coords.cpu(),
                 "shape": slat.shape}, p_slat)
        saved("slat.pt", p_slat)

    # If launched via run.sh's full flow, record the obj dir so the wrapper can run
    # the GLB step after this process exits (freeing all CLI memory first).
    manifest = os.environ.get("SAM3D_MANIFEST")
    if manifest and slat is not None:
        with open(manifest, "a") as _mf:
            _mf.write(obj_dir + "\n")
    return True


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{M}  SAM-3D Objects  ·  CLI{RST}")
    print(f"{DIM}  Apple Silicon / MPS  ·  rembg + gaussian splat{RST}")

    img_rgb  = get_image()
    obj_dir  = get_output_folder()

    # Always extract: find the largest foreground component with rembg.
    from app import _fg_components
    hdr("STEP 3 — EXTRACT (rembg)")
    step("Running rembg foreground detection…")
    comps = _fg_components(img_rgb)
    ok(f"Found {len(comps)} foreground component(s)")
    mask = comps[0] if comps else np.ones(img_rgb.shape[:2], bool)

    run_pipeline(img_rgb, mask, obj_dir)
    hdr("ALL DONE")


if __name__ == "__main__":
    main()
