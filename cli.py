#!/usr/bin/env python3
"""
SAM-3D Objects — CLI  (Apple Silicon / MPS)

Streamlined full pipeline:
    1. Ask for an image  (always treated as a raw photo — the background is
       removed here with rembg, so you never need to pre-extract it)
    2. Ask for an output folder name  (created as  ./outputs/<name>/ )
    3. Pick quality and export mode
    4. Run the full 3D pipeline immediately  (voxel → gaussian splat)

Writes  extracted.png, splat.ply, slat.pt  into  outputs/<name>/.
The textured GLB is produced afterwards by ply2glb.py (see  ./run.sh  →  full flow).
"""

import sys, os, subprocess, time, random
import numpy as np
from PIL import Image as PILImage
from sam3d_progress import CliProgress

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "notebook"))
os.environ.setdefault("SPARSE_BACKEND", "native")
# Disable MPS memory cap so the pipeline isn't silently killed mid-inference
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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


def int_env(name, default=0):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def export_progress_units(mode):
    if mode == "game":
        return 11
    if mode == "both":
        return 21
    return 10


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


def get_steps():
    """Pick diffusion steps (both stages) before the pipeline runs.
    Low (10) is the default and the only safe choice on 24 GB machines — higher
    step counts mostly just cost time (see README)."""
    hdr("STEP 3 — QUALITY")
    print(f"  {B}?{RST}  Diffusion steps (applied to both stages)")
    print(f"     {G}▶{RST} [1] Low     ·  10 steps   (recommended · required for 24 GB unified memory)")
    print(f"       [2] Medium  ·  25 steps")
    print(f"       [3] High    ·  50 steps")
    print(f"       [4] Custom")
    while True:
        raw = ask("Enter 1–4", 1)
        if raw == "1": steps = 10; break
        if raw == "2": steps = 25; break
        if raw == "3": steps = 50; break
        if raw == "4":
            while True:
                c = ask("Custom steps (positive integer)").strip()
                if c.isdigit() and int(c) > 0:
                    steps = int(c); break
                err("Enter a positive integer.")
            break
        err("Enter a number between 1 and 4.")
    if steps > 10:
        warn("More than 10 steps needs > 24 GB free and can crash on a 24 GB Mac. "
             "Quality gain is marginal (~3–5% more geometry); mostly it just takes longer.")
    ok(f"{steps} steps")
    return steps


def get_export_mode():
    hdr("STEP 4 — GLB OUTPUT")
    print(f"  {B}?{RST}  Which GLB should be generated?")
    print(f"     {G}▶{RST} [1] Game        ·  low-poly game mesh (default)")
    print(f"       [2] Unoptimised ·  original high-detail mesh")
    print(f"       [3] Both        ·  mesh_game.glb + mesh.glb")
    while True:
        raw = ask("Enter 1–3", 1)
        if raw == "1":
            ok("Game output → mesh_game.glb")
            return "game"
        if raw == "2":
            ok("Unoptimised output → mesh.glb")
            return "unoptimised"
        if raw == "3":
            ok("Both outputs → mesh_game.glb and mesh.glb")
            return "both"
        err("Enter a number between 1 and 3.")


def get_game_options(export_mode):
    if export_mode not in ("game", "both"):
        return "auto", "quality"

    hdr("STEP 5 — GAME MESH")
    print(f"  {B}?{RST}  Target triangle budget for the game mesh")
    print(f"       The exporter may keep more faces when needed to preserve the object.")
    print(f"       Examples: 2000 for simple props, 10000 for complex objects")
    while True:
        target = ask("Target faces, or auto", "auto").strip().lower()
        if target == "auto":
            break
        if target.isdigit() and int(target) >= 500:
            break
        err("Enter auto or a number >= 500.")

    ok(f"Game mesh: quality-safe target={target}")
    return target, "quality"


# ── full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(img_rgb, mask, obj_dir, steps, export_mode, game_target, game_method):
    import torch as _t

    hdr("GPU / MEMORY CHECK")
    try:
        gpu_procs = subprocess.check_output(
            ["pgrep", "-lf", "ollama|ComfyUI|webui"],
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
    pipeline = Inference(CONFIG, compile=False, low_memory_single_shot=True)
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
    step(f"Running  (steps={steps}  seed={seed})…")
    import gc
    gc.collect()
    if _t.backends.mps.is_available():
        _t.mps.empty_cache()
    t0 = time.time()
    splat_progress_units = steps * 2 + 5
    manifest = os.environ.get("SAM3D_MANIFEST")
    glb_progress_units = export_progress_units(export_mode) if manifest else 0
    progress = CliProgress(total=splat_progress_units + glb_progress_units)
    try:
        output = pipeline._pipeline.run(
            PILImage.fromarray(rgba),
            seed=seed,
            stage1_inference_steps=steps,
            stage2_inference_steps=steps,
            decode_formats=["gaussian"],   # skip mesh decoder entirely (ply2glb does it)
            # Single-shot run: free each stage's models before the next stage
            # allocates. Frees stage-1 models before SLAT and moves finished
            # splats to CPU before PLY serialization.
            free_stage_models=True,
            fail_on_nan=True,
            return_pointmap=False,
            return_latents=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=False,
            progress_callback=progress,
        )
    except FloatingPointError as e:
        progress.close()
        err(f"{e}. Free memory (quit other GPU apps) and re-run; not writing a dead splat.ply.")
        if _t.backends.mps.is_available():
            _t.mps.empty_cache()
        return False
    except RuntimeError as e:
        msg = str(e)
        if any(s in msg.lower() for s in ("mps", "metal", "out of memory", "allocate", "allocation")):
            progress.close()
            err("MPS allocation failed during generation. Quit other GPU/RAM-heavy apps "
                "and re-run; not writing a partial splat.ply.")
            if _t.backends.mps.is_available():
                _t.mps.empty_cache()
            return False
        progress.close()
        raise

    gs = output.get("gs") or output["gaussian"][0]
    finite_tensors = (gs.get_xyz, gs._features_dc, gs._scaling, gs._rotation, gs._opacity)
    if any(not bool(_t.isfinite(t).all().detach().cpu().item()) for t in finite_tensors if t is not None):
        progress.close()
        err("Output contains NaN/Inf (fp16 overflow — usually memory pressure). "
            "Free memory (quit other GPU apps) and re-run; not writing a dead splat.ply.")
        if _t.backends.mps.is_available():
            _t.mps.empty_cache()
        return False

    p_ply = os.path.join(obj_dir, "splat.ply")
    slat = output.get("slat")
    p_slat = None
    try:
        gs.save_ply(p_ply)
        if slat is not None:
            p_slat = os.path.join(obj_dir, "slat.pt")
            _t.save({"feats": slat.feats.cpu(), "coords": slat.coords.cpu(),
                     "shape": slat.shape}, p_slat)
    except ValueError as e:
        progress.close()
        err(f"{e}. Not writing a dead splat.ply.")
        return False
    except Exception:
        progress.close()
        raise

    progress.advance("Save splat outputs", 1)
    if glb_progress_units:
        progress("phase", label="Splat ready; GLB next")
        progress.close()
    else:
        progress.finish("Complete")
    ok(f"Done  ({time.time()-t0:.1f}s)")
    saved("splat.ply", p_ply)
    ok(f"{gs.get_xyz.shape[0]:,} gaussians")
    if p_slat is not None:
        saved("slat.pt", p_slat)

    # If launched via run.sh's full flow, record the obj dir so the wrapper can run
    # the GLB step after this process exits (freeing all CLI memory first).
    if manifest and slat is not None:
        with open(manifest, "a") as _mf:
            _mf.write(
                f"{obj_dir}\t{progress.done}\t{progress.total}\t"
                f"{export_mode}\t{game_target}\t{game_method}\n"
            )
    return True


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{M}  SAM-3D Objects  ·  CLI{RST}")
    print(f"{DIM}  Apple Silicon / MPS  ·  rembg + gaussian splat{RST}")

    img_rgb  = get_image()
    obj_dir  = get_output_folder()
    steps    = get_steps()
    export_mode = get_export_mode()
    game_target, game_method = get_game_options(export_mode)

    # Always extract: find the largest foreground component with rembg.
    from app import _fg_components
    extract_step = "STEP 6" if export_mode in ("game", "both") else "STEP 5"
    hdr(f"{extract_step} — EXTRACT (rembg)")
    step("Running rembg foreground detection…")
    comps = _fg_components(img_rgb)
    ok(f"Found {len(comps)} foreground component(s)")
    mask = comps[0] if comps else np.ones(img_rgb.shape[:2], bool)

    run_pipeline(img_rgb, mask, obj_dir, steps, export_mode, game_target, game_method)
    hdr("ALL DONE")


if __name__ == "__main__":
    main()
