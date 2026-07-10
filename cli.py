#!/usr/bin/env python3
"""
SAM-3D Objects — CLI  (Apple Silicon / MPS)

Streamlined full pipeline:
    1. Ask for one image, or multiple views of the same object. Images are
       treated as raw photos — the background is removed here with rembg, so
       you never need to pre-extract them.
    2. Ask for an output folder name  (created as  ./outputs/<name>/ )
    3. Pick quality and export mode
    4. Run the full 3D pipeline immediately  (voxel → gaussian splat)

Writes  extracted.png, optional extracted_view_*.png, splat.ply, slat.pt
into  outputs/<name>/.
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
    if mode in ("game", "experimental", "experimentalv2"):
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


def _load_rgb_image(path):
    path = os.path.expanduser(path.strip().strip("'\""))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    img = PILImage.open(path)
    return path, img, np.array(img.convert("RGB"))


# ── prompts ───────────────────────────────────────────────────────────────────
def get_image_views():
    """Ask for one image or multiple views. The first view is the primary
    geometry/depth view; extra views are used as conditioning references."""
    hdr("STEP 1 — IMAGES")
    print(f"  {B}?{RST}  Image input")
    print(f"     {G}▶{RST} [1] Single image")
    print(f"       [2] Multiple views of the same object")
    while True:
        mode = ask("Enter 1–2", 1)
        if mode in ("1", "2"):
            break
        err("Enter 1 or 2.")

    if mode == "1":
        while True:
            path = ask("Image path (drag file here)")
            try:
                path, img, rgb = _load_rgb_image(path)
                ok(f"{img.width}×{img.height}  ←  {path}")
                return [{"path": path, "rgb": rgb}]
            except FileNotFoundError:
                err(f"File not found: {os.path.expanduser(path)}")
            except Exception as e:
                err(f"Could not open: {e}")

    while True:
        raw_count = ask("Number of views", 2).strip()
        if raw_count.isdigit() and 2 <= int(raw_count) <= 6:
            view_count = int(raw_count)
            break
        err("Enter a number from 2 to 6.")

    views = []
    print(f"       View 1 is the primary/front view; other views guide conditioning.")
    for idx in range(view_count):
        label = "Primary image path" if idx == 0 else f"View {idx + 1} image path"
        while True:
            path = ask(label)
            try:
                path, img, rgb = _load_rgb_image(path)
                ok(f"View {idx + 1}: {img.width}×{img.height}  ←  {path}")
                views.append({"path": path, "rgb": rgb})
                break
            except FileNotFoundError:
                err(f"File not found: {os.path.expanduser(path)}")
            except Exception as e:
                err(f"Could not open: {e}")
    ok(f"{len(views)} view(s) loaded")
    return views


def _extract_masks(image_views):
    from app import _fg_components

    masks = []
    for idx, view in enumerate(image_views):
        comps = _fg_components(view["rgb"])
        ok(f"View {idx + 1}: found {len(comps)} foreground component(s)")
        masks.append(comps[0] if comps else np.ones(view["rgb"].shape[:2], bool))
    return masks


def _save_extracted_views(pipeline, image_views, masks, obj_dir):
    rgba_views = []
    for idx, (view, mask) in enumerate(zip(image_views, masks)):
        rgba = pipeline.merge_mask_to_rgba(view["rgb"], mask)
        rgba_views.append(rgba)
        filename = "extracted.png" if idx == 0 else f"extracted_view_{idx + 1:02d}.png"
        out_path = os.path.join(obj_dir, filename)
        PILImage.fromarray(rgba).save(out_path)
        saved(filename, out_path)
    return rgba_views


def _write_view_manifest(image_views, obj_dir):
    if len(image_views) <= 1:
        return
    path = os.path.join(obj_dir, "input_views.txt")
    with open(path, "w") as f:
        for idx, view in enumerate(image_views, start=1):
            f.write(f"{idx}\t{view['path']}\n")
    saved("input_views.txt", path)


def _condition_pil_images(rgba_views):
    return [PILImage.fromarray(rgba) for rgba in rgba_views[1:]]


def _primary_pil_image(rgba_views):
    return PILImage.fromarray(rgba_views[0])


def _multi_view_note(image_views):
    if len(image_views) > 1:
        ok(f"Multi-view conditioning enabled ({len(image_views)} view(s))")
    else:
        ok("Single-image conditioning")


def _image_extract_step(export_mode):
    return (
        "STEP 6"
        if export_mode in ("game", "both", "experimental", "experimentalv2")
        else "STEP 5"
    )


def _run_rembg(image_views, export_mode):
    hdr(f"{_image_extract_step(export_mode)} — EXTRACT (rembg)")
    step("Running rembg foreground detection…")
    return _extract_masks(image_views)


def _prepare_views_for_pipeline(pipeline, image_views, masks, obj_dir):
    rgba_views = _save_extracted_views(pipeline, image_views, masks, obj_dir)
    _write_view_manifest(image_views, obj_dir)
    _multi_view_note(image_views)
    return rgba_views


def _validate_views(image_views):
    if not image_views:
        raise RuntimeError("No image views loaded.")


def _condition_images_for_run(rgba_views):
    return _condition_pil_images(rgba_views)


def _primary_image_for_run(rgba_views):
    return _primary_pil_image(rgba_views)


def _format_run_view_info(image_views):
    return f"views={len(image_views)}"


def _warn_multi_view_cost(image_views):
    if len(image_views) > 1:
        warn("Multiple views compute extra depth/condition embeddings; use 2–4 views on 24 GB Macs.")


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
    print(f"     {G}▶{RST} [1] Game        ·  welded low-poly game mesh (default)")
    print(f"       [2] Unoptimised ·  original high-detail mesh")
    print(f"       [3] Both        ·  mesh_game.glb + mesh.glb")
    print(f"       [4] Experimental ·  in-repo quad-dominant retopology")
    print(f"       [5] Experimental V2 ·  quality-gated smoother quad retopology")
    while True:
        raw = ask("Enter 1–5", 1)
        if raw == "1":
            ok("Game output → mesh_game.glb")
            return "game"
        if raw == "2":
            ok("Unoptimised output → mesh.glb")
            return "unoptimised"
        if raw == "3":
            ok("Both outputs → mesh_game.glb and mesh.glb")
            return "both"
        if raw == "4":
            ok("Experimental output → mesh_experimental.glb + mesh_experimental_quads.obj")
            return "experimental"
        if raw == "5":
            ok("Experimental V2 output → mesh_experimental_v2.glb")
            return "experimentalv2"
        err("Enter a number between 1 and 5.")


def get_game_options(export_mode):
    if export_mode not in ("game", "both", "experimental", "experimentalv2"):
        return "auto", "quality"

    if export_mode in ("experimental", "experimentalv2"):
        hdr("STEP 5 — EXPERIMENTAL GENERATION")
        print(f"  {B}?{RST}  Initial triangle budget for experimental retopology")
        print(f"       The quality gate may raise this budget to preserve the source surface.")
    else:
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

    if export_mode in ("experimental", "experimentalv2"):
        label = "Experimental V2" if export_mode == "experimentalv2" else "Experimental"
        ok(f"{label} generation: quad-dominant target={target}")
        return target, export_mode
    ok(f"Game mesh: welded quality-safe target={target}")
    return target, "quality"


# ── full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(image_views, masks, obj_dir, steps, export_mode, game_target, game_method):
    import torch as _t

    _validate_views(image_views)

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
    _warn_multi_view_cost(image_views)
    rgba_views = _prepare_views_for_pipeline(pipeline, image_views, masks, obj_dir)
    primary_image = _primary_image_for_run(rgba_views)
    condition_images = _condition_images_for_run(rgba_views)

    # Single attempt only. fp16 diffusion on MPS can overflow to NaN under memory
    # pressure; we detect it and refuse to write a dead splat, but we do NOT retry
    # in-process — the prior attempt's ~15 GB stays allocated and a second run hits a
    # hard Metal allocation crash. Retry by re-running as a fresh process.
    step(f"Running  (steps={steps}  seed={seed}  {_format_run_view_info(image_views)})…")
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
            primary_image,
            seed=seed,
            stage1_inference_steps=steps,
            stage2_inference_steps=steps,
            condition_images=condition_images,
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

    image_views = get_image_views()
    obj_dir = get_output_folder()
    steps = get_steps()
    export_mode = get_export_mode()
    game_target, game_method = get_game_options(export_mode)

    # Always extract every supplied view: find the largest foreground component
    # with rembg, then merge that mask into RGBA before generation.
    masks = _run_rembg(image_views, export_mode)

    run_pipeline(image_views, masks, obj_dir, steps, export_mode, game_target, game_method)
    hdr("ALL DONE")


if __name__ == "__main__":
    main()
