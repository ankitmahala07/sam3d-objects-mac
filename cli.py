#!/usr/bin/env python3
"""
SAM-3D Objects — CLI  (Apple Silicon / MPS)

Streamlined full pipeline:
    1. Ask for one image, multiple views of the same object, or a folder batch.
       Images are treated as raw photos — the background is removed here with
       rembg, so you never need to pre-extract them.
    2. Ask for an output folder name  (created as  ./outputs/<name>/ )
    3. Pick quality and export mode
    4. Run the full 3D pipeline immediately  (voxel → gaussian splat)

Writes  extracted.png, optional extracted_view_*.png, splat.ply, slat.pt
into  outputs/<name>/  or  outputs/<batch>/<image-name>/.
The textured GLB is produced afterwards by ply2glb.py (see  ./run.sh  →  full flow).
"""

import sys, os, subprocess, time, random, re, argparse, gc
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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


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


def _load_rgb_image(path):
    path = os.path.expanduser(path.strip().strip("'\""))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    img = PILImage.open(path)
    return path, img, np.array(img.convert("RGB"))


def _probe_image(path):
    with PILImage.open(path) as img:
        img.load()
        return img.width, img.height


def _safe_folder_name(value):
    stem = os.path.splitext(os.path.basename(value.rstrip("/")))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "image"


def _discover_image_files(folder):
    paths = []
    for name in sorted(os.listdir(folder), key=str.lower):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def _materialize_job_views(job):
    if "views" in job:
        return job["views"]
    views = []
    for path in job["paths"]:
        path, img, rgb = _load_rgb_image(path)
        ok(f"{img.width}×{img.height}  ←  {path}")
        views.append({"path": path, "rgb": rgb})
    return views


def _child_output_dir(batch_root, image_path, used_names):
    base = _safe_folder_name(image_path)
    name = base
    idx = 2
    while name in used_names:
        name = f"{base}_{idx:02d}"
        idx += 1
    used_names.add(name)
    return os.path.join(batch_root, name)


# ── prompts ───────────────────────────────────────────────────────────────────
def get_image_jobs():
    """Ask for one image, multiple views, or a folder batch.

    Folder batches keep only file paths in memory. Each image is later handled
    by a fresh worker process so overnight runs do not accumulate MPS state.
    """
    hdr("STEP 1 — IMAGES")
    print(f"  {B}?{RST}  Image input")
    print(f"     {G}▶{RST} [1] Single image")
    print(f"       [2] Multiple views of the same object")
    print(f"       [3] Folder of images (sequential overnight batch)")
    while True:
        mode = ask("Enter 1–3", 1)
        if mode in ("1", "2", "3"):
            break
        err("Enter 1, 2, or 3.")

    if mode == "1":
        while True:
            path = ask("Image path (drag file here)")
            try:
                path, img, rgb = _load_rgb_image(path)
                ok(f"{img.width}×{img.height}  ←  {path}")
                return [{"name": None, "views": [{"path": path, "rgb": rgb}]}], False
            except FileNotFoundError:
                err(f"File not found: {os.path.expanduser(path)}")
            except Exception as e:
                err(f"Could not open: {e}")

    if mode == "3":
        while True:
            folder = os.path.abspath(os.path.expanduser(ask("Image folder path (drag folder here)").strip().strip("'\"")))
            if not os.path.isdir(folder):
                err(f"Folder not found: {folder}")
                continue
            files = _discover_image_files(folder)
            if not files:
                err("No supported image files found (.png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff).")
                continue
            jobs = []
            for path in files:
                try:
                    width, height = _probe_image(path)
                    ok(f"Queued {os.path.basename(path)}  ({width}×{height})")
                    jobs.append({"name": _safe_folder_name(path), "paths": [path]})
                except Exception as e:
                    warn(f"Skipping {path}: {e}")
            if jobs:
                ok(f"{len(jobs)} image(s) queued for sequential generation")
                return jobs, True
            err("No readable images found in that folder.")

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
    return [{"name": None, "views": views}], False


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
        if export_mode in ("game", "both")
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


def get_output_folder(batch_mode=False):
    """Ask for a name only; the folder is always created under ./outputs/<name>/."""
    hdr("STEP 2 — OUTPUT NAME")
    outputs_root = os.path.join(ROOT, "outputs")
    os.makedirs(outputs_root, exist_ok=True)
    while True:
        prompt = "Batch output folder name" if batch_mode else "Output folder name"
        name = ask(prompt).strip().strip("'\"")
        # Only a name is allowed — strip any path the user may have pasted.
        name = os.path.basename(name.rstrip("/"))
        if not name:
            err("Please enter a name.")
            continue
        folder = os.path.join(outputs_root, name)
        os.makedirs(folder, exist_ok=True)
        if batch_mode:
            ok(f"Batch root → {folder}")
        else:
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
    print(f"     {G}▶{RST} [1] Game        ·  quad-retopo low-poly game mesh (default)")
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
    print(f"       The quad-retopo exporter may keep more faces when needed to preserve the object.")
    print(f"       Examples: 2000 for simple props, 10000 for complex objects")
    while True:
        target = ask("Target faces, or auto", "auto").strip().lower()
        if target == "auto":
            break
        if target.isdigit() and int(target) >= 500:
            break
        err("Enter auto or a number >= 500.")

    ok(f"Game mesh: quad-retopo target={target}")
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
    except Exception:
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
    del pipeline, output, gs
    gc.collect()
    if _t.backends.mps.is_available():
        _t.mps.empty_cache()
    return True


# ── batch helpers ─────────────────────────────────────────────────────────────
def _parse_worker_args(argv):
    if "--batch-worker" not in argv:
        return None
    parser = argparse.ArgumentParser(description="Run one queued SAM-3D batch item.")
    parser.add_argument("--batch-worker", action="store_true")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--export-mode", required=True, choices=("game", "unoptimised", "both"))
    parser.add_argument("--game-target", default="auto")
    parser.add_argument("--game-method", default="quality")
    return parser.parse_args(argv)


def _append_batch_failure(batch_root, item_name, message):
    path = os.path.join(batch_root, "batch_errors.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{item_name}\t{message}\n")


def _run_batch_worker(args):
    print(f"\n{BOLD}{M}  SAM-3D Objects  ·  Batch item{RST}")
    print(f"{DIM}  Apple Silicon / MPS  ·  rembg + gaussian splat{RST}")
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        image_views = _materialize_job_views({"paths": [args.image]})
        masks = _run_rembg(image_views, args.export_mode)
        ok(f"Output → {args.output_dir}")
        success = run_pipeline(
            image_views,
            masks,
            args.output_dir,
            args.steps,
            args.export_mode,
            args.game_target,
            args.game_method,
        )
        hdr("ITEM DONE")
        return 0 if success else 2
    except KeyboardInterrupt:
        print()
        err("Interrupted")
        return 130
    except Exception as e:
        err(f"Batch item failed: {e}")
        return 1


def _run_folder_batch(jobs, batch_root, steps, export_mode, game_target, game_method):
    hdr("BATCH QUEUE")
    ok(f"{len(jobs)} image(s) will run sequentially")
    ok(f"Batch root: {batch_root}")
    failures = []
    used_names = set()
    for idx, job in enumerate(jobs, start=1):
        image_path = job["paths"][0]
        item_name = job["name"]
        obj_dir = _child_output_dir(batch_root, image_path, used_names)
        hdr(f"BATCH {idx}/{len(jobs)} — {item_name}")
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--batch-worker",
            "--image",
            image_path,
            "--output-dir",
            obj_dir,
            "--steps",
            str(steps),
            "--export-mode",
            export_mode,
            "--game-target",
            str(game_target),
            "--game-method",
            str(game_method),
        ]
        try:
            result = subprocess.run(cmd, env=os.environ.copy())
            if result.returncode == 0:
                ok(f"Queued GLB conversion for {item_name}")
            else:
                message = f"generation exited with status {result.returncode}"
                warn(f"{item_name}: {message}; continuing")
                _append_batch_failure(batch_root, item_name, message)
                failures.append((item_name, message))
        except KeyboardInterrupt:
            print()
            err("Batch interrupted")
            raise
        except Exception as e:
            message = str(e)
            warn(f"{item_name}: {message}; continuing")
            _append_batch_failure(batch_root, item_name, message)
            failures.append((item_name, message))
        finally:
            gc.collect()
    hdr("BATCH SPLAT SUMMARY")
    ok(f"Succeeded: {len(jobs) - len(failures)}")
    if failures:
        warn(f"Failed: {len(failures)}  (see {os.path.join(batch_root, 'batch_errors.log')})")
    else:
        ok("Failed: 0")
    return not failures


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{M}  SAM-3D Objects  ·  CLI{RST}")
    print(f"{DIM}  Apple Silicon / MPS  ·  rembg + gaussian splat{RST}")

    jobs, batch_mode = get_image_jobs()
    obj_dir = get_output_folder(batch_mode=batch_mode)
    steps = get_steps()
    export_mode = get_export_mode()
    game_target, game_method = get_game_options(export_mode)

    if batch_mode:
        _run_folder_batch(jobs, obj_dir, steps, export_mode, game_target, game_method)
        hdr("ALL DONE")
        return

    image_views = _materialize_job_views(jobs[0])
    # Always extract every supplied view: find the largest foreground component
    # with rembg, then merge that mask into RGBA before generation.
    masks = _run_rembg(image_views, export_mode)

    run_pipeline(image_views, masks, obj_dir, steps, export_mode, game_target, game_method)
    hdr("ALL DONE")


if __name__ == "__main__":
    worker_args = _parse_worker_args(sys.argv[1:])
    if worker_args is not None:
        sys.exit(_run_batch_worker(worker_args))
    main()
