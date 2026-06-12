# SAM-3D-Objects Apple Silicon Port Log

## Machine: Mac Mini, Apple Silicon (M-series), 24 GB, macOS

---

## Session 1 — 2026-06-12

### Milestone 0: Clean imports ✅

**Environment:**
- Python 3.11 venv at `../s3d_env`
- PyTorch 2.12.0 with MPS support (`torch.backends.mps.is_available() == True`)
- No CUDA packages installed

**Changes made:**

1. **`sam3d_objects/_kaolin_stub.py`** (new) — minimal kaolin shim.
   `check_tensor` is the only inference-path kaolin use. Stub installs
   `kaolin.utils.testing.check_tensor` as a no-op.

2. **`sam3d_objects/__init__.py`** — import `_kaolin_stub` at top.

3. **`sam3d_objects/model/backbone/tdfy_dit/modules/sparse/__init__.py`**
   — added `"native"` to allowed `SPARSE_BACKEND` values.

4. **`sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/__init__.py`**
   — added `elif BACKEND == "native": from .conv_native import *`

5. **`sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/native_backend.py`** (new)
   — `NativeSparseData`: pure-Python drop-in for `spconv.SparseConvTensor`.
   Stores `features` [N,C] and `indices` [N,4] int32 (batch,z,y,x).
   Implements `.dense()` and `.replace_feature()`.

6. **`sam3d_objects/model/backbone/tdfy_dit/modules/sparse/conv/conv_native.py`** (new)
   — Three conv ops in pure PyTorch:
   - `SparseConv3d` (dispatches to subm or strided)
   - `SparseInverseConv3d`
   Weight layout: `[K, K, K, in_ch, out_ch]` matching spconv checkpoint format.
   All ops work on CPU; coords stay int32.

7. **`sam3d_objects/model/backbone/tdfy_dit/modules/sparse/basic.py`**
   — Added `"native"` branch to `feats`/`coords` props, `__init__`, `replace`, `dense`.

8. **`notebook/inference.py`**
   — Fixed `CONDA_PREFIX` KeyError (use `.get()` with fallback).
   — Guarded `kaolin.visualize` / `pytorch3d.transforms` with try/except.
   — Added MPS auto-detection: if MPS available and no CUDA, set
     `config.device = "mps"` and `config.dtype = "float16"`.

9. **`sam3d_objects/pipeline/inference_pipeline.py`**
   — Removed `torch.cuda.current_device()` call when CUDA unavailable.
   — Replaced `with self.device:` context (CUDA-only) with `nullcontext()`.
   — Changed `device_type="cuda"` → `device_type=self.device.type` in autocast.
   — Changed checkpoint loading from `device="cuda"` → `device="cpu"`.

10. **`sam3d_objects/pipeline/inference_pipeline_pointmap.py`**
    — Guarded pytorch3d imports.
    — Changed `device_type="cuda"` → `device_type=self.device.type` in autocast.

11. **Various pytorch3d / gsplat guards** in:
    - `sam3d_objects/pipeline/inference_utils.py`
    - `sam3d_objects/pipeline/layout_post_optimization_utils.py`
    - `sam3d_objects/data/dataset/tdfy/transforms_3d.py`
    - `sam3d_objects/data/dataset/tdfy/pose_target.py`
    - `sam3d_objects/utils/visualization/scene_visualizer.py`
    - `sam3d_objects/utils/visualization/plotly/plot_scene.py`
    - `sam3d_objects/utils/visualization/image_mesh.py`
    - `sam3d_objects/model/backbone/tdfy_dit/renderers/gaussian_render.py`

12. **`sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/general_utils.py`**
    — Fixed `device="cuda"` hardcodes → `device=L.device`, `device=r.device`, etc.
    — Guarded `torch.cuda.set_device`.

13. **`sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/gaussian_model.py`**
    — Fixed `.cuda()` calls → `.to(_dev)` using the `device` constructor arg.

14. **`sam3d_objects/model/backbone/tdfy_dit/modules/attention/full_attn.py`**
    — Changed bfloat16 autocast to use float16 on non-CUDA devices.

**Verification:**
```
SPARSE_BACKEND=native ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -c "import sys; sys.path.insert(0,'notebook'); \
  from inference import Inference, load_image, load_single_mask; \
  print('Inference import OK')"
# → Inference import OK
```

---

## Milestone 1: gaussian `.ply` — BLOCKED on checkpoints

**Status:** Need to download checkpoints from HuggingFace.

**Required steps before running demo:**
1. Request access at https://huggingface.co/facebook/sam-3d-objects
2. Run `huggingface-cli login`
3. Download checkpoints (see README or plan doc)

**Run command once checkpoints are available:**
```bash
source ../s3d_env/bin/activate
export SPARSE_BACKEND=native
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export PYTORCH_ENABLE_MPS_FALLBACK=1
python demo.py
```

**Expected issues still to fix (will hit at runtime):**
- `torch.cuda.manual_seed` calls in `layout_post_optimization_utils.py`
  (lines 306-307) — gate behind `if torch.cuda.is_available():`
- Hardcoded `.cuda()` in `postprocessing_utils.py` (not in gaussian decode path)
- Any MPS-unsupported ops — `PYTORCH_ENABLE_MPS_FALLBACK=1` will handle most;
  watch for ops that are wrong (not just missing) on MPS.
- Memory: implement sequential offload if 24 GB is insufficient

---

## Environment Setup (for fresh start)

```bash
# From sam-3d-objects/ directory:
source ../s3d_env/bin/activate

# Required env vars:
export SPARSE_BACKEND=native
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Installed packages (key ones):
- torch 2.12.0 (MPS enabled, arm64)
- torchvision, torchaudio (plain, no +cu121)
- hydra-core, omegaconf, loguru, einops, timm, open3d
- safetensors, roma, rootutils, huggingface-hub
- scikit-image, trimesh, pymeshfix, xatlas, point-cloud-utils
- optree, astor, igraph, pyvista, lightning
- moge (from git), gradio, seaborn, utils3d
