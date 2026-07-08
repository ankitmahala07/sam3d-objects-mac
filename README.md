# SAM 3D Objects — Apple Silicon (macOS / MPS) port

Run Meta's **[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)**
image → 3D pipeline on an Apple Silicon Mac. Give it a photo, get back a gaussian
splat and a **textured `.glb` mesh** — no CUDA, no NVIDIA GPU.

The upstream model is CUDA-only. This fork replaces every CUDA-specific piece
(spconv, gsplat, nvdiffrast, float64 kernels) with pure-PyTorch / MPS-friendly
equivalents so the whole thing runs on the Mac's unified-memory GPU. See
[PORT_LOG.md](PORT_LOG.md) for the full list of changes, and
[README.upstream.md](README.upstream.md) for Meta's original README.

> ⚠️ This is an unofficial community port. The model, weights and the
> **SAM License** ([LICENSE](LICENSE)) belong to Meta. Your use of the weights
> is governed by that license — this repository only adds macOS glue code.

---

## Requirements

- **Apple Silicon** Mac (M1/M2/M3/M4). Tested on a Mac with ~24–30 GB unified memory.
- **macOS** with a recent PyTorch that has MPS enabled.
- **~12 GB** free disk for the model weights + working memory headroom.
- Python 3.11.

Memory is the main constraint: the pipeline is memory-heavy and runs **one stage
at a time** on purpose (see *How it works*). Quit other GPU-hungry apps (browsers,
Ollama, etc.) before a run — memory pressure is the usual cause of a failed
(all-NaN) result.

---

## Install

### Easiest: guided setup (recommended for non-technical users)

```bash
./setup.sh
```

`setup.sh` walks you through the whole thing — it checks/installs Python, creates
the environment, installs the packages, opens the Hugging Face pages you need,
logs you in, and downloads the model weights. When it finishes, just run
`./run.sh`. (Your Hugging Face token is entered into Hugging Face's own tool and
is never stored in this project.)

Prefer to do it by hand? Follow the manual steps below.

### Manual install

```bash
# 1. Create a Python 3.11 virtual environment next to the repo
python3.11 -m venv ../s3d_env
source ../s3d_env/bin/activate

# 2. Install PyTorch (MPS build) + dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio          # arm64 / MPS build
pip install -r requirements.txt
pip install -e .

# 3. rembg (background removal) + trimesh/xatlas etc. are in requirements.txt
```

`run.sh` expects the venv at `../s3d_env` (sibling of this repo). If yours lives
elsewhere, it falls back to whatever `python3` is on your `PATH`.

Key packages: `torch` (MPS), `hydra-core`, `omegaconf`, `trimesh`, `pymeshfix`,
`xatlas`, `pyvista`, `rembg`, `moge`, `utils3d`, `gradio`.

---

## Getting the model weights

The weights (~12 GB) are **not** in this repo. `./run.sh` will detect if they're
missing and print these steps. Download them once:

1. **Request access** (one-time) on Hugging Face:
   <https://huggingface.co/facebook/sam-3d-objects>

2. **Authenticate:**
   ```bash
   pip install 'huggingface-hub[cli]<1.0'
   hf auth login          # paste a token from https://hf.co/settings/tokens
   ```

3. **Download into the repo:**
   ```bash
   mkdir -p checkpoints/hf
   hf download --repo-type model --max-workers 1 \
     --local-dir checkpoints/hf-download \
     facebook/sam-3d-objects
   mv checkpoints/hf-download/checkpoints checkpoints/hf/checkpoints
   rm -rf checkpoints/hf-download
   ```

When done, this directory must exist and hold the `.ckpt` / `.yaml` files:

```
checkpoints/hf/checkpoints/
├── pipeline.yaml
├── ss_generator.ckpt        (~6.2 GB)
├── slat_generator.ckpt      (~4.6 GB)
├── ss_decoder.ckpt
├── slat_decoder_gs.ckpt
├── slat_decoder_mesh.ckpt
└── … (matching .yaml configs)
```

---

## Usage

```bash
./run.sh
```

That's the whole thing. It asks for the source image, output folder, quality,
and GLB output mode, then runs end to end:

1. **Image path** — any ordinary photo. The background is removed automatically
   (rembg); you do **not** need to pre-extract the object.
2. **Output folder name** — results are written to `outputs/<name>/`.
3. **Quality** — diffusion steps for both stages:
   `Low = 10` (default), `Medium = 25`, `High = 50`, or a custom value.
4. **GLB output** — choose the generated mesh export:
   `Game` (default), `Unoptimised`, or `Both`.
5. **Game mesh settings** — shown only for `Game` or `Both`: target face count
   and remesh method.

Output in `outputs/<name>/`:

| File            | What it is                                             |
|-----------------|--------------------------------------------------------|
| `extracted.png` | the object with background removed (RGBA)              |
| `splat.ply`     | the raw gaussian splat                                 |
| `slat.pt`       | the sparse latent (input to the mesh decoder)          |
| `mesh_game.glb` | optional/default game-oriented low-poly textured mesh  |
| `mesh.glb`      | optional unoptimised high-detail textured mesh         |

The `Game` export remeshes before UV unwrap and texture baking, so the texture is
baked directly onto the lower-poly asset. `Both` creates `mesh_game.glb` first
and then `mesh.glb` for side-by-side comparison.

Game remesh methods:

| Method | Use when | Notes |
|--------|----------|-------|
| Existing | you want the stable current flow | fast quadric decimation to the requested face budget |
| Experimental | complex objects need more shape preservation | keeps more source geometry before reduction and uses feature-aware retopo |

GLB files are runtime meshes and are stored as triangles. The experimental mode
tries to produce cleaner automatic topology, but true senior-artist quad loops
still require a dedicated retopology tool or manual cleanup in a DCC app.

**Re-bake the mesh only** (skips the expensive splat step) from an existing
`splat.ply` + `slat.pt`:

```bash
./run.sh glb outputs/<name>
```

**Create only the game mesh** from an existing result folder:

```bash
./run.sh game outputs/<name> 2000 experimental
```

Use `auto` instead of a number to pick a target automatically. Use `decimate`
instead of `experimental` for the stable existing method.

---

## Example results

Generated crate sample:

<p>
  <img src="outputs/crate/extracted.png" width="360" alt="Generated crate cutout">
</p>

| Result | Faces | File |
|--------|------:|------|
| Game mesh | 2,000 | [`outputs/crate/mesh_game.glb`](outputs/crate/mesh_game.glb) |
| Unoptimised mesh | 14,994 | [`outputs/crate/mesh.glb`](outputs/crate/mesh.glb) |

Open either `.glb` link on GitHub to use its built-in rotatable 3D viewer.

---

## Performance & memory  ⚠️

A full run takes roughly **40–80 minutes** depending on the quality you pick.

**If you have 24 GB of unified memory, use Low (10 steps) only.** Medium/High
need more headroom and will typically crash a 24 GB Mac (out-of-memory →
all-NaN result or an `Abort trap: 6`).

More steps buy you very little. Going from 10 → 50 steps produced only about
**3% more vertices and 5% more faces** in testing — while Stage 1 alone gets much
slower:

| Quality      | Steps | Stage-1 time (avg) | Geometry vs. Low |
|--------------|-------|--------------------|------------------|
| **Low**      | 10    | ~10 min            | baseline         |
| Medium       | 25    | ~24 min            | ≈ +a few %       |
| High         | 50    | ~45 min            | ~+3% verts / +5% faces |

The difference in the final mesh is minor; the main cost of higher steps is time
(and memory). Low is the recommended setting for almost everyone.

**Low-memory mode (automatic on MPS).** The splat step keeps the big diffusion
backbones in fp16 (instead of fp32, ~halving their RAM) and frees each stage's
models as soon as it's done — the depth model before Stage 1, the
sparse-structure model before Stage 2, the SLAT model before decoding. This
lowers the peak enough to run on smaller Macs; still use **Low (10 steps)** on
24 GB and close other apps first.

---

## How it works

The run is split into **two separate OS processes** so that only one
memory-heavy stage is ever resident — macOS only reclaims a process's GPU memory
when it exits.

```
 ┌── Stage 1: cli.py ───────────────┐        ┌── Stage 2: ply2glb.py ───────┐
 │  photo → rembg mask              │        │  slat.pt → mesh decoder      │
 │  → sparse-structure diffusion    │  exit  │  → decimate + fill holes     │
 │  → SLAT diffusion                │ ─────▶ │  → multi-view texture bake   │
 │  → gaussian splat  (splat.ply)   │ (frees │  → textured mesh.glb         │
 │  → sparse latent   (slat.pt)     │  mem)  │                              │
 └──────────────────────────────────┘        └──────────────────────────────┘
```

Between stages `run.sh` waits until enough memory is free before loading the mesh
decoder. Generation uses fp16 with a random seed in `0–41` (seed 42 was observed
to overflow to NaN under memory pressure); an all-NaN result is detected and
**not** written, so you never get a silently-dead `splat.ply`.

### What was ported from CUDA

- **`gsplat_silicon`** — pure-PyTorch EWA gaussian rasterizer replacing the
  CUDA-only `gsplat`.
- **`mesh_raster_silicon`** — tile-based z-buffered triangle rasterizer replacing
  `nvdiffrast` for face-id / UV rasterization and hole filling.
- **native sparse conv** — pure-PyTorch drop-in for `spconv`.
- **float64 on CPU** — MPS has no float64; camera math and splat sort keys are
  computed on CPU and moved back to the device.
- **fp32 mesh decoder** — the mesh decoder's attention overflows in fp16 on MPS,
  so it is forced to fp32.

---

## Troubleshooting

- **`MODEL WEIGHTS NOT FOUND`** — download the weights (see above).
- **Output is NaN / crash mid-run** — memory pressure. Quit other GPU apps and
  re-run; each run is a fresh process, so retries start clean.
- **`Abort trap: 6` / MTLBuffer allocation failure** — not enough free memory for
  the model. Close apps and don't run two heavy jobs at once.

---

## Credits & License

- Original model, research and weights: **Meta / SAM 3D Team** —
  [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects).
- Apple Silicon port: this repository.

Use of the model and weights is subject to Meta's **SAM License** — see
[LICENSE](LICENSE). This port is provided as-is for research/personal use under
those same terms.
