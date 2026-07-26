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

That's the whole thing. It asks for the source image(s), output folder, and
quality, then runs end to end:

1. **Images** — choose a single ordinary photo, multiple views of the same
   object, or a folder of images for a sequential overnight batch. The first
   multi-view image is the primary geometry/depth view; the other views are
   extra conditioning references. The background is removed automatically
   (rembg); you do **not** need to pre-extract the object.
2. **Output folder name** — single/multi-view results are written to
   `outputs/<name>/`. Folder batches are written as
   `outputs/<batch-name>/<image-name>/`.
3. **Quality** — diffusion steps for both stages:
   `Low = 10` (default), `Medium = 25`, `High = 50`, or a custom value.
4. **Complete mesh output** — the wrapper generates `mesh.glb`, converts it to
   `mesh.obj`, then creates the textured game asset family.

Output in `outputs/<name>/`:

| File            | What it is                                             |
|-----------------|--------------------------------------------------------|
| `extracted.png` | the object with background removed (RGBA)              |
| `extracted_view_02.png`, ... | optional extracted reference views        |
| `input_views.txt` | optional manifest of the supplied view paths          |
| `splat.ply`     | the raw gaussian splat                                 |
| `slat.pt`       | the sparse latent (input to the mesh decoder)          |
| `mesh.glb`      | high-detail textured source mesh                       |
| `mesh.obj`      | OBJ conversion of `mesh.glb`                           |
| `mesh.mtl`      | material file for `mesh.obj`                           |
| `mesh_texture.png` | texture extracted for `mesh.obj`                    |
| `mesh_game.glb` | textured game runtime mesh                             |
| `mesh_game.obj` | textured game OBJ                                      |
| `mesh_game.mtl` | material file for `mesh_game.obj`                      |
| `mesh_game_texture.png` | texture baked onto the game asset             |
| `mesh_game_normal.png` | optional tangent normal-map sidecar            |
| `mesh_game_quads.obj` | editable quad-dominant topology sidecar        |
| `mesh_game_report.json` | topology and surface-error measurements       |

Folder batches run one source image at a time. Each image is handled by a fresh
worker process, then the wrapper converts each successful `splat.ply` to the
complete mesh output family. Failed images are logged to `batch_errors.log` and
the queue continues.

Multiple views are memory-sensitive. View 1 drives depth and pose; all supplied
views are averaged into the Stage 1 and Stage 2 condition embeddings before
generation. Extra views are streamed through the condition embedders one at a
time to avoid a batch-size memory spike, but they still add depth and
conditioning work, so 2-4 views is the practical range on a 24 GB Mac.

The mesh stage builds the cleaned high-detail `mesh.glb` source first, converts
that GLB to `mesh.obj`, reloads the `mesh.glb` geometry as the game source, then
fits `mesh_game.glb` before UV unwrap and texture baking. The texture is baked
directly onto the finalized game asset. The automatic game target aims for
roughly 2k-10k vertices; it may keep more geometry when a lower count would
damage the silhouette or texture bake. It uses an in-repo signed-distance and
QEF dual-contouring implementation:

1. align a bounded grid to the source object's principal frame;
2. fit one feature-aware vertex per intersected cell;
3. connect those cells into a regular quad-dominant surface;
4. locally subdivide high-error and curved patches with manifold transition faces;
5. fair low-curvature regions while locking creases, notches, and thin tips;
6. repair ambiguous local patches and reject boundary or non-manifold results;
7. measure bidirectional surface error and raise the grid resolution when the
   requested budget loses too much shape.

No external retopology executable or service is used. The topology sidecar
preserves editable quads where the grid fit succeeds; the GLB is triangulated for
runtime compatibility and receives its texture after the game topology has been
finalized. A tangent normal-map sidecar is baked from the cleaned high-detail
source surface for shallow detail. Local transition or repair triangles may
appear around adaptively refined patches or where multiple source sheets meet
inside one grid cell; their counts are recorded in the JSON report.

For open or thin multi-part objects, the quad builder can occasionally produce a
manifold but heavier runtime mesh than the cleaned source. In that case the game
export falls back to the cleaned source mesh instead of writing a larger, worse
`mesh_game.glb`.

GLB files are runtime meshes and are stored as triangles. Set
`SAM3D_GAME_TARGET_VERTICES=2000..10000` to force a vertex target; leave it unset
for automatic quality-preserving targeting.

**Rebuild the mesh outputs only** (skips the expensive splat step):

```bash
./run.sh glb outputs/<name>
```

If `mesh.glb` already exists and is loadable, this command skips the mesh decoder
and optimizes that source GLB directly into the game asset family. If `mesh.glb`
is missing, corrupt, or `SAM3D_GAME_REBUILD_SOURCE_GLB=1` is set, `slat.pt` is
required so the high-detail source mesh can be rebuilt first.

**Compare source and game meshes from 12 directions**:

```bash
python tools/compare_glb_views.py outputs/<name>
```

This writes overlays to `outputs/<name>/comparison_12views/`.

The main quad-retopo limits can be adjusted without changing code. Some
environment variable names are kept stable for compatibility:

| Variable | Default | Purpose |
|----------|--------:|---------|
| `SAM3D_EXPERIMENTAL_ERROR_P95` | `0.015` | allowed p95 surface error as a fraction of object bounds |
| `SAM3D_EXPERIMENTAL_MAX_TARGET_MULT` | `4` | maximum automatic budget increase |
| `SAM3D_EXPERIMENTAL_MAX_FACES` | `40000` | hard automatic face ceiling |
| `SAM3D_EXPERIMENTAL_MAX_AXIS` | `144` | largest signed-distance grid dimension |
| `SAM3D_EXPERIMENTAL_QUALITY_ATTEMPTS` | `2` | reconstruction attempts before acceptance |
| `SAM3D_EXPERIMENTAL_MIN_THICKNESS_CELLS` | `16` | minimum global grid resolution across open-mesh thickness |
| `SAM3D_EXPERIMENTAL_SHELL_BAND` | `0.55` | unsigned shell width in grid-cell units |
| `SAM3D_EXPERIMENTAL_ADAPTIVE_MAX_FRACTION` | `0.12` | maximum fraction of quads selected for local subdivision |
| `SAM3D_EXPERIMENTAL_ADAPTIVE_ERROR` | `0.10` | local source-error threshold in grid-cell diagonals |
| `SAM3D_EXPERIMENTAL_ADAPTIVE_ANGLE` | `16` | local source-normal threshold in degrees |
| `SAM3D_EXPERIMENTAL_ADAPTIVE_MAX_TRANSITION_RATIO` | `0.15` | maximum runtime triangle share used by adaptive transitions |
| `SAM3D_EXPERIMENTAL_NORMAL_SIZE` | texture size | tangent normal-map resolution |
| `SAM3D_EXPERIMENTAL_ATTACH_NORMAL_MAP` | `0` | attach the optional tangent normal map to game GLBs |
| `SAM3D_EXPERIMENTAL_NORMAL_STRENGTH` | `0.35` | high-poly geometric normal contribution |
| `SAM3D_EXPERIMENTAL_ALBEDO_RELIEF` | `0.08` | subtle high-frequency crack/detail contribution |
| `SAM3D_EXPERIMENTAL_REFINEMENT_PROFILES` | `3` | number of internal smoothing candidates to evaluate |
| `SAM3D_EXPERIMENTAL_REFINEMENT_VOLUME_CHANGE` | `0.04` | maximum accepted refinement volume change |
| `SAM3D_EXPERIMENTAL_POLISH_CHAIN_MAX_COMPONENTS` | `1` | smooth only the longest sharp-edge chain by default |
| `SAM3D_EXPERIMENTAL_POLISH_CHAIN_ITERS` | `8` | sharp-edge curve fairing iterations |
| `SAM3D_EXPERIMENTAL_POLISH_CHAIN_WEIGHT` | `0.32` | sharp-edge curve fairing strength |
| `SAM3D_EXPERIMENTAL_SPLAT_GEOMETRY_WEIGHT` | `0.0` | optional splat influence over geometry; appearance still comes from splats |
| `SAM3D_EXPERIMENTAL_DIRECT_COLOR` | `0` | use faster direct splat color instead of the smoother streamed texture bake |
| `SAM3D_GAME_REBUILD_SOURCE_GLB` | `0` | rebuild `mesh.glb` before game export even when it already exists |
| `SAM3D_GAME_TARGET_VERTICES` | `auto` | optional game vertex target from 2,000 to 10,000 |
| `SAM3D_GAME_MAX_VERTICES` | `10000` | source meshes at or below this vertex count can be used directly for game output |
| `SAM3D_GAME_MAX_FACES` | `20000` | hard runtime triangle budget for bounded source fallback |
| `SAM3D_GAME_USE_SOURCE_WHEN_WITHIN_BUDGET` | `1` | keep `mesh.glb` geometry exactly when it is already game-sized |
| `SAM3D_GAME_FALLBACK_ON_RETOPO_FAILURE` | `1` | use a bounded source-derived runtime mesh instead of failing when grid retopo quality checks reject |
| `SAM3D_GAME_RETOPO_SOURCE_FALLBACK_MAX_FACES` | `20000` | allow source fallback only when the cleaned source is already game-sized |
| `SAM3D_GAME_RETOPO_SOURCE_FALLBACK_RATIO` | `1.25` | fallback when retopo produces this much more runtime geometry than the source |
| `SAM3D_EXPERIMENTAL_REJECT_SKINNY_TOPOLOGY` | `0` | turn skinny topology warnings back into hard failures |

---

## Example results

Generated crate sample:

<p>
  <img src="outputs/crate/extracted.png" width="360" alt="Generated crate cutout">
</p>

| Result | Face budget | File |
|--------|------------:|------|
| Game mesh | automatic | [`outputs/crate/mesh_game.glb`](outputs/crate/mesh_game.glb) |
| Source mesh | high detail | [`outputs/crate/mesh.glb`](outputs/crate/mesh.glb) |

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

**Large-object mesh decode fallback.** If a generated object has a very large
`slat.pt`, the fp32 mesh decoder may still exceed MPS memory. GLB conversion now
automatically decodes large SLATs on CPU, frees the decoder, then loads gaussian
splats back on MPS for texture baking. This is slower but avoids the macOS
`killed` failure during `DECODING MESH`. Override with:

```bash
SAM3D_MESH_DECODE_DEVICE=mps ./run.sh game outputs/<name> 1600
SAM3D_CPU_DECODE_VOXELS=50000 ./run.sh game outputs/<name> 1600
```

---

## How it works

The run is split across **separate OS processes** so that only one memory-heavy
stage is ever resident — macOS only reclaims a process's GPU memory when it
exits. Single-image runs exit the CLI before GLB conversion. Folder batches
spawn one fresh splat worker per image, collect successful outputs, then run the
GLB conversions afterwards.

```
 ┌── Stage 1: cli.py ───────────────┐        ┌── Stage 2: ply2glb.py ───────┐
 │  photo → rembg mask              │        │  slat.pt → mesh decoder      │
 │  → sparse-structure diffusion    │  exit  │  → selected mesh generation  │
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
