#!/usr/bin/env bash
# SAM-3D Objects — Apple Silicon runner
#
#   ./run.sh            image → gaussian splat → textured GLB   (the full flow)
#   ./run.sh glb <dir>  re-convert one output dir's splat.ply → mesh.glb
#
# The full flow runs in two separate processes on purpose: the CLI generates the
# splat and then EXITS, so macOS reclaims all of its model memory before the GLB
# decoder loads. Only one memory-heavy stage is ever alive at a time.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../s3d_env"
PY="$VENV/bin/python3"
[[ -x "$PY" ]] || PY="python3"     # fall back to whatever python3 is on PATH

# Disable the MPS memory cap so the pipeline isn't silently killed mid-inference.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1

CKPT_DIR="$SCRIPT_DIR/checkpoints/hf/checkpoints"

# --- first-launch model check --------------------------------------------------
# The model weights (~12 GB) are NOT bundled. If they are missing, print exactly
# what to download and where to put it, then stop.
check_models() {
    local missing=0 f
    local required=(
        pipeline.yaml
        ss_generator.ckpt slat_generator.ckpt
        ss_decoder.ckpt slat_decoder_gs.ckpt slat_decoder_mesh.ckpt
    )
    for f in "${required[@]}"; do
        [[ -s "$CKPT_DIR/$f" ]] || missing=1
    done
    if (( missing )); then
        cat <<EOF

──────────────────────────────────────────────────────────────
  MODEL WEIGHTS NOT FOUND
──────────────────────────────────────────────────────────────
  The SAM-3D Objects checkpoints (~12 GB) are not bundled with
  this repo. Download them once from Hugging Face, then re-run.

  1. Request access (one-time, instant-ish approval):
       https://huggingface.co/facebook/sam-3d-objects

  2. Authenticate:
       pip install 'huggingface-hub[cli]<1.0'
       hf auth login          # paste a token from hf.co/settings/tokens

  3. Download into this repo:
       mkdir -p checkpoints/hf
       hf download --repo-type model --max-workers 1 \\
         --local-dir checkpoints/hf-download \\
         facebook/sam-3d-objects
       mv checkpoints/hf-download/checkpoints checkpoints/hf/checkpoints
       rm -rf checkpoints/hf-download

  After that, this folder must contain the .ckpt / .yaml files:
       $CKPT_DIR/
──────────────────────────────────────────────────────────────
EOF
        exit 1
    fi
}

# --- available system memory (GB), macOS ---------------------------------------
# free + inactive + speculative + purgeable pages: memory the OS can hand out.
avail_gb() {
    local ps; ps=$(sysctl -n hw.pagesize)
    vm_stat | awk -v ps="$ps" '
        /^Pages free/        {gsub(/\./,"",$NF); f=$NF}
        /^Pages inactive/    {gsub(/\./,"",$NF); i=$NF}
        /^Pages speculative/ {gsub(/\./,"",$NF); s=$NF}
        /purgeable/          {gsub(/\./,"",$NF); p=$NF}
        END { printf "%d", (f+i+s+p)*ps/1073741824 }
    '
}

# Block until at least $1 GB is free. Poll for a while; if it never frees, fall
# back to asking the user to free memory and continue manually.
wait_for_memory() {
    local need="${1:-12}" waited=0 avail
    echo "  Ensuring ≥ ${need} GB free before loading the GLB models…"
    while true; do
        avail=$(avail_gb)
        if (( avail >= need )); then
            echo "  ✓ ${avail} GB free — proceeding."
            return 0
        fi
        if (( waited >= 120 )); then
            echo "  ⚠ only ${avail} GB free after ${waited}s (need ${need} GB)."
            echo "    Quit other memory-heavy apps to free it up."
            read -r -p "    Press Enter to continue anyway, or Ctrl-C to abort… " _ || true
            return 0
        fi
        echo "  … ${avail} GB free, waiting for ${need} GB…"
        sleep 5
        waited=$((waited + 5))
    done
}

# --- glb: re-convert an existing splat -----------------------------------------
if [[ "$1" == "glb" ]]; then
    check_models
    exec "$PY" "$SCRIPT_DIR/ply2glb.py" "${2:-}"
fi

# Only "", "full" run the full flow; anything else is a mistake — show usage.
if [[ -n "$1" && "$1" != "full" ]]; then
    echo "Usage:"
    echo "  ./run.sh [full]     image -> gaussian splat -> textured GLB"
    echo "  ./run.sh glb <dir>  re-convert one output dir's splat.ply -> mesh.glb"
    exit 1
fi

check_models

# --- full flow (default / 'full'): cli -> free memory -> glb -------------------
MANIFEST="$(mktemp /tmp/sam3d_manifest.XXXXXX)"
: > "$MANIFEST"

echo "──────────────────────────────────────────────"
echo "  STEP 1/2 — SPLAT GENERATION"
echo "──────────────────────────────────────────────"
# cli.py records the output dir into $SAM3D_MANIFEST, then exits so the OS
# reclaims ALL of its model memory before the GLB step starts.
SAM3D_MANIFEST="$MANIFEST" "$PY" "$SCRIPT_DIR/cli.py"

if [[ ! -s "$MANIFEST" ]]; then
    echo "  No splat output recorded (aborted or NaN) — nothing to convert."
    rm -f "$MANIFEST"
    exit 0
fi

echo "──────────────────────────────────────────────"
echo "  CLI finished & exited — its memory is now freed"
echo "──────────────────────────────────────────────"
wait_for_memory 12

echo "──────────────────────────────────────────────"
echo "  STEP 2/2 — GLB CONVERSION"
echo "──────────────────────────────────────────────"
while IFS= read -r objdir; do
    [[ -z "$objdir" ]] && continue
    wait_for_memory 12          # cheap if already free; guards multi-object runs
    echo "  → $objdir"
    "$PY" "$SCRIPT_DIR/ply2glb.py" "$objdir"
done < "$MANIFEST"

rm -f "$MANIFEST"
echo "  All done."
