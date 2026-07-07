#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  SAM 3D Objects (Apple Silicon) — guided one-time setup
#
#  For non-technical users. Run this ONCE:
#
#      ./setup.sh
#
#  It walks you through everything, opening the web pages you need and pausing
#  when it needs you to do something. When it finishes, run  ./run.sh
# ──────────────────────────────────────────────────────────────────────────────
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../s3d_env"
PY="$VENV/bin/python3"
CKPT_DIR="$SCRIPT_DIR/checkpoints/hf/checkpoints"

# ── pretty output ─────────────────────────────────────────────────────────────
BOLD=$'\033[1m'; DIM=$'\033[2m'; RST=$'\033[0m'
G=$'\033[92m'; Y=$'\033[93m'; R=$'\033[91m'; C=$'\033[96m'; W=$'\033[97m'

title() { printf "\n${BOLD}${C}══════════════════════════════════════════════════════════${RST}\n${BOLD}${W}  %s${RST}\n${DIM}══════════════════════════════════════════════════════════${RST}\n" "$1"; }
step()  { printf "  ${C}›${RST}  %s\n" "$1"; }
ok()    { printf "  ${G}✓${RST}  %s\n" "$1"; }
warn()  { printf "  ${Y}⚠${RST}  %s\n" "$1"; }
err()   { printf "  ${R}✗${RST}  %s\n" "$1"; }

pause() { printf "\n  ${BOLD}${Y}%s${RST}" "${1:-Press Enter to continue…}"; read -r _ || true; }
ask_yn() { # ask_yn "question" -> returns 0 for yes
    local q="$1" a
    printf "  ${BOLD}?${RST}  %s ${DIM}[y/N]${RST} " "$q"
    read -r a || true
    [[ "$a" =~ ^[Yy] ]]
}
open_url() { # open a link in the default browser (macOS), or just print it
    step "Opening: ${BOLD}$1${RST}"
    open "$1" 2>/dev/null || warn "Could not open a browser — please visit the link above manually."
}

# ── 0. sanity: macOS + Apple Silicon ──────────────────────────────────────────
title "SAM 3D Objects — Setup"
echo "  This will set up everything you need. It takes a while (mostly the"
echo "  ~12 GB model download). You only need to do this once."
echo

if [[ "$(uname -s)" != "Darwin" ]]; then
    err "This setup is for macOS. Detected: $(uname -s)."
    exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    warn "This is tuned for Apple Silicon (M1/M2/M3/M4). Your Mac reports '$(uname -m)'."
    ask_yn "Continue anyway?" || exit 1
else
    ok "Apple Silicon Mac detected."
fi
pause "Press Enter to begin…"

# ── 1. Python 3.11 ────────────────────────────────────────────────────────────
title "STEP 1 of 4 — Python"

find_py311() {
    for c in python3.11 python3; do
        if command -v "$c" >/dev/null 2>&1; then
            if "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)' 2>/dev/null; then
                echo "$c"; return 0
            fi
        fi
    done
    return 1
}

PY311="$(find_py311 || true)"
if [[ -n "${PY311:-}" ]]; then
    ok "Found Python 3.11 ($("$PY311" --version 2>&1))."
else
    warn "Python 3.11 is not installed."
    if command -v brew >/dev/null 2>&1; then
        step "Homebrew is installed. It can install Python 3.11 for you."
        if ask_yn "Install Python 3.11 with Homebrew now?"; then
            brew install python@3.11 || { err "Homebrew install failed."; exit 1; }
            PY311="$(find_py311 || true)"
        fi
    else
        warn "Homebrew (a Mac software installer) is not installed."
        echo "     Two options:"
        echo "       A) Install Homebrew, then re-run this setup (recommended)."
        echo "       B) Download Python 3.11 from python.org yourself."
        if ask_yn "Open the Homebrew install instructions in your browser?"; then
            open_url "https://brew.sh"
        fi
        if ask_yn "Open the Python 3.11 download page too?"; then
            open_url "https://www.python.org/downloads/release/python-3119/"
        fi
        err "Install Python 3.11, then run ./setup.sh again."
        exit 1
    fi
fi
[[ -n "${PY311:-}" ]] || { err "Still no Python 3.11 — install it and re-run."; exit 1; }

# ── 2. Virtual environment + Python packages ──────────────────────────────────
title "STEP 2 of 4 — Install Python packages"

if [[ -x "$PY" ]] && "$PY" -c 'import torch' >/dev/null 2>&1; then
    ok "Environment already set up (found PyTorch). Skipping install."
else
    if [[ ! -x "$PY" ]]; then
        step "Creating an isolated environment at: ${BOLD}$VENV${RST}"
        "$PY311" -m venv "$VENV" || { err "Could not create the environment."; exit 1; }
    fi
    step "Upgrading pip…"
    "$PY" -m pip install --quiet --upgrade pip
    step "Installing PyTorch (Apple Silicon / MPS build)… this can take a few minutes."
    "$PY" -m pip install --quiet torch torchvision torchaudio || { err "PyTorch install failed."; exit 1; }
    if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
        step "Installing the rest of the requirements… (several minutes)"
        "$PY" -m pip install --quiet -r "$SCRIPT_DIR/requirements.txt" || warn "Some requirements failed — you may need to re-run."
    fi
    step "Installing the project…"
    ( cd "$SCRIPT_DIR" && "$PY" -m pip install --quiet -e . ) || warn "Project install reported an issue."
    if "$PY" -c 'import torch; assert torch.backends.mps.is_available()' 2>/dev/null; then
        ok "PyTorch installed and Apple Silicon GPU (MPS) is available."
    else
        warn "PyTorch installed but MPS was not detected — the app may run on CPU (very slow)."
    fi
fi

# ── 3. Hugging Face access + login ─────────────────────────────────────────────
title "STEP 3 of 4 — Get access to the model"

echo "  The 3D model is published by Meta on Hugging Face. It's free, but you"
echo "  must (1) request access and (2) log in once with a personal token."
echo "  ${DIM}Your token is entered directly into Hugging Face's tool — it is never"
echo "  stored in this project.${RST}"
echo

step "First: request access to the model (click 'Agree and access repository')."
pause "Press Enter to open the access page…"
open_url "https://huggingface.co/facebook/sam-3d-objects"
echo
warn "Wait until Hugging Face shows you have access before continuing."
pause "Press Enter once you've been granted access…"

# make sure the huggingface CLI is available in our environment
if ! "$PY" -c 'import huggingface_hub' >/dev/null 2>&1; then
    step "Installing the Hugging Face downloader…"
    "$PY" -m pip install --quiet 'huggingface-hub[cli]<1.0' || { err "Could not install huggingface-hub."; exit 1; }
fi
HF="$VENV/bin/hf"; [[ -x "$HF" ]] || HF="$VENV/bin/huggingface-cli"

# already logged in?
if "$HF" auth whoami >/dev/null 2>&1 || "$HF" whoami >/dev/null 2>&1; then
    ok "Already logged in to Hugging Face."
else
    echo
    step "Next: create a personal access token (an 'access key')."
    echo "     On the page that opens, click ${BOLD}Create new token${RST} → type Read → Create,"
    echo "     then COPY the token (it looks like ${DIM}hf_xxxxxxxx${RST})."
    pause "Press Enter to open the token page…"
    open_url "https://huggingface.co/settings/tokens"
    echo
    step "Now paste the token when asked below (you will not see it as you type)."
    pause "Press Enter to start the login…"
    "$HF" auth login || "$HF" login || { err "Login failed — re-run ./setup.sh to try again."; exit 1; }
    ok "Logged in to Hugging Face."
fi

# ── 4. Download the model weights (~12 GB) ────────────────────────────────────
title "STEP 4 of 4 — Download the model (~12 GB)"

models_present() {
    local f
    for f in pipeline.yaml ss_generator.ckpt slat_generator.ckpt \
             ss_decoder.ckpt slat_decoder_gs.ckpt slat_decoder_mesh.ckpt; do
        [[ -s "$CKPT_DIR/$f" ]] || return 1
    done
    return 0
}

if models_present; then
    ok "Model weights are already downloaded. Nothing to do."
else
    echo "  This downloads ~12 GB. It can take a long time on a slow connection,"
    echo "  and it can be safely re-run if it's interrupted."
    if ask_yn "Start the download now?"; then
        mkdir -p "$SCRIPT_DIR/checkpoints/hf"
        step "Downloading… (progress is shown below)"
        "$HF" download --repo-type model --max-workers 1 \
            --local-dir "$SCRIPT_DIR/checkpoints/hf-download" \
            facebook/sam-3d-objects || { err "Download failed — re-run ./setup.sh to resume."; exit 1; }
        if [[ -d "$SCRIPT_DIR/checkpoints/hf-download/checkpoints" ]]; then
            rm -rf "$CKPT_DIR"
            mv "$SCRIPT_DIR/checkpoints/hf-download/checkpoints" "$CKPT_DIR"
        fi
        rm -rf "$SCRIPT_DIR/checkpoints/hf-download"
        if models_present; then
            ok "Model weights downloaded to checkpoints/hf/checkpoints/."
        else
            err "Download finished but some files are missing — re-run ./setup.sh."
            exit 1
        fi
    else
        warn "Skipped. You can run ./setup.sh again later to download."
        exit 0
    fi
fi

# ── done ──────────────────────────────────────────────────────────────────────
title "All set! 🎉"
echo "  Everything is installed. To create a 3D model from a photo, run:"
echo
echo "      ${BOLD}${G}./run.sh${RST}"
echo
echo "  Tip: on a 24 GB Mac, choose ${BOLD}Low (10 steps)${RST} and close other apps first."
echo
