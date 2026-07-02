# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import torch

# Route .cuda() and device="cuda" to MPS on Apple Silicon
if torch.backends.mps.is_available() and not torch.cuda.is_available():
    def _mps_cuda(self, *args, **kwargs):
        return self.to("mps")
    torch.Tensor.cuda = _mps_cuda

    import torch.nn as nn
    def _mps_module_cuda(self, device=None):
        return self.to("mps")
    nn.Module.cuda = _mps_module_cuda

import os
import shutil
import datetime
import numpy as np
from PIL import Image as PILImage

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# ── paths ──────────────────────────────────────────────────────────────────
IMAGE_PATH  = "notebook/images/pirate_character/image.png"
MASK_FOLDER = "notebook/images/pirate_character"
MASK_INDEX  = 0

# ── output folder ──────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join("outputs", timestamp)
os.makedirs(out_dir, exist_ok=True)
print(f"Output folder: {out_dir}")

# ── copy original image ────────────────────────────────────────────────────
shutil.copy(IMAGE_PATH, os.path.join(out_dir, "original.png"))

# ── load inputs ───────────────────────────────────────────────────────────
image = load_image(IMAGE_PATH)   # RGBA numpy [H,W,4]
mask  = load_single_mask(MASK_FOLDER, index=MASK_INDEX)  # bool [H,W]

# save extracted (masked) image — original RGB with background removed
rgba = image.copy()
rgba[..., 3] = (mask * 255).astype(np.uint8)
PILImage.fromarray(rgba).save(os.path.join(out_dir, "extracted.png"))

# ── load model ────────────────────────────────────────────────────────────
tag = "hf"
config_path = f"checkpoints/{tag}/checkpoints/pipeline.yaml"
inference_pipeline = Inference(config_path, compile=False)

# ── run model ─────────────────────────────────────────────────────────────
output = inference_pipeline(image, mask, seed=42)

# ── save outputs ──────────────────────────────────────────────────────────
ply_path = os.path.join(out_dir, "splat.ply")
output["gs"].save_ply(ply_path)
print(f"Gaussian splat saved to {ply_path}")

if output.get("glb") is not None:
    glb_path = os.path.join(out_dir, "output.glb")
    output["glb"].export(glb_path)
    print(f"Textured mesh saved to {glb_path}")
else:
    print("No textured mesh output (mesh decoder may have been skipped)")

print(f"\nAll outputs in: {out_dir}/")
