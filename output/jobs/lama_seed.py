#!/usr/bin/env python3
"""Inpaint seed frames with LaMa (big-lama TorchScript) inside the removal hole.

Writes <output_dir>/<i>.png at photo resolution, usable as propagate_removal.py --seed_pred_dir.
Same I/O convention as simple-lama-inpainting: RGB and mask in [0, 1], padded to a multiple of 8.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--scene_root", type=Path, required=True)
parser.add_argument("--removal_dir", type=Path, required=True)
parser.add_argument("--frames", required=True, help="e.g. '0' or '0,116' or '0-6'.")
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
args = parser.parse_args()

split = json.loads((args.scene_root / "split.json").read_text())["test"]
(_, entry), = split.items()
frames = json.loads((args.scene_root / entry["transforms_path"]).read_text())["frames"]
photo_root = args.scene_root / entry["image_root"]
args.output_dir.mkdir(parents=True, exist_ok=True)

indices = []
for part in args.frames.split(","):
    a, _, b = part.partition("-")
    indices += range(int(a), int(b or a) + 1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load(str(args.model), map_location=device).eval()

for i in indices:
    photo = np.asarray(Image.open(photo_root / frames[i]["file_path"]).convert("RGB"))
    hole = np.asarray(Image.open(args.removal_dir / "hole" / f"{i:05d}.png").convert("L")) > 127
    h, w = hole.shape
    image = torch.from_numpy(photo).permute(2, 0, 1)[None].float().div(255).to(device)
    mask = torch.from_numpy(hole)[None, None].float().to(device)
    pad_h, pad_w = (-h) % 8, (-w) % 8
    image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.inference_mode():
        result = model(image, mask)[0, :, :h, :w]
    result = (result.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    result = np.where(hole[..., None], result, photo)
    Image.fromarray(result).save(args.output_dir / f"{i:05d}.png")
    print(f"frame {i}: inpainted {hole.mean():.1%} of the image")
