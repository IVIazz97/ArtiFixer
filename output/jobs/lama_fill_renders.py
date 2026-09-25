#!/usr/bin/env python3
"""Copy a removal split, LaMa-filling the hole in every conditioning render (and reference).

ArtiFixer keeps the removed object's footprint (smudge + silhouette) visible in the render it is
conditioned on. Filling the hole with LaMa first hands it plausible texture to harmonise instead.
--opacity keeps the source opacity (hole0) or sets full opacity inside the hole (render).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--opacity", choices=("hole0", "full"), default="full")
args = parser.parse_args()

src, out = args.source_dir, args.output_dir
for sub in ("renders", "opacity", "hole", "refs"):
    (out / sub).mkdir(parents=True, exist_ok=True)
model = torch.jit.load(str(args.model), map_location="cuda").eval()
transforms = json.loads((src / "transforms.json").read_text())

for i, frame in enumerate(transforms["frames"]):
    render = np.asarray(Image.open(src / "renders" / f"{i:05d}.png").convert("RGB"))
    hole = np.asarray(Image.open(src / "hole" / f"{i:05d}.png").convert("L")) > 127
    h, w = hole.shape
    image = torch.from_numpy(render.copy()).permute(2, 0, 1)[None].float().div(255).cuda()
    mask = torch.from_numpy(hole.copy())[None, None].float().cuda()
    ph, pw = (-h) % 8, (-w) % 8
    with torch.inference_mode():
        filled = model(F.pad(image, (0, pw, 0, ph), mode="reflect"), F.pad(mask, (0, pw, 0, ph), mode="reflect"))
    filled = (filled[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    filled = np.where(hole[..., None], filled, render)
    Image.fromarray(filled).save(out / "renders" / f"{i:05d}.png")
    ref = out / "refs" / frame["file_path"]
    ref.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(filled).save(ref)
    opacity = np.asarray(Image.open(src / "opacity" / f"{i:05d}.png").convert("L"))
    if args.opacity == "full":
        opacity = np.where(hole, 255, opacity).astype(np.uint8)
    Image.fromarray(opacity).save(out / "opacity" / f"{i:05d}.png")
    shutil.copy(src / "hole" / f"{i:05d}.png", out / "hole" / f"{i:05d}.png")

for name in ("transforms.json", "selected_indices.json"):
    shutil.copy(src / name, out / name)
split = json.loads((src / "split.json").read_text())
for entry in split["test"].values():
    entry.update(transforms_path=str(out / "transforms.json"), image_root=str(out / "refs"),
                 render_dir=str(out / "renders"), opacity_dir=str(out / "opacity"),
                 selected_indices_path=str(out / "selected_indices.json"))
(out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
print(f"LaMa-filled {len(transforms['frames'])} renders -> {out} (opacity={args.opacity})")
