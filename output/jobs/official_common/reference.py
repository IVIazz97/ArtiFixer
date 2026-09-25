#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one object-removed reference view both official video-prior baselines are conditioned on.

Neither repo makes it: Omni-3DEdit takes a ``cond_view`` "obtained from image editing models
(e.g. QwenImage, GPT-4o)", MVInpainter an ``inpainted/`` first view ("we recommend Fooocus").
We run Qwen-Image-Edit-2511 (what Omni-3DEdit's paper uses) with a removal instruction, then keep
its pixels only inside the dilated object mask (feathered), so the reference is pixel-aligned
with the photo everywhere else. Both methods get this same image.

    <inputs>/reference/<stem>.png       composited reference (what the methods consume)
    <inputs>/reference/raw_<stem>.png   raw Qwen output, resized to the photo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

PROMPTS = {
    "kitchen": "Remove the orange LEGO bulldozer from the table. Fill its place with the woven placemats and the "
               "wooden table surface that are behind it. Keep everything else in the image exactly unchanged.",
    "garden": "Remove the round wooden table and the vase with dried flowers standing on it. Fill their place with the "
              "stone patio tiles and the grass lawn that are behind them. Keep everything else in the image exactly "
              "unchanged.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="output/official_inputs/<scene>.")
    parser.add_argument("--scene", required=True, choices=sorted(PROMPTS))
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--feather_px", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from diffusers import QwenImageEditPlusPipeline

    stem = json.loads((args.inputs / "views.json").read_text())["reference"]
    photo = Image.open(args.inputs / "images" / f"{stem}.png").convert("RGB")
    mask = Image.open(args.inputs / "masks" / f"{stem}.png").convert("L")

    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    edited = pipe(image=[photo], prompt=PROMPTS[args.scene], negative_prompt=" ",
                  num_inference_steps=args.steps, true_cfg_scale=args.true_cfg_scale, guidance_scale=1.0,
                  generator=torch.Generator("cuda").manual_seed(args.seed)).images[0]
    edited = edited.convert("RGB").resize(photo.size, Image.LANCZOS)

    alpha = np.asarray(mask.filter(ImageFilter.GaussianBlur(args.feather_px)), dtype=np.float32)[..., None] / 255.0
    alpha = np.maximum(alpha, np.asarray(mask, dtype=np.float32)[..., None] / 255.0)  # full weight inside the mask
    composite = alpha * np.asarray(edited, np.float32) + (1 - alpha) * np.asarray(photo, np.float32)

    out = args.inputs / "reference"
    out.mkdir(exist_ok=True)
    edited.save(out / f"raw_{stem}.png")
    Image.fromarray(composite.round().clip(0, 255).astype(np.uint8)).save(out / f"{stem}.png")
    print(f"reference {stem}: {PROMPTS[args.scene]!r} -> {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
