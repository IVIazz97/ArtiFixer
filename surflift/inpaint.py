#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SurfaceLift stage 4 (artifixer env): inpaint the chosen view inside its hole U
(SurfaceLift scripts/inpaint_view.py).

Input: select/source.png = the photo (virtual view: the render) outside the object mask, the
cleaned scene inside it (what photos confirmed), grey in U.
  --editor lama (default): big-lama TorchScript on that image with mask dilate(U, 4).
  --editor qwen: Qwen-Image-Edit-2511 from a local folder (--qwen_dir), CPU offload, SurfaceLift's
                 v3 grey-hole fill prompt with a generic surface description (no per-scene prompt).
The result is pasted inside U only (4 px feather), as inpaint_view.py does; outside U the source
stays exact.
-> select/{inpainted_raw.png, inpainted.png}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation

from surflift.common import Timer, job_summary

FILL_PROMPT = ("The grey area is a missing part of the scene, where an object stood. Fill it with the floor and the "
               "surfaces that continue around it, seamlessly: patterns, edges, joints, colour and lighting must line up "
               "with what is around the grey area. Do not add any object. Keep everything else in the image exactly unchanged.")


def lama(model_path: Path, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """big-lama TorchScript at full resolution (aurafusion/agdd.py lama_inpaint): (H, W, 3) uint8 -> uint8."""
    import torch.nn.functional as F
    model = torch.jit.load(str(model_path), map_location="cuda").eval()
    h, w = mask.shape
    x = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().div(255).cuda()
    m = torch.from_numpy(mask.astype(np.float32))[None, None].cuda()
    ph, pw = (-h) % 8, (-w) % 8
    with torch.inference_mode():
        y = model(F.pad(x, (0, pw, 0, ph), mode="reflect"), F.pad(m, (0, pw, 0, ph), mode="reflect"))
    y = (y[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return np.where(mask[..., None], y, image)


def qwen(qwen_dir: Path, image: np.ndarray, steps: int, seed: int) -> np.ndarray:
    from diffusers import QwenImageEditPlusPipeline
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(qwen_dir), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload()
    src = Image.fromarray(image)
    out = pipe(image=[src], prompt=FILL_PROMPT, negative_prompt=" ", num_inference_steps=steps, true_cfg_scale=4.0,
               guidance_scale=1.0, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    return np.asarray(out.convert("RGB").resize(src.size, Image.LANCZOS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--editor", choices=("lama", "qwen"), default="lama")
    ap.add_argument("--lama_model", type=Path, default=None)
    ap.add_argument("--qwen_dir", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask_dilate", type=int, default=4)
    args = ap.parse_args()
    d = args.work / "select"
    timer = Timer(args.work, "inpaint")
    src = np.asarray(Image.open(d / "source.png").convert("RGB"))
    hole_img = Image.open(d / "hole.png").convert("L")
    U = np.asarray(hole_img) > 127
    assert U.any(), f"{d / 'hole.png'} is empty"
    if args.editor == "qwen":
        assert args.qwen_dir is not None and args.qwen_dir.exists(), f"--qwen_dir {args.qwen_dir} missing"
        edited = qwen(args.qwen_dir, src, args.steps, args.seed)
    else:
        assert args.lama_model is not None and args.lama_model.exists(), f"--lama_model {args.lama_model} missing"
        edited = lama(args.lama_model, src, binary_dilation(U, iterations=args.mask_dilate) if args.mask_dilate else U)
    alpha = np.maximum(np.asarray(hole_img.filter(ImageFilter.GaussianBlur(4)), np.float32),
                       np.asarray(hole_img, np.float32))[..., None] / 255
    comp = alpha * edited.astype(np.float32) + (1 - alpha) * src.astype(np.float32)
    Image.fromarray(edited.astype(np.uint8)).save(d / "inpainted_raw.png")
    Image.fromarray(comp.round().clip(0, 255).astype(np.uint8)).save(d / "inpainted.png")
    job_summary(args.work, f"inpaint: {args.editor} on {int(U.sum())} hole px -> {d / 'inpainted.png'}")
    timer.done(editor=args.editor, hole_px=int(U.sum()))


if __name__ == "__main__":
    main()
