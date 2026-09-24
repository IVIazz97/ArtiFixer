#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inpaint360GS stage 7 (either env): LaMa colour and depth inpainting of the virtual views (Sec. 3.5).

As the official ``LaMa/bin/predict_color.py`` and ``predict_depth.py`` with ``refine: True``:
big-lama plus LaMa's multi-scale feature refinement (``saicinpainting/evaluation/refinement.py``:
image pyramid down to ~512 px, at every finer scale 15 Adam steps, lr 0.002, on the encoder
features so the prediction keeps the known pixels and matches the coarser result inside the
eroded hole). Depth is the removed-scene virtual depth, min-max normalised with the full-scene
depth of the same view, inpainted as a grey image and mapped back.

``--recursive_guide`` enables the paper's recursive conditional inpainting (Eq. 5-7): each frame
is inpainted side by side with the previous inpainted frame (unmasked), so the refinement's
known-pixel loss conditions it on that frame. The official script ships it off by default.

Uses the TorchScript big-lama (checkpoints/lama/big-lama.pt) split at its first FFC ResNet block
into encoder and decoder, which is the split the official refiner optimises across.

Writes <output_dir>/lama/{rgb/<i>.png, depth/<i>.npy, depth_vis/<i>.png}.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class Lama:
    """big-lama TorchScript split into the front (encoder) and rear the refiner optimises through."""

    def __init__(self, path: Path, device: str = "cuda"):
        model = torch.jit.load(str(path), map_location=device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.layers = [c for _, c in model.model.generator.model.named_children()]
        assert len(self.layers) == 36, "unexpected big-lama layout"

    def front(self, x: torch.Tensor):
        for layer in self.layers[:5]:
            x = layer(x)
        return x

    def rear(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        k = self.layers
        for i in range(5, 23):
            z1, z2 = k[i](z1, z2)
        y = k[24](k[23](z1, z2))
        y = torch.relu(k[25](y))  # the traced model reuses one in-place ReLU (layer 26)
        y = torch.relu(k[28](k[27](y)))
        y = torch.relu(k[31](k[30](y)))
        return k[35](k[34](k[33](y)))


def gaussian_blur(x: torch.Tensor) -> torch.Tensor:
    """kornia gaussian_blur2d(kernel (5, 5), sigma (1, 1)), reflect border."""
    t = torch.arange(5, dtype=x.dtype, device=x.device) - 2
    g = torch.exp(-t ** 2 / 2)
    g = g / g.sum()
    c = x.shape[1]
    x = F.pad(x, (2, 2, 2, 2), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, 5).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, g.view(1, 1, 5, 1).repeat(c, 1, 1, 1), groups=c)


def pyrdown(x: torch.Tensor, size=None) -> torch.Tensor:
    size = size or (x.shape[2] // 2, x.shape[3] // 2)
    return F.interpolate(gaussian_blur(x), size=size, mode="bilinear", align_corners=False)


def pyrdown_mask(mask: torch.Tensor, blur: bool = True, round_up: bool = True, eps: float = 1e-8) -> torch.Tensor:
    size = (mask.shape[2] // 2, mask.shape[3] // 2)
    mask = F.interpolate(gaussian_blur(mask) if blur else mask, size=size, mode="bilinear", align_corners=False)
    return (mask >= eps).float() if round_up else (mask >= 1.0 - eps).float()


def erode(mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """kornia erosion with a flat structuring element and geodesic border (outside is ignored)."""
    hits = F.conv2d(1.0 - mask, kernel[None, None], padding=kernel.shape[0] // 2)
    return mask * (hits < 0.5).float()


def pad_to_modulo(x: torch.Tensor, modulo: int = 8) -> torch.Tensor:
    h, w = x.shape[2:]
    return F.pad(x, (0, (-w) % modulo, 0, (-h) % modulo), mode="reflect")


def refine_predict(lama: Lama, image: torch.Tensor, mask: torch.Tensor, n_iters: int = 15, lr: float = 0.002,
                   min_side: int = 512, max_scales: int = 3, px_budget: int = 1_800_000) -> torch.Tensor:
    """Official ``refine_predict`` for one [1, 3, H, W] image and [1, 1, H, W] mask (1 = hole)."""
    h, w = image.shape[2:]
    if h * w > px_budget:
        ratio = math.sqrt(px_budget / float(h * w))
        h, w = int(h * ratio), int(w * ratio)
        image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        mask = (mask > 1e-8).float()
    n_scales = min(1 + int(round(max(0, math.log2(min(h, w) / min_side)))), max_scales)
    images, masks = [image], [mask]
    for _ in range(n_scales - 1):
        images.append(pyrdown(images[-1]))
        masks.append(pyrdown_mask(masks[-1]))
    kernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)).astype(np.float32)).to(image.device)

    inpainted = None
    for image, mask in zip(images[::-1], masks[::-1]):
        oh, ow = image.shape[2:]
        image, mask = pad_to_modulo(image), (pad_to_modulo(mask) >= 1e-8).float()
        with torch.no_grad():
            z1, z2 = lama.front(torch.cat([image * (1 - mask), mask], dim=1))
        z1, z2 = z1.detach().requires_grad_(True), z2.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z1, z2], lr=lr)
        mask3 = mask.repeat(1, 3, 1, 1)
        for step in range(n_iters):
            optimizer.zero_grad()
            pred = lama.rear(z1, z2)
            if inpainted is None:
                break
            pred_down = pyrdown(pred[:, :, :oh, :ow])
            mask_down = erode(pyrdown_mask(mask[:, :, :oh, :ow], blur=False, round_up=False), kernel).repeat(1, 3, 1, 1)
            known, hole = mask3 < 1e-8, mask_down >= 1e-8
            loss = (pred[known] - image[known]).abs().mean()
            if hole.any():
                loss = loss + (pred_down[hole] - inpainted[hole]).abs().mean()
            if step < n_iters - 1:
                loss.backward()
                optimizer.step()
        inpainted = (mask3 * pred + (1 - mask3) * image).detach()[:, :, :oh, :ow]
    return inpainted


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float().cuda()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True, help="Holds virtual/ and nbs/.")
    parser.add_argument("--lama_model", type=Path, required=True, help="TorchScript big-lama.pt.")
    parser.add_argument("--recursive_guide", action="store_true")
    parser.add_argument("--n_iters", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()

    root = args.output_dir
    out = root / "lama"
    for sub in ("rgb", "depth", "depth_vis"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    lama = Lama(args.lama_model)
    refine = dict(n_iters=args.n_iters, lr=args.lr)
    previous = None
    names = sorted(p.name for p in (root / "nbs" / "mask").glob("*.png"))
    for name in names:
        stem = Path(name).stem
        hole = np.asarray(Image.open(root / "nbs" / "mask" / name).convert("L")) > 127
        rgb = np.asarray(Image.open(root / "virtual" / "removed" / "rgb" / name).convert("RGB")) / 255.0
        mask = torch.from_numpy(hole)[None, None].float().cuda()
        image = to_tensor(rgb)
        if args.recursive_guide and previous is not None:
            both = refine_predict(lama, torch.cat([previous, image], dim=3), torch.cat([torch.zeros_like(mask), mask], dim=3),
                                  **refine)
            color = both[:, :, :, both.shape[3] // 2:]
            color = F.interpolate(color, size=image.shape[2:], mode="bicubic", align_corners=False).clamp(0, 1) \
                if color.shape[2:] != image.shape[2:] else color
            color = mask * color + (1 - mask) * image
        else:
            color = refine_predict(lama, image, mask, **refine)
        previous = color
        color_np = (color[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(color_np).save(out / "rgb" / name)

        depth = np.load(root / "virtual" / "removed" / "depth" / f"{stem}.npy")
        reference = np.load(root / "virtual" / "full" / "depth" / f"{stem}.npy")
        lo, hi = float(reference.min()), float(reference.max())
        normalised = (depth - lo) / max(hi - lo, 1e-8)
        completed = refine_predict(lama, to_tensor(np.repeat(normalised[..., None], 3, axis=2)), mask, **refine)
        completed = completed[0, 0].cpu().numpy()
        np.save(out / "depth" / f"{stem}.npy", (completed * (hi - lo) + lo).astype(np.float32))
        vis = cv2.applyColorMap((np.clip(completed, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(str(out / "depth_vis" / name), vis)
        print(f"{name}: hole {hole.mean():.3f}", flush=True)
    print(f"Done: {len(names)} views -> {out}")


if __name__ == "__main__":
    main()
