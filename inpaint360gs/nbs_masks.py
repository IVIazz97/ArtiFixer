#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inpaint360GS stage 6 (either env; aurafusion env for ``--mode sam2``): inpainting masks of the
virtual views, the paper's M_i = SAMT(C_i) (Sec. 3.4).

The official pipeline stops here for an interactive Segment-and-Track-Anything session on the
removed-scene virtual renders. This stage replaces the clicks:

* ``footprint`` (default): the removed objects' visible share of the full render, > threshold.
* ``sam2``: SAM2 video tracking over the removed renders, each frame box-prompted with the
  footprint's largest component (as SAM-Track propagates a first-frame selection).

Every mask then goes through the official ``tools/prepare_lama_data.enlarge``: binarise, keep
components of >= 50 px (the largest if none is), max-filter dilation by 10 px.

Writes <output_dir>/nbs/{raw,mask}/<i>.png; ``mask`` is what LaMa and the finetune use.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def enlarge(mask: np.ndarray, expand_pixels: int = 10, min_area: int = 50) -> np.ndarray:
    """Official ``enlarge``: drop components below ``min_area`` (keep the largest if all are),
    then a (2 * expand_pixels + 1)^2 max filter."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.nonzero(areas >= min_area)[0] + 1
    if len(keep) == 0:
        keep = [np.argmax(areas) + 1]
    cleaned = np.isin(labels, keep).astype(np.uint8)
    kernel = np.ones((2 * expand_pixels + 1, 2 * expand_pixels + 1), np.uint8)
    return cv2.dilate(cleaned, kernel) > 0


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    return labels == (np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)


def sam2_track(renders: list[Path], footprints: list[np.ndarray], model_id: str) -> list[np.ndarray]:
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    predictor = SAM2VideoPredictor.from_pretrained(model_id)
    masks = {}
    with tempfile.TemporaryDirectory() as frame_dir, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, path in enumerate(renders):
            Image.open(path).convert("RGB").save(Path(frame_dir) / f"{i:05d}.jpg", quality=95)
        state = predictor.init_state(video_path=frame_dir, offload_video_to_cpu=True)
        predictor.reset_state(state)
        for i, footprint in enumerate(footprints):
            component = largest_component(footprint)
            if component.any():
                ys, xs = np.nonzero(component)
                predictor.add_new_points_or_box(state, frame_idx=i, obj_id=1, box=np.array(
                    [[xs.min(), ys.min()], [xs.max(), ys.max()]], dtype=np.float32))
        for frame_idx, _, logits in predictor.propagate_in_video(state):
            masks[frame_idx] = (logits[0, 0] > 0).cpu().numpy()
    return [masks.get(i, np.zeros_like(f)) for i, f in enumerate(footprints)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True, help="Holds virtual/ from virtual_views.py.")
    parser.add_argument("--mode", choices=("footprint", "sam2"), default="footprint")
    parser.add_argument("--footprint_threshold", type=float, default=0.3)
    parser.add_argument("--expand_pixels", type=int, default=10)
    parser.add_argument("--sam2_model", default="facebook/sam2-hiera-large")
    args = parser.parse_args()

    root = args.output_dir
    renders = sorted((root / "virtual" / "removed" / "rgb").glob("*.png"))
    footprints = [np.asarray(Image.open(root / "virtual" / "footprint" / p.name).convert("L")) / 255.0
                  > args.footprint_threshold for p in renders]
    raw = footprints if args.mode == "footprint" else sam2_track(renders, footprints, args.sam2_model)
    for sub in ("raw", "mask"):
        (root / "nbs" / sub).mkdir(parents=True, exist_ok=True)
    areas = []
    for path, mask in zip(renders, raw):
        final = enlarge(mask, args.expand_pixels)
        Image.fromarray(mask.astype(np.uint8) * 255).save(root / "nbs" / "raw" / path.name)
        Image.fromarray(final.astype(np.uint8) * 255).save(root / "nbs" / "mask" / path.name)
        areas.append(final.mean())
    print(f"Done: {len(renders)} masks ({args.mode}), area mean={np.mean(areas):.3f} max={np.max(areas):.3f}, "
          f"empty={sum(a == 0 for a in areas)}")


if __name__ == "__main__":
    main()
