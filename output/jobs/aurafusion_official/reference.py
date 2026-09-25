#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the inpainted reference image the official inpaint.py expects, for a scene of our own.

360-USID and Other-360 ship a pre-inpainted ``reference/`` image; a custom scene has none, and the
official code has no step that makes one (their README tells you to inpaint that view yourself).
This builds it exactly as our port does, so both tracks start from the same 2D fill: the photo,
with the object's footprint replaced by the object-removed render, then LaMa inside the unseen
mask. Runs after remove.py + sam2_utils.py, which is when the unseen mask first exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from aurafusion.agdd import dilate, lama_inpaint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True, help="data/Other-360/<scene>.")
    parser.add_argument("--removal_renders", type=Path, required=True,
                        help="output/.../train/ours_30000_object_removal/renders (name-sorted view order).")
    parser.add_argument("--lama_model", type=Path, required=True)
    parser.add_argument("--reference_index", type=int, default=0, help="Index into the name-sorted views.")
    parser.add_argument("--object_dilate", type=int, default=7)
    args = parser.parse_args()

    import torch

    names = sorted(p.name for p in (args.data_dir / "images").iterdir())
    name = names[args.reference_index]
    stem = Path(name).stem
    photo = np.asarray(Image.open(args.data_dir / "images" / name).convert("RGB"))
    render = Image.open(args.removal_renders / f"{args.reference_index:05d}.png").convert("RGB")
    removed = np.asarray(render.resize(photo.shape[1::-1], Image.BICUBIC))

    def load(directory: str) -> np.ndarray:
        path = next(p for p in (args.data_dir / directory).iterdir() if p.stem == stem)
        return np.asarray(Image.open(path).convert("L").resize(photo.shape[1::-1], Image.NEAREST)) > 127

    obj = torch.from_numpy(load("object_masks")).float()[None, None]
    obj = dilate(obj, args.object_dilate, 3)[0, 0].numpy() > 0.5
    unseen = load("unseen_masks")
    composite = np.where(obj[..., None], removed, photo)
    reference = lama_inpaint(args.lama_model, composite, unseen)

    out = args.data_dir / "reference"
    out.mkdir(exist_ok=True)
    for stale in out.iterdir():  # the reader globs this dir; exactly one image must be in it
        stale.unlink()
    Image.fromarray(reference).save(out / f"{stem}.png")
    print(f"reference view {args.reference_index} ({name}): {int(unseen.sum())} unseen px -> {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
