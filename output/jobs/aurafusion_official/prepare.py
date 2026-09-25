#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lay a mip360 scene out the way the unmodified AuraFusion360 repo expects, and write its configs.

The official scripts resolve ``data/<dataset>/<scene>`` and ``output/<dataset>/<scene>`` relative to
the repo root (``utils/sam2_utils.py``, ``utils/LeftRefill/sdedit_utils.py`` hard-code them), and
``--dataset`` only accepts 360-USID or Other-360 -- so the scene goes under Other-360 and the model
dir is a symlink back into our output tree. Nothing in the official repo is edited.

    data/Other-360/<scene>/images        -> the same COLMAP images our 3DGUT run used
    data/Other-360/<scene>/sparse        -> the same COLMAP sparse model
    data/Other-360/<scene>/object_masks  <- datasets/masks_objid/<scene>/*.npy as PNG
    output/Other-360/<scene>             -> <repo>/output/aurafusion_official/<scene>

The inpainted ``reference/`` image cannot be made yet: it is the reference view with the hole
filled, and the hole is only known after remove.py + sam2_utils.py. ``reference.py`` writes it
between those stages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

AF = Path(__file__).resolve().parents[3]
TEMPLATE_TRAIN = """source_path = "{source}"
model_path = "{model}"

dilate_mask_kernel_size = 10
dilate_mask_iter = 1

optimize_is_masked_iter = 20000
is_masked_lr = 0.1
is_masked_3d_lr = 0.0

other_depth_lr = 0.0
"""
TEMPLATE_REMOVE = """source_path = "{source}"
model_path = "{model}"
iteration = 30000

skip_train = False
skip_test = True
skip_mesh = True
render_path = False

removal_thresh = 0.6
unseen_thresh = 0.0
outlier_factor = 1.0
aggreagte_threshold = 0.6
"""
TEMPLATE_INPAINT = """source_path = "{source}"
model_path = "{model}"
iteration = 30000
skip_train = False
skip_test = False
skip_mesh = True
render_path = False

# unseen mask args
dilate_mask_kernel_size = 5
dilate_mask_iter = 3

# Guided Depth Diffusion args
dilate_iter = 5
kernel_size = 3
optimize_iter = 8
delta = 0.3
infer_iter = 4

# finetuning
densify_until_iter = 1000
"""
TEMPLATE_SDEDIT = """dataset="Other-360"
scene="{scene}"
script=sdedit

use_ddim_inversion=True
strength=0.85
eta=1.0
scale=2.5
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official_repo", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True, help="3dgrut_input/<scene>: images/ + sparse/.")
    parser.add_argument("--mask_dir", type=Path, required=True, help="datasets/masks_objid/<scene> (uint8 .npy).")
    parser.add_argument("--model_dir", type=Path, required=True, help="Where the official run writes its output.")
    args = parser.parse_args()

    data = args.official_repo / "data" / "Other-360" / args.scene
    (data / "object_masks").mkdir(parents=True, exist_ok=True)
    for name, target in (("images", args.colmap_dir / "images"), ("sparse", args.colmap_dir / "sparse")):
        link = data / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())

    size = Image.open(sorted((data / "images").iterdir())[0]).size
    for path in sorted(args.mask_dir.glob("*.npy")):
        mask = Image.fromarray((np.load(path) > 0).astype(np.uint8) * 255)
        if mask.size != size:  # the masks were annotated at the downscaled resolution
            mask = mask.resize(size, Image.NEAREST)
        mask.save(data / "object_masks" / f"{path.stem}.png")
    print(f"{args.scene}: {len(list((data / 'object_masks').glob('*.png')))} object masks at {size} -> {data}")

    model_link = args.official_repo / "output" / "Other-360" / args.scene
    model_link.parent.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if model_link.is_symlink() or model_link.exists():
        model_link.unlink()
    model_link.symlink_to(args.model_dir.resolve())

    config_dir = args.official_repo / "configs" / "Other-360" / args.scene
    config_dir.mkdir(parents=True, exist_ok=True)
    fields = {"source": f"./data/Other-360/{args.scene}", "model": f"./output/Other-360/{args.scene}",
              "scene": args.scene}
    for name, template in (("train", TEMPLATE_TRAIN), ("remove", TEMPLATE_REMOVE),
                           ("inpaint", TEMPLATE_INPAINT), ("sdedit", TEMPLATE_SDEDIT)):
        (config_dir / f"{name}.config").write_text(template.format(**fields))
    print(f"configs -> {config_dir}, model dir -> {args.model_dir}")


if __name__ == "__main__":
    main()
