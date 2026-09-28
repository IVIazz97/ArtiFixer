#!/usr/bin/env python3
"""Object id maps from binary object masks, in the inpaint360gs associated/ format.

For every photo in --image_dir, <mask_dir>/<stem>.npy (or .png) becomes <output_dir>/<stem>.png, a
16-bit id map with 0 = background and 1 = object, and <output_dir>/scene.json says 2 classes.
inpaint360gs.distill then learns every Gaussian's object probability from them, and
inpaint360gs.remove --target_ids 1 keeps the ones above the threshold plus their hull: the
official AuraFusion360 "object-masked Gaussians" removal, with our SAM masks as the object masks.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=Path, required=True, help="The COLMAP scene's images/ (one id map per photo).")
parser.add_argument("--mask_dir", type=Path, required=True, help="Binary masks named by photo stem (.npy or .png).")
parser.add_argument("--output_dir", type=Path, required=True)
args = parser.parse_args()

photos = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
args.output_dir.mkdir(parents=True, exist_ok=True)
missing, fractions = [], []
for photo in photos:
    npy, png = args.mask_dir / f"{photo.stem}.npy", args.mask_dir / f"{photo.stem}.png"
    if npy.exists():
        mask = np.load(npy)
    elif png.exists():
        mask = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    else:
        missing.append(photo.stem)
        continue
    mask = (mask[..., 0] if mask.ndim == 3 else mask) > 0
    cv2.imwrite(str(args.output_dir / f"{photo.stem}.png"), mask.astype(np.uint16))
    fractions.append(mask.mean())
assert not missing, f"{len(missing)} of {len(photos)} photos have no mask in {args.mask_dir}, e.g. {missing[:3]}"
(args.output_dir / "scene.json").write_text(json.dumps({"num_classes": 2, "source": str(args.mask_dir)}))
print(f"{len(photos)} id maps -> {args.output_dir}; object covers {100 * np.mean(fractions):.1f}% of the pixels "
      f"on average, {sum(f > 0 for f in fractions)} views show it")
