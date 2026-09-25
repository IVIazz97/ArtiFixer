#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Put an official video-prior method's square outputs back into full-resolution frames for a 3DGS refit.

Both methods work on a square crop at their model resolution (MVInpainter 512², cropped around
the mask's bbox centre by their ``square_crop``; Omni-3DEdit 576², a centred crop by their
``transform_img_and_K``). Neither repo reconstructs 3D itself (their READMEs point to 3DGS /
AnySplat), so for a fair comparison both go through the same step: the output is resized back
to the crop and pasted into the photo **inside the dilated object mask only** (4 px feather),
the rest of the frame stays the photo. Views whose mask falls outside the crop by more than
``--max_outside`` (Omni's centred crop can cut off an off-centre object) cannot be cleaned and are
left out of the refit, and are listed in ``compose.json``.

    <out>/edited/<stem>.png               full-res composite (the method's 2D result)
    <out>/colmap/images/<name>            the same, as JPEG q100 4:4:4 under the COLMAP image name
    <out>/colmap/sparse/0/images.txt      poses of the kept views (graphdeco reads .txt when .bin is absent)
    <out>/colmap/sparse/0/{cameras.bin,points3D.bin,points3D.ply} -> the reconstruction's
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def mvinpainter_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Their ``dataloaders/realworld_dataset.square_crop`` box (x0, y0, x1, y1), same integer arithmetic."""
    h, w = mask.shape
    ys, xs = np.where(mask == 1)
    if h > w:
        c = (ys.min() + ys.max()) // 2
        y0, y1 = c - w // 2, c + w // 2
        if y0 < 0:
            y1 -= y0; y0 = 0
        elif y1 > h:
            y0 -= y1 - h; y1 = h
        return 0, int(y0), w, int(y1)
    c = (xs.min() + xs.max()) // 2
    x0, x1 = c - h // 2, c + h // 2
    if x0 < 0:
        x1 -= x0; x0 = 0
    elif x1 > w:
        x0 -= x1 - w; x1 = w
    return int(x0), 0, int(x1), h


def omni_box(w: int, h: int, size: int = 576) -> tuple[float, float, float, float]:
    """Their ``seva.eval.transform_img_and_K(mode='crop')`` centred crop, mapped back to photo pixels."""
    rfs = size / min(w, h)                        # get_resizing_factor(cover_target=True) for a square target
    rh, rw = math.ceil(rfs * h), math.ceil(rfs * w)
    ct = min(max(0, int(0.5 * rh) - size // 2), rh - size)
    cl = min(max(0, int(0.5 * rw) - size // 2), rw - size)
    return cl / rfs, ct / rfs, (cl + size) / rfs, (ct + size) / rfs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mvinpainter", "omni3dedit"), required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="output/official_inputs/<scene>.")
    parser.add_argument("--raw", type=Path, required=True,
                        help="mvinpainter: the outputs/<run> dir holding g<j>/; omni3dedit: the results/remove/<scene>_* dir.")
    parser.add_argument("--groups", type=Path, help="mvinpainter: groups.json from its prepare.py.")
    parser.add_argument("--colmap_dir", type=Path, required=True, help="$RECON/<scene>/3dgrut_input/<scene>.")
    parser.add_argument("--graphdeco", type=Path, required=True, help="Official gaussian-splatting clone (COLMAP reader).")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--feather_px", type=int, default=4)
    parser.add_argument("--max_outside", type=float, default=0.01)
    args = parser.parse_args()

    sys.path.insert(0, str(args.graphdeco))
    from scene.colmap_loader import read_extrinsics_binary  # noqa: E402

    views = json.loads((args.inputs / "views.json").read_text())
    ref = views["reference"]
    if args.method == "mvinpainter":
        raw = {}
        for g, entry in json.loads(args.groups.read_text()).items():
            for k, stem in enumerate(entry["stems"][:entry["n_real"]], start=1):
                raw[stem] = args.raw / g / f"{k:02d}_{stem}.png"
        raw[ref] = args.inputs / "reference" / f"{ref}.png"  # the reference slot's output is dropped by design
    else:
        raw = {p.stem: p for p in args.raw.glob("*.png") if p.stem != "sampled"}

    extrinsics = read_extrinsics_binary(str(args.colmap_dir / "sparse" / "0" / "images.bin"))
    by_stem = {Path(e.name).stem: e for e in extrinsics.values()}
    (args.output_dir / "edited").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "colmap" / "images").mkdir(parents=True, exist_ok=True)
    sparse = args.output_dir / "colmap" / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)

    kept, dropped, missing = [], {}, []
    for stem in views["stems"]:
        if stem not in raw or not raw[stem].exists():
            missing.append(stem)
            continue
        photo = np.asarray(Image.open(args.inputs / "images" / f"{stem}.png").convert("RGB"), np.float32)
        mask_img = Image.open(args.inputs / "masks" / f"{stem}.png").convert("L")
        mask = np.asarray(mask_img) > 127
        h, w = mask.shape
        out = Image.open(raw[stem]).convert("RGB")
        if args.method == "omni3dedit":
            out = out.crop((out.width // 2, 0, out.width, out.height))   # saved as [source | edited]
        box = (mvinpainter_box(mask.astype(np.uint8)) if args.method == "mvinpainter" else omni_box(w, h))
        if stem == ref and args.method == "mvinpainter":
            box = (0, 0, w, h)                                              # full-res reference itself
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        inside = np.zeros_like(mask)
        inside[y0:y1, x0:x1] = True
        outside = float((mask & ~inside).sum() / max(mask.sum(), 1))
        if outside > args.max_outside:
            dropped[stem] = round(outside, 4)
            continue
        canvas = photo.copy()
        canvas[y0:y1, x0:x1] = np.asarray(out.resize((x1 - x0, y1 - y0), Image.LANCZOS), np.float32)
        alpha = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(args.feather_px)), np.float32) / 255.0
        alpha = np.maximum(alpha, mask.astype(np.float32))[..., None] * inside[..., None]
        comp = Image.fromarray((alpha * canvas + (1 - alpha) * photo).round().clip(0, 255).astype(np.uint8))
        comp.save(args.output_dir / "edited" / f"{stem}.png")
        comp.save(args.output_dir / "colmap" / "images" / by_stem[stem].name, quality=100, subsampling=0)
        kept.append(stem)

    with open(sparse / "images.txt", "w") as f:
        for stem in kept:
            e = by_stem[stem]
            f.write(" ".join(map(str, [e.id, *e.qvec, *e.tvec, e.camera_id, e.name])) + "\n\n")
    for name in ("cameras.bin", "points3D.bin", "points3D.ply"):
        link = sparse / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((args.colmap_dir / "sparse" / "0" / name).resolve())
    (args.output_dir / "compose.json").write_text(json.dumps(
        {"kept": len(kept), "dropped_outside_crop": dropped, "missing": missing}, indent=1))
    print(f"{args.method}: {len(kept)} views composed, {len(dropped)} dropped (object outside crop), "
          f"{len(missing)} missing -> {args.output_dir}")
    assert not missing, f"missing outputs for {missing[:5]}..."


if __name__ == "__main__":
    main()
