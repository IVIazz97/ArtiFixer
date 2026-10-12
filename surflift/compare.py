#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Panel + fidelity check of SurfaceLift runs against another method's final renders (artifixer env).

Panel: one row per variant with its chosen view (input with the hole in grey | inpainting | the v3c
fit), then one row per sampled training view: photo | each ``--other`` method | each variant's v3c
renders | each variant's final renders.
Fidelity (SurfaceLift scripts/fidelity.py): PSNR against the photos over the pixels outside each
view's object mask ("outside") and on the 40 px ring just outside it ("ring"), every ``--stride``-th
view; the full model (object present) is the ceiling.
    python -m surflift.compare --checkpoint ... --colmap_dir ... --runs output/surflift/kitchen/{surflift,surfliftv} \
        --other af360=output/aurafusion/kitchen/lama_ref/final/rgb --out output/panels/surflift_kitchen.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from surflift.common import Scene, load_u8, to_u8


def psnr(a, b, m):
    mse = ((a[m] - b[m]) ** 2).mean()
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def tile(im: np.ndarray, w: int, label: str) -> Image.Image:
    t = Image.fromarray(im).resize((w, round(w * im.shape[0] / im.shape[1])))
    d = ImageDraw.Draw(t)
    d.rectangle((0, 0, 8 + 6 * len(label), 14), fill=(0, 0, 0))
    d.text((4, 2), label, fill=(255, 255, 0))
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--colmap_dir", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="Variant dirs holding sl/ and final/.")
    ap.add_argument("--other", nargs="*", default=[], help="name=dir of <i>.png renders")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_views", type=int, default=6)
    ap.add_argument("--views", type=int, nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--width", type=int, default=360)
    args = ap.parse_args()
    work = args.runs[0] / "sl"
    sc = Scene(args.checkpoint, args.colmap_dir, work)
    views = sc.views
    others = dict(o.split("=", 1) for o in args.other)
    cols = {"full model": None} | {k: Path(v) for k, v in others.items()}
    for r in args.runs:
        cols[f"{r.name} v3c"] = r / "sl" / "lift" / "rgb"
        cols[f"{r.name}"] = r / "final" / "rgb"

    # fidelity
    acc = {k: {"outside": [], "ring": []} for k in cols}
    with torch.no_grad():
        for v in views[::args.stride]:
            gt = sc.photo(v.index).astype(np.float32) / 255
            M = sc.mask(v.index)
            outm, ring = ~M, binary_dilation(M, iterations=40) & ~M
            for k, d in cols.items():
                if d is None:
                    im = to_u8(sc.trace(v, sc.full, geometry=False)["rgb"])
                else:
                    p = d / f"{v.index:05d}.png"
                    if not p.exists():
                        continue
                    im = load_u8(p)
                im = im.astype(np.float32) / 255
                acc[k]["outside"].append(psnr(im, gt, outm))
                if ring.any():
                    acc[k]["ring"].append(psnr(im, gt, ring))
    res = {k: {"views": len(a["outside"]), "outside_psnr": round(float(np.mean(a["outside"])), 2) if a["outside"] else None,
               "ring_psnr": round(float(np.mean(a["ring"])), 2) if a["ring"] else None} for k, a in acc.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    wd = max(len(k) for k in res)
    print(f"{'method':<{wd}}  views  outside  ring40")
    for k, r in res.items():
        if r["views"]:
            print(f"{k:<{wd}}  {r['views']:5d}  {r['outside_psnr']:7.2f}  {r['ring_psnr']}")

    # panel
    w = args.width
    rows = []
    for r in args.runs:
        s = r / "sl" / "select"
        info = json.loads((s / "select.json").read_text())
        ims = [(f"{r.name}: chosen {info['view']}{' (virtual)' if info['virtual'] else ''} input", load_u8(s / "source.png")),
               ("inpainted", load_u8(s / "inpainted.png")), ("v3c fit", load_u8(r / "sl" / "lift" / "chosen_view_fit.png"))]
        if (r / "sl" / "fit" / "sheet.jpg").exists():
            pass
        rows.append([tile(im, w, lab) for lab, im in ims])
    if args.views:
        pick = args.views
    else:
        area = np.array([sc.mask(v.index).mean() for v in views])
        cand = [v.index for v in views if area[v.index] > np.percentile(area[area > 0], 30)] if (area > 0).any() else [v.index for v in views]
        pick = [cand[int(j * len(cand) / args.n_views)] for j in range(args.n_views)]
    for i in pick:
        ims = [(f"photo {i}", sc.photo(i))]
        for k, d in cols.items():
            if d is None:
                continue
            p = d / f"{i:05d}.png"
            if p.exists():
                ims.append((k, load_u8(p)))
        rows.append([tile(im, w, lab) for lab, im in ims])
    ncol = max(len(rw) for rw in rows)
    hs = [max(t.height for t in rw) for rw in rows]
    canvas = Image.new("RGB", (w * ncol, sum(hs)), "white")
    y = 0
    for rw, h in zip(rows, hs):
        for k, t in enumerate(rw):
            canvas.paste(t, (k * w, y))
        y += h
    canvas.save(args.out, quality=88)
    print(args.out)


if __name__ == "__main__":
    main()
