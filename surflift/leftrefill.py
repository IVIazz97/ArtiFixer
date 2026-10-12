#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SurfaceLift stage 8 (AuraFusion360 2D env: aurafusion / .venv_af2d): AuraFusion360's own LeftRefill
SDEdit of our renders inside the ledger hole (SurfaceLift scripts/coin_leftrefill.py, pipeline C).

The official ``utils/LeftRefill/sdedit_utils.py`` is imported unchanged from the AuraFusion360 clone,
as aurafusion/sdedit.py does, with the benchmark settings of pipeline C: strength 0.85, eta 1.0,
CFG scale 2.5, no DDIM inversion.
  source     lift/rgb/<i>.png: the cleaned scene + the v3c fill at training view i
  mask       dilate(H_i | S_i, 4) (holes.npz); views with fewer than --min_px pixels are skipped,
             and so is the chosen view itself (it keeps its own target in the fit)
  reference  select/inpainted.png: the chosen view's inpainting (AF360 uses its LaMa reference)
Port change (default, --no_crop turns it off): LeftRefill resizes every image to 512 x 512. At
1600 x 1200 a small hole would be denoised at a third of its resolution and squashed to 1:1, so each
view is cropped to a square around its mask (side >= 512 px, 15 % padding) first and the result is
pasted back; outside the mask the source is kept exactly (LeftRefill copies it).
-> <work>/leftrefill/{source/, names/, masks/, refined/<i>.png, info.json}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def crop_box(m: np.ndarray, min_side: int = 512, pad_frac: float = 0.15) -> tuple[int, int, int, int]:
    H, W = m.shape
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * pad_frac))
    side = min(max(side, min_side), H, W)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    if max(y1 - y0, x1 - x0) > side:                   # the mask spans more than the short side: its padded bbox
        p = int(pad_frac * max(y1 - y0, x1 - x0))
        return max(y0 - p, 0), min(y1 + p, H), max(x0 - p, 0), min(x1 + p, W)
    ya = int(np.clip(cy - side // 2, 0, H - side))
    xa = int(np.clip(cx - side // 2, 0, W - side))
    return ya, ya + side, xa, xa + side


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--official_repo", type=Path, required=True, help="AuraFusion360_official clone.")
    ap.add_argument("--sd2_inpainting_ckpt", type=Path, required=True, help="512-inpainting-ema.ckpt")
    ap.add_argument("--no_shadow", action="store_true")
    ap.add_argument("--dilate", type=int, default=4)
    ap.add_argument("--min_px", type=int, default=300)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=2.5)
    ap.add_argument("--ddim_inversion", action="store_true")
    ap.add_argument("--no_crop", action="store_true")
    ap.add_argument("--max_views", type=int, default=0, help="Smoke test: only this many views.")
    args = ap.parse_args()
    t0 = time.time()
    w = args.work
    out = w / "leftrefill"
    for d in ("source", "names", "masks", "refined"):
        if (out / d).exists():
            shutil.rmtree(out / d)
        (out / d).mkdir(parents=True)
    z = np.load(w / "holes.npz")
    hinfo = json.loads((w / "holes.json").read_text())
    shape = tuple(hinfo["shape"])
    un = lambda k: np.unpackbits(z[k])[: shape[0] * shape[1]].reshape(shape).astype(bool)  # noqa: E731
    sel = json.loads((w / "select" / "camera.json").read_text())
    ref_index = -1 if sel["virtual"] else int(sel["index"])
    use = []
    for key in hinfo["per_view"]:
        i = int(key)
        m = un(f"H/{i}")
        if not args.no_shadow and f"S/{i}" in z:
            m = m | un(f"S/{i}")
        m = binary_dilation(m, iterations=args.dilate) if args.dilate else m
        if i == ref_index or m.sum() < args.min_px:
            continue
        use.append((i, m))
    if args.max_views:
        use = use[:: max(1, len(use) // args.max_views)][: args.max_views]
    boxes = {}
    for i, m in use:
        src = np.asarray(Image.open(w / "lift" / "rgb" / f"{i:05d}.png").convert("RGB"))
        box = (0, m.shape[0], 0, m.shape[1]) if args.no_crop else crop_box(m)
        y0, y1, x0, x1 = box
        boxes[i] = box
        Image.fromarray(src[y0:y1, x0:x1]).save(out / "source" / f"{i:05d}.png")
        Image.fromarray(src[y0:y1, x0:x1]).save(out / "names" / f"{i:05d}.png")
        Image.fromarray((m[y0:y1, x0:x1] * 255).astype(np.uint8)).save(out / "masks" / f"{i:05d}.png")
    ref_img = out / "reference.png"
    shutil.copy(w / "select" / "inpainted.png", ref_img)
    print(f"{len(use)} views, reference = chosen view {sel['name']} inpainting, crop={not args.no_crop}", flush=True)

    leftrefill = (args.official_repo / "utils" / "LeftRefill").resolve()
    pretrained = leftrefill / "pretrained_models" / "512-inpainting-ema.ckpt"
    if not pretrained.exists():                  # sdedit_utils would otherwise wget it at import
        pretrained.parent.mkdir(parents=True, exist_ok=True)
        pretrained.symlink_to(args.sd2_inpainting_ckpt.resolve())
    t1 = time.time()
    if use:
        cwd = os.getcwd()
        sys.path.insert(0, str(leftrefill))
        os.chdir(leftrefill)                     # LeftRefill resolves its checkpoints relative to its own dir
        try:
            sys.argv = [sys.argv[0]]
            import sdedit_utils                  # noqa: E402  (builds SD2-inpainting + the LeftRefill prompt)
            sdedit_utils.LeftRefill(str(ref_img), str(out / "source"), str(out / "names"), str(out / "masks"),
                                    str(out / "refined"), strength=None if args.strength >= 1 else args.strength,
                                    eta=args.eta, scale=args.scale, use_ddim_inversion=args.ddim_inversion)
        finally:
            os.chdir(cwd)
    t2 = time.time()
    for i, m in use:                             # paste the refined crops back into the full renders
        full = np.asarray(Image.open(w / "lift" / "rgb" / f"{i:05d}.png").convert("RGB")).copy()
        y0, y1, x0, x1 = boxes[i]
        crop = np.asarray(Image.open(out / "refined" / f"{i:05d}.png").convert("RGB"))
        assert crop.shape[:2] == (y1 - y0, x1 - x0), (i, crop.shape, boxes[i])
        full[y0:y1, x0:x1] = np.where(m[y0:y1, x0:x1, None], crop, full[y0:y1, x0:x1])
        Image.fromarray(full).save(out / "refined" / f"{i:05d}.png")
    info = {"views": len(use), "ref_view_skipped": ref_index, "strength": args.strength, "eta": args.eta, "scale": args.scale,
            "ddim_inversion": args.ddim_inversion, "dilate": args.dilate, "shadow": not args.no_shadow,
            "crop": not args.no_crop, "sdedit_seconds": round(t2 - t1, 1), "seconds": round(time.time() - t0, 1),
            "indices": [i for i, _ in use]}
    (out / "info.json").write_text(json.dumps(info, indent=1))
    with open(w / "stages.txt", "a") as f:
        try:
            import torch
            peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        except Exception:                        # noqa: BLE001
            peak = None
        f.write(json.dumps({"stage": "leftrefill", "seconds": info["seconds"], "peak_alloc_gib": peak, "views": len(use)}) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "indices"}), flush=True)
    line = (f"leftrefill: {len(use)} views refined in {info['sdedit_seconds']:.0f} s (crop={not args.no_crop}, shadow={not args.no_shadow})"
            f" -> {out / 'refined'}")
    with open(w.parent / "job_summary.txt", "a") as f:
        f.write(line + "\n")
    print("SUMMARY:", line, flush=True)
    if not use:
        print("WARNING: no view has a ledger hole >= --min_px outside the chosen view; the fit uses only the chosen view")


if __name__ == "__main__":
    main()
