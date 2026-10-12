#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SurfaceLift stage 7 (artifixer env): ledger hole H_v of EVERY training view, the fill's coverage
C_v, and the shadow extension S_v (SurfaceLift scripts/coin_holes.py --shadow).

H_v = the chosen view's hole rule on every view: (M & alpha < 0.9) | (M0 & failed), M = M0 dilated
16 px, failed = revealed pixel of the cleaned scene that no other photo confirms; open 2 / close 4 /
dilate 2 inside M. C_v = coverage of the v3c fill > 0.3. S_v = M0 pixels the photos confirm (so not
in H_v), on the support surface (depth within 5 % of the harmonic depth of the ring), more than 25 %
darker than the harmonic luminance of the ring outside M0: the object's own shadow.
-> <work>/holes.npz (packed bits H/<i>, C/<i>, S/<i>), holes.json, holes_sheet.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_opening, distance_transform_edt

from surflift.align import harmonic_fill
from surflift.clean import load_cleaned
from surflift.common import Scene, Timer, job_summary
from surflift.ledger import Witnesses, check_view
from surflift.lifted import LiftedGaussians, normals_from_depth


def hole(A, M0, failed, alpha, beyond):
    M = binary_dilation(M0, iterations=beyond) if beyond else M0.copy()
    U = (M & (A < alpha)) | (M0 & failed)
    U = binary_opening(U, iterations=2)
    U = binary_closing(U, iterations=4) & M
    return binary_dilation(U, iterations=2) & M


def _harmonic_from_ring(val: np.ndarray, M0: np.ndarray, ring: np.ndarray, factor: int = 4) -> np.ndarray:
    vals = np.where(ring, val, 0.0)
    _, (iy, ix) = distance_transform_edt(~ring, return_indices=True)
    full = vals[iy, ix].astype(np.float64)
    es = harmonic_fill(full[::factor, ::factor].copy(), M0[::factor, ::factor])
    Hh, Ww = M0.shape
    return F.interpolate(torch.tensor(es, dtype=torch.float32)[None, None], size=(Hh, Ww), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def shadow_map(img, M0, confirmed, A, D, thr, surf_tol=0.05):
    ring = binary_dilation(M0, iterations=3) & ~M0 & (A > 0.9)
    if ring.sum() < 20 or not M0.any():
        return np.zeros_like(M0)
    L = np.log(img.astype(np.float32).mean(-1) / 255.0 + 0.02)
    Lh = _harmonic_from_ring(L, M0, ring)
    Dh = _harmonic_from_ring(D, M0, ring)
    on_surface = np.abs(D - Dh) < surf_tol * np.maximum(Dh, 1e-6)
    dark = L < Lh - np.log(1.0 / (1.0 - thr))
    S = M0 & confirmed & dark & on_surface & (A > 0.9)
    S = binary_opening(S, iterations=2)
    return binary_closing(S, iterations=3) & M0


def load_holes(work: Path) -> tuple[dict, dict]:
    z = np.load(work / "holes.npz")
    info = json.loads((work / "holes.json").read_text())
    shape = tuple(info["shape"])
    out = {}
    for key in info["per_view"]:
        i = int(key)
        un = lambda k: np.unpackbits(z[k])[: shape[0] * shape[1]].reshape(shape).astype(bool)  # noqa: E731
        out[i] = {"H": un(f"H/{i}"), "C": un(f"C/{i}"), "S": un(f"S/{i}") if f"S/{i}" in z else None}
    return out, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--colmap_dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--beyond", type=int, default=16)
    ap.add_argument("--col_tol", type=float, default=0.1)
    ap.add_argument("--cov", type=float, default=0.3)
    ap.add_argument("--no_shadow", action="store_true")
    ap.add_argument("--shadow_thr", type=float, default=0.25)
    args = ap.parse_args()
    w = args.work
    timer = Timer(w, "holes")
    sc = Scene(args.checkpoint, args.colmap_dir, w)
    clean, _ = load_cleaned(sc, w)
    views = sc.views
    masks = lambda v: sc.mask(v.index)                               # noqa: E731
    photos = lambda v: sc.photo(v.index)                              # noqa: E731
    wit = Witnesses(sc, sc.full, views, masks, photos)
    lifted = LiftedGaussians.load(w / "lift" / "lifted.pt")
    packed, per, sheet = {}, {}, []
    with torch.no_grad():
        for k, v in enumerate(views):
            M0 = masks(v)
            if not M0.any():
                continue
            r = sc.trace(v, clean, geometry=True)
            A = r["alpha"].cpu().numpy()
            conf, fail, _ = check_view(sc, clean, v, M0, wit, skip=k, col_tol=args.col_tol, render=r)
            H = hole(A, M0, fail, args.alpha, args.beyond)
            C = sc.coverage(v, lifted, clean).cpu().numpy() > args.cov
            rf = sc.render(v, clean, lifted, with_geometry=True)
            HC = H & C
            front = 0.0
            if HC.any():
                n = normals_from_depth(rf["depth"].float(), v.fx, v.fy, v.cx, v.cy).cpu().numpy()
                ys, xs = np.nonzero(HC)
                ray = np.stack([(xs - v.cx) / v.fx, (ys - v.cy) / v.fy, np.ones(len(xs))], 1)
                ray /= np.linalg.norm(ray, axis=1, keepdims=True)
                front = float(np.abs((n[ys, xs] * ray).sum(1)).mean())
            packed[f"H/{v.index}"] = np.packbits(H.ravel())
            packed[f"C/{v.index}"] = np.packbits(C.ravel())
            d = {"M0": int(M0.sum()), "H": int(H.sum()), "HC": int(HC.sum()), "C": int(C.sum()),
                 "confirmed": int(conf.sum()), "failed": int(fail.sum()), "frontal": round(front, 4)}
            S = None
            if not args.no_shadow:
                img = rf["rgb"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                img = np.where(M0[..., None], img, photos(v).astype(np.float32))
                S = shadow_map(img, M0, conf, A, r["depth"].cpu().numpy(), args.shadow_thr) & ~H
                packed[f"S/{v.index}"] = np.packbits(S.ravel())
                d["S"] = int(S.sum())
            per[str(v.index)] = d
            if k % max(1, len(views) // 8) == 0 and len(sheet) < 8:
                rgb = (rf["rgb"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
                rgb[H & ~C] = (0.5 * rgb[H & ~C] + [127, 0, 0]).astype(np.uint8)
                rgb[HC] = (0.5 * rgb[HC] + [0, 127, 0]).astype(np.uint8)
                if S is not None:
                    rgb[S] = (0.5 * rgb[S] + [0, 0, 127]).astype(np.uint8)
                sheet.append(Image.fromarray(rgb).resize((480, round(480 * v.H / v.W))))
            if k % 25 == 0:
                print(k, v.stem, d, flush=True)
    np.savez_compressed(w / "holes.npz", **packed)
    H_all = np.array([p["H"] for p in per.values()])
    info = {"shape": [views[0].H, views[0].W], "views": len(views), "views_with_object": len(per), "params": vars(args) | {
        "checkpoint": str(args.checkpoint), "colmap_dir": str(args.colmap_dir), "work": str(w)},
            "sum_H": int(H_all.sum()), "views_H_gt_500": int((H_all > 500).sum()), "per_view": per}
    (w / "holes.json").write_text(json.dumps(info, indent=1))
    if sheet:
        W2, H2 = sheet[0].size
        canvas = Image.new("RGB", (W2 * 4, H2 * ((len(sheet) + 3) // 4)))
        for i, im in enumerate(sheet):
            canvas.paste(im, ((i % 4) * W2, (i // 4) * H2))
        canvas.save(w / "holes_sheet.jpg", quality=85)
    job_summary(w, f"holes: {len(per)} views with the object, {info['views_H_gt_500']} with a ledger hole > 500 px, "
                   f"mean hole {int(H_all.mean()) if len(H_all) else 0} px, shadow px {sum(p.get('S', 0) for p in per.values())}")
    timer.done(sum_H=info["sum_H"], views_H_gt_500=info["views_H_gt_500"])


if __name__ == "__main__":
    main()
