#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SurfaceLift stage 3 (artifixer env): choose the fill view, then its hole = only what no photo saw
(SurfaceLift scripts/select_view.py + scripts/unseen_view.py --consistency).

Object points P: the full model's depth inside the eroded object masks of up to 40 views.

``--mode real`` (variant surflift): SurfaceLift's real-view scoring over every training view,
    score = coverage^2 x size x (0.4 + 0.6 steepness), hole = the view's object mask M0.
``--mode contain`` (variant surfliftv): the view must hold the WHOLE hole inside the frame, with a
    margin. Candidates:
      * training views whose object mask stays ``margin`` px clear of the border;
      * SurfaceLift's 144 virtual look-at cameras around the object, and every training camera
        re-aimed at the object; each pulled back along its viewing line (x1 ... x3) until every
        point of P projects ``margin`` px inside the frame (first distance that fits);
    rejected when blocked (> ``max_near`` of the frame closer than 0.25 x the object distance: inside
    or right behind geometry) or mostly empty (mean alpha < 0.6). Virtual views score with
    SurfaceLift's context term (ring opacity / (1 + (d / 0.5)^2), d = distance to the nearest
    training camera over the object distance); their hole M0 = P projected, closed, filled, dilated.
    Without any valid candidate it falls back to the best real view (logged).

Hole U in the chosen view (unseen_view.py --consistency, the benchmark rule): with M = M0 dilated
``--beyond`` px and the cleaned scene's render (alpha A, depth D),
    U = (M & A < alpha) | (M0 & (~observed | contradicted | (see_through & ~confirmed)))
opened 2, closed 4, dilated 2, inside M.

-> <work>/select/{camera.json, source.png (photo/render outside M0, cleaned render in M0, grey in U),
   hole.png (U), object_mask.png (M0), removed_{render.png,depth.npy,alpha.npy}, select.json, top.jpg}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion, binary_fill_holes, binary_opening,
                           distance_transform_edt, label)

from surflift.align import harmonic_fill
from surflift.clean import load_cleaned
from surflift.common import Scene, Timer, View, job_summary, look_at, orbit_candidates, save_mask, save_u8, save_view, scene_up, to_u8, unproject_np
from surflift.ledger import Witnesses, check_view


def object_points(sc: Scene, n_views: int = 40, seed: int = 0) -> np.ndarray:
    with_obj = [v for v in sc.views if sc.mask(v.index).any()]
    step = max(1, len(with_obj) // n_views)
    pts = []
    with torch.no_grad():
        for v in with_obj[::step]:
            r = sc.trace(v, sc.full, geometry=True)
            M0 = sc.mask(v.index)
            for er in (14, 6, 0):
                m = (binary_erosion(M0, iterations=er) if er else M0) & (r["alpha"].cpu().numpy() > 0.5)
                if m.sum() > 50:
                    break
            pts.append(unproject_np(v, r["depth"].cpu().numpy(), m))
    P = np.concatenate(pts)
    med = np.median(P, 0)
    mad = np.median(np.linalg.norm(P - med, axis=1))
    P = P[np.linalg.norm(P - med, axis=1) < 4 * mad]
    return P[np.random.default_rng(seed).choice(len(P), min(60000, len(P)), replace=False)]


def projected_hole(v: View, P: np.ndarray, dilate: int = 12) -> np.ndarray:
    uv, z = v.project(P)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < v.W - 0.5) & (uv[:, 1] >= 0) & (uv[:, 1] < v.H - 0.5)
    hole = np.zeros((v.H, v.W), bool)
    q = np.round(uv[ok]).astype(int)
    hole[q[:, 1], q[:, 0]] = True
    return binary_dilation(binary_fill_holes(binary_closing(hole, iterations=6)), iterations=dilate)


def fits(v: View, P: np.ndarray, margin: float) -> bool:
    uv, z = v.project(P)
    return bool((z > 1e-3).all() and (uv[:, 0] >= margin).all() and (uv[:, 0] <= v.W - 1 - margin).all()
                and (uv[:, 1] >= margin).all() and (uv[:, 1] <= v.H - 1 - margin).all())


def pulled_back(template: View, d: np.ndarray, r0: float, center: np.ndarray, up: np.ndarray, Ps: np.ndarray,
                margin: float, name: str, ladder=(1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0)):
    for f in ladder:
        v = look_at(template, center + d * r0 * f, center, up, f"{name}_x{int(round(f * 100)):03d}")
        if fits(v, Ps, margin):
            return v, f
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--colmap_dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--mode", choices=("real", "contain"), default="real")
    ap.add_argument("--margin", type=float, default=0.04, help="contain: border margin, fraction of the short side (>= 40 px).")
    ap.add_argument("--max_near", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--beyond", type=int, default=16)
    ap.add_argument("--see_through", type=float, default=0.10)
    ap.add_argument("--depth_tol", type=float, default=0.03)
    ap.add_argument("--col_tol", type=float, default=0.1)
    ap.add_argument("--view", type=int, default=None, help="Force this training view (index).")
    ap.add_argument("--min_island", type=float, default=0.05,
                    help="Virtual views: drop hole components smaller than this fraction of the largest (and < 200 px).")
    ap.add_argument("--virtual_only", action="store_true", help="contain: test the virtual cameras (reject every real view).")
    args = ap.parse_args()
    w = args.work
    out = w / "select"
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer(w, "select")
    sc = Scene(args.checkpoint, args.colmap_dir, w)
    clean, _ = load_cleaned(sc, w)
    views = sc.views
    up = scene_up(views)
    P = object_points(sc)
    center = np.median(P, 0)
    cam_obj = float(np.median([np.linalg.norm(v.center - center) for v in views]))
    train_centers = np.array([v.center for v in views])
    W0, H0 = views[0].W, views[0].H
    margin = max(round(args.margin * min(W0, H0)), 40)
    Ps = P[np.random.default_rng(1).choice(len(P), min(5000, len(P)), replace=False)]
    print(f"object: {len(P)} points, centre {center.round(3).tolist()}, camera distance {cam_obj:.3f}, margin {margin} px", flush=True)

    cands: list[tuple[View, dict]] = []
    for v in views:
        M0 = sc.mask(v.index)
        if not M0.any():
            continue
        if args.view is not None and v.index != args.view:
            continue
        ys, xs = np.nonzero(M0)
        contained = bool(xs.min() >= margin and xs.max() <= v.W - 1 - margin and ys.min() >= margin and ys.max() <= v.H - 1 - margin)
        cands.append((v, {"kind": "real", "contained": contained}))
    n_real = len(cands)
    if args.mode == "contain" and args.view is None:
        seeds = [(vv, "orbit") for vv in orbit_candidates(views, center, up)]
        seeds += [(look_at(v, v.center, center, up, f"aim_{v.index:05d}"), "reaim") for v in views]
        for v0, kind in seeds:
            d = v0.center - center
            r0 = float(np.linalg.norm(d))
            v, f = pulled_back(v0, d / r0, r0, center, up, Ps, margin, v0.name)
            if v is not None:
                cands.append((v, {"kind": kind, "contained": True, "pull": f}))
    print(f"{len(cands)} candidates ({n_real} real)", flush=True)

    rows = []
    with torch.no_grad():
        for v, meta in cands:
            uv, z = v.project(P)
            inside = (z > 0) & (uv[:, 0] >= 0.02 * v.W) & (uv[:, 0] < 0.98 * v.W) & (uv[:, 1] >= 0.02 * v.H) & (uv[:, 1] < 0.98 * v.H)
            r = sc.trace(v, clean, geometry=True)
            depth, alpha = r["depth"].cpu().numpy(), r["alpha"].cpu().numpy()
            hole = projected_hole(v, P) if v.virtual else sc.mask(v.index)
            q = np.round(uv[inside]).astype(int)
            front = depth[q[:, 1], q[:, 0]]
            occluded = (alpha[q[:, 1], q[:, 0]] > 0.5) & (front < z[inside] * 0.85)
            coverage = inside.mean() * (1 - occluded.mean() if len(q) else 0)
            a = hole.mean()
            size = min(1.0, a / 0.12) * (1.0 if a < 0.4 else float(np.exp(-(a - 0.4) / 0.1)))
            steep = float(np.clip(v.forward @ -up, 0, 1))
            ring = binary_dilation(hole, iterations=30) & ~hole
            dist_obj = float(np.linalg.norm(v.center - center))
            near = float(((alpha > 0.5) & (depth < 0.25 * dist_obj)).mean())
            context = 1.0
            if v.virtual:
                context = float(alpha[ring].mean()) if ring.any() else 0.0
                dmin = np.min(np.linalg.norm(train_centers - v.center, axis=1)) / cam_obj
                context /= 1 + (dmin / 0.5) ** 2
            score = coverage ** 2 * size * (0.4 + 0.6 * steep) * context
            reject = None
            if args.virtual_only and not v.virtual:
                reject = "virtual_only"
            elif args.mode == "contain" and args.view is None and not v.virtual and not meta["contained"]:
                reject = "mask cut by the frame margin"
            elif args.mode == "contain" and v.virtual:
                if near > args.max_near:
                    reject = f"blocked {near:.3f}"
                elif alpha.mean() < 0.6:
                    reject = f"empty {alpha.mean():.2f}"
            rows.append({"name": v.stem, "index": v.index, "virtual": v.virtual, **meta, "score": float(score) if reject is None else 0.0,
                         "raw_score": float(score), "coverage": float(coverage), "hole_area": float(a), "steepness": steep,
                         "context": context, "near": near, "reject": reject})
    order = sorted(range(len(cands)), key=lambda i: -rows[i]["score"])
    best = order[0] if rows and rows[order[0]]["score"] > 0 else None
    fallback = None
    if best is None:                             # nothing holds the whole hole: best real view by the real rule
        fallback = "no valid contained candidate: best real view"
        print("WARNING:", fallback, flush=True)
        reals = [i for i in range(len(cands)) if not cands[i][0].virtual]
        if not reals:
            raise SystemExit("no valid candidate view and no real view with the object (contain mode keeps only real views "
                             "whose mask is inside the margin; rerun with --mode real)")
        best = max(reals, key=lambda i: rows[i]["raw_score"])
    view, meta = cands[best]
    print("chosen:", json.dumps(rows[best]), flush=True)

    # ---- the chosen view's hole (unseen_view.py --consistency)
    photos = lambda u: sc.photo(u.index)                                # noqa: E731
    masks = lambda u: sc.mask(u.index)                                  # noqa: E731
    wit = Witnesses(sc, sc.full, views, masks, photos)
    with torch.no_grad():
        r = sc.trace(view, clean, geometry=True)
    rgb, Dn, An = r["rgb"], r["depth"].cpu().numpy(), r["alpha"].cpu().numpy()
    M0 = projected_hole(view, P) if view.virtual else sc.mask(view.index)
    M = binary_dilation(M0, iterations=args.beyond) if args.beyond else M0
    confirmed, contradicted, observed = check_view(sc, clean, view, M, wit, skip=None if view.virtual else view.index,
                                                   depth_tol=args.depth_tol, col_tol=args.col_tol, alpha_min=0.5, render=r)
    edge = binary_dilation(M, iterations=2) & ~M & (An > 0.9)
    if edge.sum() > 20:
        vals = np.where(edge, Dn, 0.0)
        _, (iy, ix) = distance_transform_edt(~edge, return_indices=True)
        expected = harmonic_fill(vals[iy, ix], M)
        through = M & (Dn > expected * (1 + args.see_through))
    else:
        through = np.zeros_like(M)
    U = (M & (An < args.alpha)) | (M0 & (~observed | contradicted | (through & ~confirmed)))
    U = binary_opening(U, iterations=2)
    U = binary_closing(U, iterations=4) & M
    U = binary_dilation(U, iterations=2) & M
    dropped_islands = 0
    if view.virtual and U.any():
        # a render from a new viewpoint is blurrier than the photos, so the colour test flags scattered
        # islands on well-observed surface; lifting them at the monocular depth made floaters
        lab, n = label(U)
        sizes = np.bincount(lab.ravel())[1:]
        small = np.nonzero(sizes < max(200, args.min_island * sizes.max()))[0] + 1
        dropped_islands = int(np.isin(lab, small).sum())
        U &= ~np.isin(lab, small)
    render = rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    img = render if view.virtual else sc.photo(view.index).astype(np.float32) / 255
    source = np.where(M0[..., None], render, img)
    source[U] = 0.5
    save_view(view, out / "camera.json")
    save_u8((source * 255).round(), out / "source.png")
    save_u8(to_u8(rgb), out / "removed_render.png")
    if not view.virtual:
        save_u8(sc.photo(view.index), out / "photo.png")
    np.save(out / "removed_depth.npy", Dn)
    np.save(out / "removed_alpha.npy", An)
    save_mask(U, out / "hole.png")
    save_mask(M0, out / "object_mask.png")
    ys, xs = np.nonzero(U) if U.any() else (np.array([0]), np.array([0]))
    info = {"mode": args.mode, "view": view.stem, "index": view.index, "virtual": view.virtual, "meta": meta,
            "fallback": fallback, "margin_px": margin, "center": center.tolist(), "up": up.tolist(), "camera_distance": cam_obj,
            "mask_px": int(M.sum()), "hole_px": int(U.sum()), "hole_frac_of_mask": round(float(U.sum() / max(M.sum(), 1)), 3),
            "hole_touches_border": bool(xs.min() == 0 or ys.min() == 0 or xs.max() == view.W - 1 or ys.max() == view.H - 1),
            "confirmed_px": int(confirmed.sum()), "contradicted_px": int(contradicted.sum()),
            "unobserved_px": int((M0 & (An >= args.alpha) & ~observed).sum()), "see_through_px": int(through.sum()), "dropped_island_px": dropped_islands,
            "candidates": len(cands), "real_candidates": n_real, "rejected": sum(1 for x in rows if x["reject"]),
            "chosen": rows[best], "top": [rows[i] for i in order[:40]]}
    (out / "select.json").write_text(json.dumps(info, indent=1))

    # contact sheet: top 6 candidates (cleaned render, hole outlined)
    tiles = []
    with torch.no_grad():
        for i in order[:6]:
            v = cands[i][0]
            im = to_u8(sc.trace(v, clean, geometry=False)["rgb"]).copy()
            hm = projected_hole(v, P) if v.virtual else sc.mask(v.index)
            edge_px = binary_dilation(hm, iterations=3) & ~hm
            im[edge_px] = (255, 255, 0)
            t = Image.fromarray(im).resize((360, round(360 * v.H / v.W)))
            ImageDraw.Draw(t).text((4, 4), f"{rows[i]['name']} s={rows[i]['score']:.3f} cov={rows[i]['coverage']:.2f} "
                                           f"a={rows[i]['hole_area']:.2f}{' REJ' if rows[i]['reject'] else ''}", fill=(255, 255, 0))
            tiles.append(t)
    if tiles:
        sheet = Image.new("RGB", (360 * 3, tiles[0].height * 2))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % 3) * 360, (k // 3) * tiles[0].height))
        sheet.save(out / "top.jpg", quality=88)
    if not U.any():
        raise SystemExit(f"select: the hole of view {view.stem} is empty (every revealed pixel is confirmed by a photo): "
                         "nothing to fill; see select/select.json")
    where = f"virtual {view.name} [{meta['kind']}, pulled x{meta.get('pull')}]" if view.virtual else f"real #{view.index} {view.name}"
    frame = "touches the frame border" if info["hole_touches_border"] else "fully in frame"
    job_summary(w, f"select ({args.mode}): chosen {where}, score {rows[best]['raw_score']:.3f}, hole {int(U.sum())} px "
                   f"({100 * U.mean():.1f} % of the frame), {frame}" + (f", FALLBACK: {fallback}" if fallback else "")
                   + f"; {len(cands)} candidates, {info['rejected']} rejected")
    timer.done(view=view.stem, virtual=view.virtual, hole_px=int(U.sum()), candidates=len(cands))


if __name__ == "__main__":
    main()
