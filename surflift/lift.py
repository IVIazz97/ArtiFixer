#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SurfaceLift stage 6 (artifixer env): lift the inpainted view onto its aligned depth, fit it, and
align its colours (SurfaceLift scripts/lift.py --base ledger_c --mv_real 0.2 --margin 8, then
scripts/blend.py --round 0: the v3c result).

1. Align MoGe's depth to the cleaned scene's rendered depth (surflift.align: Huber scale/shift on a
   ring + harmonic boundary correction).
2. One surface-bound Gaussian per pixel of dilate(U, margin) (surflift.lifted), coloured by the
   inpainting (real view: times the per-channel render/photo gain on a ring, so the fill carries the
   model's exposure).
3. ``iters`` Adam steps: 0.8 L1 + 0.2 D-SSIM inside dilate(U, max(6, margin)) of the chosen view,
   rendered with the frozen cleaned scene; ``mv_real`` of the steps after 200 fit another photo on
   its real pixels (outside its object mask) wherever the fill shows.
4. Harmonic colour alignment (surflift.appearance.align_colour) in the chosen view.
5. Render cleaned scene + fill at every training view (the LeftRefill sources).
-> <work>/lift/{aligned_depth.npy, lifted_fit.pt (after 3), lifted.pt (after 4), chosen_view_fit.png,
   rgb/<i>.png, info.json}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation

from surflift.align import align_depth
from surflift.appearance import align_colour
from surflift.clean import load_cleaned
from surflift.common import Scene, Timer, job_summary, load_u8, load_view, save_u8, subset, to_u8
from surflift.lifted import LiftedGaussians
from surflift.losses import gaussian_window, ssim_map


def ring_gain(target, render, hole, ralpha):
    """lift.py's exposure match: per-channel gain render / photo on a 4..40 px ring of opaque pixels around the hole."""
    ring = binary_dilation(hole, iterations=40) & ~binary_dilation(hole, iterations=4) & (ralpha > 0.95)
    if ring.sum() < 50:
        return np.ones(3, np.float32)
    return np.clip(render[ring].mean(0) / np.maximum(target[ring].mean(0), 1e-3), 0.5, 2.0).astype(np.float32)


def frustum_subset(base, view):
    dev = base.xyz.device
    Xc = base.xyz @ torch.tensor(view.R.T, dtype=torch.float32, device=dev) + torch.tensor(view.t, dtype=torch.float32, device=dev)
    zz = Xc[:, 2].clamp(min=1e-6)
    uu, vv = view.fx * Xc[:, 0] / zz + view.cx, view.fy * Xc[:, 1] / zz + view.cy
    return subset(base, (Xc[:, 2] > 0) & (uu > -0.2 * view.W) & (uu < 1.2 * view.W) & (vv > -0.2 * view.H) & (vv < 1.2 * view.H))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--colmap_dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--max_slide", type=float, default=1.5)
    ap.add_argument("--margin", type=int, default=8)
    ap.add_argument("--mv_real", type=float, default=0.2)
    ap.add_argument("--no_exposure_match", action="store_true")
    ap.add_argument("--no_colour_align", action="store_true")
    args = ap.parse_args()
    w = args.work
    sel = w / "select"
    out = w / "lift"
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer(w, "lift")
    torch.manual_seed(0)
    np.random.seed(0)

    view = load_view(sel / "camera.json")
    target = load_u8(sel / "inpainted.png").astype(np.float32) / 255
    hole = np.asarray(Image.open(sel / "hole.png")) > 127
    assert hole.any(), f"{sel / 'hole.png'} is empty"
    rd, ra = np.load(sel / "removed_depth.npy"), np.load(sel / "removed_alpha.npy")
    info = {"view": view.stem, "virtual": view.virtual, "hole_px": int(hole.sum())}
    if not view.virtual and not args.no_exposure_match:
        render = load_u8(sel / "removed_render.png").astype(np.float32) / 255
        gain = ring_gain(target, render, hole, ra)
        target = np.where(hole[..., None], np.clip(target * gain, 0, 1), target)
        info["exposure_gain"] = gain.round(3).tolist()
    mono = np.load(sel / "mono_depth.npy")
    aligned, ainfo = align_depth(mono, rd, ra, hole)
    info.update(ainfo)
    np.save(out / "aligned_depth.npy", aligned)
    rel = ainfo["ring_abs_err_global"] / max(float(np.median(rd[ra > 0.5])) if (ra > 0.5).any() else 1.0, 1e-6)
    print(f"depth alignment: scale {ainfo['scale']:.3f} shift {ainfo['shift']:.3f}, ring {ainfo['ring_px']} px, median ring error "
          f"{ainfo['ring_abs_err_global']:.4f} ({100 * rel:.1f} % of the median depth)", flush=True)
    if not np.isfinite(aligned).all() or ainfo["scale"] <= 0:
        raise SystemExit(f"lift: depth alignment failed (scale {ainfo['scale']}, non-finite depth?)")

    sc = Scene(args.checkpoint, args.colmap_dir, w)
    base, _ = load_cleaned(sc, w)
    base_fit = frustum_subset(base, view)
    lift_px = binary_dilation(hole, iterations=args.margin) if args.margin else hole
    lifted = LiftedGaussians(aligned, target, lift_px, view, max_slide=args.max_slide)
    print(f"{len(lifted)} lifted Gaussians, base {len(base)} ({len(base_fit)} in the chosen frustum)", flush=True)
    opt = lifted.optimizer()
    region = torch.tensor(binary_dilation(hole, iterations=max(6, args.margin)), device="cuda")
    ys, xs = np.nonzero(region.cpu().numpy())
    y0, y1, x0, x1 = max(ys.min() - 8, 0), ys.max() + 9, max(xs.min() - 8, 0), xs.max() + 9
    tgt = torch.tensor(target, device="cuda").permute(2, 0, 1)
    win = gaussian_window()
    views = sc.views
    photo_cache = {}
    t0, log = time.time(), []
    for it in range(args.iters + 1):
        if args.mv_real > 0 and it > 200 and np.random.rand() < args.mv_real:
            v = views[np.random.randint(len(views))]
            if v.index == view.index and not view.virtual:
                continue
            if v.index not in photo_cache:
                cov = sc.coverage(v, lifted, base)
                M = torch.as_tensor(sc.mask(v.index), device="cuda")
                m = (cov > 0.05) & ~M
                photo_cache[v.index] = None if m.sum() < 20 else (torch.tensor(sc.photo(v.index), device="cuda").permute(2, 0, 1), m)
            if photo_cache[v.index] is None:
                continue
            gt, m = photo_cache[v.index]
            img = sc.render(v, base, lifted, with_geometry=False)["rgb"]
            loss = ((img - gt.float() / 255).abs().mean(0) * m).sum() / m.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            continue
        img = sc.render(view, base_fit, lifted, with_geometry=False)["rgb"]
        a, b, m = img[:, y0:y1, x0:x1], tgt[:, y0:y1, x0:x1], region[y0:y1, x0:x1].float()
        l1 = ((a - b).abs().mean(0) * m).sum() / m.sum()
        ssim = (ssim_map(a[None], b[None], win)[0].mean(0) * m).sum() / m.sum()
        loss = 0.8 * l1 + 0.2 * (1 - ssim)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 100 == 0:
            log.append({"it": it, "l1": round(float(l1), 4), "ssim": round(float(ssim), 4), "s": round(time.time() - t0, 1)})
            print(log[-1], flush=True)
    del photo_cache
    lifted.save(out / "lifted_fit.pt")
    with torch.no_grad():
        uv_shift = lifted.duv.norm(dim=1).cpu().numpy()
        info.update({"n_lifted": len(lifted), "uv_shift_px_median": float(np.median(uv_shift)),
                     "uv_shift_px_p95": float(np.percentile(uv_shift, 95)), "train_log": log, "margin": args.margin,
                     "fit_seconds": round(time.time() - t0, 1)})
        fit = to_u8(sc.render(view, base, lifted, with_geometry=False)["rgb"])
        save_u8(fit, out / "chosen_view_fit_before_align.png")
        if not args.no_colour_align:
            info["colour_align"] = align_colour(lifted, fit.astype(np.float32) / 255, hole, ra)
        lifted.save(out / "lifted.pt")
        save_u8(to_u8(sc.render(view, base, lifted, with_geometry=False)["rgb"]), out / "chosen_view_fit.png")
        for v in views:
            save_u8(to_u8(sc.render(v, base, lifted, with_geometry=False)["rgb"]), out / "rgb" / f"{v.index:05d}.png")
    (out / "info.json").write_text(json.dumps(info, indent=1))
    job_summary(w, f"lift: {len(lifted)} Gaussians on view {view.stem}, ring err {ainfo['ring_abs_err_global']:.4f} "
                   f"({100 * rel:.1f} % of depth), final L1 {log[-1]['l1'] if log else 'n/a'} SSIM {log[-1]['ssim'] if log else 'n/a'}, "
                   f"colour gain median {info.get('colour_align', {}).get('gain_median')} -> {out / 'rgb'}")
    timer.done(n_lifted=len(lifted), view=view.stem)


if __name__ == "__main__":
    main()
