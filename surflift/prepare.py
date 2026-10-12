#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SurfaceLift stage 1 (artifixer env): the removal and the object masks.

Removal: the FlashSplat object of the AuraFusion360 port (``aurafusion.render_views.removal_keep_masks``:
labelled object Gaussians + everything inside their trimmed convex hull, see ``object_gaussians`` for
the one difference), or an ``inpaint360gs.remove`` removal.pt (``--removal_pt``). The "removed
scene" is the full model minus those Gaussians.

Object mask M0 of every view (SurfaceLift scene_data.object_mask): the 2D object mask (``--mask_dir``
<stem>.npy/.png > 0, e.g. the bakerh SAM3 masks_npy or datasets/masks_objid) united with the removed
Gaussians' rendered opacity > 0.05, dilated ``--dilate`` px. Without masks, only the opacity.

-> <work>/removal.pt {background, deleted: bool over the full model}, cameras.json (surflift View
   list, dataset order), masks/<i>.png, prepare.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation

from surflift.common import Scene, Timer, job_summary, save_mask, subset


def load_2d_mask(mask_dir: Path, stem: str, H: int, W: int) -> np.ndarray | None:
    for ext in (".npy", ".png"):
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            m = np.load(p) if ext == ".npy" else cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            m = (m[..., 0] if m.ndim == 3 else m) > 0
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            return m
    return None


def object_gaussians(model, labels, contribution, hit_count, hull_expand=0.0, object_id=1, min_total_contribution=0.1,
                     hull_trim_percentile=99.5) -> torch.Tensor:
    """The FlashSplat object of aurafusion.render_views.removal_keep_masks (labelled object Gaussians with
    mean contribution > 0.1, their trimmed convex hull, same subsampling) but with the hull test applied
    to EVERY Gaussian. removal_keep_masks keeps only visible label-0 Gaussians outside the hull as the
    removed scene, i.e. it also drops every low-contribution Gaussian anywhere in the scene (kitchen:
    1.36M of 2M); here those stay, so the removed scene equals the full model away from the object."""
    from scipy.spatial import ConvexHull
    device = model.positions.device
    total = contribution.sum(0) / hit_count.float().clamp_min(1.0)
    foreground = (labels[object_id].to(device) & (total > min_total_contribution).to(device))
    fg_pos = model.positions.detach()[foreground].cpu().numpy()
    if hull_trim_percentile < 100:
        dist = np.linalg.norm(fg_pos - fg_pos.mean(0), axis=1)
        fg_pos = fg_pos[dist <= np.percentile(dist, hull_trim_percentile)]
    if fg_pos.shape[0] > 10_000:
        fg_pos = fg_pos[np.random.default_rng(0).choice(fg_pos.shape[0], 10_000, replace=False)]
    hull = ConvexHull(fg_pos)
    pos = model.positions.detach().cpu().numpy()
    tol = 1e-7 * max(float(np.ptp(fg_pos, axis=0).max()), 1.0)
    margin = hull_expand * float(np.linalg.norm(fg_pos - fg_pos.mean(0), axis=1).mean())
    inside = np.zeros(len(pos), dtype=bool)
    for s in range(0, len(pos), 65536):
        p = pos[s:s + 65536]
        inside[s:s + 65536] = ((p @ hull.equations[:, :3].T + hull.equations[:, 3]) <= tol + margin).all(axis=1)
    return foreground | torch.from_numpy(inside).to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--colmap_dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--flashsplat_dir", type=Path, default=None, help="labels.pt, contribution.pt, hit_count.pt")
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--removal_pt", type=Path, default=None, help="Instead: an inpaint360gs.remove removal.pt.")
    ap.add_argument("--hull_expand", type=float, default=0.0)
    ap.add_argument("--mask_dir", type=Path, default=None)
    ap.add_argument("--dilate", type=int, default=12)
    ap.add_argument("--opacity_thr", type=float, default=0.05)
    args = ap.parse_args()
    w = args.work
    w.mkdir(parents=True, exist_ok=True)
    timer = Timer(w, "prepare")
    sc = Scene(args.checkpoint, args.colmap_dir, w, photos=False)
    model = sc.model
    if args.removal_pt is not None:
        removal = torch.load(args.removal_pt, map_location="cpu")
        assert removal["removed"].shape[0] == len(sc.full), "removal.pt is for another checkpoint"
        removed = removal["removed"].bool()
        background = ~removed
    else:
        fs = args.flashsplat_dir
        labels = torch.load(args.labels or fs / "labels.pt", map_location="cpu")
        contribution = torch.load(fs / "contribution.pt", map_location="cpu")
        hit_count = torch.load(fs / "hit_count.pt", map_location="cpu")
        background = ~object_gaussians(model, labels, contribution, hit_count, args.hull_expand).cpu()
    background = background.bool().cpu()
    deleted = ~background
    if not deleted.any():
        raise SystemExit("prepare: the removal deletes no Gaussian (FlashSplat labels / removal.pt empty?)")
    torch.save({"background": background, "deleted": deleted}, w / "removal.pt")
    print(f"{len(sc.full)} Gaussians: kept {int(background.sum())}, deleted (object + hull) {int(deleted.sum())}", flush=True)

    obj = subset(sc.full, deleted.to(sc.full.xyz.device))
    stats, missing = [], 0
    with torch.no_grad():
        for v in sc.views:
            a = sc.alpha(v, obj).cpu().numpy()
            m = a > args.opacity_thr
            m2 = load_2d_mask(args.mask_dir, v.stem, v.H, v.W) if args.mask_dir else None
            if m2 is None:
                missing += 1
            else:
                m = m | m2
            m = binary_dilation(m, iterations=args.dilate) if args.dilate else m
            save_mask(m, w / "masks" / f"{v.index:05d}.png")
            stats.append(float(m.mean()))
            if v.index % 50 == 0:
                print(f"mask {v.index}/{len(sc.views)}: {100 * m.mean():.1f} %", flush=True)
    (w / "cameras.json").write_text(json.dumps([v.to_json() for v in sc.views]))
    info = {"views": len(sc.views), "W": sc.views[0].W, "H": sc.views[0].H, "gaussians": len(sc.full),
            "deleted": int(deleted.sum()), "mask_dir": str(args.mask_dir), "views_without_2d_mask": missing,
            "mask_frac_mean": round(float(np.mean(stats)), 4), "views_with_object": int(sum(s > 0 for s in stats))}
    (w / "prepare.json").write_text(json.dumps(info, indent=1))
    if args.mask_dir and missing:
        print(f"WARNING: {missing} views have no mask in {args.mask_dir}; their M0 is the object opacity only")
    if info["views_with_object"] == 0:
        raise SystemExit("prepare: no view has an object mask (masks empty and object opacity 0)")
    job_summary(w, f"prepare: {len(sc.full)} Gaussians, object + hull {int(deleted.sum())} removed; {len(sc.views)} views "
                   f"{info['W']}x{info['H']}, object in {info['views_with_object']}, mean mask {100 * info['mask_frac_mean']:.1f} %, "
                   f"{missing} views without a 2D mask ({args.mask_dir})")
    timer.done(**info)


if __name__ == "__main__":
    main()
