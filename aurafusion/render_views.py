#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion stage 1 (ArtiFixer env): object removal and per-view renders.

Removes the FlashSplat-labelled object with the same filters as
``data_processing.render_flashsplat_extraction`` (visible-contribution threshold, label-0
background, background Gaussians inside the trimmed object hull dropped) -- the official code's
removal is the same idea: labelled Gaussians plus everything inside their convex hull.

Writes, for every COLMAP view in dataset order:
  removed/rgb/<i>.png      removed-scene RGB
  removed/depth/<i>.npy    removed-scene camera z-depth (float32), the paper's D^incomplete
  removed/opacity/<i>.png  removed-scene opacity
  object/opacity/<i>.png   opacity of the removed Gaussians alone = removal region R_i
  deleted/opacity/<i>.png  opacity of the background Gaussians the hull filter dropped: surface
                           that the removal took away, which the unseen test cannot see as missing
and cameras.json, ckpt_removed.pt (the fixed background for the inpainting stages).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError

from aurafusion.scene import (
    all_views_overrides, build_dataset, load_model, project, render_view, save_cameras, save_png, unproject,
    view_camera,
)
from flashsplat import export


def removal_keep_masks(model, labels, contribution, hit_count, object_id=1, min_total_contribution=0.1,
                       hull_trim_percentile=99.5, hull_expand=0.0):
    """Foreground/background keep masks, identical to render_flashsplat_extraction's
    ``--normalize-by-hit-count --min-total-contribution 0.1 --background label0
    --background-convex-hull --convex-hull-trim-percentile 99.5``.

    ``hull_expand`` pushes every hull plane outwards by that fraction of the object's mean
    radius. FlashSplat's argmax leaves thin, low-contribution parts of the object (a fork, a
    handle) in the background, and those protrude past the trimmed hull, so they survive the
    removal and no later stage can fix them -- they sit outside the unseen mask. A wider hull
    takes them, at the cost of removing some real background near the object, which is the
    part AuraFusion inpaints anyway."""
    device = model.positions.device
    total = contribution.sum(0) / hit_count.float().clamp_min(1.0)
    visible = (total > min_total_contribution).to(device)
    foreground = labels[object_id].to(device) & visible
    background = labels[0].to(device) & visible

    fg_pos = model.positions.detach()[foreground].cpu().numpy()
    if hull_trim_percentile < 100:
        dist = np.linalg.norm(fg_pos - fg_pos.mean(0), axis=1)
        fg_pos = fg_pos[dist <= np.percentile(dist, hull_trim_percentile)]
    if fg_pos.shape[0] > 10_000:
        fg_pos = fg_pos[np.random.default_rng(0).choice(fg_pos.shape[0], 10_000, replace=False)]
    try:
        hull = ConvexHull(fg_pos)
    except QhullError as error:
        raise SystemExit(f"Could not compute object convex hull: {error}") from error
    bg_idx = torch.where(background)[0]
    bg_pos = model.positions.detach()[bg_idx].cpu().numpy()
    tol = 1e-7 * max(float(np.ptp(fg_pos, axis=0).max()), 1.0)
    margin = hull_expand * float(np.linalg.norm(fg_pos - fg_pos.mean(0), axis=1).mean())  # hull planes are unit-normal
    inside = np.zeros(len(bg_idx), dtype=bool)
    for s in range(0, len(bg_idx), 65536):
        pts = bg_pos[s:s + 65536]
        inside[s:s + 65536] = ((pts @ hull.equations[:, :3].T + hull.equations[:, 3]) <= tol + margin).all(axis=1)
    background[bg_idx[torch.from_numpy(inside).to(device)]] = False
    return foreground, background, visible & ~foreground & ~background


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--flashsplat_dir", type=Path, required=True, help="Holds contribution.pt and hit_count.pt.")
    parser.add_argument("--labels", type=Path, default=None, help="Defaults to <flashsplat_dir>/labels.pt.")
    parser.add_argument("--object_id", type=int, default=1)
    parser.add_argument("--hull_expand", type=float, default=0.0,
                        help="Grow the object hull by this fraction of its mean radius before dropping the "
                             "background Gaussians inside it. Catches thin object parts FlashSplat mislabels.")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir
    model, conf, checkpoint = load_model(args.checkpoint, all_views_overrides(args.colmap_dir))
    dataset, loader = build_dataset(conf)
    labels = torch.load(args.labels or args.flashsplat_dir / "labels.pt", map_location="cpu")
    contribution = torch.load(args.flashsplat_dir / "contribution.pt", map_location="cpu")
    hit_count = torch.load(args.flashsplat_dir / "hit_count.pt", map_location="cpu")
    foreground, background, deleted = removal_keep_masks(model, labels, contribution, hit_count, args.object_id,
                                                        hull_expand=args.hull_expand)
    print(f"{model.num_gaussians} Gaussians: object={int(foreground.sum())}, kept background={int(background.sum())}, "
          f"background deleted inside the hull={int(deleted.sum())}")

    object_model = export.segment_model(model, foreground, setup_optimizer=False)
    removed_model = export.segment_model(model, background, setup_optimizer=False)
    deleted_model = export.segment_model(model, deleted, setup_optimizer=False)
    object_model.renderer = removed_model.renderer = deleted_model.renderer = model.renderer
    out.mkdir(parents=True, exist_ok=True)
    export.save_segmented_checkpoint(model, background, out / "ckpt_removed.pt", int(checkpoint["global_step"]))

    cameras = []
    for index, batch in enumerate(loader):
        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
        camera = view_camera(dataset, index, gpu_batch)
        cameras.append(camera)
        removed = render_view(removed_model, gpu_batch, index)
        obj = render_view(object_model, gpu_batch, index)
        gone = render_view(deleted_model, gpu_batch, index)
        save_png(removed["rgb"], out / "removed" / "rgb" / f"{index:05d}.png")
        save_png(removed["opacity"], out / "removed" / "opacity" / f"{index:05d}.png")
        save_png(obj["opacity"], out / "object" / "opacity" / f"{index:05d}.png")
        save_png(gone["opacity"], out / "deleted" / "opacity" / f"{index:05d}.png")
        (out / "removed" / "depth").mkdir(parents=True, exist_ok=True)
        np.save(out / "removed" / "depth" / f"{index:05d}.npy", removed["depth"].cpu().numpy().astype(np.float32))

        if index == 0:  # geometry self-check: unproject then reproject into the same view
            depth = removed["depth"]
            valid = removed["opacity"] > 0.5
            points = unproject(camera, depth, valid)
            u, v, _ = project([camera], points)
            rows, cols = torch.nonzero(valid, as_tuple=True)
            err = torch.maximum((u[0] - cols).abs(), (v[0] - rows).abs()).max().item()
            print(f"reprojection self-check on view 0: max pixel error {err}")
            assert err <= 1, "camera convention mismatch between render rays and pinhole projection"
        if index % 50 == 0:
            print(f"rendered {index + 1}/{len(dataset)}", flush=True)
    save_cameras(cameras, out / "cameras.json")
    print(f"Done: {len(cameras)} views -> {out}")


if __name__ == "__main__":
    main()
