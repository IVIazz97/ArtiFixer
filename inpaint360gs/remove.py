#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inpaint360GS stage 4 (ArtiFixer env): remove the target (and occluding) objects.

As the official ``edit_object_removal.py``: an object is every Gaussian whose distilled class
probability exceeds 0.7, plus every Gaussian inside the convex hull of those points after an
IQR outlier filter (factor 1.0). ``--target_ids`` are removed for good; ``--surrounding_ids``
(occluders next to the target, found with YOLOv8 in the paper and by hand in the official code)
are removed only while the hole is inpainted and are put back by ``finetune``. The object
radius, the 80th percentile distance of the filtered points from their mean (of all selected
objects together when there are several), sets the virtual camera distance.

Target ids come from ``--target_ids`` (read them off associated_numbered/) or, automatically,
from ``--target_masks_dir``: binary reference masks of the object (.npy or .png per image stem,
e.g. datasets/masks_objid/<scene>); every associated id with at least ``--auto_min_inside`` of
its pixels inside the reference masks is selected. ``--flashsplat_dir`` instead removes the
FlashSplat object, for comparisons with the removal the other ports share; the convex hull is
taken there too. ``--hull_expand`` grows that hull about the object's centre.

Writes <output_dir>/removal/{ckpt_removed.pt, ckpt_target_removed.pt (only if surrounding ids),
removal.pt (per-Gaussian masks: removed, surrounding, footprint), removal.json, rgb/<i>.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError

from flashsplat import export
from inpaint360gs.common import (
    all_views_overrides, build_dataset, dataset_views, load_ids, load_model, render_view, save_png,
)


def hull_mask(positions: torch.Tensor, mask: torch.Tensor, outlier_factor: float = 1.0,
              expand: float = 1.0):
    """Official ``points_inside_convex_hull`` (IQR-filtered) and ``get_hull_size``.

    ``expand`` scales the hull about the object's centre; 1.0 is the official hull, which is not
    expanded. A bigger hull also takes what sits just around the object, e.g. the Gaussians of
    its contact shadow. It does not change the returned radius, which is a property of the
    object's own points.

    Returns (Gaussians inside the hull [N] bool, object radius)."""
    points = positions[mask].cpu().numpy().astype(np.float64)
    if len(points) == 0:
        return torch.zeros_like(mask), 0.0
    q1, q3 = np.percentile(points, 25, axis=0), np.percentile(points, 75, axis=0)
    iqr = q3 - q1
    points = points[~np.any((points < q1 - outlier_factor * iqr) | (points > q3 + outlier_factor * iqr), axis=1)]
    if len(points) == 0:
        return torch.zeros_like(mask), 0.0
    radius = float(np.percentile(np.linalg.norm(points - points.mean(0), axis=1), 80))
    try:
        hull = ConvexHull(points)
    except QhullError:
        return torch.zeros_like(mask), radius
    equations = torch.as_tensor(hull.equations, dtype=torch.float64, device=positions.device)
    normals, offsets = equations[:, :3], equations[:, 3]
    if expand != 1.0:  # x is in the scaled hull iff centre + (x - centre) / expand is in the hull
        centre = torch.as_tensor(points.mean(0), dtype=torch.float64, device=positions.device)
        offsets = expand * (normals @ centre + offsets) - normals @ centre
    tol = 1e-9 * max(float(np.ptp(points, axis=0).max()), 1.0)
    inside = torch.zeros_like(mask)
    for s in range(0, positions.shape[0], 1 << 20):
        chunk = positions[s:s + (1 << 20)].double()
        inside[s:s + (1 << 20)] = ((chunk @ normals.T + offsets) <= tol).all(dim=1)
    return inside, radius


def load_reference_mask(directory: Path, stem: str, height: int, width: int) -> np.ndarray | None:
    for suffix in (".npy", ".png"):
        path = directory / f"{stem}{suffix}"
        if path.exists():
            ref = np.load(path) if suffix == ".npy" else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            ref = ref[..., 0] if ref.ndim == 3 else ref
            return cv2.resize((ref > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return None


def auto_target_ids(root: Path, cameras, masks_dir: Path, min_inside: float, min_share: float) -> list[int]:
    """Associated ids lying mostly inside the reference object masks."""
    inside, total, ref_pixels = {}, {}, 0
    for camera in cameras:
        stem = Path(camera.name).stem
        ref = load_reference_mask(masks_dir, stem, camera.height, camera.width)
        if ref is None:
            continue
        ids = load_ids(root / "associated" / f"{stem}.png", camera.height, camera.width)
        ref_pixels += int(ref.sum())
        for label, count in zip(*np.unique(ids, return_counts=True)):
            total[label] = total.get(label, 0) + int(count)
        for label, count in zip(*np.unique(ids[ref], return_counts=True)):
            inside[label] = inside.get(label, 0) + int(count)
    chosen = []
    for label in sorted(inside):
        fraction, share = inside[label] / total[label], inside[label] / max(ref_pixels, 1)
        if label > 0 and fraction >= min_inside and share >= min_share:
            chosen.append(int(label))
            print(f"  id {label}: {100 * fraction:.0f}% inside the reference masks, {100 * share:.1f}% of them")
    if not chosen:
        raise SystemExit(f"no associated id lies >= {min_inside:.0%} inside the masks in {masks_dir}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True, help="Holds associated/ and identity/.")
    parser.add_argument("--target_ids", type=int, nargs="*", default=None)
    parser.add_argument("--target_masks_dir", type=Path, default=None)
    parser.add_argument("--auto_min_inside", type=float, default=0.5)
    parser.add_argument("--auto_min_share", type=float, default=0.005)
    parser.add_argument("--surrounding_ids", type=int, nargs="*", default=[])
    parser.add_argument("--removal_thresh", type=float, default=0.7)
    parser.add_argument("--hull_expand", type=float, default=1.0,
                        help="Scale the objects' convex hull about their centre before taking "
                             "the Gaussians inside it. 1.0 is the official hull.")
    parser.add_argument("--flashsplat_dir", type=Path, default=None,
                        help="Remove the FlashSplat object (labels.pt, contribution.pt, hit_count.pt) instead.")
    parser.add_argument("--flashsplat_labels", type=Path, default=None, help="Defaults to <flashsplat_dir>/labels.pt.")
    args = parser.parse_args()

    root = args.output_dir
    out = root / "removal"
    out.mkdir(parents=True, exist_ok=True)
    model, conf, checkpoint = load_model(args.checkpoint, all_views_overrides(args.colmap_dir))
    dataset, loader = build_dataset(conf)
    batches, cameras = dataset_views(dataset, loader)
    positions = model.positions.detach()
    objects: dict[int, torch.Tensor] = {}
    radii: dict[int, float] = {}

    if args.flashsplat_dir is not None:
        from aurafusion.render_views import removal_keep_masks

        load = lambda name: torch.load(args.flashsplat_dir / name, map_location="cpu")  # noqa: E731
        labels = torch.load(args.flashsplat_labels or args.flashsplat_dir / "labels.pt", map_location="cpu")
        foreground, background = removal_keep_masks(model, labels, load("contribution.pt"), load("hit_count.pt"))
        inside, radii[1] = hull_mask(positions, foreground, expand=args.hull_expand)
        objects[1], target_ids, surrounding_ids = ~background | inside, [1], []
        footprint = foreground | inside  # ~background also holds never-seen Gaussians
        print(f"flashsplat: {int(foreground.sum())} foreground, {int(inside.sum())} inside the "
              f"hull (expand {args.hull_expand}), {int(objects[1].sum())} removed")
        source = "flashsplat"
    else:
        if not args.target_ids and args.target_masks_dir is None:
            raise SystemExit("give --target_ids, --target_masks_dir or --flashsplat_dir")
        target_ids = args.target_ids or auto_target_ids(root, cameras, args.target_masks_dir, args.auto_min_inside,
                                                        args.auto_min_share)
        surrounding_ids = [i for i in args.surrounding_ids if i not in target_ids]
        identity = torch.load(root / "identity" / "identity.pt", map_location="cuda")
        classifier = torch.nn.Conv2d(identity["features"].shape[1], identity["num_classes"], kernel_size=1).cuda()
        classifier.load_state_dict(identity["classifier"])
        selected = target_ids + surrounding_ids
        prob = torch.empty(len(selected), model.num_gaussians, device="cuda")  # softmax over all classes
        with torch.no_grad():
            for s in range(0, model.num_gaussians, 1 << 18):
                chunk = identity["features"][s:s + (1 << 18)].T[:, :, None]
                prob[:, s:s + (1 << 18)] = torch.softmax(classifier(chunk)[..., 0], dim=0)[selected]
        for row, obj_id in enumerate(selected):
            mask = prob[row] > args.removal_thresh
            inside, radii[obj_id] = hull_mask(positions, mask, expand=args.hull_expand)
            objects[obj_id] = mask | inside
            print(f"object {obj_id}: {int(mask.sum())} above {args.removal_thresh}, {int(objects[obj_id].sum())} with hull")
        source = "distilled"

    removed = torch.zeros_like(positions[:, 0], dtype=torch.bool)
    for mask in objects.values():
        removed |= mask
    if source == "distilled":
        footprint = removed
    if not removed.any():
        raise SystemExit(f"nothing to remove: no Gaussian of ids {target_ids} has probability > {args.removal_thresh}")
    surrounding = torch.zeros_like(removed)
    for obj_id in surrounding_ids:
        surrounding |= objects[obj_id]
    if len(objects) > 1:
        _, radius = hull_mask(positions, removed)
    else:
        radius = radii[target_ids[0]]

    step = int(checkpoint["global_step"])
    kept = export.save_segmented_checkpoint(model, ~removed, out / "ckpt_removed.pt", step)
    if surrounding_ids:
        export.save_segmented_checkpoint(model, ~removed | surrounding, out / "ckpt_target_removed.pt", step)
    torch.save({"removed": removed.cpu(), "surrounding": surrounding.cpu(), "footprint": footprint.cpu(),
                "objects": {k: v.cpu() for k, v in objects.items()}}, out / "removal.pt")
    (out / "removal.json").write_text(json.dumps({
        "source": source, "target_ids": target_ids, "surrounding_ids": surrounding_ids,
        "removal_thresh": args.removal_thresh, "hull_expand": args.hull_expand,
        "target_object_radius": radius,
        "num_removed": int(removed.sum()), "num_kept": kept, "num_surrounding": int(surrounding.sum()),
    }, indent=1))
    print(f"removed {int(removed.sum())} of {model.num_gaussians} Gaussians, object radius {radius:.4f}")

    removed_model = export.segment_model(model, ~removed, setup_optimizer=False)
    removed_model.renderer = model.renderer
    for index, batch in enumerate(batches):
        save_png(render_view(removed_model, batch, index)["rgb"], out / "rgb" / f"{index:05d}.png")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
