#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inpaint360GS stage 2 (ArtiFixer env): 2D mask association via 3D Gaussians (paper Sec. 3.2).

As the official ``seg/mask_associate.py``. Views are visited in image-name order. For every
mask of a view, the Gaussian centres projecting inside it are split per patch of a 16 x 16 image
grid; within a patch their camera depths are 2-means clustered, and the nearest 30% of the near
cluster are the mask's foreground Gaussians. The first view seeds the key-object database with
all its masks. Every later mask is matched to the database entry with the highest GS-IoU; below
0.1 it becomes a new object. A matched entry absorbs the mask's not-yet-assigned Gaussians.
The official default score, used here, is |I| / (|mask| + |I|); ``--score iou`` gives the
paper's |I| / |U|.

Deviations: the depth 2-means is solved exactly (best split of the sorted depths) instead of
sklearn's KMeans, and patches are indexed consistently in (row, column) order; the official
code reshapes the transposed mask as [H, W] when it looks for patches the mask touches.

Writes <output_dir>/associated/<stem>.png (16-bit, 0 = background), scene.json, and
associated_color/ and associated_numbered/ (photo with each id written at its centroid, the
official ``add_label_num_hqsam`` view used to pick target ids).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from inpaint360gs.common import (
    all_views_overrides, build_dataset, colorize_ids, dataset_views, load_ids, load_model, project, save_ids,
)


def foreground_gaussians(u, v, z, ids, height, width, patches=16, keep_fraction=0.3):
    """Per-mask foreground Gaussians of one view.

    ``u, v, z`` are pixel column/row and camera depth of the in-view Gaussians, ``ids`` the
    [H, W] mask ids. Returns (gaussian positions into u/v/z [K], their mask ids [K]).
    """
    mask_id = ids[v, u]
    keep = mask_id > 0
    index = torch.nonzero(keep, as_tuple=True)[0]
    mask_id, z = mask_id[keep], z[keep]
    patch_h, patch_w = -(-height // patches), -(-width // patches)
    patch = (v[keep] // patch_h) * patches + (u[keep] // patch_w)
    group = mask_id * patches * patches + patch

    order = torch.argsort(z, stable=True)
    order = order[torch.argsort(group[order], stable=True)]  # by group, then by depth
    group, z, index, mask_id = group[order], z[order].double(), index[order], mask_id[order]
    _, gid, count = torch.unique_consecutive(group, return_inverse=True, return_counts=True)
    start = torch.cumsum(count, 0) - count
    local = torch.arange(len(group), device=group.device) - start[gid]  # rank inside the group
    n = count[gid].double()

    # Exact 1D 2-means: split after rank s (left = s nearest) maximising Sl^2/s + Sr^2/(n-s).
    csum = torch.cumsum(z, 0)
    base = (csum - z)[start][gid]  # prefix sum before the group
    total = csum[start + count - 1][gid] - base
    s = (local + 1).double()
    left = csum - base
    objective = left ** 2 / s + (total - left) ** 2 / (n - s).clamp_min(1)
    objective[s >= n] = -1.0  # the split must leave both clusters non-empty
    best = torch.full((len(count),), -2.0, dtype=torch.float64, device=z.device).scatter_reduce(
        0, gid, objective, "amax")
    rank_of_best = torch.where(objective == best[gid], local, torch.full_like(local, 1 << 30))
    split = torch.full((len(count),), 1 << 30, device=z.device).scatter_reduce(0, gid, rank_of_best, "amin") + 1
    # Keep the nearest 30% of the near cluster; a lone Gaussian is kept (official fallback).
    quota = torch.where(count >= 2, (split.double() * keep_fraction).long(), torch.ones_like(count))
    selected = local < quota[gid]
    return index[selected], mask_id[selected]


class KeyObjectDatabase:
    """The official key-object database: one Gaussian index set per object id."""

    def __init__(self, num_gaussians: int, device, score: str = "official", threshold: float = 0.1):
        self.entries: list[torch.Tensor] = []
        self.assigned = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
        self.marker = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
        self.score, self.threshold = score, threshold

    def intersections(self, current: torch.Tensor) -> torch.Tensor:
        if not self.entries:
            return torch.zeros(0, dtype=torch.long, device=current.device)
        members = torch.cat(self.entries)
        owner = torch.repeat_interleave(torch.arange(len(self.entries), device=current.device),
                                        torch.tensor([len(e) for e in self.entries], device=current.device))
        self.marker[current] = True
        hits = torch.bincount(owner[self.marker[members]], minlength=len(self.entries))
        self.marker[current] = False
        return hits

    def match(self, current: torch.Tensor) -> int:
        """Id (0-based) of the entry ``current`` joins, creating a new one below the threshold."""
        inter = self.intersections(current).double()
        if len(inter):
            sizes = torch.tensor([len(e) for e in self.entries], device=current.device, dtype=torch.float64)
            if self.score == "iou":
                score = inter / (sizes + len(current) - inter + 1e-8)
            else:  # official "IOU_heighlight"
                score = inter / (len(current) + inter + 1e-8)
            best = int(torch.argmax(score))
            if score[best] >= self.threshold:
                fresh = current[~self.assigned[current]]
                self.entries[best] = torch.unique(torch.cat([self.entries[best], fresh]))
                self.assigned[current] = True
                return best
        return self.add(current)

    def add(self, current: torch.Tensor) -> int:
        self.entries.append(current)
        self.assigned[current] = True
        return len(self.entries) - 1


def draw_numbers(image_bgr: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Official add_label_num_hqsam: every id written at the centroid of each of its contours."""
    out = image_bgr.copy()
    for label in np.unique(ids):
        if label == 0:
            continue
        contours, _ = cv2.findContours((ids == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                cv2.putText(out, str(int(label)), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True, help="Holds raw_masks/ from raw_masks.py.")
    parser.add_argument("--patches", type=int, default=16)
    parser.add_argument("--score", choices=("official", "iou"), default="official")
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    root = args.output_dir
    model, conf, _ = load_model(args.checkpoint, all_views_overrides(args.colmap_dir))
    dataset, loader = build_dataset(conf)
    _, cameras = dataset_views(dataset, loader)
    positions = model.positions.detach()
    database = KeyObjectDatabase(positions.shape[0], positions.device, args.score, args.threshold)
    out = root / "associated"
    for sub in ("associated", "associated_color", "associated_numbered"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    for step, camera in enumerate(sorted(cameras, key=lambda c: c.name)):
        stem = Path(camera.name).stem
        raw = load_ids(root / "raw_masks" / f"{stem}.png", camera.height, camera.width)
        u, v, z = (t[0] for t in project([camera], positions))
        inside = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height) & (z > 0)
        gaussians = torch.nonzero(inside, as_tuple=True)[0]
        raw_t = torch.from_numpy(raw).to(positions.device)
        chosen, owner = foreground_gaussians(u[inside], v[inside], z[inside], raw_t, camera.height, camera.width,
                                             args.patches)
        chosen = gaussians[chosen]
        num_masks = int(raw.max(initial=0))
        per_mask = [chosen[owner == m] for m in range(1, num_masks + 1)]
        if step == 0:
            labels = [database.add(g) for g in per_mask]
        else:
            labels = [database.match(g) for g in per_mask]
        lut = np.zeros(num_masks + 1, dtype=np.int64)
        lut[1:] = np.asarray(labels, dtype=np.int64) + 1
        associated = lut[raw]
        save_ids(associated, out / f"{stem}.png")
        cv2.imwrite(str(root / "associated_color" / f"{stem}.png"), cv2.cvtColor(colorize_ids(associated), cv2.COLOR_RGB2BGR))
        photo = cv2.resize(cv2.imread(str(Path(dataset.image_paths[camera.index])), cv2.IMREAD_COLOR),
                           (camera.width, camera.height))
        cv2.imwrite(str(root / "associated_numbered" / f"{stem}.jpg"), draw_numbers(photo, associated))
        if step % 25 == 0:
            print(f"{step + 1}/{len(cameras)} {camera.name}: {num_masks} masks, {len(database.entries)} objects",
                  flush=True)

    num_classes = len(database.entries) + 1
    if num_classes > 256:
        print(f"note: {num_classes} classes; the official code caps ids at 255 (uint8 masks)")
    (out / "scene.json").write_text(json.dumps({
        "num_classes": num_classes, "patches": args.patches, "score": args.score, "threshold": args.threshold,
        "gaussians_per_object": [len(e) for e in database.entries],
    }, indent=1))
    print(f"Done: {num_classes} classes (incl. background) over {len(cameras)} views -> {out}")


if __name__ == "__main__":
    main()
