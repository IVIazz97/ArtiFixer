#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inpaint360GS stage 3 (ArtiFixer env): object-ID distillation into the Gaussians (Sec. 3.3).

As the official ``seg/distillation.py`` with ``config/object_distill/train_distill.json``: the
reconstruction is frozen; every Gaussian gets a 16-dim identity feature (initialised as the
official RGB2SH(rand)), rendered by alpha blending (Eq. 2) and mapped per pixel to object
logits by a 1x1 convolution. Loss: cross-entropy against the associated masks divided by
log(num_classes) (Eq. 3), plus every 50 iterations 0.0005 x (1 - mean cosine similarity)
between 1000 random Gaussians' features and those of their 5 nearest neighbours among at most
200k Gaussians (Eq. 4; the official code computes but drops the KL term, as here).
2000 iterations, Adam lr 0.0025 on features and 5e-4 on the classifier.

Writes <output_dir>/identity/{identity.pt, pred/<stem>.png, pred_color/<stem>.png}.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from inpaint360gs.common import (
    SH_C0, all_views_overrides, build_dataset, colorize_ids, dataset_views, load_ids, load_model, render_features,
    save_ids,
)


def neighbour_cosine_loss(xyz, features, k=5, max_points=200_000, sample_size=1000):
    """Official ``loss_cls_3d_cosin`` similarity term (1 - mean cosine to the k nearest neighbours)."""
    if xyz.shape[0] > max_points:
        keep = torch.randperm(xyz.shape[0], device=xyz.device)[:max_points]
        xyz, features = xyz[keep], features[keep]
    sample = torch.randperm(xyz.shape[0], device=xyz.device)[:sample_size]
    neighbours = torch.cdist(xyz[sample], xyz).topk(k, largest=False).indices
    cosine = F.cosine_similarity(features[sample][:, None].expand(-1, k, -1), features[neighbours], dim=-1)
    return 1 - cosine.mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True, help="Holds associated/ from associate.py.")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--num_features", type=int, default=16)
    parser.add_argument("--feature_lr", type=float, default=0.0025)
    parser.add_argument("--classifier_lr", type=float, default=5e-4)
    parser.add_argument("--reg3d_interval", type=int, default=50)
    parser.add_argument("--reg3d_k", type=int, default=5)
    parser.add_argument("--reg3d_max_points", type=int, default=200_000)
    parser.add_argument("--reg3d_sample_size", type=int, default=1000)
    parser.add_argument("--sim_weight", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    root = args.output_dir
    out = root / "identity"
    num_classes = json.loads((root / "associated" / "scene.json").read_text())["num_classes"]
    model, conf, _ = load_model(args.checkpoint, all_views_overrides(args.colmap_dir))
    for name in ("positions", "rotation", "scale", "density", "features_albedo", "features_specular"):
        getattr(model, name).requires_grad_(False)
    dataset, loader = build_dataset(conf, num_workers=0)
    batches, cameras = dataset_views(dataset, loader)
    targets = [torch.from_numpy(load_ids(root / "associated" / f"{Path(c.name).stem}.png", c.height, c.width)).cuda()
               for c in cameras]
    print(f"distilling {num_classes} classes into {model.num_gaussians} Gaussians over {len(cameras)} views")

    n = model.num_gaussians
    features = torch.nn.Parameter((torch.rand(n, args.num_features, device="cuda") - 0.5) / SH_C0)
    classifier = torch.nn.Conv2d(args.num_features, num_classes, kernel_size=1).cuda()
    feature_optimizer = torch.optim.Adam([features], lr=args.feature_lr, eps=1e-15)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr)
    xyz = model.positions.detach()
    log_classes = math.log(num_classes)

    order: list[int] = []
    for iteration in range(1, args.iterations + 1):
        if not order:
            order = torch.randperm(len(batches)).tolist()
        index = order.pop()
        rendered = render_features(model, batches[index], features, frame_id=index)
        logits = classifier(rendered.permute(2, 0, 1)[None])
        loss_2d = F.cross_entropy(logits, targets[index][None]) / log_classes
        loss = loss_2d
        if iteration % args.reg3d_interval == 0:
            loss = loss + args.sim_weight * neighbour_cosine_loss(
                xyz, features, args.reg3d_k, args.reg3d_max_points, args.reg3d_sample_size)
        feature_optimizer.zero_grad(set_to_none=True)
        classifier_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        feature_optimizer.step()
        classifier_optimizer.step()
        if iteration % 200 == 0 or iteration == 1:
            print(f"iter {iteration}/{args.iterations}: loss_2d {loss_2d.item():.4f}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features.detach(), "classifier": classifier.state_dict(), "num_classes": num_classes,
                "num_gaussians": n}, out / "identity.pt")
    accuracy = []
    with torch.no_grad():
        for index, (batch, camera) in enumerate(zip(batches, cameras)):
            logits = classifier(render_features(model, batch, features, frame_id=index).permute(2, 0, 1)[None])
            pred = logits[0].argmax(0)
            accuracy.append((pred == targets[index]).float().mean().item())
            pred = pred.cpu().numpy()
            stem = Path(camera.name).stem
            save_ids(pred, out / "pred" / f"{stem}.png")
            (out / "pred_color").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "pred_color" / f"{stem}.png"), cv2.cvtColor(colorize_ids(pred), cv2.COLOR_RGB2BGR))
    print(f"Done: pixel accuracy vs associated masks {np.mean(accuracy):.3f} -> {out}")


if __name__ == "__main__":
    main()
