#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion stage 5 (ArtiFixer env): lift the inpainted reference into new Gaussians.

As the official ``inpaint_init`` + ``GaussianModel.inpaint_setup``: every pixel of the reference's
processed unseen mask is unprojected with the AGDD-aligned depth, and becomes one Gaussian with
the inpainted reference colour (SH DC, higher SH bands zero), scale = sqrt(mean squared distance
to its 3 nearest neighbours), random rotation, and near-full opacity (0.99 here; the official
uses 1.0, which is inverse_sigmoid(1) = inf). The removed-scene Gaussians are left untouched and
come first in the checkpoint; ``aurafusion_num_fixed`` records how many there are.

Writes <render_dir>/init/{ckpt_init.pt, rgb/<i>.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from aurafusion.scene import (
    ViewCamera, all_views_overrides, build_dataset, load_model, render_view, save_png, unproject,
)

SH_C0 = 0.28209479177387814


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--reference_index", type=int, default=0)
    parser.add_argument("--initial_opacity", type=float, default=0.99)
    args = parser.parse_args()

    root, r = args.render_dir, args.reference_index
    out = root / "init"
    camera = ViewCamera.from_json(json.loads((root / "cameras.json").read_text())[r])
    mask = torch.from_numpy(np.asarray(Image.open(root / "unseen_dilated" / f"{r:05d}.png")) > 127).cuda()
    depth = torch.from_numpy(np.load(root / "reference" / "depth_aligned.npy")).cuda()
    reference = torch.from_numpy(np.asarray(Image.open(root / "reference" / "reference.png").convert("RGB"))).cuda()
    points = unproject(camera, depth, mask)
    colors = reference[mask].float() / 255.0
    print(f"unprojected {points.shape[0]} unseen pixels of reference view {r}")

    model, conf, checkpoint = load_model(root / "ckpt_removed.pt", all_views_overrides(args.colmap_dir))
    num_fixed = model.num_gaussians
    distances, _ = cKDTree(points.cpu().numpy()).query(points.cpu().numpy(), k=4)
    scale = torch.from_numpy(np.sqrt((distances[:, 1:] ** 2).mean(1)).clip(1e-7)).float().cuda()
    n = points.shape[0]
    new = {
        "positions": points,
        "rotation": torch.rand((n, 4), device="cuda"),
        "scale": model.scale_activation_inv(scale)[:, None].repeat(1, 3),
        "density": model.density_activation_inv(torch.full((n, 1), args.initial_opacity, device="cuda")),
        "features_albedo": (colors - 0.5) / SH_C0,
        "features_specular": torch.zeros((n, model.features_specular.shape[1]), device="cuda"),
    }
    for name, value in new.items():
        old = getattr(model, name)
        setattr(model, name, torch.nn.Parameter(torch.cat([old.detach(), value.to(old.dtype)], dim=0)))
    model.set_optimizable_parameters()
    model.setup_optimizer()
    model.validate_fields()
    model.build_acc()
    out.mkdir(parents=True, exist_ok=True)
    parameters = model.get_model_parameters() | {"global_step": int(checkpoint["global_step"]), "epoch": 0,
                                                 "aurafusion_num_fixed": num_fixed}
    torch.save(parameters, out / "ckpt_init.pt")
    print(f"checkpoint: {num_fixed} fixed + {n} new Gaussians -> {out / 'ckpt_init.pt'}")

    dataset, loader = build_dataset(conf)
    for index, batch in enumerate(loader):
        rendered = render_view(model, dataset.get_gpu_batch_with_intrinsics(batch), index)
        save_png(rendered["rgb"], out / "rgb" / f"{index:05d}.png")
    print(f"Done: rendered {len(dataset)} views -> {out / 'rgb'}")


if __name__ == "__main__":
    main()
