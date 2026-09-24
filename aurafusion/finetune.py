#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion stage 7 (ArtiFixer env): finetune only the new Gaussians on the SDEdit targets.

As the official ``inpaint_finetune`` (paper Eq. 12): the removed-scene Gaussians are frozen
(their rows get zero gradient); per random view, inside the processed unseen mask
(1 - 0.2) L1 + 0.2 (1 - SSIM) + 0.5 LPIPS (VGG, on 2x2 patches of the mask's bounding box),
outside it 0.8 L1 + 0.2 (1 - SSIM); 10k iterations with 3DGS learning rates.
Deviations: no densification (the official densifies every 300 iterations until 1000) and no
2DGS normal/distortion regularisers, which have no 3DGUT counterpart.

Writes <render_dir>/final/{ckpt_final.pt, rgb/<i>.png}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aurafusion.scene import all_views_overrides, build_dataset, load_model, render_view, save_png


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from fused_ssim import fused_ssim

    return fused_ssim(a.permute(2, 0, 1)[None].contiguous(), b.permute(2, 0, 1)[None].contiguous())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lambda_lpips", type=float, default=0.5)
    args = parser.parse_args()
    root = args.render_dir
    out = root / "final"

    model, conf, checkpoint = load_model(root / "init" / "ckpt_init.pt", all_views_overrides(args.colmap_dir))
    num_fixed = int(checkpoint["aurafusion_num_fixed"])
    print(f"finetuning {model.num_gaussians - num_fixed} new Gaussians, {num_fixed} frozen")
    names = ("positions", "rotation", "scale", "density", "features_albedo", "features_specular")
    lrs = {name: conf.optimizer.params[name].lr for name in names}
    lrs["positions"] *= float(model.scene_extent)
    groups = []
    for name in names:
        param = getattr(model, name)
        param.requires_grad_(True)
        param.register_hook(lambda g: torch.cat([torch.zeros_like(g[:num_fixed]), g[num_fixed:]]))
        groups.append({"params": [param], "lr": lrs[name], "name": name})
    optimizer = torch.optim.Adam(groups, eps=1e-15)
    position_lr_final = 1.6e-6 * float(model.scene_extent)

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda().eval()
    for p in lpips.parameters():
        p.requires_grad_(False)

    dataset, _ = build_dataset(conf, num_workers=0)
    batches, targets, masks = [], [], []
    for index in range(len(dataset)):
        batches.append(dataset.get_gpu_batch_with_intrinsics(torch.utils.data.default_collate([dataset[index]])))
        targets.append(torch.from_numpy(np.asarray(Image.open(root / "sdedit" / f"{index:05d}.png").convert("RGB"))).cuda())
        masks.append(torch.from_numpy(np.asarray(Image.open(root / "unseen_dilated" / f"{index:05d}.png")) > 127).cuda())

    generator = torch.Generator().manual_seed(0)
    order: list[int] = []
    for iteration in range(1, args.iterations + 1):
        if not order:
            order = torch.randperm(len(dataset), generator=generator).tolist()
        index = order.pop()
        t = iteration / args.iterations  # 3DGS log-linear position lr decay
        optimizer.param_groups[0]["lr"] = float(np.exp((1 - t) * np.log(lrs["positions"]) + t * np.log(position_lr_final)))

        image = model(batches[index], train=True, frame_id=index)["pred_rgb"][0]
        gt = targets[index].float() / 255.0
        m = masks[index][..., None].float()
        loss = 0.8 * F.l1_loss(image * (1 - m), gt * (1 - m)) + 0.2 * (1 - ssim(image * (1 - m), gt * (1 - m)))
        if masks[index].any():
            loss = loss + (1 - args.lambda_dssim) * F.l1_loss(image * m, gt * m) \
                + args.lambda_dssim * (1 - ssim(image * m, gt * m))
            ys, xs = torch.nonzero(masks[index], as_tuple=True)
            crop = (slice(ys.min().item(), ys.max().item() + 1), slice(xs.min().item(), xs.max().item() + 1))
            ci, cg = image[crop].permute(2, 0, 1)[None], gt[crop].permute(2, 0, 1)[None]
            ph, pw = ci.shape[2] // 2, ci.shape[3] // 2
            if ph >= 16 and pw >= 16:  # 2x2 patches, as the official divide_into_patches(K=2)
                patches_i = torch.cat([ci[..., a * ph:(a + 1) * ph, b * pw:(b + 1) * pw] for a in range(2) for b in range(2)])
                patches_g = torch.cat([cg[..., a * ph:(a + 1) * ph, b * pw:(b + 1) * pw] for a in range(2) for b in range(2)])
                loss = loss + args.lambda_lpips * lpips(patches_i.clamp(0, 1), patches_g)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration % 1000 == 0 or iteration == 1:
            print(f"iter {iteration}/{args.iterations}: loss {loss.item():.4f}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    model.set_optimizable_parameters()
    model.setup_optimizer()
    torch.save(model.get_model_parameters() | {"global_step": int(checkpoint["global_step"]), "epoch": 0,
                                               "aurafusion_num_fixed": num_fixed}, out / "ckpt_final.pt")
    with torch.no_grad():
        for index, batch in enumerate(batches):
            save_png(render_view(model, batch, index)["rgb"], out / "rgb" / f"{index:05d}.png")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
