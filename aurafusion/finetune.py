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

af360v (virtual views from ``virtual_views``): ``--in_frame_only`` keeps the in-mask terms only for
views whose unseen mask is fully inside the image (``frame_views.in_frame``); a view that cuts the
hole only gets the outside term. With ``--virtual_json`` the virtual views' final renders go to
final/virtual_rgb/<i>.png, so final/rgb holds the real views only, numbered as without them.

Writes <render_dir>/final/{ckpt_final.pt, rgb/<i>.png}.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--in_frame_only", action="store_true",
                        help="In-mask loss only for views whose unseen mask does not touch the image border.")
    parser.add_argument("--frame_margin", type=float, default=0.01, help="frame_views.in_frame margin.")
    parser.add_argument("--virtual_json", type=Path, default=None,
                        help="virtual.json: render the virtual views to final/virtual_rgb instead of final/rgb.")
    parser.add_argument("--summary", type=Path, default=None, help="Append a one-line human-readable summary here.")
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
    missing = [i for i in range(len(dataset)) if not (root / "sdedit" / f"{i:05d}.png").exists()]
    if missing:
        raise SystemExit(f"finetune: {len(missing)} of {len(dataset)} SDEdit targets missing in {root / 'sdedit'} "
                         f"(e.g. {missing[:5]}); rerun sdedit")
    for index in range(len(dataset)):
        batches.append(dataset.get_gpu_batch_with_intrinsics(torch.utils.data.default_collate([dataset[index]])))
        targets.append(torch.from_numpy(np.asarray(Image.open(root / "sdedit" / f"{index:05d}.png").convert("RGB"))).cuda())
        masks.append(torch.from_numpy(np.asarray(Image.open(root / "unseen_dilated" / f"{index:05d}.png")) > 127).cuda())
    supervise_hole = [True] * len(dataset)
    if args.in_frame_only:
        from aurafusion.frame_views import in_frame

        supervise_hole = [in_frame(m.cpu().numpy(), args.frame_margin) for m in masks]
        print(f"in-mask loss on {sum(supervise_hole)}/{len(dataset)} views (unseen mask inside the frame): "
              f"{[i for i, s in enumerate(supervise_hole) if s]}")
        print(f"outside loss only (mask cut by the border or empty): {[i for i, s in enumerate(supervise_hole) if not s]}")
        if not any(supervise_hole):  # e.g. a partial capture: better plain af360 than a failed job
            print("WARNING FALLBACK: no view frames the whole unseen mask; the in-mask loss goes back to every view "
                  "(as af360; check virtual_views / frame_views logs)")
            supervise_hole = [True] * len(dataset)

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
        if masks[index].any() and supervise_hole[index]:
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
        last_loss = loss.item()
        if iteration % 1000 == 0 or iteration == 1:
            print(f"iter {iteration}/{args.iterations}: loss {loss.item():.4f}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    model.set_optimizable_parameters()
    model.setup_optimizer()
    torch.save(model.get_model_parameters() | {"global_step": int(checkpoint["global_step"]), "epoch": 0,
                                               "aurafusion_num_fixed": num_fixed}, out / "ckpt_final.pt")
    num_real = len(batches)
    if args.virtual_json is not None:
        num_real = json.loads(args.virtual_json.read_text())["num_real"]
    with torch.no_grad():
        for index, batch in enumerate(batches):
            folder = "rgb" if index < num_real else "virtual_rgb"
            save_png(render_view(model, batch, index)["rgb"], out / folder / f"{index:05d}.png")
    print(f"final renders: {out / 'rgb'} ({num_real} real views)"
          + (f", {out / 'virtual_rgb'} ({len(batches) - num_real} virtual)" if num_real < len(batches) else ""))
    if args.summary is not None:
        with open(args.summary, "a") as f:
            f.write(f"af360v finetune: {args.iterations} iters, in-hole loss on {sum(supervise_hole)}/{len(batches)} views, "
                    f"last loss {last_loss:.4f}, {model.num_gaussians - num_fixed} new Gaussians -> {out}\n")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
