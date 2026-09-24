#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inpaint360GS stage 9 (ArtiFixer env): 3D hybrid-supervision inpainting (Sec. 3.5, Eq. 9).

As the official ``edit_object_inpaint.py`` with ``config/object_inpaint/common.json``, on the 30
virtual views with the LaMa images as targets, for 5000 iterations (a random view each):

    loss = 0.2 * L1 outside the hole + 0.8 * (1 - SSIM of the whole image)
           + 0.0005 * LPIPS(VGG) over the 2 x 2 patches of the hole's bounding box

(the paper writes the L1 inside the mask; the code, followed here, puts it outside). All
Gaussians receive gradients, as in the official code where concatenating frozen and trainable
tensors into one ``nn.Parameter`` makes all of them trainable, with the 3DGS learning rates.
Between iterations 500 and 5000, every 100, the new Gaussians alone are cloned, split and
pruned (opacity < 0.005, larger than 20 px or 0.1 x scene extent). After training, remaining-scene
Gaussians are reset to their original values except those projecting into virtual view 0's
hole mask (grown by 10% of its area) near the seed point cloud (within 3 x its mean std).
Finally the surrounding objects are put back.

3DGUT gives no screen-space gradients or radii, so densification uses 3DGRUT's own proxy
(|world-space position gradient| x camera distance / 2) averaged over the views that have the
Gaussian in frustum (3DGS's radii > 0), and 3 sigma x f / z as the radius.

Deviation: positions. The official position learning rate is 0 (``spatial_lr_scale`` is 0 for
a loaded scene), so clones stay on top of their source, keep its gradient and are cloned again
every round; on 3DGUT that compounds past 5M Gaussians. Here the new Gaussians move with the
3DGS rate (1.6e-4 x scene extent) while the scene's stay pinned as upstream, and densification
stops adding Gaussians beyond ``--max_new_gaussians``.

Writes <output_dir>/final/{ckpt_final.pt, rgb/<i>.png (dataset views), virtual_rgb/<i>.png}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from inpaint360gs.common import (
    all_views_overrides, build_dataset, dataset_views, load_cameras, load_model, load_rgb, project, render_view,
    save_png, template_batch, unproject, virtual_batch,
)

NAMES = ("positions", "rotation", "scale", "density", "features_albedo", "features_specular")


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from fused_ssim import fused_ssim

    return fused_ssim(a.permute(2, 0, 1)[None].contiguous(), b.permute(2, 0, 1)[None].contiguous())


class Optimizer:
    """Adam over the model fields, one group per field, with 3DGS-style row surgery."""

    def __init__(self, model, lrs: dict[str, float]):
        self.model = model
        groups = []
        for name in NAMES:
            param = torch.nn.Parameter(getattr(model, name).detach().clone(), requires_grad=True)
            setattr(model, name, param)
            groups.append({"params": [param], "lr": lrs[name], "name": name})
        self.adam = torch.optim.Adam(groups, eps=1e-15)

    def edit(self, fn) -> None:
        """Replace every field p by fn(name, p, False) and its Adam moments m by fn(name, m, True)."""
        for group in self.adam.param_groups:
            name, old = group["name"], group["params"][0]
            new = torch.nn.Parameter(fn(name, old.detach(), False), requires_grad=True)
            state = self.adam.state.pop(old, None)
            if state:
                state["exp_avg"] = fn(name, state["exp_avg"], True)
                state["exp_avg_sq"] = fn(name, state["exp_avg_sq"], True)
                self.adam.state[new] = state
            group["params"][0] = new
            setattr(self.model, name, new)


class Densifier:
    """3DGS clone / split / prune restricted to rows >= num_fixed (the new Gaussians)."""

    def __init__(self, model, num_fixed: int, extent: float, grad_threshold: float, percent_dense: float,
                 max_new: int, min_opacity: float = 0.005, max_screen_size: float = 20):
        self.model, self.num_fixed, self.extent, self.max_new = model, num_fixed, extent, max_new
        self.grad_threshold, self.percent_dense = grad_threshold, percent_dense
        self.min_opacity, self.max_screen_size = min_opacity, max_screen_size
        self.reset()

    def reset(self) -> None:
        n = self.model.num_gaussians
        self.accum = torch.zeros(n, device="cuda")
        self.denom = torch.zeros(n, device="cuda")
        self.max_radii = torch.zeros(n, device="cuda")

    @torch.no_grad()
    def add_stats(self, camera) -> None:
        """3DGRUT's screen-gradient proxy and a 3-sigma radius, averaged over in-frustum views."""
        positions, grad = self.model.positions, self.model.positions.grad
        u, v, z = (t[0] for t in project([camera], positions))
        in_frustum = (z > 0.2) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        position = torch.as_tensor(camera.c2w[:3, 3], dtype=positions.dtype, device=positions.device)
        distance = (positions[in_frustum] - position).norm(dim=1)
        self.accum[in_frustum] += (grad[in_frustum] * distance[:, None]).norm(dim=1) / 2
        self.denom[in_frustum] += 1
        radius = 3 * self.model.get_scale()[in_frustum].max(dim=1).values * camera.fx / z[in_frustum]
        self.max_radii[in_frustum] = torch.maximum(self.max_radii[in_frustum], radius)

    @torch.no_grad()
    def step(self, optimizer: Optimizer) -> None:
        from threedgrut.utils.misc import quaternion_to_so3

        model = self.model
        grads = self.accum / self.denom.clamp_min(1)
        new_rows = torch.arange(model.num_gaussians, device="cuda") >= self.num_fixed
        big = model.get_scale().max(dim=1).values > self.percent_dense * self.extent

        if model.num_gaussians - self.num_fixed >= self.max_new:
            grads = torch.zeros_like(grads)  # at the cap: prune only
        clone = (grads >= self.grad_threshold) & ~big & new_rows  # under-reconstruction
        optimizer.edit(lambda name, t, moment: torch.cat([t, torch.zeros_like(t[clone]) if moment else t[clone]]))
        n_clone = int(clone.sum())

        pad = torch.zeros(n_clone, dtype=torch.bool, device="cuda")
        split = torch.cat([(grads >= self.grad_threshold) & big & new_rows, pad])  # over-reconstruction
        scale = model.get_scale()[split].repeat(2, 1)
        samples = torch.normal(torch.zeros_like(scale), scale)
        offsets = torch.bmm(quaternion_to_so3(model.rotation[split]).repeat(2, 1, 1), samples[..., None])[..., 0]
        values = {name: getattr(model, name).detach()[split].repeat(2, 1) for name in NAMES}
        values["positions"] = values["positions"] + offsets
        values["scale"] = model.scale_activation_inv(scale / (0.8 * 2))
        optimizer.edit(lambda name, t, moment: torch.cat(
            [t[~split], torch.zeros_like(values[name]) if moment else values[name]]))
        n_split = int(split.sum())

        max_radii = torch.cat([self.max_radii, torch.zeros(n_clone, device="cuda")])
        max_radii = torch.cat([max_radii[~split], torch.zeros(2 * n_split, device="cuda")])
        prune = (model.get_density()[:, 0] < self.min_opacity) | (max_radii > self.max_screen_size) \
            | (model.get_scale().max(dim=1).values > 0.1 * self.extent)
        prune[:self.num_fixed] = False
        optimizer.edit(lambda name, t, moment: t[~prune])
        print(f"densify: +{n_clone} cloned, {n_split} split, -{int(prune.sum())} pruned -> "
              f"{model.num_gaussians - self.num_fixed} new Gaussians", flush=True)
        self.reset()


def hole_mask_grown(mask: np.ndarray, growth: float = 1.10) -> np.ndarray:
    """Official: dilate with the smallest odd ellipse (3..99) that grows the area by 10%."""
    target = int(mask.sum() * growth)
    dilated = mask.astype(np.uint8)
    for k in range(3, 101, 2):
        dilated = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if dilated.sum() >= target:
            break
    return dilated > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True, help="Holds removal/, virtual/, nbs/, lama/, init/.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="The full scene (for surrounding objects).")
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--lambda_dssim", type=float, default=0.8)
    parser.add_argument("--lambda_lpips", type=float, default=0.0005)
    parser.add_argument("--position_lr", type=float, default=1.6e-4,
                        help="New Gaussians' position lr, times the scene extent; scene Gaussians stay put.")
    parser.add_argument("--max_new_gaussians", type=int, default=2_000_000)
    parser.add_argument("--densify_from", type=int, default=500)
    parser.add_argument("--densify_until", type=int, default=5000)
    parser.add_argument("--densify_interval", type=int, default=100)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    parser.add_argument("--percent_dense", type=float, default=0.01)
    parser.add_argument("--seed_view", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    root = args.output_dir
    out = root / "final"
    model, conf, checkpoint = load_model(root / "init" / "ckpt_init.pt", all_views_overrides(args.colmap_dir))
    num_fixed = int(checkpoint["inpaint360gs_num_fixed"])
    original = {name: getattr(model, name).detach()[:num_fixed].clone() for name in NAMES}
    extent = float(model.scene_extent)
    print(f"finetuning {model.num_gaussians - num_fixed} new + {num_fixed} scene Gaussians, extent {extent:.3f}")

    lrs = {name: conf.optimizer.params[name].lr for name in NAMES}
    lrs["positions"] = args.position_lr * extent
    optimizer = Optimizer(model, lrs)
    densifier = Densifier(model, num_fixed, extent, args.densify_grad_threshold, args.percent_dense,
                          args.max_new_gaussians)

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda().eval()
    for p in lpips.parameters():
        p.requires_grad_(False)

    dataset, loader = build_dataset(conf, num_workers=0)
    template = template_batch(dataset, json.loads((root / "virtual" / "virtual.json").read_text())["reference_view"])
    cameras = load_cameras(root / "virtual" / "cameras.json")
    batches = [virtual_batch(template, c.c2w) for c in cameras]
    targets = [load_rgb(root / "lama" / "rgb" / f"{c.index:05d}.png") for c in cameras]
    holes = [torch.from_numpy(np.asarray(Image.open(root / "nbs" / "mask" / f"{c.index:05d}.png").convert("L")) > 127).cuda()
             for c in cameras]

    generator = torch.Generator().manual_seed(args.seed)
    for iteration in range(args.iterations):
        index = int(torch.randint(len(batches), (1,), generator=generator))
        image = model(batches[index], train=True, frame_id=0)["pred_rgb"][0]
        gt, hole = targets[index], holes[index]
        outside = (~hole)[..., None].float()
        l1 = ((image - gt).abs() * outside).sum() / (outside.sum() * 3).clamp_min(1)
        loss = (1 - args.lambda_dssim) * l1 + args.lambda_dssim * (1 - ssim(image, gt))
        if hole.any():
            ys, xs = torch.nonzero(hole, as_tuple=True)
            crop = (slice(ys.min().item(), ys.max().item() + 1), slice(xs.min().item(), xs.max().item() + 1))
            ci, cg = image[crop].permute(2, 0, 1)[None], gt[crop].permute(2, 0, 1)[None]
            ph, pw = ci.shape[2] // 2, ci.shape[3] // 2
            if ph >= 32 and pw >= 32:  # official divide_into_patches(K=2)
                patches_i = torch.cat([ci[..., a * ph:(a + 1) * ph, b * pw:(b + 1) * pw] for a in range(2) for b in range(2)])
                patches_g = torch.cat([cg[..., a * ph:(a + 1) * ph, b * pw:(b + 1) * pw] for a in range(2) for b in range(2)])
                loss = loss + args.lambda_lpips * lpips(patches_i.clamp(0, 1), patches_g)
        optimizer.adam.zero_grad(set_to_none=True)
        loss.backward()
        if iteration < args.densify_until:
            densifier.add_stats(cameras[index])
        optimizer.adam.step()
        with torch.no_grad():
            model.positions[:num_fixed] = original["positions"]  # the scene's positions stay put
        if args.densify_from < iteration < args.densify_until and iteration % args.densify_interval == 0:
            densifier.step(optimizer)
        if iteration % 500 == 0:
            print(f"iter {iteration}/{args.iterations}: loss {loss.item():.4f}, "
                  f"{model.num_gaussians - num_fixed} new Gaussians", flush=True)

    with torch.no_grad():
        # Reset the remaining scene except next to the hole of virtual view 0 and the seed cloud.
        cam0, seed = cameras[0], cameras[args.seed_view]
        grown = torch.from_numpy(hole_mask_grown(holes[0].cpu().numpy())).cuda()
        u, v, z = (t[0] for t in project([cam0], model.positions.detach()))
        inside = (u >= 0) & (u < cam0.width) & (v >= 0) & (v < cam0.height) & (z > 0)
        near_hole = torch.zeros_like(inside)
        near_hole[inside] = grown[v[inside], u[inside]]
        seed_points = unproject(seed, torch.from_numpy(np.load(root / "lama" / "depth" / f"{seed.index:05d}.npy")).cuda(),
                                holes[args.seed_view]).cpu().numpy()
        candidates = torch.nonzero(near_hole, as_tuple=True)[0]
        if len(candidates) and len(seed_points):
            distance, _ = cKDTree(seed_points).query(model.positions[candidates].cpu().numpy())
            near_hole[:] = False
            near_hole[candidates[torch.from_numpy(distance < seed_points.std(axis=0).mean() * 3.0).cuda()]] = True
        keep_trained = near_hole[:num_fixed]
        for name in NAMES:
            field = getattr(model, name)
            field[:num_fixed][~keep_trained] = original[name][~keep_trained]
        print(f"kept the trained values of {int(keep_trained.sum())} scene Gaussians near the hole")

        removal = json.loads((root / "removal" / "removal.json").read_text())
        if removal["surrounding_ids"]:
            full, _, _ = load_model(args.checkpoint, all_views_overrides(args.colmap_dir))
            surrounding = torch.load(root / "removal" / "removal.pt")["surrounding"].cuda()
            extra = {name: getattr(full, name).detach()[surrounding] for name in NAMES}
            optimizer.edit(lambda name, t, moment: torch.cat([t, torch.zeros_like(extra[name]) if moment else extra[name]]))
            print(f"put back {int(surrounding.sum())} Gaussians of surrounding objects {removal['surrounding_ids']}")

    model.set_optimizable_parameters()
    model.setup_optimizer()
    model.validate_fields()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.get_model_parameters() | {"global_step": int(checkpoint["global_step"]), "epoch": 0,
                                               "inpaint360gs_num_fixed": num_fixed}, out / "ckpt_final.pt")
    with torch.no_grad():
        for batch, camera in zip(batches, cameras):
            save_png(render_view(model, batch, 0)["rgb"], out / "virtual_rgb" / f"{camera.index:05d}.png")
        views, _ = dataset_views(dataset, loader)
        for index, batch in enumerate(views):
            save_png(render_view(model, batch, index)["rgb"], out / "rgb" / f"{index:05d}.png")
    print(f"Done: {model.num_gaussians} Gaussians -> {out}")


if __name__ == "__main__":
    main()
