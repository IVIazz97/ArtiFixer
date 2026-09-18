# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export FlashSplat segmentation results: checkpoints, PLYs, and overlay renders."""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
import torch


def segment_model(model, keep_mask: torch.Tensor, setup_optimizer: bool = True):
    """Slice a ``MixtureOfGaussians`` down to the Gaussians selected by ``keep_mask``.

    Uses the model's own ``__getitem__`` slicing, then re-establishes optimizable
    parameters so the result can produce a loadable checkpoint.
    """
    # MixtureOfGaussians.copy_fields reads these unconditionally, but __init__ only sets
    # them when progressive training is on (init_n_features < max_n_features). ArtiFixer's
    # configs always enable it; fill in defaults so a config that does not still works.
    for attribute, default in (("feature_dim_increase_interval", 0), ("feature_dim_increase_step", 0)):
        if not hasattr(model, attribute):
            setattr(model, attribute, default)

    sliced = model[keep_mask.to(model.positions.device)]
    if setup_optimizer:
        sliced.set_optimizable_parameters()
        sliced.setup_optimizer()
    sliced.validate_fields()
    return sliced


def save_segmented_checkpoint(model, keep_mask: torch.Tensor, path: str | Path, global_step: int) -> int:
    """Write a checkpoint holding only the selected Gaussians.

    The dict layout matches ``threedgrut.trainer.Trainer.save_checkpoint`` so that
    ``threedgrut.render.Renderer.from_checkpoint`` loads it directly. Optimizer state is
    freshly initialised rather than sliced -- these checkpoints are for rendering and
    export, not for resuming training.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sliced = segment_model(model, keep_mask)
    parameters = sliced.get_model_parameters()
    parameters |= {"global_step": global_step, "epoch": 0}
    torch.save(parameters, path)
    return sliced.num_gaussians


def save_segmented_ply(model, keep_mask: torch.Tensor, path: str | Path) -> int:
    """Write the selected Gaussians as a PLY via the model's existing exporter."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sliced = segment_model(model, keep_mask, setup_optimizer=False)
    sliced.export_ply(str(path))
    return sliced.num_gaussians


def id_to_rgb(object_id: int) -> np.ndarray:
    """Distinct colour per object id, golden-ratio hue spacing (as in FlashSplat)."""
    rgb = np.zeros((3,), dtype=np.uint8)
    if object_id == 0:  # background
        return rgb
    golden_ratio = 1.6180339887
    hue = (object_id * golden_ratio) % 1
    saturation = 0.5 + (object_id % 2) * 0.5
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, saturation)
    rgb[:] = (int(r * 255), int(g * 255), int(b * 255))
    return rgb


@torch.no_grad()
def save_overlay_renders(
    model,
    dataset,
    dataloader,
    labels: torch.Tensor,
    out_dir: str | Path,
    alpha_threshold: float = 0.5,
    blend: float = 0.4,
) -> int:
    """Render each solved object and blend its colour over the RGB render.

    Objects are composited in id order; an object only claims pixels no nearer object has
    already claimed, resolved by the rendered depth (as in FlashSplat's ``colormask.py``).
    """
    import torchvision

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_objects = labels.shape[0] - 1
    written = 0

    # Slice once up front: each MixtureOfGaussians constructor builds its own rasterizer,
    # so doing this per view would rebuild them for every frame.
    sliced_models = {}
    for object_id in range(1, num_objects + 1):
        keep = labels[object_id]
        if not bool(keep.any()):
            continue
        sliced = segment_model(model, keep, setup_optimizer=False)
        sliced.renderer = model.renderer
        sliced_models[object_id] = sliced

    for index, batch in enumerate(dataloader):
        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)

        base = model(gpu_batch, train=False, frame_id=index)
        overlay = base["pred_rgb"][0].permute(2, 0, 1).clone()  # [3, H, W]
        claimed_depth = None

        for object_id, sliced in sliced_models.items():
            outputs = sliced(gpu_batch, train=False, frame_id=index)

            opacity = outputs["pred_opacity"][0, ..., 0]  # [H, W]
            depth = outputs["pred_dist"][0, ..., 0]
            visible = opacity > alpha_threshold

            if claimed_depth is None:
                claimed_depth = torch.full_like(depth, float("inf"))
            # Only claim pixels where this object is in front of whatever claimed them.
            visible &= depth < claimed_depth
            if not bool(visible.any()):
                continue

            colour = torch.tensor(
                id_to_rgb(object_id) / 255.0, dtype=overlay.dtype, device=overlay.device
            ).view(3, 1, 1)
            overlay = torch.where(visible.unsqueeze(0), (1 - blend) * overlay + blend * colour, overlay)
            claimed_depth = torch.where(visible, depth, claimed_depth)

        torchvision.utils.save_image(overlay.clamp(0, 1), out_dir / f"{index:05d}.png")
        written += 1

    return written
