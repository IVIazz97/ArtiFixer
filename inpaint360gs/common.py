# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3DGUT helpers shared by the Inpaint360GS stages that run in the ArtiFixer environment.

Scene loading, dataset cameras, rendering and (un)projection come from ``aurafusion.scene``;
this adds what Inpaint360GS needs on top: renders from virtual poses, alpha-blended per-Gaussian
identity features (paper Eq. 2) with gradients, and integer id-mask I/O.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import torch

from aurafusion.scene import (  # noqa: F401  (re-exported for the stage modules)
    ViewCamera, all_views_overrides, build_dataset, load_cameras, load_model, pixel_grid, project, render_view,
    save_cameras, save_png, unproject, view_camera,
)

SH_C0 = 0.28209479177387814
# 3DGUT decodes radiance as max(C0 * dc + 0.5, 0). Identity features are rendered as radiance
# f + FEATURE_OFFSET, so the clamp never fires, and the offset times the opacity is subtracted.
FEATURE_OFFSET = 32.0


def dataset_views(dataset, loader) -> tuple[list, list[ViewCamera]]:
    """GPU batches and pinhole cameras of every dataset view, in dataset order."""
    batches, cameras = [], []
    for index, batch in enumerate(loader):
        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
        batches.append(gpu_batch)
        cameras.append(view_camera(dataset, index, gpu_batch))
    return batches, cameras


def template_batch(dataset, name: str):
    """GPU batch of the dataset view whose image file is ``name`` (the virtual views' intrinsics)."""
    index = next(i for i, p in enumerate(dataset.image_paths) if Path(str(p)).name == name)
    return dataset.get_gpu_batch_with_intrinsics(torch.utils.data.default_collate([dataset[index]]))


def virtual_batch(template, c2w: np.ndarray):
    """A copy of ``template`` (same intrinsics and camera-space rays) seen from OpenCV ``c2w``."""
    pose = torch.as_tensor(np.asarray(c2w), dtype=template.T_to_world.dtype, device=template.T_to_world.device)
    pose = pose[: template.T_to_world.shape[1], :][None]
    return dataclasses.replace(template, T_to_world=pose.contiguous(), rgb_gt=None, mask=None)


def render_features(model, gpu_batch, features: torch.Tensor, frame_id: int = 0) -> torch.Tensor:
    """[H, W, K] alpha-blended ``features`` [N, K], F = sum_i f_i alpha_i T_i, differentiable in
    ``features`` only (geometry is detached).

    3DGUT hard-codes 3 radiance channels, so the K channels are rendered 3 at a time as SH-DC
    radiance with the active SH degree set to 0. The background model is bypassed, so empty
    pixels render 0 as in the official rasterizer.
    """
    from threedgut_tracer.tracer import Tracer

    tracer = model.renderer
    sensor, poses = Tracer._Tracer__create_camera_parameters(gpu_batch)
    n, k = features.shape
    pad = (-k) % 3
    shifted = torch.nn.functional.pad(features, (0, pad)) + FEATURE_OFFSET
    rest = features.new_zeros(n, model.features_specular.shape[1])
    geometry = [t.detach().contiguous() for t in
                (model.positions, model.get_rotation(), model.get_scale(), model.get_density())]
    height, width = gpu_batch.rays_dir.shape[1:3]
    chunks, alpha = [], None
    for c in range(0, k + pad, 3):
        dc = (shifted[:, c:c + 3] - 0.5) / SH_C0
        rgba, _, _, _ = Tracer._Autograd.apply(
            tracer.tracer_wrapper, frame_id, 0, gpu_batch.rays_ori.contiguous(), gpu_batch.rays_dir.contiguous(),
            *geometry, torch.cat([dc, rest], dim=1).contiguous(), sensor, poses,
        )
        rgba = rgba.reshape(height, width, 4)
        chunks.append(rgba[..., :3])
        alpha = rgba[..., 3:].detach()
    return torch.cat(chunks, dim=-1)[..., :k] - FEATURE_OFFSET * alpha


def save_ids(ids: np.ndarray, path: Path) -> None:
    """Integer id map as a 16-bit PNG (ids are never rescaled)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    assert ids.max(initial=0) < 65536, "id map exceeds 16 bits"
    cv2.imwrite(str(path), ids.astype(np.uint16))


def load_ids(path: Path, height: int | None = None, width: int | None = None) -> np.ndarray:
    """Integer id map, nearest-resized to (height, width) if given."""
    ids = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if ids is None:
        raise FileNotFoundError(path)
    if ids.ndim == 3:
        ids = ids[..., 0]
    if height is not None and ids.shape != (height, width):
        ids = cv2.resize(ids, (width, height), interpolation=cv2.INTER_NEAREST)
    return ids.astype(np.int64)


def colorize_ids(ids: np.ndarray) -> np.ndarray:
    """[H, W, 3] uint8 visualisation, one colour per id, black background."""
    from flashsplat.export import id_to_rgb

    palette = np.stack([id_to_rgb(i) for i in range(int(ids.max(initial=0)) + 1)])
    return palette[ids]


def load_rgb(path: Path) -> torch.Tensor:
    """[H, W, 3] float image in [0, 1] on the GPU."""
    image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).cuda().float() / 255.0
