# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-view accumulation of per-Gaussian alpha-blending weights for FlashSplat."""

from __future__ import annotations

from pathlib import Path

import torch

from flashsplat.masks import load_object_id_mask, mask_path_for_frame
from flashsplat.threedgut_flashsplat_tracer import Tracer as FlashSplatTracer

# 4 bytes per float32 entry of the [num_objects + 1, num_gaussians] accumulator.
_ACCUMULATOR_WARN_BYTES = 4 << 30


def load_model(checkpoint_path: str | Path, config_overrides: dict | None = None):
    """Load a 3DGUT checkpoint and swap in the FlashSplat-instrumented rasterizer.

    ``MixtureOfGaussians.__init__`` builds the stock ``threedgut_tracer.Tracer``; we
    replace it afterwards. In practice the stock extension is already in the torch
    extension cache from training, so this costs a cache hit rather than a rebuild.
    """
    from threedgrut.model.model import MixtureOfGaussians

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    conf = checkpoint["config"]

    if conf["render"]["method"] != "3dgut":
        raise ValueError(
            f"FlashSplat segmentation requires a 3DGUT checkpoint, got render.method="
            f"{conf['render']['method']!r}. The 3DGRT ray tracer has no equivalent hook."
        )

    for key, value in (config_overrides or {}).items():
        target = conf
        *parents, leaf = key.split(".")
        for part in parents:
            target = target[part]
        target[leaf] = value

    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint)
    model.renderer = FlashSplatTracer(conf)
    model.build_acc()

    return model, conf, int(checkpoint["global_step"])


def build_test_dataloader(conf):
    """Same test-split dataloader construction as ``threedgrut.render.Renderer``."""
    from threedgrut import datasets
    from threedgrut.datasets.utils import configure_dataloader_for_platform

    dataset = datasets.make_test(name=conf.dataset.type, config=conf)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        **configure_dataloader_for_platform(
            {"num_workers": 8, "batch_size": 1, "shuffle": False, "collate_fn": None}
        ),
    )
    return dataset, dataloader


def allocate_accumulator(num_objects: int, num_gaussians: int, device: str = "cuda") -> torch.Tensor:
    """Allocate the ``[num_objects + 1, num_gaussians]`` accumulator, once for the sweep."""
    num_bytes = 4 * (num_objects + 1) * num_gaussians
    if num_bytes > _ACCUMULATOR_WARN_BYTES:
        raise MemoryError(
            f"FlashSplat accumulator would need {num_bytes / 2**30:.1f} GiB for "
            f"{num_objects} objects x {num_gaussians} Gaussians. Reduce --num_objects, or "
            f"segment in batches of objects."
        )
    return torch.zeros((num_objects + 1, num_gaussians), dtype=torch.float32, device=device)


@torch.no_grad()
def accumulate_contributions(
    model,
    dataset,
    dataloader,
    mask_dir: str | Path,
    num_objects: int,
    on_view=None,
) -> torch.Tensor:
    """Render every view with its object-id mask, summing alpha*T per Gaussian per label.

    Args:
        on_view: optional ``(index, batch, outputs) -> None`` hook, for saving renders.

    Returns:
        ``[num_objects + 1, num_gaussians]`` float32 accumulator on the model's device.
    """
    contribution = allocate_accumulator(num_objects, model.num_gaussians)

    for index, batch in enumerate(dataloader):
        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
        height, width = gpu_batch.rays_ori.shape[1:3]

        mask_path = mask_path_for_frame(mask_dir, str(dataset.image_paths[index]))
        object_ids = load_object_id_mask(mask_path, height, width).to(contribution.device)

        max_id = int(object_ids.max())
        if max_id > num_objects:
            raise ValueError(
                f"{mask_path} contains object id {max_id}, but num_objects={num_objects}. "
                f"Ids must lie in 0..{num_objects}."
            )

        outputs = model.renderer.accumulate_contribution(
            model, gpu_batch, object_ids, contribution, frame_id=index
        )
        if on_view is not None:
            on_view(index, gpu_batch, outputs)

    return contribution
