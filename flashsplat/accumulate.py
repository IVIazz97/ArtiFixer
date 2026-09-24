# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-view accumulation of per-Gaussian alpha-blending weights for FlashSplat."""

from __future__ import annotations

from pathlib import Path

import torch

from flashsplat.masks import find_mask_path_for_frame, load_object_id_mask
from flashsplat.threedgut_flashsplat_tracer import Tracer as FlashSplatTracer

# 4 bytes per float32 entry of the [num_objects + 1, num_gaussians] accumulator.
_ACCUMULATOR_WARN_BYTES = 4 << 30


class _UnbuiltTracer:
    """Stands in for threedgut_tracer.Tracer wherever MixtureOfGaussians builds one.

    Both the stock ``lib3dgut_cc`` and this package's ``lib3dgut_flashsplat_cc`` define a
    C++ ``SplatRaster`` class; pybind11's per-process type registry raises "already
    registered" if both are ever imported into the same process (torch sets
    ``RTLD_GLOBAL``, so the same-named RTTI collides on the second import). Every caller in
    this module and in ``flashsplat.export`` either replaces ``.renderer`` immediately
    (``load_model``) or never renders through the freshly constructed model at all
    (``segment_model``'s checkpoint/PLY export), so the stock tracer is never needed here.
    """

    def __init__(self, _conf):
        pass


def _patch_stock_tracer() -> None:
    import threedgut_tracer

    if threedgut_tracer.Tracer is not _UnbuiltTracer:
        threedgut_tracer.Tracer = _UnbuiltTracer


_patch_stock_tracer()


def load_model(checkpoint_path: str | Path, config_overrides: dict | None = None):
    """Load a 3DGUT checkpoint and swap in the FlashSplat-instrumented rasterizer.

    ``MixtureOfGaussians.__init__`` would otherwise unconditionally build the stock
    ``threedgut_tracer.Tracer`` here, which we discard immediately below -- see
    ``_UnbuiltTracer`` for why that construction is skipped process-wide.
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



def parse_config_overrides(items: list[str] | None) -> dict:
    """Parse ``KEY=VALUE`` strings (dotted keys, YAML values) into a ``load_model`` override dict.

    ``selected_indices_file=null`` is the common case: a prepared ArtiFixer checkpoint selects
    every view for training, which leaves its held-out test split -- the one swept here -- empty.
    """
    import yaml

    overrides = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Config override must look like KEY=VALUE, got {item!r}")
        overrides[key] = yaml.safe_load(value)
    return overrides


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
    return_hit_count: bool = False,
) -> torch.Tensor:
    """Render every view with its object-id mask, summing alpha*T per Gaussian per label.

    Args:
        on_view: optional ``(index, batch, outputs) -> None`` hook, for saving renders.
        return_hit_count: also return per-Gaussian count of views with positive contribution.

    Returns:
        ``[num_objects + 1, num_gaussians]`` float32 accumulator on the model's device.
    """
    contribution = allocate_accumulator(num_objects, model.num_gaussians)
    hit_count = torch.zeros(model.num_gaussians, dtype=torch.int32, device=contribution.device)

    num_used = 0
    num_skipped = 0
    for index, batch in enumerate(dataloader):
        image_name = str(dataset.image_paths[index])
        mask_path = find_mask_path_for_frame(mask_dir, image_name)
        if mask_path is None:
            # Sparse mask sets only annotate a subset of views; skip the rest.
            num_skipped += 1
            continue

        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
        height, width = gpu_batch.rays_ori.shape[1:3]

        object_ids = load_object_id_mask(mask_path, height, width).to(contribution.device)

        max_id = int(object_ids.max())
        if max_id > num_objects:
            raise ValueError(
                f"{mask_path} contains object id {max_id}, but num_objects={num_objects}. "
                f"Ids must lie in 0..{num_objects}."
            )

        previous_contribution = contribution if not return_hit_count else contribution.clone()
        outputs = model.renderer.accumulate_contribution(
            model, gpu_batch, object_ids, contribution, frame_id=index
        )
        if return_hit_count:
            hit_count += (contribution > previous_contribution).any(dim=0).to(torch.int32)
        num_used += 1
        if on_view is not None:
            on_view(index, gpu_batch, outputs)

    print(f"FlashSplat accumulation: used {num_used} views with masks, skipped {num_skipped} without")
    if num_used == 0:
        raise ValueError(f"No masks found in {mask_dir} matching any of the {len(dataset)} views")

    return (contribution, hit_count) if return_hit_count else contribution
