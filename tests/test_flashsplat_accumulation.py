#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness check for the FlashSplat accumulation hook in the 3DGUT rasterizer.

Verifies the hook against a physical invariant of alpha compositing, so it needs no
reference implementation:

    sum over Gaussians of sum over pixels of (alpha * T)  ==  sum over pixels of (1 - T_final)

The right-hand side is exactly what 3DGUT already writes as the alpha channel of its
render. So with an all-ones mask, `contribution[1].sum()` must equal `pred_opacity.sum()`.

Also asserts that enabling the hook does not perturb the render at all.

Needs a CUDA device and a trained 3DGUT checkpoint:
    python tests/test_flashsplat_accumulation.py --checkpoint <ckpt.pt> --colmap_dir <scene>
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "thirdparty" / "3DGRUT-ArtiFixer"))

import torch

from flashsplat import accumulate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This test requires a CUDA device"

    model, conf, _ = accumulate.load_model(args.checkpoint, {"path": str(args.colmap_dir)})
    dataset, dataloader = accumulate.build_test_dataloader(conf)

    failures = []
    for index, batch in enumerate(dataloader):
        if index >= args.num_views:
            break

        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
        height, width = gpu_batch.rays_ori.shape[1:3]

        # Reference render with the hook disabled.
        reference = model.renderer.render(model, gpu_batch, train=False, frame_id=index)

        # All pixels labelled object 1, so every hit lands in row 1 and none in row 0.
        object_ids = torch.ones(height * width, dtype=torch.int32, device="cuda")
        contribution = torch.zeros((2, model.num_gaussians), dtype=torch.float32, device="cuda")
        instrumented = model.renderer.accumulate_contribution(
            model, gpu_batch, object_ids, contribution, frame_id=index
        )

        total_weight = float(contribution[1].sum())
        total_alpha = float(reference["pred_opacity"].sum())
        background_weight = float(contribution[0].sum())
        rgb_delta = float((instrumented["pred_rgb"] - reference["pred_rgb"]).abs().max())

        print(
            f"view {index}: sum(alpha*T)={total_weight:.4f} sum(1-T)={total_alpha:.4f} "
            f"bg={background_weight:.6f} max|rgb delta|={rgb_delta:.3e}"
        )

        if abs(total_weight - total_alpha) > args.rtol * max(total_alpha, 1.0):
            failures.append(f"view {index}: accumulated weight {total_weight} != opacity {total_alpha}")
        if background_weight != 0.0:
            failures.append(f"view {index}: background row is non-zero ({background_weight})")
        if rgb_delta != 0.0:
            failures.append(f"view {index}: the hook perturbed the render (max delta {rgb_delta})")

    # A mask of all -1 must accumulate nothing at all.
    gpu_batch = dataset.get_gpu_batch_with_intrinsics(next(iter(dataloader)))
    height, width = gpu_batch.rays_ori.shape[1:3]
    ignored = torch.full((height * width,), -1, dtype=torch.int32, device="cuda")
    contribution = torch.zeros((2, model.num_gaussians), dtype=torch.float32, device="cuda")
    model.renderer.accumulate_contribution(model, gpu_batch, ignored, contribution)
    if float(contribution.abs().sum()) != 0.0:
        failures.append("negative object ids were accumulated instead of skipped")

    if failures:
        print("\nFAIL:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
