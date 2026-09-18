# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashSplat's closed-form 2D-to-3D Gaussian segmentation solver.

Port of ``multi_instance_opt`` from https://github.com/florinshen/FlashSplat
(``objremoval.py``; the ``colormask.py`` copy of the same function is broken -- it calls
``enumerate(all_contrib, desc=...)``, and ``enumerate`` takes no ``desc``).

The rendered mask for object *e* is a linear function of the per-Gaussian labels, with
coefficient equal to each Gaussian's accumulated alpha-blending weight. So the optimal
assignment is a per-Gaussian comparison of "weight I contributed to object e" against
"weight I contributed to everything else", which is what this computes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_instance_opt(all_contrib: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
    """Assign each Gaussian to each object.

    Args:
        all_contrib: ``[num_objects + 1, num_gaussians]`` accumulated alpha-blending weight,
            row 0 being background. This is what the instrumented rasterizer produces.
        gamma: background bias in ``[-1, 1]``. Larger values pull Gaussians towards
            background, giving tighter (more conservative) objects; negative values
            grow them. FlashSplat calls this the slackness.

    Returns:
        Boolean ``[num_objects + 1, num_gaussians]``, where ``[i, j]`` is True when
        Gaussian *j* belongs to object *i*. Rows are solved independently, so a Gaussian
        may belong to several objects where masks genuinely overlap.
    """
    if all_contrib.ndim != 2:
        raise ValueError(f"all_contrib must be [num_objects + 1, num_gaussians], got {tuple(all_contrib.shape)}")

    all_contrib = all_contrib.float()
    total_contrib = all_contrib.sum(dim=0)

    labels = torch.zeros_like(all_contrib, dtype=torch.bool)
    for obj_idx, obj_contrib in enumerate(all_contrib):
        if obj_contrib.sum() == 0:  # object never seen; leave its row empty
            continue
        # Row 0 is "everything but this object", row 1 is "this object".
        pair = torch.stack([total_contrib - obj_contrib, obj_contrib], dim=0)
        pair = F.normalize(pair, dim=0)
        pair[0] += gamma
        labels[obj_idx] = pair.argmax(dim=0).bool()
    return labels
