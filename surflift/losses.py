# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Vendored unchanged from SurfaceLift surflift/losses.py (2026-10-12).

"""Photometric losses shared by the single-view fit (lift.py) and the multi-view blend (blend.py)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_window(size=11, sigma=1.5, device="cuda"):
    g = torch.exp(-((torch.arange(size, device=device) - size // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[:, None]
    return (g @ g.T)[None, None].repeat(3, 1, 1, 1)


def ssim_map(a, b, win):
    mu_a = F.conv2d(a, win, padding=5, groups=3); mu_b = F.conv2d(b, win, padding=5, groups=3)
    s_a = F.conv2d(a * a, win, padding=5, groups=3) - mu_a ** 2
    s_b = F.conv2d(b * b, win, padding=5, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, win, padding=5, groups=3) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_a + s_b + C2))


def masked_l1_dssim(img, tgt, mask, win, box=None, w_ssim=0.2):
    """0.8 L1 + 0.2 (1 - SSIM) averaged over ``mask`` (H, W), both maps cropped to ``box``."""
    if box is not None:
        y0, y1, x0, x1 = box
        img, tgt, mask = img[:, y0:y1, x0:x1], tgt[:, y0:y1, x0:x1], mask[y0:y1, x0:x1]
    m = mask.float()
    l1 = ((img - tgt).abs().mean(0) * m).sum() / m.sum().clamp(min=1)
    ssim = (ssim_map(img[None], tgt[None], win)[0].mean(0) * m).sum() / m.sum().clamp(min=1)
    return (1 - w_ssim) * l1 + w_ssim * (1 - ssim), l1, ssim
