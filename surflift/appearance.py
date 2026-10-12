# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harmonic colour alignment of the lifted fill (SurfaceLift surflift/appearance.py, ``align_colour``
and its helpers only; the view-dependent transfer is opt-in there and not used by v3c).

Port change: the lifted colour is one (N, 3) DC tensor (``color_dc.data[:]`` instead of ``[:, 0]``).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_dilation, gaussian_filter


def _norm_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``img`` (H, W, C) over ``mask`` in a Gaussian window: normalised convolution."""
    w = gaussian_filter(mask.astype(np.float64), sigma)
    v = gaussian_filter(img * mask[..., None], (sigma, sigma, 0))
    return v / np.maximum(w, 1e-6)[..., None], w


@torch.no_grad()
def align_colour(lifted, fit: np.ndarray, hole: np.ndarray, ralpha: np.ndarray, band: int = 16,
                 max_log: float = 0.9, max_chroma: float = 0.08, log=print) -> dict:
    """Harmonic colour alignment of the fill in the chosen view (Poisson blending, in log colour).

    The inpainting's low frequencies (brightness, tint) need not match the surface around it
    (cleaner, lighter tiles than the real mossy ones), which reads as a patch from every view. At
    each point of the hole's edge, the local mean colour just outside (removed scene, opaque) is
    compared with the local mean just inside (the fill): the log ratio, mostly luminance (chroma
    limited to +-``max_chroma``), is the jump to remove. It is extended over the hole as a harmonic
    function, so the fill keeps all its own detail and only its smooth shading is bent to meet the
    boundary, and every lifted Gaussian's base colour is multiplied by it at its pixel."""
    H, W = hole.shape
    img = fit.astype(np.float64)
    inner = _erode(hole, 2) & ~_erode(hole, band)                         # fill, 2..band px inside
    outer = _dilate(hole, band) & ~_dilate(hole, 2) & (ralpha > 0.95)
    mu_in, w_in = _norm_blur(img, inner, band / 2)
    mu_out, w_out = _norm_blur(img, outer, band / 2)
    edge = _dilate(hole, 1) & ~hole                                          # Dirichlet pixels
    ok = edge & (w_in > 0.05) & (w_out > 0.05)
    d = np.log(mu_out + 0.02) - np.log(mu_in + 0.02)
    lum = d.mean(-1, keepdims=True)
    d = np.clip(lum, -max_log, max_log) + np.clip(d - lum, -max_chroma, max_chroma)
    vals = np.zeros((H, W, 3))
    vals[ok] = d[ok]
    # boundary pixels without a reliable pair take the nearest reliable one (the harmonic solve
    # needs values on the whole edge), then smooth along the edge
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(~ok, return_indices=True)
    vals = vals[iy, ix]
    vals_s, _ = _norm_blur(vals, edge, band / 2)
    vals = np.where(edge[..., None], vals_s, 0.0)
    from .align import harmonic_fill
    corr = np.stack([harmonic_fill(vals[..., c], hole) for c in range(3)], -1)
    uv = lifted.uv().round().long()
    g = torch.tensor(corr, dtype=torch.float32, device=lifted.color_dc.device)[uv[:, 1].clamp(0, H - 1), uv[:, 0].clamp(0, W - 1)]
    c = lifted.base_color() * torch.exp(g)
    lifted.color_dc.data[:] = (c - 0.5) / 0.28209479177387814
    info = {"edge_px": int(ok.sum()), "gain_median": np.exp(np.median(corr[hole], 0)).round(3).tolist(),
            "gain_p05": np.exp(np.percentile(corr[hole].mean(1), 5)).round(3).item(),
            "gain_p95": np.exp(np.percentile(corr[hole].mean(1), 95)).round(3).item()}
    log(info)
    return info


def _dilate(m, n):
    return binary_dilation(m, iterations=n)


def _erode(m, n):
    from scipy.ndimage import binary_erosion
    return binary_erosion(m, iterations=n)
