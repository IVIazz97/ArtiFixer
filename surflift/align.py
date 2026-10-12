# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Vendored unchanged from SurfaceLift surflift/align.py (2026-10-12).

"""Align a monocular depth map of the inpainted view to the scene's rendered depth.

1. Global fit: rendered ≈ s * mono + t on a ring of reliable pixels just outside the hole
   (render alpha > 0.95), by iteratively reweighted least squares with Huber weights, so floaters
   and silhouette pixels in the ring do not drag the fit.
2. Boundary correction (``harmonic``): the fit's residual on the ring is extended into the hole
   as a harmonic function (Laplace equation, Dirichlet values on the hole boundary). The aligned
   surface then meets the known surface continuously at the hole's edge instead of floating
   above or sinking below it, and the correction decays smoothly towards the hole's middle, where
   the monocular shape is kept.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import binary_dilation, distance_transform_edt


def huber_fit(x: np.ndarray, y: np.ndarray, iters: int = 20) -> tuple[float, float]:
    A = np.stack([x, np.ones_like(x)], 1)
    w = np.ones_like(x)
    for _ in range(iters):
        s, t = np.linalg.lstsq(A * w[:, None] ** 0.5, y * w ** 0.5, rcond=None)[0]
        r = y - (s * x + t)
        mad = np.median(np.abs(r - np.median(r))) + 1e-9
        delta = 1.345 * 1.4826 * mad
        w = np.where(np.abs(r) <= delta, 1.0, delta / np.abs(r))
    return float(s), float(t)


def harmonic_fill(values: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Solve Laplace(x) = 0 on ``hole`` with x = values on the hole's 4-neighbourhood boundary."""
    H, W = hole.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.nonzero(hole)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, data, b = [], [], [], np.zeros(len(ys))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)
        inside = hole[ny, nx] & ~((ny == ys) & (nx == xs))
        rows.append(np.arange(len(ys))[inside]); cols.append(idx[ny[inside], nx[inside]]); data.append(-np.ones(inside.sum()))
        border = ~hole[ny, nx]
        b[border] += values[ny[border], nx[border]]
    deg = np.zeros(len(ys))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        deg += ((ny >= 0) & (ny < H) & (nx >= 0) & (nx < W))
    A = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(len(ys), len(ys)))
    A = A + sp.diags(deg)
    x, info = spla.cg(A, b, x0=np.full(len(ys), b.sum() / max(deg.sum(), 1)), rtol=1e-6, maxiter=4000)
    out = values.copy()
    out[ys, xs] = x
    return out


def align_depth(mono: np.ndarray, rendered: np.ndarray, alpha: np.ndarray, hole: np.ndarray,
                ring_px: int = 40, mode: str = "harmonic") -> tuple[np.ndarray, dict]:
    """-> aligned depth (H, W) for the whole view (only the hole is used downstream) + diagnostics."""
    near = binary_dilation(hole, iterations=ring_px) & ~binary_dilation(hole, iterations=2)
    ring = near & (alpha > 0.95) & (rendered > 0) & np.isfinite(rendered) & (mono > 0)
    if ring.sum() < 200:                                  # too little context: widen the ring
        near = binary_dilation(hole, iterations=ring_px * 3) & ~hole
        ring = near & (alpha > 0.9) & (rendered > 0) & (mono > 0)
    s, t = huber_fit(mono[ring], rendered[ring])
    aligned = s * mono + t
    info = {"scale": s, "shift": t, "ring_px": int(ring.sum()),
            "ring_abs_err_global": float(np.median(np.abs(rendered[ring] - aligned[ring])))}
    if mode == "harmonic":
        resid = np.zeros_like(mono)
        resid[ring] = rendered[ring] - aligned[ring]
        r = resid[ring]
        lim = 3 * 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        resid[ring] = np.clip(r, np.median(r) - lim, np.median(r) + lim)   # drop floater spikes
        _, (iy, ix) = distance_transform_edt(~ring, return_indices=True)   # boundary values from nearest ring pixel
        resid = resid[iy, ix]
        corr = harmonic_fill(resid, hole)
        aligned = np.where(hole, aligned + corr, aligned)
        info["boundary_correction_median"] = float(np.median(np.abs(corr[hole])))
    aligned = np.maximum(aligned, 1e-3)
    return aligned, info
