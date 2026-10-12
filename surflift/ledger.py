# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation ledger: which Gaussians real pixels have actually seen (SurfaceLift surflift/ledger.py).

For every Gaussian, its compositing weight w_i(p) = T_i(p) alpha_i(p) summed over all training
views, split by two pixel sets A, B (default: inside / outside the view's object mask M_v). One
render + one backward pass per view gives both: with every Gaussian white through its SH DC term,
d/d(dc_i) of  sum_{p in A} R_0(p) + sum_{p in B} R_1(p)  is C0 * A-weight on channel 0 and
C0 * B-weight on channel 1. On 3DGUT the same holds: its radiance is max(C0 dc + 0.5, 0) = 1 > 0.

``revealed_consistency`` is the pixel-level test of what a removal reveals (the v3c "consistency"
rule): a revealed pixel passes if another photo saw its 3D point (in frame, unoccluded against
that photo's full-model depth, outside its object mask) with an agreeing colour.

Port changes: the witness data (full-model depth, mask, blurred photo of every view) is computed
once and kept on the GPU at a reduced resolution when the views would not fit the budget
(``Witnesses``), and the test can run on any camera, including virtual ones (``check_view``).
"""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn.functional as F

from .common import C0, Frozen, Scene, View, scene_up, white_features


def observation_ledger(sc: Scene, model: Frozen, views: list[View], masks, log=functools.partial(print, flush=True), split=None) -> dict:
    """``masks``: callable view -> (H, W) bool object mask. ``split``: optional callable view ->
    (A, B) bool maps; then ``masked`` sums over A and ``seen`` over B. -> dict of (N,) float32."""
    dev = model.xyz.device
    N = len(model)
    feats = white_features(N, sc.F, dev).requires_grad_(True)
    m = Frozen(model.xyz, model.rot, model.scale, model.opacity, feats)
    seen = torch.zeros(N, device=dev)
    masked = torch.zeros(N, device=dev)
    for k, v in enumerate(views):
        if split is None:
            M = torch.as_tensor(masks(v), device=dev)
            A, B = M, ~M
        else:
            A, B = (torch.as_tensor(x, device=dev) for x in split(v))
        out = sc.trace(v, m, geometry=False)["rgb"]
        loss = (out[0] * A).sum() + (out[1] * B).sum()
        g, = torch.autograd.grad(loss, feats)
        masked += g[:, 0] / C0
        seen += g[:, 1] / C0
        if k % 50 == 0:
            log(f"ledger {k}/{len(views)}")
    return {"seen": seen.detach().cpu().numpy().astype(np.float32), "masked": masked.detach().cpu().numpy().astype(np.float32)}


class Witnesses:
    """Every training photo as a witness: its full-model depth, object mask and 3x3-box-blurred
    photo (tolerates misregistration), stored at 1/``s`` resolution if the views exceed ``budget_px``."""

    def __init__(self, sc: Scene, full: Frozen, views: list[View], masks, photos, budget_px: float = 4.0e8, log=functools.partial(print, flush=True)):
        dev = full.xyz.device
        H, W = views[0].H, views[0].W
        self.s = 1
        while len(views) * (H // self.s) * (W // self.s) > budget_px:
            self.s *= 2
        self.views = views
        self.Df, self.Ms, self.Ph = [], [], []
        with torch.no_grad():
            for u in views:
                d = sc.trace(u, full, geometry=True)["depth"]
                m = torch.as_tensor(masks(u), device=dev)
                ph = torch.tensor(photos(u), device=dev).permute(2, 0, 1)[None].float() / 255
                ph = F.avg_pool2d(ph, 3, 1, 1)
                if self.s > 1:            # nearest for depth / mask (no mixing across edges), area for the photo
                    d = d[self.s // 2::self.s, self.s // 2::self.s]
                    m = F.max_pool2d(m[None, None].float(), self.s)[0, 0] > 0
                    ph = F.avg_pool2d(ph, self.s)
                self.Df.append(d.half())
                self.Ms.append(m)
                self.Ph.append(ph[0].half())
        self.R = torch.stack([torch.tensor(u.R, dtype=torch.float32, device=dev) for u in views])
        self.t = torch.stack([torch.tensor(u.t, dtype=torch.float32, device=dev) for u in views])
        log(f"witnesses: {len(views)} views at 1/{self.s} resolution")

    def best_diff(self, X: torch.Tensor, col: torch.Tensor, skip: int | None, depth_tol: float):
        """For world points X (P, 3) with colours col (P, 3): the smallest mean |RGB difference| over the
        photos that saw them (9.0 if none), and whether any photo saw them."""
        best = torch.full((len(X),), 9.0, device=X.device)
        seen = torch.zeros(len(X), dtype=torch.bool, device=X.device)
        s = self.s
        for j, u in enumerate(self.views):
            if j == skip:
                continue
            Xc = X @ self.R[j].T + self.t[j]
            zc = Xc[:, 2]
            uu = u.fx * Xc[:, 0] / zc.clamp(min=1e-6) + u.cx
            ww = u.fy * Xc[:, 1] / zc.clamp(min=1e-6) + u.cy
            ok = (zc > 0) & (uu >= 0) & (uu <= u.W - 1) & (ww >= 0) & (ww <= u.H - 1)
            if not ok.any():
                continue
            h, w = self.Ms[j].shape
            ui = (uu.round().long() // s).clamp(0, w - 1)
            wi = (ww.round().long() // s).clamp(0, h - 1)
            vis = ok & ((self.Df[j][wi, ui].float() - zc).abs() < depth_tol * zc) & ~self.Ms[j][wi, ui]
            diff = (self.Ph[j][:, wi, ui].T.float() - col).abs().mean(1)
            best = torch.where(vis, torch.minimum(best, diff), best)
            seen |= vis
        return best, seen


@torch.no_grad()
def check_view(sc: Scene, scene: Frozen, view: View, region: np.ndarray, wit: Witnesses, skip: int | None,
               depth_tol: float = 0.03, col_tol: float = 0.1, alpha_min: float = 0.5, render=None):
    """revealed_consistency for one camera (real or virtual): pixels of ``region`` where ``scene``
    renders content (alpha > alpha_min). -> (pass_map, fail_map, observed_map) bool (H, W) numpy;
    observed = some photo saw the point (colour ignored), the unseen_view.py test."""
    dev = scene.xyz.device
    H, W = view.H, view.W
    r = render if render is not None else sc.trace(view, scene, geometry=True)
    rgb, D, A = r["rgb"], r["depth"], r["alpha"]
    P = torch.as_tensor(region, device=dev) & (A > alpha_min)
    ys, xs = torch.nonzero(P, as_tuple=True)
    pm = np.zeros((H, W), bool)
    fm = np.zeros((H, W), bool)
    om = np.zeros((H, W), bool)
    if len(ys) == 0:
        return pm, fm, om
    z = D[ys, xs]
    Pc = torch.stack([(xs - view.cx) / view.fx * z, (ys - view.cy) / view.fy * z, z], 1)
    c2w = torch.tensor(view.c2w, dtype=torch.float32, device=dev)
    X = Pc @ c2w[:3, :3].T + c2w[:3, 3]
    col = rgb[:, ys, xs].T
    good = torch.zeros(len(X), dtype=torch.bool, device=dev)
    obs = torch.zeros(len(X), dtype=torch.bool, device=dev)
    for c0 in range(0, len(X), 1 << 20):
        best, seen = wit.best_diff(X[c0:c0 + (1 << 20)], col[c0:c0 + (1 << 20)], skip, depth_tol)
        good[c0:c0 + (1 << 20)] = best < col_tol
        obs[c0:c0 + (1 << 20)] = seen
    ys, xs, good, obs = ys.cpu().numpy(), xs.cpu().numpy(), good.cpu().numpy(), obs.cpu().numpy()
    pm[ys[good], xs[good]] = True
    fm[ys[~good], xs[~good]] = True
    om[ys[obs], xs[obs]] = True
    return pm, fm, om


@torch.no_grad()
def revealed_consistency(sc: Scene, scene: Frozen, views: list[View], masks, wit: Witnesses, col_tol: float = 0.1,
                         log=functools.partial(print, flush=True)) -> tuple[dict, dict]:
    """SurfaceLift's revealed_consistency over every training view (each view is checked against all
    the others). -> ({view index: (pass_map, fail_map)}, summary counts)."""
    out, n_pass, n_fail = {}, 0, 0
    for k, v in enumerate(views):
        pm, fm, _ = check_view(sc, scene, v, masks(v), wit, skip=k, col_tol=col_tol)
        out[v.index] = (pm, fm)
        n_pass += int(pm.sum())
        n_fail += int(fm.sum())
        if k % 50 == 0:
            log(f"consistency {k}/{len(views)}: pass {int(pm.sum())} fail {int(fm.sum())}")
    return out, {"pass_px": n_pass, "fail_px": n_fail}
