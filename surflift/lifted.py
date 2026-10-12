# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Surface-bound Gaussians lifted from one inpainted view (SurfaceLift surflift/lifted.py).

Each new Gaussian is parametrised by a point (u, v) on the inpainted view's image plane:

    position(u, v) = c2w · ( (u - cx) / fx · D(u, v),  (v - cy) / fy · D(u, v),  D(u, v) )

with D the aligned depth map, sampled bilinearly, so the photometric gradient can only slide a
Gaussian *along* the aligned surface. The normal comes from the depth map's normals at (u, v); only
an in-plane angle is free. Free per Gaussian: (u, v) offset (bounded to +-1.5 px), two tangent
scales (bounded to [1/2, 2] x the pixel footprint), in-plane angle, opacity, base colour (SH DC).

Port changes: 3DGUT has no surfels, so every lifted Gaussian is SurfaceLift's ``kind='3dgs'`` flat
3D Gaussian (thickness 5 % of the smaller tangent scale; the Inpaint360GS variant); the principal
point is the view's (cx, cy) instead of the image centre; the colour is one (N, 3) DC tensor and
``features(F)`` pads 3DGUT's specular SH with zeros; ``save``/``load`` keep everything needed to
rebuild the set in one file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import C0, View


def normals_from_depth(D: torch.Tensor, fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    """Camera-space unit normals (H, W, 3) of the surface z = D(u, v), facing the camera."""
    H, W = D.shape
    v, u = torch.meshgrid(torch.arange(H, device=D.device, dtype=D.dtype),
                          torch.arange(W, device=D.device, dtype=D.dtype), indexing="ij")
    P = torch.stack([(u - cx) / fx * D, (v - cy) / fy * D, D], -1)
    du = torch.zeros_like(P); dv = torch.zeros_like(P)
    du[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / 2; du[:, 0] = P[:, 1] - P[:, 0]; du[:, -1] = P[:, -1] - P[:, -2]
    dv[1:-1] = (P[2:] - P[:-2]) / 2; dv[0] = P[1] - P[0]; dv[-1] = P[-1] - P[-2]
    n = F.normalize(torch.cross(dv, du, dim=-1), dim=-1)
    flip = (n * P).sum(-1, keepdim=True) > 0
    return torch.where(flip, -n, n)


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) -> (N, 4) as (w, x, y, z) (graphdeco and 3DGUT convention)."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.stack([
        torch.sqrt(torch.clamp(1 + t, min=1e-8)) / 2,
        torch.sqrt(torch.clamp(1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-8)) / 2,
        torch.sqrt(torch.clamp(1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], min=1e-8)) / 2,
        torch.sqrt(torch.clamp(1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2], min=1e-8)) / 2], 1)
    k = q.argmax(1)
    w, x, y, z = q.unbind(1)
    s01, s02, s12 = m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1]
    d10, d02, d21 = m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] - m[:, 2, 0], m[:, 2, 1] - m[:, 1, 2]
    out = torch.empty_like(q)
    b = k == 0
    out[b] = torch.stack([w[b], d21[b] / (4 * w[b]), d02[b] / (4 * w[b]), d10[b] / (4 * w[b])], 1)
    b = k == 1
    out[b] = torch.stack([d21[b] / (4 * x[b]), x[b], s01[b] / (4 * x[b]), s02[b] / (4 * x[b])], 1)
    b = k == 2
    out[b] = torch.stack([d02[b] / (4 * y[b]), s01[b] / (4 * y[b]), y[b], s12[b] / (4 * y[b])], 1)
    b = k == 3
    out[b] = torch.stack([d10[b] / (4 * z[b]), s02[b] / (4 * z[b]), s12[b] / (4 * z[b]), z[b]], 1)
    return F.normalize(out, dim=1)


class LiftedGaussians(nn.Module):
    def __init__(self, depth: np.ndarray, rgb: np.ndarray, hole: np.ndarray, view: View, init_opacity: float = 0.9,
                 scale_mult: float = 0.7071, max_slide: float = 1.5, device: str = "cuda"):
        super().__init__()
        H, W = depth.shape
        self.view = view
        self.W, self.H, self.fx, self.fy, self.cx, self.cy = W, H, view.fx, view.fy, view.cx, view.cy
        D = torch.tensor(depth, dtype=torch.float32, device=device)
        self.register_buffer("D", D)
        self.register_buffer("Ncam", normals_from_depth(D, self.fx, self.fy, self.cx, self.cy))
        self.register_buffer("c2w", torch.tensor(view.c2w, dtype=torch.float32, device=device))
        self.register_buffer("hole", torch.tensor(np.asarray(hole, bool), device=device))
        ys, xs = np.nonzero(hole)
        base = torch.tensor(np.stack([xs, ys], 1), dtype=torch.float32, device=device)
        self.register_buffer("base_uv", base)
        n = len(base)
        z = D[ys, xs]
        # WonderWorld's footprint z / (sqrt(2) f cos(theta)), grazing angles clamped
        ray = F.normalize(torch.stack([(base[:, 0] - self.cx) / self.fx, (base[:, 1] - self.cy) / self.fy,
                                       torch.ones(n, device=device)], 1), dim=1)
        cos = (-(self.Ncam[ys, xs] * ray).sum(1)).clamp(min=0.25)
        footprint = z / self.fx * scale_mult / cos
        col = torch.tensor(rgb[ys, xs], dtype=torch.float32, device=device)
        self.max_slide = max_slide
        self.register_buffer("log_footprint", torch.log(footprint.clamp_min(1e-8))[:, None].repeat(1, 2))
        self.duv_raw = nn.Parameter(torch.zeros(n, 2, device=device))
        self.scale_raw = nn.Parameter(torch.zeros(n, 2, device=device))
        self.theta = nn.Parameter(torch.zeros(n, device=device))
        self.opacity_logit = nn.Parameter(torch.full((n, 1), float(np.log(init_opacity / (1 - init_opacity))), device=device))
        self.color_dc = nn.Parameter((col - 0.5) / C0)

    @property
    def duv(self) -> torch.Tensor:
        return self.max_slide * torch.tanh(self.duv_raw)

    def uv(self) -> torch.Tensor:
        uv = self.base_uv + self.duv
        return torch.stack([uv[:, 0].clamp(0, self.W - 1), uv[:, 1].clamp(0, self.H - 1)], 1)

    def _sample(self, img: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        g = torch.stack([uv[:, 0] / (self.W - 1) * 2 - 1, uv[:, 1] / (self.H - 1) * 2 - 1], 1)
        out = F.grid_sample(img.permute(2, 0, 1)[None], g[None, :, None, :], mode="bilinear", align_corners=True)
        return out[0, :, :, 0].T

    def xyz(self) -> torch.Tensor:
        uv = self.uv()
        z = self._sample(self.D[..., None], uv)[:, 0]
        Pc = torch.stack([(uv[:, 0] - self.cx) / self.fx * z, (uv[:, 1] - self.cy) / self.fy * z, z], 1)
        return Pc @ self.c2w[:3, :3].T + self.c2w[:3, 3]

    def rotation(self) -> torch.Tensor:
        n_cam = F.normalize(self._sample(self.Ncam, self.uv()), dim=1)
        n = n_cam @ self.c2w[:3, :3].T
        helper = torch.where((n[:, :1].abs() < 0.9), torch.tensor([1.0, 0, 0], device=n.device), torch.tensor([0, 1.0, 0], device=n.device))
        t1 = F.normalize(torch.cross(n, helper, dim=1), dim=1)
        t2 = torch.cross(n, t1, dim=1)
        c, s = torch.cos(self.theta)[:, None], torch.sin(self.theta)[:, None]
        a, b = c * t1 + s * t2, -s * t1 + c * t2
        return rotmat_to_quat(torch.stack([a, b, n], 2))

    def scaling(self) -> torch.Tensor:
        s = torch.exp(self.log_footprint + 0.6931 * torch.tanh(self.scale_raw))
        return torch.cat([s, 0.05 * s.min(1, keepdim=True).values], 1)

    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logit)

    def features(self, F_dim: int) -> torch.Tensor:
        return torch.cat([self.color_dc, self.color_dc.new_zeros(len(self.color_dc), F_dim - 3)], 1)

    def base_color(self) -> torch.Tensor:
        return (C0 * self.color_dc + 0.5).clamp(min=0)

    def __len__(self) -> int:
        return self.base_uv.shape[0]

    def save(self, path: Path) -> None:
        torch.save({"view": self.view.to_json(), "depth": self.D.cpu(), "hole": self.hole.cpu(), "max_slide": self.max_slide,
                    "params": {k: v.detach().cpu() for k, v in self.named_parameters()}}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cuda") -> "LiftedGaussians":
        d = torch.load(path, map_location="cpu", weights_only=False)
        hole = d["hole"].numpy()
        g = cls(d["depth"].numpy(), np.zeros((*hole.shape, 3), np.float32), hole, View.from_json(d["view"]),
                max_slide=d["max_slide"], device=device)
        with torch.no_grad():
            for k, v in d["params"].items():
                getattr(g, k).copy_(v.to(device))
        return g

    def optimizer(self, lr_uv=0.05, lr_scale=0.01, lr_theta=0.01, lr_opacity=0.05, lr_color=0.0025):
        return torch.optim.Adam([
            {"params": [self.duv_raw], "lr": lr_uv}, {"params": [self.scale_raw], "lr": lr_scale},
            {"params": [self.theta], "lr": lr_theta}, {"params": [self.opacity_logit], "lr": lr_opacity},
            {"params": [self.color_dc], "lr": lr_color}], eps=1e-15)
