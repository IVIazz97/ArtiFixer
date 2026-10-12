# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3DGUT backend of the SurfaceLift port: cameras, the frozen scene, rendering, checkpoint export.

Replaces SurfaceLift's ``surflift/backends.py`` (the base methods' own ``render()`` through a
``Composite`` model), ``surflift/cameras.py`` (COLMAP views, virtual look-at cameras) and
``surflift/scene_data.py`` (photos, object masks). One ``Scene`` holds the 3DGUT reconstruction
(the FULL model, object present), every dataset view in COLMAP/dataset order, their photos at the
renderer's resolution and the object masks written by ``surflift.prepare``.

Rendering calls the stock 3DGUT tracer's autograd function directly on concatenated tensors
(frozen Gaussians + lifted ones), so the lifted parameters get gradients and arbitrary per-Gaussian
colours can be traced (SurfaceLift's "white" / "lifted" override modes and the ledger's gradient
trick). Depth is 3DGUT's ``pred_dist / pred_opacity`` turned into camera z (alpha-normalised mean
depth, as SurfaceLift's Inpaint360GS backend); alpha is ``pred_opacity``.

Pixel convention: 3DGUT's rays go through pixel centres ``index + 0.5`` with the dataset's principal
point, so a camera point projects to pixel index ``fx * x / z + cx`` with ``cx = principal_x - 0.5``
(``View.cx``). SurfaceLift's code used ``(W - 1) / 2`` for that; every use was replaced by ``View.cx``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import aurafusion.scene  # noqa: F401  (puts thirdparty/3DGRUT-ArtiFixer on sys.path)
from aurafusion.scene import all_views_overrides, build_dataset, load_model, view_camera

C0 = 0.28209479177387814


# ---------------------------------------------------------------------------------------------
# cameras (SurfaceLift surflift/cameras.py, with a principal point and the 3DGUT intrinsics key)

@dataclass
class View:
    index: int                # dataset (COLMAP) order index; -1 for a virtual camera
    name: str                 # image file name, or a virtual camera's name
    W: int
    H: int
    fx: float
    fy: float
    cx: float                 # principal point in pixel-index coordinates (u_index = fx x / z + cx)
    cy: float
    R: np.ndarray             # (3, 3) world -> camera (OpenCV axes)
    t: np.ndarray             # (3,)
    intr: int = 0             # 3DGUT intrinsics id (which template batch holds this camera's rays)
    virtual: bool = False

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    @property
    def c2w(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.R.T
        m[:3, 3] = -self.R.T @ self.t
        return m

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        return self.R[2]

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (N, 3) -> pixel-index coords (N, 2) and camera depth z (N,)."""
        Xc = X @ self.R.T + self.t
        z = Xc[:, 2]
        u = self.fx * Xc[:, 0] / np.maximum(z, 1e-8) + self.cx
        v = self.fy * Xc[:, 1] / np.maximum(z, 1e-8) + self.cy
        return np.stack([u, v], 1), z

    def to_json(self) -> dict:
        d = {k: getattr(self, k) for k in ("index", "name", "W", "H", "fx", "fy", "cx", "cy", "intr", "virtual")}
        return d | {"R": self.R.tolist(), "t": self.t.tolist()}

    @staticmethod
    def from_json(d: dict) -> "View":
        return View(**{**d, "R": np.asarray(d["R"], np.float64), "t": np.asarray(d["t"], np.float64)})


def save_view(view: View, path: Path) -> None:
    Path(path).write_text(json.dumps(view.to_json(), indent=1))


def load_view(path: Path) -> View:
    return View.from_json(json.loads(Path(path).read_text()))


def scene_up(views: list[View]) -> np.ndarray:
    """World 'up': the mean of the cameras' -y axes (photos are taken roughly upright)."""
    up = -np.mean([v.R[1] for v in views], 0)
    return up / np.linalg.norm(up)


def look_at(template: View, eye: np.ndarray, target: np.ndarray, up: np.ndarray, name: str) -> View:
    """A virtual camera at ``eye`` looking at ``target``, with the template's intrinsics."""
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)                    # OpenCV: x right, y down (~ -up), z forward
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0]))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return dataclasses.replace(template, index=-1, name=name, R=R, t=-R @ eye, virtual=True)


def orbit_candidates(views: list[View], center: np.ndarray, up: np.ndarray,
                     elev_offsets=(0, 15, 30, 45), radii=(1.0, 0.75), n_azim=18) -> list[View]:
    """SurfaceLift's virtual look-at cameras: spheres around the object at the training cameras'
    median distance (and closer), azimuths all around, elevations from the training cameras'
    median up to +45 deg."""
    rel = np.array([v.center - center for v in views])
    dist = np.linalg.norm(rel, axis=1)
    elev0 = np.degrees(np.median(np.arcsin(np.clip(rel @ up / dist, -1, 1))))
    a = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    out = []
    for rr in radii:
        r = np.median(dist) * rr
        for de in elev_offsets:
            e = np.radians(min(elev0 + de, 85.0))
            for k in range(n_azim):
                az = 2 * np.pi * k / n_azim
                eye = center + r * (np.cos(e) * (np.cos(az) * a + np.sin(az) * b) + np.sin(e) * up)
                out.append(look_at(views[0], eye, center, up, f"virtual_r{int(rr * 100):03d}_e{de:02d}_a{k:02d}"))
    return out


def unproject_np(view: View, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    Pc = np.stack([(xs - view.cx) / view.fx * z, (ys - view.cy) / view.fy * z, z], 1)
    return Pc @ view.c2w[:3, :3].T + view.c2w[:3, 3]


# ---------------------------------------------------------------------------------------------
# Gaussians

@dataclass
class Frozen:
    """Activated 3DGUT Gaussian attributes: positions, unit quaternion (w, x, y, z), scales,
    opacity (N, 1) and radiance features (N, 3 + specular): SH DC first, as 3DGUT stores them."""
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    features: torch.Tensor

    def __len__(self) -> int:
        return self.xyz.shape[0]


FIELDS = ("xyz", "rot", "scale", "opacity", "features")


def subset(f: Frozen, keep) -> Frozen:
    keep = torch.as_tensor(keep, device=f.xyz.device)
    return Frozen(*(getattr(f, n)[keep] for n in FIELDS))


def concat(a: Frozen, b: Frozen) -> Frozen:
    return Frozen(*(torch.cat([getattr(a, n), getattr(b, n)], 0) for n in FIELDS))


def white_features(n: int, F: int, device) -> torch.Tensor:
    f = torch.zeros(n, F, device=device)
    f[:, :3] = 0.5 / C0                     # radiance max(C0 dc + 0.5, 0) = 1 in every channel
    return f


# ---------------------------------------------------------------------------------------------
# scene

def _to_view(cam, intr: int) -> View:
    c2w = np.asarray(cam.c2w, np.float64)
    R = c2w[:3, :3].T
    return View(cam.index, cam.name, cam.width, cam.height, cam.fx, cam.fy, cam.cx - 0.5, cam.cy - 0.5,
                R, -R @ c2w[:3, 3], intr=intr)


class Scene:
    """The 3DGUT reconstruction + every dataset view (+ photos, + object masks under ``work``)."""

    def __init__(self, checkpoint: Path, colmap_dir: Path, work: Path | None = None, photos: bool = True,
                 log=print):
        from threedgut_tracer.tracer import Tracer

        self._Tracer = Tracer
        model, conf, ckpt = load_model(Path(checkpoint), all_views_overrides(Path(colmap_dir)))
        self.model, self.conf = model, conf
        self.global_step = int(ckpt.get("global_step", 0))
        del ckpt
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.full = Frozen(model.positions.detach().float().contiguous(), model.get_rotation().detach().float().contiguous(),
                               model.get_scale().detach().float().contiguous(), model.get_density().detach().float().contiguous(),
                               model.get_features().detach().float().contiguous())
        self.F = self.full.features.shape[1]
        self.n_active = int(model.n_active_features)
        dataset, loader = build_dataset(conf, num_workers=4)
        self.views: list[View] = []
        self.templates: dict = {}
        self._photos: dict[int, np.ndarray] = {}
        for i, batch in enumerate(loader):
            intr = int(batch["intr"][0])
            gb = dataset.get_gpu_batch_with_intrinsics(batch)
            self.views.append(_to_view(view_camera(dataset, i, gb), intr))
            if intr not in self.templates:
                self.templates[intr] = dataclasses.replace(gb, rgb_gt=None, mask=None)
            if photos:
                a = batch["data"][0].numpy()
                a = a.reshape(a.shape[-3:])          # (1, H, W, 3) -> (H, W, 3)
                self._photos[i] = a if a.dtype == np.uint8 else np.clip(np.round(a), 0, 255).astype(np.uint8)
        log(f"scene: {len(self.full)} Gaussians, {len(self.views)} views, {len(self.templates)} intrinsics, "
            f"{self.views[0].W}x{self.views[0].H}")
        self.work = None if work is None else Path(work)
        self._masks: dict[int, np.ndarray] = {}

    # ---- data
    def photo(self, i: int) -> np.ndarray:
        return self._photos[i]

    def mask(self, i: int) -> np.ndarray:
        """(H, W) bool object mask M0 of view i (surflift.prepare: SAM mask | object opacity, dilated)."""
        if i not in self._masks:
            self._masks[i] = np.asarray(Image.open(self.work / "masks" / f"{i:05d}.png")) > 127
        return self._masks[i]

    # ---- rendering
    def trace(self, view: View, g: Frozen, features: torch.Tensor | None = None, geometry: bool = True) -> dict:
        """Render Gaussians ``g`` (optionally with override features) from ``view``.
        -> rgb (3, H, W); with geometry also alpha (H, W) and depth (H, W, camera z, 0 where empty)."""
        from inpaint360gs.common import virtual_batch

        Tr = self._Tracer
        batch = virtual_batch(self.templates[view.intr], view.c2w)
        sensor, poses = Tr._Tracer__create_camera_parameters(batch)
        feats = g.features if features is None else features
        rgba, dist, _, _ = Tr._Autograd.apply(
            self.model.renderer.tracer_wrapper, 0, self.n_active, batch.rays_ori.contiguous(), batch.rays_dir.contiguous(),
            g.xyz.contiguous(), g.rot.contiguous(), g.scale.contiguous(), g.opacity.contiguous(), feats.contiguous(),
            sensor, poses)
        H, W = view.H, view.W
        rgba = rgba.reshape(H, W, 4)
        rgb, alpha = rgba[..., :3], rgba[..., 3]
        rgb, _ = self.model.background(batch.T_to_world.contiguous(), batch.rays_dir, rgb[None], alpha[None, ..., None], False)
        out = {"rgb": rgb.reshape(H, W, 3).permute(2, 0, 1)}
        if geometry:
            d = dist.reshape(H, W, -1)[..., 0] / alpha.clamp_min(1e-6) * batch.rays_dir.reshape(H, W, 3)[..., 2]
            out["alpha"] = alpha
            out["depth"] = torch.where(alpha > 1e-4, d, torch.zeros_like(d))
        return out

    def compose(self, base: Frozen, lifted=None) -> Frozen:
        if lifted is None:
            return base
        lf = Frozen(lifted.xyz(), lifted.rotation(), lifted.scaling(), lifted.opacity(), lifted.features(self.F))
        return concat(base, lf)

    def render(self, view: View, base: Frozen, lifted=None, with_geometry: bool = True) -> dict:
        return self.trace(view, self.compose(base, lifted), geometry=with_geometry)

    @torch.no_grad()
    def coverage(self, view: View, lifted, base: Frozen) -> torch.Tensor:
        """(H, W) visible opacity of the lifted Gaussians alone, occlusion by ``base`` included."""
        g = self.compose(base, lifted)
        f = white_features(len(g), self.F, g.xyz.device)
        f[:len(base), :3] = -0.5 / C0
        return self.trace(view, g, f, geometry=False)["rgb"][0]

    @torch.no_grad()
    def alpha(self, view: View, g: Frozen) -> torch.Tensor:
        return self.trace(view, g, white_features(len(g), self.F, g.xyz.device), geometry=False)["rgb"][0]

    # ---- export
    def raw_params(self, keep=None) -> dict[str, torch.Tensor]:
        """Pre-activation checkpoint tensors of the full model (rows ``keep``)."""
        m = self.model
        out = {"positions": m.positions, "rotation": m.rotation, "scale": m.scale, "density": m.density,
               "features_albedo": m.features_albedo, "features_specular": m.features_specular}
        out = {k: v.detach() for k, v in out.items()}
        if keep is not None:
            keep = torch.as_tensor(keep, device=m.positions.device)
            out = {k: v[keep] for k, v in out.items()}
        return out

    def raw_from_frozen(self, g: Frozen) -> dict[str, torch.Tensor]:
        """Activated attributes -> pre-activation checkpoint tensors (inverse activations of the model)."""
        m = self.model
        dt = m.positions.dtype
        return {"positions": g.xyz.detach().to(dt), "rotation": g.rot.detach().to(dt),
                "scale": m.scale_activation_inv(g.scale.detach().clamp_min(1e-8)).to(dt),
                "density": m.density_activation_inv(g.opacity.detach().clamp(1e-4, 1 - 1e-4)).to(dt),
                "features_albedo": g.features[:, :3].detach().to(dt), "features_specular": g.features[:, 3:].detach().to(dt)}

    def save_checkpoint(self, parts: list[dict[str, torch.Tensor]], path: Path, extra: dict | None = None) -> int:
        """Concatenate raw parameter dicts into this model and save a loadable 3DGUT checkpoint
        (the layout aurafusion/finetune.py writes). Replaces the in-memory model's Gaussians."""
        m = self.model
        for name in ("positions", "rotation", "scale", "density", "features_albedo", "features_specular"):
            setattr(m, name, torch.nn.Parameter(torch.cat([p[name] for p in parts], 0).contiguous()))
        m.set_optimizable_parameters()
        m.setup_optimizer()
        m.validate_fields()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.get_model_parameters() | {"global_step": self.global_step, "epoch": 0} | (extra or {}), path)
        return int(m.positions.shape[0])


def to_u8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float tensor in [0, 1] -> (H, W, 3) uint8."""
    return (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def save_u8(a: np.ndarray, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(a).astype(np.uint8)).save(path)


def save_mask(m: np.ndarray, path: Path) -> None:
    save_u8((np.asarray(m, bool) * 255).astype(np.uint8), path)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def load_u8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def fov_x(view: View) -> float:
    return math.degrees(2 * math.atan(view.W / (2 * view.fx)))


def job_summary(work: Path, line: str) -> None:
    """One human-readable line to <work>/../job_summary.txt (the bakerh launcher copies its tail into
    REPORT.txt) and to stdout. The variant dir is the parent of the work dir (<variant>/sl)."""
    p = Path(work).parent / "job_summary.txt"
    with open(p, "a") as f:
        f.write(line.rstrip() + "\n")
    print("SUMMARY:", line, flush=True)


class Timer:
    """Per-stage wall time + peak GPU memory, appended to <work>/stages.txt."""

    def __init__(self, work: Path, name: str):
        import time
        self.work, self.name, self.t0 = Path(work), name, time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    def done(self, **info) -> dict:
        import time
        s = time.time() - self.t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
        line = {"stage": self.name, "seconds": round(s, 1), "peak_alloc_gib": round(peak, 2)} | info
        self.work.mkdir(parents=True, exist_ok=True)
        with open(self.work / "stages.txt", "a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        return line
