# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3DGUT scene access shared by the AuraFusion stages that run in the ArtiFixer environment.

Uses the stock ``threedgut_tracer`` (never ``flashsplat.accumulate``, which replaces it
process-wide), so the same helpers serve both rendering and the gradient-based finetune.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "thirdparty" / "3DGRUT-ArtiFixer", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch
from PIL import Image


@dataclass
class ViewCamera:
    """Pinhole camera of one dataset view. ``c2w`` maps OpenCV camera coordinates to world."""

    index: int
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray

    def to_json(self) -> dict:
        return {**{k: getattr(self, k) for k in ("index", "name", "width", "height", "fx", "fy", "cx", "cy")},
                "c2w": self.c2w.tolist()}

    @staticmethod
    def from_json(d: dict) -> "ViewCamera":
        return ViewCamera(**{**d, "c2w": np.asarray(d["c2w"], dtype=np.float64)})


def load_cameras(path: Path) -> list[ViewCamera]:
    return [ViewCamera.from_json(d) for d in json.loads(Path(path).read_text())]


def save_cameras(cameras: list[ViewCamera], path: Path) -> None:
    Path(path).write_text(json.dumps([c.to_json() for c in cameras], indent=1))


def load_model(checkpoint_path: Path, config_overrides: dict | None = None, setup_optimizer: bool = False):
    """Load a 3DGUT checkpoint with the stock tracer. Returns (model, conf, checkpoint dict)."""
    from threedgrut.model.model import MixtureOfGaussians

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    conf = checkpoint["config"]
    for key, value in (config_overrides or {}).items():
        target = conf
        *parents, leaf = key.split(".")
        for part in parents:
            target = target[part]
        target[leaf] = value
    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=setup_optimizer)
    model.build_acc()
    return model, conf, checkpoint


def all_views_overrides(colmap_dir: Path) -> dict:
    """Config overrides that make the dataset's test split cover every COLMAP view, in COLMAP order.

    Prepared ArtiFixer checkpoints carry a selected_indices_file; with it the test split is only
    the held-out views. Same overrides as the FlashSplat stages use.
    """
    return {"path": str(colmap_dir), "dataset.test_split_interval": -1, "selected_indices_file": None}


def build_dataset(conf, shuffle: bool = False, num_workers: int = 4):
    from threedgrut import datasets
    from threedgrut.datasets.utils import configure_dataloader_for_platform

    dataset = datasets.make_test(name=conf.dataset.type, config=conf)
    loader = torch.utils.data.DataLoader(
        dataset,
        **configure_dataloader_for_platform(
            {"num_workers": num_workers, "batch_size": 1, "shuffle": shuffle, "collate_fn": None}
        ),
    )
    return dataset, loader


def view_camera(dataset, index: int, gpu_batch) -> ViewCamera:
    """Pinhole parameters of a view, read from the batch the renderer actually uses."""
    intrinsics = next(v for k, v in vars(gpu_batch).items() if k.startswith("intrinsics_") and v)
    focal = np.asarray(intrinsics["focal_length"], dtype=np.float64).reshape(-1)
    principal = np.asarray(intrinsics["principal_point"], dtype=np.float64).reshape(-1)
    for key in ("radial_coeffs", "tangential_coeffs", "thin_prism_coeffs"):
        if key in intrinsics:
            assert np.allclose(np.asarray(intrinsics[key], dtype=np.float64), 0.0, atol=1e-6), (
                f"view {index}: {key} is non-zero; the AuraFusion port assumes undistorted pinhole cameras"
            )
    height, width = gpu_batch.rays_dir.shape[1:3]
    c2w = gpu_batch.T_to_world[0].detach().cpu().numpy().astype(np.float64)
    if c2w.shape == (3, 4):
        c2w = np.vstack([c2w, [0, 0, 0, 1]])
    return ViewCamera(
        index=index,
        name=Path(str(dataset.image_paths[index])).name,
        width=int(width), height=int(height),
        fx=float(focal[0]), fy=float(focal[1]), cx=float(principal[0]), cy=float(principal[1]),
        c2w=c2w,
    )


@torch.no_grad()
def render_view(model, gpu_batch, frame_id: int) -> dict[str, torch.Tensor]:
    """Render RGB, opacity and camera-space z-depth (normalised by opacity) of one view.

    3DGUT's ``pred_dist`` is the front-to-back sum of weight * ray distance, i.e. not divided by
    the accumulated opacity; the rays are unit length in camera space, so z = t * dir_z.
    """
    outputs = model(gpu_batch, train=False, frame_id=frame_id)
    rgb = outputs["pred_rgb"][0].clamp(0, 1)
    opacity = outputs["pred_opacity"][0, ..., 0]
    distance = outputs["pred_dist"][0, ..., 0] / opacity.clamp_min(1e-6)
    depth_z = distance * gpu_batch.rays_dir[0, ..., 2]
    return {"rgb": rgb, "opacity": opacity, "depth": depth_z}


def pixel_grid(camera: ViewCamera, device) -> torch.Tensor:
    """[H, W, 2] continuous image coordinates of pixel centres (3DGRUT convention: index + 0.5)."""
    v, u = torch.meshgrid(
        torch.arange(camera.height, device=device, dtype=torch.float32),
        torch.arange(camera.width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([u + 0.5, v + 0.5], dim=-1)


def unproject(camera: ViewCamera, depth: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """World points of the (masked) pixels of ``camera`` at camera-space z ``depth`` [H, W]."""
    uv = pixel_grid(camera, depth.device)
    if mask is not None:
        uv, depth = uv[mask], depth[mask]
    else:
        uv, depth = uv.reshape(-1, 2), depth.reshape(-1)
    x = (uv[:, 0] - camera.cx) / camera.fx * depth
    y = (uv[:, 1] - camera.cy) / camera.fy * depth
    points_cam = torch.stack([x, y, depth], dim=-1)
    c2w = torch.as_tensor(camera.c2w, dtype=torch.float32, device=depth.device)
    return points_cam @ c2w[:3, :3].T + c2w[:3, 3]


def project(cameras: list[ViewCamera], points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world points [P, 3] into every camera. Returns pixel column/row indices [J, P] and z [J, P]."""
    w2c = torch.as_tensor(np.stack([np.linalg.inv(c.c2w) for c in cameras]), dtype=torch.float32, device=points.device)
    intr = torch.as_tensor([[c.fx, c.fy, c.cx, c.cy] for c in cameras], dtype=torch.float32, device=points.device)
    cam = torch.einsum("jab,pb->jpa", w2c[:, :3, :3], points) + w2c[:, None, :3, 3]
    z = cam[..., 2]
    safe_z = z.clamp_min(1e-6)
    u = cam[..., 0] / safe_z * intr[:, 0:1] + intr[:, 2:3]
    v = cam[..., 1] / safe_z * intr[:, 1:2] + intr[:, 3:4]
    return torch.floor(u).long(), torch.floor(v).long(), z


def save_png(array: np.ndarray | torch.Tensor, path: Path) -> None:
    """Save a [H,W] mask/opacity in [0,1] or bool, or an [H,W,3] image in [0,1], as 8-bit PNG."""
    if torch.is_tensor(array):
        array = array.detach().float().cpu().numpy()
    if array.dtype == bool:
        array = array.astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8)).save(path)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127
