# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loading of per-view object-id masks for FlashSplat segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

MASK_SUFFIXES = (".png", ".PNG", ".npy")


def _load_mask_array(mask_path: str | Path) -> np.ndarray:
    """Load the raw mask as a 2D (or 3D RGB) array, ids untouched."""
    mask_path = Path(mask_path)
    if mask_path.suffix == ".npy":
        return np.load(mask_path)
    return np.array(Image.open(mask_path))


def load_object_id_mask(mask_path: str | Path, actual_h: int, actual_w: int) -> torch.Tensor:
    """Load a single-channel object-id mask as an int32 ``[actual_h * actual_w]`` tensor.

    Pixel values are object ids used verbatim: ``0`` is background, ``1..K`` are objects.

    Unlike ``threedgrut.datasets.dataset_colmap.load_mask_image_tensor`` this must *not*
    call ``Image.convert("L")`` (which remaps palette indices through the palette and would
    turn id 3 into some grey level) and must *not* divide by 255 or threshold. Resizing is
    nearest-neighbour only, for the same reason.
    """
    mask = _load_mask_array(mask_path)
    if mask.ndim == 3:  # RGB(A) masks: ids are replicated across channels
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"{mask_path}: expected a 2D mask, got shape {mask.shape}")

    if mask.shape != (actual_h, actual_w):
        # Nearest-neighbour only: any interpolation would invent ids between labels.
        mask = np.array(Image.fromarray(mask).resize((actual_w, actual_h), Image.NEAREST))

    return torch.from_numpy(mask.astype(np.int32)).reshape(-1)


def find_mask_path_for_frame(mask_dir: str | Path, image_name: str) -> Path | None:
    """Resolve the mask for a frame, matching on stem so mask and image extensions may differ.

    Returns ``None`` if no mask exists for this frame, for sparse mask sets where only a
    subset of views were annotated.
    """
    mask_dir = Path(mask_dir)
    stem = Path(image_name).stem
    for suffix in MASK_SUFFIXES:
        candidate = mask_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def mask_path_for_frame(mask_dir: str | Path, image_name: str) -> Path:
    """Like ``find_mask_path_for_frame``, but raises if the frame has no mask."""
    mask_path = find_mask_path_for_frame(mask_dir, image_name)
    if mask_path is None:
        raise FileNotFoundError(f"No mask for frame {image_name!r} in {mask_dir} (tried {MASK_SUFFIXES})")
    return mask_path


def infer_num_objects(mask_dir: str | Path) -> int:
    """Largest object id present across the mask directory (background excluded)."""
    mask_dir = Path(mask_dir)
    paths = sorted(p for p in mask_dir.iterdir() if p.suffix in MASK_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    max_id = 0
    for path in paths:
        mask = _load_mask_array(path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        max_id = max(max_id, int(mask.max()))
    return max_id
