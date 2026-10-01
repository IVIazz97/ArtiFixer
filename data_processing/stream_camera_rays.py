# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy camera rays to the GPU one camera at a time, as the images already are.

The stock ColmapDataset copies the ray maps (rays_ori, rays_dir: [1, H, W, 3] float32 each) of every
COLMAP camera to the GPU at the first batch and keeps them there. With one camera per frame that is
36 MiB per frame at 1600x900, e.g. 43 GiB for 1240 frames. ``install()`` makes the per-worker cache
copy only the camera a batch asks for and keep the last ``KEEP`` of them, so frames sharing a camera
still cost one copy in total. Applied in-process, since thirdparty/3DGRUT-ArtiFixer is upstream code.
"""

from __future__ import annotations

from collections import OrderedDict

KEEP = 2


class StreamedRays:
    """intr_id -> (params, rays_ori, rays_dir, camera_name) on ``device``, copying a camera's rays from
    the CPU ``intrinsics`` only when it is asked for and keeping the ``keep`` most recently used."""

    def __init__(self, intrinsics, device, keep=KEEP):
        self.intrinsics, self.device, self.keep = intrinsics, device, keep
        self.cached = OrderedDict()

    def __getitem__(self, intr_id):
        if intr_id in self.cached:
            self.cached.move_to_end(intr_id)
        else:
            params, rays_ori, rays_dir, name = self.intrinsics[intr_id]
            self.cached[intr_id] = (params, rays_ori.to(self.device, non_blocking=True),
                                    rays_dir.to(self.device, non_blocking=True), name)
            while len(self.cached) > self.keep:
                self.cached.popitem(last=False)  # a batch built from it still holds its tensors
        return self.cached[intr_id]

    def __len__(self):
        return len(self.intrinsics)


def install() -> None:
    """Patch ColmapDataset (idempotent)."""
    from threedgrut.datasets.dataset_colmap import ColmapDataset
    from threedgrut.datasets.utils import get_worker_id

    if getattr(ColmapDataset, "_streams_camera_rays", False):
        return

    def lazy_worker_intrinsics_cache(self):
        worker_id = get_worker_id()
        cache = self._worker_gpu_cache.get(worker_id)
        if cache is None or cache.intrinsics is not self.intrinsics:
            cache = self._worker_gpu_cache[worker_id] = StreamedRays(self.intrinsics, self.device)
        return cache

    ColmapDataset._lazy_worker_intrinsics_cache = lazy_worker_intrinsics_cache
    ColmapDataset._streams_camera_rays = True
