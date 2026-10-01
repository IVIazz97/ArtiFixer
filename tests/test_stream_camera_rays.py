# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test of data_processing/stream_camera_rays.py against a stand-in ColmapDataset."""

import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class FakeRays:
    """A CPU ray map that records every copy to a device."""

    copies = []

    def __init__(self, name):
        self.name = name

    def to(self, device, non_blocking=False):
        FakeRays.copies.append((self.name, device))
        return f"{self.name}@{device}"


class FakeColmapDataset:
    """The stock behaviour: every camera's rays copied to the device on the first batch."""

    def __init__(self, n_cameras):
        self.device = "cuda"
        self._worker_gpu_cache = {}
        self.intrinsics = {i: ({"id": i}, FakeRays(f"ori{i}"), FakeRays(f"dir{i}"), "OpenCVPinholeCameraModelParameters")
                           for i in range(1, n_cameras + 1)}

    def _lazy_worker_intrinsics_cache(self):
        if "main" not in self._worker_gpu_cache:
            self._worker_gpu_cache["main"] = {i: (p, o.to(self.device), d.to(self.device), n)
                                              for i, (p, o, d, n) in self.intrinsics.items()}
        return self._worker_gpu_cache["main"]

    def get_gpu_batch_with_intrinsics(self, intr):  # the part of the stock method that reads the cache
        return self._lazy_worker_intrinsics_cache()[intr]


class StreamCameraRaysTest(unittest.TestCase):
    def setUp(self):
        FakeRays.copies = []
        colmap = types.ModuleType("threedgrut.datasets.dataset_colmap")
        colmap.ColmapDataset = type("ColmapDataset", (FakeColmapDataset,), {})
        utils = types.ModuleType("threedgrut.datasets.utils")
        utils.get_worker_id = lambda: "main"
        self.modules = {"threedgrut": types.ModuleType("threedgrut"),
                        "threedgrut.datasets": types.ModuleType("threedgrut.datasets"),
                        "threedgrut.datasets.dataset_colmap": colmap, "threedgrut.datasets.utils": utils}
        self.saved = {name: sys.modules.get(name) for name in self.modules}
        sys.modules.update(self.modules)
        self.Dataset = colmap.ColmapDataset

    def tearDown(self):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_stock_copies_every_camera(self):
        self.Dataset(1240).get_gpu_batch_with_intrinsics(1)
        self.assertEqual(len(FakeRays.copies), 2 * 1240)

    def test_only_the_asked_camera_is_copied_and_the_last_two_are_kept(self):
        from data_processing import stream_camera_rays

        stream_camera_rays.install()
        stream_camera_rays.install()  # idempotent
        dataset = self.Dataset(1240)
        params, ori, dirs, name = dataset.get_gpu_batch_with_intrinsics(7)
        self.assertEqual((params, ori, dirs), ({"id": 7}, "ori7@cuda", "dir7@cuda"))
        self.assertEqual(FakeRays.copies, [("ori7", "cuda"), ("dir7", "cuda")])
        for intr in (7, 7, 8, 7, 8):  # both kept: no new copies after the first of each
            dataset.get_gpu_batch_with_intrinsics(intr)
        self.assertEqual(len(FakeRays.copies), 4)
        dataset.get_gpu_batch_with_intrinsics(9)  # evicts 7, the least recently used
        dataset.get_gpu_batch_with_intrinsics(8)
        self.assertEqual(len(FakeRays.copies), 6)
        dataset.get_gpu_batch_with_intrinsics(7)
        self.assertEqual(len(FakeRays.copies), 8)
        self.assertLessEqual(len(dataset._worker_gpu_cache["main"].cached), stream_camera_rays.KEEP)

    def test_new_intrinsics_start_a_new_cache(self):
        from data_processing import stream_camera_rays

        stream_camera_rays.install()
        dataset = self.Dataset(3)
        dataset.get_gpu_batch_with_intrinsics(1)
        dataset.intrinsics = {1: ({"id": "low-res"}, FakeRays("ori1b"), FakeRays("dir1b"), "OpenCVPinholeCameraModelParameters")}
        self.assertEqual(dataset.get_gpu_batch_with_intrinsics(1)[1], "ori1b@cuda")


if __name__ == "__main__":
    unittest.main()
