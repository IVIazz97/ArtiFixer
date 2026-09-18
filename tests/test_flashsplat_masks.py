# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from flashsplat.masks import infer_num_objects, load_object_id_mask, mask_path_for_frame


class ObjectIdMaskTest(unittest.TestCase):
    def test_preserves_raw_object_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "00000.png"
            Image.fromarray(np.array([[0, 3, 7], [7, 0, 3]], dtype=np.uint8)).save(path)
            ids = load_object_id_mask(path, actual_h=2, actual_w=3)

        # Ids must survive verbatim -- not normalised to 0/1, not remapped through a palette.
        self.assertEqual(ids.tolist(), [0, 3, 7, 7, 0, 3])
        self.assertEqual(ids.dtype.__str__(), "torch.int32")

    def test_nearest_resize_does_not_invent_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "00000.png"
            Image.fromarray(np.array([[0, 4], [4, 9]], dtype=np.uint8)).save(path)
            ids = load_object_id_mask(path, actual_h=6, actual_w=8)

        self.assertEqual(ids.shape[0], 48)
        # Bilinear resizing would blend 0 and 4 into 1, 2, 3 -- nearest must not.
        self.assertEqual(sorted(set(ids.tolist())), [0, 4, 9])

    def test_rgb_mask_uses_first_channel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "00000.png"
            rgb = np.zeros((2, 2, 3), dtype=np.uint8)
            rgb[..., 0] = np.array([[0, 5], [5, 0]], dtype=np.uint8)
            Image.fromarray(rgb).save(path)
            ids = load_object_id_mask(path, actual_h=2, actual_w=2)

        self.assertEqual(ids.tolist(), [0, 5, 5, 0])

    def test_mask_path_matches_on_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "frame_0007.png").write_bytes(b"")
            resolved = mask_path_for_frame(tmpdir, "/data/scene/images/frame_0007.JPG")
            self.assertEqual(resolved.name, "frame_0007.png")

    def test_missing_mask_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                mask_path_for_frame(tmpdir, "frame_0001.jpg")

    def test_infer_num_objects_takes_the_max_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Image.fromarray(np.array([[0, 2]], dtype=np.uint8)).save(Path(tmpdir) / "a.png")
            Image.fromarray(np.array([[0, 5]], dtype=np.uint8)).save(Path(tmpdir) / "b.png")
            self.assertEqual(infer_num_objects(tmpdir), 5)


if __name__ == "__main__":
    unittest.main()
