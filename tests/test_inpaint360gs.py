# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the Inpaint360GS port's pure functions.

The pose tests compare against the official ``utils/pose_utils.py`` when a clone is found at
$INPAINT360GS_OFFICIAL (default /leonardo_scratch/fast/IscrC_EditGS/opt/Inpaint360GS_official).
"""

import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inpaint360gs import poses  # noqa: E402
from inpaint360gs.associate import KeyObjectDatabase, foreground_gaussians  # noqa: E402
from inpaint360gs.init_gaussians import statistical_outliers  # noqa: E402
from inpaint360gs.lama import erode, gaussian_blur  # noqa: E402
from inpaint360gs.nbs_masks import enlarge  # noqa: E402
from inpaint360gs.remove import hull_mask  # noqa: E402

OFFICIAL = Path(os.environ.get("INPAINT360GS_OFFICIAL", "/leonardo_scratch/fast/IscrC_EditGS/opt/Inpaint360GS_official"))


def look_at(position, target, up=np.array([0.0, -1.0, 0.0])):
    """OpenCV camera-to-world looking from ``position`` at ``target`` (world y down)."""
    z = target - position
    z = z / np.linalg.norm(z)
    x = np.cross(-up, z) if abs(np.dot(up, z)) < 0.99 else np.array([1.0, 0, 0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], axis=1)
    c2w[:3, 3] = position
    return c2w


def capture_rig(n=40, seed=0):
    """An object-centric 360 capture around a point off the origin, with jitter."""
    rng = np.random.default_rng(seed)
    target = np.array([0.3, 0.1, -0.2])
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.stack([target[0] + 3 * np.cos(angles), target[1] - 1.0 + 0.1 * rng.standard_normal(n),
                          target[2] + 2.5 * np.sin(angles)], axis=1)
    return np.stack([look_at(p, target + 0.05 * rng.standard_normal(3)) for p in positions]), target


class PoseTest(unittest.TestCase):
    def test_circle_looks_at_focus_with_orthonormal_rotations(self):
        c2ws, target = capture_rig()
        path = poses.circle_path(c2ws, n_frames=30, circle_radius=0.5)
        self.assertEqual(path.shape, (30, 4, 4))
        for pose in path:
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.0, places=9)
        # every optical axis passes (nearly) through the capture's focus point
        to_target = target - path[:, :3, 3]
        cos = np.einsum("nd,nd->n", path[:, :3, 2], to_target / np.linalg.norm(to_target, axis=1, keepdims=True))
        self.assertGreater(cos.min(), 0.99)

    def test_virtual_radius_frames_object(self):
        c2ws, _ = capture_rig()
        near = poses.virtual_radius(c2ws, 1.0, 0.8, object_radius=0.1)
        far = poses.virtual_radius(c2ws, 1.0, 0.8, object_radius=0.4)
        self.assertAlmostEqual(far / near, 4.0, places=6)  # linear in the object radius

    @unittest.skipUnless((OFFICIAL / "utils" / "pose_utils.py").exists(), "official clone not found")
    def test_matches_official(self):
        sys.path.insert(0, str(OFFICIAL))
        try:
            from utils import pose_utils
        finally:
            sys.path.remove(str(OFFICIAL))
        c2ws, _ = capture_rig()
        views = []
        for c2w in c2ws:  # 3DGS convention: R = c2w rotation, T = w2c translation
            w2c = np.linalg.inv(c2w)
            views.append(types.SimpleNamespace(R=c2w[:3, :3], T=w2c[:3, 3], FoVx=1.0, FoVy=0.8))
        ratio = pose_utils.generate_virtual_radius(views, target_object_radius=0.2)
        self.assertAlmostEqual(poses.virtual_radius(c2ws, 1.0, 0.8, 0.2), ratio, places=9)
        official = [np.linalg.inv(w2c) for w2c in
                    pose_utils.generate_ellipse_path(views, n_frames=30, is_circle=True, circle_radius=ratio)]
        ours = poses.circle_path(c2ws, n_frames=30, circle_radius=ratio)
        for theirs, mine in zip(official, ours):
            np.testing.assert_allclose(mine[:3, 3], theirs[:3, 3], atol=1e-6)  # camera centres
            axes = theirs[:3, :3] / np.linalg.norm(theirs[:3, :3], axis=0)   # official keeps the PCA scale
            np.testing.assert_allclose(mine[:3, :3], axes, atol=1e-6)


class AssociationTest(unittest.TestCase):
    def test_two_means_split_is_optimal(self):
        rng = np.random.default_rng(1)
        depth = np.concatenate([rng.normal(1.0, 0.05, 30), rng.normal(3.0, 0.2, 70)]).astype(np.float32)
        height = width = 16
        u = torch.zeros(100, dtype=torch.long)
        v = torch.zeros(100, dtype=torch.long)
        ids = torch.ones(height, width, dtype=torch.long)
        chosen, owner = foreground_gaussians(u, v, torch.from_numpy(depth), ids, height, width, patches=1)
        # brute force: best split of the sorted depths
        d = np.sort(depth.astype(np.float64))
        sse = [((d[:s] - d[:s].mean()) ** 2).sum() + ((d[s:] - d[s:].mean()) ** 2).sum() for s in range(1, 100)]
        split = int(np.argmin(sse)) + 1
        self.assertEqual(split, 30)
        self.assertEqual(len(chosen), int(split * 0.3))
        np.testing.assert_allclose(np.sort(depth[chosen.numpy()]), d[: int(split * 0.3)])
        self.assertTrue(bool((owner == 1).all()))

    def test_groups_per_mask_and_patch(self):
        # two masks side by side, each with a near and a far layer; a lone Gaussian is kept
        height, width = 8, 8
        ids = torch.zeros(height, width, dtype=torch.long)
        ids[:, :4], ids[:, 4:] = 1, 2
        u = torch.tensor([0] * 10 + [7] * 10 + [2])
        v = torch.tensor([0] * 10 + [0] * 10 + [7])
        z = torch.tensor([1.0] * 5 + [5.0] * 5 + [2.0] * 5 + [9.0] * 5 + [4.0])
        chosen, owner = foreground_gaussians(u, v, z, ids, height, width, patches=2)
        self.assertEqual(sorted(chosen.tolist()), [0, 10, 20])  # int(5 * 0.3) = 1 per two-layer patch
        self.assertEqual(owner[chosen.argsort()].tolist(), [1, 2, 1])

    def test_database_matching(self):
        db = KeyObjectDatabase(100, "cpu")
        self.assertEqual(db.add(torch.arange(0, 20)), 0)
        self.assertEqual(db.add(torch.arange(50, 60)), 1)
        # overlaps object 0 heavily: joins it, contributing only unassigned Gaussians
        self.assertEqual(db.match(torch.arange(10, 30)), 0)
        self.assertEqual(db.entries[0].tolist(), list(range(30)))
        # disjoint: a new object
        self.assertEqual(db.match(torch.arange(80, 90)), 2)
        # an empty mask never matches
        self.assertEqual(db.match(torch.zeros(0, dtype=torch.long)), 3)


class MaskAndFilterTest(unittest.TestCase):
    def test_enlarge_drops_specks_and_dilates(self):
        mask = np.zeros((60, 60), dtype=bool)
        mask[20:30, 20:30] = True  # 100 px, kept
        mask[50, 50] = True        # 1 px, dropped
        out = enlarge(mask, expand_pixels=3)
        self.assertFalse(out[50, 50])
        self.assertTrue(out[17:33, 17:33].all())
        self.assertFalse(out[16, 20])

    def test_enlarge_keeps_largest_when_all_small(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:4, 2:4] = True
        mask[10:15, 10:15] = True
        out = enlarge(mask, expand_pixels=0)
        self.assertTrue(out[12, 12])
        self.assertFalse(out[2, 2])

    def test_hull_expand_scales_about_the_centre(self):
        """The unit cube's hull, scaled by 2 about its centre, spans [-0.5, 1.5]."""
        corners = torch.tensor([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
        probes = torch.tensor([[0.5, 0.5, 0.5], [1.2, 0.5, 0.5], [1.45, 1.45, 1.45], [1.7, 0.5, 0.5]])
        positions = torch.cat([corners, probes])
        is_corner = torch.zeros(len(positions), dtype=torch.bool)
        is_corner[:len(corners)] = True

        tight, radius = hull_mask(positions, is_corner)
        self.assertEqual(tight[len(corners):].tolist(), [True, False, False, False])
        self.assertAlmostEqual(radius, float(np.percentile(np.full(8, np.sqrt(0.75)), 80)), places=6)

        wide, wide_radius = hull_mask(positions, is_corner, expand=2.0)
        self.assertEqual(wide[len(corners):].tolist(), [True, True, True, False])
        self.assertEqual(wide[:len(corners)].tolist(), [True] * 8)
        self.assertAlmostEqual(wide_radius, radius, places=12)  # the radius is the object's own

    def test_statistical_outliers(self):
        rng = np.random.default_rng(0)
        points = np.concatenate([rng.normal(0, 0.01, (500, 3)), [[5.0, 5.0, 5.0]]])
        keep = statistical_outliers(points)
        self.assertFalse(keep[-1])
        self.assertGreater(keep[:-1].mean(), 0.99)

    def test_blur_preserves_constants_and_erode_shrinks(self):
        x = torch.full((1, 3, 16, 16), 0.7)
        torch.testing.assert_close(gaussian_blur(x), x)
        mask = torch.zeros(1, 1, 16, 16)
        mask[..., 4:12, 4:12] = 1
        kernel = torch.ones(3, 3)
        eroded = erode(mask, kernel)
        self.assertEqual(int(eroded.sum()), 36)
        full = erode(torch.ones(1, 1, 8, 8), kernel)  # geodesic border: the image edge does not erode
        self.assertEqual(int(full.sum()), 64)

    def test_blur_matches_kornia(self):
        try:
            from kornia.filters import gaussian_blur2d
        except ImportError:
            self.skipTest("kornia not installed")
        x = torch.rand(1, 3, 20, 24)
        torch.testing.assert_close(gaussian_blur(x), gaussian_blur2d(x, (5, 5), (1.0, 1.0)), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
