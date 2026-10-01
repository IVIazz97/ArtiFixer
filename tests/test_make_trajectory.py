# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of scripts/bakerh/make_trajectory.py.

The scene is an analytic box room (walls, floor and ceiling; optional pillar and doorway), both as
a ray-cast renderer standing in for the 3DGUT render checks and as tiled flat Gaussians for the
occupancy checks. World up is +y, cameras are OpenGL camera-to-world as in transforms.json.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
_spec = importlib.util.spec_from_file_location("make_trajectory", REPO_ROOT / "scripts/bakerh/make_trajectory.py")
mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mt)

UP = np.array([0.0, 1.0, 0.0])
INTR = (48.0, 48.0, 32.0, 24.0, 64, 48)  # fx, fy, cx, cy, w, h of the check camera


def look_at(position, target):
    return np.r_[np.c_[mt.look_at(np.asarray(position, float), np.asarray(target, float), UP), position], [[0, 0, 0, 1]]]


class Room:
    """Axis-aligned room [x0, x1] x [0, height] x [z0, z1], an optional vertical pillar (x, z, r) and
    an optional doorway (z-range x y-range) in the wall x = x1."""

    def __init__(self, x=(-3, 3), z=(-2, 2), height=3.0, pillar=None, doorway=None):
        self.x, self.z, self.height, self.pillar, self.doorway = x, z, height, pillar, doorway

    def render(self, c2w_gl):
        fx, fy, cx, cy, w, h = INTR
        v, u = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")
        rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1).reshape(-1, 3)  # OpenCV, z = 1
        depths, opacities = [], []
        for pose in np.asarray(c2w_gl):
            c2w = pose @ mt.FLIP
            o, d = c2w[:3, 3], rays @ c2w[:3, :3].T
            t = np.full(len(d), np.inf)
            for axis, value, (a, b), bounds in (
                (0, self.x[0], (1, 2), ((0, self.height), self.z)), (0, self.x[1], (1, 2), ((0, self.height), self.z)),
                (2, self.z[0], (0, 1), (self.x, (0, self.height))), (2, self.z[1], (0, 1), (self.x, (0, self.height))),
                (1, 0.0, (0, 2), (self.x, self.z)), (1, self.height, (0, 2), (self.x, self.z)),
            ):
                with np.errstate(divide="ignore", invalid="ignore"):
                    tt = (value - o[axis]) / d[:, axis]
                p = o + tt[:, None] * d
                hit = (tt > 1e-9) & (p[:, a] >= bounds[0][0]) & (p[:, a] <= bounds[0][1]) \
                    & (p[:, b] >= bounds[1][0]) & (p[:, b] <= bounds[1][1])
                if axis == 0 and value == self.x[1] and self.doorway is not None:
                    (z0, z1), (y0, y1) = self.doorway
                    hit &= ~((p[:, 2] > z0) & (p[:, 2] < z1) & (p[:, 1] > y0) & (p[:, 1] < y1))
                t = np.where(hit & (tt < t), tt, t)
            if self.pillar is not None:
                px, pz, r = self.pillar
                ox, oz = o[0] - px, o[2] - pz
                a2 = d[:, 0] ** 2 + d[:, 2] ** 2
                b2 = 2 * (ox * d[:, 0] + oz * d[:, 2])
                disc = b2**2 - 4 * a2 * (ox**2 + oz**2 - r**2)
                with np.errstate(invalid="ignore", divide="ignore"):
                    tt = (-b2 - np.sqrt(disc)) / (2 * a2)
                y = o[1] + tt * d[:, 1]
                hit = (disc > 0) & (tt > 1e-9) & (y > 0) & (y < self.height)
                t = np.where(hit & (tt < t), tt, t)
            found = np.isfinite(t)
            depths.append(np.where(found, t, 0.0).reshape(h, w))  # z-depth: the rays have z = 1
            opacities.append(found.astype(float).reshape(h, w))
        return torch.tensor(np.array(depths)), torch.tensor(np.array(opacities))

    def gaussians(self, step=0.05, blob=None):
        """Flat Gaussians tiling every surface (and an isotropic blob box (centre, half) if given)."""
        mu, scale = [], []
        xs, zs = np.arange(self.x[0], self.x[1] + 1e-9, step), np.arange(self.z[0], self.z[1] + 1e-9, step)
        ys = np.arange(0, self.height + 1e-9, step)
        for value in self.x:
            g = np.stack(np.meshgrid([value], ys, zs, indexing="ij"), -1).reshape(-1, 3)
            if value == self.x[1] and self.doorway is not None:
                (z0, z1), (y0, y1) = self.doorway
                g = g[~((g[:, 2] > z0) & (g[:, 2] < z1) & (g[:, 1] > y0) & (g[:, 1] < y1))]
            mu.append(g)
            scale.append(np.tile([0.002, step, step], (len(g), 1)))
        for value in self.z:
            g = np.stack(np.meshgrid(xs, ys, [value], indexing="ij"), -1).reshape(-1, 3)
            mu.append(g)
            scale.append(np.tile([step, step, 0.002], (len(g), 1)))
        for value in (0.0, self.height):
            g = np.stack(np.meshgrid(xs, [value], zs, indexing="ij"), -1).reshape(-1, 3)
            mu.append(g)
            scale.append(np.tile([step, 0.002, step], (len(g), 1)))
        if self.pillar is not None:
            px, pz, r = self.pillar
            a, y = np.meshgrid(np.arange(0, 2 * np.pi, step / r), ys)
            g = np.stack([px + r * np.cos(a), y, pz + r * np.sin(a)], -1).reshape(-1, 3)
            mu.append(g)
            scale.append(np.full((len(g), 3), step))
        n_scene = sum(len(m) for m in mu)
        if blob is not None:
            centre, half = blob
            g = np.random.default_rng(0).uniform(-1, 1, (6000, 3)) * half + centre
            mu.append(g)
            scale.append(np.full((len(g), 3), 0.04))
        mu, scale = np.concatenate(mu), np.concatenate(scale)
        return mu, scale, n_scene


def write_scene(folder, room, photos_c2w, blob=None):
    """transforms.json (+ checkpoint, and FlashSplat files for the blob) in ``folder``."""
    folder = Path(folder)
    json.dump({"camera_model": "OPENCV", "w": 64, "h": 48, "fl_x": 48, "fl_y": 48, "cx": 32, "cy": 24,
               "frames": [{"file_path": f"images/{k:04d}.jpg", "transform_matrix": m.tolist()} for k, m in enumerate(photos_c2w)]},
              open(folder / "transforms.json", "w"))
    mu, scale, n_scene = room.gaussians(blob=blob)
    n = len(mu)
    torch.save({"positions": torch.tensor(mu).float(), "rotation": torch.tensor(np.tile([1.0, 0, 0, 0], (n, 1))).float(),
                "scale": torch.tensor(np.log(scale)).float(), "density": torch.full((n, 1), float(np.log(0.95 / 0.05))),
                "config": SimpleNamespace(model=SimpleNamespace(density_activation="sigmoid", scale_activation="exp"))},
               folder / "ckpt.pt")
    if blob is not None:
        fs = folder / "flashsplat"
        fs.mkdir(exist_ok=True)
        is_obj = np.arange(n) >= n_scene
        torch.save(torch.from_numpy(np.stack([~is_obj, is_obj])), fs / "labels.pt")
        torch.save(torch.ones(2, n), fs / "contribution.pt")
        torch.save(torch.ones(n, dtype=torch.int32), fs / "hit_count.pt")
    return folder


def ring(n, radius_x, radius_z, target, height=1.5, arc=(0, 2 * np.pi)):
    angles = np.linspace(arc[0], arc[1], n, endpoint=arc[1] - arc[0] < 2 * np.pi - 1e-9)
    return np.array([look_at([radius_x * np.cos(a), height, radius_z * np.sin(a)], target) for a in angles])


def head_travel(transforms, side=2.0, up=0.5, periods=2.0, sigma=2.0, jump=4.0):
    """The travel algorithm as committed at HEAD (scripts/bakerh/make_trajectory.py before the modes)."""
    frames = sorted(transforms["frames"], key=lambda f: f["file_path"])
    c2w = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
    n = len(c2w)
    pos = c2w[:, :3, 3]
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    spacing = float(np.median(steps))
    segments = np.split(np.arange(n), np.where(steps > jump * spacing)[0] + 1)
    path = []
    for seg in segments:
        rot = Rotation.from_matrix(c2w[seg, :3, :3])
        smooth = []
        for i in seg:
            w = np.exp(-0.5 * ((seg - i) / sigma) ** 2)
            w /= w.sum()
            m = np.eye(4)
            m[:3, :3] = rot.mean(weights=w).as_matrix()
            m[:3, 3] = (w[:, None] * pos[seg]).sum(0)
            smooth.append(m)
        smooth = np.stack(smooth)
        travel = np.gradient(smooth[:, :3, 3], axis=0) if len(seg) > 1 else np.zeros((1, 3))
        for m, i, t in zip(smooth, seg, travel):
            cam_up = m[:3, 1]
            across = np.cross(t, cam_up)
            across = across / np.linalg.norm(across) if np.linalg.norm(across) > 1e-9 * spacing else m[:3, 0]
            m[:3, 3] += side * spacing * np.sin(2 * np.pi * periods * i / (n - 1)) * across + up * spacing * cam_up
            path.append(m)
    keys = ("camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2")
    out = {k: transforms[k] for k in keys if k in transforms}
    out["frames"] = [{"transform_matrix": m.tolist()} for m in np.stack(path)]
    return json.dumps(out, indent=2) + "\n"


class TravelModeTest(unittest.TestCase):
    def test_bit_identical_to_head_without_checkpoint(self):
        photos = ring(40, 1.6, 1.2, [0, 0.7, 0])
        with tempfile.TemporaryDirectory() as tmp:
            write_scene(tmp, Room(), photos)
            transforms = json.load(open(Path(tmp) / "transforms.json"))
            np.random.default_rng(1).shuffle(transforms["frames"])  # file order, not list order
            json.dump(transforms, open(Path(tmp) / "transforms.json", "w"))
            expected = head_travel(transforms)
            for extra in ([], ["--mode", "travel"]):
                out = Path(tmp) / f"traj{len(extra)}.json"
                self.assertEqual(mt.main(["--transforms", f"{tmp}/transforms.json", "--output", str(out)] + extra), 0)
                self.assertEqual(out.read_text(), expected)


    def test_max_frames_keeps_every_kth_frame(self):
        photos = ring(40, 1.6, 1.2, [0, 0.7, 0])
        with tempfile.TemporaryDirectory() as tmp:
            write_scene(tmp, Room(), photos)
            full, capped = Path(tmp) / "full.json", Path(tmp) / "capped.json"
            self.assertEqual(mt.main(["--transforms", f"{tmp}/transforms.json", "--output", str(full)]), 0)
            self.assertEqual(mt.main(["--transforms", f"{tmp}/transforms.json", "--output", str(capped), "--max_frames", "15"]), 0)
            all_frames = json.load(open(full))["frames"]
            self.assertEqual(json.load(open(capped))["frames"], all_frames[::3])
            self.assertEqual(json.load(open(capped.with_name("capped_info.json")))["frame_stride"], 3)

    def test_a_gap_in_the_photos_makes_two_pieces(self):
        photos = np.concatenate([ring(20, 1.6, 1.2, [0, 0.7, 0], arc=(0, 0.5 * np.pi)),
                                 ring(20, 1.6, 1.2, [0, 0.7, 0], arc=(np.pi, 1.5 * np.pi))])  # a jump after photo 19
        with tempfile.TemporaryDirectory() as tmp:
            write_scene(tmp, Room(), photos)
            for extra, pieces in (([], [[0, 19], [20, 39]]), (["--max_frames", "15"], [[0, 6], [7, 13]])):
                out = Path(tmp) / "traj.json"
                self.assertEqual(mt.main(["--transforms", f"{tmp}/transforms.json", "--output", str(out)] + extra), 0)
                info = json.load(open(out.with_name("traj_info.json")))
                self.assertEqual(info["pieces"], pieces)
                self.assertEqual(len(info["jumps_spacings"]), 1)
                self.assertGreater(info["jumps_spacings"][0], 4)


    def test_phase_offset_shifts_the_weave(self):
        photos = ring(40, 1.6, 1.2, [0, 0.7, 0])
        with tempfile.TemporaryDirectory() as tmp:
            write_scene(tmp, Room(), photos)
            paths = {}
            for offset in ("0", "0.5", "1"):
                out = Path(tmp) / f"traj{offset}.json"
                self.assertEqual(mt.main(["--transforms", f"{tmp}/transforms.json", "--output", str(out), "--phase_offset", offset]), 0)
                paths[offset] = np.array([f["transform_matrix"] for f in json.load(open(out))["frames"]])
            self.assertGreater(np.abs(paths["0.5"] - paths["0"]).max(), 0.01)
            np.testing.assert_allclose(paths["1"], paths["0"], atol=1e-9)  # a whole period later: the same path


class OrbitPoseTest(unittest.TestCase):
    def test_anchor_stays_centred_and_level(self):
        anchor = np.array([0.5, 0.3, -0.2])
        photo = look_at([2.0, 1.4, 1.0], anchor)
        for yaw, pitch in ((0.4, 0.0), (-0.3, 0.25), (0.2, -0.5)):
            m = mt.orbit_pose(photo, anchor, yaw, pitch, UP, np.radians(60))
            local = m[:3, :3].T @ (anchor - m[:3, 3])
            np.testing.assert_allclose(local[:2], 0, atol=1e-9)  # on the optical axis
            self.assertLess(local[2], 0)  # in front (OpenGL looks along -z)
            self.assertAlmostEqual(np.linalg.norm(m[:3, 3] - anchor), np.linalg.norm(photo[:3, 3] - anchor), places=9)
            np.testing.assert_allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(m[:3, :3]), 1.0, places=9)
            self.assertAlmostEqual(m[:3, 0] @ UP, 0.0, places=9)  # no roll: the x axis stays horizontal

    def test_elevation_is_capped(self):
        anchor = np.zeros(3)
        m = mt.orbit_pose(look_at([2.0, 0.0, 0.0], anchor), anchor, 0.0, np.radians(80), UP, np.radians(60))
        elevation = np.degrees(np.arcsin(m[1, 3] / np.linalg.norm(m[:3, 3])))
        self.assertAlmostEqual(elevation, 60.0, places=6)


class ViewCheckerTest(unittest.TestCase):
    def checker(self, room, photos, tol=0.05):
        return mt.ViewChecker(room.render, photos, INTR, depth_tol=tol)

    def test_photo_pose_is_seen_and_unphotographed_ceiling_is_not(self):
        room = Room()
        photos = ring(24, 1.6, 1.2, [0, 0.7, 0])
        checker = self.checker(room, photos)
        stats = checker.measure(photos[:3], 0.5)
        self.assertTrue((stats["unseen"] < 0.01).all(), stats)
        ceiling = look_at([0, 1.5, 0], [0.0, 3.0, 0.01])
        self.assertGreater(checker.measure(ceiling[None], 0.5)["unseen"][0], 0.9)

    def test_occlusion_behind_a_pillar_is_unseen(self):
        room = Room(pillar=(0.0, 0.0, 0.35))
        photo = look_at([-2.5, 1.5, 0.0], [3.0, 1.5, 0.0])
        view = look_at([-2.5, 1.5, 1.2], [3.0, 1.5, 0.0])  # sees the wall behind the pillar
        with_depth = self.checker(room, photo[None]).measure(view[None], 0.5)["unseen"][0]
        without_depth = self.checker(room, photo[None], tol=1e9).measure(view[None], 0.5)["unseen"][0]
        self.assertGreater(with_depth, without_depth + 0.02)

    def test_near_wall_and_doorway(self):
        room = Room(doorway=((-0.5, 0.5), (0.0, 2.0)))
        photo = look_at([-1.0, 1.0, 0.0], [3.0, 1.0, 0.0])
        checker = self.checker(room, photo[None])
        near = look_at([2.8, 1.5, 1.2], [3.5, 1.5, 1.2])  # 0.2 from the wall, facing it
        self.assertGreater(checker.measure(near[None], 0.5, unseen=False)["near_frac"][0], 0.95)
        stats = checker.measure(photo[None], 0.5)
        self.assertGreater(stats["empty"][0], 0.05)  # the doorway: no geometry
        self.assertLess(stats["unseen"][0], 0.01)  # ...which never counts as unseen

    def test_anchor_on_central_ray(self):
        room = Room()
        photo = look_at([-1.0, 1.5, 0.0], [3.0, 0.5, 0.0])
        anchor = self.checker(room, photo[None]).anchor(0, photo)
        local = photo[:3, :3].T @ (anchor - photo[:3, 3])
        np.testing.assert_allclose(local[:2], 0, atol=1e-6)
        self.assertLess(abs(anchor[0] - 3.0), 0.1)  # on the far wall, near where the ray meets it


class VidsplatModeTest(unittest.TestCase):
    ROOM = Room(x=(-4, 4), z=(-3, 3), height=3.0)

    def run_mode(self, tmp, extra=()):
        photos = ring(48, 1.5, 1.2, [3.5, 1.0, 0.0], arc=(0.5 * np.pi, 1.5 * np.pi))  # half ring facing +x
        write_scene(tmp, self.ROOM, photos)
        out = Path(tmp) / "traj.json"
        argv = ["--transforms", f"{tmp}/transforms.json", "--output", str(out), "--mode", "vidsplat",
                "--checkpoint", f"{tmp}/ckpt.pt", "--clearance", "0.3", "--metric_scale", "1.0"] + list(extra)
        code = mt.main(argv, renderer=(self.ROOM.render, INTR))
        info = json.loads(out.with_name("traj_info.json").read_text())
        return code, out, info, photos

    def test_valid_orbits_layout_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, info, photos = self.run_mode(tmp, ["--budget", "97"])
            self.assertEqual(code, 0, info.get("failed"))
            frames = np.array([f["transform_matrix"] for f in json.load(open(out))["frames"]])
            L, clips = info["clip_len"], info["clips"]
            self.assertGreaterEqual(len(clips), info["min_clips"])
            self.assertEqual(len(frames), 1 + L * len(clips))
            self.assertEqual([c["seed_index"] for c in clips], sorted(c["seed_index"] for c in clips))
            np.testing.assert_allclose(frames[0], photos[clips[0]["seed_index"]])
            self.assertEqual(info["pieces"], [[0, L]] + [[1 + L * c, L * (c + 1)] for c in range(1, len(clips))])
            self.assertEqual(len(info["jumps_spacings"]), len(clips) - 1)
            for c_i, clip in enumerate(clips):
                start = 1 + L * c_i
                self.assertEqual(start % 4, 1)  # never shares a 4-frame VAE group with the previous clip
                seg = frames[start:start + L]
                np.testing.assert_allclose(np.linalg.norm(seg[:, :3, 3] - clip["anchor"], axis=1), clip["anchor_distance"], rtol=1e-6)
                k = clip["keyframes"]
                self.assertTrue(max(k["unseen"]) <= 0.4 and max(k["near_frac"]) <= 0.02, k)
                inside = (np.abs(seg[:, 0, 3]) < 4 - 0.3) & (np.abs(seg[:, 2, 3]) < 3 - 0.3) \
                    & (seg[:, 1, 3] > 0.3) & (seg[:, 1, 3] < 3 - 0.3)
                self.assertTrue(inside.all(), f"clip {c_i} leaves the room or touches a wall")
            again = Path(tmp) / "again.json"
            argv = ["--transforms", f"{tmp}/transforms.json", "--output", str(again), "--mode", "vidsplat",
                    "--checkpoint", f"{tmp}/ckpt.pt", "--clearance", "0.3", "--metric_scale", "1.0", "--budget", "97"]
            self.assertEqual(mt.main(argv, renderer=(self.ROOM.render, INTR)), 0)
            self.assertEqual(again.read_text(), out.read_text())

    def test_seen_views_count_as_observed(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, info, photos = self.run_mode(tmp, ["--budget", "97"])
            self.assertEqual(code, 0)
            first = {(c["seed_index"], c["direction"], c["extent_deg"]) for c in info["clips"]}
            again = Path(tmp) / "round2.json"
            argv = ["--transforms", f"{tmp}/transforms.json", "--output", str(again), "--mode", "vidsplat",
                    "--checkpoint", f"{tmp}/ckpt.pt", "--clearance", "0.3", "--metric_scale", "1.0", "--budget", "97",
                    "--seen_transforms", str(out), "--min_clips", "0"]
            self.assertEqual(mt.main(argv, renderer=(self.ROOM.render, INTR)), 0)
            info2 = json.loads(again.with_name("round2_info.json").read_text())
            second = {(c["seed_index"], c["direction"], c["extent_deg"]) for c in info2["clips"]}
            self.assertFalse(first & second, "round 2 repeated an orbit whose views are already seen")
            self.assertGreater(info2["rejected"]["nothing_new"], info["rejected"]["nothing_new"])

    def test_seeds_spread_over_the_capture_and_between_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            rounds = []
            for k in range(3):  # a small scene's loop: one orbit per round, phase (k-1)/3 of a seed spacing
                code, out, info, photos = self.run_mode(tmp, ["--budget", "17", "--phase_offset", str(k / 3)])
                self.assertEqual(code, 0, info.get("failed"))
                self.assertEqual(info["clips_wanted"], 1)
                seeds = info["seed_indices"]
                self.assertEqual(len(seeds), 12)
                gaps = np.diff(seeds + [seeds[0] + len(photos)])
                self.assertLessEqual(gaps.max() - gaps.min(), 1, seeds)  # evenly over the whole capture
                self.assertIn(info["clips"][0]["seed_index"], info["valid_seeds"])
                rounds.append(set(seeds))
            for a in range(3):
                for b in range(a + 1, 3):
                    self.assertFalse(rounds[a] & rounds[b], f"rounds {a + 1} and {b + 1} try the same seed photos")

    def test_one_orbit_per_stretch_of_the_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, info, photos = self.run_mode(tmp, ["--budget", "49", "--seeds", "24"])
            self.assertEqual(code, 0, info.get("failed"))
            n, wanted = len(photos), info["clips_wanted"]
            self.assertEqual((wanted, info["seeds_tried"]), (3, 24))
            stretches = {min(s * wanted // n, wanted - 1) for s in info["valid_seeds"]}
            got = [min(c["seed_index"] * wanted // n, wanted - 1) for c in info["clips"]]
            self.assertEqual(sorted(set(got)), sorted(stretches)[:len(info["clips"])], got)

    def test_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, info, _ = self.run_mode(tmp, ["--s_high", "0.001"])
            self.assertEqual(code, 2)
            self.assertIn("valid VidSplat orbits", info["failed"])
            self.assertGreater(info["rejected"]["too_unseen"], 0)
            self.assertFalse(out.exists())
            self.assertTrue(out.with_suffix(".png").exists() or importlib.util.find_spec("matplotlib") is None)

    def test_clip_len_must_be_multiple_of_4(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self.run_mode(tmp, ["--clip_len", "10"])


class ObjectModeTest(unittest.TestCase):
    BLOB = (np.array([0.0, 0.5, 0.0]), np.array([0.5, 0.5, 0.4]))

    def run_mode(self, tmp, room, photos, extra=()):
        write_scene(tmp, room, photos, blob=self.BLOB)
        out = Path(tmp) / "traj.json"
        argv = ["--transforms", f"{tmp}/transforms.json", "--output", str(out), "--checkpoint", f"{tmp}/ckpt.pt",
                "--flashsplat_dir", f"{tmp}/flashsplat", "--clearance", "0.25"] + list(extra)
        code = mt.main(argv)
        return code, out, json.loads(out.with_name("traj_info.json").read_text())

    def test_roomy_room_passes_with_a_photo_near_a_wall(self):
        photos = ring(60, 2.75, 1.8, [0, 0.5, 0])  # the photo at angle 0 is 0.25 from the wall x = 3
        with tempfile.TemporaryDirectory() as tmp:
            code, out, info = self.run_mode(tmp, Room(x=(-3, 3), z=(-3, 3)), photos)
            self.assertEqual(code, 0, info.get("failed"))
            self.assertLessEqual(info["collapsed_share"], 0.5)
            path = np.array([f["transform_matrix"] for f in json.load(open(out))["frames"]])
            self.assertEqual(len(path), len(photos))

    def test_photos_close_to_walls_move_no_closer(self):
        photos = ring(60, 2.85, 1.85, [0, 0.5, 0])  # 9 photos within the clearance of the walls x = +-3
        room = Room(x=(-3, 3), z=(-2.3, 2.3))
        with tempfile.TemporaryDirectory() as tmp:
            code, out, info = self.run_mode(tmp, room, photos)
            self.assertEqual(code, 0, info.get("failed"))
            self.assertGreater(info["base_cameras_touching"], 0)
            self.assertLessEqual(info["collapsed_share"], 0.25)

            def wall_distance(p):
                return np.minimum.reduce([p[:, 0] + 3, 3 - p[:, 0], p[:, 2] + 2.3, 2.3 - p[:, 2]])

            path = np.array([f["transform_matrix"] for f in json.load(open(out))["frames"]])
            self.assertGreaterEqual(wall_distance(path[:, :3, 3]).min(), 0.9 * wall_distance(photos[:, :3, 3]).min())

    def test_existing_extraction_without_hit_count(self):
        photos = ring(60, 2.75, 1.8, [0, 0.5, 0])
        with tempfile.TemporaryDirectory() as tmp:
            write_scene(tmp, Room(x=(-3, 3), z=(-3, 3)), photos, blob=self.BLOB)
            (Path(tmp) / "flashsplat" / "hit_count.pt").unlink()
            out = Path(tmp) / "traj.json"
            argv = ["--transforms", f"{tmp}/transforms.json", "--output", str(out), "--checkpoint", f"{tmp}/ckpt.pt",
                    "--flashsplat_dir", f"{tmp}/flashsplat", "--clearance", "0.25"]
            self.assertEqual(mt.main(argv), 0)
            info = json.loads(out.with_name("traj_info.json").read_text())
            self.assertEqual(info["object"]["gaussians"], 6000)
            np.testing.assert_allclose(info["object"]["centre"], self.BLOB[0], atol=0.05)

    def test_tight_walls_fail_loudly_unless_fallback(self):
        photos = ring(60, 1.45, 1.3, [0, 0.5, 0])
        room = Room(x=(-1.6, 1.6), z=(-1.45, 1.45))  # the orbit has no room: walls just outside the photos
        with tempfile.TemporaryDirectory() as tmp:
            code, out, info = self.run_mode(tmp, room, photos)
            self.assertEqual(code, 2)
            self.assertIn("collapsed", info["failed"])
            self.assertFalse(out.exists())
        with tempfile.TemporaryDirectory() as tmp:
            code, out, info = self.run_mode(tmp, room, photos, ["--allow_fallback"])
            self.assertEqual(code, 0)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
