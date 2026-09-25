#!/usr/bin/env python3
"""Write a novel camera path for vanilla ArtiFixer3D from a prepared scene's photo cameras.

Reads the prepared nerfstudio transforms.json (OpenGL camera-to-world, as written by
data_processing.prepare_colmap_artifixer_inputs) and writes one path camera per photo, in photo
(file name) order:
  - position and rotation are Gaussian-smoothed over neighbouring photos (--sigma frames), within
    runs of photos without a jump (a step larger than --jump x the median photo spacing);
  - each camera is then moved by --side x spacing x sin(...) perpendicular to both the direction of
    travel and its up axis (toward and away from the object on a capture that circles it, left and
    right on a walk), weaving --periods times from one side of the photo path to the other, and up
    by --up x spacing. Rotations are kept, so every path camera looks where the photos look, from
    viewpoints no photo was taken from.

The output is the transforms-style trajectory JSON that prepare_colmap_artifixer_inputs
--trajectory_path takes: the photos' shared OPENCV intrinsics and frames with transform_matrix only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("--transforms", type=Path, required=True, help="Prepared <root>/3dgrut_input/<scene>/nerfstudio/transforms.json")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--side", type=float, default=2.0, help="Weave amplitude across the direction of travel, in median photo spacings.")
parser.add_argument("--up", type=float, default=0.5, help="Constant upward shift, in median photo spacings.")
parser.add_argument("--periods", type=float, default=2.0, help="Left-right weaves over the whole path.")
parser.add_argument("--sigma", type=float, default=2.0, help="Smoothing width, in photos.")
parser.add_argument("--jump", type=float, default=4.0, help="A step above this many spacings splits the path.")
args = parser.parse_args()

transforms = json.loads(args.transforms.read_text())
assert "applied_transform" not in transforms, "expected a prepared transforms.json without applied_transform"
frames = sorted(transforms["frames"], key=lambda f: f["file_path"])
c2w = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
n = len(c2w)
assert n >= 2, f"need at least 2 photos, got {n}"
pos = c2w[:, :3, 3]
steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
spacing = float(np.median(steps))
assert spacing > 0, "photo cameras do not move"
segments = np.split(np.arange(n), np.where(steps > args.jump * spacing)[0] + 1)

path = []
for seg in segments:
    rot = Rotation.from_matrix(c2w[seg, :3, :3])
    smooth = []
    for i in seg:
        w = np.exp(-0.5 * ((seg - i) / args.sigma) ** 2)
        w /= w.sum()
        m = np.eye(4)
        m[:3, :3] = rot.mean(weights=w).as_matrix()
        m[:3, 3] = (w[:, None] * pos[seg]).sum(0)
        smooth.append(m)
    smooth = np.stack(smooth)
    travel = np.gradient(smooth[:, :3, 3], axis=0) if len(seg) > 1 else np.zeros((1, 3))
    for m, i, t in zip(smooth, seg, travel):
        up = m[:3, 1]  # OpenGL camera axes: x right, y up, z back
        across = np.cross(t, up)
        across = across / np.linalg.norm(across) if np.linalg.norm(across) > 1e-9 * spacing else m[:3, 0]
        side = args.side * spacing * np.sin(2 * np.pi * args.periods * i / (n - 1))
        m[:3, 3] += side * across + args.up * spacing * up
        path.append(m)
path = np.stack(path)

keys = ("camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2")
out = {k: transforms[k] for k in keys if k in transforms}
out["frames"] = [{"transform_matrix": m.tolist()} for m in path]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, indent=2) + "\n")

nearest = np.linalg.norm(path[:, None, :3, 3] - pos[None], axis=2).min(1) / spacing
angle = np.degrees(np.linalg.norm((Rotation.from_matrix(path[:, :3, :3]) * Rotation.from_matrix(c2w[:, :3, :3]).inv()).as_rotvec(), axis=1))
print(f"photos={n} path_frames={len(path)} segments={len(segments)} median_spacing={spacing:.4f} (scene units)")
print(f"path camera to nearest photo camera, in spacings: mean {nearest.mean():.2f} max {nearest.max():.2f}")
print(f"rotation change vs. the photo at the same index: mean {angle.mean():.1f} deg, max {angle.max():.1f} deg")
print(f"wrote {args.output}")
