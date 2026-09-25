#!/usr/bin/env python3
"""Move background Gaussians resting on top of the segmented object into the object label.

Used for garden: the SAM2 mask covers the table but not the vase on it, so removing the table
leaves the vase floating. World "up" is estimated as the mean camera up vector (mip360 captures
are taken upright). Background Gaussians inside the object's footprint (2D convex hull in the
plane orthogonal to up) and between the object's top and top + height_factor * object height
are relabelled as object.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib.path import Path as Polygon
from scipy.spatial import ConvexHull

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--transforms", type=Path, required=True)
parser.add_argument("--flashsplat_dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--object_id", type=int, default=1)
parser.add_argument("--height_factor", type=float, default=1.5)
parser.add_argument("--min_total_contribution", type=float, default=0.1)
args = parser.parse_args()

positions = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["positions"].detach().float().numpy()
labels = torch.load(args.flashsplat_dir / "labels.pt", map_location="cpu").clone()
contribution = torch.load(args.flashsplat_dir / "contribution.pt", map_location="cpu")
hit_count = torch.load(args.flashsplat_dir / "hit_count.pt", map_location="cpu").float()
visible = (contribution.sum(0) / hit_count.clamp_min(1)) > args.min_total_contribution

c2ws = np.array([f["transform_matrix"] for f in json.loads(args.transforms.read_text())["frames"]])
up = c2ws[:, :3, 1].mean(0)  # OpenGL camera +y is up
up /= np.linalg.norm(up)
e1 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
e1 /= np.linalg.norm(e1)
e2 = np.cross(up, e1)

obj = (labels[args.object_id] & visible).numpy()
obj_pos = positions[obj]
height = positions @ up
obj_h = obj_pos @ up
top, bottom = np.percentile(obj_h, 99), np.percentile(obj_h, 1)
camera_h = (c2ws[:, :3, 3] @ up).mean()
print(f"up={up.round(3)} object height {bottom:.3f}..{top:.3f}, mean camera height {camera_h:.3f}")
assert camera_h > top, "cameras should be above the object top; up vector sign looks wrong"

planar = np.stack([positions @ e1, positions @ e2], 1)
obj_planar = planar[obj]
centroid = obj_planar.mean(0)
dist = np.linalg.norm(obj_planar - centroid, axis=1)
obj_planar = obj_planar[dist <= np.percentile(dist, 99.5)]
hull = ConvexHull(obj_planar)
footprint = Polygon(obj_planar[hull.vertices])

in_band = (height > top - 0.02 * (top - bottom)) & (height < top + args.height_factor * (top - bottom))
candidates = np.where(in_band & ~labels[args.object_id].numpy())[0]
inside = footprint.contains_points(planar[candidates])
moved = candidates[inside]
labels[args.object_id, moved] = True
labels[0, moved] = False
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(labels, args.output)
print(f"moved {len(moved)} Gaussians into object {args.object_id}; wrote {args.output}")
