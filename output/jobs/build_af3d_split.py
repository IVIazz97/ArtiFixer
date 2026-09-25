#!/usr/bin/env python3
"""Build the ArtiFixer3D input for object removal from propagate_removal.py output.

ArtiFixer3D distils a split into 3DGUT: `selected` frames are anchors (read from a COLMAP
folder, full L1+SSIM loss), every other frame is a target (read from --artifixer_frames_dir,
LPIPS-only loss). Here every frame image is an object-free composite (photo outside the removal
mask, ArtiFixer inside), so anchors never show the removed object.

Writes to --output_dir:
  colmap/images/<name>   composites under their original COLMAP names (JPEG q95)
  colmap/sparse          symlink to the prepared COLMAP model (poses + points3D)
  transforms.json        prepared transforms (unchanged)
  selected_indices.json  anchor frame indices
  split.json             prepared-scene split for data_processing.run_artifixer3d
"""

import argparse
import json
from pathlib import Path

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--scene_root", type=Path, required=True)
parser.add_argument("--propagated_dir", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--anchor_every", type=int, default=3)
args = parser.parse_args()

split = json.loads((args.scene_root / "split.json").read_text())["test"]
(scene_id, entry), = split.items()
transforms_path = args.scene_root / entry["transforms_path"]
frames = json.loads(transforms_path.read_text())["frames"]
seeds = [int(i) for i, r in json.loads((args.propagated_dir / "order.json").read_text()).items() if r == 0]

out = args.output_dir
(out / "colmap" / "images").mkdir(parents=True, exist_ok=True)
sparse = out / "colmap" / "sparse"
if not sparse.exists():
    sparse.symlink_to((args.scene_root / entry["image_root"] / "sparse").resolve())
for i, frame in enumerate(frames):
    with Image.open(args.propagated_dir / "anchors" / f"{i:05d}.png") as image:
        image.convert("RGB").save(out / "colmap" / frame["file_path"], quality=95)

anchors = sorted(set(range(0, len(frames), args.anchor_every)) | set(seeds))
(out / "selected_indices.json").write_text(json.dumps(anchors))
(out / "transforms.json").write_text(transforms_path.read_text())
(out / "split.json").write_text(json.dumps({"test": {scene_id: {
    "scene_id": scene_id,
    "transforms_path": "transforms.json",
    "image_root": "colmap",
    "selected_indices_path": "selected_indices.json",
    "prompt_path": str((args.scene_root / entry["prompt_path"]).resolve()),
    "camera_scale": entry["camera_scale"],
    "metric_scale": entry.get("metric_scale"),
    "has_gt": False,
}}}, indent=2) + "\n")
print(f"{scene_id}: {len(anchors)} anchors, {len(frames) - len(anchors)} targets -> {out / 'split.json'}")
