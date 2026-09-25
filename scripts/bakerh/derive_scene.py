#!/usr/bin/env python3
"""Derive a scene root for the bakerh removal runs from a prepared scene.

The removal scripts (output/jobs/build_removal_split.py, propagate_removal.py, build_af3d_split.py)
read <scene_root>/split.json and use its transforms_path, image_root, prompt_path, camera_scale and
metric_scale. This writes such a split.json into --output_dir, with three optional changes:

  --frames       keep only these frames (0-based inclusive ranges in transforms.json order, i.e. the
                 numbering of the reconstruction renders), e.g. "49-148,284-304"
  --prompt_path  use another caption.h5 (e.g. the "empty floor" prompt) for every later stage
  resolution     when the photos are larger than the 3DGUT renders (scene trained on downsampled
                 images), write the photos resized to the render size and scale the intrinsics,
                 so photos, renders, hole masks and ArtiFixer outputs all share one size

Writes to --output_dir:
  split.json        {"test": {scene_id: {...}}} with absolute paths
  transforms.json   prepared transforms restricted to --frames
  colmap/images     resized photos, colmap/sparse -> source sparse (only when resizing)
  frames.json       source index of every frame, window starts, and auto seed ranges
"""

import argparse
import json
from pathlib import Path

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--scene_root", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--frames", default="", help="e.g. '49-148,284-304'; empty keeps every frame.")
parser.add_argument("--prompt_path", type=Path, default=None)
parser.add_argument("--seed_length", type=int, default=7, help="Frames per auto seed range (one per window).")
args = parser.parse_args()

split = json.loads((args.scene_root / "split.json").read_text())["test"]
(scene_id, entry), = split.items()
root = args.scene_root.resolve()
transforms = json.loads((root / entry["transforms_path"]).read_text())
frames = transforms["frames"]
image_root = (root / entry["image_root"]).resolve()

if args.frames:
    indices = []
    for part in args.frames.split(","):
        a, _, b = part.strip().partition("-")
        indices += range(int(a), int(b or a) + 1)
else:
    indices = list(range(len(frames)))
assert indices and max(indices) < len(frames) and min(indices) >= 0, f"--frames out of range [0, {len(frames) - 1}]"
assert len(set(indices)) == len(indices), "--frames overlap"
window_starts = [k for k in range(len(indices)) if k == 0 or indices[k] != indices[k - 1] + 1]
window_ends = window_starts[1:] + [len(indices)]
auto_seeds = ",".join(f"{s}-{min(s + args.seed_length, e) - 1}" for s, e in zip(window_starts, window_ends))

out = args.output_dir
out.mkdir(parents=True, exist_ok=True)
subset = [dict(frames[i]) for i in indices]

with Image.open(image_root / subset[0]["file_path"]) as photo:
    photo_size = photo.size
render_size = photo_size
if "render_dir" in entry:
    first_render = (root / entry["render_dir"]) / f"{indices[0]:05d}.png"
    if first_render.is_file():
        with Image.open(first_render) as render:
            render_size = render.size
    else:
        print(f"warning: {first_render} missing; assuming photos match the render size")

if render_size != photo_size:
    sx, sy = render_size[0] / photo_size[0], render_size[1] / photo_size[1]
    print(f"photos {photo_size} -> renders {render_size}: resizing photos, scaling intrinsics by ({sx:.4f}, {sy:.4f})")
    new_root = out / "colmap"
    (new_root / "images").mkdir(parents=True, exist_ok=True)
    sparse = new_root / "sparse"
    if not sparse.exists():
        sparse.symlink_to((image_root / "sparse").resolve())
    for frame in subset:
        with Image.open(image_root / frame["file_path"]) as photo:
            target = new_root / frame["file_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            photo.convert("RGB").resize(render_size, Image.LANCZOS).save(target, quality=95)
    for mapping in [transforms, *subset]:
        for key, scale in (("fl_x", sx), ("cx", sx), ("fl_y", sy), ("cy", sy)):
            if key in mapping:
                mapping[key] = mapping[key] * scale
        if "w" in mapping:
            mapping["w"], mapping["h"] = render_size
    image_root = new_root.resolve()

(out / "transforms.json").write_text(json.dumps({**transforms, "frames": subset}, indent=2))
(out / "frames.json").write_text(json.dumps(
    {"source_indices": indices, "window_starts": window_starts, "auto_seeds": auto_seeds}, indent=1))
prompt_path = (args.prompt_path or root / entry["prompt_path"]).resolve()
assert prompt_path.is_file(), f"Missing prompt {prompt_path}"
(out / "split.json").write_text(json.dumps({"test": {scene_id: {
    "scene_id": scene_id,
    "transforms_path": str((out / "transforms.json").resolve()),
    "image_root": str(image_root),
    "prompt_path": str(prompt_path),
    "camera_scale": entry["camera_scale"],
    "metric_scale": entry.get("metric_scale"),
}}}, indent=2) + "\n")
print(f"{scene_id}: {len(indices)}/{len(frames)} frames, windows at {window_starts}, auto seeds {auto_seeds}, "
      f"prompt {prompt_path} -> {out / 'split.json'}")
