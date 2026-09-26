#!/usr/bin/env python3
"""COLMAP scene restricted to a subset of views, to run the removal baselines on part of a capture.

--frames uses the numbering of derive_scene.py and run_removal.sh FRAMES: 0-based inclusive ranges in
the prepared transforms.json order of --scene_root (the reconstruction render numbering), so
COMPRESSOR_FRAMES picks the same photos for ArtiFixer and for the baselines.

Writes <output_dir>/sparse/0/{cameras,images}.bin (+ a points3D.bin symlink) and, for every images*
folder of --colmap_dir (images, images_2, ...), the same folder with symlinks to the kept photos.
The 3DGUT checkpoint trained on the full scene renders this subset as it is.
"""

import argparse
import json
import shutil
from pathlib import Path

from data_processing.prepare_colmap_artifixer_inputs import ColmapScene, image_basename, read_colmap_scene, write_sparse_model

parser = argparse.ArgumentParser()
parser.add_argument("--scene_root", type=Path, required=True, help="Prepared scene root with split.json.")
parser.add_argument("--colmap_dir", type=Path, required=True, help="The 3DGUT input: images*/ and sparse/0/.")
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--frames", required=True, help="e.g. '49-148,284-304'")
args = parser.parse_args()

(_, entry), = json.loads((args.scene_root / "split.json").read_text())["test"].items()
frames = json.loads((args.scene_root / entry["transforms_path"]).read_text())["frames"]
indices = []
for part in args.frames.split(","):
    a, _, b = part.strip().partition("-")
    indices += range(int(a), int(b or a) + 1)
assert indices and min(indices) >= 0 and max(indices) < len(frames), f"--frames out of range [0, {len(frames) - 1}]"
assert len(set(indices)) == len(indices), "--frames overlap"
names = [Path(frames[i]["file_path"]).name for i in indices]

scene = read_colmap_scene(args.colmap_dir / "sparse" / "0")
by_name = {image_basename(image): image for image in scene.images}
missing = [name for name in names if name not in by_name]
assert not missing, f"{len(missing)} frames not in {args.colmap_dir}/sparse/0, e.g. {missing[:3]}"

out = args.output_dir
if out.exists():
    shutil.rmtree(out)
(out / "sparse" / "0").mkdir(parents=True)
write_sparse_model(args.colmap_dir / "sparse" / "0", out / "sparse" / "0",
                   ColmapScene(cameras=scene.cameras, images=[by_name[name] for name in names]))
folders = [f for f in sorted(args.colmap_dir.glob("images*")) if f.is_dir()]
for folder in folders:
    (out / folder.name).mkdir()
    for name in names:
        if (folder / name).exists():
            (out / folder.name / name).symlink_to((folder / name).resolve())
print(f"{len(names)}/{len(scene.images)} views ({args.frames}), folders {[f.name for f in folders]} -> {out}")
