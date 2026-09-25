#!/usr/bin/env python3
"""Turn FlashSplat background renders into a reconstructed_colmap split for object-removal inpainting.

Per frame (in the prepared transforms.json order that run_inference indexes by):
  renders/<i>.png   background-only 3DGUT render (object Gaussians removed)
  opacity/<i>.png   background opacity, forced to 0 inside the dilated object footprint so
                    ArtiFixer treats that region as unobserved and regenerates it
  refs/<file_path>  real photo with the dilated footprint replaced by the background render,
                    so the reference views ArtiFixer copies from no longer show the object
  hole/<i>.png      the dilated footprint itself (for inspection / masked metrics)
The footprint is the union of the SAM2 mask and the projected removed-object opacity.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull

parser = argparse.ArgumentParser()
parser.add_argument("--scene_root", type=Path, required=True)
parser.add_argument("--flashsplat_dir", type=Path, required=True)
parser.add_argument("--mask_dir", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--dilate_px", type=int, default=12)
parser.add_argument("--fg_opacity_threshold", type=float, default=0.05)
parser.add_argument("--opacity", choices=("hole0", "render"), default="hole0",
                    help="hole0: zero opacity in the footprint; render: raw background opacity.")
parser.add_argument("--refs", choices=("composite", "render", "photo"), default="composite",
                    help="Reference views: photo with footprint patched, pure background render, or raw photo.")
parser.add_argument("--hole_shape", choices=("mask", "hull"), default="mask",
                    help="hull: fill the 2D convex hull of the footprint, hiding the object's silhouette.")
args = parser.parse_args()

split = json.loads((args.scene_root / "split.json").read_text())["test"]
(scene_id, entry), = split.items()
transforms_path = args.scene_root / entry["transforms_path"]
image_root = args.scene_root / entry["image_root"]
frames = json.loads(transforms_path.read_text())["frames"]
index_by_name = {name: int(idx) for idx, name in json.loads((args.flashsplat_dir / "frame_names.json").read_text()).items()}

out = args.output_dir
for sub in ("renders", "opacity", "hole"):
    (out / sub).mkdir(parents=True, exist_ok=True)


def load(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert(mode))


hole_fractions = []
for i, frame in enumerate(frames):
    name = Path(frame["file_path"]).name
    j = index_by_name[name]
    background = load(args.flashsplat_dir / "background_renders" / f"{j:05d}.png", "RGB")
    background_opacity = load(args.flashsplat_dir / "background_renders_opacity" / f"{j:05d}.png", "L")
    foreground_opacity = load(args.flashsplat_dir / "foreground_renders_opacity" / f"{j:05d}.png", "L")
    real = load(image_root / frame["file_path"], "RGB")
    h, w = background.shape[:2]
    assert real.shape[:2] == (h, w), f"{name}: photo {real.shape[:2]} vs render {(h, w)}"

    sam = np.load(args.mask_dir / f"{Path(name).stem}.npy")
    if sam.shape != (h, w):
        sam = np.asarray(Image.fromarray(sam).resize((w, h), Image.NEAREST))
    hole = (sam > 0) | (foreground_opacity / 255.0 > args.fg_opacity_threshold)
    if args.hole_shape == "hull":
        ys, xs = np.nonzero(hole)
        points = np.stack([xs, ys], 1)
        polygon = Image.new("L", (w, h), 0)
        ImageDraw.Draw(polygon).polygon([tuple(p) for p in points[ConvexHull(points).vertices]], fill=1)
        hole = np.asarray(polygon, dtype=bool)
    hole = binary_dilation(hole, iterations=args.dilate_px)
    hole_fractions.append(hole.mean())

    Image.fromarray(background).save(out / "renders" / f"{i:05d}.png")
    Image.fromarray(np.where(hole & (args.opacity == "hole0"), 0, background_opacity).astype(np.uint8)).save(out / "opacity" / f"{i:05d}.png")
    Image.fromarray((hole * 255).astype(np.uint8)).save(out / "hole" / f"{i:05d}.png")
    ref_path = out / "refs" / frame["file_path"]
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref = {"composite": np.where(hole[..., None], background, real), "render": background, "photo": real}[args.refs]
    Image.fromarray(ref).save(ref_path.with_suffix(".png"))
    frame_ref = ref_path.with_suffix(".png").relative_to(out / "refs").as_posix()
    frame["file_path"] = frame_ref

transforms = json.loads(transforms_path.read_text())
transforms["frames"] = frames
(out / "transforms.json").write_text(json.dumps(transforms, indent=2))
(out / "selected_indices.json").write_text(json.dumps(list(range(len(frames)))))

removal_entry = {
    "scene_id": scene_id,
    "transforms_path": str(out / "transforms.json"),
    "image_root": str(out / "refs"),
    "render_dir": str(out / "renders"),
    "opacity_dir": str(out / "opacity"),
    "selected_indices_path": str(out / "selected_indices.json"),
    "prompt_path": str(args.scene_root / entry["prompt_path"]),
    "camera_scale": entry["camera_scale"],
    "metric_scale": entry.get("metric_scale"),
    "has_gt": False,
}
(out / "split.json").write_text(json.dumps({"test": {scene_id: removal_entry}}, indent=2) + "\n")
print(f"wrote {out / 'split.json'}: {len(frames)} frames, hole fraction mean={np.mean(hole_fractions):.3f} "
      f"min={np.min(hole_fractions):.3f} max={np.max(hole_fractions):.3f}")
