#!/usr/bin/env python3
"""Debug view of a vanilla run: where the camera is on the novel path, what ArtiFixer got, what it made.

For a run_removal.sh vanilla[_<mode>] output dir, writes to <variant_dir>/debug/:
  0_debug.mp4             one frame per path frame, side by side:
                            top view of the photos (blue) and the path (orange) with the current camera
                            (red, with its view direction; for vidsplat the current orbit in red too),
                            the 3DGUT render of the path (what ArtiFixer gets),
                            the ArtiFixer output and, once it exists, the ArtiFixer3D+ output;
                          with a caption: frame, trajectory mode and, per mode, the offset scale
                          (object/travel) or the orbit, its seed photo, direction and extent (vidsplat).
  frames/<i>.png          the frames of 0_debug.mp4
  1_path_render.mp4       links to the videos run_inference already writes (3DGUT render of the path,
  2_artifixer.mp4         ArtiFixer output, ArtiFixer3D+ output), whichever exist
  3_artifixer3d_plus.mp4
Before ArtiFixer has run, the render column comes from the path renders of the prepared scene and the
other columns are left out. Uses only what is on disk: rerun it any time.

    python scripts/bakerh/debug_video.py --variant_dir output/bakerh_removal/PCV/vanilla_vidsplat
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

AF = Path(__file__).resolve().parents[2]
if str(AF) not in sys.path:
    sys.path.insert(0, str(AF))


def newest_batch(root):
    """The newest run_inference frames/batch_0000 dir under root (None if none)."""
    batches = [p for p in Path(root).glob("**/frames/batch_0000") if (p / "pred").is_dir()] if Path(root).is_dir() else []
    return max(batches, key=lambda p: p.stat().st_mtime) if batches else None


def frame_files(folder, count):
    """Path frame i -> <folder>/<i:05d>.png, else the i-th PNG in name order (None where missing)."""
    if folder is None or not Path(folder).is_dir():
        return None
    files = sorted(Path(folder).glob("*.png"))
    by_name = {p.stem: p for p in files}
    out = [by_name.get(f"{i:05d}") for i in range(count)]
    if sum(p is not None for p in out) < len(files) // 2:  # other naming: fall back to order
        out = (files + [None] * count)[:count]
    return out if any(out) else None


def panel(path, height, label):
    if path is None:
        image = Image.new("RGB", (height * 16 // 9, height), (40, 40, 40))
    else:
        image = Image.open(path).convert("RGB")
        image = image.resize((max(1, round(image.width * height / image.height)), height), Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((6, 4), label)
    draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return image


class Minimap:
    """Top view (on the plane across the photos' mean up axis) of photos, path and the current camera."""

    def __init__(self, photos_c2w, path, size, anchors=None, centre=None):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        up = photos_c2w[:, :3, 1].mean(0)
        up /= np.linalg.norm(up)
        cams = np.concatenate([photos_c2w[:, :3, 3], path[:, :3, 3]])
        spread = cams - cams.mean(0)
        spread -= np.outer(spread @ up, up)
        h1 = np.linalg.eigh(spread.T @ spread)[1][:, -1]
        h2 = np.cross(up, h1)
        self.flat = lambda x: np.stack([np.asarray(x) @ h1, np.asarray(x) @ h2], -1)  # noqa: E731
        shown = self.flat(np.concatenate([cams] + ([np.asarray(anchors).reshape(-1, 3)] if anchors else [])))
        lo, hi = shown.min(0), shown.max(0)
        pad = 0.1 * (hi - lo).max()
        self.scale = 0.06 * (hi - lo).max()
        self.path = self.flat(path[:, :3, 3])
        self.look = self.flat(-path[:, :3, 2])
        self.fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        ax = self.fig.add_axes([0.01, 0.01, 0.98, 0.98])
        ax.plot(*self.flat(photos_c2w[:, :3, 3]).T, ".", c="tab:blue", ms=2)
        ax.plot(*self.path.T, "-", c="tab:orange", lw=0.6)
        if anchors:
            ax.plot(*self.flat(anchors).T, "x", c="tab:red", ms=4)
        if centre is not None:
            ax.plot(*self.flat(np.asarray(centre)[None])[0], "*", c="tab:red", ms=9)
        self.segment, = ax.plot([], [], "-", c="tab:red", lw=1.4, alpha=0.6)
        self.camera, = ax.plot([], [], "o", c="tab:red", ms=5)
        self.view, = ax.plot([], [], "-", c="tab:red", lw=1.5)
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])

    def draw(self, i, segment=None):
        p = self.path[i]
        self.camera.set_data([p[0]], [p[1]])
        tip = p + self.scale * self.look[i] / max(np.linalg.norm(self.look[i]), 1e-9)
        self.view.set_data([p[0], tip[0]], [p[1], tip[1]])
        seg = self.path[segment] if segment is not None else np.empty((0, 2))
        self.segment.set_data(seg[:, 0], seg[:, 1])
        self.fig.canvas.draw()
        return Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy())


def caption(info, i, n):
    mode = info.get("mode", "?")
    text = f"frame {i}/{n - 1}   trajectory: {mode}"
    if mode == "vidsplat" and info.get("clips"):
        L = info["clip_len"]
        if i == 0:
            return text + f"   seed photo {info['clips'][0]['seed']} (start)", None
        k, j = (i - 1) // L, (i - 1) % L
        clip = info["clips"][min(k, len(info["clips"]) - 1)]
        text += (f"   orbit {k + 1}/{len(info['clips'])}: seed {clip['seed']}, {clip['direction']} "
                 f"{clip['extent_deg']:.0f} deg, frame {j + 1}/{L}")
        return text, list(range(1 + k * L, min(n, 1 + (k + 1) * L)))
    scale = info.get("offset_scale")
    if scale:
        stride = info.get("frame_stride", 1)
        text += f"   offset scale {scale[min(i * stride, len(scale) - 1)]:.2f}"
    if info.get("failed"):
        text += "   (path FAILED: " + info["failed"][:60] + "...)"
    return text, None


def link(src, dst):
    if src is None or not Path(src).exists():
        return False
    dst.unlink(missing_ok=True)
    os.symlink(os.path.relpath(src, dst.parent), dst)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant_dir", type=Path, required=True, help="output/bakerh_removal/<scene>/vanilla[_<mode>]")
    parser.add_argument("--scene", default=None, help="Scene id (default: the prepared root inside variant_dir).")
    parser.add_argument("--height", type=int, default=360, help="Height of every panel, pixels.")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args(argv)

    O = args.variant_dir
    scene = args.scene or next((p.name for p in O.iterdir() if (p / "3dgrut_input").is_dir()), None)
    assert scene, f"no prepared scene root (<scene>/3dgrut_input) in {O}"
    V = O / scene
    trajectory = json.loads((O / "trajectory_input.json").read_text())
    info_path = O / "trajectory_input_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    path = np.array([f["transform_matrix"] for f in trajectory["frames"]], dtype=np.float64)
    photos = json.loads((V / "3dgrut_input" / scene / "nerfstudio" / "transforms.json").read_text())
    photos_c2w = np.array([f["transform_matrix"] for f in sorted(photos["frames"], key=lambda f: f["file_path"])])
    n = len(path)

    batch = newest_batch(O / "artifixer")
    plus = newest_batch(O / "af3d_plus")
    columns = []  # (label, frame files)
    render_dir = batch / "rendered" if batch else V / "recon_results" / scene / "reconstruction" / scene / "ours_30000" / "trajectory" / "renders"
    renders = frame_files(render_dir, n)
    if renders:
        columns.append(("3DGUT render of the path (ArtiFixer input)", renders))
    if batch and frame_files(batch / "pred", n):
        columns.append(("ArtiFixer", frame_files(batch / "pred", n)))
    if plus and frame_files(plus / "pred", n):
        columns.append(("ArtiFixer3D+", frame_files(plus / "pred", n)))
    if not columns:
        print(f"nothing rendered yet under {O}: only the path top view")

    out = O / "debug"
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    centre = info.get("object", {}).get("centre")
    minimap = Minimap(photos_c2w, path, args.height, anchors=info.get("anchors"), centre=centre)
    files = []
    for i in range(n):
        text, segment = caption(info, i, n)
        tiles = [minimap.draw(i, segment)] + [panel(col[i], args.height, label) for label, col in columns]
        width = sum(t.width for t in tiles)
        frame = Image.new("RGB", (width + width % 2, args.height + 24), (0, 0, 0))
        x = 0
        for tile in tiles:
            frame.paste(tile, (x, 24))
            x += tile.width
        ImageDraw.Draw(frame).text((6, 6), text, fill=(255, 255, 255))
        files.append(frames_dir / f"{i:05d}.png")
        frame.save(files[-1])
    from model_training.utils.video_io import save_video

    save_video(files, out / "0_debug.mp4", fps=args.fps)
    print(f"debug video: {out / '0_debug.mp4'} ({n} frames; columns: top view"
          + "".join(f", {label}" for label, _ in columns) + ")")

    videos = [("1_path_render.mp4", batch, "rendered"), ("2_artifixer.mp4", batch, "pred"), ("3_artifixer3d_plus.mp4", plus, "pred")]
    for name, b, kind in videos:
        src = b.parent.parent / "videos" / f"batch_0000_{kind}.mp4" if b else None
        if link(src, out / name):
            print(f"  {out / name} -> {src}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
