#!/usr/bin/env python3
"""Progressive reference propagation for object removal with ArtiFixer.

Seed frames (already cleanly inpainted) become anchors. Each round, ArtiFixer generates the
next `--step` frames at every frontier of the anchor set, conditioned on the nearest anchors
(covisibility). Each generated frame is composited -- real photo outside the removal mask,
generation inside it (feathered) -- and added to the anchors. Every rollout is short and starts
right next to clean anchors, so the autoregressive drift seen in a single 185-frame rollout
cannot build up. The model is loaded once for all rounds.

Outputs in --output_dir:
  anchors/<i>.png   composited frame (photo or render outside mask, ArtiFixer inside)
  pred/<i>.png      raw ArtiFixer output
  rounds/<k>/       per-round mini split and generation (deleted unless --keep_rounds)
  order.json        which round produced each frame
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from model_eval.datasets.reconstructed_colmap_eval import ReconstructedColmapEvalDataset
from model_eval.run_inference import (
    get_eval_pipe,
    load_transformer_checkpoint,
    parse_args,
    process_items_with_context_parallel,
)
from model_training.data.utils import NeighborSelectionMode

parser = argparse.ArgumentParser()
parser.add_argument("--scene_root", type=Path, required=True)
parser.add_argument("--removal_dir", type=Path, required=True, help="build_removal_split.py output (renders/opacity/hole).")
parser.add_argument("--seed_pred_dir", type=Path, required=True, help="Pass-1 ArtiFixer pred frames.")
parser.add_argument("--seed_frames", required=True, help="e.g. '0-6' or '0-6,116-122'.")
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--step", type=int, default=7)
parser.add_argument("--num_views", type=int, default=12)
parser.add_argument("--feather_sigma", type=float, default=3.0)
parser.add_argument("--jump_factor", type=float, default=4.0, help="Camera step > factor*median splits sequences.")
parser.add_argument("--outside", choices=("photo", "render"), default="photo",
                    help="What anchors show outside the removal mask: the real photo, or the object-free render.")
parser.add_argument("--keep_rounds", action="store_true")
parser.add_argument("--checkpoint_pt", required=True)
parser.add_argument("--model_id", required=True)
args = parser.parse_args()

split = json.loads((args.scene_root / "split.json").read_text())["test"]
(scene_id, entry), = split.items()
transforms = json.loads((args.scene_root / entry["transforms_path"]).read_text())
frames = transforms["frames"]
n = len(frames)
photo_root = args.scene_root / entry["image_root"]
out = args.output_dir
for sub in ("anchors", "pred"):
    (out / sub).mkdir(parents=True, exist_ok=True)

# Sequence structure: split at camera jumps; a sequence whose ends are close is cyclic.
centers = np.array([f["transform_matrix"] for f in frames])[:, :3, 3]
steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
median_step = np.median(steps)
cuts = [i + 1 for i, s in enumerate(steps) if s > args.jump_factor * median_step]
bounds = list(zip([0] + cuts, cuts + [n]))
cyclic = len(bounds) == 1 and np.linalg.norm(centers[0] - centers[-1]) < args.jump_factor * median_step
seq_of = np.zeros(n, dtype=int)
for k, (a, b) in enumerate(bounds):
    seq_of[a:b] = k
print(f"{scene_id}: {n} frames, sequences={bounds}, cyclic={cyclic}")


def neighbor(i: int, direction: int) -> int | None:
    j = i + direction
    if cyclic:
        return j % n
    if j < 0 or j >= n:
        return None
    return j


def crosses_cut(i: int, j: int) -> bool:
    return seq_of[i] != seq_of[j]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).astype(np.float32)


def composite(i: int, generated_path: Path) -> None:
    if args.outside == "photo":
        photo = load_rgb(photo_root / frames[i]["file_path"])
    else:
        photo = load_rgb(args.removal_dir / "renders" / f"{i:05d}.png")
    generated = load_rgb(generated_path)
    with Image.open(args.removal_dir / "hole" / f"{i:05d}.png") as hole_image:
        hole = np.asarray(hole_image.convert("L")).astype(np.float32) / 255.0
    alpha = np.clip(gaussian_filter(hole, args.feather_sigma), 0, 1)[..., None]
    alpha = np.maximum(alpha, hole[..., None])  # never let the photo (with the object) leak into the hole
    mixed = alpha * generated + (1 - alpha) * photo
    Image.fromarray(mixed.round().clip(0, 255).astype(np.uint8)).save(out / "anchors" / f"{i:05d}.png")
    shutil.copy(generated_path, out / "pred" / f"{i:05d}.png")


done: dict[int, int] = {}  # frame -> round that produced it (0 = seed)
for part in args.seed_frames.split(","):
    a, _, b = part.partition("-")
    for i in range(int(a), int(b or a) + 1):
        composite(i, args.seed_pred_dir / f"{i:05d}.png")
        done[i] = 0

run_args = parse_args([
    "--evalset", "reconstructed_colmap", "--checkpoint_pt", args.checkpoint_pt, "--model_id", args.model_id,
    "--save_dir", str(out / "unused"), "--split_path", str(out / "unused.json"),
    "--render_trajectory", "trajectory", "--neighbor_selection_mode", "covisibility", "--save_frame_outputs_only",
])
device = torch.device("cuda:0")
with torch.inference_mode():
    pipe = get_eval_pipe(run_args, device)
    load_transformer_checkpoint(pipe.transformer, run_args)
    pipe.transformer.eval()

round_idx = 0
while len(done) < n:
    round_idx += 1
    # Each frontier: a done frame whose neighbour in some direction is not done yet.
    chunks, claimed = [], set()
    for i in sorted(done):
        for direction in (1, -1):
            # A chunk may start across a camera jump (new rollout), but never spans one.
            chunk, j = [], neighbor(i, direction)
            while j is not None and j not in done and j not in claimed and len(chunk) < args.step:
                if chunk and crosses_cut(chunk[-1], j):
                    break
                chunk.append(j)
                claimed.add(j)
                j = neighbor(j, direction)
            if chunk:
                chunks.append((i, chunk))
    assert chunks, f"No frontier left but {n - len(done)} frames undone (sequence without a seed?)"

    # Mini split: [frontier anchor, chunk...] per chunk, then every other anchor. Anchors between
    # chunks split them into separate rollouts; all anchors are reference candidates.
    order, selected, targets = [], [], []
    for anchor, chunk in chunks:
        selected.append(len(order)); order.append(anchor)
        for j in chunk:
            targets.append(len(order)); order.append(j)
    frontier_anchors = {anchor for anchor, _ in chunks}
    for i in sorted(done):
        if i not in frontier_anchors:
            selected.append(len(order)); order.append(i)

    rdir = out / "rounds" / f"{round_idx:03d}"
    if rdir.exists():
        shutil.rmtree(rdir)
    for sub in ("images", "renders", "opacity"):
        (rdir / sub).mkdir(parents=True)
    mini_frames = []
    for pos, i in enumerate(order):
        frame = dict(frames[i])
        if i in done:
            frame["file_path"] = f"images/{pos:05d}.png"
            (rdir / "images" / f"{pos:05d}.png").symlink_to((out / "anchors" / f"{i:05d}.png").resolve())
        for sub in ("renders", "opacity"):
            (rdir / sub / f"{pos:05d}.png").symlink_to((args.removal_dir / sub / f"{i:05d}.png").resolve())
        mini_frames.append(frame)
    (rdir / "transforms.json").write_text(json.dumps({**transforms, "frames": mini_frames}))
    (rdir / "selected.json").write_text(json.dumps(selected))
    (rdir / "targets.json").write_text(json.dumps(targets))
    (rdir / "split.json").write_text(json.dumps({"test": {scene_id: {
        "transforms_path": "transforms.json", "image_root": ".", "render_dir": "renders", "opacity_dir": "opacity",
        "selected_indices_path": "selected.json", "target_indices_path": "targets.json",
        "prompt_path": str((args.scene_root / entry["prompt_path"]).resolve()),
        "camera_scale": entry["camera_scale"], "has_gt": False}}}))

    dataset = ReconstructedColmapEvalDataset(
        split="test", split_path=rdir / "split.json", num_views=min(args.num_views, len(selected)),
        neighbor_selection_mode=NeighborSelectionMode.COVISIBILITY, use_target_indices=True,
    )
    with torch.inference_mode():
        process_items_with_context_parallel(pipe, dataset, run_args, rdir / "gen", 0, 1, device)

    pred_dir = rdir / "gen" / scene_id / "frames" / "batch_0000" / "pred"
    for pos in targets:
        composite(order[pos], pred_dir / f"{pos:05d}.png")
        done[order[pos]] = round_idx
    print(f"round {round_idx}: generated {[c for _, c in chunks]} -> {len(done)}/{n} done", flush=True)
    if not args.keep_rounds:
        shutil.rmtree(rdir)

(out / "order.json").write_text(json.dumps({f"{i:05d}": r for i, r in sorted(done.items())}, indent=1))
print(f"Done: {n} frames in {round_idx} rounds -> {out}")
