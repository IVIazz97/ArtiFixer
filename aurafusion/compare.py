#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect an AuraFusion run: unseen-mask agreement with the official masks, and image panels.

``--official_data`` (360-USID Other-360/<scene> layout, e.g. kitchen) enables per-view IoU of our
unseen masks against the released official ones; views are matched by image name.
``--official_results`` adds the released official inpainted renders as a panel column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.NEAREST)
    return np.asarray(image) > 127


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_dir", type=Path, required=True, help="A variant dir (holds cameras.json, final/).")
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--official_data", type=Path, default=None)
    parser.add_argument("--official_results", type=Path, default=None,
                        help="Dir of official final renders (00000.png...), indexed in name-sorted view order.")
    parser.add_argument("--views", default="0,40,90,140")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--width", type=int, default=360)
    args = parser.parse_args()

    root = args.render_dir
    cameras = json.loads((root / "cameras.json").read_text())

    if args.official_data is not None:
        rows = []
        for c in cameras:
            official = args.official_data / "unseen_masks" / c["name"]
            if not official.exists():
                continue
            size = (c["width"], c["height"])
            ours_raw, ours = load_mask(root / "unseen" / f"{c['index']:05d}.png", size), \
                load_mask(root / "unseen_dilated" / f"{c['index']:05d}.png", size)
            theirs = load_mask(official, size)
            union = (ours_raw | theirs).sum()
            rows.append(((ours_raw & theirs).sum() / union if union else np.nan,
                         ours_raw.mean(), theirs.mean(), ours.mean()))
        rows = np.array(rows, dtype=np.float64)
        print(f"unseen masks vs official on {len(rows)} views: IoU mean={np.nanmean(rows[:, 0]):.3f} "
              f"median={np.nanmedian(rows[:, 0]):.3f}; area ours={rows[:, 1].mean():.4f} "
              f"official={rows[:, 2].mean():.4f} ours_dilated={rows[:, 3].mean():.4f}")

    columns = [("photo", lambda c: args.colmap_dir / "images" / c["name"]),
               ("removed", lambda c: root / "removed" / "rgb" / f"{c['index']:05d}.png"),
               ("unseen (ours)", lambda c: root / "unseen_dilated" / f"{c['index']:05d}.png"),
               ("init", lambda c: root / "init" / "rgb" / f"{c['index']:05d}.png"),
               ("SDEdit target", lambda c: root / "sdedit" / f"{c['index']:05d}.png"),
               ("final (ours)", lambda c: root / "final" / "rgb" / f"{c['index']:05d}.png")]
    if args.official_data is not None:
        columns.insert(3, ("unseen (official)", lambda c: args.official_data / "unseen_masks" / c["name"]))
    if args.official_results is not None:  # the official code sorts views by image name
        official_index = {name: i for i, name in enumerate(sorted(c["name"] for c in cameras))}
        columns.append(("final (official)", lambda c: args.official_results / f"{official_index[c['name']]:05d}.png"))

    views = [int(v) for v in args.views.split(",")]
    first = cameras[views[0]]
    w = args.width
    h = round(first["height"] * w / first["width"])
    panel = Image.new("RGB", (len(columns) * w, len(views) * h + 20), "white")
    draw = ImageDraw.Draw(panel)
    for col, (label, _) in enumerate(columns):
        draw.text((col * w + 4, 4), label, fill="black")
    for row, v in enumerate(views):
        for col, (_, path_of) in enumerate(columns):
            path = path_of(cameras[v])
            if Path(path).exists():
                panel.paste(Image.open(path).convert("RGB").resize((w, h)), (col * w, 20 + row * h))
    args.panel.parent.mkdir(parents=True, exist_ok=True)
    panel.save(args.panel, quality=88)
    print(f"panel -> {args.panel}")


if __name__ == "__main__":
    main()
