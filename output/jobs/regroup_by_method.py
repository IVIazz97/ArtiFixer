#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-off: regroup output/ by method instead of by dataset.

    output/mip360/<scene>/{3dgrut_input,3dgrut_runs,...}  ->  output/recon/<scene>/...
    output/mip360/<scene>/removal/flashsplat              ->  output/flashsplat/<scene>
    output/mip360/garden/removal/flashsplat_vase          ->  output/flashsplat/garden_vase
    output/mip360/<scene>/removal/aurafusion              ->  output/aurafusion/<scene>
    output/mip360/<scene>/removal/inpaint360gs            ->  output/inpaint360gs/<scene>
    output/mip360/<scene>/removal/{v0..v9,propagation,af3d} -> output/artifixer/<scene>/...

Everything is one filesystem, so the moves are renames. Absolute paths inside symlink targets
and JSON/YAML/text files under output/ are rewritten in one pass, as in ``reorganize_removal.py``.
Run with --dry-run first; it prints the plan and what it would rewrite.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1]
OLD_ROOT = OUTPUT / "mip360"
RECON_ITEMS = ("3dgrut_input", "3dgrut_runs", "captions", "metric_alignment", "recon_results",
               "split.json", "selected_images.txt", "selected_indices.json")
METHOD_OF = {"flashsplat": "flashsplat", "aurafusion": "aurafusion", "inpaint360gs": "inpaint360gs"}
ARTIFIXER = tuple(f"v{i}" for i in range(10)) + ("propagation", "af3d")
REWRITE_SUFFIXES = {".json", ".txt", ".yaml", ".yml", ".md", ".cfg", ""}


def plan() -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for scene_dir in sorted(p for p in OLD_ROOT.iterdir() if p.is_dir()):
        scene = scene_dir.name
        for item in RECON_ITEMS:
            if (scene_dir / item).exists():
                moves.append((scene_dir / item, OUTPUT / "recon" / scene / item))
        removal = scene_dir / "removal"
        if not removal.is_dir():
            continue
        for entry in sorted(removal.iterdir()):
            name = entry.name
            if name in METHOD_OF:
                moves.append((entry, OUTPUT / METHOD_OF[name] / scene))
            elif name.startswith("flashsplat_"):  # e.g. flashsplat_vase -> flashsplat/garden_vase
                moves.append((entry, OUTPUT / "flashsplat" / f"{scene}_{name.split('_', 1)[1]}"))
            elif name in ARTIFIXER:
                moves.append((entry, OUTPUT / "artifixer" / scene / name))
            else:
                raise SystemExit(f"unmapped directory: {entry}")
    return moves


def rewriter(moves: list[tuple[Path, Path]]):
    """One regex pass over old -> new absolute prefixes, longest first so the most specific wins."""
    mapping = {str(old): str(new) for old, new in moves}
    for scene_dir in sorted(p for p in OLD_ROOT.iterdir() if p.is_dir()):  # bare scene root, last resort
        mapping.setdefault(str(scene_dir), str(OUTPUT / "recon" / scene_dir.name))
    pattern = re.compile("|".join(re.escape(o) for o in sorted(mapping, key=len, reverse=True)) + r'(?=[/"\'\s]|$)')
    return lambda text: pattern.sub(lambda m: mapping[m.group(0)], text), pattern


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    moves = plan()
    rewrite, pattern = rewriter(moves)
    print(f"{len(moves)} moves:")
    for old, new in moves:
        print(f"  {old.relative_to(OUTPUT)} -> {new.relative_to(OUTPUT)}")

    if not args.dry_run:
        for old, new in moves:
            new.parent.mkdir(parents=True, exist_ok=True)
            if new.exists():
                raise SystemExit(f"target already exists: {new}")
            shutil.move(str(old), str(new))
        for scene_dir in sorted(p for p in OLD_ROOT.iterdir() if p.is_dir()):
            for leftover in (scene_dir / "removal", scene_dir):
                if leftover.is_dir() and not any(leftover.iterdir()):
                    leftover.rmdir()

    roots = [OUTPUT] if not args.dry_run else [OUTPUT]
    links, files = 0, 0
    for root in roots:
        for path in root.rglob("*"):
            if "logs" in path.parts:
                continue
            if path.is_symlink():
                target = str(path.readlink())
                new_target = rewrite(target)
                if new_target != target:
                    links += 1
                    if not args.dry_run:
                        path.unlink()
                        path.symlink_to(new_target)
            elif path.is_file() and path.suffix in REWRITE_SUFFIXES and path.stat().st_size < 20 << 20:
                try:
                    text = path.read_text()
                except (UnicodeDecodeError, OSError):
                    continue
                if pattern.search(text):
                    files += 1
                    if not args.dry_run:
                        path.write_text(rewrite(text))
    print(f"{'would rewrite' if args.dry_run else 'rewrote'} {links} symlink targets and {files} files")


if __name__ == "__main__":
    main()
