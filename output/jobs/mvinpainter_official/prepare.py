#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lay a mip360 scene out as the unmodified MVInpainter ``test_removal.py --dataset_names realworld`` reads it.

Their RealWorldDataset treats every sub-folder of ``--dataset_root`` as one sequence and runs
``nframe`` frames through the model at once; frame 0 of the batch is overwritten by the inpainted
reference (``inpainted/``) and its output is dropped. A 279-view scene does not fit one 24-frame
batch, and their ``--reference_split N`` (interleaved groups ``views[j::N]``, a duplicate of the
group's first frame prepended as the slot the reference overwrites) only yields equal-size
batches when N divides the view count. So we materialise the same interleaved groups as separate
sequences of exactly ``nframe`` files:

    <root>/g<j>/images/00_ref.png         dummy slot (the reference photo; replaced by inpainted/)
    <root>/g<j>/images/<k>_<stem>.png     views[j::G], padded with the group's own views to nframe-1
    <root>/g<j>/masks/...                 same names, dilated object footprint (official_inputs)
    <root>/g<j>/inpainted/<ref>.png       the shared Qwen-Image-Edit reference
    <root>/groups.json                    group -> stems (padding marked) for compose.py

The ``<k>_`` prefix keeps their filename sort in trajectory order with the dummy first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="output/official_inputs/<scene>.")
    parser.add_argument("--root", type=Path, required=True, help="The --dataset_root to create.")
    parser.add_argument("--nframe", type=int, default=24)
    args = parser.parse_args()

    views = json.loads((args.inputs / "views.json").read_text())
    stems, ref = views["stems"], views["reference"]
    per_group = args.nframe - 1
    n_groups = math.ceil(len(stems) / per_group)
    groups = {}
    for j in range(n_groups):
        members = stems[j::n_groups]
        padded = members + [members[i % len(members)] for i in range(per_group - len(members))]
        g = args.root / f"g{j:02d}"
        for sub in ("images", "masks", "inpainted"):
            (g / sub).mkdir(parents=True, exist_ok=True)

        def link(src: Path, dst: Path) -> None:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src, dst)

        link(args.inputs / "images" / f"{ref}.png", g / "images" / "00_ref.png")
        link(args.inputs / "masks" / f"{ref}.png", g / "masks" / "00_ref.png")
        for k, stem in enumerate(padded, start=1):
            link(args.inputs / "images" / f"{stem}.png", g / "images" / f"{k:02d}_{stem}.png")
            link(args.inputs / "masks" / f"{stem}.png", g / "masks" / f"{k:02d}_{stem}.png")
        link(args.inputs / "reference" / f"{ref}.png", g / "inpainted" / f"{ref}.png")
        groups[g.name] = {"stems": padded, "n_real": len(members)}
    (args.root / "groups.json").write_text(json.dumps(groups, indent=1))
    print(f"{len(stems)} views -> {n_groups} groups of {per_group} (+ reference slot) under {args.root}")


if __name__ == "__main__":
    main()
