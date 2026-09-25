#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lay a mip360 scene out as the unmodified Omni-3DEdit test loader (``seva/seva_test_dataset_edit.py``) reads it.

Their loader globs ``<json_folder>/{dataset}_{editing_type}_*.json`` (the remove config allows
dataset "demo", type "remove"), keeps the first successful pair as the fixed edited reference and
walks the rest in chunks of 9 (padding the last chunk cyclically), so one JSON covers the whole
scene: 1 reference + all other views, in name (= capture trajectory) order. Paths are relative
to the JSON folder, as in their demo. Omni-3DEdit is mask-free and pose-free (VGGT per chunk);
the only edited input it reads is the reference's ``edited_path``. For the other views the loader
also loads ``edited_path`` as the target slot, but only frames flagged as inputs are encoded
(``SevaConditioner.forward``), so we point it at the source photo.

    <root>/jsons/demo_remove_<scene>.json
    <root>/remove/<scene>/source_view/<stem>.png  -> official_inputs images
    <root>/remove/<scene>/cond_view/<ref>.png     -> the shared Qwen-Image-Edit reference
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="output/official_inputs/<scene>.")
    parser.add_argument("--root", type=Path, required=True, help="The json_folder's parent to create.")
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()

    views = json.loads((args.inputs / "views.json").read_text())
    ref = views["reference"]
    order = [ref] + [s for s in views["stems"] if s != ref]
    src = args.root / "remove" / args.scene / "source_view"
    cond = args.root / "remove" / args.scene / "cond_view"
    for d in (src, cond, args.root / "jsons"):
        d.mkdir(parents=True, exist_ok=True)

    def link(target: Path, dst: Path) -> None:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(target, dst)

    for stem in order:
        link(args.inputs / "images" / f"{stem}.png", src / f"{stem}.png")
    link(args.inputs / "reference" / f"{ref}.png", cond / f"{ref}.png")
    rel = f"../remove/{args.scene}"
    pairs = [{
        "original_path": f"{rel}/source_view/{stem}.png",
        "edited_path": f"{rel}/{'cond_view' if stem == ref else 'source_view'}/{stem}.png",
        "original_source_path": "", "success": True, "device_id": 0,
    } for stem in order]
    (args.root / "jsons" / f"demo_remove_{args.scene}.json").write_text(json.dumps({
        "editing_type": "remove", "instruction": "", "scene_name": args.scene,
        "num_images": len(pairs), "successful_edits": len(pairs), "image_pairs": pairs,
    }, indent=2))
    print(f"{len(pairs)} views (reference {ref} first, {-(-(len(pairs) - 1) // 9)} chunks) -> {args.root}")


if __name__ == "__main__":
    main()
