#!/usr/bin/env python3
"""One-off: gather every object-removal artifact of a scene under <scene>/removal/.

Reconstruction outputs (3dgrut_input, 3dgrut_runs, recon_results, captions, metric_alignment,
split.json) stay at the scene root, where prepare_colmap_artifixer_inputs expects them.
Absolute paths inside moved JSON files and symlink targets are rewritten to the new layout.
"""

import os
import re
import shutil
import sys
from pathlib import Path

OUT = Path("/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/mip360")


def plan(scene: str) -> list[tuple[str, str]]:
    root = OUT / scene
    moves = [("flashsplat", "removal/flashsplat"), ("flashsplat_v5", "removal/flashsplat_vase"),
             ("removal", "removal/v0/input"), ("artifixer_removal", "removal/v0/artifixer"),
             ("lama_seed", "removal/propagation/lama_seed_frames"),
             ("propagated_lama", "removal/propagation/lama_seed"),
             ("af3d_lama", "removal/af3d/lama_seed")]
    for v in range(1, 10):
        moves += [(f"removal_v{v}", f"removal/v{v}/input"), (f"artifixer_removal_v{v}", f"removal/v{v}/artifixer")]
    for v in ("v3", "v5"):
        moves += [(f"propagated_{v}", "removal/propagation/artifixer_seed"), (f"af3d_{v}", "removal/af3d/artifixer_seed")]
    return [(old, new) for old, new in moves if (root / old).exists()]


def main() -> None:
    for scene in sys.argv[1:]:
        root = OUT / scene
        moves = plan(scene)
        # Old absolute path -> new absolute path, applied in ONE regex pass so a rewritten path
        # is never matched again (e.g. removal_v1/ -> removal/v1/input/ must not then hit removal/).
        mapping = {str(root / old): str(root / new) for old, new in moves}
        pattern = re.compile("|".join(re.escape(o) for o in sorted(mapping, key=len, reverse=True)) + r'(?=[/"]|$)')

        def rewrite(text: str) -> str:
            return pattern.sub(lambda m: mapping[m.group(0)], text)

        # "removal" is both a source and the new parent: park it first.
        moves = [("_removal_v0_tmp" if old == "removal" else old, new) for old, new in moves]
        if (root / "removal").is_dir():
            (root / "removal").rename(root / "_removal_v0_tmp")
        (root / "removal").mkdir(exist_ok=True)
        for old, new in moves:
            (root / new).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root / old), str(root / new))
            print(f"{scene}: {old} -> {new}")

        for dirpath, _, filenames in os.walk(root / "removal"):
            for name in filenames:
                path = Path(dirpath) / name
                if path.is_symlink():
                    target = os.readlink(path)
                    new_target = rewrite(target)
                    if new_target != target:
                        path.unlink()
                        path.symlink_to(new_target)
                elif name.endswith(".json"):
                    text = path.read_text()
                    new_text = rewrite(text)
                    if new_text != text:
                        path.write_text(new_text)


if __name__ == "__main__":
    main()
