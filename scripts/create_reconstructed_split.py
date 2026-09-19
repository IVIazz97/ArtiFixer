#!/usr/bin/env python3
"""Write split.json for a reconstructed COLMAP render trajectory."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SCENE_ROOT")

    scene_root = Path(sys.argv[1]).resolve()
    scene_id = scene_root.name
    prepared = scene_root / "3dgrut_input" / scene_id
    render_root = scene_root / "recon_results" / scene_id / "reconstruction" / scene_id / "ours_30000" / "transforms"
    checkpoint = scene_root / "3dgrut_runs" / scene_id / scene_id / "ours_30000" / "ckpt_30000.pt"
    selected_indices = render_root / "selected_indices.json"
    caption = scene_root / "captions" / scene_id / "caption.h5"
    scale_info = scene_root / "metric_alignment" / "scale_info.txt"

    if not all(path.exists() for path in (prepared / "nerfstudio" / "transforms.json", render_root / "renders", render_root / "opacity", checkpoint, selected_indices, caption, scale_info)):
        raise SystemExit("incomplete reconstructed scene artifacts")

    scale = None
    for line in scale_info.read_text().splitlines():
        if line.startswith("Scale factor:"):
            scale = float(line.split(":", 1)[1].split()[0])
            break
    if scale is None:
        raise SystemExit(f"missing scale factor in {scale_info}")

    def relative(path: Path) -> str:
        return path.relative_to(scene_root).as_posix()

    entry = {
        "scene_id": scene_id,
        "transforms_path": relative(prepared / "nerfstudio" / "transforms.json"),
        "image_root": relative(prepared),
        "render_dir": relative(render_root / "renders"),
        "opacity_dir": relative(render_root / "opacity"),
        "selected_indices_path": relative(selected_indices),
        "prompt_path": relative(caption),
        "reconstruction_checkpoint": relative(checkpoint),
        "metric_scale": scale,
        "camera_scale": scale * 0.01,
    }
    (scene_root / "split.json").write_text(json.dumps({"test": {scene_id: entry}}, indent=2) + "\n")
    print(f"wrote {scene_root / 'split.json'}")


if __name__ == "__main__":
    main()
