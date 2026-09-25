#!/usr/bin/env python3
"""Side-by-side removal panel: our ArtiFixer result vs the official-code baselines, same views.

Rows are views spread over the capture trajectory (the reference view of the video-prior methods
first); columns are the photo and every method whose output exists (missing ones are skipped):

    ArtiFixer        output/artifixer/<scene>/<v>/artifixer/.../frames/batch_0000/pred/<i>.png
    AuraFusion360    CVPR2027/AuraFusion360/output/Other-360/<scene>[_robustdepth]/train/ours_10000_object_inpaint/renders/<i>.png
                     (_robustdepth, 0.99-quantile AGDD depth range, when it exists; else the unmodified official run)
    Inpaint360GS     CVPR2027/Inpaint360GS/output/others/<scene>/train/ours_object_inpaint_virtual/iteration_*/renders/<stem>.png
                     (training views only: held-out test views stay blank)
    MVInpainter 2D   CVPR2027/MVInpainter/output/mip360/<scene>/edited/<stem>.png  (before the 3DGS refit)
    MVInpainter 3DGS CVPR2027/MVInpainter/output/mip360/<scene>/3dgs/train/ours_30000/renders/<i>.png
    Omni-3DEdit 2D / 3DGS   same layout under CVPR2027/Omni3DEdit (and .../perchunk: a reference per 10-view chunk)

<i> is the index in name-sorted view order (all of these render the same COLMAP views).
    python output/jobs/panel_video_priors.py --scene kitchen   # -> output/panels/kitchen_video_priors.jpg
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from PIL import Image, ImageDraw

AF = Path(__file__).resolve().parents[2]
REPOS = AF.parent
ARTIFIXER_VARIANT = {"kitchen": "v3", "garden": "v5"}
REFERENCE = {"kitchen": "DSCF0687", "garden": "DSC08131"}


def columns(scene: str) -> dict[str, callable]:
    af = glob.glob(str(AF / f"output/artifixer/{scene}/{ARTIFIXER_VARIANT[scene]}/artifixer/*/*/{scene}/frames/batch_0000/pred"))
    cols = {
        f"ArtiFixer {ARTIFIXER_VARIANT[scene]}": (lambda i, s, d=Path(af[0]) if af else None: d / f"{i:05d}.png" if d else None),
    }
    af360 = REPOS / "AuraFusion360/output/Other-360"
    robust = af360 / f"{scene}_robustdepth/train/ours_10000_object_inpaint/renders"
    name = "AuraFusion360 robust depth" if robust.is_dir() else "AuraFusion360"
    cols[name] = lambda i, s, d=robust if robust.is_dir() else af360 / f"{scene}/train/ours_10000_object_inpaint/renders": d / f"{i:05d}.png"
    i360 = sorted(glob.glob(str(REPOS / f"Inpaint360GS/output/others/{scene}/train/ours_object_inpaint_virtual/iteration_*/renders")))
    cols["Inpaint360GS"] = lambda i, s, d=Path(i360[-1]) if i360 else None: d / f"{s}.png" if d else None
    for name, repo, sub in (("MVInpainter", "MVInpainter", ""), ("Omni-3DEdit", "Omni3DEdit", ""),
                            ("Omni-3DEdit per-chunk ref", "Omni3DEdit", "perchunk")):
        out = REPOS / repo / "output/mip360" / scene / sub
        cols[f"{name} 2D"] = lambda i, s, o=out: o / "edited" / f"{s}.png"
        cols[f"{name} 3DGS"] = lambda i, s, o=out: o / "3dgs/train/ours_30000/renders" / f"{i:05d}.png"
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=sorted(REFERENCE))
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--width", type=int, default=360)
    args = parser.parse_args()

    photos = sorted((AF / f"output/recon/{args.scene}/3dgrut_input/{args.scene}/images").iterdir())
    stems = [p.stem for p in photos]
    ref = stems.index(REFERENCE[args.scene])
    picks = [(ref + k * len(stems) // args.rows) % len(stems) for k in range(args.rows)]
    cols = {"photo": lambda i, s: photos[i]}
    cols.update({n: f for n, f in columns(args.scene).items()
                 if any(f(i, stems[i]) and f(i, stems[i]).exists() for i in picks)})

    w = args.width
    h = round(w * Image.open(photos[0]).height / Image.open(photos[0]).width)
    canvas = Image.new("RGB", (w * len(cols), 24 + h * len(picks)), "white")
    draw = ImageDraw.Draw(canvas)
    for c, (name, path_of) in enumerate(cols.items()):
        draw.text((c * w + 6, 6), name, fill="black")
        for r, i in enumerate(picks):
            path = path_of(i, stems[i])
            if path and path.exists():
                canvas.paste(Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS), (c * w, 24 + r * h))
    out = AF / f"output/panels/{args.scene}_video_priors.jpg"
    canvas.save(out, quality=92)
    print(f"{out}: views {[stems[i] for i in picks]}, columns {list(cols)}")


if __name__ == "__main__":
    main()
