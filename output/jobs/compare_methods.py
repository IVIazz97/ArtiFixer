#!/usr/bin/env python3
"""Final-result comparison of every removal method on kitchen (lego), garden (table + vase) and bonsai (plant).

Writes, per scene:
    output/panels/<scene>_all_methods.jpg        static grid: rows = views, columns = final results
    output/panels/viewer/<scene>/<key>.jpg       one vertical strip per method (all picked views)
    output/panels/viewer/manifest.json           methods, views and tile size, for the HTML viewer

"Final" = what each method would hand over as its 3D result: the 3DGS/2DGS refit renders for the
baselines, ArtiFixer's own output for ours. The video-prior methods' 2D outputs (before their 3DGS
refit) are added as extra, non-final entries. Views are spread over the whole capture trajectory,
starting at the reference view of the video-prior methods, and restricted to views every final
method renders (Inpaint360GS renders only its training views).
    python output/jobs/compare_methods.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

AF = Path(__file__).resolve().parents[2]
REPOS = AF.parent
OUT = AF / "output/panels"
REFERENCE = {"kitchen": "DSCF0687", "garden": "DSC08131", "bonsai": None}   # bonsai: no video-prior runs
N_VIEWS, TILE_W = 24, 400


def first(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern))
    return Path(hits[0]) if hits else None


AF_MAIN = {"kitchen": "v3", "garden": "v5"}          # the ArtiFixer variant shown among the final results
AF_VARIANTS = {"v0": "mask +12px, opacity 0, refs = photo patched with render",
               "v1": "mask +12px, opacity = render, refs = photo patched",
               "v2": "mask +12px, opacity = render, refs = background render",
               "v3": "mask +12px, opacity 0, refs = background render",
               "v4": "hull +30px, opacity 0, refs = background render",
               "v5": "vase merged, mask +12px, opacity 0, refs = background render",
               "v6": "LaMa-filled render, full opacity", "v7": "LaMa-filled render, opacity 0",
               "v8": "hull +40px, opacity 0", "v9": "hull +80px, opacity 0"}
# Verdicts from looking at the 24 viewer views: (works | partly | fails) + one line. Missing = n/a.
VERDICTS = {
    "kitchen": {
        "artifixer_v3": ("mixed", "lego gone; grey placemat smeared and blurred in back views"),
        "aurafusion360_unmodified": ("fail", "black blob + floating shards (AGDD depth range broken by far floaters)"),
        "aurafusion360": ("good", "clean; placemat stripes a little soft"),
        "inpaint360gs": ("good", "good; one soft dent in the placemat"),
        "mvinpainter": ("good", "clean and sharp all around; faint stain on the pink mat"),
        "omni3dedit": ("mixed", "clean near the reference; pink smear where the lego was in far views"),
        "omni3dedit_perchunk": ("good", "clean in most views; a soft smudge where the lego was in a few"),
        "port_aurafusion": ("fail", "brown blob and leftover bucket pieces"),
        "port_aurafusion_hull": ("mixed", "lego gone; pale washed-out patch on the placemats"),
        "port_inpaint360gs": ("good", "clean; filled patch a little blurry"),
        "port_inpaint360gs_fs": ("good", "clean; filled patch a little blurry"),
    },
    "garden": {
        "artifixer_v5": ("fail", "glowing ring around the table footprint + vase stump"),
        "aurafusion360_unmodified": ("fail", "shards + dark smear"),
        "aurafusion360": ("mixed", "table gone; soft dark patch under the legs"),
        "inpaint360gs": ("mixed", "paving kept; dark smudge under the legs"),
        "mvinpainter": ("fail", "far views invent new objects; 3DGS averages them into a dark blur"),
        "omni3dedit": ("fail", "far views re-generate the table; ghost table in the 3DGS"),
        "omni3dedit_perchunk": ("fail", "blurry ghost table in every view"),
        "artifixer3d": ("fail", "pale ghost table in every view"),
        "artifixer3d_lama": ("fail", "pale ghost table in every view"),
        "port_aurafusion": ("mixed", "table gone; dark X-shaped smear under the legs"),
        "port_aurafusion_hull": ("fail", "orange / pale blob where the table stood"),
        "port_inpaint360gs": ("fail", "vase stays (segmentation misses it); grass smeared over the paving"),
        "port_inpaint360gs_fs": ("mixed", "table and vase gone; paving blurred into a smooth patch"),
    },
    "bonsai": {
        "aurafusion360_unmodified": ("fail", "smears"),
        "aurafusion360": ("mixed", "smears / haze where the plant was"),
        "inpaint360gs": ("mixed", "faint grey haze where the plant was"),
        "port_aurafusion": ("fail", "dark smear streaking off the stand"),
        "port_aurafusion_hull": ("fail", "orange smear across the view"),
        "port_inpaint360gs": ("fail", "dark smoke cloud where the plant was"),
        "port_inpaint360gs_fs": ("fail", "dark smoke cloud where the plant was"),
    },
}


# One line per scene for the top of the viewer: what works, from the verdicts above.
SUMMARY = {
    "kitchen": "Works: MVInpainter, AuraFusion360 robust depth, Inpaint360GS (official and both 3DGUT ports), Omni-3DEdit "
               "per chunk. Partly: ArtiFixer v3, Omni-3DEdit 1 ref. Fails: AuraFusion360 official.",
    "garden": "Nothing is clean. Closest: AuraFusion360 robust depth, Inpaint360GS and its FlashSplat port (dark or blurred "
              "patch under the legs). Fails: ArtiFixer v5 / ArtiFixer3D (ring, ghost table), the video priors (ghost table, "
              "invented objects), AuraFusion360 official.",
    "bonsai": "Nothing is clean: every method leaves a smear, haze or smoke where the plant was. Least bad: Inpaint360GS "
              "official (faint haze). No ArtiFixer or video-prior runs on bonsai.",
}


def methods(scene: str) -> list[dict]:
    """key, label, note, final?, path_of(index, stem) for every method with output on this scene."""
    by_index = lambda d: (lambda i, s: d / f"{i:05d}.png") if d else None
    by_stem = lambda d: (lambda i, s: d / f"{s}.png") if d else None
    photos = AF / f"output/recon/{scene}/3dgrut_input/{scene}/images"
    ext = next(photos.iterdir()).suffix
    names = sorted(p.name for p in photos.iterdir())
    i360_dirs = [first(str(REPOS / f"Inpaint360GS/output/others/{scene}/{split}/ours_object_inpaint_virtual/iteration_*/renders"))
                 for split in ("train", "test")]                    # training + held-out test views
    i360 = lambda i, s, ds=[d for d in i360_dirs if d]: next((d / f"{s}.png" for d in ds if (d / f"{s}.png").exists()),
                                                           Path("/nonexistent"))
    fs_dir = AF / f"output/flashsplat/{'garden_vase' if scene == 'garden' else scene}"
    fs_index = ({n: int(j) for j, n in json.loads((fs_dir / "frame_names.json").read_text()).items()}
                if (fs_dir / "frame_names.json").exists() else {})
    flashsplat = (lambda i, s: fs_dir / "background_renders" / f"{fs_index[names[i]]:05d}.png") if fs_index else None
    omni, mvi = REPOS / "Omni3DEdit/output/mip360" / scene, REPOS / "MVInpainter/output/mip360" / scene
    af_pred = lambda v: first(str(AF / f"output/artifixer/{scene}/{v}/artifixer/*/*/{scene}/frames/batch_0000/pred"))

    m = [("photo", "Input photo", "with the object", True, lambda i, s: photos / f"{s}{ext}")]
    main_v = AF_MAIN.get(scene)
    if main_v:
        m.append((f"artifixer_{main_v}", f"ArtiFixer {main_v}", "ours: " + AF_VARIANTS[main_v], True, by_index(af_pred(main_v))))
    if scene == "garden":
        m.append(("artifixer3d", "ArtiFixer3D", "ours: 3DGS on ArtiFixer-seeded views", True,
                  by_index(first(str(AF / "output/artifixer/garden/af3d/artifixer_seed/artifixer3d/recon_results/garden/artifixer3d/garden/ours_30000/renders")))))
    m += [
        ("aurafusion360_unmodified", "AuraFusion360 official", "official 2DGS code as released", True,
         by_index(REPOS / f"AuraFusion360/output/Other-360/{scene}/train/ours_10000_object_inpaint/renders")),
        ("aurafusion360", "AuraFusion360 robust depth", "official 2DGS, 0.99-quantile AGDD depth range", True,
         by_index(REPOS / f"AuraFusion360/output/Other-360/{scene}_robustdepth/train/ours_10000_object_inpaint/renders")),
        ("inpaint360gs", "Inpaint360GS", "official 3DGS: LaMa + virtual views", True, i360),
        ("mvinpainter", "MVInpainter", "official: AnimateDiff video prior, 1 Qwen ref → 3DGS", True, by_index(mvi / "3dgs/train/ours_30000/renders")),
        ("omni3dedit", "Omni-3DEdit (1 ref)", "official: SEVA video prior, 1 Qwen ref → 3DGS", True,
         by_index(omni / "3dgs/train/ours_30000/renders")),
        ("omni3dedit_perchunk", "Omni-3DEdit (ref per chunk)", "authors' 360-USID protocol: a Qwen ref per 10 views → 3DGS", True,
         by_index(omni / "perchunk/3dgs/train/ours_30000/renders")),
        ("flashsplat", "Removal only", "FlashSplat background render, no inpainting (the hole to fill)", False, flashsplat),
        ("port_aurafusion", "AuraFusion360 on 3DGUT", "our port (aurafusion/): LaMa reference, official hull", False,
         by_index(AF / f"output/aurafusion/{scene}/lama_ref/final/rgb")),
        ("port_aurafusion_hull", "AuraFusion360 on 3DGUT (hull)", "our port: removal = FlashSplat + convex hull", False,
         by_index(AF / f"output/aurafusion/{scene}/hull/final/rgb")),
        ("port_inpaint360gs", "Inpaint360GS on 3DGUT", "our port (inpaint360gs/): paper segmentation + distillation", False,
         by_index(AF / f"output/inpaint360gs/{scene}/distill/final/rgb")),
        ("port_inpaint360gs_fs", "Inpaint360GS on 3DGUT (FlashSplat)", "our port: FlashSplat removal, hull x1.3", False,
         by_index(AF / f"output/inpaint360gs/{scene}/flashsplat_hull13/final/rgb")),
    ]
    for v, desc in AF_VARIANTS.items():
        if v != main_v:
            m.append((f"artifixer_{v}", f"ArtiFixer {v}", "ours: " + desc, False, by_index(af_pred(v))))
    if scene == "garden":
        m.append(("artifixer3d_lama", "ArtiFixer3D (LaMa seed)", "ours: 3DGS on LaMa-seeded views", False,
                  by_index(first(str(AF / "output/artifixer/garden/af3d/lama_seed/artifixer3d/recon_results/garden/artifixer3d/garden/ours_30000/renders")))))
    m += [
        ("mvinpainter_2d", "MVInpainter 2D", "its 2D output, before the 3DGS refit", False, by_stem(mvi / "edited")),
        ("omni3dedit_2d", "Omni-3DEdit (1 ref) 2D", "its 2D output, before the 3DGS refit", False, by_stem(omni / "edited")),
        ("omni3dedit_perchunk_2d", "Omni-3DEdit (ref per chunk) 2D", "its 2D output, before the 3DGS refit", False,
         by_stem(omni / "perchunk/edited")),
    ]
    level_text = VERDICTS.get(scene, {})
    return [{"key": k, "label": lab, "note": note, "final": fin, "path_of": f,
             "verdict": dict(zip(("level", "text"), level_text[k])) if k in level_text else None}
            for k, lab, note, fin, f in m if f is not None]


def pick_views(scene: str, ms: list[dict]) -> tuple[list[str], list[int]]:
    photos = sorted((AF / f"output/recon/{scene}/3dgrut_input/{scene}/images").iterdir())
    stems = [p.stem for p in photos]
    finals = [m for m in ms if m["final"]]
    ok = [i for i, s in enumerate(stems) if all(m["path_of"](i, s).exists() for m in finals if m["key"] != "omni3dedit_perchunk")]
    ref = stems.index(REFERENCE[scene]) if REFERENCE[scene] else 0
    ok.sort(key=lambda i: (i - ref) % len(stems))       # trajectory order, starting at the reference
    step = len(ok) / N_VIEWS
    picks = [ok[int(k * step)] for k in range(min(N_VIEWS, len(ok)))]
    return [stems[i] for i in picks], picks


def tile(path: Path, w: int, h: int) -> Image.Image:
    if path.exists():
        return Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)
    t = Image.new("RGB", (w, h), (40, 40, 40))
    ImageDraw.Draw(t).text((w // 2 - 40, h // 2 - 6), "not available", fill=(170, 170, 170))
    return t


def font(size: int):
    for f in ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def main() -> None:
    manifest = {"tile_w": TILE_W, "scenes": {}}
    for scene in REFERENCE:
        ms = methods(scene)
        ms = [m for m in ms if m["key"] == "photo" or any(m["path_of"](i, s).exists()
              for i, s in enumerate(p.stem for p in sorted((AF / f"output/recon/{scene}/3dgrut_input/{scene}/images").iterdir())))]
        stems, idx = pick_views(scene, ms)
        ph = Image.open(ms[0]["path_of"](idx[0], stems[0]))
        th = round(TILE_W * ph.height / ph.width)
        vdir = OUT / "viewer" / scene
        vdir.mkdir(parents=True, exist_ok=True)
        for m in ms:
            strip = Image.new("RGB", (TILE_W, th * len(idx)))
            for r, (i, s) in enumerate(zip(idx, stems)):
                strip.paste(tile(m["path_of"](i, s), TILE_W, th), (0, r * th))
            strip.save(vdir / f"{m['key']}.jpg", quality=84)
        manifest["scenes"][scene] = {
            "tile_h": th, "views": stems, "reference": REFERENCE[scene], "summary": SUMMARY.get(scene, ""),
            "methods": [{k: m[k] for k in ("key", "label", "note", "final", "verdict") if m[k] is not None} for m in ms]}

        # static panel: final results only, 6 views
        finals = [m for m in ms if m["final"]]
        rows = [k * len(idx) // 6 for k in range(6)]
        pw, phh, head = 300, round(300 * th / TILE_W), 54
        canvas = Image.new("RGB", (pw * len(finals), head + phh * len(rows)), "white")
        d = ImageDraw.Draw(canvas)
        for c, m in enumerate(finals):
            d.text((c * pw + 8, 6), m["label"], fill="black", font=font(17))
            d.text((c * pw + 8, 30), m["note"][:44], fill=(90, 90, 90), font=font(12))
            for r, k in enumerate(rows):
                canvas.paste(tile(m["path_of"](idx[k], stems[k]), pw, phh), (c * pw, head + r * phh))
        canvas.save(OUT / f"{scene}_all_methods.jpg", quality=90)
        print(f"{scene}: {len(ms)} methods, {len(idx)} views -> {OUT / f'{scene}_all_methods.jpg'}")
    (OUT / "viewer/manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
