#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SurfaceLift stage 2 (artifixer env): observation ledger -> the cleaned removed scene
(SurfaceLift scripts/ledger.py --rule consistency, ``decide_consistency``).

1. seen_i = weight of every full-model Gaussian on real pixels (outside the object masks), all views.
2. Start from the removal's kept Gaussians + the deleted ones that real pixels saw (seen >= min_seen).
3. ``passes`` times: every revealed pixel (inside a view's mask, where the candidate scene renders
   content) is checked against the other photos (surflift.ledger.revealed_consistency); a Gaussian
   is dropped when its weight on failing pixels exceeds its weight on passing ones plus
   ``seen_weight`` x seen_i.
Everything kept is bit-identical to the full model.

-> <work>/ledger/{seen.npy, keep.npy (bool over the full model), dropped.npy, stats.json,
   check_<i>.jpg: photo | full | removal | cleaned}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from surflift.common import Scene, Timer, job_summary, subset, to_u8
from surflift.ledger import Witnesses, observation_ledger, revealed_consistency


def load_cleaned(sc: Scene, work: Path):
    keep = np.load(work / "ledger" / "keep.npy")
    return subset(sc.full, torch.as_tensor(keep, device=sc.full.xyz.device)), keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--colmap_dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--min_seen", type=float, default=1.0)
    ap.add_argument("--min_revealed", type=float, default=0.5)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--col_tol", type=float, default=0.1)
    ap.add_argument("--seen_weight", type=float, default=1.0,
                    help="SurfaceLift mip360 v3c used 0.25, its benchmark 1.0 (0.25 deleted well-observed background on "
                         "forward-facing captures).")
    ap.add_argument("--check", type=int, default=4, help="Write this many check sheets (views with the largest masks).")
    args = ap.parse_args()
    w = args.work
    out = w / "ledger"
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer(w, "ledger")
    sc = Scene(args.checkpoint, args.colmap_dir, w)
    full = sc.full
    dev = full.xyz.device
    views = sc.views
    masks = lambda v: sc.mask(v.index)                                   # noqa: E731
    photos = lambda v: sc.photo(v.index)                                 # noqa: E731
    led = observation_ledger(sc, full, views, masks)
    seen = led["seen"]
    np.save(out / "seen.npy", seen)
    background = torch.load(w / "removal.pt")["background"].numpy()
    deleted = ~background
    restore = deleted & (seen >= args.min_seen)
    keep = background | restore
    wit = Witnesses(sc, full, views, masks, photos)
    dropped = np.zeros_like(keep)
    rounds = []
    for r in range(args.passes):
        cand = subset(full, torch.as_tensor(keep, device=dev))
        maps, summ = revealed_consistency(sc, cand, views, masks, wit, col_tol=args.col_tol)
        led = observation_ledger(sc, cand, views, masks, split=lambda v: maps[v.index][::-1])   # A = fail, B = pass
        fail = np.zeros(len(keep), np.float32)
        ok = np.zeros(len(keep), np.float32)
        fail[keep], ok[keep] = led["masked"], led["seen"]
        bad = keep & (fail >= args.min_revealed) & (fail > ok + args.seen_weight * seen)
        rounds.append({"pass": r, "candidates": int(keep.sum()), **summ, "dropped": int(bad.sum())})
        print(rounds[-1], flush=True)
        dropped |= bad
        keep &= ~bad
        del maps
        if not bad.any():
            break
    np.save(out / "keep.npy", keep)
    np.save(out / "dropped.npy", dropped)
    stats = {"rule": "consistency", "full": int(len(keep)), "removed_scene": int(background.sum()),
             "deleted_by_removal": int(deleted.sum()), "restored_seen": int((restore & keep).sum()),
             "dropped_from_removed_scene": int((dropped & background).sum()), "cleaned_scene": int(keep.sum()),
             "passes": rounds, "witness_scale": wit.s,
             "thresholds": {"min_seen": args.min_seen, "col_tol": args.col_tol, "seen_weight": args.seen_weight,
                            "min_revealed": args.min_revealed}}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    wit_s = wit.s
    del wit
    torch.cuda.empty_cache()

    clean = subset(full, torch.as_tensor(keep, device=dev))
    removed = subset(full, torch.as_tensor(background, device=dev))
    order = sorted(views, key=lambda v: -sc.mask(v.index).sum())[:args.check]
    with torch.no_grad():
        for v in order:
            ims = [("photo", sc.photo(v.index)), ("full", to_u8(sc.trace(v, full, geometry=False)["rgb"])),
                   ("removal", to_u8(sc.trace(v, removed, geometry=False)["rgb"])),
                   ("cleaned (ledger)", to_u8(sc.trace(v, clean, geometry=False)["rgb"]))]
            wd = 480
            hd = round(v.H * wd / v.W)
            canvas = Image.new("RGB", (wd * len(ims), hd + 20), "white")
            d = ImageDraw.Draw(canvas)
            for k, (name, im) in enumerate(ims):
                canvas.paste(Image.fromarray(im).resize((wd, hd)), (k * wd, 20))
                d.text((k * wd + 6, 4), f"{name} [{v.index}]", fill="black")
            canvas.save(out / f"check_{v.index:05d}.jpg", quality=88)
    job_summary(w, f"ledger: restored {stats['restored_seen']} / dropped {stats['dropped_from_removed_scene']} Gaussians of the "
                   f"removed scene ({stats['removed_scene']}); cleaned {stats['cleaned_scene']}; revealed px pass/fail per pass "
                   + ", ".join(f"{r['pass_px']}/{r['fail_px']}" for r in rounds) + f"; witnesses at 1/{wit_s}")
    timer.done(**{k: stats[k] for k in ("restored_seen", "dropped_from_removed_scene", "cleaned_scene")})


if __name__ == "__main__":
    main()
