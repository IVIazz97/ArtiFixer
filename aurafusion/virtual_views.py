#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion af360v, stage 0 (ArtiFixer env): virtual cameras that frame the whole hole.

On close-up captures the removed object is often cut by the image border, so the unseen mask, the
LaMa/AGDD reference and the SDEdit targets only cover part of the hole. This stage adds views in
which the entire object (and so the entire hole) is inside the image:

1. The object is the same removal as ``render_views`` (FlashSplat labels + hull filter, or
   ``--removal_pt``). Its extent: the object Gaussians' positions, trimmed at the 99.5th distance
   percentile (as the hull), boxed along their principal axes (8 corners) and grown by
   ``--box_grow``.
2. One candidate per real camera: re-aimed at the box centre (keeping the camera's up direction),
   then pulled back along the new viewing ray until every box corner projects at least
   ``--margin`` of the image side inside the border. Never moved closer than the real camera; same
   intrinsics and resolution as that camera.
3. Rejections, from renders of the removed scene and of the object alone: other geometry in front
   of the object (``--max_occlusion`` of its pixels), the camera inside or against geometry
   (``--max_near`` of the image closer than ``--near_ratio`` x the distance to the hole), an
   empty view outside the reconstruction (mean opacity below that of the real view minus 0.1,
   or below ``--min_opacity``), pull-back beyond ``--max_pullback`` x the real distance.
4. ``--num_views`` of the survivors, farthest-point sampled on their viewing directions (from the
   one closest to its real camera), ordered by source camera so the SAM2 "video" stays smooth.
   If fewer than ``--min_views`` survive, the least-bad rejected ones fill up (logged).
5. The full scene (object present) rendered from each: the pseudo-photo.

Writes a derived COLMAP scene ``--output_dir/colmap_virtual``: the real views (symlinks, same
order) followed by the virtual ones, named ``zzz_virtual_<k>.png`` so they sort last (3DGRUT
orders views by name), reusing the source camera's COLMAP camera id; every images* folder gets
the pseudo-photo at that folder's resolution. Also ``virtual.json`` (num_real, virtual indices
and names, source view, pull-back, rejection stats) and ``virtual_views.png`` (a contact sheet).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from aurafusion.render_views import removal_keep_masks
from aurafusion.scene import all_views_overrides, build_dataset, load_model, render_view, view_camera
from flashsplat import export


def object_box(positions: np.ndarray, trim_percentile: float, grow: float) -> tuple[np.ndarray, np.ndarray]:
    """Centre [3] and the 8 corners [8, 3] of the PCA-aligned box of the trimmed points."""
    dist = np.linalg.norm(positions - positions.mean(0), axis=1)
    positions = positions[dist <= np.percentile(dist, trim_percentile)]
    mean = positions.mean(0)
    _, _, axes = np.linalg.svd(positions - mean, full_matrices=False)
    local = (positions - mean) @ axes.T
    lo, hi = local.min(0), local.max(0)
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * (1 + grow)
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=np.float64)
    centre = mid @ axes + mean
    return centre, centre + (signs * half) @ axes


def aimed_pose(c2w: np.ndarray, centre: np.ndarray, corners: np.ndarray, camera, margin: float) -> tuple[np.ndarray, float, float]:
    """OpenCV c2w looking at ``centre`` with the camera's up, at the distance where every corner fits.

    Returns (c2w, distance to centre, real camera's distance to centre)."""
    position = c2w[:3, 3]
    forward = centre - position
    d0 = float(np.linalg.norm(forward))
    forward /= max(d0, 1e-9)
    down = c2w[:3, 1]
    right = np.cross(down, forward)
    if np.linalg.norm(right) < 1e-3:  # looking along its own up axis: keep its right axis instead
        right = c2w[:3, 0] - forward * float(c2w[:3, 0] @ forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=1)
    q = (corners - centre) @ rotation  # corner offsets in camera axes; camera at centre - d * forward
    mx, my = margin * camera.width, margin * camera.height
    room = {"+x": camera.width - mx - camera.cx, "-x": camera.cx - mx, "+y": camera.height - my - camera.cy, "-y": camera.cy - my}
    need = [float(-q[:, 2].min()) + 1e-3 * d0]  # every corner in front of the camera
    for qx, qy, qz in q:  # |f * q / (qz + d)| <= room  =>  d >= f |q| / room - qz
        need.append(camera.fx * abs(qx) / room["+x" if qx > 0 else "-x"] - qz)
        need.append(camera.fy * abs(qy) / room["+y" if qy > 0 else "-y"] - qz)
    distance = max(d0, max(need))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = centre - distance * forward
    return pose, distance, d0


def virtual_batch(template, c2w: np.ndarray):
    pose = torch.as_tensor(c2w, dtype=template.T_to_world.dtype, device=template.T_to_world.device)
    return dataclasses.replace(template, T_to_world=pose[: template.T_to_world.shape[1]][None].contiguous(),
                               rgb_gt=None, mask=None)


def farthest_point_order(directions: np.ndarray, start: int, count: int) -> list[int]:
    chosen = [start]
    best = 1 - directions @ directions[start]  # cosine distance to the chosen set
    while len(chosen) < min(count, len(directions)):
        nxt = int(np.argmax(best))
        chosen.append(nxt)
        best = np.minimum(best, 1 - directions @ directions[nxt])
    return chosen


def write_colmap(colmap_dir: Path, out: Path, real_names: list[str], virtual: list[dict], photos: list[np.ndarray]) -> None:
    from data_processing.prepare_colmap_artifixer_inputs import ColmapScene, image_basename, read_colmap_scene, write_sparse_model
    from threedgrut.datasets.utils import Image as ColmapImage

    scene = read_colmap_scene(colmap_dir / "sparse" / "0")
    by_name = {image_basename(image): image for image in scene.images}
    images = [by_name[name] for name in real_names]
    next_id = max(image.id for image in scene.images) + 1
    for k, view in enumerate(virtual):
        w2c = np.linalg.inv(np.asarray(view["c2w"]))
        x, y, z, w = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        images.append(ColmapImage(id=next_id + k, qvec=np.array([w, x, y, z]), tvec=w2c[:3, 3].copy(),
                                  camera_id=by_name[view["source_name"]].camera_id, name=view["name"],
                                  xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), dtype=np.int64)))
    if out.exists():
        shutil.rmtree(out)
    (out / "sparse" / "0").mkdir(parents=True)
    write_sparse_model(colmap_dir / "sparse" / "0", out / "sparse" / "0", ColmapScene(cameras=scene.cameras, images=images))
    for folder in sorted(f for f in colmap_dir.glob("images*") if f.is_dir()):
        (out / folder.name).mkdir()
        for name in real_names:
            if (folder / name).exists():
                (out / folder.name / name).symlink_to((folder / name).resolve())
        for view, photo in zip(virtual, photos):
            source = folder / view["source_name"]
            if not source.exists():
                continue
            with Image.open(source) as image:
                size = image.size
            pseudo = Image.fromarray(photo)
            if pseudo.size != size:
                pseudo = pseudo.resize(size, Image.LANCZOS)
            pseudo.save(out / folder.name / view["name"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True, help="The real views (e.g. run_removal.sh PCOLMAP).")
    parser.add_argument("--flashsplat_dir", type=Path, default=None)
    parser.add_argument("--removal_pt", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--object_id", type=int, default=1)
    parser.add_argument("--hull_expand", type=float, default=0.0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_views", type=int, default=24)
    parser.add_argument("--min_views", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.06, help="Border kept free around the object, fraction of the side.")
    parser.add_argument("--box_grow", type=float, default=0.05)
    parser.add_argument("--max_pullback", type=float, default=4.0)
    parser.add_argument("--max_occlusion", type=float, default=0.15)
    parser.add_argument("--near_ratio", type=float, default=0.3)
    parser.add_argument("--max_near", type=float, default=0.3)
    parser.add_argument("--min_opacity", type=float, default=0.6)
    parser.add_argument("--summary", type=Path, default=None, help="Append a one-line human-readable summary here.")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    model, conf, _ = load_model(args.checkpoint, all_views_overrides(args.colmap_dir))
    dataset, loader = build_dataset(conf)
    if args.removal_pt is not None:
        removal = torch.load(args.removal_pt, map_location="cpu")
        removed = removal["removed"].to(model.positions.device)
        foreground = removal["footprint"].to(removed.device) & removed
        background = ~removed
    else:
        assert args.flashsplat_dir is not None, "give --flashsplat_dir or --removal_pt"
        labels = torch.load(args.labels or args.flashsplat_dir / "labels.pt", map_location="cpu")
        contribution = torch.load(args.flashsplat_dir / "contribution.pt", map_location="cpu")
        hit_count = torch.load(args.flashsplat_dir / "hit_count.pt", map_location="cpu")
        foreground, background, _ = removal_keep_masks(model, labels, contribution, hit_count, args.object_id,
                                                       hull_expand=args.hull_expand)
    if int(foreground.sum()) < 10:
        raise SystemExit(f"virtual_views: only {int(foreground.sum())} object Gaussians in the removal "
                         f"({args.removal_pt or args.flashsplat_dir}); cannot place cameras around the object")
    print(f"removal: {int(foreground.sum())} object Gaussians, {int(background.sum())} kept background Gaussians")
    centre, corners = object_box(model.positions.detach()[foreground].cpu().numpy().astype(np.float64), 99.5, args.box_grow)
    radius = float(np.linalg.norm(corners - centre, axis=1).max())
    print(f"object box centre {np.round(centre, 3).tolist()}, half-diagonal {radius:.3f}")
    object_model = export.segment_model(model, foreground, setup_optimizer=False)
    removed_model = export.segment_model(model, background, setup_optimizer=False)
    object_model.renderer = removed_model.renderer = model.renderer

    candidates, templates, cameras = [], [], []
    for index, batch in enumerate(loader):
        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
        camera = view_camera(dataset, index, gpu_batch)
        cameras.append(camera)
        templates.append(gpu_batch)
        real_opacity = float(render_view(removed_model, gpu_batch, index)["opacity"].mean())
        pose, distance, d0 = aimed_pose(camera.c2w, centre, corners, camera, args.margin)
        vb = virtual_batch(gpu_batch, pose)
        bg, obj = render_view(removed_model, vb, index), render_view(object_model, vb, index)
        on_object = obj["opacity"] > 0.5
        occluded = (bg["opacity"] > 0.5) & (bg["depth"] < obj["depth"] - 0.1 * radius) & on_object
        occlusion = float(occluded.sum()) / max(int(on_object.sum()), 1)
        near = float(((bg["opacity"] > 0.5) & (bg["depth"] < args.near_ratio * (distance - radius))).float().mean())
        opacity = float(bg["opacity"].mean())
        floor = min(args.min_opacity, real_opacity - 0.1)
        reasons = []
        if distance > args.max_pullback * d0:
            reasons.append("pullback")
        if not on_object.any():
            reasons.append("no_object")
        if occlusion > args.max_occlusion:
            reasons.append("occluded")
        if near > args.max_near:
            reasons.append("near_geometry")
        if opacity < floor:
            reasons.append("empty")
        penalty = (occlusion / args.max_occlusion + near / args.max_near + max(0.0, floor - opacity) * 10
                   + max(0.0, distance / d0 / args.max_pullback - 1) + (0 if on_object.any() else 100))
        candidates.append({"source_index": index, "source_name": camera.name, "c2w": pose.tolist(),
                           "distance": distance, "pullback": distance / d0, "occlusion": occlusion, "near": near,
                           "opacity": opacity, "real_opacity": real_opacity, "rejected": reasons, "penalty": penalty})
        if index % 50 == 0 or reasons:
            print(f"candidate {index + 1}/{len(dataset)}: pullback {distance / d0:.2f} occlusion {occlusion:.2f} "
                  f"near {near:.2f} opacity {opacity:.2f} {reasons or 'ok'}", flush=True)

    valid = [i for i, c in enumerate(candidates) if not c["rejected"]]
    counts = {r: sum(r in c["rejected"] for c in candidates) for r in ("pullback", "no_object", "occluded", "near_geometry", "empty")}
    print(f"{len(valid)}/{len(candidates)} candidates pass (one per real camera); rejected: {counts} "
          f"(thresholds: occlusion > {args.max_occlusion}, near > {args.max_near}, pull-back > {args.max_pullback}x, "
          f"opacity < min({args.min_opacity}, real view - 0.1))")
    pool = valid
    if len(valid) < args.min_views:
        pool = sorted(range(len(candidates)), key=lambda i: candidates[i]["penalty"])[:max(args.min_views, len(valid))]
        print(f"WARNING: fewer than {args.min_views} candidates pass; using the {len(pool)} least-bad ones")
    directions = np.stack([np.asarray(candidates[i]["c2w"])[:3, 2] for i in pool])
    start = min(range(len(pool)), key=lambda k: candidates[pool[k]]["pullback"])
    if not pool:
        raise SystemExit("virtual_views: no candidate camera at all (no real views in --colmap_dir?)")
    picked = sorted(pool[k] for k in farthest_point_order(directions, start, args.num_views))

    real_names = [c.name for c in cameras]
    virtual, photos = [], []
    for k, i in enumerate(picked):
        view = dict(candidates[i], name=f"zzz_virtual_{k:03d}.png")
        assert all(name < view["name"] for name in real_names), f"a real image name sorts after {view['name']}"
        rendered = render_view(model, virtual_batch(templates[view["source_index"]], np.asarray(view["c2w"])), view["source_index"])
        photos.append((rendered["rgb"].cpu().numpy() * 255).round().astype(np.uint8))
        virtual.append(view)
    write_colmap(args.colmap_dir, out / "colmap_virtual", real_names, virtual, photos)
    num_real = len(real_names)
    for k, view in enumerate(virtual):
        view["index"] = num_real + k
    (out / "virtual.json").write_text(json.dumps({
        "num_real": num_real, "virtual_indices": [v["index"] for v in virtual], "views": virtual,
        "object_centre": centre.tolist(), "object_corners": corners.tolist(), "margin": args.margin,
        "passed": len(valid), "candidates": len(candidates), "rejected": counts,
        "rejected_candidates": [{k: c[k] for k in ("source_index", "pullback", "occlusion", "near", "opacity", "rejected")}
                                for c in candidates if c["rejected"]],
    }, indent=1))

    # Contact sheet of the pseudo-photos (6 per row, 240 px wide).
    thumbs = [Image.fromarray(p).resize((240, max(1, round(240 * p.shape[0] / p.shape[1])))) for p in photos]
    th = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (240 * min(6, len(thumbs)), th * ((len(thumbs) + 5) // 6)))
    for k, t in enumerate(thumbs):
        sheet.paste(t, (240 * (k % 6), th * (k // 6)))
    sheet.save(out / "virtual_views.png")
    fallback = sum(1 for v in virtual if v["rejected"])
    for v in virtual:
        print(f"virtual view {v['index']} ({v['name']}): from real view {v['source_index']} ({v['source_name']}), "
              f"pull-back {v['pullback']:.2f}x, occlusion {v['occlusion']:.2f}, near {v['near']:.2f}, "
              f"opacity {v['opacity']:.2f}{' FALLBACK ' + ','.join(v['rejected']) if v['rejected'] else ''}")
    rejected_text = ", ".join(f"{n} {r}" for r, n in counts.items() if n) or "none"
    line = (f"af360v: {len(virtual)} virtual cams from {len(candidates)} candidates ({len(candidates) - len(valid)} rejected: "
            f"{rejected_text}{f'; {fallback} picked from the rejected as fallback' if fallback else ''}), "
            f"pull-back {min(v['pullback'] for v in virtual):.2f}-{max(v['pullback'] for v in virtual):.2f}x, "
            f"object box half-diagonal {radius:.3f}")
    print(line)
    if args.summary is not None:
        with open(args.summary, "a") as f:
            f.write(line + "\n")
    print(f"wrote {out / 'colmap_virtual'}, {out / 'virtual.json'}, {out / 'virtual_views.png'}")
    print(f"Done: {num_real} real + {len(virtual)} virtual views (sources {[v['source_index'] for v in virtual]}, "
          f"pull-back {np.round([v['pullback'] for v in virtual], 2).tolist()}) -> {out / 'colmap_virtual'}")


if __name__ == "__main__":
    main()
