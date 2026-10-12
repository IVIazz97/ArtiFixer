#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SurfaceLift stage 9 (artifixer env): multi-view colour fit of ONLY the lifted Gaussians to the
LeftRefill images inside each view's ledger hole, then export (SurfaceLift scripts/coin_fit.py
--targets leftrefill --views nontrivial --shadow: the surflift_leftrefill_shadow row).

Start: the v3c single-view fit (lift/lifted_fit.pt, before colour alignment) on the cleaned scene.
Observed Gaussians never change, except (shadow) the cleaned scene's Gaussians drawn mostly into
the shadow extensions S_v (weight in S >= 0.4 of their weight, ledger split), whose SH-DC colour is
trainable. The lifted geometry is frozen (uv slide, scales, angle); colour and opacity train.
``iters`` Adam steps, each one of:
  target   (1 - ref_weight - mv_real): a training view with a LeftRefill image, 0.8 L1 + 0.2 D-SSIM on
           dilate(H_v | S_v, 3) & C_v; only views with >= min_px such pixels (``nontrivial``)
  ref      (ref_weight): the chosen view with its own target (inpainted.png with lift.py's exposure
           gain) on dilate(U, max(6, margin)); with the shadow, also on its S (target: LaMa of the
           chosen view over U | S; the benchmark used its object-free reference photo there)
  mv_real  (mv_real): another photo, outside its object mask, wherever the fill shows
Then the harmonic colour alignment on the chosen view, and the export:
-> <work>/fit/{info.json, sheet.jpg}; <final>/ckpt_final.pt (removed scene, cleaned by the ledger,
   + the fill: a 3DGUT checkpoint, aurafusion/finetune.py layout, ``surflift_num_fixed`` = observed
   Gaussians first) and <final>/rgb/<i>.png for every view (rendered from the reloaded checkpoint).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation

from aurafusion.scene import all_views_overrides, build_dataset, load_model, render_view, save_png
from surflift.appearance import align_colour
from surflift.clean import load_cleaned
from surflift.common import Frozen, Scene, Timer, job_summary, load_u8, load_view, subset, to_u8
from surflift.holes import load_holes
from surflift.lift import ring_gain
from surflift.lifted import LiftedGaussians
from surflift.ledger import observation_ledger
from surflift.losses import gaussian_window, masked_l1_dssim


def box_of(m: np.ndarray, pad: int = 8):
    ys, xs = np.nonzero(m)
    return max(ys.min() - pad, 0), ys.max() + pad + 1, max(xs.min() - pad, 0), xs.max() + pad + 1


class Group:
    """The lifted fill + the shadow Gaussians (geometry, opacity, view-dependent SH frozen; SH-DC trainable)."""

    def __init__(self, lifted, sh: Frozen | None):
        self.l, self.s = lifted, sh
        self.s_dc = None if sh is None else torch.nn.Parameter(sh.features[:, :3].clone())

    def _cat(self, a, b):
        return a if self.s is None else torch.cat([a, b])

    def xyz(self): return self._cat(self.l.xyz(), self.s.xyz if self.s is not None else None)
    def rotation(self): return self._cat(self.l.rotation(), self.s.rot if self.s is not None else None)
    def scaling(self): return self._cat(self.l.scaling(), self.s.scale if self.s is not None else None)
    def opacity(self): return self._cat(self.l.opacity(), self.s.opacity if self.s is not None else None)

    def features(self, F):
        f = self.l.features(F)
        return f if self.s is None else torch.cat([f, torch.cat([self.s_dc, self.s.features[:, 3:]], 1)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--colmap_dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True, help="Output dir: ckpt_final.pt + rgb/")
    ap.add_argument("--targets", type=Path, default=None, help="Default <work>/leftrefill/refined.")
    ap.add_argument("--lama_model", type=Path, default=None, help="For the chosen view's shadow target.")
    ap.add_argument("--no_shadow", action="store_true")
    ap.add_argument("--shadow_ratio", type=float, default=0.4)
    ap.add_argument("--min_px", type=int, default=500)
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--ref_weight", type=float, default=0.2)
    ap.add_argument("--mv_real", type=float, default=0.2)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr_color", type=float, default=0.0025)
    ap.add_argument("--lr_opacity", type=float, default=0.01)
    ap.add_argument("--no_align", action="store_true")
    ap.add_argument("--no_guard", dest="guard", action="store_false",
                    help="Port addition (default on): also fit the fill to the cleaned scene's own render where the fill "
                         "shows over confirmed pixels of the object mask (outside the ledger hole).")
    args = ap.parse_args()
    t0 = time.time()
    w = args.work
    out = w / "fit"
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer(w, "fit")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    sc = Scene(args.checkpoint, args.colmap_dir, w)
    views = sc.views
    base, keep = load_cleaned(sc, w)
    clean_idx = np.nonzero(keep)[0]
    sel = w / "select"
    ref = load_view(sel / "camera.json")
    lifted = LiftedGaussians.load(w / "lift" / "lifted_fit.pt")
    for p in lifted.parameters():
        p.requires_grad_(False)
    lifted.color_dc.requires_grad_(True)
    lifted.opacity_logit.requires_grad_(True)
    holes, hinfo = load_holes(w)
    shape = tuple(hinfo["shape"])
    empty = np.zeros(shape, bool)
    shadow = not args.no_shadow
    sh = np.zeros(len(base), bool)
    shadow_info = None
    if shadow:
        Smap = {v.index: (holes[v.index]["S"] if v.index in holes and holes[v.index]["S"] is not None else empty) for v in views}
        led = observation_ledger(sc, base, views, None, split=lambda v: (Smap[v.index], ~Smap[v.index]))
        ins, outs = led["masked"], led["seen"]
        sh = (ins >= 1.0) & (ins >= args.shadow_ratio * (ins + outs))
        shadow_info = {"shadow_gaussians": int(sh.sum()), "S_px": int(sum(m.sum() for m in Smap.values()))}
        print("shadow", shadow_info, flush=True)
    shm = torch.as_tensor(sh, device=base.xyz.device)
    G = Group(lifted, subset(base, shm) if sh.any() else None)
    fixed = subset(base, ~shm)
    groups = [{"params": [lifted.color_dc], "lr": args.lr_color}, {"params": [lifted.opacity_logit], "lr": args.lr_opacity}]
    if G.s_dc is not None:
        groups.append({"params": [G.s_dc], "lr": args.lr_color})
    opt = torch.optim.Adam(groups, eps=1e-15)

    tdir = args.targets or w / "leftrefill" / "refined"
    cov_thr = hinfo["params"]["cov"]
    data, used, guard_px = [], [], []
    for v in views:
        if (not ref.virtual and v.index == ref.index) or v.index not in holes:
            continue
        p = tdir / f"{v.index:05d}.png"
        if not p.exists():
            continue
        h = holes[v.index]
        m = h["H"] | (h["S"] if (shadow and h["S"] is not None) else False)
        if G.s is not None:
            C = (sc.coverage(v, G, fixed) > cov_thr).cpu().numpy()
        else:
            C = h["C"]
        m = binary_dilation(m, iterations=args.dilate) & C if args.dilate else m & C
        if m.sum() < args.min_px:
            continue
        tgt = load_u8(p)
        assert tgt.shape[:2] == (v.H, v.W), (p, tgt.shape)
        n_guard = 0
        if args.guard:
            # guard: where the fill shows over what the photos confirmed (the object mask outside the ledger hole),
            # the target is the cleaned scene itself, so lifted Gaussians poking through there fade or take its colour
            cov = sc.coverage(v, G, fixed).cpu().numpy() > 0.05
            hm = h["H"] | (h["S"] if (shadow and h["S"] is not None) else False)
            guard = binary_dilation(sc.mask(v.index), iterations=4) & ~binary_dilation(hm, iterations=args.dilate + 2) & cov
            if guard.any():
                cl = to_u8(sc.trace(v, base, geometry=False)["rgb"])
                tgt = np.where(guard[..., None], cl, tgt)
                m = m | guard
                n_guard = int(guard.sum())
        guard_px.append(n_guard)
        data.append((v, torch.tensor(tgt, device="cuda").permute(2, 0, 1), torch.tensor(m, device="cuda"), box_of(m)))
        used.append(v.index)
    print(f"{len(data)} target views from {tdir}; guard px per view mean {np.mean(guard_px) if guard_px else 0:.0f}", flush=True)

    hole = np.asarray(Image.open(sel / "hole.png")) > 127
    rtarget = load_u8(sel / "inpainted.png").astype(np.float32) / 255
    ralpha = np.load(sel / "removed_alpha.npy")
    rr = load_u8(sel / "removed_render.png").astype(np.float32) / 255
    if not ref.virtual:
        rtarget = np.where(hole[..., None], np.clip(rtarget * ring_gain(rtarget, rr, hole, ralpha), 0, 1), rtarget)
    margin = json.loads((w / "lift" / "info.json").read_text()).get("margin", 0)
    rregion = binary_dilation(hole, iterations=max(6, margin))
    S_ref = holes[ref.index]["S"] if (shadow and not ref.virtual and ref.index in holes) else None
    if S_ref is not None and S_ref.any() and args.lama_model is not None and args.lama_model.exists():
        from surflift.inpaint import lama
        S_ref = binary_dilation(S_ref, iterations=args.dilate)
        src = load_u8(sel / "source.png")
        refimg = lama(args.lama_model, src, binary_dilation(hole | S_ref, iterations=4)).astype(np.float32) / 255
        refimg = np.clip(refimg * ring_gain(refimg, rr, hole | S_ref, ralpha), 0, 1)
        rtarget = np.where((S_ref & ~hole)[..., None], refimg, rtarget)
        rregion = rregion | S_ref
        Image.fromarray((rtarget * 255).round().astype(np.uint8)).save(out / "ref_target.png")
    rtgt = torch.tensor(rtarget, device="cuda").permute(2, 0, 1)
    rreg, rbox = torch.tensor(rregion, device="cuda"), box_of(rregion)

    win = gaussian_window()
    photo_cache, log, counts = {}, [], {"target": 0, "ref": 0, "mv_real": 0}
    for it in range(args.iters + 1):
        r = rng.random()
        if r < args.mv_real and it > 100:
            v = views[rng.integers(len(views))]
            if v.index not in photo_cache:
                cov = sc.coverage(v, G, fixed)
                M = torch.as_tensor(sc.mask(v.index), device="cuda")
                mm = (cov > 0.05) & ~M
                photo_cache[v.index] = None if mm.sum() < 20 else (torch.tensor(sc.photo(v.index), device="cuda").permute(2, 0, 1), mm)
            if photo_cache[v.index] is None:
                continue
            gt, mm = photo_cache[v.index]
            img = sc.render(v, fixed, G, with_geometry=False)["rgb"]
            loss = ((img - gt.float() / 255).abs().mean(0) * mm).sum() / mm.sum()
            kind = "mv_real"
        elif r < args.mv_real + args.ref_weight or not data:
            img = sc.render(ref, fixed, G, with_geometry=False)["rgb"]
            loss, _, _ = masked_l1_dssim(img, rtgt, rreg, win, rbox)
            kind = "ref"
        else:
            v, tgt, m, box = data[rng.integers(len(data))]
            img = sc.render(v, fixed, G, with_geometry=False)["rgb"]
            loss, _, _ = masked_l1_dssim(img, tgt.float() / 255, m, win, box)
            kind = "target"
        counts[kind] += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 250 == 0:
            log.append({"it": it, "kind": kind, "loss": round(float(loss), 4), "s": round(time.time() - t0, 1)})
            print(log[-1], flush=True)
    del photo_cache
    fit_s = time.time() - t0

    with torch.no_grad():
        info = {"target_views": used, "n_target_views": len(used), "counts": counts, "log": log, "fit_seconds": round(fit_s, 1),
                "ref_view": ref.stem, "ref_virtual": ref.virtual, "opacity_mean": float(lifted.opacity().mean()),
                "shadow": shadow_info, "args": {k: str(v) for k, v in vars(args).items()}}
        if not args.no_align:
            fit = to_u8(sc.render(ref, fixed, G, with_geometry=False)["rgb"]).astype(np.float32) / 255
            info["colour_align"] = align_colour(lifted, fit, hole, ralpha)
        lifted.save(out / "lifted.pt")
        # sheet: target | v3c | final, the 4 target views with the largest masks
        order = sorted(range(len(data)), key=lambda j: -int(data[j][2].sum()))[:4]
        rows = []
        for j in order:
            v, tgt, m, (y0, y1, x0, x1) = data[j]
            after = to_u8(sc.render(v, fixed, G, with_geometry=False)["rgb"])
            before = load_u8(w / "lift" / "rgb" / f"{v.index:05d}.png")
            tg = tgt.permute(1, 2, 0).cpu().numpy()
            row = np.concatenate([a[y0:y1, x0:x1] for a in (tg, before, after)], 1)
            rows.append(Image.fromarray(row).resize((900, max(1, round(900 * (y1 - y0) / (3 * (x1 - x0)))))))
        if rows:
            canvas = Image.new("RGB", (900, sum(x.size[1] for x in rows)))
            y = 0
            for x in rows:
                canvas.paste(x, (0, y))
                y += x.size[1]
            canvas.save(out / "sheet.jpg", quality=88)
        # export: observed Gaussians (minus the shadow ones), shadow Gaussians with their new colour, the fill
        check_view = views[used[0]] if used else (views[ref.index] if not ref.virtual else views[0])
        check_fn = to_u8(sc.render(check_view, fixed, G, with_geometry=False)["rgb"])
        fixed_rows = clean_idx[~sh]
        parts = [sc.raw_params(fixed_rows)]
        if sh.any():
            sp = sc.raw_params(clean_idx[sh])
            sp["features_albedo"] = G.s_dc.detach().to(sp["features_albedo"].dtype)
            parts.append(sp)
        lf = Frozen(lifted.xyz(), lifted.rotation(), lifted.scaling(), lifted.opacity(), lifted.features(sc.F))
        parts.append(sc.raw_from_frozen(lf))
        n = sc.save_checkpoint(parts, args.final / "ckpt_final.pt", extra={"surflift_num_fixed": int(len(fixed_rows))})
        info |= {"final_gaussians": n, "fixed": int(len(fixed_rows)), "shadow_recoloured": int(sh.sum()), "lifted": len(lifted)}
    del sc
    torch.cuda.empty_cache()

    # final renders from the reloaded checkpoint (what every other method's final/ holds)
    model, conf, _ = load_model(args.final / "ckpt_final.pt", all_views_overrides(args.colmap_dir))
    dataset, loader = build_dataset(conf)
    with torch.no_grad():
        for index, batch in enumerate(loader):
            rgb = render_view(model, dataset.get_gpu_batch_with_intrinsics(batch), index)["rgb"]
            save_png(rgb, args.final / "rgb" / f"{index:05d}.png")
            if index == check_view.index:
                a = (rgb.cpu().numpy() * 255).round()
                mse = float(((a - check_fn.astype(np.float64)) ** 2).mean())
                info["export_check_psnr_db"] = round(10 * np.log10(255 ** 2 / max(mse, 1e-10)), 2)
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(f"export check (reloaded checkpoint vs in-memory render, view {check_view.index}): "
          f"{info.get('export_check_psnr_db')} dB", flush=True)
    if info.get("export_check_psnr_db", 0) < 40:
        raise SystemExit(f"fit: the reloaded {args.final / 'ckpt_final.pt'} renders differently from the fitted model "
                         f"({info.get('export_check_psnr_db')} dB at view {check_view.index}): export broken")
    job_summary(w, f"fit: {len(used)} LeftRefill target views, {counts}, last loss {log[-1]['loss'] if log else 'n/a'}, "
                   f"{info['shadow_recoloured']} shadow Gaussians recoloured; final {info['final_gaussians']} Gaussians "
                   f"({info['lifted']} lifted), export check {info.get('export_check_psnr_db')} dB -> {args.final / 'ckpt_final.pt'}, "
                   f"{args.final / 'rgb'}")
    timer.done(n_target_views=len(used), final_gaussians=info["final_gaussians"], export_check_psnr_db=info.get("export_check_psnr_db"))


if __name__ == "__main__":
    main()
