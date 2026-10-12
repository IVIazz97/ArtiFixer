<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# SurfaceLift on 3DGUT

A port of our own hole-filling method SurfaceLift (research code in `CVPR2027/SurfaceLift`, snapshot
2026-10-12) onto the 3DGUT reconstructions and the FlashSplat removal used by the other baselines in
this repo. It runs the best SurfaceLift recipe:

- **v3c**: observation ledger (never change what a photo saw), hole = only what no photo saw,
  surface-bound single-view lift, harmonic colour alignment;
- **pipeline C + shadow** of the benchmark (`surflift_leftrefill_shadow`): ledger hole of every view
  plus the object's shadow, AuraFusion360's LeftRefill SDEdit of our renders inside it, and a
  multi-view colour fit of the lifted Gaussians to those images.

Two variants (`scripts/bakerh/run_removal.sh`, `output/jobs/surflift/run.sbatch`):

| variant | fill view |
|---|---|
| `surflift` | the best **real** view, SurfaceLift's `select_view` real scoring (coverage² × hole size × steepness) |
| `surfliftv` | a view that holds the **whole** hole inside the frame (40 px / 4 % margin): real views whose mask clears the margin, SurfaceLift's 144 virtual look-at cameras, and every training camera re-aimed at the object; virtual ones are pulled back (×1 … ×3) until the object's 3D points all project inside the margin, and rejected when blocked (> 5 % of the frame nearer than 0.25 × the object distance) or mostly empty. Falls back to the best real view. |

`surfliftv` exists for the Baker Hughes captures: close-ups of large objects often cut the hole at the
frame border, and a single-view fill of a cut hole leaves part of it empty.

## Stages

All stages write under one work dir (`<variant>/sl/`); the final model goes to `<variant>/final/`.

| # | module | env | what it does | SurfaceLift source |
|---|---|---|---|---|
| 1 | `prepare` | artifixer | FlashSplat object (labels + trimmed hull, as `aurafusion.render_views`) → `removal.pt`; object mask M0 of every view = 2D mask (`--mask_dir`) ∪ object opacity > 0.05, dilated 12 px | `scene_data.object_mask`, `flashsplat_remove.py` |
| 2 | `clean` | artifixer | Observation ledger, consistency rule, 2 passes → `ledger/keep.npy` (cleaned removed scene, bit-identical to the full model) | `scripts/ledger.py --rule consistency` |
| 3 | `select_view` | artifixer | `--mode real` / `--mode contain` (above), then the unseen hole U of that view with the `--consistency` rule, alpha 0.9, beyond 16 px | `select_view.py`, `unseen_view.py` |
| 4 | `inpaint` | artifixer | big-lama (default) or Qwen-Image-Edit-2511 (`--editor qwen --qwen_dir`, CPU offload) on the view, pasted inside U | `inpaint_view.py` |
| 5 | `depth` | artifixer | MoGe-2 ViT-L with the camera's FoV | `depth_view.py` |
| 6 | `lift` | artifixer | Align depth, lift dilate(U, 8) into surface-bound flat Gaussians, 1500 steps (`--mv_real 0.2`), harmonic colour alignment, render every view | `lift.py`, `blend.py --round 0` |
| 7 | `holes` | artifixer | H_v, coverage C_v and shadow S_v of every view | `coin_holes.py --shadow` |
| 8 | `leftrefill` | aurafusion / `.venv_af2d` | AF360's official `sdedit_utils.LeftRefill` (strength 0.85, eta 1, scale 2.5, no DDIM inversion) on our renders inside dilate(H_v ∪ S_v, 4), reference = the chosen view's inpainting | `coin_leftrefill.py` |
| 9 | `fit` | artifixer | 3000 steps on the lifted colour/opacity (+ shadow Gaussians' DC): 60 % LeftRefill targets, 20 % chosen view, 20 % real photos; colour alignment; export `final/ckpt_final.pt` + `final/rgb/<i>.png` | `coin_fit.py --views nontrivial --shadow` |
| – | `compare` | artifixer | Panel + fidelity (PSNR vs photos outside the masks / 40 px ring) | `fidelity.py` |

`common.py` replaces SurfaceLift's `backends.py` / `cameras.py` / `scene_data.py`. `ledger.py`,
`lifted.py`, `appearance.py` (only `align_colour`), `align.py` and `losses.py` are vendored;
`align.py` and `losses.py` unchanged.

`final/ckpt_final.pt` is a plain 3DGUT checkpoint (the layout `aurafusion/finetune.py` writes): the
cleaned observed Gaussians first (`surflift_num_fixed`), then the shadow Gaussians with their new
colour, then the fill. `final/rgb/` is rendered from the reloaded checkpoint; the fit stage logs the
PSNR between that and its in-memory render (`fit/info.json: export_check_psnr_db`).

## Deviations from the official SurfaceLift

- **Renderer: 3DGUT** instead of the base methods' 2DGS/3DGS rasterisers. Rendering calls the 3DGUT
  tracer's autograd function on concatenated tensors (frozen + lifted), so per-Gaussian override
  colours (white / lifted-only coverage) and the ledger's colour-gradient trick work as before. Depth
  is 3DGUT's alpha-normalised mean z (as SurfaceLift's Inpaint360GS backend). The principal point is
  the dataset's, not the image centre.
- **Lifted Gaussians** are flat 3D Gaussians (SurfaceLift's Inpaint360GS kind, 5 % thick): 3DGUT has
  no surfels.
- **Base scene**: the full 3DGUT reconstruction minus the FlashSplat object (labels + every Gaussian
  inside the object's trimmed convex hull). The AF360/I360 ports' `removal_keep_masks` also drops every
  low-contribution Gaussian anywhere in the scene (kitchen: 1.36 M of 2 M); here those stay, so the
  removed scene equals the full model away from the object.
- **Object masks**: the 2D masks given (`--mask_dir`; bakerh: the SAM3 `masks_npy`, kitchen:
  `datasets/masks_objid`) ∪ the removed Gaussians' opacity, dilated 12 px; SurfaceLift used SAM2 ∪
  FlashSplat foreground opacity on kitchen/garden.
- **Inpainter: big-lama** (TorchScript) by default, on the source image with the unseen hole in grey
  (photo outside the mask, the cleaned scene inside it). SurfaceLift v3c pasted a whole-photo
  Qwen-Image-Edit removal edit into the hole; Qwen (54 GB) is optional here with a generic fill prompt
  (no per-scene object prompt).
- **View selection**: scored on the ledger-cleaned scene (SurfaceLift scored on the base method's
  removed scene, then cleaned). The `surfliftv` containment mode is new.
- **Hole rule**: `unseen_view --consistency` (the benchmark rule) on every scene; kitchen/garden v3c ran
  without it.
- **Ledger `--seen_weight 1.0`** (the benchmark value; mip360 v3c used 0.25). On kitchen 0.25 drops
  ~10 k more Gaussians and looks the same.
- **LeftRefill crop**: LeftRefill resizes the whole image to 512 × 512. Each view is first cropped to a
  square around its mask (≥ 512 px, 15 % padding) and pasted back, so a hole is not denoised at a third
  of its resolution on 1600 × 1200 photos (`--no_crop` = the official behaviour).
- **Shadow target of the chosen view**: the benchmark used its real object-free reference photo on
  the chosen view's shadow pixels; here LaMa over U ∪ S of that view.
- **Fit guard** (port addition, `fit.py --no_guard` turns it off): where the fill shows over pixels of a
  view's object mask that the photos confirmed (outside its ledger hole), the fit target is the cleaned
  scene's own render. Without it, lifted Gaussians at the hole edge poked through the real mat from
  other views as a bright jagged rim (kitchen); the guard fades / recolours them.
- **Virtual fill views**: hole components smaller than 5 % of the largest (and < 200 px) are dropped
  (`select_view --min_island`): a render from a new viewpoint is blurrier than the photos, so the colour
  test flags scattered islands on observed surface, and lifting them made floaters.
- **Witness memory**: the consistency test keeps every photo's full-model depth, mask and blurred
  photo on the GPU, at 1/2 or 1/4 resolution once the views exceed 4·10⁸ pixels (923 Compressor views).

## Run

```bash
# Leonardo (mip360):
sbatch --job-name=SL_kitchen --export=ALL,SCENE=kitchen,VARIANT=surflift output/jobs/surflift/run.sbatch
sbatch --job-name=SLv_kitchen --export=ALL,SCENE=kitchen,VARIANT=surfliftv output/jobs/surflift/run.sbatch
python -m surflift.compare --checkpoint ... --colmap_dir ... --runs output/surflift/kitchen/{surflift,surfliftv} \
    --other af360=output/aurafusion/kitchen/lama_ref/final/rgb --out output/panels/surflift_kitchen.jpg
# Baker Hughes VM:
bash scripts/bakerh/run_removal.sh TA_BLUE_MOTOR surflift
bash scripts/bakerh/run_removal.sh TA_BLUE_MOTOR surfliftv
```

Per-stage wall time and peak GPU memory: `<variant>/sl/stages.txt`; one human-readable line per stage
(chosen view, hole size, in-frame or not, ledger drops, ring error, fit loss, export check):
`<variant>/job_summary.txt` (the bakerh launcher copies it into REPORT.txt).

## Kitchen check (2026-10-12, `output/panels/surflift_kitchen.jpg`)

PSNR vs the photos outside the object masks / on the 40 px ring, every 3rd view (93):

| | outside | ring |
|---|---|---|
| full model (ceiling) | 34.48 | 35.42 |
| AuraFusion360 port (`aurafusion/kitchen/lama_ref`) | 33.74 | 33.42 |
| surflift v3c (after `lift`) | 33.67 | 34.30 |
| **surflift** (final) | 33.68 | **34.37** |
| surfliftv with virtual views forced (`--virtual_only`), final | 33.09 | 33.11 |

`surfliftv` picks the same real view as `surflift` on kitchen (DSCF0704 holds the whole hole), so the
virtual path was tested with `--virtual_only` (chosen: training view 113 re-aimed, pulled back ×1.3).
LeftRefill ran on 100 of the 278 views there (`--max_views 100`, debug-queue time limit).
