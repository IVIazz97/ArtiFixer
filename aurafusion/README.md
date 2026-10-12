<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# AuraFusion360 on 3DGUT

A port of [AuraFusion360](https://github.com/kkennethwu/AuraFusion360_official) (Wu et al., CVPR 2025)
for filling the hole left after removing an object from a 3DGUT reconstruction. The object is
selected with [FlashSplat](../flashsplat/README.md) labels. Only the 3D parts are rewritten for
3DGUT. The 2D models (SAM2, Marigold, and LeftRefill on SD2-inpainting) are the official ones,
and SDEdit calls the official `sdedit_utils.py` unchanged.

## Stages

| # | module | env | what it does | official counterpart |
|---|---|---|---|---|
| 1 | `render_views` | artifixer | Removes the object (FlashSplat label + convex-hull filter). For every view, renders the removed scene (RGB, z-depth, opacity) and the removed object's opacity (= removal region R). Writes `cameras.json` and `ckpt_removed.pt`. | `remove.py` |
| 2 | `unseen_contour` | either | Paper Eq. 1–2. Lifts R_n with the removed-scene depth into every view where the object is visible. A pixel is kept if it is hidden in more than 60% of those views. | `GaussianExtractor.depth_aware_unseen_mask_generation` |
| 3 | `sam2_unseen` | aurafusion | Eq. 3. The contour's bounding box prompts SAM2 on the removed renders. The mask is then opened, reduced to its largest component and dilated (kernel 5 × 3). | `utils/sam2_utils.py`, `camera_utils.py` |
| 4 | `agdd` | aurafusion | Builds the reference image: photo, removed render over the object, LaMa in the unseen mask. `--reference_image` supplies one instead. Then Adaptive Guided Depth Diffusion (Eq. 5–8, Marigold v1-0). | `inpaint_init` + `utils/depth_utils.py`, `marigold_di_utils.AGDDv2` |
| 5 | `init_gaussians` | artifixer | Unprojects the reference's unseen pixels into one new Gaussian each. The removed scene stays frozen. Renders all views. | `inpaint_init` + `GaussianModel.inpaint_setup` |
| 6 | `sdedit` | aurafusion | LeftRefill SDEdit (Eq. 9–11): DDIM inversion to strength 0.85, CFG 2.5, conditioned on the reference image. | `utils/LeftRefill/sdedit_utils.py` (called unchanged) |
| 7 | `finetune` | artifixer | 10k iterations on only the new Gaussians (Eq. 12): L1/SSIM + 0.5 LPIPS inside the unseen mask, L1/SSIM outside it. | `inpaint_finetune` |

`compare.py` scores our unseen masks against the released official ones and draws image panels.
`output/jobs/aurafusion/run.sbatch` runs the stages and switches envs (`STEPS=a+b+c`,
`VARIANT`, `REF_IMAGE`, `REF_INDEX`). Hyperparameters default to the official
`configs/Other-360/kitchen/*.config`.

The two envs:

- `$ENVS/artifixer` has the 3DGUT tracer.
- `$ENVS/aurafusion` holds the 2D models, built by `output/jobs/aurafusion_env.sh`. LeftRefill's
  vendored `ldm` needs pytorch_lightning, open_clip and CLIP with transformers < 5, and those must
  stay out of the Wan/ArtiFixer env.

Models are pre-fetched into `checkpoints/aurafusion/` (compute nodes are offline):

- SD2-inpainting `512-inpainting-ema.ckpt`, from `sd2-community`; the stabilityai repo returns 401;
- Marigold v1-0 (fp16);
- SAM2 hiera-large;
- the official 360-USID kitchen reference image and masks, plus the official results.

OpenCLIP ViT-H laion2b weights live in the HF cache.

## Variant af360v: virtual views that frame the whole hole

In close-up captures, the image border often cuts the removed object. The reference (LaMa + AGDD) and
the SDEdit targets are then built on only part of the hole, so the inpainting is inconsistent.
af360v (`run_removal.sh SCENE af360v`, or `output/jobs/aurafusion/run_af360v.sbatch` on mip360) adds
virtual views and runs the same stages on them:

1. `virtual_views` (artifixer env) uses the same removal as `render_views`.
   - It boxes the object's Gaussians: a PCA box, trimmed at 99.5%.
   - From every real camera it builds one candidate. The candidate is re-aimed at the box centre,
     keeps the real camera's up direction and intrinsics, and is pulled back until all 8 corners lie
     6% inside the image border.
   - It rejects a candidate when other geometry hides more than 15% of the object, more than 30% of
     the image is very near, the view is mostly empty, or the camera had to move back more than 4×.
   - It keeps 24 candidates by farthest-point sampling on view direction.
   - It renders the full scene from each kept view as its "photo".
   - Output: a COLMAP scene `colmap_virtual/` with the real views first, unchanged, then
     `zzz_virtual_*.png`, plus `virtual.json` and the contact sheet `virtual_views.png`.
2. `render_views`, `agdd` and `init_gaussians` run on `colmap_virtual` without changes.
3. `unseen_contour --virtual_json` lets only real views vote on seen/unseen. Virtual views still get a
   contour.
4. `frame_views` picks the reference. A view's processed unseen mask is "in frame" when it stays 1% of
   the shorter side away from the border. The reference is the largest in-frame real view, else the
   largest in-frame virtual view. The result goes to `in_frame.json`.
5. `finetune --in_frame_only --virtual_json` applies the in-mask L1/SSIM/LPIPS loss only on in-frame
   views. Every view keeps the outside loss. Real views render to `final/rgb/<i>.png` with af360's
   numbering; virtual views render to `final/virtual_rgb/`.

`virtual_views`, `frame_views` and `finetune` take `--summary FILE`. With it, each appends one line
about what it decided to that file: camera counts and rejection reasons, the reference and whether it is
real or virtual, and how many views supervise the hole. `run_removal.sh` points it at
`$O/job_summary.txt`, which `run_all.sh` copies into REPORT.txt.

## Validation

On mip360 kitchen (same frames as the official Other-360/kitchen), our stage 1–3 unseen masks
reach mean IoU 0.77 (median 0.81) against the released official ones over all 279 views.

## Deviations from the official code

- **3DGUT instead of 2DGS.**
  - Depth is `pred_dist / pred_opacity` converted to camera z. It is an alpha-weighted mean rather than 2DGS's median surface depth.
  - New Gaussians are isotropic 3D, not surfels.
  - There are no normal or distortion regularisers.
- **Removal** uses FlashSplat labels and our convex-hull filter, not their learned per-Gaussian mask.
- **Removal region R** is object opacity > 0.02 instead of the exact test "depth changed". Points behind a camera count as seen.
- **SAM2 box prompt** comes from the contour's largest connected component, so a stray floater pixel cannot inflate it.
- **Unseen contour on partial captures:** when the official mean (over every view showing the object, out-of-frame
  counting as seen) leaves the contour empty in every view, `unseen_contour --denominator auto` (default) recomputes
  it averaging only over the views the point lands in, with points no other view frames counting as unseen.
- **Reference image:**
  - For Other-360 the official code loads a pre-inpainted reference image.
  - Here it is built from the photo, the removed render and LaMa, or taken from `--reference_image`.
- **AGDD optimiser:** torch AdamW with an fp32 master latent, instead of bitsandbytes AdamW8bit on the fp16 latent.
- **New Gaussians** start at opacity 0.99 (the official 1.0 is `inverse_sigmoid(1) = inf`).
- **Finetune** has no densification; the official densifies every 300 iterations until 1000.
