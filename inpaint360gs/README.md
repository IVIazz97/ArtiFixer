<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inpaint360GS on 3DGUT

A port of [Inpaint360GS](https://github.com/dfki-av/Inpaint360GS) (Wang et al., WACV 2026,
[arXiv:2511.06457](https://arxiv.org/abs/2511.06457)) for removing objects from a 3DGUT
reconstruction and inpainting the hole. The paper's own object-aware segmentation is ported too:
2D mask association through the Gaussians, and object-ID distillation. The 3D parts are
rewritten for 3DGUT and follow the official code, including where the code differs from the
paper. Hyperparameters default to the official `config/*/common.json` and `train_distill.json`.

## Stages

| # | module | env | what it does | official counterpart |
|---|---|---|---|---|
| 1 | `raw_masks` | aurafusion | Segments every photo on its own into an id map. Uses SAM2 automatic masks (64 points per side, IoU threshold 0.8). Nested masks are painted largest first. | `seg/raw_mask_sam.py` |
| 2 | `associate` | artifixer | Sec. 3.2, GS-IoU association. Each mask's foreground Gaussians are the nearest 30% of the near depth cluster in each 16×16-grid patch. Masks are matched to a key-object database, and a score below 0.1 starts a new object. Writes consistent ids, plus `associated_numbered/` for picking target ids. | `seg/mask_associate.py`, `tools/add_label_num_hqsam.py` |
| 3 | `distill` | artifixer | Sec. 3.3. Learns a 16-dim identity feature per Gaussian and a 1×1 classifier; the scene stays frozen. Loss: CE / log C, plus a 3D k-NN cosine term every 50 iterations. Runs 2000 iterations. | `seg/distillation.py` |
| 4 | `remove` | artifixer | Removes Gaussians with class probability > 0.7, plus everything inside their IQR-filtered convex hull. Records the object radius. Target ids come from `--target_ids` or automatically from reference masks. `--flashsplat_dir` removes the FlashSplat object instead, hull included. `--hull_expand` scales that hull about the object's centre, which also takes what sits right against the object. | `edit_object_removal.py` |
| 5 | `virtual_views` | artifixer | Sec. 3.4. Places 30 PCA-aligned circular poses around the focus point, at the radius where the object fills 70% of the view. Renders the full and the removed scene (RGB, z-depth, opacity), and the removed objects' footprint: their visible share Σα·T of the full render. | `tools/virtual_pose.py`, `utils/pose_utils.py` |
| 6 | `nbs_masks` | either (sam2: aurafusion) | Builds inpainting masks from the footprint, or with SAM2 video tracking prompted by it. Then applies the official `enlarge`: drop components under 50 px, dilate 10 px. | interactive Segment-and-Track-Anything, `tools/prepare_lama_data.enlarge` |
| 7 | `lama` | either | Sec. 3.5. Inpaints colour and min–max normalised depth with big-lama and LaMa's multi-scale feature refinement. `--recursive_guide` turns on recursive conditional inpainting. | `LaMa/bin/predict_color.py`, `predict_depth.py` |
| 8 | `init_gaussians` | artifixer | Unprojects the hole pixels of virtual view 4 with the completed depth and removes statistical outliers. Each point becomes one Gaussian: colour from LaMa, opacity 0.1, scale/rotation/SH from its 5 nearest scene Gaussians. | `edit_object_removal_plyfusion.py`, `GaussianModel.inpaint_setup` |
| 9 | `finetune` | artifixer | Eq. 9: 0.2 L1 outside the hole, 0.8 full-image D-SSIM, 0.0005 patch LPIPS. Runs 5000 iterations on the virtual views and densifies the new Gaussians. Then resets scene Gaussians away from the hole and puts back the surrounding objects. Renders every dataset view. | `edit_object_inpaint.py` |

`output/jobs/inpaint360gs/run.sbatch` runs the stages and switches envs. It takes `STEPS=a+b+c`,
`REMOVAL=distill|flashsplat`, `TARGET_IDS`, `SURROUNDING_IDS`, `HULL_EXPAND`, `NBS_MODE` and
`RECURSIVE=1`.
Stages 1–3 write to `output/inpaint360gs/<scene>/base/`, which every variant links to. Stage
4 onwards write to `output/inpaint360gs/<scene>/<variant>/` (see `output/LAYOUT.md`).

Envs and models are the ones the AuraFusion port uses (`aurafusion/README.md`):

- SAM2 hiera-large from the HF cache.
- `checkpoints/lama/big-lama.pt` (TorchScript). The refiner splits it at its first FFC ResNet
  block, the same encoder/decoder split the official refiner optimises across.

## Picking the object

Upstream, you choose `target_id` by looking at `images_<r>_num/` after association. Here the
same view is `base/associated_numbered/`. Two ways to select:

- `--target_ids` removes exactly those ids.
- `--target_masks_dir` (the default in the job, `$MASKS/<scene>`) selects every associated id
  that has at least half of its pixels inside the binary reference masks.

`--surrounding_ids` are occluders. They are removed while the hole is inpainted and put back
afterwards. The paper finds them with YOLOv8; the official code takes them by hand.

## Validation (mip360 kitchen, 1/4 resolution, one A100)

| stage | result | time |
|---|---|---|
| raw masks | 45 SAM2 masks per view (max 94) | ~16 min |
| association | 807 classes. One id holds 60% of the bulldozer's reference-mask pixels, 91% of that id inside the mask. | ~2 min |
| distillation | predicted ids match the associated masks on 79% of pixels | ~5 min |
| removal | auto-selected ids 7, 13, 15, 168; removed 376k of 2M Gaussians; radius 0.75 | <1 min |
| virtual → finetune | circle radius 0.92 × camera spread; holes cover 21% of each view; 84k seed Gaussians, 1.2M new after densification | 8.5 min |

On the 279 dataset views, away from the object (reference mask dilated by 15 px), PSNR against
the photos is 32.3 dB for the final model and 30.9 dB for the removed-only scene. There is no
object-free ground truth for mip360, so the hole itself is not scored.

## Tests

`tests/test_inpaint360gs.py` (CPU) covers:

- the virtual trajectory and radius, which match the official `pose_utils.py` exactly;
- the exact depth 2-means and per-patch grouping;
- the key-object database;
- `enlarge`, the statistical outlier filter, and the LaMa refiner's blur (checked against
  kornia) and erosion.

## Deviations from the official code

- **3DGUT instead of 3DGS.**
  - Identity features go through 3DGUT's 3-channel SH radiance with degree 0, three channels
    per pass, shifted so the radiance clamp at 0 never fires.
  - Depth is alpha-normalised z. The official depth is the un-normalised sum of z·α·T; the two
    agree wherever the render is opaque.
  - Densification has no screen-space gradients or radii. It uses 3DGRUT's own proxy,
    |∇position|·distance/2, averaged over in-frustum views, and 3σ·f/z for the radius.
- **New Gaussians move** (position lr 1.6e-4 × extent), and at most 2M are added. Upstream's
  position lr is 0 because a loaded scene has `spatial_lr_scale` 0. Clones then sit on top of
  their source and are re-cloned every round: on kitchen that reached 5.4M new Gaussians and ran
  out of memory. Scene Gaussians stay pinned, as upstream.
- **2D masks:** SAM2 automatic masks instead of CropFormer entity segmentation, whose
  detectron2 build the official "hqsam" path needs. The official "sam" path is broken upstream
  (`generate(...)["masks"]` on a list).
- **Association:**
  - The depth 2-means is solved exactly instead of with sklearn KMeans.
  - Patches are indexed in row/column order. Upstream reshapes the transposed mask as [H, W]
    when it looks for the patches a mask touches.
  - Ids are 16-bit; upstream wraps them at 255.
- **`--hull_expand`** (default 1.0, the official hull) is ours. It scales the hull about the
  object's centre, for holes that also swallow the contact shadow. The virtual-camera radius
  stays the object's own, so runs at different expansions share a trajectory.
- **NBS masks** are automatic (footprint, or SAM2 tracking) instead of an interactive
  SAM-Track session.
- **Surrounding objects** are put back whenever there are any. Upstream only does it for more
  than one (`len(surrounding_ids) > 1`).
- **All views** (train + test) are used for distillation and for the virtual-radius PCA.
  Upstream uses the `--eval` train split.

Kept as in the official code, where it differs from the paper:

- The L1 term is outside the hole.
- Every Gaussian is trainable during the finetune (positions only for the new ones).
- Only one virtual view (4) seeds the new Gaussians.
- The association score is |I| / (|mask| + |I|).
- LaMa's refinement is a single scale (no optimisation) when the image's short side is under
  ~724 px, e.g. mip360 at 1/4 resolution.
- Recursive conditioning is off by default.
