# Running the inpainting pipelines from a trained 3DGUT scene

Written for a Claude/engineer session on a GPU machine (not Leonardo). You have a scene already
reconstructed with 3DGUT and want to run object removal + inpainting three ways:

1. **AuraFusion360 port** (`aurafusion/`) — see `aurafusion/README.md`
2. **Inpaint360GS port** (`inpaint360gs/`) — see `inpaint360gs/README.md`
3. **ArtiFixer pipeline** — FlashSplat removal → ArtiFixer 1.3B video diffusion → propagation → ArtiFixer3D refit

The canonical runners are `output/jobs/*.sbatch` and `output/jobs/<port>/run.sbatch`. They were
written for the Leonardo SLURM cluster: ignore the `#SBATCH` headers and the
`source .../CVPR2027/env.sh` line inside `output/jobs/common.sh`, and run the python commands
they contain directly. Everything below tells you how to reproduce their environment on a plain
GPU box. **Read the sbatch script for a stage before running it — it is the source of truth for
flags and stage order.** Output layout: `output/LAYOUT.md`.

## 0. What must already exist

- This repo, with the submodule: `git submodule update --init --recursive`, then apply the
  FlashSplat patch (needed by the removal render method):
  ```bash
  git -C thirdparty/3DGRUT-ArtiFixer apply ../patches/3DGRUT-ArtiFixer-flashsplat.patch
  ```
- A trained 3DGUT scene. Two levels of readiness:
  - **Ports only (AuraFusion / Inpaint360GS):** a 3DGUT checkpoint `ckpt_30000.pt` plus the
    prepared COLMAP dir it was trained from.
  - **ArtiFixer pipeline:** the full per-scene layout that
    `data_processing.prepare_colmap_artifixer_inputs` writes (stage A,
    `output/jobs/stageA_recon.sbatch`): `3dgrut_input/<scene>`, `3dgrut_runs/…/ours_30000/ckpt_30000.pt`,
    `recon_results/`, `split.json`, and `captions/<scene>/caption.h5` (written by
    `output/jobs/write_caption.py`; needs a one-sentence scene caption — examples in
    `output/jobs/common.sh`). If you only have a bare checkpoint, rerun stage A first.
- **Object-id masks** for the object you want removed: `datasets/masks_objid/<scene>/` — PNG/NPY
  id maps (0 = background, 1..K = objects) named after the image they annotate. FlashSplat and
  the ports' auto target-id selection both read these. They are gitignored, so they arrive by
  copy, not by clone.

## 1. Environments

Two venvs (`output/jobs/common.sh` calls them `$ENVS/artifixer` and `$ENVS/aurafusion`):

- **artifixer** — the main env: torch 2.11 + CUDA, diffusers/transformers for Wan, and the 3DGUT
  tracer (`threedgrut` importable, CUDA extension built). On the recovery machine this is the
  repo's `.venv` (see `ENVIRONMENT_RECOVERY.md`). Runs every 3D stage and ArtiFixer inference.
- **aurafusion** — the 2D-models env: SAM2, Marigold, LeftRefill (vendored `ldm` needs
  pytorch_lightning, open_clip, CLIP, transformers < 5 — must stay out of the Wan env). Build it
  with `output/jobs/aurafusion_env.sh` (adapt the two `source`/`uv venv` lines to local paths).

The stage tables in `aurafusion/README.md` and `inpaint360gs/README.md` say which env each stage
uses; the run.sbatch scripts encode it as `python` (artifixer env, the active one) vs `$AF_PY`
(the aurafusion env's python).

## 2. Model checkpoints (all gitignored — download or copy)

```bash
bash scripts/download_baseline_checkpoints.sh    # SAM2, Marigold, SD2-inpainting, OpenCLIP, big-lama (~9 GB)
hf download nvidia/ArtiFixer artifixer-1.3b.pt --local-dir checkpoints/artifixer
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir checkpoints/Wan2.1-T2V-1.3B-Diffusers
```

Expected result: `checkpoints/{artifixer/artifixer-1.3b.pt, Wan2.1-T2V-1.3B-Diffusers/,
aurafusion/{sam2-hiera-large,marigold-v1-0,sd2-inpainting}, lama/big-lama.pt}` plus SAM2/OpenCLIP
in the HF cache. `checkpoints/moge-2-vitl-normal/model.pt` (MOGE_MODEL_PATH) is only needed by
stage-A reconstruction, not by the pipelines below.

## 3. Shell setup (replaces `output/jobs/common.sh`)

```bash
export AF=/path/to/ArtiFixer                       # this repo
cd "$AF"
export PYTHONPATH="$AF:$AF/thirdparty/3DGRUT-ArtiFixer:${PYTHONPATH:-}"
export MODEL_ID="$AF/checkpoints/Wan2.1-T2V-1.3B-Diffusers"
export CHECKPOINT_PT="$AF/checkpoints/artifixer/artifixer-1.3b.pt"
export OUT="$AF/output" RECON="$OUT/recon" MASKS="$AF/datasets/masks_objid"
export SCENE=kitchen                               # or your scene
CKPT="$RECON/$SCENE/3dgrut_runs/$SCENE/$SCENE/ours_30000/ckpt_30000.pt"
COLMAP="$RECON/$SCENE/3dgrut_input/$SCENE"
```

Skip `HF_HUB_OFFLINE` if the machine has internet (Leonardo compute nodes don't).

## 4. FlashSplat segmentation (prerequisite for AuraFusion and the ArtiFixer pipeline)

Run the `seg` and `extract` steps of `output/jobs/stageB_removal.sbatch`
(`data_processing.run_flashsplat_segmentation` then `data_processing.render_flashsplat_extraction`)
with `$CKPT`, `$COLMAP`, `--mask_dir $MASKS/$SCENE`, output `$OUT/flashsplat/$SCENE`. This writes
`labels.pt`, `contribution.pt`, `hit_count.pt` and the foreground/background renders. The main
`README.md` ("FlashSplat Object Removal") documents the flags. Inpaint360GS does not need this
unless you run its `REMOVAL=flashsplat` variant.

## 5. AuraFusion360 port

Follow `output/jobs/aurafusion/run.sbatch` top to bottom. Stages, in order:
`render, contour, sam2, agdd, init, sdedit, finetune` — sam2/agdd/sdedit run in the aurafusion
env, the rest in the artifixer env. Stages 1–3 write the shared `base/`; each variant dir
symlinks them (the script does this). Outputs: `output/aurafusion/<scene>/<variant>/`.
Knobs: `VARIANT`, `REF_IMAGE` (your own inpainted reference instead of the LaMa one),
`REF_INDEX`, `HULL_EXPAND`.

## 6. Inpaint360GS port

Follow `output/jobs/inpaint360gs/run.sbatch`. Stages, in order:
`masks, associate, distill, remove, virtual, nbs, lama, init, finetune` — `masks` (and `nbs` in
sam2 mode) run in the aurafusion env, the rest in the artifixer env. Stages 1–3 write
`output/inpaint360gs/<scene>/base/`, shared by all variants. Knobs: `REMOVAL=distill|flashsplat`,
`TARGET_IDS` (or auto-pick from `$MASKS/$SCENE`), `SURROUNDING_IDS`, `HULL_EXPAND`, `NBS_MODE`,
`RECURSIVE=1`. Pick target ids by eye from `base/associated_numbered/`.

## 7. ArtiFixer pipeline

Four stages; outputs under `output/artifixer/<scene>/` (variant table in `output/LAYOUT.md`):

1. **Removal split + single-rollout inference** — `output/jobs/stageB_removal.sbatch`, steps
   `split,infer`: `output/jobs/build_removal_split.py` builds `vN/input/` (renders, hole masks,
   reference views, split.json) from the FlashSplat result, then `model_eval.run_inference`
   (`--evalset reconstructed_colmap --render_trajectory all_frames`) writes `vN/artifixer/`.
   Variant knobs (`--opacity`, `--refs`, `--dilate_px`, `--hole_shape`) are looped by
   `output/jobs/stageC_variants.sbatch`; the tested combinations are the v0–v9 table in
   `output/LAYOUT.md` (v3 was the pick for kitchen, v5 for garden).
2. **Propagation** — `output/jobs/stageE_propagate.sbatch`:
   `output/jobs/propagate_removal.py` progressively propagates chosen seed frames (photo outside
   the mask, ArtiFixer prediction inside) to all views.
3. **ArtiFixer3D + ArtiFixer3D+** — `output/jobs/stageF_af3d.sbatch`:
   `output/jobs/build_af3d_split.py`, then `data_processing.run_artifixer3d` distills the
   composites back into 3DGUT, then a `val_frames` `run_inference` pass for ArtiFixer3D+.
4. Optional LaMa-seeded comparison: `stageG_lamafill.sbatch` / `output/jobs/lama_seed.py`.

## 8. Comparing methods

`output/jobs/compare_methods.py` builds side-by-side panels across
aurafusion / inpaint360gs / artifixer outputs; `aurafusion/compare.py` scores unseen masks
against the official ones (kitchen data via `WITH_OFFICIAL=1` on the download script).
