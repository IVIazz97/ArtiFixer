# ArtiFixer Handoff

A recap of this repo's `README.md` plus the exact state of an in-progress session, written so a
fresh Claude instance picking this up elsewhere has full context without re-deriving it.

_As of 2026-09-18. This fork adds the FlashSplat work below on top of `nv-tlabs/artifixer`; the
upstream repo (`origin` remote) is untouched._

## Handoff status

**Done, on the source machine (a Mac):**

- Downloaded `nvidia/ArtiFixer` → `artifixer-1.3b.pt` (6.7 GB) and the full
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` repo (28.9 GB), both unauthenticated — these are public
  Hugging Face repos, no token needed.
- Saved under `download/artifixer-checkpoints/` and `download/Wan2.1-T2V-1.3B-Diffusers/` at the
  repo root. `download/` is gitignored — these files are local-only and were never pushed here.
- Confirmed via `model_eval/checkpoint_loading.py` that the checkpoint holds *only* transformer
  weights; the base model supplies the VAE, text encoder, tokenizer, and scheduler. Estimated
  in-memory footprint at inference (bf16 cast): ~15 GB (text encoder ~11.3 GB + transformer ~3.4 GB
  + VAE ~0.25 GB).

**Blocked:**

- The target machine is a corporate-managed Windows 365 Cloud PC, reached from macOS via
  Microsoft's **Windows App** client (RDP-based). The exact organization isn't recorded here
  deliberately — ask the user if it matters.
- RDP folder redirection was configured client-side (Windows App → Settings → General → Choose
  Folder) and Windows App was granted Full Disk Access on macOS. After a clean reconnect, the
  `TSCLIENT` share still appears empty inside the remote session.
- Personal OneDrive/cloud storage is not reachable from the source Mac for this account —
  consistent with a conditional-access / managed-device policy on the corporate tenant.
- Working theory: both restrictions are server-side policy on the Cloud PC / tenant, not a client
  misconfiguration. Further client-side tweaking on the Mac is unlikely to fix this.

**Unknown — confirm with the user before doing more work:**

- **Does the Cloud PC have a GPU at all?** Standard Windows 365 SKUs are CPU-only. Every workload
  in this repo (the diffusion transformer, the 3DGRUT/FlashSplat CUDA rasterizer) requires CUDA.
  If there's no GPU, nothing here can run on that machine regardless of file transfer.
- The actual downstream task was never fully pinned down in the source session — "get the weights
  onto the VM" is as far as it got. Confirm what the user actually wants to run once files are in
  place: plain ArtiFixer inference, the FlashSplat segmentation work below, or something else.

**Try this first, before any manual file transfer:** both Hugging Face repos above are public and
were downloaded with no token. If the Cloud PC has outbound internet access to `huggingface.co` —
plausibly more permitted than personal cloud storage from an unmanaged device, since it's outbound
from a corporate asset rather than inbound to one — the simplest fix is to skip transferring 33 GB
entirely and just re-download directly on the Cloud PC:

```powershell
pip install huggingface_hub
hf download nvidia/ArtiFixer artifixer-1.3b.pt --local-dir download\artifixer-checkpoints
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir download\Wan2.1-T2V-1.3B-Diffusers
```

Only fall back to wrestling with RDP redirection, cloud storage, or IT tickets if the Cloud PC
itself can't reach Hugging Face either.

---

## Repository layout

- `model_training/` — model definition, data loaders, training loop, diffusion pipelines.
- `model_eval/` — inference entry point and metric computation (DL3DV, Nerfbusters).
- `data_processing/` — data-prep wrappers, split generation, captioning, sparse-recon conversion.
- `thirdparty/` — the 3DGRUT submodule (`nv-tlabs/3DGRUT-ArtiFixer`), pinned to a fixed commit.
- `flashsplat/` — **not part of upstream ArtiFixer**, added on this fork only (see below).

## Setup

1. Clone with submodules: `git clone --recurse-submodules <this-repo-url>`, or
   `git submodule update --init --recursive` if already cloned without them.
2. Build one of the CUDA Dockerfiles — `Dockerfile.cuda12`, `Dockerfile.cuda13`, or
   `Dockerfile.cuda13-aarch64` for ARM64/GB200. This is the recommended environment; everything
   below assumes it (i.e. assumes a CUDA GPU is present).
3. Run with `docker run --gpus all --ipc=host -v "$PWD":/workspace/artifixer -v /path/to/data:/data ...`

## Checkpoints

The checkpoint stores **only transformer weights**. Every inference/eval command needs
`--model_id` pointed at the matching base model, or loading fails with shape mismatches (the
default is the 14B base).

| Checkpoint | Base model | Params | Status |
| --- | --- | --- | --- |
| `artifixer-1.3b.pt` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | ~1.68B | **downloaded** (see above) |
| `artifixer-14b.pt` | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | ~16.9B | not fetched (67.6 GB checkpoint alone) |

```bash
export CHECKPOINT_PT=download/artifixer-checkpoints/artifixer-1.3b.pt
export MODEL_ID=download/Wan2.1-T2V-1.3B-Diffusers   # or the HF id, to pull fresh instead
```

## Inference

Try it on one scene by pulling a DL3DV archive (`scripts/download_dl3dv_scene.py`), or prepare
your own COLMAP capture arranged as:

```text
<COLMAP_SCENE>/
  images/
  sparse/0/
    cameras.bin  images.bin  points3D.bin
```

```bash
python -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir /path/to/COLMAP_SCENE \
    --output_root /path/to/artifixer-prep/my_scene
```

This trains a 3DGRUT COLMAP MCMC reconstruction (10k iterations by default), renders it, estimates
metric scale with MoGe, and writes caption embeddings — producing a `split.json` that
`model_eval.run_inference` consumes:

```bash
python -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
    --save_dir "$SAVE_DIR" --split_path "$SCENE_ROOT/split.json" \
    --render_trajectory all_frames   # or val_frames / trajectory
```

<details>
<summary>ArtiFixer3D and ArtiFixer3D+ (chained refinement stages)</summary>

ArtiFixer3D trains a fresh 3DGRUT optimization on real anchor views + ArtiFixer-generated targets:

```bash
python -m data_processing.run_artifixer3d \
    --scene_root "$SCENE_ROOT" \
    --artifixer_frames_dir "$ARTIFIXER_OUTPUT_DIR/$SCENE_ID/frames/batch_0000/pred"
```

ArtiFixer3D+ re-runs ArtiFixer on that output using the generated `split_artifixer3d_plus.json` —
the same `model_eval.run_inference` call, pointed at the new split path.

</details>

## Training data preparation

Training needs three prepared inputs: DL3DV scene archives, per-scene reconstruction HDF5s, and
prompt HDF5s.

<details>
<summary>Full 5-step pipeline (download → caption → reconstruct → split → export paths)</summary>

1. Download DL3DV scenes: `huggingface-cli download DL3DV/DL3DV-ALL-960P --repo-type dataset ...`
2. Generate prompt HDF5s: `python -m data_processing.run_captioning ...`
3. Generate reconstruction HDF5s: `python -m data_processing.run_sparse_reconstruction ...` — runs
   covisibility split, 3DGRUT training, metric alignment (MoGe), HDF5 conversion. MoGe weights
   auto-download from HF unless `MOGE_MODEL_PATH` points at a local copy.
4. Build train/test split: `python -m data_processing.trainval_test_split ...`
5. Export `SPLIT_PATH` / `DL3DV_ROOT` / `PROMPT_ROOT` for training and eval to share.

</details>

## Training — three stages

1. **Stage 1** — supervised finetuning on reconstruction-conditioned DL3DV clips
   (`model_training.train`).
2. **Stage 2** — block-causal diffusion-forcing finetuning from the stage 1 checkpoint
   (`model_training.diffusion_forcing`).
3. **Stage 3** — DMD distillation: the stage 2 checkpoint initializes the student/generator, stage
   1 initializes the critic (`model_training.distillation`).

All launched via `accelerate launch --multi_gpu ...`. Default model is Wan2.1 14B; pass
`--model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers` to every stage for the 1.3B variant. Keep
`num_processes × gradient_accumulation_steps = 128`.

## Evaluation

Release evaluation reports four rows — `3DGUT` (base reconstruction), `ArtiFixer`,
`ArtiFixer3D`, `ArtiFixer3D+` — over DL3DV (our split + DiFix split) and NerfBusters, via
`model_eval.run_inference` then `model_eval.compute_metrics_*`. See `README.md` for the full
per-dataset commands — they're mostly environment-variable plumbing around the same two entry
points.

## FlashSplat (this fork only, not upstream)

Lives entirely under `flashsplat/` plus one CLI in `data_processing/run_flashsplat_segmentation.py`.
Segments the Gaussians of a trained 3DGUT/3DGRUT reconstruction using 2D object-id mask PNGs, via
[FlashSplat](https://github.com/florinshen/FlashSplat)'s closed-form solve: the rasterizer
accumulates per-Gaussian α·T per label across views, then a per-Gaussian argmax assigns labels —
no optimization loop.

```bash
python data_processing/run_flashsplat_segmentation.py \
    --checkpoint /path/to/ckpt_last.pt --colmap_dir /path/to/colmap_scene \
    --mask_dir /path/to/object_id_masks --num_objects 3 --gamma 0.0 \
    --output_root /path/to/out --outputs labels checkpoints ply overlays
```

**Verification status** (from `flashsplat/README.md`): the solver and mask-loading logic are
CPU-tested and passing (11 tests). The CUDA hook itself — the actual accumulation inside the 3DGUT
rasterizer — has **never been compiled or run**; this was authored on a GPU-less macOS machine. It
needs to run inside the CUDA Docker image at least once (`tests/test_flashsplat_build.py`, then
`tests/test_flashsplat_accumulation.py --checkpoint <ckpt> --colmap_dir <scene>`) before it can be
trusted.

Only `THIRD-PARTY-NOTICES.md` was modified in the pre-existing tree; the 3DGRUT submodule itself
is untouched and still on its pinned commit.

---

Sources: `README.md`, `flashsplat/README.md`, `model_eval/checkpoint_loading.py`. Verify file
paths and current git status still match before acting on anything above — this was written to
capture one session's state, not maintained going forward.
