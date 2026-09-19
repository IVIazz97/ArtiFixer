# Artifixer Environment Recovery Note

Last verified: 2026-09-01

## What happened

This repo was checked after a Docker container restart while recovering UV-managed environments in the workspace.

The Artifixer `.venv` was still healthy because `.venv/bin/python` points to `/usr/bin/python3.12`, not the missing UV-managed Python cache under `/root/.local/share/uv/python`.

## Verified working state

- Local env: `.venv`
- Python: `3.12.3`
- Torch: `2.11.0+cu128`
- Torchvision: `0.26.0+cu128`
- `torch.cuda.is_available()`: `True`
- GPU seen during verification: NVIDIA A100-SXM4-40GB, `sm_80`
- CUDA toolkit to use for this container: `/usr/local/cuda-12.8`

## Important detail

On A100 (`sm_80`), Artifixer uses PyTorch auto SDPA/cuDNN instead of requiring external FA3/FA4 packages. The original sanity check treated FlashAttention packages as required even on A100, which made the env look broken despite the runtime path being valid. The sanity check is now hardware-aware: FA3/FA4 are required only on Hopper/Blackwell-style paths where the code can use them.

For 3DGRUT-related scripts, keep this path exported:

```bash
export PYTHONPATH="$PWD/thirdparty/3DGRUT-ArtiFixer:${PYTHONPATH:-}"
```

## Smoke checks that passed

- `torch`
- `torchvision`
- `diffusers`
- `transformers`
- `accelerate`
- Diffusers Wan internals used by Artifixer
- `threedgrut`
- `threedgrut.datasets.camera_models`
- `model_training.net.transformer`
- Artifixer training/data utility imports from `tests/container_sanity_check.py`

## Recommended way to reactivate this repo

From the workspace root:

```bash
source tmp/activate_repo_env.sh artifixer
```

From inside this repo:

```bash
source ../tmp/activate_repo_env.sh artifixer
```

## Quick health check

From inside this repo:

```bash
PYTHONPATH="$PWD/thirdparty/3DGRUT-ArtiFixer:${PYTHONPATH:-}" \
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-} \
.venv/bin/python tests/container_sanity_check.py
```

## Workspace-level runbook

See `../tmp/environment_recovery.md` for the full workspace recovery record.