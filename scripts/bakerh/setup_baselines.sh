#!/usr/bin/env bash
# One-time VM setup for the Inpaint360GS and AuraFusion360 3DGUT ports (run_baselines.sh).
# Needs PyPI, download.pytorch.org and GitHub; never Hugging Face:
#   1. checks the local model folders under MODELS
#   2. clones the official AuraFusion360 repo (only its utils/LeftRefill code + learned prompt are used)
#   3. builds .venv_af2d: SAM2, Marigold, LeftRefill (pytorch_lightning 1.9, open_clip 2.16,
#      transformers<5 must stay out of the ArtiFixer .venv)
#   4. makes sure the ArtiFixer .venv has OpenCV (the ports import cv2)
#   5. links SAM2 and OpenCLIP ViT-H into the HF cache layout, so from_pretrained() works offline
#   6-7. self-test: loads every model once on GPU $GPU
# Safe to rerun: finished steps are skipped. Everything is logged to
# output/bakerh_removal/setup/SETUP_REPORT.txt; the last line says SETUP OK or SETUP FAIL.
#
# Usage (from any directory):  bash scripts/bakerh/setup_baselines.sh
# Knobs: MODELS, AF360_REPO, VENV2D, GPU (1).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AF=$(cd "$HERE/../.." && pwd)
cd "$AF"
MODELS=${MODELS:-/workspace/amazzucchelli/fbk-3dworld/models}
AF360_REPO=${AF360_REPO:-/workspace/amazzucchelli/fbk-3dworld/AuraFusion360_official}
VENV2D=${VENV2D:-$AF/.venv_af2d}
GPU=${GPU:-1}
PY=$AF/.venv/bin/python
PY2=$VENV2D/bin/python
LOG_DIR=$AF/output/bakerh_removal/setup
mkdir -p "$LOG_DIR"
REPORT=$LOG_DIR/SETUP_REPORT.txt
exec > >(tee "$REPORT") 2>&1

failed=()
step() {  # step NAME CMD...: run, print OK/FAIL, remember failures
  local name=$1; shift
  echo; echo "==================== $name ===================="
  if "$@"; then echo "---- $name: OK"; else echo "---- $name: FAIL"; failed+=("$name"); fi
}
echo "setup_baselines $(date -Is)   repo=$AF   git=$(git rev-parse --short HEAD)"
echo "MODELS=$MODELS   AF360_REPO=$AF360_REPO   VENV2D=$VENV2D   GPU=$GPU"

check_models() {
  local ok=0 f
  for f in LaMa/big-lama.pt sam2-hiera-large/sam2_hiera_large.pt \
           CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin \
           stable-diffusion-2-inpainting/512-inpainting-ema.ckpt marigold-depth-v1-0/model_index.json; do
    if [ -s "$MODELS/$f" ]; then echo "  ok       $MODELS/$f"; else echo "  MISSING  $MODELS/$f"; ok=1; fi
  done
  ls -la "$MODELS/marigold-depth-v1-0/unet" "$MODELS/marigold-depth-v1-0/vae" "$MODELS/CLIP-ViT-H-14-laion2B-s32B-b79K"
  return $ok
}

clone_af360() {
  if [ ! -d "$AF360_REPO/utils/LeftRefill" ]; then
    git clone --depth 1 https://github.com/kkennethwu/AuraFusion360_official.git "$AF360_REPO" || return 1
  fi
  local prompt=$AF360_REPO/utils/LeftRefill/check_points/ref_guided_inpainting/ckpts/epoch=7-step=6039.ckpt
  ls -la "$prompt" || return 1
  [ "$(stat -c %s "$prompt")" -gt 100000 ] || { echo "LeftRefill prompt checkpoint looks like a git-lfs pointer"; return 1; }
}

find_uv() {  # prints a uv binary, installing uv with pip if there is none
  local u
  for u in "$(command -v uv 2>/dev/null)" "$AF/.venv/bin/uv" "$HOME/.local/bin/uv"; do
    [ -n "$u" ] && [ -x "$u" ] && { echo "$u"; return 0; }
  done
  "$PY" -m pip install -q uv >&2 || python3 -m pip install -q --user uv >&2 || return 1
  for u in "$AF/.venv/bin/uv" "$HOME/.local/bin/uv"; do [ -x "$u" ] && { echo "$u"; return 0; }; done
  return 1
}

build_venv2d() {
  if [ -x "$PY2" ] && "$PY2" -c "import sam2, open_clip, pytorch_lightning, diffusers, clip, lpips" 2>/dev/null; then
    echo "found $VENV2D"; return 0
  fi
  local uv base
  uv=$(find_uv) || { echo "no uv and could not pip-install it"; return 1; }
  base=$("$PY" -c 'import sys; print(sys._base_executable)')
  echo "uv=$uv base python=$base"
  "$uv" venv --python "$base" "$VENV2D" || return 1
  "$uv" pip install --python "$PY2" torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu126 || return 1
  "$uv" pip install --python "$PY2" setuptools wheel "diffusers==0.37.1" "transformers>=4.46,<5" accelerate \
      huggingface_hub safetensors "pytorch_lightning==1.9.5" "open_clip_torch==2.16.0" kornia lpips scikit-image \
      torchmetrics omegaconf einops opencv-python-headless scipy natsort configargparse sentencepiece ftfy regex \
      hydra-core iopath pillow "numpy<2" "git+https://github.com/openai/CLIP.git" || return 1
  SAM2_BUILD_CUDA=0 "$uv" pip install --python "$PY2" --no-build-isolation "git+https://github.com/facebookresearch/sam2.git" || return 1
  "$PY2" -c "import torch, diffusers, transformers, pytorch_lightning, open_clip, clip, sam2; print('venv ok', torch.__version__, transformers.__version__)"
}

main_venv_cv2() {
  "$PY" -c "import cv2; print('cv2', cv2.__version__)" 2>/dev/null && return 0
  local uv
  if uv=$(find_uv); then "$uv" pip install --python "$PY" opencv-python-headless; else "$PY" -m pip install opencv-python-headless; fi
  "$PY" -c "import cv2; print('cv2', cv2.__version__)"
}

link_hf() {  # link_hf REPO_ID SRC_DIR FILE...: HF cache entry whose snapshot points at the local files
  local repo=$1 src=$2; shift 2
  local hub rev d f
  hub=$("$PY2" -c 'from huggingface_hub import constants; print(constants.HF_HUB_CACHE)') || return 1
  rev=$(git -C "$src" rev-parse HEAD 2>/dev/null || echo 0123456789abcdef0123456789abcdef01234567)
  d=$hub/models--${repo//\//--}
  mkdir -p "$d/refs" "$d/snapshots/$rev"
  printf '%s' "$rev" > "$d/refs/main"
  for f in "$@"; do
    if [ ! -f "$src/$f" ]; then
      [ "$f" = open_clip_config.json ] && continue
      echo "missing $src/$f"; return 1
    fi
    ln -sfn "$(readlink -f "$src/$f")" "$d/snapshots/$rev/$f"
  done
  ls -la "$d/snapshots/$rev"
}
link_models() {
  link_hf facebook/sam2-hiera-large "$MODELS/sam2-hiera-large" sam2_hiera_large.pt || return 1
  link_hf laion/CLIP-ViT-H-14-laion2B-s32B-b79K "$MODELS/CLIP-ViT-H-14-laion2B-s32B-b79K" \
      open_clip_pytorch_model.bin open_clip_config.json
}

test_main_venv() {
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$AF:$AF/thirdparty/3DGRUT-ArtiFixer" "$PY" - "$MODELS/LaMa/big-lama.pt" <<'EOF'
import sys, torch, cv2, threedgrut, flashsplat
import aurafusion.scene, inpaint360gs.lama, inpaint360gs.finetune, aurafusion.finetune
lama = torch.jit.load(sys.argv[1], map_location="cpu")
print("lama TorchScript ok,", sum(1 for _ in lama.parameters()), "tensors; cv2", cv2.__version__)
EOF
}

test_venv2d() {
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH="$AF" \
    "$PY2" - "$MODELS" "$AF360_REPO" <<'EOF'
import os, sys, traceback
from pathlib import Path
import torch
models, repo = Path(sys.argv[1]), Path(sys.argv[2])
bad = []
def check(name, fn):
    try:
        fn(); print(f"  ok    {name}", flush=True)
    except Exception:
        traceback.print_exc(); print(f"  FAIL  {name}", flush=True); bad.append(name)

def sam2():
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
def lama():
    torch.jit.load(str(models / "LaMa" / "big-lama.pt"), map_location="cpu")
def marigold():
    from diffusers import MarigoldDepthPipeline
    try:
        MarigoldDepthPipeline.from_pretrained(str(models / "marigold-depth-v1-0"), variant="fp16", torch_dtype=torch.float16)
        print("        (fp16 variant)")
    except (OSError, ValueError):
        MarigoldDepthPipeline.from_pretrained(str(models / "marigold-depth-v1-0"), torch_dtype=torch.float16)
        print("        (no fp16 files; fp32 weights cast to fp16)")
def openclip():
    import open_clip
    open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
def leftrefill():  # what aurafusion/sdedit.py does before LeftRefill: SD2 ckpt + CLIP + learned prompt
    import aurafusion.sdedit  # noqa: F401  (import check only)
    leftrefill = (repo / "utils" / "LeftRefill").resolve()
    pretrained = leftrefill / "pretrained_models" / "512-inpainting-ema.ckpt"
    if not pretrained.exists():
        pretrained.parent.mkdir(parents=True, exist_ok=True)
        pretrained.symlink_to((models / "stable-diffusion-2-inpainting" / "512-inpainting-ema.ckpt").resolve())
    os.chdir(leftrefill); sys.path.insert(0, str(leftrefill)); sys.argv = [sys.argv[0]]
    import sdedit_utils  # noqa: F401  (builds SD2-inpainting + LeftRefill, ~1 min)

for name, fn in [("sam2", sam2), ("lama", lama), ("marigold", marigold), ("open_clip ViT-H", openclip),
                 ("LeftRefill (SD2 + CLIP + prompt)", leftrefill)]:
    check(name, fn)
sys.exit(1 if bad else 0)
EOF
}

step "1 model folders" check_models
step "2 AuraFusion360 code" clone_af360
step "3 venv for SAM2/Marigold/LeftRefill" build_venv2d
step "4 OpenCV in the ArtiFixer .venv" main_venv_cv2
step "5 HF cache links (SAM2, OpenCLIP)" link_models
step "6 self-test: ArtiFixer .venv" test_main_venv
step "7 self-test: models in $VENV2D" test_venv2d

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "SETUP OK   next: bash scripts/bakerh/run_baselines.sh"
else
  echo "SETUP FAIL: ${failed[*]}   send back $REPORT"
  exit 1
fi
