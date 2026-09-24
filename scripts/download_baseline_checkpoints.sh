#!/bin/bash
# Download every pretrained model the AuraFusion360 (aurafusion/) and Inpaint360GS (inpaint360gs/)
# ports need. Run it on a machine with internet (on Leonardo: the login node; compute nodes are offline).
#
#   bash scripts/download_baseline_checkpoints.sh              # models only (~9 GB)
#   WITH_OFFICIAL=1 bash scripts/download_baseline_checkpoints.sh  # + official kitchen data/results for validation
#
# Env: CKPT_DIR (default <repo>/checkpoints), HF_HOME (HF cache; SAM2 and OpenCLIP are loaded from it).
#
# | model                         | used by                          | source                                                        |
# |-------------------------------|----------------------------------|---------------------------------------------------------------|
# | SAM2 hiera-large              | aurafusion sam2_unseen,          | https://huggingface.co/facebook/sam2-hiera-large              |
# |                               | inpaint360gs raw_masks/nbs_masks |                                                               |
# | Marigold v1-0 (fp16)          | aurafusion agdd                  | https://huggingface.co/prs-eth/marigold-v1-0                  |
# | SD2-inpainting 512-ema .ckpt  | aurafusion sdedit (LeftRefill)   | https://huggingface.co/sd2-community/stable-diffusion-2-inpainting |
# | OpenCLIP ViT-H-14 laion2B     | aurafusion sdedit (LeftRefill)   | https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K  |
# | big-lama TorchScript          | inpaint360gs lama, aurafusion    | https://huggingface.co/fashn-ai/LaMa (big-lama.pt)            |
#
# The 3DGUT scene checkpoint (ckpt_30000.pt) is trained per scene, not downloaded.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CKPT_DIR=${CKPT_DIR:-$REPO/checkpoints}
AF=$CKPT_DIR/aurafusion

command -v hf >/dev/null || { echo "hf CLI not found: pip install -U huggingface_hub"; exit 1; }
echo "checkpoints -> $CKPT_DIR   HF cache -> ${HF_HOME:-$HOME/.cache/huggingface}"

# SAM2: the code calls SAM2*.from_pretrained("facebook/sam2-hiera-large"), which reads the HF cache.
hf download facebook/sam2-hiera-large
hf download facebook/sam2-hiera-large --local-dir "$AF/sam2-hiera-large"

# Marigold: passed as a local dir (--marigold $AF/marigold-v1-0).
hf download prs-eth/marigold-v1-0 --local-dir "$AF/marigold-v1-0" \
    --include "*.json" "*.txt" "tokenizer/*" "*fp16.safetensors" "vae/diffusion_pytorch_model.safetensors"

# SD2-inpainting: the stabilityai repo returns 401, sd2-community mirrors the same file.
hf download sd2-community/stable-diffusion-2-inpainting 512-inpainting-ema.ckpt --local-dir "$AF/sd2-inpainting"

# OpenCLIP ViT-H: LeftRefill's FrozenOpenCLIPEmbedder loads it through open_clip from the HF cache.
hf download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin open_clip_config.json

# big-lama TorchScript (205,803,670 bytes; inpaint360gs/lama.py asserts its layer layout).
hf download fashn-ai/LaMa big-lama.pt --local-dir "$CKPT_DIR/lama"

if [ "${WITH_OFFICIAL:-0}" = 1 ]; then
    hf download kkennethwu/360-USID --repo-type dataset --local-dir "$AF/official_data" \
        --include "Other-360/kitchen/reference/*" "Other-360/kitchen/unseen_masks/*" "Other-360/kitchen/object_masks/*"
    hf download kkennethwu/AuraFusion360_Results --repo-type dataset --local-dir "$AF/official_results" \
        --include "Other-360/kitchen/train/ours_10000_object_inpaint/renders/*" \
                  "Other-360/kitchen/train/ours_object_inpaint_init/renders/*"
fi

echo "DOWNLOAD_DONE"
ls -lh "$CKPT_DIR/lama/big-lama.pt" "$AF/sd2-inpainting/512-inpainting-ema.ckpt" "$AF/sam2-hiera-large/model.safetensors"
