#!/bin/bash
# Build $ENVS/aurafusion: the 2D models of the AuraFusion360 port (SAM2, Marigold AGDD, LeftRefill SDEdit).
# Kept separate from $ENVS/artifixer: LeftRefill's vendored ldm needs pytorch_lightning/open_clip/clip
# and transformers<5, which the ArtiFixer (Wan) stack must not see.
set -e
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/env.sh >/dev/null 2>&1
uv venv --python 3.12 --managed-python $ENVS/aurafusion
source $ENVS/aurafusion/bin/activate
uv pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install "diffusers==0.37.1" "transformers>=4.46,<5" accelerate huggingface_hub safetensors \
    "pytorch_lightning==1.9.5" "open_clip_torch==2.16.0" kornia lpips scikit-image torchmetrics omegaconf einops \
    opencv-python-headless scipy natsort configargparse sentencepiece ftfy regex hydra-core iopath pillow "numpy<2" \
    "git+https://github.com/openai/CLIP.git"
SAM2_BUILD_CUDA=0 uv pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
python -c "import torch, diffusers, transformers, pytorch_lightning, open_clip, clip, sam2; print('aurafusion env ok', torch.__version__, diffusers.__version__, transformers.__version__, pytorch_lightning.__version__)"
