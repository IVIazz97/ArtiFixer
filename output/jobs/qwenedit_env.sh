#!/bin/bash
# Build $ENVS/qwenedit and fetch Qwen/Qwen-Image-Edit-2511: the 2D editor that makes the single
# object-removed reference view both official video-prior baselines (Omni-3DEdit, MVInpainter) need.
# Omni-3DEdit's paper uses Qwen-Image edits for its cond view. Login node only (needs internet). ~20 min.
set -e
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/env.sh >/dev/null 2>&1
export TMPDIR=/leonardo_scratch/fast/IscrC_EditGS/tmp_amazzucc
uv venv --python 3.11 --managed-python --allow-existing $ENVS/qwenedit
source $ENVS/qwenedit/bin/activate
uv pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install "diffusers>=0.36" "transformers>=4.51,<5" accelerate safetensors sentencepiece pillow scipy huggingface_hub
python -c "from diffusers import QwenImageEditPlusPipeline; print('qwenedit env ok')"
hf download Qwen/Qwen-Image-Edit-2511
echo QWENEDIT_READY
