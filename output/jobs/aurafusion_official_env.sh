#!/bin/bash
# Build $ENVS/aurafusion_official: the unmodified AuraFusion360 repo (2DGS), run as a baseline
# against our 3DGUT port. Separate from $ENVS/aurafusion because it needs the 2DGS CUDA
# rasterizer, open3d and a torch old enough to build it.
# Run on the login node (compute nodes have no internet). ~15 min.
set -e
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/env.sh >/dev/null 2>&1
export TMPDIR=/leonardo_scratch/fast/IscrC_EditGS/tmp_amazzucc
O=/leonardo_scratch/fast/IscrC_EditGS/opt/AuraFusion360_official
uv venv --python 3.10 --managed-python --allow-existing $ENVS/aurafusion_official
source $ENVS/aurafusion_official/bin/activate
uv pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install ninja "setuptools<81" wheel "numpy<2"   # pytorch_lightning 1.9 needs pkg_resources, dropped in setuptools 81
# 2DGS rasterizer + knn: old CUDA extensions, so no build isolation (they need the installed torch).
TORCH_CUDA_ARCH_LIST=8.0 uv pip install --no-build-isolation $O/submodules/diff-surfel-rasterization
TORCH_CUDA_ARCH_LIST=8.0 uv pip install --no-build-isolation $O/submodules/simple-knn
uv pip install open3d==0.18.0 mediapy lpips scikit-image tqdm trimesh plyfile opencv-python-headless \
    huggingface_hub diffusers accelerate "transformers<5" ConfigArgParse tensorboard timm natsort \
    pytorch-fid torchmetrics bitsandbytes "matplotlib==3.8.0" "einops==0.6.1" "pytorch_lightning==1.9.5" "open_clip_torch==2.16.0" \
    pie-torch kornia omegaconf ftfy regex sentencepiece hydra-core iopath "git+https://github.com/openai/CLIP.git"
SAM2_BUILD_CUDA=0 uv pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
python -c "import torch, open3d, diff_surfel_rasterization, simple_knn, sam2, open_clip, pytorch_lightning; \
print('aurafusion_official env ok', torch.__version__)"
