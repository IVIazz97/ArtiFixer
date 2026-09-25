#!/bin/bash
# Build $ENVS/mvinpainter_official: the unmodified MVInpainter repo (NeurIPS'24, SD1.5-inpainting +
# AnimateDiff motion modules). Follows its README: python 3.8, requirements.txt pins, mmcv-full + mmflow,
# then the patched raft_decoder.py copied into site-packages. mmcv-full is built from source (no
# prebuilt 1.x wheels for torch 2.1); 1.6.2 because mmflow 0.5.x caps mmcv-full below 1.7. Its setup.py
# hard-codes -std=c++14, which torch 2.1 headers reject, so the sdist is built with c++17 instead.
# Run on the login node (compute nodes have no internet). ~25 min, mostly the mmcv-full build.
set -e
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/env.sh >/dev/null 2>&1
export TMPDIR=/leonardo_scratch/fast/IscrC_EditGS/tmp_amazzucc
M=/leonardo_scratch/fast/IscrC_EditGS/opt/MVInpainter_official
uv venv --python 3.8 --managed-python --allow-existing $ENVS/mvinpainter_official
source $ENVS/mvinpainter_official/bin/activate
uv pip install torch==2.1.2 torchvision==0.16.2 xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r $M/requirements.txt "setuptools<70" wheel "numpy<1.25" huggingface_hub==0.23.4
cd $TMPDIR && rm -rf mmcv-full-1.6.2* && python -m pip download --no-deps --no-binary :all: mmcv-full==1.6.2 -q
tar xzf mmcv-full-1.6.2.tar.gz && sed -i 's/c++14/c++17/g' mmcv-full-1.6.2/setup.py
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=16 uv pip install --no-build-isolation ./mmcv-full-1.6.2
cd /   # never import mmcv from inside its source tree (no compiled _ext there)
uv pip install --no-build-isolation mmflow==0.5.2
SP=$(python -c "import mmflow, os; print(os.path.dirname(mmflow.__file__))")
cp $M/check_points/mmflow/raft_decoder.py $SP/models/decoders/
python -c "import torch, xformers, mmcv, mmflow, diffusers, peft; from mmflow.apis import init_model; print('mvinpainter_official env ok', torch.__version__, mmcv.__version__)"
