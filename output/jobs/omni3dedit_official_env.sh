#!/bin/bash
# Build $ENVS/omni3dedit_official: the unmodified Omni-3DEdit repo (CVPR'26, SEVA-based feed-forward
# multi-view editing). Their requirements.txt pins torch 2.5.0 + an xformers git commit; we take the
# requirement pins as-is except torch 2.5.1 + the prebuilt xformers 0.0.28.post3 wheel (same xformers
# series, avoids a 40-min source build) and VGGT from the same pinned commit.
# Run on the login node (compute nodes have no internet).
set -e
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/env.sh >/dev/null 2>&1
export TMPDIR=/leonardo_scratch/fast/IscrC_EditGS/tmp_amazzucc
O=/leonardo_scratch/fast/IscrC_EditGS/opt/Omni3DEdit_official
uv venv --python 3.10 --managed-python --allow-existing $ENVS/omni3dedit_official
source $ENVS/omni3dedit_official/bin/activate
uv pip install torch==2.5.1 torchvision==0.20.1 xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124
grep -v -E '^(torch|torchvision|torchaudio|triton|xformers|nvidia-|-e |jupyter|notebook|ipykernel|ipywidgets|widgetsnbextension|gevent|embreex|yt-dlp)' \
    $O/requirements.txt > $TMPDIR/omni3dedit_reqs.txt
uv pip install -r $TMPDIR/omni3dedit_reqs.txt
uv pip install --no-deps "git+https://github.com/facebookresearch/vggt.git@b02cc03ceee70821ed1231a530c1992507ef9862"
python -c "import torch, xformers, vggt, open_clip, pytorch_lightning, kornia; print('omni3dedit_official env ok', torch.__version__, xformers.__version__)"
