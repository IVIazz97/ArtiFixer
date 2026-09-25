#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# The unmodified AuraFusion360 repo (2DGS) on one of our mip360 scenes -- the vanilla baseline for
# the 3DGUT port in aurafusion/. Only prepare.py (data layout + configs) and reference.py (the
# inpainted reference image, which 360-USID ships and a custom scene has not) are ours.
#   sbatch --job-name=AFO_kitchen --export=ALL,SCENE=kitchen output/jobs/aurafusion_official/run.sbatch
# Optional: STEPS (subset of prep,train,remove,sam2,ref,inpaint,sdedit,finetune; pass a+b+c).
set -euo pipefail
AF=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer
source $AF/output/jobs/common.sh                        # TMPDIR, HF cache, OUT/RECON
STEPS=${STEPS:-prep,train,remove,sam2,ref,inpaint,sdedit,finetune}; STEPS=${STEPS//+/,}
O=/leonardo_scratch/fast/IscrC_EditGS/opt/AuraFusion360_official
M="$OUT/aurafusion_official/$SCENE"                     # their model_path, symlinked from $O/output/Other-360/$SCENE
D="$O/data/Other-360/$SCENE"
CFG="$O/configs/Other-360/$SCENE"
PY=$ENVS/aurafusion_official/bin/python
export PYTHONPATH=""                                    # their utils/ scene/ arguments/ must win
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PORT=$((6100 + RANDOM % 900))
cd "$O"
echo "host=$(hostname) scene=$SCENE steps=$STEPS model=$M"
has() { [[ ",$STEPS," == *",$1,"* ]]; }

has prep && PYTHONPATH="$AF" $PY $AF/output/jobs/aurafusion_official/prepare.py --official_repo "$O" --scene "$SCENE" \
    --colmap_dir "$RECON/$SCENE/3dgrut_input/$SCENE" --mask_dir "$AF/datasets/masks_objid/$SCENE" --model_dir "$M"

has train && $PY train.py --config "$CFG/train.config" --port $PORT
has remove && $PY remove.py --config "$CFG/remove.config"
has sam2 && $PY utils/sam2_utils.py --dataset Other-360 --scene "$SCENE"
has ref && PYTHONPATH="$AF" $PY $AF/output/jobs/aurafusion_official/reference.py --data_dir "$D" \
    --removal_renders "$M/train/ours_30000_object_removal/renders" --lama_model "$AF/checkpoints/lama/big-lama.pt"
has inpaint && $PY inpaint.py --config "$CFG/inpaint.config"
has sdedit && $PY utils/LeftRefill/sdedit_utils.py --config "$CFG/sdedit.config"
has finetune && $PY inpaint.py --config "$CFG/inpaint.config" --images inpaint --finetune_iteration 10000
echo AURAFUSION_OFFICIAL_DONE
