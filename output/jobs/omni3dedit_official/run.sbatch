#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# The unmodified Omni-3DEdit repo (CVPR'26; SEVA multi-view video diffusion, feed-forward, mask- and
# pose-free) on one of our mip360 scenes, then the shared graphdeco 3DGS refit. Needs
# official_inputs/<scene> (prepare_inputs.py) and its Qwen reference (reference.sbatch) first, and the
# gated stabilityai/stable-virtual-camera weights in the HF cache (their test_step loads SEVA too).
#   sbatch --job-name=O3E_kitchen --export=ALL,SCENE=kitchen output/jobs/omni3dedit_official/run.sbatch
# Optional: STEPS (subset of prep,infer,refit; pass a+b).
set -euo pipefail
AF=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer
source $AF/output/jobs/common.sh
STEPS=${STEPS:-prep,infer,refit}; STEPS=${STEPS//+/,}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
O=/leonardo_scratch/fast/IscrC_EditGS/opt/Omni3DEdit_official
D="$OUT/omni3dedit_official/$SCENE"
PY=$ENVS/omni3dedit_official/bin/python
echo "host=$(hostname) scene=$SCENE steps=$STEPS"

has prep && $PY $AF/output/jobs/omni3dedit_official/prepare.py --inputs "$OUT/official_inputs/$SCENE" --root "$D/data" --scene "$SCENE"

if has infer; then
  cd "$O"
  # README "Object Removal" command on 1 GPU; json folder + save dir overridden through their OmegaConf dotlist.
  CUDA_VISIBLE_DEVICES=0 USE_FLASH_ATTENTION=1 PYTHONPATH=. $PY -u -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
      --master_addr=localhost --master_port=$((12000 + RANDOM % 2000)) main.py \
      --base=configs/train/seva_edit_mmdit_remove.yaml --no_date --train=False --debug \
      --resume=checkpoints/omni3dedit_remove.ckpt \
      test_data.params.json_folder="$D/data/jsons" VERSION_DICT.options.test_save_dir="$D/results" </dev/null
  cd "$AF"
fi

if has refit; then
  METHOD=omni3dedit RAW=$(ls -d "$D/results/remove/${SCENE}_iter_"*) GROUPS=""
  source $AF/output/jobs/official_common/refit.sh
fi
echo OMNI3DEDIT_OFFICIAL_DONE
