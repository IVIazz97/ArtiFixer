#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# The unmodified MVInpainter repo (NeurIPS'24; SD1.5-inpainting + AnimateDiff motion modules, flow
# guided) on one of our mip360 scenes, then the shared graphdeco 3DGS refit. Needs
# official_inputs/<scene> (prepare_inputs.py) and its Qwen reference (reference.sbatch) first.
#   sbatch --job-name=MVI_kitchen --export=ALL,SCENE=kitchen output/jobs/mvinpainter_official/run.sbatch
# Optional: STEPS (subset of prep,infer,refit; pass a+b).
set -euo pipefail
AF=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer
source $AF/output/jobs/common.sh
STEPS=${STEPS:-prep,infer,refit}; STEPS=${STEPS//+/,}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
M=/leonardo_scratch/fast/IscrC_EditGS/opt/MVInpainter_official
D="$OUT/mvinpainter_official/$SCENE"
ROOT="$D/dataset"                              # their --dataset_root: one sub-folder per 24-frame group
RUN="mip360_$SCENE"                            # their --output_path; they write outputs/<run>_interval1.0_... under cwd
PY=$ENVS/mvinpainter_official/bin/python
echo "host=$(hostname) scene=$SCENE steps=$STEPS"

has prep && $PY $AF/output/jobs/mvinpainter_official/prepare.py --inputs "$OUT/official_inputs/$SCENE" --root "$ROOT" --nframe 24

if has infer; then
  mkdir -p "$D/outputs" && ln -sfn "$D/outputs" "$M/outputs"   # their outputs/ -> ours (one scene at a time)
  cd "$M"
  # README "removal" command for the 512 model, with our dataset root.
  PYTHONPATH="" $PY test_removal.py --load_path check_points/mvinpainter_f_512 --dataset_root "$ROOT" \
      --output_path "$RUN" --resume_from_checkpoint best --val_cfg 1.0 --img_size 512 \
      --sampling_interval 1.0 --dataset_names realworld --reference_path inpainted --nframe 24 --save_images
  cd "$AF"
fi

if has refit; then
  METHOD=mvinpainter RAW=$(ls -d "$D/outputs/${RUN}_interval1.0"*) GROUPS="$ROOT/groups.json"
  source $AF/output/jobs/official_common/refit.sh
fi
echo MVINPAINTER_OFFICIAL_DONE
