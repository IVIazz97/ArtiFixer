#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# v6/v7: LaMa-filled conditioning renders (full / zero opacity in the hole), then ArtiFixer.
# Usage: sbatch --job-name=G_garden --export=ALL,SCENE=garden,BASE=v5 stageG_lamafill.sbatch
set -euo pipefail
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/jobs/common.sh
G="$RECON/$SCENE"; A="$OUT/artifixer/$SCENE"
for v in v6:full v7:hole0; do
  IFS=: read -r name opacity <<< "$v"
  python output/jobs/lama_fill_renders.py --source_dir "$A/$BASE/input" --output_dir "$A/$name/input" \
      --model "$AF/checkpoints/lama/big-lama.pt" --opacity "$opacity"
  python -m model_eval.run_inference --evalset reconstructed_colmap \
      --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" --save_dir "$A/$name/artifixer" \
      --split_path "$A/$name/input/split.json" --render_trajectory all_frames --replace_if_exists
done
echo STAGE_G_DONE
