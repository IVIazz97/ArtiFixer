#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# ArtiFixer3D (distill composites into 3DGUT) + ArtiFixer3D+ on a propagated removal result.
# Usage: sbatch --job-name=F_garden_lama --export=ALL,SCENE=garden,PROP=lama stageF_af3d.sbatch
set -euo pipefail
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/jobs/common.sh
G="$RECON/$SCENE"; A="$OUT/artifixer/$SCENE"; D="$A/af3d/$PROP"
python output/jobs/build_af3d_split.py --scene_root "$G" --propagated_dir "$A/propagation/$PROP" --output_dir "$D"
python -m data_processing.run_artifixer3d --scene_root "$D" --split_path "$D/split.json" \
    --artifixer_frames_dir "$A/propagation/$PROP/anchors" --output_root "$D/artifixer3d" \
    --artifixer3d_plus_inference_split_path "$D/split_artifixer3d_plus.json"
# ArtiFixer3D+: val_frames regenerates only the targets, each segment between two anchors is a
# fresh short rollout, so there is no long-rollout drift.
python -m model_eval.run_inference --evalset reconstructed_colmap \
    --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" --save_dir "$D/artifixer3d_plus" \
    --split_path "$D/split_artifixer3d_plus.json" --render_trajectory val_frames \
    --neighbor_selection_mode covisibility --replace_if_exists
echo STAGE_F_DONE
