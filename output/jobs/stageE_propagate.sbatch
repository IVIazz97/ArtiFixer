#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# Progressive reference propagation. Usage:
# sbatch --job-name=E_garden --export=ALL,SCENE=garden,VARIANT=v5,SEEDS=0-6 (use + between ranges, e.g. 0-6+116-122) stageE_propagate.sbatch
set -euo pipefail
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/jobs/common.sh
G="$RECON/$SCENE"; A="$OUT/artifixer/$SCENE"
python output/jobs/propagate_removal.py --scene_root "$G" --removal_dir "$A/$VARIANT/input" \
    --seed_pred_dir "${SEED_DIR:-$A/$VARIANT/artifixer/artifixer-1.3b/distilled_views_reconstructed_colmap_auto12_evenly_spaced_sink7_all_frames/$SCENE/frames/batch_0000/pred}" \
    --seed_frames "${SEEDS//+/,}" --output_dir "$A/propagation/${OUTNAME:-artifixer_seed}" \
    --outside "${OUTSIDE:-photo}" --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID"
echo STAGE_E_DONE
