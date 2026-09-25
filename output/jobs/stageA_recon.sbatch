#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# Usage: sbatch --job-name=A_garden --export=ALL,SCENE=garden stageA_recon.sbatch
set -euo pipefail
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/jobs/common.sh
echo "host=$(hostname) scene=$SCENE"; nvidia-smi --query-gpu=name --format=csv,noheader

python -m data_processing.prepare_colmap_artifixer_inputs \
    --colmap_dir "$AF/datasets/mip360_ds4/$SCENE" \
    --output_root "$RECON/$SCENE" \
    --selected_image_names_file "$AF/datasets/mip360_ds4/$SCENE/train_images.txt" \
    --phases prepare,reconstruct,render,scale \
    --reconstruction_steps 30000 \
    --max_gaussians 2000000

python output/jobs/write_caption.py \
    --caption "${CAPTION[$SCENE]}" \
    --output_path "$RECON/$SCENE/captions/$SCENE/caption.h5" \
    --text_encoder_model_id "$MODEL_ID"

echo STAGE_A_DONE
