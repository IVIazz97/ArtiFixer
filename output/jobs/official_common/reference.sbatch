#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# Qwen-Image-Edit-2511 reference view for the official video-prior baselines (see reference.py).
#   sbatch --job-name=REF_kitchen --export=ALL,SCENE=kitchen output/jobs/official_common/reference.sbatch
set -euo pipefail
AF=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer
source $AF/output/jobs/common.sh
$ENVS/qwenedit/bin/python $AF/output/jobs/official_common/reference.py --inputs "$OUT/official_inputs/$SCENE" --scene "$SCENE"
