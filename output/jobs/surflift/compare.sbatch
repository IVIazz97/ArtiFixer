#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# Panel + fidelity of the SurfaceLift kitchen runs vs the AuraFusion360 port:
#   sbatch --export=ALL,SCENE=kitchen output/jobs/surflift/compare.sbatch
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/jobs/common.sh
export PYTHONPATH="$AF:$AF/thirdparty/3DGRUT-ArtiFixer:${PYTHONPATH:-}"
G="$RECON/$SCENE"
python -m surflift.compare --checkpoint "$G/3dgrut_runs/$SCENE/$SCENE/ours_30000/ckpt_30000.pt" --colmap_dir "$G/3dgrut_input/$SCENE" \
    --runs $OUT/surflift/$SCENE/surflift $OUT/surflift/$SCENE/surfliftv_virtual \
    --other af360=$OUT/aurafusion/$SCENE/lama_ref/final/rgb --out $OUT/panels/surflift_$SCENE.jpg ${COMPARE_ARGS:-}
