#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# AuraFusion360 port on a mip360 scene. Usage:
#   sbatch --job-name=AF_kitchen --export=ALL,SCENE=kitchen output/jobs/aurafusion/run.sbatch
# Optional: STEPS (subset of render,contour,sam2,agdd,init,sdedit,finetune), VARIANT (output subdir,
# default "lama_ref"), REF_IMAGE (inpainted reference image; default: LaMa), REF_INDEX (default 0),
# HULL_EXPAND (widen the removed object's convex hull, default 0).
set -euo pipefail
AF=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer
source $AF/output/jobs/common.sh                        # artifixer env, TMPDIR, offline HF
STEPS=${STEPS:-render,contour,sam2,agdd,init,sdedit,finetune}; STEPS=${STEPS//+/,}   # sbatch --export splits on commas: pass a+b+c
VARIANT=${VARIANT:-lama_ref}
REF_INDEX=${REF_INDEX:-0}
G="$RECON/$SCENE"; CKPT="$G/3dgrut_runs/$SCENE/$SCENE/ours_30000/ckpt_30000.pt"; COLMAP="$G/3dgrut_input/$SCENE"
LABELS="$OUT/flashsplat/$SCENE/labels.pt"
[ "$SCENE" = garden ] && LABELS="$OUT/flashsplat/garden_vase/labels.pt"   # table + vase (vase floats otherwise)
HULL_EXPAND=${HULL_EXPAND:-0}                           # >0 widens the object hull; own base dir, the removal differs
BASE="$OUT/aurafusion/$SCENE/base"
if [ "$HULL_EXPAND" != 0 ]; then BASE="$OUT/aurafusion/$SCENE/base_hull$HULL_EXPAND"; fi
R="$OUT/aurafusion/$SCENE/$VARIANT"
CK=$AF/checkpoints/aurafusion
export PYTHONPATH="$AF:$AF/thirdparty/3DGRUT-ArtiFixer:${PYTHONPATH:-}"
AF_PY=$ENVS/aurafusion/bin/python                        # SAM2 / Marigold / LeftRefill env
echo "host=$(hostname) scene=$SCENE variant=$VARIANT steps=$STEPS"
has() { [[ ",$STEPS," == *",$1,"* ]]; }

has render  && python -m aurafusion.render_views --checkpoint "$CKPT" --colmap_dir "$COLMAP" \
                 --flashsplat_dir "$OUT/flashsplat/$SCENE" --labels "$LABELS" --output_dir "$BASE" \
                 --hull_expand "$HULL_EXPAND"
has contour && python -m aurafusion.unseen_contour --render_dir "$BASE"
SAM2_ARGS=""; if [ "$HULL_EXPAND" != 0 ]; then SAM2_ARGS="--union_deleted"; fi
has sam2    && $AF_PY -m aurafusion.sam2_unseen --render_dir "$BASE" $SAM2_ARGS
# Variant dir: stage 1-3 outputs are shared through symlinks.
mkdir -p "$R"
for f in cameras.json ckpt_removed.pt removed object deleted unseen_contour unseen unseen_dilated; do
    [ -e "$R/$f" ] || ln -s "$BASE/$f" "$R/$f"
done
has agdd    && $AF_PY -m aurafusion.agdd --render_dir "$R" --colmap_dir "$COLMAP" --reference_index "$REF_INDEX" \
                 --lama_model "$AF/checkpoints/lama/big-lama.pt" --marigold "$CK/marigold-v1-0" \
                 ${REF_IMAGE:+--reference_image "$REF_IMAGE"}
has init    && python -m aurafusion.init_gaussians --render_dir "$R" --colmap_dir "$COLMAP" --reference_index "$REF_INDEX"
has sdedit  && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 $AF_PY -m aurafusion.sdedit --render_dir "$R" --reference_index "$REF_INDEX" \
                 --official_repo /leonardo_scratch/fast/IscrC_EditGS/opt/AuraFusion360_official \
                 --sd2_inpainting_ckpt "$CK/sd2-inpainting/512-inpainting-ema.ckpt"
has finetune && python -m aurafusion.finetune --render_dir "$R" --colmap_dir "$COLMAP"
echo AURAFUSION_DONE
