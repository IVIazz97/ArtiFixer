#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# Inpaint360GS port on a mip360 scene. Usage:
#   sbatch --job-name=I360_kitchen --export=ALL,SCENE=kitchen output/jobs/inpaint360gs/run.sbatch
# Optional:
#   PRED_IOU         SAM2 predicted-IoU threshold (default 0.8); STABILITY, CROP_LAYERS likewise
#   OVERWRITE=1      redo raw masks that already exist
#   STEPS            subset of masks,associate,distill,remove,virtual,nbs,lama,init,finetune (pass a+b+c)
#   REMOVAL          distill (paper: association + distilled ids, default) or flashsplat (shared removal)
#   TARGET_IDS       ids to remove, space separated (default: picked from the reference masks $MASKS/$SCENE)
#   SURROUNDING_IDS  occluders removed only while inpainting, space separated
#   HULL_EXPAND      scale the removal's convex hull about the object's centre (default 1.0)
#   NBS_MODE         footprint (default) or sam2
#   RECURSIVE=1      LaMa recursive conditional inpainting
#   VARIANT          output subdir (default: $REMOVAL)
set -euo pipefail
AF=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer
source $AF/output/jobs/common.sh                        # artifixer env, TMPDIR, offline HF
STEPS=${STEPS:-masks,associate,distill,remove,virtual,nbs,lama,init,finetune}; STEPS=${STEPS//+/,}
REMOVAL=${REMOVAL:-distill}
HULL_EXPAND=${HULL_EXPAND:-1.0}
NBS_MODE=${NBS_MODE:-footprint}
VARIANT=${VARIANT:-$REMOVAL}
G="$RECON/$SCENE"; CKPT="$G/3dgrut_runs/$SCENE/$SCENE/ours_30000/ckpt_30000.pt"; COLMAP="$G/3dgrut_input/$SCENE"
BASE="$OUT/inpaint360gs/$SCENE/base"; R="$OUT/inpaint360gs/$SCENE/$VARIANT"
export PYTHONPATH="$AF:$AF/thirdparty/3DGRUT-ArtiFixer:${PYTHONPATH:-}"
AF_PY=$ENVS/aurafusion/bin/python                        # SAM2 env
echo "host=$(hostname) scene=$SCENE removal=$REMOVAL hull_expand=$HULL_EXPAND variant=$VARIANT steps=$STEPS"
has() { [[ ",$STEPS," == *",$1,"* ]]; }

# Segmentation (stages 1-3) is shared by every variant.
has masks     && $AF_PY -m inpaint360gs.raw_masks --image_dir "$COLMAP/images" --output_dir "$BASE" \
                    --pred_iou_thresh "${PRED_IOU:-0.8}" --stability_score_thresh "${STABILITY:-0.95}" \
                    --crop_n_layers "${CROP_LAYERS:-0}" ${OVERWRITE:+--overwrite}
has associate && python -m inpaint360gs.associate --checkpoint "$CKPT" --colmap_dir "$COLMAP" --output_dir "$BASE"
has distill   && python -m inpaint360gs.distill --checkpoint "$CKPT" --colmap_dir "$COLMAP" --output_dir "$BASE"
mkdir -p "$R"
for f in raw_masks associated associated_color associated_numbered identity; do
    [ -e "$R/$f" ] || ln -s "$BASE/$f" "$R/$f"
done

if has remove; then
    if [ "$REMOVAL" = flashsplat ]; then
        LABELS="$OUT/flashsplat/$SCENE/labels.pt"
        [ "$SCENE" = garden ] && LABELS="$OUT/flashsplat/garden_vase/labels.pt"   # table + vase
        python -m inpaint360gs.remove --checkpoint "$CKPT" --colmap_dir "$COLMAP" --output_dir "$R" \
            --flashsplat_dir "$OUT/flashsplat/$SCENE" --flashsplat_labels "$LABELS" --hull_expand "$HULL_EXPAND"
    else
        if [ -n "${TARGET_IDS:-}" ]; then TARGET=(--target_ids $TARGET_IDS); else TARGET=(--target_masks_dir "$MASKS/$SCENE"); fi
        python -m inpaint360gs.remove --checkpoint "$CKPT" --colmap_dir "$COLMAP" --output_dir "$R" "${TARGET[@]}" \
            --hull_expand "$HULL_EXPAND" ${SURROUNDING_IDS:+--surrounding_ids $SURROUNDING_IDS}
    fi
fi
has virtual  && python -m inpaint360gs.virtual_views --checkpoint "$CKPT" --colmap_dir "$COLMAP" --output_dir "$R"
if has nbs; then
    NBS_PY=python; [ "$NBS_MODE" = sam2 ] && NBS_PY=$AF_PY
    $NBS_PY -m inpaint360gs.nbs_masks --output_dir "$R" --mode "$NBS_MODE"
fi
has lama     && python -m inpaint360gs.lama --output_dir "$R" --lama_model "$AF/checkpoints/lama/big-lama.pt" \
                    ${RECURSIVE:+--recursive_guide}
has init     && python -m inpaint360gs.init_gaussians --output_dir "$R" --colmap_dir "$COLMAP"
has finetune && python -m inpaint360gs.finetune --output_dir "$R" --checkpoint "$CKPT" --colmap_dir "$COLMAP"
echo INPAINT360GS_DONE
