#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# FlashSplat segmentation -> object extraction -> removal split -> ArtiFixer 1.3B inference.
# Usage: sbatch --job-name=B_garden --export=ALL,SCENE=garden stageB_removal.sbatch
# Set STEPS to a subset of "seg,extract,split,infer" to rerun only part.
set -euo pipefail
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/jobs/common.sh
STEPS=${STEPS:-seg,extract,split,infer}
SCENE_ROOT="$RECON/$SCENE"
CKPT="$SCENE_ROOT/3dgrut_runs/$SCENE/$SCENE/ours_30000/ckpt_30000.pt"
COLMAP="$SCENE_ROOT/3dgrut_input/$SCENE"
FS="$OUT/flashsplat/$SCENE"
REMOVAL="$OUT/artifixer/$SCENE/v0/input"
echo "host=$(hostname) scene=$SCENE steps=$STEPS"

if [[ $STEPS == *seg* ]]; then
python -m data_processing.run_flashsplat_segmentation \
    --checkpoint "$CKPT" --colmap_dir "$COLMAP" --mask_dir "$MASKS/$SCENE" \
    --num_objects 1 --output_root "$FS" \
    --outputs labels ply overlays --save-hit-count \
    --test_split_interval -1 --config_override selected_indices_file=null
fi

if [[ $STEPS == *extract* ]]; then
python -m data_processing.render_flashsplat_extraction \
    --checkpoint "$CKPT" --colmap_dir "$COLMAP" \
    --labels "$FS/labels.pt" --contribution "$FS/contribution.pt" --hit-count "$FS/hit_count.pt" \
    --normalize-by-hit-count --min-total-contribution 0.1 \
    --background-convex-hull --convex-hull-trim-percentile 99.5 \
    --background label0 --object_id 1 \
    --output_root "$FS" --save_opacity \
    --test_split_interval -1 --config_override selected_indices_file=null
fi

if [[ $STEPS == *split* ]]; then
python output/jobs/build_removal_split.py \
    --scene_root "$SCENE_ROOT" --flashsplat_dir "$FS" --mask_dir "$MASKS/$SCENE" --output_dir "$REMOVAL"
fi

if [[ $STEPS == *infer* ]]; then
python -m model_eval.run_inference \
    --evalset reconstructed_colmap \
    --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
    --save_dir "$OUT/artifixer/$SCENE/v0/artifixer" \
    --split_path "$REMOVAL/split.json" \
    --render_trajectory all_frames --replace_if_exists
fi
echo STAGE_B_DONE
