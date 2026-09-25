#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# Removal-split variants + single-rollout ArtiFixer, all under artifixer/<scene>/<name>/{input,artifixer}.
# VARIANTS: space-separated name:opacity:refs:dilate_px:hole_shape:flashsplat_subdir
#   e.g. "v8:hole0:render:60:hull:flashsplat_vase" (the subdir maps to $OUT/flashsplat/<scene>_vase)
set -euo pipefail
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/jobs/common.sh
R="$OUT/artifixer/$SCENE"
for v in $VARIANTS; do
  IFS=: read -r name opacity refs dilate shape fs <<< "$v"
  echo "=== $SCENE $name opacity=$opacity refs=$refs dilate=${dilate:-12} shape=${shape:-mask} fs=${fs:-flashsplat}"
  FS="$OUT/flashsplat/$SCENE"; [ -n "${fs:-}" ] && [ "$fs" != flashsplat ] && FS="$OUT/flashsplat/${SCENE}_${fs#flashsplat_}"
  python output/jobs/build_removal_split.py --scene_root "$RECON/$SCENE" --flashsplat_dir "$FS" \
      --mask_dir "$MASKS/$SCENE" --output_dir "$R/$name/input" \
      --opacity "$opacity" --refs "$refs" --dilate_px "${dilate:-12}" --hole_shape "${shape:-mask}"
  python -m model_eval.run_inference --evalset reconstructed_colmap \
      --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
      --save_dir "$R/$name/artifixer" --split_path "$R/$name/input/split.json" \
      --render_trajectory all_frames --replace_if_exists
done
echo STAGE_C_DONE
