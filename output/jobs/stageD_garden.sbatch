#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# Garden v4 (convex-hull hole) and v5 (vase removed too), both on top of v3 (hole0 + render refs).
set -euo pipefail
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/jobs/common.sh
S=garden; G="$RECON/$S"; CKPT="$G/3dgrut_runs/$S/$S/ours_30000/ckpt_30000.pt"; FS5="$OUT/flashsplat/${S}_vase"
A="$OUT/artifixer/$S"; FS="$OUT/flashsplat/$S"
infer() { python -m model_eval.run_inference --evalset reconstructed_colmap \
    --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" --save_dir "$A/$1/artifixer" \
    --split_path "$A/$1/input/split.json" --render_trajectory all_frames --replace_if_exists; }

echo "=== v4"
python output/jobs/build_removal_split.py --scene_root "$G" --flashsplat_dir "$FS" --mask_dir "$MASKS/$S" \
    --output_dir "$A/v4/input" --opacity hole0 --refs render --hole_shape hull --dilate_px 30
infer v4

echo "=== v5"
python output/jobs/extrude_labels.py --checkpoint "$CKPT" --transforms "$G/3dgrut_input/$S/nerfstudio/transforms.json" \
    --flashsplat_dir "$FS" --output "$FS5/labels.pt"
python -m data_processing.render_flashsplat_extraction --checkpoint "$CKPT" --colmap_dir "$G/3dgrut_input/$S" \
    --labels "$FS5/labels.pt" --contribution "$FS/contribution.pt" --hit-count "$FS/hit_count.pt" \
    --normalize-by-hit-count --min-total-contribution 0.1 --background-convex-hull --convex-hull-trim-percentile 99.5 \
    --background label0 --object_id 1 --output_root "$FS5" --save_opacity \
    --test_split_interval -1 --config_override selected_indices_file=null
# The SAM2 masks miss the vase; the projected object opacity (fg_opacity) now covers it.
python output/jobs/build_removal_split.py --scene_root "$G" --flashsplat_dir "$FS5" --mask_dir "$MASKS/$S" \
    --output_dir "$A/v5/input" --opacity hole0 --refs render
infer v5
echo STAGE_D_DONE
