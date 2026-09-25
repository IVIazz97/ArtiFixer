#!/bin/bash
# Refit step shared by the official video-prior baselines: compose.py (method outputs -> full-res
# frames + filtered COLMAP view list), then upstream graphdeco 3DGS (the reconstruction MVInpainter's
# README points to) trained on those frames, then rendered at *every* original view (-s the full
# scene, so views compose.py dropped are rendered too).
# Sourced from a method's run.sbatch with METHOD, SCENE, RAW (and GROUPS for mvinpainter) set.
set -euo pipefail
G=/leonardo_scratch/fast/IscrC_EditGS/opt/gaussian-splatting_official
GPY=$ENVS/gaussian-splatting/bin/python
D="$OUT/${METHOD}_official/$SCENE"
$GPY $AF/output/jobs/official_common/compose.py --method "$METHOD" --inputs "$OUT/official_inputs/$SCENE" \
    --raw "$RAW" ${GROUPS:+--groups "$GROUPS"} --colmap_dir "$RECON/$SCENE/3dgrut_input/$SCENE" \
    --graphdeco "$G" --output_dir "$D"
cd "$G"
PYTHONPATH="" $GPY train.py -s "$D/colmap" -m "$D/3dgs" --disable_viewer --quiet --test_iterations -1
PYTHONPATH="" $GPY render.py -m "$D/3dgs" -s "$RECON/$SCENE/3dgrut_input/$SCENE" --skip_test --quiet
cd "$AF"
echo "REFIT_DONE $D/3dgs/train/ours_30000/renders"
