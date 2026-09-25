#!/usr/bin/env bash
# Vanilla ArtiFixer3D+ (no FlashSplat, no removal, no masks) on TA_BLUE_MOTOR, then PCV, on GPU 1:
# the existing 3DGUT model renders a smooth camera path offset from the photos, ArtiFixer fixes those
# renders with the photos as references, ArtiFixer3D distills photos + fixed frames into a new
# 3DGUT, and ArtiFixer3D+ fixes that model's renders of the path.
# (run_removal.sh SCENE vanilla does each scene, see its header; run_all.sh does the detaching,
# logging and REPORT.txt exactly as for the removal jobs.)
#
# Usage (from any directory):
#   bash scripts/bakerh/run_vanilla.sh        starts in the background
#   bash scripts/bakerh/run_vanilla.sh --fg   runs in this terminal instead
# Report: output/bakerh_removal/logs/latest/REPORT.txt, the same place as run_all.sh, so the two
# cannot run at the same time on the GPU. Outputs: output/bakerh_removal/<scene>/vanilla/
# Knobs: TRAJ_SIDE (2) and TRAJ_UP (0.5) set how far the path is from the photos, in median photo
# spacings; GPU (1), NUM_VIEWS (6), AF3D_STEPS (30000), FORCE=1 pass through.
export JOBS=${JOBS:-"TA_BLUE_MOTOR:vanilla PCV:vanilla"}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_all.sh" "$@"
