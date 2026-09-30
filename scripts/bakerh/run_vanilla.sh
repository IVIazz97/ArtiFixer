#!/usr/bin/env bash
# Vanilla ArtiFixer3D+ (no removal, no inpainting) on TA_BLUE_MOTOR, then PCV, on GPU 1:
# the existing 3DGUT model renders a novel camera path that keeps clear of walls (TRAJ_MODE: object,
# the default, orbits the object's FlashSplat centre of mass scaled to its size; vidsplat takes short
# VidSplat-style orbits about the surface points the photos look at; travel is the earlier weave),
# ArtiFixer fixes those renders with the photos as references, ArtiFixer3D distills photos + fixed
# frames into a new 3DGUT, and ArtiFixer3D+ fixes that model's renders of the path.
# (run_removal.sh SCENE vanilla does each scene, see its header; run_all.sh does the detaching,
# logging and REPORT.txt exactly as for the removal jobs.)
#
# Usage (from any directory):
#   bash scripts/bakerh/run_vanilla.sh        starts in the background
#   bash scripts/bakerh/run_vanilla.sh --fg   runs in this terminal instead
# Report: output/bakerh_removal/logs/latest/REPORT.txt, the same place as run_all.sh, so the two
# cannot run at the same time on the GPU. Outputs: output/bakerh_removal/<scene>/vanilla/
# Knobs: TRAJ_MODE (object), TRAJ_CLEARANCE_M (0.3) the minimum distance to walls; object: TRAJ_ORBIT
# (0.6), TRAJ_RADIAL (0.3), TRAJ_RISE (0.3) object radii, TRAJ_RENDER_CHECK, TRAJ_ALLOW_FALLBACK; vidsplat:
# TRAJ_CLIP (16), TRAJ_ORBIT_DEG (15,30,45), TRAJ_S_LOW (0.03), TRAJ_S_HIGH (0.4), TRAJ_D0_M (0.5),
# TRAJ_BUDGET; travel: TRAJ_SIDE (2), TRAJ_UP (0.5). A path that cannot be made safe fails the job with
# the reason in REPORT.txt. GPU (1), NUM_VIEWS (6), AF3D_STEPS (30000), FORCE=1 pass through.
export JOBS=${JOBS:-"TA_BLUE_MOTOR:vanilla PCV:vanilla"}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_all.sh" "$@"
