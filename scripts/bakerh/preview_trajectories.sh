#!/usr/bin/env bash
# Preview of the novel paths before any ArtiFixer run: for TA_BLUE_MOTOR, PCV and TA_TURBINE and every
# path type (object, vidsplat, travel) it builds the path and renders it with the scene's 3DGUT model,
# nothing else (run_trajectories.sh with PREVIEW=1: phases [vobject,] vprep, vdebug). Minutes per job
# instead of hours. Per job, in output/bakerh_removal/<scene>/vanilla_<mode>/:
#   trajectory_input.png        top view: walls at camera height, photos, path (and object / orbit anchors)
#   trajectory_input_info.json  per-frame offset scale or per-orbit stats, rejected candidates, failure reason
#   debug/0_debug.mp4           the path frame by frame: top view with the current camera | 3DGUT render
#   <scene>/recon_results/<scene>/reconstruction/<scene>/ours_30000/trajectory/renders/   the renders
# A path that cannot be made safe fails its job with the reason in REPORT.txt.
# When the paths look right, run the real thing: bash scripts/bakerh/run_trajectories.sh
# It reuses these paths and renders as long as the TRAJ_* settings stay the same.
#
# Usage (from any directory):
#   bash scripts/bakerh/preview_trajectories.sh        checks the inputs, then starts in the background
#   bash scripts/bakerh/preview_trajectories.sh --fg   runs in this terminal instead
# Knobs as run_trajectories.sh: JOBS / SCENES / MODES (e.g. MODES=vidsplat), TRAJ_* , GPU, FORCE=1.
# Report: output/bakerh_removal/logs/latest/REPORT.txt
export PREVIEW=1
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_trajectories.sh" "$@"
