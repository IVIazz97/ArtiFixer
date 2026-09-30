#!/usr/bin/env bash
# ArtiFixer reconstruction loop on TA_BLUE_MOTOR, PCV and TA_TURBINE, one scene after the other on GPU 1:
# LOOP_ROUNDS (3) rounds per scene, each adding a new novel path to the reconstruction.
#   round k: choose a path on the round k-1 model (vidsplat: orbits aimed at what neither the photos
#            nor the earlier rounds' views have seen), render all paths so far with that model,
#            ArtiFixer fixes the new frames, ArtiFixer3D distills photos + every fixed frame so far
#            (round 1 from scratch, AF3D_STEPS; later rounds continue the previous model for
#            LOOP_STEPS (10000) more steps; LOOP_DISTILL=scratch retrains each round instead)
#   end:     ArtiFixer3D+ over all paths with the last model
# Each round gets 1/LOOP_ROUNDS of the frame budget (one frame per photo in all, or TRAJ_MAX_FRAMES).
# A later round that finds no acceptable new path ends that scene's loop early with the model so far.
#
# Usage (from any directory):
#   bash scripts/bakerh/run_loop.sh        checks the inputs, then starts in the background
#   bash scripts/bakerh/run_loop.sh --fg   runs in this terminal instead
# Knobs: MODES (vidsplat; object or travel repeat that path with a shifted weave each round), SCENES,
# JOBS (e.g. JOBS="PCV:loop_vidsplat"), LOOP_ROUNDS, LOOP_STEPS, LOOP_DISTILL, TRAJ_*, GPU, FORCE=1.
# Rerunning skips finished rounds. Report: output/bakerh_removal/logs/latest/REPORT.txt
# Outputs: output/bakerh_removal/<scene>/loop_<mode>/
#   round_<k>/            the round: new_path.png (top view of the new path), trajectory_input.json (all
#                         paths so far), artifixer/ (ArtiFixer on the new frames), pred_all/ (every fixed
#                         frame so far), artifixer3d/ (the round's model, see model.txt),
#                         debug/0_debug.mp4 (the new frames: top view | render | ArtiFixer)
#   final -> round_<k>    the last round; final/af3d_plus/ is ArtiFixer3D+ over all paths
#   debug/0_debug.mp4     all paths: top view | final-model render | ArtiFixer | ArtiFixer3D+
export PREFIX=loop
export MODES=${MODES:-vidsplat}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_trajectories.sh" "$@"
