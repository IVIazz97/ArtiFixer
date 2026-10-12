#!/usr/bin/env bash
# Inpainting comparison on the bakerh scenes TA_BLUE_MOTOR, PCV and Compressor (views
# COMPRESSOR_FRAMES, default 49-148,284-304), one job after the other on GPU 1. Per scene:
#   af360      AuraFusion360 3DGUT port; its AGDD depth already uses the robust 0.99-quantile range
#   af360v     the same, with virtual cameras that hold the whole hole in frame: the reference view
#              comes from them, and only views with the hole fully in frame supervise its inside
#   surflift   SurfaceLift (ours) on 3DGUT: observation ledger, single-view surface-bound fill from
#              the best real view, LeftRefill refinement
#   surfliftv  the same, filled from a virtual view that holds the whole hole in frame
# All four use the same FlashSplat removal. Finished jobs are skipped in seconds (af360 ran on all
# three scenes on 2026-09-28), so rerunning this after a failure only redoes what is missing.
# The report starts with the output of check_surflift.sh (models, envs, finished runs), then one
# section per job: phase durations, peak GPU memory, the job's job_summary.txt (chosen views,
# camera counts) and, for a failed job, the last 80 lines of the failing phase's log.
# Needs setup_baselines.sh to have passed once. run_all.sh does the detaching, logging and REPORT.txt.
#
# Usage (from any directory):
#   bash scripts/bakerh/run_inpaint.sh        starts in the background
#   bash scripts/bakerh/run_inpaint.sh --fg   runs in this terminal instead
# Knobs: SCENES ("TA_BLUE_MOTOR PCV Compressor"), VARIANTS ("af360 af360v surflift surfliftv"),
# JOBS (an explicit list, e.g. JOBS="PCV:surfliftv"), COMPRESSOR_FRAMES, GPU, FORCE=1 (redo finished jobs).
# Report: output/bakerh_removal/logs/latest/REPORT.txt. Outputs: output/bakerh_removal/<scene>/<variant>/
SCENES=${SCENES:-"TA_BLUE_MOTOR PCV Compressor"}
VARIANTS=${VARIANTS:-"af360 af360v surflift surfliftv"}
if [ -z "${JOBS:-}" ]; then
  for s in $SCENES; do for v in $VARIANTS; do JOBS="${JOBS:-} $s:$v"; done; done
fi
export JOBS
# The model/input check goes into REPORT.txt first, so one file says what is missing and what failed.
export PRECHECK=${PRECHECK-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_surflift.sh}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_all.sh" "$@"
