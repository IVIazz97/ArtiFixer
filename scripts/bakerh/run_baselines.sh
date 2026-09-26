#!/usr/bin/env bash
# Inpaint360GS and AuraFusion360 3DGUT ports on TA_BLUE_MOTOR, PCV and Compressor, one after the
# other on GPU 1 (six jobs: run_removal.sh SCENE i360 / af360, see its "Baselines" section).
# Run scripts/bakerh/setup_baselines.sh once first. run_all.sh does the detaching, logging and
# REPORT.txt exactly as for the ArtiFixer jobs.
#
# Usage (from any directory):
#   bash scripts/bakerh/run_baselines.sh        starts in the background
#   bash scripts/bakerh/run_baselines.sh --fg   runs in this terminal instead
# Compressor views: COMPRESSOR_FRAMES (0-based inclusive ranges, same numbering as the ArtiFixer
# FRAMES; default 49-148,284-304), e.g.  COMPRESSOR_FRAMES=0-399 bash scripts/bakerh/run_baselines.sh
# TA_BLUE_MOTOR and PCV use every view. JOBS="PCV:af360" runs a subset; REF_INDEX, GPU pass through.
# Report: output/bakerh_removal/logs/latest/REPORT.txt (one run at a time with run_all.sh).
# Outputs: output/bakerh_removal/<scene>/{i360,af360}/final/rgb
export JOBS=${JOBS:-"TA_BLUE_MOTOR:i360 TA_BLUE_MOTOR:af360 PCV:i360 PCV:af360 Compressor:i360 Compressor:af360"}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_all.sh" "$@"
