#!/usr/bin/env bash
# Everything still to run on the bakerh scenes, one job after the other on GPU 1:
#   1. vanilla ArtiFixer3D+ (no removal, no inpainting; a novel path next to the photos is fixed):
#        TA_BLUE_MOTOR, PCV
#   2. object removal + inpainting on Compressor (views COMPRESSOR_FRAMES, default 49-148,284-304),
#      then PCV, then TA_BLUE_MOTOR, each with
#        i360 / af360        Inpaint360GS / AuraFusion360 3DGUT ports on the FlashSplat removal
#        i360sam / af360sam  the same ports on their own SAM-based segmentation, pointed at the object
#                            by the SAM masks FlashSplat uses
#      and, for the Compressor only, ArtiFixer itself (bighull), which never finished there.
# Jobs already finished are skipped in seconds (TA and Compressor af360 from the last run), so
# rerunning this after a failure only redoes what is missing. See run_removal.sh for each variant.
# Needs setup_baselines.sh to have passed once. run_all.sh does the detaching, logging and REPORT.txt.
#
# Usage (from any directory):
#   bash scripts/bakerh/run_sequence.sh        starts in the background
#   bash scripts/bakerh/run_sequence.sh --fg   runs in this terminal instead
# Knobs: COMPRESSOR_FRAMES, JOBS (a subset, e.g. JOBS="PCV:af360sam"), GPU, FORCE=1 (redo finished jobs).
# Report: output/bakerh_removal/logs/latest/REPORT.txt. Outputs: output/bakerh_removal/<scene>/<variant>/
export JOBS=${JOBS:-"TA_BLUE_MOTOR:vanilla PCV:vanilla \
Compressor:i360 Compressor:af360 Compressor:i360sam Compressor:af360sam Compressor:bighull \
PCV:i360 PCV:af360 PCV:i360sam PCV:af360sam \
TA_BLUE_MOTOR:i360 TA_BLUE_MOTOR:af360 TA_BLUE_MOTOR:i360sam TA_BLUE_MOTOR:af360sam"}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_all.sh" "$@"
