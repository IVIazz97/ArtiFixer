#!/usr/bin/env bash
# The three ArtiFixer reconstruction loops on TA_BLUE_MOTOR, PCV, TA_TURBINE and G35, one job after the
# other on GPU 1: first every scene's loop_vidsplat, then every scene's once_vidsplat, then every
# scene's scratch_vidsplat. Each runs up to LOOP_ROUNDS (5) rounds of new vidsplat paths fixed by
# ArtiFixer, then ArtiFixer3D+ once on the final model:
#   loop_vidsplat     "+5K":   each round continues the previous model (round 1: the reconstruction)
#                              for LOOP_STEPS (5000) steps on photos + every fixed frame so far
#   once_vidsplat     "+30K":  no training between rounds; ArtiFixer3D from scratch once (30000
#                              steps) on photos + all rounds' fixed frames
#   scratch_vidsplat  "+150K": each round retrains ArtiFixer3D from scratch (30000 steps) on photos +
#                              every fixed frame so far
#
# Usage (from any directory):
#   bash scripts/bakerh/run_loop_all.sh        checks the inputs, then starts in the background
#   bash scripts/bakerh/run_loop_all.sh --fg   runs in this terminal instead
# Rerunning resumes: finished rounds are kept. A loop folder made with other settings, or by code from
# before the settings were recorded, moves to <scene>/previous/ by itself; FRESH=1 moves every one.
# Knobs: LOOP_VARIANTS ("loop once scratch"), SCENES, MODES (vidsplat), G35_ROOT (G35's scene folder,
# default output/bakerh_undis/G35), LOOP_ROUNDS, LOOP_STEPS, LOOP_LR, AF3D_STEPS, TRAJ_*, GPU.
# Report: output/bakerh_removal/logs/latest/REPORT.txt. Outputs: output/bakerh_removal/<scene>/<variant>/
# (round_<k>/, final -> last round, final/af3d_plus/, debug/0_debug.mp4; see run_loop.sh).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AF=$(cd "$HERE/../.." && pwd)
SCENES=${SCENES:-"TA_BLUE_MOTOR PCV TA_TURBINE G35"}
MODES=${MODES:-vidsplat}
LOOP_VARIANTS=${LOOP_VARIANTS:-"loop once scratch"}
JOBS=""
for v in $LOOP_VARIANTS; do
  case $v in loop|once|scratch) ;; *) echo "unknown loop variant '$v' (loop, once, scratch)"; exit 2 ;; esac
  for s in $SCENES; do for m in $MODES; do JOBS="$JOBS $s:${v}_$m"; done; done
done
export JOBS=${JOBS# }

missing=0
for s in $SCENES; do
  sr=${SCENE_ROOT:-$AF/output/bakerh_undis/$s}
  [ "$s" != G35 ] || sr=${G35_ROOT:-$sr}
  ckpt=$sr/3dgrut_runs/$s/$s/ours_30000/ckpt_30000.pt
  if [ ! -f "$ckpt" ]; then
    echo "$s: no 3DGUT checkpoint at $ckpt"
    [ "$s" != G35 ] || echo "  set G35_ROOT to G35's scene folder (with 3dgrut_input/G35 and 3dgrut_runs/G35/G35/ours_30000)"
    missing=1
  fi
done
[ "$missing" -eq 0 ] || { echo "nothing started"; exit 1; }
exec bash "$HERE/run_trajectories.sh" "$@"
