#!/usr/bin/env bash
# Vanilla ArtiFixer3D+ with every novel-path type on TA_BLUE_MOTOR, PCV and TA_TURBINE: nine jobs,
# one after the other on GPU 1. Each job builds its path (make_trajectory.py), renders it with the
# scene's 3DGUT model, fixes the renders with ArtiFixer (photos as references), distills photos +
# fixed frames into a new 3DGUT (ArtiFixer3D) and fixes that model's renders (ArtiFixer3D+):
#   vanilla_object    orbits the object's FlashSplat centre of mass, scaled to its size
#   vanilla_vidsplat  VidSplat-style (SIGGRAPH 2026) short orbits about the surface points the photos see
#   vanilla_travel    the earlier weave across the direction of travel
# All three keep clear of walls; a path that cannot be made safe fails its job (the reason is in
# REPORT.txt) and the next job runs. run_removal.sh documents every phase and TRAJ_* knob.
#
# Usage (from any directory):
#   bash scripts/bakerh/run_trajectories.sh        checks the inputs, then starts in the background
#   bash scripts/bakerh/run_trajectories.sh --fg   runs in this terminal instead
# Rerunning skips what is finished (same trajectory settings) and redoes only what failed or is missing.
# Knobs: JOBS (a subset, e.g. JOBS="TA_TURBINE:vanilla_vidsplat"), GPU (1), NUM_VIEWS (6),
# TRAJ_MAX_FRAMES (cap on path frames, for a scene with many photos), TRAJ_* (see run_removal.sh), FORCE=1.
# Report: output/bakerh_removal/logs/latest/REPORT.txt (one run at a time with run_all.sh).
# Outputs: output/bakerh_removal/<scene>/vanilla_<mode>/  (trajectory_input.png: top view of the path;
# artifixer/: ArtiFixer on the path; <scene>/artifixer3d/: ArtiFixer3D; af3d_plus/: ArtiFixer3D+;
# debug/0_debug.mp4: path top view with the current camera | 3DGUT render | ArtiFixer | ArtiFixer3D+,
# plus debug/1_path_render.mp4, 2_artifixer.mp4, 3_artifixer3d_plus.mp4)
# vanilla_object takes the object from the FlashSplat extraction the scene already has
# (<scene root>/flashsplat_out*/labels.pt, or this pipeline's own); it segments only if there is none.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AF=$(cd "$HERE/../.." && pwd)
SCENES=${SCENES:-"TA_BLUE_MOTOR PCV TA_TURBINE"}
MODES=${MODES:-"object vidsplat travel"}
PREFIX=${PREFIX:-vanilla}  # run_loop.sh: loop
if [ -z "${JOBS:-}" ]; then
  JOBS=""
  for s in $SCENES; do for m in $MODES; do JOBS="$JOBS $s:${PREFIX}_$m"; done; done
fi
export JOBS=${JOBS# }

# Preflight (read-only): what each scene has, so a missing input shows now, not hours into the run.
SAM_MASKS=${SAM_MASKS:-/workspace/amazzucchelli/fbk-3dworld/FlashSplat/sam3_single_object_anchor_tracking}
echo "jobs: $JOBS"
for s in $(printf '%s\n' $JOBS | cut -d: -f1 | awk '!seen[$0]++'); do
  sr=${SCENE_ROOT:-$AF/output/bakerh_undis/$s}
  [ "$s" != G35 ] || sr=${G35_ROOT:-$sr}
  ckpt=$sr/3dgrut_runs/$s/$s/ours_30000/ckpt_30000.pt
  photos=$(find "$sr/3dgrut_input/$s/images" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
  labels=""
  for d in "$sr"/flashsplat_out* "$AF/output/bakerh_removal/$s/flashsplat"; do
    [ -f "$d/labels.pt" ] || continue
    if [ -f "$d/hit_count.pt" ]; then labels=$d; break; fi
    [ -n "$labels" ] || labels=$d
  done
  if [ -n "$labels" ]; then
    object="existing FlashSplat extraction ${labels#$AF/}$([ -f "$labels/hit_count.pt" ] || echo ' (labels only)')"
  elif [ -d "$SAM_MASKS/$s/full/masks_npy" ] || [ -d "$SAM_MASKS/$s/0_400/masks_npy" ]; then
    object="SAM masks found (vobject segments the object first)"
  else
    object="NO labels or SAM masks: ${PREFIX}_object will fail (the other modes still run)"
  fi
  scale=$(sed -n 's/^Scale factor: *\([^ ]*\).*/\1/p' "$sr/metric_alignment/scale_info.txt" 2>/dev/null | head -1)
  printf '  %-14s checkpoint %-8s photos %-5s metric scale %-10s object: %s\n' "$s" \
    "$([ -f "$ckpt" ] && echo ok || echo MISSING)" "${photos:-0}" "${scale:-none}" "$object"
  if [ "${photos:-0}" -gt 600 ] && [ -z "${TRAJ_MAX_FRAMES:-}" ]; then
    echo "    $s has $photos photos: one path frame each may not fit through ArtiFixer (923 did not). Consider TRAJ_MAX_FRAMES=400."
  fi
done
exec bash "$HERE/run_all.sh" "$@"
