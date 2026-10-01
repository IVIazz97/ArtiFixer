#!/usr/bin/env bash
# run_loop.sh from a clean start: every loop_<mode> folder it is about to run moves to
# output/bakerh_removal/<scene>/previous/loop_<mode>_<stamp>/ first, so no round of an earlier run
# (with other rounds, steps or seeds) is picked up as finished. Then run_loop.sh starts.
#
# Usage (from any directory):
#   bash scripts/bakerh/run_loop_fresh.sh        moves the old folders, then starts in the background
#   bash scripts/bakerh/run_loop_fresh.sh --fg   runs in this terminal instead
# Same knobs as run_loop.sh (SCENES, MODES, JOBS, LOOP_*, TRAJ_*, GPU). Running it again moves the
# results of this run aside too: to resume a stopped run instead, use run_loop.sh.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AF=$(cd "$HERE/../.." && pwd)
OUT_ROOT=${OUT_ROOT:-$AF/output/bakerh_removal}
SCENES=${SCENES:-"TA_BLUE_MOTOR PCV TA_TURBINE"}
MODES=${MODES:-vidsplat}
JOBS=${JOBS:-}
if [ -z "$JOBS" ]; then
  for s in $SCENES; do for m in $MODES; do JOBS="$JOBS $s:loop_$m"; done; done
fi
JOBS=${JOBS# }

LATEST=$OUT_ROOT/logs/latest
if [ -f "$LATEST/pgid" ] && kill -0 -- "-$(cat "$LATEST/pgid")" 2>/dev/null; then
  echo "a run is already going (log: $LATEST/all.log); nothing moved. Stop it first: kill -- -$(cat "$LATEST/pgid")"
  exit 1
fi

stamp=$(date +%Y%m%d_%H%M%S)
for job in $JOBS; do
  s=${job%%:*} variant=${job#*:}
  [[ $variant == loop_* ]] || { echo "$job is not a loop job"; exit 1; }
  if [ -e "$OUT_ROOT/$s/$variant" ]; then
    mkdir -p "$OUT_ROOT/$s/previous"
    mv "$OUT_ROOT/$s/$variant" "$OUT_ROOT/$s/previous/${variant}_$stamp"
    echo "moved $s/$variant -> $s/previous/${variant}_$stamp"
  fi
done
export JOBS
exec bash "$HERE/run_loop.sh" "$@"
