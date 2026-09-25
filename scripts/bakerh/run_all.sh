#!/usr/bin/env bash
# Baker Hughes object removal + ArtiFixer3D+, all five jobs end to end on GPU 1, one after another:
#   TA_BLUE_MOTOR normal, PCV normal, TA_BLUE_MOTOR bighull, PCV bighull, Compressor bighull
# (run_removal.sh does each job; its header documents the phases and knobs.)
#
# Usage, from the repo root:
#   bash scripts/bakerh/run_all.sh
#
# Everything is logged under output/bakerh_removal/logs/<timestamp>/:
#   all.log     full console output of every job and phase
#   REPORT.txt  environment, inputs found/missing, status + duration of each job, per-phase
#               durations and peak GPU memory, and the last 80 lines of every failed phase.
#               Updated after each job. If anything fails, send this file back.
# Per-phase logs: output/bakerh_removal/<scene>/<variant>/logs/<timestamp>/NN_<phase>.log
#
# Propagation seeds default to the first 7 frames of each frame window (SEEDS=auto); after
# looking at the single-rollout video, rerun one job with better seeds, e.g.
#   SEEDS=10-16 bash scripts/bakerh/run_removal.sh PCV bighull
# A failed job is skipped and the next one runs. Rerunning run_all.sh redoes only what is
# missing (a job whose single-rollout output exists restarts from propagation).
# JOBS="PCV:bighull Compressor:bighull" runs a subset. GPU (1), NUM_VIEWS (6), ... pass through.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AF=$(cd "$HERE/../.." && pwd)
cd "$AF"
export GPU=${GPU:-1}
export SEEDS=${SEEDS:-auto}
export OUT_ROOT=${OUT_ROOT:-$AF/output/bakerh_removal}
PY=${PY:-$AF/.venv/bin/python}
JOBS=${JOBS:-"TA_BLUE_MOTOR:normal PCV:normal TA_BLUE_MOTOR:bighull PCV:bighull Compressor:bighull"}

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$OUT_ROOT/logs/$STAMP
mkdir -p "$RUN_DIR"
ln -sfn "$STAMP" "$OUT_ROOT/logs/latest"
exec > >(tee -a "$RUN_DIR/all.log") 2>&1
REPORT=$RUN_DIR/REPORT.txt
log() { echo "[$(date '+%F %T')] $*"; }
section() { printf '\n==================== %s ====================\n' "$*" >> "$REPORT"; }
check() { if [ -e "$1" ]; then echo "  ok       $1"; else echo "  MISSING  $1"; fi >> "$REPORT"; }

log "run_all $STAMP   jobs: $JOBS   seeds: $SEEDS   gpu: $GPU"
log "report: $REPORT"
{
  echo "bakerh removal report   started $(date -Is)   run $STAMP"
  echo "jobs: $JOBS"
  echo "seeds=$SEEDS gpu=$GPU num_views=${NUM_VIEWS:-6} repo=$AF"
} > "$REPORT"

section "git"
{
  git log --oneline -3
  git status --short | head -40
} >> "$REPORT" 2>&1

section "GPU"
nvidia-smi >> "$REPORT" 2>&1 || echo "nvidia-smi failed" >> "$REPORT"

section "python"
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$AF:$AF/thirdparty/3DGRUT-ArtiFixer" timeout 600 "$PY" - >> "$REPORT" 2>&1 <<'EOF'
import importlib.util, sys
from importlib import metadata
print("python", sys.version.split()[0], sys.executable)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print("device", torch.cuda.get_device_name(0), f"free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
for name in ("diffusers", "transformers", "threedgrut", "moge", "h5py", "scipy"):
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = "?"
    print(f"{name:13s} {'ok ' + version if importlib.util.find_spec(name) else 'MISSING'}")
EOF

section "inputs"
MASKS=/workspace/amazzucchelli/fbk-3dworld/FlashSplat/sam3_single_object_anchor_tracking
for p in "$PY" "$AF/checkpoints/ArtiFixer/artifixer-1.3b.pt" "$AF/checkpoints/Wan2.1-T2V-1.3B-Diffusers/text_encoder" \
         "$AF/output/bakerh_undis/TA_BLUE_MOTOR/3dgrut_runs/TA_BLUE_MOTOR/TA_BLUE_MOTOR/ours_30000/ckpt_30000.pt" \
         "$AF/output/bakerh_undis/PCV/3dgrut_runs/PCV/PCV/ours_30000/ckpt_30000.pt" \
         "$AF/output/bakerh_undis_ds2_from1600/Compressor/3dgrut_runs/Compressor/Compressor/ours_30000/ckpt_30000.pt" \
         "$AF/output/bakerh_undis_ds2_from1600/Compressor/split.json" \
         "$AF/output/bakerh_undis_ds2_from1600/Compressor/flashsplat_out_per_view_norm_gt0p1_hull_trim995/hit_count.pt"; do
  check "$p"
done
for s in TA_BLUE_MOTOR PCV Compressor; do
  for sub in full 0_400; do
    d=$MASKS/$s/$sub/masks_npy
    if [ -d "$d" ]; then echo "  ok       $d ($(find "$d" -name '*.npy' | wc -l) masks)"; else echo "  absent   $d"; fi >> "$REPORT"
  done
done

results=()
for job in $JOBS; do
  scene=${job%%:*} variant=${job##*:}
  start=$(date +%s)
  log "job $job start"
  bash "$HERE/run_removal.sh" "$scene" "$variant"
  code=$?
  took=$(($(date +%s) - start))
  if [ "$code" -eq 0 ]; then status=OK; else status="FAILED(exit $code)"; fi
  results+=("$(printf '%-24s %-18s %dh%02dm' "$job" "$status" $((took / 3600)) $((took % 3600 / 60)))")
  log "job $job $status after $((took / 60)) min"

  job_logs=$OUT_ROOT/$scene/$variant/logs/latest
  section "job $job: $status ($((took / 60)) min)"
  cat "$job_logs/summary.log" >> "$REPORT" 2>/dev/null
  if [ "$code" -ne 0 ] && [ -f "$job_logs/FAILED" ]; then
    failed_log=$(sed -n 's/.*log=//p' "$job_logs/FAILED")
    echo "--- last 80 lines of $failed_log" >> "$REPORT"
    tail -80 "$failed_log" >> "$REPORT" 2>&1
  elif [ "$code" -ne 0 ]; then
    echo "--- no phase log; last 60 lines of all.log" >> "$REPORT"
    tail -60 "$RUN_DIR/all.log" >> "$REPORT"
  fi
done

section "summary   finished $(date -Is)"
printf '%s\n' "${results[@]}" >> "$REPORT"
log "run_all done"
printf '%s\n' "${results[@]}"
log "report: $REPORT"
