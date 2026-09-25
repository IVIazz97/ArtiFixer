#!/usr/bin/env bash
# Object removal + ArtiFixer3D+ on one Baker Hughes scene, on one GPU, with a log per phase.
#
# Usage, from the repo root:
#   bash scripts/bakerh/run_removal.sh SCENE VARIANT
#     SCENE    TA_BLUE_MOTOR | PCV | Compressor
#     VARIANT  normal   hole = SAM mask + projected object, dilated 12 px; scene caption
#              bighull  hole = 2D convex hull of that, dilated 80 px; "empty floor" caption
#   Both variants give ArtiFixer the object-free background renders as reference views, never the
#   photos, so nothing it is conditioned on shows the object.
#
# Part 1 (default): prep, flashsplat, derive, split, infer. Ends with a single-rollout ArtiFixer
# video; look at it and note frames where the hole is filled cleanly. Then part 2:
#   SEEDS=0-6 bash scripts/bakerh/run_removal.sh SCENE VARIANT      -> propagate, af3d, af3dplus
#   SEEDS=auto   seeds = the first 7 frames of each frame window (also runs part 1 if missing)
# PHASES=split,infer reruns only those phases. FORCE=1 redoes prep and flashsplat even when done.
#
# Env knobs: GPU (1), NUM_VIEWS (6), DILATE, HOLE_SHAPE, FRAMES, MASK_DIR, SCENE_CAPTION,
# EMPTY_CAPTION, METRIC_SCALE, OUTSIDE (photo|render), AF3D_STEPS (30000), PY, SCENE_ROOT, FS_SRC, OUT_ROOT.
# Outputs: output/bakerh_removal/<scene>/<variant>/. Logs: .../logs/<timestamp>/NN_<phase>.log and
# .../logs/summary.log (start, end, duration and peak GPU memory of every phase, across runs).
set -euo pipefail
SCENE=${1:?usage: run_removal.sh SCENE VARIANT   (SCENE: TA_BLUE_MOTOR|PCV|Compressor, VARIANT: normal|bighull)}
VARIANT=${2:?usage: run_removal.sh SCENE VARIANT   (VARIANT: normal|bighull)}
AF=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$AF"

# One GPU; expandable segments and one-at-a-time reference encoding avoided the earlier OOMs.
GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$AF:$AF/thirdparty/3DGRUT-ArtiFixer:${PYTHONPATH:-}"
if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda-12.8 ]; then
  export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
fi
# Activate the repo venv: its bin/ must be on PATH (ninja for the 3DGUT CUDA JIT build).
VENV=${VENV:-$AF/.venv}
if [ -f "$VENV/bin/activate" ]; then
  set +u; source "$VENV/bin/activate"; set -u
fi
PY=${PY:-$VENV/bin/python}
# torch's JIT build shells out to `ninja`; if the venv lacks it, install the ninja wheel into it.
if ! command -v ninja >/dev/null 2>&1; then
  echo "ninja not on PATH: installing the ninja wheel with $PY" >&2
  "$PY" -m pip install -q ninja || { command -v uv >/dev/null 2>&1 && uv pip install --python "$PY" ninja; } || true
  NINJA_BIN=$("$PY" -c 'import ninja; print(ninja.BIN_DIR)' 2>/dev/null || true)
  if [ -n "$NINJA_BIN" ]; then export PATH="$NINJA_BIN:$PATH"; fi
  command -v ninja >/dev/null 2>&1 || { echo "ninja still missing: run '$PY -m pip install ninja', then rerun" >&2; exit 3; }
fi
MODEL_ID=${MODEL_ID:-$AF/checkpoints/Wan2.1-T2V-1.3B-Diffusers}
CHECKPOINT_PT=${CHECKPOINT_PT:-$AF/checkpoints/ArtiFixer/artifixer-1.3b.pt}
NUM_VIEWS=${NUM_VIEWS:-6}
MEM_ARGS=(--max_neighbors_per_encode 1)
AF3D_STEPS=${AF3D_STEPS:-30000}
OUTSIDE=${OUTSIDE:-photo}
SAM_MASKS=${SAM_MASKS:-/workspace/amazzucchelli/fbk-3dworld/FlashSplat/sam3_single_object_anchor_tracking}
EMPTY_CAPTION=${EMPTY_CAPTION:-"An empty floor without holes: one continuous, clean floor surface with nothing standing on it, in an industrial hall."}

S=$SCENE
case $S in
  TA_BLUE_MOTOR)
    SR=${SCENE_ROOT:-$AF/output/bakerh_undis/$S}; FS_SRC=${FS_SRC:-$SR/flashsplat_out}; FRAMES=${FRAMES-}
    SCENE_CAPTION=${SCENE_CAPTION:-"An industrial hall with a large blue electric motor standing on the floor, surrounded by equipment."} ;;
  PCV)
    SR=${SCENE_ROOT:-$AF/output/bakerh_undis/$S}; FS_SRC=${FS_SRC:-$SR/flashsplat_out}; FRAMES=${FRAMES-}
    SCENE_CAPTION=${SCENE_CAPTION:-"An industrial hall with a large pressure control valve assembly, pipes and flanges standing on the floor."} ;;
  Compressor)
    # The ds2 scene already has split.json, caption, metric scale and the tuned FlashSplat labels.
    # Only frames 49-148 and 284-304 go through ArtiFixer: all 923 at once ran out of memory.
    SR=${SCENE_ROOT:-$AF/output/bakerh_undis_ds2_from1600/$S}
    FS_SRC=${FS_SRC:-$SR/flashsplat_out_per_view_norm_gt0p1_hull_trim995}; FRAMES=${FRAMES-49-148,284-304}
    SCENE_CAPTION=${SCENE_CAPTION:-} ;;
  *) echo "unknown SCENE '$S' (TA_BLUE_MOTOR, PCV, Compressor)" >&2; exit 2 ;;
esac
case $VARIANT in
  normal)  HOLE_SHAPE=${HOLE_SHAPE:-mask}; DILATE=${DILATE:-12} ;;
  bighull) HOLE_SHAPE=${HOLE_SHAPE:-hull}; DILATE=${DILATE:-80} ;;
  *) echo "unknown VARIANT '$VARIANT' (normal, bighull)" >&2; exit 2 ;;
esac
if [ -z "${MASK_DIR:-}" ]; then
  for d in "$SAM_MASKS/$S/full/masks_npy" "$SAM_MASKS/$S/0_400/masks_npy"; do
    if [ -d "$d" ]; then MASK_DIR=$d; break; fi
  done
fi
CKPT=$SR/3dgrut_runs/$S/$S/ours_30000/ckpt_30000.pt
COLMAP=$SR/3dgrut_input/$S
OUT_ROOT=${OUT_ROOT:-$AF/output/bakerh_removal}
FS=$OUT_ROOT/$S/flashsplat      # shared by both variants
O=$OUT_ROOT/$S/$VARIANT

pred_dir() {  # newest single-rollout prediction dir under $1 (empty if none)
  [ -d "$1" ] || return 0
  find "$1" -type d -path '*/frames/batch_0000/pred' -printf '%T@ %p\n' | sort -nr | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print }'
}

if [ -z "${PHASES:-}" ]; then
  PHASES=prep,flashsplat,derive,split,infer
  if [ -n "${SEEDS:-}" ]; then
    if [ -n "$(pred_dir "$O/artifixer")" ]; then PHASES=propagate,af3d,af3dplus
    else PHASES=$PHASES,propagate,af3d,af3dplus; fi
  fi
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$O/logs/$STAMP
mkdir -p "$LOG_DIR"
ln -sfn "$STAMP" "$O/logs/latest"
SUMMARY=$O/logs/summary.log
N=0
# One line to the screen, this run's summary.log and the variant's cumulative summary.log.
note() { echo "[$(date '+%F %T')] $S/$VARIANT $*" | tee -a "$SUMMARY" "$LOG_DIR/summary.log"; }

missing=()
for f in "$PY" "$CKPT" "$CHECKPOINT_PT" "$MODEL_ID" "$COLMAP" "${MASK_DIR:-<MASK_DIR: no masks_npy under $SAM_MASKS/$S>}"; do
  [ -e "$f" ] || missing+=("$f")
done
if [ ${#missing[@]} -gt 0 ]; then
  for f in "${missing[@]}"; do note "missing: $f"; done
  echo "phase=setup log=$LOG_DIR/summary.log" > "$LOG_DIR/FAILED"
  exit 1
fi
hms() { printf '%dh%02dm%02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

# run_phase NAME: runs phase_NAME with its output in NN_NAME.log (and on screen), samples GPU
# memory every 5 s, records OK/FAILED + duration + peak memory in summary.log, stops on failure.
run_phase() {
  local name=$1 log start status peak sampler
  N=$((N + 1))
  log=$LOG_DIR/$(printf %02d "$N")_$name.log
  note "START $name   log: $log"
  start=$(date +%s)
  nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits -lms 5000 > "$log.gpumem" 2>/dev/null &
  sampler=$!
  set +e
  ( set -euo pipefail; "phase_$name" ) 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  set -e
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  peak=$(sort -n "$log.gpumem" 2>/dev/null | tail -1)
  if [ "$status" -eq 0 ]; then
    note "END   $name   OK   $(hms $(($(date +%s) - start)))   peak GPU mem ${peak:-?} MiB"
    return
  fi
  note "END   $name   FAILED (exit $status)   $(hms $(($(date +%s) - start)))   peak GPU mem ${peak:-?} MiB"
  echo "phase=$name log=$log" > "$LOG_DIR/FAILED"
  if grep -q -i -E 'out of memory|OutOfMemoryError' "$log"; then
    note "      cause: CUDA out of memory. Try NUM_VIEWS=3, or fewer FRAMES."
  fi
  echo "---- last 30 lines of $log"
  tail -30 "$log"
  exit "$status"
}

phase_prep() {
  # Scene-level: metric scale + split.json (prepare_colmap_artifixer_inputs, scale phase only;
  # it reuses the existing 3DGUT checkpoint and renders) and, for the normal run, the caption.
  if [ -n "${FORCE:-}" ] || [ ! -f "$SR/split.json" ]; then
    "$PY" -m data_processing.prepare_colmap_artifixer_inputs --colmap_dir "$COLMAP" --output_root "$SR" \
        --phases scale --reconstruction_steps 30000 ${METRIC_SCALE:+--metric_scale "$METRIC_SCALE"} ${FORCE:+--replace}
  else
    echo "found $SR/split.json"
  fi
  [ -f "$SR/split.json" ] || { echo "prepare wrote no $SR/split.json (reconstruction renders incomplete?)"; return 1; }
  [ "$VARIANT" = normal ] || return 0
  local caption
  caption=$("$PY" -c 'import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
entry = next(iter(json.loads((root / "split.json").read_text())["test"].values()))
print((root / entry["prompt_path"]).resolve())' "$SR")
  if [ -f "$caption" ] && [ -z "${FORCE:-}" ]; then
    echo "found caption $caption"
  else
    [ -n "$SCENE_CAPTION" ] || { echo "no caption at $caption: set SCENE_CAPTION"; return 1; }
    echo "caption: $SCENE_CAPTION"
    "$PY" output/jobs/write_caption.py --caption "$SCENE_CAPTION" --output_path "$caption" --text_encoder_model_id "$MODEL_ID"
  fi
}

phase_flashsplat() {
  # Object/background renders + their opacity maps (build_removal_split needs *_opacity and
  # frame_names.json, which the earlier bakerh FlashSplat runs did not write). Labels are reused
  # when FS_SRC has a hit count; otherwise segmentation is rerun with --save-hit-count.
  if [ -z "${FORCE:-}" ] && [ -f "$FS/frame_names.json" ] && [ -d "$FS/background_renders_opacity" ]; then
    echo "found $FS"
    return
  fi
  local labels=$FS_SRC
  if [ -f "$FS_SRC/hit_count.pt" ]; then
    echo "reusing labels, contribution and hit count from $FS_SRC"
  else
    echo "no hit_count.pt in $FS_SRC: segmenting again with masks $MASK_DIR"
    "$PY" -m data_processing.run_flashsplat_segmentation --checkpoint "$CKPT" --colmap_dir "$COLMAP" \
        --mask_dir "$MASK_DIR" --num_objects 1 --output_root "$FS" --outputs labels overlays --save-hit-count \
        --test_split_interval -1 --config_override selected_indices_file=null
    labels=$FS
  fi
  "$PY" -m data_processing.render_flashsplat_extraction --checkpoint "$CKPT" --colmap_dir "$COLMAP" \
      --labels "$labels/labels.pt" --contribution "$labels/contribution.pt" --hit-count "$labels/hit_count.pt" \
      --normalize-by-hit-count --min-total-contribution 0.1 \
      --background-convex-hull --convex-hull-trim-percentile 99.5 \
      --background label0 --object_id 1 --output_root "$FS" --save_opacity \
      --test_split_interval -1 --config_override selected_indices_file=null
}

phase_derive() {
  # Per-variant scene root: frame subset (FRAMES), prompt, photos matched to the render size.
  local prompt_args=()
  if [ "$VARIANT" = bighull ]; then
    echo "caption: $EMPTY_CAPTION"
    "$PY" output/jobs/write_caption.py --caption "$EMPTY_CAPTION" --output_path "$O/prompt/caption.h5" \
        --text_encoder_model_id "$MODEL_ID"
    prompt_args=(--prompt_path "$O/prompt/caption.h5")
  fi
  rm -rf "$O/scene"
  "$PY" scripts/bakerh/derive_scene.py --scene_root "$SR" --output_dir "$O/scene" --frames "$FRAMES" \
      ${prompt_args[@]+"${prompt_args[@]}"}
}

phase_split() {
  rm -rf "$O/input"
  "$PY" output/jobs/build_removal_split.py --scene_root "$O/scene" --flashsplat_dir "$FS" --mask_dir "$MASK_DIR" \
      --output_dir "$O/input" --opacity hole0 --refs render --dilate_px "$DILATE" --hole_shape "$HOLE_SHAPE"
}

phase_infer() {
  "$PY" -m model_eval.run_inference --evalset reconstructed_colmap \
      --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
      --save_dir "$O/artifixer" --split_path "$O/input/split.json" --render_trajectory all_frames \
      --num_views "$NUM_VIEWS" "${MEM_ARGS[@]}" --replace_if_exists
}

phase_propagate() {
  local pred seeds
  pred=$(pred_dir "$O/artifixer")
  [ -n "$pred" ] || { echo "no single-rollout output under $O/artifixer: run part 1 first"; return 1; }
  seeds=${SEEDS:?set SEEDS, e.g. SEEDS=0-6 or SEEDS=auto}
  if [ "$seeds" = auto ]; then
    seeds=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["auto_seeds"])' "$O/scene/frames.json")
  fi
  echo "seed frames: $seeds (from $pred)"
  rm -rf "$O/propagation"
  "$PY" output/jobs/propagate_removal.py --scene_root "$O/scene" --removal_dir "$O/input" \
      --seed_pred_dir "$pred" --seed_frames "${seeds//+/,}" --output_dir "$O/propagation" \
      --outside "$OUTSIDE" --num_views "$NUM_VIEWS" "${MEM_ARGS[@]}" \
      --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID"
}

phase_af3d() {
  rm -rf "$O/af3d"
  "$PY" output/jobs/build_af3d_split.py --scene_root "$O/scene" --propagated_dir "$O/propagation" --output_dir "$O/af3d"
  "$PY" -m data_processing.run_artifixer3d --scene_root "$O/af3d" --split_path "$O/af3d/split.json" \
      --artifixer_frames_dir "$O/propagation/anchors" --output_root "$O/af3d/artifixer3d" \
      --artifixer3d_plus_inference_split_path "$O/af3d/split_artifixer3d_plus.json" --artifixer3d_steps "$AF3D_STEPS"
}

phase_af3dplus() {
  # ArtiFixer3D+: regenerate the non-anchor frames from the distilled 3DGUT renders.
  "$PY" -m model_eval.run_inference --evalset reconstructed_colmap \
      --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
      --save_dir "$O/af3d_plus" --split_path "$O/af3d/split_artifixer3d_plus.json" --render_trajectory val_frames \
      --neighbor_selection_mode covisibility --num_views "$NUM_VIEWS" "${MEM_ARGS[@]}" --replace_if_exists
}

{
  echo "scene=$S variant=$VARIANT phases=$PHASES seeds=${SEEDS:-} gpu=$GPU"
  echo "scene_root=$SR"; echo "flashsplat_src=$FS_SRC"; echo "flashsplat_out=$FS"; echo "masks=$MASK_DIR"
  echo "frames=${FRAMES:-all} hole=$HOLE_SHAPE+${DILATE}px refs=render outside=$OUTSIDE num_views=$NUM_VIEWS"
  echo "scene_caption=$SCENE_CAPTION"; echo "empty_caption=$EMPTY_CAPTION"
  echo "python=$PY venv=${VIRTUAL_ENV:-<not active>}"; echo "model_id=$MODEL_ID"; echo "checkpoint=$CHECKPOINT_PT"
  echo "on PATH: python=$(command -v python) ninja=$(command -v ninja) nvcc=$(command -v nvcc) CUDA_HOME=${CUDA_HOME:-}"
  echo "git=$(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l) changed files"
  nvidia-smi -i "$GPU" 2>&1 | head -20 || true
} > "$LOG_DIR/00_config.log"
note "RUN $STAMP   phases=$PHASES   gpu=$GPU   config: $LOG_DIR/00_config.log"

IFS=, read -ra phase_list <<< "$PHASES"
for p in "${phase_list[@]}"; do
  declare -F "phase_$p" > /dev/null || { echo "unknown phase '$p'" >&2; exit 2; }
done
for p in "${phase_list[@]}"; do
  run_phase "$p"
done
note "DONE  phases=$PHASES"

set +e  # the hints below must never turn a finished run into a failure
if [[ ",$PHASES," == *,infer,* ]] && [ -z "${SEEDS:-}" ]; then
  echo
  echo "Part 1 done. Single-rollout ArtiFixer output:"
  find "$O/artifixer" -name '*.mp4' | head -5
  echo "  frames: $(pred_dir "$O/artifixer")"
  echo "  hole masks: $O/input/hole"
  echo "Pick frames where the hole is filled cleanly (frame windows start at: $("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["window_starts"])' "$O/scene/frames.json")), then:"
  echo "  SEEDS=0-6 bash scripts/bakerh/run_removal.sh $S $VARIANT"
fi
if [[ ",$PHASES," == *,af3dplus,* ]]; then
  echo
  echo "ArtiFixer3D renders:  $(find "$O/af3d/artifixer3d" -type d -name renders 2>/dev/null | head -1)"
  echo "ArtiFixer3D+ output:"
  find "$O/af3d_plus" -name '*.mp4' | head -5
fi
exit 0
