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
#              vanilla  no removal, no FlashSplat, no masks (TA_BLUE_MOTOR, PCV): plain ArtiFixer3D+
#                       on a novel camera path; see "Vanilla" below. run_vanilla.sh runs both scenes.
#              i360     Inpaint360GS 3DGUT port (inpaint360gs/): FlashSplat removal, virtual views, LaMa
#              af360    AuraFusion360 3DGUT port (aurafusion/): SAM2 unseen masks, Marigold AGDD,
#                       LeftRefill SDEdit. Both need setup_baselines.sh once; see "Baselines" below.
#                       run_baselines.sh runs both on all three scenes.
#
# Part 1 (default): prep, flashsplat, derive, split, infer. Ends with a single-rollout ArtiFixer
# video; look at it and note frames where the hole is filled cleanly. Then part 2:
#   SEEDS=0-6 bash scripts/bakerh/run_removal.sh SCENE VARIANT      -> propagate, af3d, af3dplus
#   SEEDS=auto   seeds = the first 7 frames of each frame window (also runs part 1 if missing)
# PHASES=split,infer reruns only those phases. FORCE=1 redoes prep and flashsplat even when done.
#
# Vanilla (phases vprep, vinfer, vaf3d, vaf3dplus): make_trajectory.py builds a smooth path, one
# camera per photo, weaving TRAJ_SIDE (2) median photo spacings to either side of the photo path
# (across the direction of travel) and TRAJ_UP (0.5) spacings above it. vprep renders it with the
# existing 3DGUT model into a prepared root (every photo is a real anchor, every path frame a
# target; caption and metric scale reused from the scene), vinfer fixes the path renders with the
# photos as references, vaf3d distills photos + fixed frames into a new 3DGUT, vaf3dplus runs
# ArtiFixer again on its path renders.
# FORCE=1 redoes vprep and vinfer even when done.
#
# Baselines (i360, af360): both remove the same FlashSplat object as the ArtiFixer runs (labels,
# contribution and hit count from FS_SRC, else from the flashsplat phase). With FRAMES they run on
# those views only (subset_colmap.py; the Compressor defaults to COMPRESSOR_FRAMES, else
# 49-148,284-304, the ArtiFixer frames). af360's reference view is the one with the largest unseen
# mask (REF_INDEX overrides). Models come from the local folders in MODELS (no Hugging Face); the
# af360 2D stages run in .venv_af2d. Final renders of every view: <variant>/final/rgb.
#
# Env knobs: GPU (1), NUM_VIEWS (6), DILATE, HOLE_SHAPE, FRAMES, COMPRESSOR_FRAMES, MASK_DIR, SCENE_CAPTION,
# EMPTY_CAPTION, METRIC_SCALE, OUTSIDE (photo|render), AF3D_STEPS (30000), PY, SCENE_ROOT, FS_SRC, OUT_ROOT,
# MODELS, REF_INDEX.
# Outputs: output/bakerh_removal/<scene>/<variant>/. Logs: .../logs/<timestamp>/NN_<phase>.log and
# .../logs/summary.log (start, end, duration and peak GPU memory of every phase, across runs).
set -euo pipefail
SCENE=${1:?usage: run_removal.sh SCENE VARIANT   (SCENE: TA_BLUE_MOTOR|PCV|Compressor, VARIANT: normal|bighull|vanilla|i360|af360)}
VARIANT=${2:?usage: run_removal.sh SCENE VARIANT   (VARIANT: normal|bighull|vanilla|i360|af360)}
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
    FS_SRC=${FS_SRC:-$SR/flashsplat_out_per_view_norm_gt0p1_hull_trim995}; FRAMES=${FRAMES-${COMPRESSOR_FRAMES-49-148,284-304}}
    SCENE_CAPTION=${SCENE_CAPTION:-} ;;
  *) echo "unknown SCENE '$S' (TA_BLUE_MOTOR, PCV, Compressor)" >&2; exit 2 ;;
esac
case $VARIANT in
  normal)  HOLE_SHAPE=${HOLE_SHAPE:-mask}; DILATE=${DILATE:-12} ;;
  bighull) HOLE_SHAPE=${HOLE_SHAPE:-hull}; DILATE=${DILATE:-80} ;;
  vanilla) HOLE_SHAPE=none; DILATE=0
    # One path frame per photo: all 923 Compressor photos would not fit through ArtiFixer.
    [ "$S" != Compressor ] || { echo "vanilla is for TA_BLUE_MOTOR and PCV only" >&2; exit 2; } ;;
  i360|af360) HOLE_SHAPE=none; DILATE=0; export HF_HUB_OFFLINE=1 ;;
  *) echo "unknown VARIANT '$VARIANT' (normal, bighull, vanilla, i360, af360)" >&2; exit 2 ;;
esac
TRAJ_SIDE=${TRAJ_SIDE:-2}
TRAJ_UP=${TRAJ_UP:-0.5}
# Baselines: local model folders (no Hugging Face on the VM) and the AuraFusion360 2D-model venv.
MODELS=${MODELS:-/workspace/amazzucchelli/fbk-3dworld/models}
LAMA=${LAMA:-$MODELS/LaMa/big-lama.pt}
MARIGOLD=${MARIGOLD:-$MODELS/marigold-depth-v1-0}
SD2_CKPT=${SD2_CKPT:-$MODELS/stable-diffusion-2-inpainting/512-inpainting-ema.ckpt}
AF360_REPO=${AF360_REPO:-/workspace/amazzucchelli/fbk-3dworld/AuraFusion360_official}
AF_PY=${AF_PY:-$AF/.venv_af2d/bin/python}
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
V=$O/$S                         # vanilla: prepared root (its name is the scene id)
# Baselines: FlashSplat dir with labels, contribution and hit count; the COLMAP views they run on.
if [ -f "$FS_SRC/hit_count.pt" ]; then FSP=$FS_SRC; else FSP=$FS; fi
if [ -n "${FRAMES:-}" ]; then PCOLMAP=$O/colmap; else PCOLMAP=$COLMAP; fi

pred_dir() {  # newest single-rollout prediction dir under $1 (empty if none)
  [ -d "$1" ] || return 0
  find "$1" -type d -path '*/frames/batch_0000/pred' -printf '%T@ %p\n' | sort -nr | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print }'
}

if [ -z "${PHASES:-}" ] && [ "$VARIANT" = vanilla ]; then
  PHASES=vprep,vinfer,vaf3d,vaf3dplus
elif [ -z "${PHASES:-}" ] && [ "$VARIANT" = i360 ]; then
  PHASES=flashsplat,views,i360remove,i360virtual,i360nbs,i360lama,i360init,i360finetune
elif [ -z "${PHASES:-}" ] && [ "$VARIANT" = af360 ]; then
  PHASES=flashsplat,views,afrender,afcontour,afsam2,afagdd,afinit,afsdedit,affinetune
elif [ -z "${PHASES:-}" ]; then
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
needed=("$PY" "$CKPT" "$CHECKPOINT_PT" "$MODEL_ID" "$COLMAP")
[ "$VARIANT" = vanilla ] || needed+=("${MASK_DIR:-<MASK_DIR: no masks_npy under $SAM_MASKS/$S>}")
case $VARIANT in
  i360)  needed+=("$LAMA") ;;
  af360) needed+=("$LAMA" "$MARIGOLD" "$SD2_CKPT" "$AF360_REPO/utils/LeftRefill" "$AF_PY") ;;
esac
for f in "${needed[@]}"; do
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
  # Size the photos like the FlashSplat renders (half size for the Compressor, trained downsampled).
  "$PY" scripts/bakerh/derive_scene.py --scene_root "$SR" --output_dir "$O/scene" --frames "$FRAMES" \
      --render_like "$FS/background_renders/00000.png" ${prompt_args[@]+"${prompt_args[@]}"}
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

phase_vprep() {
  # Prepared root $V with the photos as anchors and the novel path as targets, rendered from the
  # existing 3DGUT checkpoint (no reconstruction). Caption and metric scale come from the scene.
  if [ -z "${FORCE:-}" ] && [ -f "$V/split.json" ]; then
    echo "found $V/split.json"
    return
  fi
  local prep=("$PY" -m data_processing.prepare_colmap_artifixer_inputs --colmap_dir "$COLMAP" --output_root "$V"
              --reconstruction_checkpoint "$CKPT" --reconstruction_steps 30000 ${FORCE:+--replace})
  "${prep[@]}" --phases prepare
  "$PY" scripts/bakerh/make_trajectory.py --transforms "$V/3dgrut_input/$S/nerfstudio/transforms.json" \
      --output "$O/trajectory_input.json" --side "$TRAJ_SIDE" --up "$TRAJ_UP"

  local caption=$V/captions/$S/caption.h5 src=""
  if [ -f "$SR/split.json" ]; then
    src=$("$PY" -c 'import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
entry = next(iter(json.loads((root / "split.json").read_text())["test"].values()))
print((root / entry["prompt_path"]).resolve())' "$SR")
  fi
  mkdir -p "$(dirname "$caption")"
  if [ -n "$src" ] && [ -f "$src" ]; then
    echo "caption from $src"
    cp -L "$src" "$caption"
  else
    echo "caption: $SCENE_CAPTION"
    "$PY" output/jobs/write_caption.py --caption "$SCENE_CAPTION" --output_path "$caption" --text_encoder_model_id "$MODEL_ID"
  fi

  local scale
  scale=$(sed -n 's/^Scale factor: *\([^ ]*\).*/\1/p' "$SR/metric_alignment/scale_info.txt" 2>/dev/null | head -1)
  echo "metric scale: ${scale:-<none in $SR/metric_alignment: estimating with MoGe>}"
  "${prep[@]}" --phases render,scale --trajectory_path "$O/trajectory_input.json" ${scale:+--metric_scale "$scale"}
  [ -f "$V/split.json" ] || { echo "prepare wrote no $V/split.json (trajectory renders incomplete?)"; return 1; }
}

phase_vinfer() {
  # ArtiFixer on the path renders, photos as reference views (upstream default neighbour selection).
  if [ -z "${FORCE:-}" ] && [ -n "$(pred_dir "$O/artifixer")" ]; then
    echo "found $(pred_dir "$O/artifixer")"
    return
  fi
  "$PY" -m model_eval.run_inference --evalset reconstructed_colmap \
      --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
      --save_dir "$O/artifixer" --split_path "$V/split.json" --render_trajectory trajectory \
      --num_views "$NUM_VIEWS" "${MEM_ARGS[@]}" --replace_if_exists
}

phase_vaf3d() {
  local pred
  pred=$(pred_dir "$O/artifixer")
  [ -n "$pred" ] || { echo "no ArtiFixer output under $O/artifixer: run vinfer first"; return 1; }
  echo "ArtiFixer frames: $pred"
  "$PY" -m data_processing.run_artifixer3d --scene_root "$V" --artifixer_frames_dir "$pred" \
      --artifixer3d_steps "$AF3D_STEPS"
}

phase_vaf3dplus() {
  "$PY" -m model_eval.run_inference --evalset reconstructed_colmap \
      --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
      --save_dir "$O/af3d_plus" --split_path "$V/split_artifixer3d_plus.json" --render_trajectory trajectory \
      --num_views "$NUM_VIEWS" "${MEM_ARGS[@]}" --replace_if_exists
}

phase_views() {
  # Baselines on a subset of the views (FRAMES): a COLMAP copy that holds only those photos.
  if [ -z "${FRAMES:-}" ]; then
    echo "all views of $COLMAP"
    return
  fi
  "$PY" scripts/bakerh/subset_colmap.py --scene_root "$SR" --colmap_dir "$COLMAP" --output_dir "$PCOLMAP" \
      --frames "$FRAMES"
}

# Inpaint360GS 3DGUT port (inpaint360gs/README.md); the FlashSplat removal replaces its SAM2 +
# association + distillation stages.
phase_i360remove() {
  "$PY" -m inpaint360gs.remove --checkpoint "$CKPT" --colmap_dir "$PCOLMAP" --output_dir "$O" --flashsplat_dir "$FSP"
}
phase_i360virtual() {
  "$PY" -m inpaint360gs.virtual_views --checkpoint "$CKPT" --colmap_dir "$PCOLMAP" --output_dir "$O"
}
phase_i360nbs() {
  "$PY" -m inpaint360gs.nbs_masks --output_dir "$O" --mode footprint
}
phase_i360lama() {
  "$PY" -m inpaint360gs.lama --output_dir "$O" --lama_model "$LAMA"
}
phase_i360init() {
  "$PY" -m inpaint360gs.init_gaussians --output_dir "$O" --colmap_dir "$PCOLMAP"
}
phase_i360finetune() {
  "$PY" -m inpaint360gs.finetune --output_dir "$O" --checkpoint "$CKPT" --colmap_dir "$PCOLMAP"
}

# AuraFusion360 3DGUT port (aurafusion/README.md); SAM2, AGDD and SDEdit run in .venv_af2d.
af_ref() {  # REF_INDEX, else the view with the largest unseen mask (it sees most of what must be filled)
  if [ -n "${REF_INDEX:-}" ]; then
    echo "$REF_INDEX"
    return
  fi
  if [ ! -s "$O/reference_index.txt" ]; then
    "$PY" -c 'import sys
from pathlib import Path
import numpy as np
from PIL import Image
paths = sorted(Path(sys.argv[1]).glob("*.png"))
area = [np.count_nonzero(np.asarray(Image.open(p)) > 127) for p in paths]
print(int(paths[int(np.argmax(area))].stem))' "$O/unseen_dilated" > "$O/reference_index.txt"
  fi
  cat "$O/reference_index.txt"
}
phase_afrender() {
  "$PY" -m aurafusion.render_views --checkpoint "$CKPT" --colmap_dir "$PCOLMAP" --flashsplat_dir "$FSP" --output_dir "$O"
}
phase_afcontour() {
  "$PY" -m aurafusion.unseen_contour --render_dir "$O"
}
phase_afsam2() {
  rm -f "$O/reference_index.txt"
  "$AF_PY" -m aurafusion.sam2_unseen --render_dir "$O"
}
phase_afagdd() {
  local ref
  ref=$(af_ref)
  echo "reference view: $ref"
  "$AF_PY" -m aurafusion.agdd --render_dir "$O" --colmap_dir "$PCOLMAP" --reference_index "$ref" \
      --lama_model "$LAMA" --marigold "$MARIGOLD"
}
phase_afinit() {
  local ref
  ref=$(af_ref)
  "$PY" -m aurafusion.init_gaussians --render_dir "$O" --colmap_dir "$PCOLMAP" --reference_index "$ref"
}
phase_afsdedit() {
  local ref
  ref=$(af_ref)
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$AF_PY" -m aurafusion.sdedit --render_dir "$O" --reference_index "$ref" \
      --official_repo "$AF360_REPO" --sd2_inpainting_ckpt "$SD2_CKPT"
}
phase_affinetune() {
  "$PY" -m aurafusion.finetune --render_dir "$O" --colmap_dir "$PCOLMAP"
}

{
  echo "scene=$S variant=$VARIANT phases=$PHASES seeds=${SEEDS:-} gpu=$GPU"
  echo "scene_root=$SR"; echo "flashsplat_src=$FS_SRC"; echo "flashsplat_out=$FS"; echo "masks=${MASK_DIR:-}"
  echo "vanilla path: side=$TRAJ_SIDE up=$TRAJ_UP spacings (used by vprep only)"
  echo "baselines: flashsplat=$FSP views=$PCOLMAP (frames ${FRAMES:-all}) models=$MODELS af360_repo=$AF360_REPO af_py=$AF_PY"
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
if [[ ",$PHASES," == *,vaf3dplus,* ]]; then
  echo
  echo "3DGUT path renders:   $V/recon_results/$S/reconstruction/$S/ours_30000/trajectory/renders"
  echo "ArtiFixer output:"; find "$O/artifixer" -name '*.mp4' | head -5
  echo "ArtiFixer3D renders:  $(find "$V/artifixer3d" -type d -name renders 2>/dev/null | head -1)"
  echo "ArtiFixer3D+ output:"; find "$O/af3d_plus" -name '*.mp4' | head -5
fi
if [[ ",$PHASES," == *,i360finetune,* || ",$PHASES," == *,affinetune,* ]]; then
  echo
  echo "Final renders ($VARIANT, every view): $O/final/rgb"
  [ -d "$O/final/virtual_rgb" ] && echo "Virtual views: $O/final/virtual_rgb"
fi
exit 0
