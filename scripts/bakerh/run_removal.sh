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
#              i360sam  Inpaint360GS with its own segmentation (SAM2 automatic masks, association,
#                       identity distillation); the object is picked with MASK_DIR, the SAM masks
#                       FlashSplat uses
#              af360sam AuraFusion360 with its own removal: an object probability distilled into the
#                       Gaussians from the MASK_DIR masks, > 0.6 plus the IQR-filtered hull (official)
#              run_sequence.sh runs vanilla, then every inpainting variant on all three scenes.
#
# Part 1 (default): prep, flashsplat, derive, split, infer. Ends with a single-rollout ArtiFixer
# video; look at it and note frames where the hole is filled cleanly. Then part 2:
#   SEEDS=0-6 bash scripts/bakerh/run_removal.sh SCENE VARIANT      -> propagate, af3d, af3dplus
#   SEEDS=auto   seeds = the first 7 frames of each frame window (also runs part 1 if missing)
# PHASES=split,infer reruns only those phases. FORCE=1 redoes prep and flashsplat even when done.
#
# Vanilla (phases [vobject,] vprep, vinfer, vaf3d, vaf3dplus): make_trajectory.py builds the novel
# path, TRAJ_MODE:
#   object (default)  one camera per photo on the smoothed photo path, orbiting the object's centre of
#                     mass (FlashSplat object Gaussians, weighted by opacity; vobject segments them if
#                     no labels exist yet) by up to TRAJ_ORBIT (0.6) object radii, toward/away by
#                     TRAJ_RADIAL (0.3) and up by TRAJ_RISE (0.3) radii, looking at the object. Fails
#                     the phase when walls leave no room (more than half the frames collapse onto the
#                     photos); TRAJ_ALLOW_FALLBACK=1 accepts that. TRAJ_RENDER_CHECK=1 also renders
#                     every frame and rejects views a near surface blocks.
#   vidsplat          VidSplat-style (SIGGRAPH 2026) short orbits: for seed photos over the capture, a
#                     TRAJ_CLIP (16) frame orbit about the surface point the photo looks at, the
#                     direction and extent (TRAJ_ORBIT_DEG 15,30,45) that reveal the most unseen
#                     surface while no keyframe shows more than TRAJ_S_HIGH (0.4) unseen area, none is
#                     blocked by a surface closer than TRAJ_D0_M (0.5) m, and the last shows at least
#                     TRAJ_S_LOW (0.03) more than the photo. TRAJ_BUDGET frames in all (default one per
#                     photo, at most TRAJ_MAX_FRAMES). Fails the phase when fewer than half the
#                     orbits are valid.
#   travel            the earlier weave: TRAJ_SIDE (2) median photo spacings across the direction of
#                     travel and TRAJ_UP (0.5) up, keeping the photo rotations.
# PREVIEW=1 stops after the path and its 3DGUT renders (phases [vobject,] vprep, vdebug: top view +
# debug/0_debug.mp4), to check a path before ArtiFixer runs (preview_trajectories.sh does all nine).
# VARIANT loop_object / loop_vidsplat / loop_travel runs up to LOOP_ROUNDS (5) rounds of that path type,
# each on the previous round's model (round 1: the 3DGUT reconstruction), distilling photos + every fixed
# frame so far (see "Loop" further down; run_loop.sh runs loop_vidsplat on TA_BLUE_MOTOR, PCV and
# TA_TURBINE). LOOP_STEPS (5000) extra steps per round, LOOP_LR, LOOP_DISTILL=continue|scratch.
# VARIANT vanilla_object / vanilla_vidsplat / vanilla_travel sets TRAJ_MODE and writes to its own
# directory (run_trajectories.sh runs all three on TA_BLUE_MOTOR, PCV and TA_TURBINE). TRAJ_MAX_FRAMES
# caps the path length (object/travel: every k-th frame) for scenes with many photos.
# Every mode checks the 3DGUT Gaussians: no path camera comes closer than TRAJ_CLEARANCE_M (0.3) metres
# to a surface or crosses a wall; object/travel cameras that would are pulled back toward the photo
# path. Top view of walls, object and path: $O/trajectory_input.png, details (rejections, per-orbit
# stats, failure reason): $O/trajectory_input_info.json. vprep renders the path with the existing 3DGUT
# model into a prepared root (every photo is a real anchor, every path frame a target; caption and
# metric scale reused from the scene), vinfer fixes the path renders with the photos as references,
# vaf3d distills photos + fixed frames into a new 3DGUT, vaf3dplus runs ArtiFixer again on its path
# renders. Changing a TRAJ_* knob redoes the path and ArtiFixer (the old outputs move to
# $O/previous/<timestamp>/); FORCE=1 redoes vprep and vinfer regardless.
#
# Baselines (i360, af360): both remove the same FlashSplat object as the ArtiFixer runs (labels,
# contribution and hit count from FS_SRC, else from the flashsplat phase). With FRAMES they run on
# those views only (subset_colmap.py; the Compressor defaults to COMPRESSOR_FRAMES, else
# 49-148,284-304, the ArtiFixer frames). af360's reference view is the one with the largest unseen
# mask (REF_INDEX overrides). Models come from the local folders in MODELS (no Hugging Face); the
# af360 2D stages run in .venv_af2d. Final renders of every view: <variant>/final/rgb.
# A baseline job whose final model exists for the same views is skipped (FORCE=1 redoes it).
#
# Env knobs: GPU (1), NUM_VIEWS (6), TRAJ_* (vanilla path, see above), DILATE, HOLE_SHAPE, FRAMES, COMPRESSOR_FRAMES, MASK_DIR, SCENE_CAPTION,
# EMPTY_CAPTION, METRIC_SCALE, OUTSIDE (photo|render), AF3D_STEPS (30000), PY, SCENE_ROOT, FS_SRC, OUT_ROOT,
# MODELS, REF_INDEX.
# Outputs: output/bakerh_removal/<scene>/<variant>/. Logs: .../logs/<timestamp>/NN_<phase>.log and
# .../logs/summary.log (start, end, duration and peak GPU memory of every phase, across runs).
set -euo pipefail
SCENE=${1:?usage: run_removal.sh SCENE VARIANT   (SCENE: TA_BLUE_MOTOR|PCV|TA_TURBINE|Compressor, VARIANT: normal|bighull|vanilla[_object|_vidsplat|_travel]|i360|af360|i360sam|af360sam)}
VARIANT=${2:?usage: run_removal.sh SCENE VARIANT   (VARIANT: normal|bighull|vanilla[_object|_vidsplat|_travel]|i360|af360|i360sam|af360sam)}
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
  TA_TURBINE)
    SR=${SCENE_ROOT:-$AF/output/bakerh_undis/$S}; FS_SRC=${FS_SRC:-$SR/flashsplat_out}; FRAMES=${FRAMES-}
    SCENE_CAPTION=${SCENE_CAPTION:-"An industrial hall with a large turbine standing on the floor, surrounded by pipes and equipment."} ;;
  Compressor)
    # The ds2 scene already has split.json, caption, metric scale and the tuned FlashSplat labels.
    # Only frames 49-148 and 284-304 go through ArtiFixer: all 923 at once ran out of memory.
    SR=${SCENE_ROOT:-$AF/output/bakerh_undis_ds2_from1600/$S}
    FS_SRC=${FS_SRC:-$SR/flashsplat_out_per_view_norm_gt0p1_hull_trim995}; FRAMES=${FRAMES-${COMPRESSOR_FRAMES-49-148,284-304}}
    SCENE_CAPTION=${SCENE_CAPTION:-} ;;
  *) echo "unknown SCENE '$S' (TA_BLUE_MOTOR, PCV, TA_TURBINE, Compressor)" >&2; exit 2 ;;
esac
case $VARIANT in
  normal)  HOLE_SHAPE=${HOLE_SHAPE:-mask}; DILATE=${DILATE:-12} ;;
  bighull) HOLE_SHAPE=${HOLE_SHAPE:-hull}; DILATE=${DILATE:-80} ;;
  loop_object|loop_vidsplat|loop_travel) HOLE_SHAPE=none; DILATE=0; TRAJ_MODE=${VARIANT#loop_}
    [ "$S" != Compressor ] || { echo "the loop is for TA_BLUE_MOTOR, PCV and TA_TURBINE only" >&2; exit 2; } ;;
  vanilla|vanilla_object|vanilla_vidsplat|vanilla_travel) HOLE_SHAPE=none; DILATE=0
    # vanilla_<mode> fixes TRAJ_MODE and has its own output dir, so the three paths can sit side by side.
    [ "$VARIANT" = vanilla ] || TRAJ_MODE=${VARIANT#vanilla_}
    # One path frame per photo: all 923 Compressor photos would not fit through ArtiFixer.
    [ "$S" != Compressor ] || { echo "vanilla is for TA_BLUE_MOTOR, PCV and TA_TURBINE only" >&2; exit 2; } ;;
  i360|af360|i360sam|af360sam) HOLE_SHAPE=none; DILATE=0; export HF_HUB_OFFLINE=1 ;;
  *) echo "unknown VARIANT '$VARIANT' (normal, bighull, vanilla, vanilla_object, vanilla_vidsplat, vanilla_travel, loop_object, loop_vidsplat, loop_travel, i360, af360, i360sam, af360sam)" >&2; exit 2 ;;
esac
TRAJ_MODE=${TRAJ_MODE:-object}
case $TRAJ_MODE in object|vidsplat|travel) ;; *) echo "unknown TRAJ_MODE '$TRAJ_MODE' (object, vidsplat, travel)" >&2; exit 2 ;; esac
TRAJ_ORBIT=${TRAJ_ORBIT:-0.6}
TRAJ_RADIAL=${TRAJ_RADIAL:-0.3}
TRAJ_RISE=${TRAJ_RISE:-0.3}
TRAJ_SIDE=${TRAJ_SIDE:-2}
TRAJ_UP=${TRAJ_UP:-0.5}
TRAJ_CLEARANCE_M=${TRAJ_CLEARANCE_M:-0.3}
TRAJ_D0_M=${TRAJ_D0_M:-0.5}
TRAJ_RENDER_CHECK=${TRAJ_RENDER_CHECK:-0}
TRAJ_ALLOW_FALLBACK=${TRAJ_ALLOW_FALLBACK:-0}
TRAJ_CLIP=${TRAJ_CLIP:-16}
TRAJ_ORBIT_DEG=${TRAJ_ORBIT_DEG:-15,30,45}
TRAJ_S_LOW=${TRAJ_S_LOW:-0.03}
TRAJ_S_HIGH=${TRAJ_S_HIGH:-0.4}
TRAJ_BUDGET=${TRAJ_BUDGET:-}
TRAJ_MAX_FRAMES=${TRAJ_MAX_FRAMES:-}
LOOP_ROUNDS=${LOOP_ROUNDS:-5}
LOOP_STEPS=${LOOP_STEPS:-5000}
LOOP_LR=${LOOP_LR:-1}
LOOP_DISTILL=${LOOP_DISTILL:-continue}
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

USER_PHASES=${PHASES:-}
if [ -z "${PHASES:-}" ] && [[ $VARIANT == vanilla* ]]; then
  PHASES=vprep,vinfer,vaf3d,vaf3dplus
  [ -z "${PREVIEW:-}" ] || PHASES=vprep,vdebug  # only the path and its 3DGUT renders, no ArtiFixer
  [ "$TRAJ_MODE" != object ] || PHASES=vobject,$PHASES
elif [ -z "${PHASES:-}" ] && [[ $VARIANT == loop_* ]]; then
  PHASES=vloopplus
  for ((k = LOOP_ROUNDS; k >= 1; k--)); do PHASES=vround$k,$PHASES; done
  [ "$TRAJ_MODE" != object ] || PHASES=vobject,$PHASES
elif [ -z "${PHASES:-}" ] && [ "$VARIANT" = i360 ]; then
  PHASES=flashsplat,views,i360remove,i360virtual,i360nbs,i360lama,i360init,i360finetune
elif [ -z "${PHASES:-}" ] && [ "$VARIANT" = af360 ]; then
  PHASES=flashsplat,views,afrender,afcontour,afsam2,afagdd,afinit,afsdedit,affinetune
elif [ -z "${PHASES:-}" ] && [ "$VARIANT" = i360sam ]; then
  PHASES=views,i360masks,i360associate,i360distill,i360remove,i360virtual,i360nbs,i360lama,i360init,i360finetune
elif [ -z "${PHASES:-}" ] && [ "$VARIANT" = af360sam ]; then
  PHASES=views,afsegmasks,afsegdistill,afsegremove,afrender,afcontour,afsam2,afagdd,afinit,afsdedit,affinetune
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

if [[ $VARIANT == i360* || $VARIANT == af360* ]] && [ -z "$USER_PHASES" ] && [ -z "${FORCE:-}" ] \
    && [ -f "$O/final/ckpt_final.pt" ] \
    && [ "$(cat "$O/views.txt" 2>/dev/null || echo "${FRAMES:-all}")" = "${FRAMES:-all}" ]; then
  note "SKIP: already finished on views ${FRAMES:-all} ($O/final; FORCE=1 redoes it)"
  exit 0
fi

missing=()
needed=("$PY" "$CKPT" "$CHECKPOINT_PT" "$MODEL_ID" "$COLMAP")
[[ $VARIANT == vanilla* || $VARIANT == loop_* ]] || needed+=("${MASK_DIR:-<MASK_DIR: no masks_npy under $SAM_MASKS/$S>}")
case $VARIANT in
  i360)           needed+=("$LAMA") ;;
  i360sam)        needed+=("$LAMA" "$AF_PY") ;;
  af360|af360sam) needed+=("$LAMA" "$MARIGOLD" "$SD2_CKPT" "$AF360_REPO/utils/LeftRefill" "$AF_PY") ;;
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
  elif [ -z "${FORCE:-}" ] && [ -f "$FS/hit_count.pt" ]; then
    echo "reusing labels, contribution and hit count from $FS"
    labels=$FS
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

object_labels() {  # the FlashSplat extraction of this scene's object that already exists (empty if none)
  # FS_SRC first, then any other extraction next to the scene, then this pipeline's own; among them
  # one with hit_count.pt (the removal's exact visibility filter) wins over labels.pt alone.
  local d best=""
  for d in "$FS_SRC" "$SR"/flashsplat_out* "$FS"; do
    [ -f "$d/labels.pt" ] || continue
    if [ -f "$d/hit_count.pt" ]; then echo "$d"; return; fi
    [ -n "$best" ] || best=$d
  done
  echo "$best"
}

phase_vobject() {
  # The object the vanilla path is built around: an existing FlashSplat extraction of it, else a new
  # segmentation (labels only, no renders).
  local found
  found=$(object_labels)
  if [ -n "$found" ]; then
    echo "object from the existing FlashSplat extraction $found"
    return
  fi
  [ -n "${MASK_DIR:-}" ] || { echo "no FlashSplat labels and no MASK_DIR (masks_npy under $SAM_MASKS/$S): set MASK_DIR, or TRAJ_MODE=vidsplat|travel"; return 1; }
  "$PY" -m data_processing.run_flashsplat_segmentation --checkpoint "$CKPT" --colmap_dir "$COLMAP" \
      --mask_dir "$MASK_DIR" --num_objects 1 --output_root "$FS" --outputs labels --save-hit-count \
      --test_split_interval -1 --config_override selected_indices_file=null
}

metric_scale() {  # metres per scene unit of the scene, if known (empty, not an error, when it is not)
  [ -f "$SR/metric_alignment/scale_info.txt" ] || return 0
  sed -n 's/^Scale factor: *\([^ ]*\).*/\1/p' "$SR/metric_alignment/scale_info.txt" | head -1
}

ckpt_steps() {  # ckpt_steps .../ckpt_N.pt -> N: renders land in ours_N, so prepare must look there (loop rounds > 1)
  local name
  name=$(basename "$1" .pt)
  echo "${name#ckpt_}"
}

prepare_photos() {  # prepare_photos ROOT CHECKPOINT: prepared scene root (photos, transforms) over CHECKPOINT
  local prep=("$PY" -m data_processing.prepare_colmap_artifixer_inputs --colmap_dir "$COLMAP" --output_root "$1"
              --reconstruction_checkpoint "$2" --reconstruction_steps "$(ckpt_steps "$2")" ${FORCE:+--replace})
  "${prep[@]}" --phases prepare
}

render_path() {  # render_path ROOT CHECKPOINT TRAJECTORY: caption, then photos and path rendered with CHECKPOINT
  local root=$1 ckpt=$2 trajectory=$3 scale caption=$1/captions/$S/caption.h5 src=""
  scale=$(metric_scale)
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
  "$PY" -m data_processing.prepare_colmap_artifixer_inputs --colmap_dir "$COLMAP" --output_root "$root" \
      --reconstruction_checkpoint "$ckpt" --reconstruction_steps "$(ckpt_steps "$ckpt")" ${FORCE:+--replace} \
      --phases render,scale --trajectory_path "$trajectory" ${scale:+--metric_scale "$scale"}
  [ -f "$root/split.json" ] || { echo "prepare wrote no $root/split.json (trajectory renders incomplete?)"; return 1; }
}

build_traj_args() {  # build_traj_args CHECKPOINT: make_trajectory.py arguments for TRAJ_MODE, in TRAJ_ARGS
  local scale labels
  scale=$(metric_scale)
  TRAJ_ARGS=(--checkpoint "$1" --clearance_m "$TRAJ_CLEARANCE_M" --d0_m "$TRAJ_D0_M" ${scale:+--metric_scale "$scale"}
             ${TRAJ_MAX_FRAMES:+--max_frames "$TRAJ_MAX_FRAMES"})
  case $TRAJ_MODE in
    object)
      labels=$(object_labels)
      [ -n "$labels" ] || { echo "no FlashSplat labels for $S (vobject should have made them)"; return 1; }
      TRAJ_ARGS+=(--mode object --flashsplat_dir "$labels" --orbit "$TRAJ_ORBIT" --radial "$TRAJ_RADIAL" --rise "$TRAJ_RISE")
      [ "$TRAJ_RENDER_CHECK" = 0 ] || TRAJ_ARGS+=(--render_check --colmap_dir "$COLMAP")
      [ "$TRAJ_ALLOW_FALLBACK" = 0 ] || TRAJ_ARGS+=(--allow_fallback) ;;
    vidsplat)
      TRAJ_ARGS+=(--mode vidsplat --colmap_dir "$COLMAP" --clip_len "$TRAJ_CLIP" --orbit_deg "$TRAJ_ORBIT_DEG"
                  --s_low "$TRAJ_S_LOW" --s_high "$TRAJ_S_HIGH" ${TRAJ_BUDGET:+--budget "$TRAJ_BUDGET"}) ;;
    travel)
      TRAJ_ARGS+=(--mode travel --side "$TRAJ_SIDE" --up "$TRAJ_UP") ;;
  esac
}

phase_vprep() {
  # Prepared root $V with the photos as anchors and the novel path as targets, rendered from the
  # existing 3DGUT checkpoint (no reconstruction). Caption and metric scale come from the scene.
  local scale
  scale=$(metric_scale)
  build_traj_args "$CKPT" || return 1
  local traj=("${TRAJ_ARGS[@]}")
  # The path is redone when its settings change: a stale split.json would silently keep the old one.
  local settings
  settings=$(printf '%s\n' "${traj[@]}")
  if [ -z "${FORCE:-}" ] && [ -f "$V/split.json" ]; then
    if [ "$(cat "$O/trajectory_input.args" 2>/dev/null)" = "$settings" ]; then
      echo "found $V/split.json (same trajectory settings)"
      return
    fi
    local old
    old=$O/previous/$(date +%Y%m%d_%H%M%S)
    mkdir -p "$old"
    echo "trajectory settings changed: new path; the previous path and ArtiFixer outputs move to $old"
    for f in "$O/artifixer" "$O/af3d_plus" "$O"/trajectory_input*; do
      [ ! -e "$f" ] || mv "$f" "$old/"
    done
  fi
  prepare_photos "$V" "$CKPT"
  echo "metric scale: ${scale:-<none in $SR/metric_alignment: estimating with MoGe; clearance from the object size>}"
  echo "trajectory: $TRAJ_MODE"
  "$PY" scripts/bakerh/make_trajectory.py --transforms "$V/3dgrut_input/$S/nerfstudio/transforms.json" \
      --output "$O/trajectory_input.json" "${traj[@]}" \
    || { echo "make_trajectory.py found no acceptable $TRAJ_MODE path: see $O/trajectory_input_info.json and $O/trajectory_input.png"; return 1; }
  render_path "$V" "$CKPT" "$O/trajectory_input.json"
  echo "$settings" > "$O/trajectory_input.args"
}

debug_videos() {  # best effort: $O/debug/0_debug.mp4 (path top view | render | ArtiFixer | ArtiFixer3D+)
  "$PY" scripts/bakerh/debug_video.py --variant_dir "$O" --scene "$S" \
    || echo "WARNING: debug video failed; the run goes on"
}

phase_vdebug() {
  debug_videos
}

# Loop (loop_<mode>): up to LOOP_ROUNDS rounds, each adding a new path of TRAJ_MODE to the reconstruction.
# Round k: the path is chosen on the round k-1 model (round 0: the 3DGUT reconstruction; vidsplat counts
# every earlier generated view as seen; object/travel shift their weave by (k-1)/LOOP_ROUNDS of a
# period), all paths so far are rendered with that model, ArtiFixer fixes only the new frames, and
# ArtiFixer3D continues that model on photos + every fixed frame so far for LOOP_STEPS more steps, with
# the from-scratch densification windows and position LR decay squeezed into those steps (the LR
# restarts at LOOP_LR x its initial value). LOOP_DISTILL=scratch instead retrains every round from
# scratch for AF3D_STEPS. Each round gets 1/LOOP_ROUNDS of the frame budget. A later round that finds no
# acceptable new path ends the loop.
# vloopplus runs ArtiFixer3D+ over all paths with the last model. Outputs: $O/round_<k>/, $O/final.
loop_round() {
  local k=$1 R=$O/round_$1 prev=$O/round_$(($1 - 1)) base steps first per_round n_photos offset pred i
  local C=$R/$S
  if [ -f "$O/loop_ended" ] && [ "$(cat "$O/loop_ended")" -le "$k" ]; then
    echo "the loop ended at round $(cat "$O/loop_ended") (no acceptable new path): nothing to do"
    return 0
  fi
  if [ -z "${FORCE:-}" ] && [ -f "$R/model.txt" ] && [ -f "$(cat "$R/model.txt")" ]; then
    echo "round $k done: model $(cat "$R/model.txt")"
    return 0
  fi
  if [ "$k" -eq 1 ]; then base=$CKPT; else base=$(cat "$prev/model.txt"); fi
  echo "round $k/$LOOP_ROUNDS: paths chosen and rendered with $base"
  mkdir -p "$R"
  prepare_photos "$C" "$base"
  n_photos=$("$PY" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["frames"]))' \
      "$C/3dgrut_input/$S/nerfstudio/transforms.json")
  per_round=$(( ${TRAJ_MAX_FRAMES:-$n_photos} / LOOP_ROUNDS ))
  [ "$per_round" -ge $((TRAJ_CLIP + 1)) ] || per_round=$((TRAJ_CLIP + 1))
  offset=$("$PY" -c "print(($k - 1) / $LOOP_ROUNDS)")
  build_traj_args "$base" || return 1
  local seen=()
  [ "$k" -eq 1 ] || seen=(--seen_transforms "$prev/trajectory_input.json")
  if ! "$PY" scripts/bakerh/make_trajectory.py --transforms "$C/3dgrut_input/$S/nerfstudio/transforms.json" \
      --output "$R/new_path.json" "${TRAJ_ARGS[@]}" --max_frames "$per_round" --phase_offset "$offset" \
      ${seen[@]+"${seen[@]}"}; then
    if [ "$k" -eq 1 ]; then
      echo "round 1 found no acceptable $TRAJ_MODE path: see $R/new_path_info.json and $R/new_path.png"
      return 1
    fi
    echo "round $k found no acceptable new path (see $R/new_path_info.json): the loop ends with round $((k - 1))"
    echo "$k" > "$O/loop_ended"
    return 0
  fi
  # All paths so far, the new one last: ArtiFixer fixes the new block, ArtiFixer3D distills them all.
  first=$("$PY" - "$R" "$prev" "$k" <<'PYEOF'
import json, shutil, sys
from pathlib import Path
R, prev, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
new = json.loads((R / "new_path.json").read_text())
old = json.loads((prev / "trajectory_input.json").read_text())["frames"] if k > 1 else []
new["frames"] = old + new["frames"]
(R / "trajectory_input.json").write_text(json.dumps(new, indent=2) + "\n")
info = json.loads((R / "new_path_info.json").read_text())
# Pieces of the whole path: every earlier round's, then the new round's after the offset.
before = json.loads((prev / "trajectory_input_info.json").read_text()).get("pieces") if k > 1 else []
before = before or ([[0, len(old) - 1]] if old else [])
info["pieces"] = before + [[a + len(old), b + len(old)] for a, b in info.get("pieces") or [[0, len(new["frames"]) - len(old) - 1]]]
info.update(frame_offset=len(old), loop_round=k)
(R / "trajectory_input_info.json").write_text(json.dumps(info, indent=1) + "\n")
if (R / "new_path.png").exists():
    shutil.copy(R / "new_path.png", R / "trajectory_input.png")
print(len(old))
PYEOF
)
  render_path "$C" "$base" "$R/trajectory_input.json"
  "$PY" - "$C" "$first" <<'PYEOF'
import json, sys
from pathlib import Path
root, first = Path(sys.argv[1]), int(sys.argv[2])
split = json.loads((root / "split.json").read_text())
(scene, entry), = split["test"].items()
targets = json.loads((root / entry["target_indices_path"]).read_text())
new = [t for t in targets if t >= first]
assert new == list(range(first, first + len(new))) and new, f"new frames {first}.. are not the last targets"
(root / "target_indices_round.json").write_text(json.dumps(new) + "\n")
entry["target_indices_path"] = str((root / "target_indices_round.json").resolve())
(root / "split_round.json").write_text(json.dumps(split, indent=2) + "\n")
print(f"ArtiFixer on frames {first}-{first + len(new) - 1} of {len(targets)}")
PYEOF
  "$PY" -m model_eval.run_inference --evalset reconstructed_colmap \
      --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
      --save_dir "$R/artifixer" --split_path "$C/split_round.json" --render_trajectory trajectory \
      --num_views "$NUM_VIEWS" "${MEM_ARGS[@]}" --replace_if_exists
  pred=$(pred_dir "$R/artifixer")
  [ -n "$pred" ] || { echo "no ArtiFixer output under $R/artifixer"; return 1; }
  rm -rf "$R/pred_all"
  mkdir -p "$R/pred_all"
  if [ "$k" -gt 1 ]; then
    for i in "$prev"/pred_all/*.png; do ln -s "$(readlink -f "$i")" "$R/pred_all/$(basename "$i")"; done
  fi
  for i in "$pred"/*.png; do ln -sfn "$(readlink -f "$i")" "$R/pred_all/$(basename "$i")"; done
  local distill=()
  if [ "$LOOP_DISTILL" = scratch ]; then
    steps=$AF3D_STEPS
  else
    steps=$(( $("$PY" -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["global_step"]))' "$base") + LOOP_STEPS ))
    distill=(--base_checkpoint "$base" --rescale_schedule --lr_scale "$LOOP_LR")
  fi
  echo "ArtiFixer3D: photos + $(ls "$R/pred_all" | wc -l) fixed frames, $([ ${#distill[@]} -gt 0 ] && echo "continuing $base to step $steps" || echo "from scratch, $steps steps")"
  "$PY" -m data_processing.run_artifixer3d --scene_root "$C" --artifixer_frames_dir "$R/pred_all" \
      --output_root "$R/artifixer3d" --artifixer3d_steps "$steps" --phases distill ${FORCE:+--replace} \
      ${distill[@]+"${distill[@]}"}
  local model
  model=$(find "$R/artifixer3d" -name "ckpt_${steps}.pt" | head -1)
  [ -n "$model" ] || { echo "ArtiFixer3D wrote no ckpt_${steps}.pt under $R/artifixer3d"; return 1; }
  echo "$model" > "$R/model.txt"
  "$PY" scripts/bakerh/debug_video.py --variant_dir "$R" --scene "$S" \
    || echo "WARNING: debug video failed; the run goes on"
}

phase_vloopplus() {
  # ArtiFixer3D+ over every path of the loop, rendered with the last round's model.
  local k R="" C model steps done_pred
  for k in $(seq "$LOOP_ROUNDS" -1 1); do
    if [ -f "$O/round_$k/model.txt" ]; then R=$O/round_$k; break; fi
  done
  [ -n "$R" ] || { echo "no finished loop round under $O"; return 1; }
  C=$R/$S
  model=$(cat "$R/model.txt")
  steps=$(basename "$model" .pt)
  steps=${steps#ckpt_}
  ln -sfn "$(basename "$R")" "$O/final"
  done_pred=$(pred_dir "$R/af3d_plus")
  if [ -z "${FORCE:-}" ] && [ -n "$done_pred" ] && [ "$done_pred" -nt "$model" ]; then
    echo "found $done_pred (newer than the final model)"
  else
    echo "ArtiFixer3D+ with $model over $(ls "$R/pred_all" | wc -l) path frames"
    "$PY" -m data_processing.run_artifixer3d --scene_root "$C" --output_root "$R/artifixer3d" \
        --artifixer3d_steps "$steps" --phases render,prepare_artifixer3d_plus
    "$PY" -m model_eval.run_inference --evalset reconstructed_colmap \
        --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
        --save_dir "$R/af3d_plus" --split_path "$C/split_artifixer3d_plus.json" --render_trajectory trajectory \
        --num_views "$NUM_VIEWS" "${MEM_ARGS[@]}" --replace_if_exists
  fi
  done_pred=$(pred_dir "$R/af3d_plus")
  "$PY" scripts/bakerh/debug_video.py --variant_dir "$R" --scene "$S" --first_frame 0 --pred_dir "$R/pred_all" \
      --render_dir "$(dirname "$done_pred")/rendered" --out_dir "$O/debug" \
    || echo "WARNING: debug video failed; the run goes on"
}

phase_vinfer() {
  # ArtiFixer on the path renders, photos as reference views (upstream default neighbour selection).
  if [ -z "${FORCE:-}" ] && [ -n "$(pred_dir "$O/artifixer")" ]; then
    echo "found $(pred_dir "$O/artifixer")"
    debug_videos
    return
  fi
  "$PY" -m model_eval.run_inference --evalset reconstructed_colmap \
      --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
      --save_dir "$O/artifixer" --split_path "$V/split.json" --render_trajectory trajectory \
      --num_views "$NUM_VIEWS" "${MEM_ARGS[@]}" --replace_if_exists
  touch "$O/af3d_stale"  # new ArtiFixer frames: vaf3d must distill again, not reuse its checkpoint
  debug_videos
}

phase_vaf3d() {
  local pred
  pred=$(pred_dir "$O/artifixer")
  [ -n "$pred" ] || { echo "no ArtiFixer output under $O/artifixer: run vinfer first"; return 1; }
  echo "ArtiFixer frames: $pred"
  local replace=()
  if [ -n "${FORCE:-}" ] || [ -e "$O/af3d_stale" ]; then replace=(--replace); fi
  "$PY" -m data_processing.run_artifixer3d --scene_root "$V" --artifixer_frames_dir "$pred" \
      --artifixer3d_steps "$AF3D_STEPS" ${replace[@]+"${replace[@]}"}
  rm -f "$O/af3d_stale"
}

phase_vaf3dplus() {
  local done_pred
  done_pred=$(pred_dir "$O/af3d_plus")
  if [ -z "${FORCE:-}" ] && [ -n "$done_pred" ] && [ "$done_pred" -nt "$(pred_dir "$O/artifixer")" ]; then
    echo "found $done_pred (newer than the ArtiFixer frames it builds on)"
    debug_videos
    return
  fi
  "$PY" -m model_eval.run_inference --evalset reconstructed_colmap \
      --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
      --save_dir "$O/af3d_plus" --split_path "$V/split_artifixer3d_plus.json" --render_trajectory trajectory \
      --num_views "$NUM_VIEWS" "${MEM_ARGS[@]}" --replace_if_exists
  debug_videos
}

phase_views() {
  # Baselines on a subset of the views (FRAMES): a COLMAP copy that holds only those photos.
  rm -f "$O/final/ckpt_final.pt"  # the job is being redone; views.txt + this file mark it finished
  echo "${FRAMES:-all}" > "$O/views.txt"
  if [ -z "${FRAMES:-}" ]; then
    echo "all views of $COLMAP"
    return
  fi
  "$PY" scripts/bakerh/subset_colmap.py --scene_root "$SR" --colmap_dir "$COLMAP" --output_dir "$PCOLMAP" \
      --frames "$FRAMES"
}

# Inpaint360GS 3DGUT port (inpaint360gs/README.md). i360: the FlashSplat removal replaces its SAM2 +
# association + distillation stages. i360sam: its own stages, the target picked with MASK_DIR.
phase_i360masks() {
  "$AF_PY" -m inpaint360gs.raw_masks --image_dir "$PCOLMAP/images" --output_dir "$O"
}
phase_i360associate() {
  "$PY" -m inpaint360gs.associate --checkpoint "$CKPT" --colmap_dir "$PCOLMAP" --output_dir "$O"
}
phase_i360distill() {
  "$PY" -m inpaint360gs.distill --checkpoint "$CKPT" --colmap_dir "$PCOLMAP" --output_dir "$O"
}
phase_i360remove() {
  if [ "$VARIANT" = i360sam ]; then
    "$PY" -m inpaint360gs.remove --checkpoint "$CKPT" --colmap_dir "$PCOLMAP" --output_dir "$O" --target_masks_dir "$MASK_DIR"
  else
    "$PY" -m inpaint360gs.remove --checkpoint "$CKPT" --colmap_dir "$PCOLMAP" --output_dir "$O" --flashsplat_dir "$FSP"
  fi
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
# af360sam: the official object-masked-Gaussians removal. The MASK_DIR masks become 2-class id maps,
# distilled into the Gaussians (inpaint360gs.distill); object probability > 0.6 (official
# removal_thresh) plus the IQR-filtered hull (inpaint360gs.remove) is removed.
phase_afsegmasks() {
  "$PY" scripts/bakerh/masks_to_ids.py --image_dir "$PCOLMAP/images" --mask_dir "$MASK_DIR" --output_dir "$O/seg/associated"
}
phase_afsegdistill() {
  "$PY" -m inpaint360gs.distill --checkpoint "$CKPT" --colmap_dir "$PCOLMAP" --output_dir "$O/seg"
}
phase_afsegremove() {
  "$PY" -m inpaint360gs.remove --checkpoint "$CKPT" --colmap_dir "$PCOLMAP" --output_dir "$O/seg" \
      --target_ids 1 --removal_thresh 0.6
}
phase_afrender() {
  local removal=(--flashsplat_dir "$FSP")
  [ "$VARIANT" = af360sam ] && removal=(--removal_pt "$O/seg/removal/removal.pt")
  "$PY" -m aurafusion.render_views --checkpoint "$CKPT" --colmap_dir "$PCOLMAP" "${removal[@]}" --output_dir "$O"
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
  echo "vanilla path: mode=$TRAJ_MODE clearance=${TRAJ_CLEARANCE_M} m d0=${TRAJ_D0_M} m; object: orbit=$TRAJ_ORBIT radial=$TRAJ_RADIAL rise=$TRAJ_RISE radii render_check=$TRAJ_RENDER_CHECK allow_fallback=$TRAJ_ALLOW_FALLBACK; vidsplat: clip=$TRAJ_CLIP orbit_deg=$TRAJ_ORBIT_DEG s_low=$TRAJ_S_LOW s_high=$TRAJ_S_HIGH budget=${TRAJ_BUDGET:-photos}; travel: side=$TRAJ_SIDE up=$TRAJ_UP spacings"
  echo "baselines: flashsplat=$FSP views=$PCOLMAP (frames ${FRAMES:-all}) models=$MODELS af360_repo=$AF360_REPO af_py=$AF_PY"
  echo "frames=${FRAMES:-all} hole=$HOLE_SHAPE+${DILATE}px refs=render outside=$OUTSIDE num_views=$NUM_VIEWS"
  echo "scene_caption=$SCENE_CAPTION"; echo "empty_caption=$EMPTY_CAPTION"
  echo "python=$PY venv=${VIRTUAL_ENV:-<not active>}"; echo "model_id=$MODEL_ID"; echo "checkpoint=$CHECKPOINT_PT"
  echo "on PATH: python=$(command -v python) ninja=$(command -v ninja) nvcc=$(command -v nvcc) CUDA_HOME=${CUDA_HOME:-}"
  echo "git=$(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l) changed files"
  nvidia-smi -i "$GPU" 2>&1 | head -20 || true
} > "$LOG_DIR/00_config.log"
note "RUN $STAMP   phases=$PHASES   gpu=$GPU   config: $LOG_DIR/00_config.log"

for ((k = 1; k <= LOOP_ROUNDS; k++)); do eval "phase_vround$k() { loop_round $k; }"; done
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
if [[ ",$PHASES," == *,vdebug,* ]]; then
  echo
  echo "Path preview ($TRAJ_MODE):"
  echo "  top view:     $O/trajectory_input.png   details: $O/trajectory_input_info.json"
  echo "  3DGUT renders: $V/recon_results/$S/reconstruction/$S/ours_30000/trajectory/renders"
  echo "  video:        $O/debug/0_debug.mp4 (top view with the current camera | render)"
  echo "The full run reuses this path and its renders while the TRAJ_* settings stay the same."
fi
if [[ ",$PHASES," == *,vaf3dplus,* ]]; then
  echo
  echo "Path top view:        $O/trajectory_input.png"
  echo "Debug video:          $O/debug/0_debug.mp4 (top view | render | ArtiFixer | ArtiFixer3D+)"
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
