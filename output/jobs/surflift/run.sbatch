#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# SurfaceLift 3DGUT port (surflift/README.md) on a mip360 scene, the same stages as the bakerh
# run_removal.sh variants surflift / surfliftv. Usage:
#   sbatch --job-name=SL_kitchen --export=ALL,SCENE=kitchen,VARIANT=surflift output/jobs/surflift/run.sbatch
# VARIANT surflift (best real view) | surfliftv (a view that holds the whole hole, virtual allowed).
# STEPS (a+b, default all): prep, ledger, select, inpaint, depth, lift, holes, refill, fit.
# LEDGER_ARGS / SELECT_ARGS: extra flags; VARIANT surfliftv* selects with --mode contain.
# SHARE=1 (default): surfliftv reuses surflift's prep + ledger when they exist.
set -euo pipefail
AF=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer
source $AF/output/jobs/common.sh                        # artifixer env, TMPDIR, offline HF
VARIANT=${VARIANT:-surflift}
STEPS=${STEPS:-prep,ledger,select,inpaint,depth,lift,holes,refill,fit}; STEPS=${STEPS//+/,}
G="$RECON/$SCENE"; CKPT="$G/3dgrut_runs/$SCENE/$SCENE/ours_30000/ckpt_30000.pt"; COLMAP="$G/3dgrut_input/$SCENE"
FSD="$OUT/flashsplat/$SCENE"; LABELS="$FSD/labels.pt"
[ "$SCENE" = garden ] && LABELS="$OUT/flashsplat/garden_vase/labels.pt"
O="$OUT/surflift/$SCENE/$VARIANT"; W="$O/sl"
CK=$AF/checkpoints/aurafusion
AF_PY=$ENVS/aurafusion/bin/python
MODE=${MODE:-real}; [[ "$VARIANT" == surfliftv* ]] && MODE=contain
export PYTHONPATH="$AF:$AF/thirdparty/3DGRUT-ArtiFixer:${PYTHONPATH:-}"
COMMON=(--checkpoint "$CKPT" --colmap_dir "$COLMAP" --work "$W")
echo "host=$(hostname) scene=$SCENE variant=$VARIANT mode=$MODE steps=$STEPS job=${SLURM_JOB_ID:-none}"
has() { [[ ",$STEPS," == *",$1,"* ]]; }
mkdir -p "$W"
S0="$OUT/surflift/$SCENE/surflift/sl"
if [ "$VARIANT" != surflift ] && [ "${SHARE:-1}" = 1 ] && [ -f "$S0/ledger/keep.npy" ] && has prep && has ledger; then
  echo "reusing prep + ledger of $S0"
  rm -rf "$W/masks" "$W/ledger"
  cp -r "$S0/masks" "$S0/ledger" "$W/"; cp "$S0/removal.pt" "$S0/cameras.json" "$S0/prepare.json" "$W/"
  STEPS=${STEPS/prep,/}; STEPS=${STEPS/ledger,/}
fi
has prep    && python -m surflift.prepare "${COMMON[@]}" --flashsplat_dir "$FSD" --labels "$LABELS" --mask_dir "$MASKS/$SCENE"
has ledger  && python -m surflift.clean "${COMMON[@]}" ${LEDGER_ARGS:-}
has select  && python -m surflift.select_view "${COMMON[@]}" --mode $MODE ${SELECT_ARGS:-}
has inpaint && python -m surflift.inpaint --work "$W" --lama_model "$AF/checkpoints/lama/big-lama.pt"
has depth   && python -m surflift.depth --work "$W" --moge "$AF/checkpoints/moge-2-vitl-normal/model.pt"
has lift    && python -m surflift.lift "${COMMON[@]}"
has holes   && python -m surflift.holes "${COMMON[@]}"
has refill  && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 $AF_PY -m surflift.leftrefill --work "$W" \
                 --official_repo /leonardo_scratch/fast/IscrC_EditGS/opt/AuraFusion360_official \
                 --sd2_inpainting_ckpt "$CK/sd2-inpainting/512-inpainting-ema.ckpt" ${REFILL_ARGS:-}
has fit     && python -m surflift.fit "${COMMON[@]}" --final "$O/final" --lama_model "$AF/checkpoints/lama/big-lama.pt"
echo SURFLIFT_DONE
