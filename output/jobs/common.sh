# Shared environment for the mip360 object-removal jobs. Source from an sbatch script.
# sbatch --export=ALL carries the submitting shell's TMPDIR, which may not exist on the
# compute node; pin a job-local one before env.sh derives cache paths from it.
export TMPDIR=/leonardo_scratch/fast/IscrC_EditGS/tmp/job_${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/env.sh artifixer >/dev/null 2>&1
export AF=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer
cd "$AF"
export PYTHONPATH="$AF/thirdparty/3DGRUT-ArtiFixer:$AF:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
export MOGE_MODEL_PATH="$AF/checkpoints/moge-2-vitl-normal/model.pt"
export MODEL_ID="$AF/checkpoints/Wan2.1-T2V-1.3B-Diffusers"
export CHECKPOINT_PT="$AF/checkpoints/artifixer/artifixer-1.3b.pt"
export OUT="$AF/output"                                  # grouped per method: $OUT/<method>/<scene>
export RECON="$OUT/recon"                                # reconstructions: $RECON/<scene>
export MASKS="$AF/datasets/masks_objid"

declare -A CAPTION=(
  [garden]="A small backyard garden on an overcast day, with an old paved patio of weathered hexagonal stone tiles in the foreground, a patch of green lawn, leafy shrubs and a potted plant, and an ivy-covered brick wall with a black wooden door in the background."
  [kitchen]="An indoor dining room with a wooden table covered by a woven grey placemat and a pink table runner, wooden chairs, a leafy green houseplant and a bright window in the background."
  [bonsai]="An indoor room with a small round table draped in a knitted purple cloth holding a wooden stand, an orange road bicycle leaning against a white wall behind it, an electronic keyboard, a black amplifier and cardboard boxes on a wooden floor."
)
