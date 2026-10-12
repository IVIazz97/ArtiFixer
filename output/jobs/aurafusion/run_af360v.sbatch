#!/bin/bash
#SBATCH --account=IscrC_EditGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/logs/%x_%j.log
# af360v (AuraFusion360 with virtual views framing the whole hole) on a mip360 scene, the same stage
# commands as scripts/bakerh/run_removal.sh's af360v phases. Usage:
#   sbatch --job-name=AFV_kitchen --export=ALL,SCENE=kitchen output/jobs/aurafusion/run_af360v.sbatch
# Optional: STEPS (subset of virtual+render+contour+sam2+ref+agdd+init+sdedit+finetune), VARIANT (output subdir),
# COLMAP_DIR (other views of the same reconstruction), PLAIN=1 (af360 as run_removal.sh runs it: no virtual
# views, no flags, reference = largest unseen mask; for side-by-side tests).
set -euo pipefail
AF=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer
source $AF/output/jobs/common.sh
STEPS=${STEPS:-virtual,render,contour,sam2,ref,agdd,init,sdedit,finetune}; STEPS=${STEPS//+/,}
VARIANT=${VARIANT:-af360v}
G="$AF/output/mip360/$SCENE"; CKPT="$G/3dgrut_runs/$SCENE/$SCENE/ours_30000/ckpt_30000.pt"; COLMAP="${COLMAP_DIR:-$G/3dgrut_input/$SCENE}"
FSP="$G/removal/flashsplat"
O="$G/removal/aurafusion/$VARIANT"; VCOLMAP="$O/colmap_virtual"
CK=$AF/checkpoints/aurafusion
export PYTHONPATH="$AF:$AF/thirdparty/3DGRUT-ArtiFixer:${PYTHONPATH:-}"
AF_PY=$ENVS/aurafusion/bin/python
echo "host=$(hostname) scene=$SCENE variant=$VARIANT steps=$STEPS"
has() { [[ ",$STEPS," == *",$1,"* ]]; }
t() { local s=$SECONDS; "$@"; echo "TIME $3: $((SECONDS - s)) s"; }
mkdir -p "$O"
VFLAGS=(--virtual_json "$O/virtual.json"); FFLAGS=(--in_frame_only "${VFLAGS[@]}")
if [ -n "${PLAIN:-}" ]; then VCOLMAP=$COLMAP; VFLAGS=(); FFLAGS=(); STEPS=${STEPS//virtual,/}; fi
has virtual && t python -m aurafusion.virtual_views --checkpoint "$CKPT" --colmap_dir "$COLMAP" --flashsplat_dir "$FSP" \
                 --output_dir "$O"
has render  && t python -m aurafusion.render_views --checkpoint "$CKPT" --colmap_dir "$VCOLMAP" --flashsplat_dir "$FSP" \
                 --output_dir "$O"
has contour && t python -m aurafusion.unseen_contour --render_dir "$O" "${VFLAGS[@]}"
has sam2    && t $AF_PY -m aurafusion.sam2_unseen --render_dir "$O"
has ref     && { python -m aurafusion.frame_views --render_dir "$O" "${VFLAGS[@]}" > "$O/reference_index.txt"; }
# PLAIN: af360's rule (largest unseen mask over all views), as run_removal.sh af_ref
if has ref && [ -n "${PLAIN:-}" ]; then python -c 'import sys
from pathlib import Path
import numpy as np
from PIL import Image
paths = sorted(Path(sys.argv[1]).glob("*.png"))
area = [np.count_nonzero(np.asarray(Image.open(p)) > 127) for p in paths]
print(int(paths[int(np.argmax(area))].stem))' "$O/unseen_dilated" > "$O/reference_index.txt"; fi
REF=$(cat "$O/reference_index.txt" 2>/dev/null || echo 0); echo "reference view: $REF"
has agdd    && t $AF_PY -m aurafusion.agdd --render_dir "$O" --colmap_dir "$VCOLMAP" --reference_index "$REF" \
                 --lama_model "$AF/checkpoints/lama/big-lama.pt" --marigold "$CK/marigold-v1-0"
has init    && t python -m aurafusion.init_gaussians --render_dir "$O" --colmap_dir "$VCOLMAP" --reference_index "$REF"
has sdedit  && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 t $AF_PY -m aurafusion.sdedit --render_dir "$O" --reference_index "$REF" \
                 --official_repo /leonardo_scratch/fast/IscrC_EditGS/opt/AuraFusion360_official \
                 --sd2_inpainting_ckpt "$CK/sd2-inpainting/512-inpainting-ema.ckpt"
has finetune && t python -m aurafusion.finetune --render_dir "$O" --colmap_dir "$VCOLMAP" "${FFLAGS[@]}"
echo AURAFUSION_DONE
