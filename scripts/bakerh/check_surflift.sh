#!/usr/bin/env bash
# Read-only check: is everything SurfaceLift needs on this VM, and which AuraFusion360 / Inpaint360GS
# baseline runs are finished? Downloads nothing, changes nothing.
#
# SurfaceLift (CVPR2027/SurfaceLift) fills the hole left by an object removal from ONE chosen view:
# inpaint that view, MoGe-2 depth aligned to the removed scene, Gaussians bound to that surface,
# observation ledger (never change what a photo saw). On the bakerh scenes it would run on top of the
# 3DGUT removal the i360/af360 ports already made (output/bakerh_removal/<scene>/<variant>/).
#
# Usage (from any directory):  bash scripts/bakerh/check_surflift.sh
# Knobs: MODELS, AF360_REPO, VENV, VENV2D, GPU (1), LOAD_TEST (1 = load MoGe + LaMa once on the GPU).
# Writes output/bakerh_surflift/CHECK_REPORT.txt; send that file back. The last line says
# CHECK OK or CHECK: N MISSING, and the "TO DOWNLOAD" block lists exactly what to fetch and where it goes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AF=$(cd "$HERE/../.." && pwd)
cd "$AF"
MODELS=${MODELS:-/workspace/amazzucchelli/fbk-3dworld/models}
AF360_REPO=${AF360_REPO:-/workspace/amazzucchelli/fbk-3dworld/AuraFusion360_official}
VENV=${VENV:-$AF/.venv}
VENV2D=${VENV2D:-$AF/.venv_af2d}
SAM_MASKS=${SAM_MASKS:-/workspace/amazzucchelli/fbk-3dworld/FlashSplat/sam3_single_object_anchor_tracking}
GPU=${GPU:-1}
LOAD_TEST=${LOAD_TEST:-1}
PY=$VENV/bin/python
OUT=$AF/output/bakerh_surflift
mkdir -p "$OUT"
REPORT=$OUT/CHECK_REPORT.txt
exec > >(tee "$REPORT") 2>&1

missing=()   # "need|what|put it here|source|size"  -> printed in the TO DOWNLOAD block
todo=()      # things that are not downloads (run a job, install a package)
section() { echo; echo "==================== $1 ===================="; }

echo "check_surflift $(date -Is)   repo=$AF   git=$(git rev-parse --short HEAD 2>/dev/null)"
echo "MODELS=$MODELS   AF360_REPO=$AF360_REPO   GPU=$GPU"

# ---------------------------------------------------------------------------------------------
# Models. Columns: need (required|engine|optional), label, path to check, min bytes, source, size.
#   required = core SurfaceLift (chosen-view inpainting with LaMa, MoGe-2 depth)
#   engine   = LeftRefill refinement inside the ledger hole (our best 360-USID variant)
#   af360    = only the AuraFusion360 runs themselves need it (SAM2 unseen masks, Marigold AGDD)
#   optional = Qwen-Image-Edit-2511 chosen-view editor (the v3c default on Leonardo; LaMa replaces it)
# ---------------------------------------------------------------------------------------------
section "models"
check_file() {  # need label path min_bytes source size
  local need=$1 label=$2 path=$3 min=$4 src=$5 size=$6 got
  if [ -f "$path" ]; then
    got=$(stat -L -c %s "$path" 2>/dev/null || echo 0)
    if [ "$got" -ge "$min" ]; then
      printf '  ok       %-9s %-34s %s (%s bytes)\n' "$need" "$label" "$path" "$got"; return
    fi
    printf '  TOO SMALL %-8s %-34s %s (%s bytes; a git-lfs pointer or a partial copy?)\n' "$need" "$label" "$path" "$got"
  else
    printf '  MISSING  %-9s %-34s %s\n' "$need" "$label" "$path"
  fi
  missing+=("$need|$label|$path|$src|$size")
}
check_dir() {  # need label dir "file1 file2 ..." source size
  local need=$1 label=$2 dir=$3 files=$4 src=$5 size=$6 f bad=()
  for f in $files; do [ -e "$dir/$f" ] || bad+=("$f"); done
  if [ ${#bad[@]} -eq 0 ]; then
    printf '  ok       %-9s %-34s %s (%s)\n' "$need" "$label" "$dir" "$(du -shL "$dir" 2>/dev/null | cut -f1)"
  else
    printf '  MISSING  %-9s %-34s %s (lacks: %s)\n' "$need" "$label" "$dir" "${bad[*]}"
    missing+=("$need|$label|$dir/|$src|$size")
  fi
}

check_file required "MoGe-2 ViT-L normal (depth)" "$AF/checkpoints/moge-2-vitl-normal/model.pt" 1000000000 \
  "hf download Ruicheng/moge-2-vitl-normal model.pt --local-dir moge-2-vitl-normal" "1.3 GB"
check_file required "big-lama TorchScript (inpaint)" "$MODELS/LaMa/big-lama.pt" 200000000 \
  "hf download fashn-ai/LaMa big-lama.pt --local-dir LaMa" "206 MB"
check_file engine "SD2-inpainting 512-ema" "$MODELS/stable-diffusion-2-inpainting/512-inpainting-ema.ckpt" 5000000000 \
  "hf download sd2-community/stable-diffusion-2-inpainting 512-inpainting-ema.ckpt --local-dir stable-diffusion-2-inpainting" "5.2 GB"
check_file engine "OpenCLIP ViT-H-14 laion2B" "$MODELS/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin" 3000000000 \
  "hf download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_pytorch_model.bin open_clip_config.json --local-dir CLIP-ViT-H-14-laion2B-s32B-b79K" "3.9 GB"
check_file engine "LeftRefill learned prompt" \
  "$AF360_REPO/utils/LeftRefill/check_points/ref_guided_inpainting/ckpts/epoch=7-step=6039.ckpt" 100000 \
  "git clone https://github.com/kkennethwu/AuraFusion360_official.git (or: bash scripts/bakerh/setup_baselines.sh)" "GitHub"
check_file af360 "SAM2 hiera-large" "$MODELS/sam2-hiera-large/sam2_hiera_large.pt" 800000000 \
  "hf download facebook/sam2-hiera-large sam2_hiera_large.pt --local-dir sam2-hiera-large" "900 MB"
check_dir af360 "Marigold depth v1-0" "$MODELS/marigold-depth-v1-0" "model_index.json unet vae text_encoder tokenizer scheduler" \
  "hf download prs-eth/marigold-v1-0 --local-dir marigold-depth-v1-0" "~5 GB (fp32) / 2.6 GB (fp16 files only)"
check_dir optional "Qwen-Image-Edit-2511 (editor)" "$MODELS/Qwen-Image-Edit-2511" \
  "model_index.json transformer text_encoder vae processor tokenizer scheduler" \
  "hf download Qwen/Qwen-Image-Edit-2511 --local-dir Qwen-Image-Edit-2511" "54 GB"

# ---------------------------------------------------------------------------------------------
section "python (.venv: SurfaceLift core runs here)"
if [ ! -x "$PY" ]; then
  echo "  MISSING  $PY"; todo+=("create the ArtiFixer .venv ($PY)")
else
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$AF:$AF/thirdparty/3DGRUT-ArtiFixer" "$PY" - <<'EOF' || todo+=("pip install the .venv packages marked MISSING above")
import importlib, sys
bad = []
def has(mod, attr=None, need="required"):
    try:
        m = importlib.import_module(mod)
        if attr and not hasattr(m, attr):
            raise ImportError(f"{mod} has no {attr}")
        v = getattr(m, "__version__", "")
        print(f"  ok       {need:9s} {mod}{'.' + attr if attr else ''} {v}")
    except Exception as e:
        print(f"  MISSING  {need:9s} {mod}{'.' + attr if attr else ''}  ({type(e).__name__}: {e})")
        if need == "required":
            bad.append(mod)
import torch
print(f"  torch {torch.__version__} cuda {torch.version.cuda} available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"  device {torch.cuda.get_device_name()} free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
for mod in ("numpy", "scipy", "PIL", "cv2", "threedgrut", "flashsplat"):
    has(mod)
has("moge.model.v2", "MoGeModel")
has("diffusers", "QwenImageEditPlusPipeline", need="optional")
sys.exit(1 if bad else 0)
EOF
fi

section "python (.venv_af2d: LeftRefill engine runs here)"
if [ -x "$VENV2D/bin/python" ] && "$VENV2D/bin/python" -c "import open_clip, pytorch_lightning, diffusers" 2>/dev/null; then
  echo "  ok       engine    $VENV2D (open_clip, pytorch_lightning, diffusers)"
else
  echo "  MISSING  engine    $VENV2D"; todo+=("engine: run 'bash scripts/bakerh/setup_baselines.sh' once (builds .venv_af2d, clones AuraFusion360)")
fi

# ---------------------------------------------------------------------------------------------
# Per-scene inputs: trained 3DGUT, cameras, object masks, and a finished removal to build on.
# These are produced on the VM, never downloaded.
# ---------------------------------------------------------------------------------------------
section "scene inputs"
for S in TA_BLUE_MOTOR PCV Compressor; do
  if [ "$S" = Compressor ]; then SR=$AF/output/bakerh_undis_ds2_from1600/$S; else SR=$AF/output/bakerh_undis/$S; fi
  echo "  -- $S"
  for p in "$SR/3dgrut_runs/$S/$S/ours_30000/ckpt_30000.pt" "$SR/3dgrut_input/$S/sparse"; do
    if [ -e "$p" ]; then echo "     ok       $p"; else echo "     MISSING  $p"; todo+=("$S: $p is missing (made on the VM, not downloadable)"); fi
  done
  md=""
  for d in "$SAM_MASKS/$S/full/masks_npy" "$SAM_MASKS/$S/0_400/masks_npy"; do [ -d "$d" ] && { md=$d; break; }; done
  if [ -n "$md" ]; then echo "     ok       $md ($(ls "$md" | wc -l) masks)"
  else echo "     MISSING  SAM3 masks under $SAM_MASKS/$S"; todo+=("$S: SAM3 object masks missing"); fi
  rem=()
  for v in i360 i360sam af360 af360sam; do
    f=$(find "$AF/output/bakerh_removal/$S/$v" -maxdepth 3 -name ckpt_removed.pt 2>/dev/null | head -1)
    [ -n "$f" ] && rem+=("$v")
  done
  if [ ${#rem[@]} -gt 0 ]; then echo "     ok       removed scene from: ${rem[*]}"
  else echo "     MISSING  no ckpt_removed.pt under output/bakerh_removal/$S/{i360,i360sam,af360,af360sam}"
       todo+=("$S: no finished removal yet: run 'bash scripts/bakerh/run_removal.sh $S i360' first"); fi
done

# ---------------------------------------------------------------------------------------------
# Baseline runs (run_removal.sh): finished = final/ckpt_final.pt, the same rule run_removal.sh uses to
# skip a job. views = the FRAMES the run used (views.txt); renders = final/rgb PNGs; last = the last
# line of the variant's cumulative logs/summary.log (where an unfinished run stopped).
# ---------------------------------------------------------------------------------------------
section "baseline runs (output/bakerh_removal/<scene>/<variant>)"
printf '  %-14s %-9s %-10s %-14s %-8s %s\n' scene variant status views renders last
for S in TA_BLUE_MOTOR PCV Compressor; do
  for v in af360 af360sam i360 i360sam; do
    O=$AF/output/bakerh_removal/$S/$v
    views=$(cat "$O/views.txt" 2>/dev/null || echo -)
    n=$(ls "$O/final/rgb" 2>/dev/null | grep -c '\.png$')
    last=$(tail -n 1 "$O/logs/summary.log" 2>/dev/null | cut -c1-110)
    if [ -f "$O/final/ckpt_final.pt" ]; then st=DONE
    elif [ -d "$O" ]; then st=UNFINISHED; todo+=("$S $v unfinished: bash scripts/bakerh/run_removal.sh $S $v")
    else st="not run"; todo+=("$S $v not run: bash scripts/bakerh/run_removal.sh $S $v")
    fi
    printf '  %-14s %-9s %-10s %-14s %-8s %s\n' "$S" "$v" "$st" "$views" "$n" "${last:--}"
  done
done

# ---------------------------------------------------------------------------------------------
section "disk"
df -h "$AF" "$MODELS" 2>/dev/null | awk 'NR == 1 || !seen[$0]++'

# ---------------------------------------------------------------------------------------------
if [ "$LOAD_TEST" = 1 ] && [ -x "$PY" ]; then
  section "load test (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" - "$AF/checkpoints/moge-2-vitl-normal/model.pt" "$MODELS/LaMa/big-lama.pt" <<'EOF' \
    || todo+=("a model file exists but does not load: see 'load test' above")
import sys, traceback
from pathlib import Path
import torch
bad = 0
def run(name, fn):
    global bad
    try:
        fn(); print(f"  ok       {name}", flush=True)
    except Exception:
        traceback.print_exc(); print(f"  FAIL     {name}", flush=True); bad += 1
moge, lama = map(Path, sys.argv[1:3])
if not torch.cuda.is_available():
    print("  SKIP     no CUDA device visible: load test not run"); sys.exit(0)
def load_moge():
    from moge.model.v2 import MoGeModel
    m = MoGeModel.from_pretrained(str(moge)).cuda().eval()
    with torch.no_grad():
        out = m.infer(torch.rand(3, 480, 640, device="cuda"))
    print(f"           MoGe-2 depth {tuple(out['depth'].shape)}, normals {'normal' in out}")
if moge.exists(): run("MoGe-2 load + one inference", load_moge)
if lama.exists(): run("big-lama TorchScript load", lambda: torch.jit.load(str(lama), map_location="cpu"))
sys.exit(1 if bad else 0)
EOF
fi

# ---------------------------------------------------------------------------------------------
section "summary"
n_req=0
for m in "${missing[@]}"; do [ "${m%%|*}" != optional ] && n_req=$((n_req + 1)); done
if [ ${#missing[@]} -gt 0 ]; then
  echo "TO DOWNLOAD (the VM cannot reach Hugging Face: fetch on a machine with internet, then copy here)"
  i=0
  for m in "${missing[@]}"; do
    IFS='|' read -r need label path src size <<< "$m"
    i=$((i + 1))
    echo "  $i. [$need] $label   ($size)"
    echo "       get:     $src"
    echo "       put at:  $path"
  done
  echo "  required = SurfaceLift cannot run without it; engine = LeftRefill refinement (best variant);"
  echo "  af360 = only the AuraFusion360 baseline runs need it;"
  echo "  optional = Qwen editor for the chosen view (LaMa is used instead when it is absent)."
fi
if [ ${#todo[@]} -gt 0 ]; then
  echo "TO DO ON THE VM (not downloads)"
  for t in "${todo[@]}"; do echo "  - $t"; done
fi
echo
if [ $n_req -eq 0 ] && [ ${#todo[@]} -eq 0 ]; then
  echo "CHECK OK$( [ ${#missing[@]} -gt 0 ] && echo " (optional items missing: see TO DOWNLOAD)")   report: $REPORT"
else
  echo "CHECK: $n_req MISSING downloads, ${#todo[@]} VM to-dos   send back $REPORT"
  exit 1
fi
