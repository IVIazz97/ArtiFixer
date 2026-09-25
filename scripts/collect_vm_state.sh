#!/usr/bin/env bash
# Snapshot a machine's ArtiFixer checkout so scripts for it can be written elsewhere.
#
# Run from the repo root on the target machine (no AI needed there):
#   curl -fsSL https://raw.githubusercontent.com/IVIazz97/ArtiFixer/main/scripts/collect_vm_state.sh -o /tmp/collect.sh
#   bash /tmp/collect.sh
#
# Writes one text file, $OUT (default ~/artifixer_vm_state.txt), made of these sections:
#   report.txt        system, GPU/CUDA, git state, python envs, data inventory
#   local.patch       uncommitted changes of the main repo (tracked + small untracked text files)
#   sub_<name>.patch  same, per initialized submodule
#   unpushed.patch    local commits not on the upstream branch (format-patch)
#   ignored_code.patch  code files (.py/.sh/.yaml/...) living in gitignored paths, as new-file diffs
# Each section starts with a line `##### FILE: <name> #####`. The working tree, index and
# branches of this checkout are not touched. No environment dump, shell history or tokens are
# collected; remote URLs are redacted.
set -uo pipefail

REPO=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "run this from inside the ArtiFixer repo" >&2; exit 1; }
cd "$REPO"
OUT=${OUT:-$HOME/artifixer_vm_state.txt}
MAX_UNTRACKED=${MAX_UNTRACKED:-524288}   # bytes; larger untracked files are listed, not included
OUT_DIR=$(mktemp -d)
trap 'rm -rf "$OUT_DIR"' EXIT
R="$OUT_DIR/report.txt"

sec() { printf '\n==================== %s ====================\n' "$*" >> "$R"; }
run() { printf '$ %s\n' "$*" >> "$R"; timeout 120 bash -c "$*" >> "$R" 2>&1; echo >> "$R"; }
redact() { sed -E 's#(https?://)[^/@ ]+@#\1***@#g'; }

# Full diff (tracked changes + small untracked text files) of the repo at $1 against its HEAD,
# built in a throwaway index so the real one is left alone.
patch_of() {
  local dir=$1 idx
  idx=$(mktemp)
  cp "$(git -C "$dir" rev-parse --absolute-git-dir)/index" "$idx" 2>/dev/null || : > "$idx"
  GIT_INDEX_FILE=$idx git -C "$dir" add -u 2>/dev/null
  git -C "$dir" ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do
    local p="$dir/$f"
    if [ -f "$p" ] && [ "$(stat -c %s "$p")" -le "$MAX_UNTRACKED" ] && grep -Iq . "$p"; then
      GIT_INDEX_FILE=$idx git -C "$dir" add -- "$f" 2>/dev/null
    fi
  done
  GIT_INDEX_FILE=$idx git -C "$dir" diff --cached --binary HEAD
  rm -f "$idx"
}

git_state() {
  local dir=$1
  run "git -C '$dir' log --oneline -8 --decorate"
  run "git -C '$dir' status --short --branch --untracked-files=all | head -300"
  run "git -C '$dir' stash list"
  printf '$ untracked files > %s bytes or binary (not in the patch)\n' "$MAX_UNTRACKED" >> "$R"
  git -C "$dir" ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do
    local p="$dir/$f"
    if [ -f "$p" ] && { [ "$(stat -c %s "$p")" -gt "$MAX_UNTRACKED" ] || ! grep -Iq . "$p"; }; then
      printf '  %10s  %s\n' "$(stat -c %s "$p")" "$f"
    fi
  done >> "$R"
}

echo "collecting (2-5 min) ..."
{ echo "ArtiFixer VM state, $(date -Is)"; echo "repo: $REPO"; } > "$R"

sec "SYSTEM"
run "grep PRETTY_NAME /etc/os-release; uname -srm"
run "[ -f /.dockerenv ] && echo 'inside docker' || echo 'not docker'; nproc; free -g | head -2"
run "df -h '$REPO' '$HOME' /tmp 2>/dev/null"
for t in sbatch docker conda mamba uv pip python python3 nvcc gcc ninja cmake; do
  printf '%-8s %s\n' "$t" "$(command -v "$t" || echo -)"
done >> "$R"
run "gcc --version | head -1; cmake --version | head -1"
run "curl -s -o /dev/null -m 10 -w 'huggingface.co HTTP %{http_code}\n' https://huggingface.co; curl -s -o /dev/null -m 10 -w 'github.com HTTP %{http_code}\n' https://github.com"

sec "GPU / CUDA"
run "nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv"
run "nvidia-smi | head -5"
run "nvcc --version | tail -2"
run "ls -d /usr/local/cuda* 2>/dev/null"
run "echo CUDA_HOME=\${CUDA_HOME:-} CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-} TORCH_CUDA_ARCH_LIST=\${TORCH_CUDA_ARCH_LIST:-}"
run "echo VIRTUAL_ENV=\${VIRTUAL_ENV:-} CONDA_PREFIX=\${CONDA_PREFIX:-} ENVS=\${ENVS:-}; echo PYTHONPATH=\${PYTHONPATH:-}"
run "echo HF_HOME=\${HF_HOME:-} HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-} TORCH_HOME=\${TORCH_HOME:-}"

sec "GIT: main repo"
git remote -v | redact >> "$R"
run "git rev-parse HEAD; git rev-parse --abbrev-ref HEAD; git rev-parse --abbrev-ref @{u}"
GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND='ssh -o BatchMode=yes' timeout 60 git fetch -q 2>/dev/null
run "git rev-list --left-right --count @{u}...HEAD | awk '{print \"behind upstream:\", \$1, \" ahead:\", \$2}'"
git_state "$REPO"
patch_of "$REPO" > "$OUT_DIR/local.patch"
git format-patch --stdout '@{u}..HEAD' > "$OUT_DIR/unpushed.patch" 2>/dev/null
# Code edited inside gitignored paths (e.g. output/jobs on checkouts older than dc0f50e).
timeout 120 git ls-files -z --others --ignored --exclude-standard 2>/dev/null \
  | grep -z -E '\.(py|sh|sbatch|ya?ml|patch|diff|md|cfg|toml|ini)$' \
  | grep -z -v -E '(^|/)(\.?venv|envs|site-packages|__pycache__|node_modules|checkpoints|datasets|download|build|\.cache|wandb|tmp)/' \
  | head -z -n 300 | while IFS= read -r -d '' f; do
    [ "$(stat -c %s "$f")" -le "$MAX_UNTRACKED" ] && git diff --no-index --binary /dev/null "$f"
  done > "$OUT_DIR/ignored_code.patch"
run "git diff --stat HEAD; wc -l '$OUT_DIR/local.patch' '$OUT_DIR/unpushed.patch' '$OUT_DIR/ignored_code.patch'"
run "find . -name '*.patch' -o -name '*.diff' -not -path './.git/*' | grep -v '^./.git/' | head -50"

sec "GIT: submodules"
run "git submodule status --recursive"
git submodule foreach --quiet --recursive 'echo "$toplevel/$sm_path"' 2>/dev/null | while read -r sub; do
  name=$(echo "${sub#"$REPO"/}" | tr '/' '_')
  sec "submodule ${sub#"$REPO"/}"
  git -C "$sub" remote -v | redact >> "$R"
  git_state "$sub"
  patch_of "$sub" > "$OUT_DIR/sub_$name.patch"
  run "git -C '$sub' diff --stat HEAD"
  if [ -f thirdparty/patches/3DGRUT-ArtiFixer-flashsplat.patch ] && [ "$name" = thirdparty_3DGRUT-ArtiFixer ]; then
    run "git -C '$sub' apply --check --reverse '$REPO/thirdparty/patches/3DGRUT-ArtiFixer-flashsplat.patch' && echo 'flashsplat patch: APPLIED' || echo 'flashsplat patch: not (fully) applied'"
  fi
done

sec "ACTIVATION SCRIPTS"
for f in tmp/activate_repo_env.sh ../tmp/activate_repo_env.sh env.sh ../env.sh; do
  [ -f "$f" ] && { echo "--- $f"; cat "$f"; } >> "$R"
done

sec "PYTHON ENVS"
declare -A SEEN
cands=()
[ -x .venv/bin/python ] && cands+=("$REPO/.venv/bin/python")
for d in "${ENVS:-/nonexistent}"/* "$REPO"/../envs/* "$HOME"/envs/* "$HOME"/.venvs/*; do
  [ -x "$d/bin/python" ] && cands+=("$d/bin/python")
done
if command -v conda >/dev/null; then
  while read -r p; do [ -x "$p/bin/python" ] && cands+=("$p/bin/python"); done \
    < <(conda env list 2>/dev/null | awk '!/^#/ && NF {print $NF}')
fi
for p in $(which -a python python3 2>/dev/null); do cands+=("$p"); done
while read -r p; do cands+=("$p"); done < <(timeout 60 find "$REPO/.." "$HOME" /workspace /opt \
  -maxdepth 4 -path '*/bin/python' \( -type f -o -type l \) 2>/dev/null | grep -v -e site-packages -e '/.cache/' | head -30)

n=0
for py in "${cands[@]}"; do
  prefix=$(timeout 30 "$py" -c 'import sys; print(sys.prefix)' 2>/dev/null) || continue
  [ -n "${SEEN[$prefix]:-}" ] && continue
  SEEN[$prefix]=1; n=$((n + 1)); [ $n -gt 10 ] && break
  sec "env $prefix  (via $py)"
  PYTHONPATH="$REPO:$REPO/thirdparty/3DGRUT-ArtiFixer" timeout 300 "$py" - >> "$R" 2>&1 <<'EOF'
import importlib.util, sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
          [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
except Exception as e:
    print("torch: FAIL", type(e).__name__, e)
mods = ["torchvision", "diffusers", "transformers", "accelerate", "threedgrut", "hydra", "omegaconf",
        "sam2", "pytorch_lightning", "open_clip", "clip", "ldm", "cv2", "lpips", "kornia", "h5py",
        "simple_knn", "diff_gaussian_rasterization", "gsplat", "tinycudann", "flash_attn"]
print("modules found:", " ".join(m for m in mods if importlib.util.find_spec(m)))
print("modules missing:", " ".join(m for m in mods if not importlib.util.find_spec(m)))
try:
    import threedgrut  # noqa: F401
    print("import threedgrut: OK")
except Exception as e:
    print("import threedgrut: FAIL", type(e).__name__, e)
import importlib.metadata as md
print("--- packages")
print("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in md.distributions()}, key=str.lower)))
EOF
done
run "ls ~/.cache/torch_extensions/* 2>/dev/null | head -40"

sec "DATA: repo dirs"
run "ls -la checkpoints datasets output 2>&1 | head -120"
run "find checkpoints -maxdepth 3 2>/dev/null | head -150"
run "du -sh checkpoints/* datasets/* output/* 2>/dev/null"
run "find datasets -maxdepth 3 -type d 2>/dev/null | head -150"
run "find output -maxdepth 4 -type d -not -path 'output/jobs*' 2>/dev/null | head -250"
run "ls \${HF_HOME:-\$HOME/.cache/huggingface}/hub 2>/dev/null"

sec "DATA: trained scenes / COLMAP / masks anywhere"
roots=$(for d in "$REPO" "$REPO/.." "$HOME" /data /workspace /mnt /scratch /datasets; do
  [ -d "$d" ] && (cd "$d" && pwd -P); done | sort -u | tr '\n' ' ')
run "timeout 180 find $roots -maxdepth 9 \\( -name .git -o -name site-packages -o -name node_modules -o -name .cache \\) -prune -o \
  \\( -name 'ckpt_*.pt' -o -name 'cameras.bin' -o -name 'cameras.txt' -o -name 'transforms*.json' -o -name 'caption.h5' \
     -o -name 'labels.pt' -o -name 'split.json' -o -name 'parsed.yaml' -o -name 'point_cloud.ply' \\) -print 2>/dev/null | sort -u | head -300"
run "timeout 120 find $roots -maxdepth 7 -type d \\( -name 'masks*' -o -name 'images' -o -name 'images_[0-9]*' \\) -not -path '*/.git/*' 2>/dev/null | sort -u | head -150"

sec "DONE"
(cd "$OUT_DIR" && wc -c ./*) >> "$R"
for f in "$OUT_DIR"/report.txt "$OUT_DIR"/local.patch "$OUT_DIR"/unpushed.patch "$OUT_DIR"/ignored_code.patch \
         "$OUT_DIR"/sub_*.patch; do
  [ -f "$f" ] || continue
  printf '##### FILE: %s #####\n' "$(basename "$f")"
  cat "$f"
  echo
done > "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)). Send that one file."
