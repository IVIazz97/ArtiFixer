#!/bin/bash
# Smoke-test ArtiFixer 1.3B inference end to end on 21 garden frames, using the real photos
# as stand-in renders and full opacity. Checks model loading and the pipeline, not quality.
set -euo pipefail
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/jobs/common.sh
S=$AF/output/smoke
rm -rf "$S"; mkdir -p "$S/renders" "$S/opacity"
python - <<EOF
import json
from pathlib import Path
from PIL import Image
src = Path("$OUT/garden/3dgrut_input/garden")
t = json.loads((src / "nerfstudio/transforms.json").read_text())
t["frames"] = t["frames"][:21]
Path("$S/transforms.json").write_text(json.dumps(t))
for i, f in enumerate(t["frames"]):
    im = Image.open(src / f["file_path"]).convert("RGB")
    im.save(f"$S/renders/{i:05d}.png")
    Image.new("L", im.size, 255).save(f"$S/opacity/{i:05d}.png")
Path("$S/selected_indices.json").write_text(json.dumps(list(range(0, 21, 2))))
split = {"test": {"garden": {
    "transforms_path": "$S/transforms.json", "image_root": str(src), "render_dir": "$S/renders",
    "opacity_dir": "$S/opacity", "selected_indices_path": "$S/selected_indices.json",
    "prompt_path": "$S/caption.h5", "camera_scale": 0.01, "has_gt": True}}}
Path("$S/split.json").write_text(json.dumps(split, indent=2))
EOF
python output/jobs/write_caption.py --caption "${CAPTION[garden]}" --output_path "$S/caption.h5" --text_encoder_model_id "$MODEL_ID"
python -m model_eval.run_inference --evalset reconstructed_colmap \
    --checkpoint_pt "$CHECKPOINT_PT" --model_id "$MODEL_ID" \
    --save_dir "$S/out" --split_path "$S/split.json" --render_trajectory all_frames --replace_if_exists
find "$S/out" -maxdepth 5 | head -30
echo SMOKE_DONE
