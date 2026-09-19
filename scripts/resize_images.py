#!/usr/bin/env python3
from pathlib import Path
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("target", type=Path)
args = parser.parse_args()
args.target.mkdir(parents=True, exist_ok=True)
for path in args.source.glob("*.jpg"):
    output = args.target / path.name
    if output.exists():
        continue
    with Image.open(path) as image:
        image.resize((max(1, image.width // 2), max(1, image.height // 2)), Image.Resampling.LANCZOS).save(
            output, quality=95
        )
print(f"resized images in {args.target}")
