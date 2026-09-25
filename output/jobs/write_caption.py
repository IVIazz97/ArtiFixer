#!/usr/bin/env python3
"""Write a caption.h5 from a fixed caption string, skipping the Qwen3-VL captioner.

Same on-disk format as data_processing.captioning.generate_captions.generate_caption_hdf5.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from data_processing.captioning.generate_captions import generate_text_embedding, get_text_encoder_and_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--caption", required=True)
parser.add_argument("--output_path", type=Path, required=True)
parser.add_argument("--text_encoder_model_id", required=True)
parser.add_argument("--max_sequence_length", type=int, default=512)
args = parser.parse_args()

text_encoder, tokenizer = get_text_encoder_and_tokenizer(args.text_encoder_model_id)
with torch.inference_mode():
    embedding = generate_text_embedding(args.caption, text_encoder, tokenizer, args.max_sequence_length)
args.output_path.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(args.output_path, "w") as hf:
    dataset = hf.create_dataset("frame_00000", data=embedding)
    dataset.attrs["caption"] = args.caption
    dataset.attrs["image_indices"] = np.array([0])
print(f"wrote {args.output_path} embedding={embedding.shape}")
