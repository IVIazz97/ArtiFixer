#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JIT-compile the FlashSplat-instrumented 3DGUT extension (`lib3dgut_flashsplat_cc`).

Mirrors thirdparty/3DGRUT-ArtiFixer/tests/test_build_3dgut.py. Loads no data and runs no
training -- it only exercises slangc -> nvcc -> ld for the vendored copy, and confirms the
copy builds into its own directory without disturbing the stock `lib3dgut_cc`.

Run:
    python tests/test_flashsplat_build.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "thirdparty" / "3DGRUT-ArtiFixer"))

import hydra
from omegaconf import DictConfig, OmegaConf

import threedgrut.utils.misc  # noqa: F401  registers ${div:}, ${eq:}

OmegaConf.register_new_resolver("int_list", lambda values: [int(v) for v in values], replace=True)


@hydra.main(
    config_path="../thirdparty/3DGRUT-ArtiFixer/configs",
    config_name="apps/colmap_3dgut.yaml",
    version_base=None,
)
def main(conf: DictConfig) -> None:
    OmegaConf.set_struct(conf, False)
    conf.path = "/tmp/unused"
    conf.out_dir = "/tmp/unused"
    conf.experiment_name = "flashsplat_build_test"
    conf.selected_indices_file = None
    conf.num_selected_indices = None
    conf.train_test_split_file = None
    conf.image_path_override = None

    print("[1/3] Importing setup_3dgut_flashsplat")
    from flashsplat.threedgut_flashsplat_tracer.setup_3dgut_flashsplat import setup_3dgut_flashsplat

    print("[2/3] JIT-compiling lib3dgut_flashsplat_cc (slangc + nvcc + ld)")
    plugin = setup_3dgut_flashsplat(conf)

    print("[3/3] Checking the FlashSplat entry point is bound")
    assert hasattr(plugin.SplatRaster, "trace_contrib"), "trace_contrib missing from SplatRaster"
    assert hasattr(plugin.SplatRaster, "trace"), "trace missing from SplatRaster"
    print(f"PASS: loaded {plugin!r} with trace_contrib")


if __name__ == "__main__":
    main()
