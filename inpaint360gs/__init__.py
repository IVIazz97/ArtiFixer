# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inpaint360GS object-aware removal and inpainting, ported onto 3DGUT.

Follows Wang et al., "Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting
for 360° Scenes" (WACV 2026, arXiv:2511.06457) and its official code
(github.com/dfki-av/Inpaint360GS). See ``inpaint360gs/README.md`` for the stage list, which
environment runs each stage, and every deviation from the official implementation.
"""
