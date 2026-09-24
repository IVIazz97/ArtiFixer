# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion360 object-removal inpainting, ported onto 3DGUT + FlashSplat.

Follows Wu et al., "AuraFusion360: Augmented Unseen Region Alignment for Reference-based 360°
Unbounded Scene Inpainting" (CVPR 2025) and its official code
(github.com/kkennethwu/AuraFusion360_official). See ``aurafusion/README.md`` for the stage list,
which environment runs each stage, and every deviation from the official implementation.
"""
