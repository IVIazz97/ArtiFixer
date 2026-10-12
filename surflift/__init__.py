# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SurfaceLift (our method), ported onto 3DGUT + the FlashSplat removal.

Vendored from the research code in ``CVPR2027/SurfaceLift`` (snapshot 2026-10-12, not a git repo):
v3c (observation ledger, unseen-only hole, surface-bound single-view lift, harmonic colour
alignment) followed by pipeline C of the benchmark (ledger hole of every view + shadow extension,
AuraFusion360's LeftRefill SDEdit of our renders, multi-view colour fit of the lifted Gaussians):
the ``surflift_leftrefill_shadow`` row. Each module names the file it comes from. See
``surflift/README.md`` for the stage table and every deviation from the official code.
"""
