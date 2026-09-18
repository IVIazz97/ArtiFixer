# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashSplat segmentation of 3DGUT Gaussians from 2D masks.

The 3DGUT rasterizer in ``threedgut_flashsplat_tracer`` is a vendored copy of the one in
``thirdparty/3DGRUT-ArtiFixer``, instrumented to accumulate each Gaussian's alpha-blending
weight per mask label. ``solver`` turns that accumulation into labels in closed form.
"""
