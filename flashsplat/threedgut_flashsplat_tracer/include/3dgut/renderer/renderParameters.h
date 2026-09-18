// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <3dgut/sensors/sensors.h>
#include <3dgut/utils/bounding_box.h>

namespace threedgut {

using TTrackInstancePose = tcnn::vec<7>;

struct MemoryHandles {

    const uint64_t* handles;

    template <typename T>
    inline TCNN_HOST_DEVICE T* bufferPtr(int index) {
        return reinterpret_cast<T*>(handles[index]);
    }
};

// FlashSplat: per-Gaussian, per-label accumulation of the alpha-blending weight (alpha * T).
// Disabled (and zero-cost) whenever `contribution` is null, which is the case for every
// render path except the segmentation pass.
struct FlashSplatAccumulator {
    const int32_t* rayObjectId = nullptr; ///< [H*W] object id per pixel, negative to skip
    float* contribution        = nullptr; ///< [numObjects, numParticles] accumulator
    uint32_t numParticles      = 0;       ///< row stride of `contribution`
};

struct RenderParameters {
    uint32_t id;
    tcnn::ivec2 resolution;
    float hitTransmittance;
    threedgut::BoundingBox objectAABB;
    TSensorModel sensorModel;
    TSensorState sensorState;
    // tcnn::mat4x3 colorCorrectionMatrix;
};

} // namespace threedgut
