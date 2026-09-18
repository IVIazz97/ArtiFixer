<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FlashSplat-instrumented 3DGUT rasterizer

A copy of `thirdparty/3DGRUT-ArtiFixer/threedgut_tracer`, instrumented so the forward
rasterizer accumulates each Gaussian's alpha-blending weight per mask label. It builds as a
**separate** torch extension, `lib3dgut_flashsplat_cc`, so the stock `lib3dgut_cc` used by
ArtiFixer training and rendering is untouched and unaffected.

**Copied from:** `nv-tlabs/3DGRUT-ArtiFixer` @ `62e1038b74b2edc01440fd4ddf5f080109b6faba`
(2026-06-09), the commit ArtiFixer pins the submodule to.

## What differs from the original

The delta is deliberately small, so it can be re-applied by hand if the submodule is bumped.

| File | Change |
| --- | --- |
| `include/3dgut/renderer/renderParameters.h` | New `FlashSplatAccumulator` POD (mask pointer, accumulator pointer, row stride). |
| `include/3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh` | **The hook.** `processHitParticle` does `atomicAdd(&contribution[objId * N + particleIdx], hitWeight)` on each forward hit. Same in `evalForwardNoKBufferBalanced`, which also needed `particleIdx` hoisted out of an inner scope. The accumulator is threaded through `eval` and `evalKBuffer`. |
| `include/3dgut/kernels/cuda/renderers/gutRenderer.cuh` | `render` and `renderBalanced` kernels take the accumulator by value. `renderBackward` untouched. |
| `include/3dgut/renderer/gutRenderer.h`, `src/gutRenderer.cu` | `renderForward` takes two defaulted pointers and builds the accumulator. |
| `include/3dgut/splatRaster.h`, `src/splatRaster.cpp` | `trace` and the new `traceContrib` both delegate to a shared `traceImpl`. `traceContrib` validates the mask and accumulator tensors. |
| `bindings.cpp` | Binds `trace_contrib`. |
| `setup_3dgut.py` → `setup_3dgut_flashsplat.py` | Renames the extension to `lib3dgut_flashsplat_cc`; points tiny-cuda-nn and slang includes back at the submodule so those are not duplicated on disk. |
| `tracer.py` | Imports the renamed extension; adds `Tracer.accumulate_contribution`. |

Everything is inert when no accumulator is passed: `trace` forwards null pointers, and the
kernel's `if (flashSplat.contribution != nullptr)` guard skips the accumulation entirely.

## Resyncing after a submodule bump

```bash
diff -ru thirdparty/3DGRUT-ArtiFixer/threedgut_tracer flashsplat/threedgut_flashsplat_tracer
```

Re-copy the submodule directory, re-apply the table above, and update the commit recorded
here. `tests/test_flashsplat_build.py` then confirms the copy still compiles, and
`tests/test_flashsplat_accumulation.py` confirms the hook is still correct.
