<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FlashSplat segmentation for 3DGUT

Handoff document. Describes every change made to this repository to add FlashSplat
segmentation of 3DGUT Gaussians, why each one exists, and what has and has not been
verified. Nothing outside the files listed here was touched.

---

## 1. Goal

Segment the Gaussians of a trained 3DGUT reconstruction using 2D masks.

[FlashSplat](https://github.com/florinshen/FlashSplat) (ECCV 2024) solves this in closed
form. Its insight: a rendered mask is a **linear** function of the per-Gaussian labels,
where each Gaussian's coefficient at a pixel is its alpha-blending weight `α·T`. So if the
rasterizer accumulates, per Gaussian and per mask label, the sum of `α·T` over every pixel
of every view, the optimal label assignment is a per-Gaussian argmax. No optimization loop.

The work required is therefore:

1. make the 3DGUT rasterizer accumulate `α·T` per Gaussian per label (§3), and
2. apply the FlashSplat solver to that accumulation (§4).

**Hard constraint from the user: all of this lives inside the `artifixer` repository.**
The 3DGUT rasterizer normally lives in `thirdparty/3DGRUT-ArtiFixer`, which is a **git
submodule pointing at a different repository** (`nv-tlabs/3DGRUT-ArtiFixer`). Editing it
would put the work outside artifixer. So the rasterizer was **copied** into artifixer's own
tree instead. The submodule is byte-for-byte untouched — verify with
`git -C thirdparty/3DGRUT-ArtiFixer status`.

## 2. Why the port is small

3DGUT computes the blend weight in exactly one place, and it means exactly what `α·T`
means in 3DGS. From `include/3dgut/kernels/slang/models/gaussianParticles.slang:234`:

```slang
const float weight = backToFront ? alpha : alpha * transmittance;   // <- this is α·T
...
transmittance *= (1 - alpha);
```

That is exposed to CUDA as `densityIntegrateHit`, whose only forward caller is
`processHitParticle`. Every forward hit funnels through that one function:

- with `k_buffer_size: 0` (ArtiFixer's default), `HitParticleKBuffer<0>` is a specialization
  whose `full()` always returns true and whose `closestHit(h)` returns `h`, so `evalKBuffer`
  calls `processHitParticle` immediately on each hit;
- with `k_buffer_size > 0`, hits are buffered and flushed through the same function.

Separately, `RayPayload::idx` (`rayPayload.cuh:88`) is already the row-major pixel index
`x + W*y`, so it indexes a flattened `[H, W]` mask directly with no coordinate math.

FlashSplat's own change to the 3DGS rasterizer is three lines
(`forward.cu:308` and `:398` upstream). This port is the same three lines plus plumbing.

## 3. The rasterizer copy

**Location:** `flashsplat/threedgut_flashsplat_tracer/`
**Copied from:** `thirdparty/3DGRUT-ArtiFixer/threedgut_tracer` @ `62e1038b74b2edc01440fd4ddf5f080109b6faba`
(the commit artifixer pins the submodule to).
**Builds as:** `lib3dgut_flashsplat_cc`, a *separate* torch JIT extension.

Note for resyncing: `threedgut_tracer` is byte-identical between the pinned commit and the
current 3DGRUT-ArtiFixer branch tip (`6e91363`), verified with
`git diff --stat 62e1038 6e91363 -- threedgut_tracer`. Other parts of the submodule do
differ between those commits (see §6), so check this before assuming a pin bump is a no-op
for the copy.

The rename matters: the stock extension is `lib3dgut_cc`. Different names mean different
`torch.utils.cpp_extension` build directories and different generated `threedgutSlang.cuh`,
so the two coexist and normal ArtiFixer training/rendering is bit-identical.

`flashsplat/threedgut_flashsplat_tracer/README.md` holds the per-file delta table and the
resync procedure. Summary of the nine changed files:

| File | Change |
| --- | --- |
| `include/3dgut/renderer/renderParameters.h:38` | New `FlashSplatAccumulator` POD: `{const int32_t* rayObjectId; float* contribution; uint32_t numParticles;}`. |
| `include/3dgut/kernels/cuda/renderers/gutKBufferRenderer.cuh:165` | **The hook.** See below. |
| ` " ` `:457` | Same hook in the balanced kernel; `particleIdx` hoisted out of an inner scope (`:373`) to be visible there. |
| ` " ` `:109, :185, :231, :312` | Accumulator threaded through `processHitParticle`, `eval`, `evalKBuffer`, `evalForwardNoKBufferBalanced`. |
| `include/3dgut/kernels/cuda/renderers/gutRenderer.cuh:83, :125` | `render` and `renderBalanced` kernels take the accumulator by value. `renderBackward` (`:231`) untouched. |
| `include/3dgut/renderer/gutRenderer.h`, `src/gutRenderer.cu:241` | `renderForward` gains two defaulted pointer params and builds the accumulator. |
| `include/3dgut/splatRaster.h`, `src/splatRaster.cpp` | `trace` (`:251`) and new `traceContrib` (`:270`) both delegate to a shared private `traceImpl` (`:174`). |
| `bindings.cpp:84` | Binds `trace_contrib`. |
| `setup_3dgut.py` → `setup_3dgut_flashsplat.py` | Extension renamed; tiny-cuda-nn and slang includes repointed at the submodule so those are not duplicated on disk. |
| `tracer.py:352` | Imports the renamed extension; adds `Tracer.accumulate_contribution`. |

### The hook

`gutKBufferRenderer.cuh:165`, in the forward branch of `processHitParticle`, immediately
after `hitWeight` is computed:

```cpp
if (flashSplat.contribution != nullptr) {
    const int32_t objId = flashSplat.rayObjectId[ray.idx];
    if (objId >= 0) {
        atomicAdd(&flashSplat.contribution[objId * flashSplat.numParticles + hitParticle.idx],
                  hitWeight);
    }
}
```

Three properties to preserve if you modify this:

- **Inert by default.** `trace` passes null pointers, the guard short-circuits, nothing
  changes. Only `traceContrib` supplies an accumulator.
- **Forward only.** It sits inside `if constexpr (Backward) {...} else {...}`'s else branch.
  Gradients are untouched; there is deliberately no backward twin.
- **Negative ids are skipped**, which is how a caller excludes pixels entirely.

## 4. The Python pipeline

All under `flashsplat/`, plus one CLI in `data_processing/`.

| File | Role |
| --- | --- |
| `flashsplat/masks.py` | `load_object_id_mask(path, h, w) -> int32 [h*w]`. **Must not** use `Image.convert("L")` (remaps palette indices), must not divide by 255 or threshold, and resizes with `Image.NEAREST` only — all three would corrupt object ids. Also `mask_path_for_frame`/`find_mask_path_for_frame` (stem matching; the latter returns `None` instead of raising, for sparse mask sets) and `infer_num_objects`. |
| `flashsplat/solver.py` | `multi_instance_opt(all_contrib [K+1, N], gamma) -> bool [K+1, N]`. Per object: stack `[total - obj, obj]`, `F.normalize(dim=0)`, add `gamma` to the background row, `argmax`. |
| `flashsplat/accumulate.py` | `load_model` (loads a checkpoint, **swaps in the FlashSplat tracer**), `build_test_dataloader`, `allocate_accumulator`, `accumulate_contributions` (the multi-view sweep; skips views without a mask, so `--mask_dir` may cover only a subset of views). |
| `flashsplat/export.py` | `segment_model`, `save_segmented_checkpoint`, `save_segmented_ply`, `save_overlay_renders`. |
| `data_processing/run_flashsplat_segmentation.py` | CLI tying it together, styled after `run_artifixer3d.py`. |

### Data flow

```
mask PNGs ──> masks.load_object_id_mask ──> int32 [H*W]
                                              │
checkpoint ──> accumulate.load_model ──> model (FlashSplat tracer swapped in)
                                              │
                    Tracer.accumulate_contribution  (per view)
                              │  SplatRaster.trace_contrib
                              │  renderForward  ->  render kernel  ->  processHitParticle
                              ▼
              contribution [K+1, N] float32, atomicAdd-accumulated in place
                              │
                    solver.multi_instance_opt(gamma)
                              ▼
                     labels [K+1, N] bool
                              │
        ┌─────────────────────┼──────────────────┬────────────────────┐
    labels.pt            ckpt.pt +           object.ply +         overlays/
   contribution.pt      ckpt_removed.pt      removed.ply
```

### Three design points worth knowing

- **The accumulator is caller-owned and accumulated in place across views.** It is allocated
  once by `allocate_accumulator` and passed into every `trace_contrib` call. This avoids a
  per-frame multi-GB allocation and a per-frame device-to-device add. If you change this,
  note that `[K+1, N]` float32 is `4·(K+1)·N` bytes — fine at K=16/N=3M (~200 MB), 5 GB at
  K=256/N=5M. `allocate_accumulator` raises above 4 GiB rather than OOMing opaquely.
- **`load_model` swaps `model.renderer` after construction.** `MixtureOfGaussians.__init__`
  unconditionally builds the stock `threedgut_tracer.Tracer`; there is no config hook to
  prevent it. In practice `lib3dgut_cc` is already in the torch extension cache from
  training, so this is a cache hit, not a rebuild.
- **`accumulate_contribution` bypasses `Tracer._Autograd`** and calls `trace_contrib`
  directly under `@torch.no_grad()`. FlashSplat is inference-only.

### Mask convention (a user decision, not an arbitrary one)

Masks are **multi-object integer-id PNGs in their own directory**: `0` = background,
`1..K` = object ids, negative = ignore. Binary segmentation is just `K=1`.

This deliberately does **not** reuse 3DGRUT's existing `<image>_mask.png` sidecars
(loaded inline at `dataset_colmap.py:598` in the pinned submodule, surfaced as
`Batch.mask`). Those are binary, are read with `.convert("L")` and a `> 0.5` threshold, and
already mean "invalid pixels to exclude from the training loss" — overloading them would be
ambiguous and would destroy object ids. `Batch.mask` is untouched by this work.

## 5. Usage

```bash
python data_processing/run_flashsplat_segmentation.py \
    --checkpoint  /path/to/ckpt_last.pt \
    --colmap_dir  /path/to/colmap_scene \
    --mask_dir    /path/to/object_id_masks \
    --num_objects 3 \
    --gamma       0.0 \
    --output_root /path/to/out \
    --outputs labels checkpoints ply overlays
```

`--num_objects` defaults to the largest id found in `--mask_dir`.
`--gamma` is FlashSplat's background bias in `[-1, 1]`: larger is more conservative
(tighter objects), negative grows them.
`--contribution out/contribution.pt` re-solves at a different `--gamma` **without
re-rendering**, which is the main reason the raw accumulator is saved.

Outputs, per object `i`:

```
out/contribution.pt          [K+1, N] float32, the raw accumulation
out/labels.pt                [K+1, N] bool
out/summary.json             counts, gamma, global_step
out/object_00i/ckpt.pt       segmented 3DGRUT checkpoint (loadable by Renderer.from_checkpoint)
out/object_00i/ckpt_removed.pt   the complement — object removal
out/object_00i/object.ply    segmented PLY
out/object_00i/removed.ply
out/overlays/*.png           colored overlay renders
```

## 6. Verification status — read this before trusting anything

### Verified (run, passing)

- `tests/test_flashsplat_solver.py` — 5 tests. Correct dominant-label assignment,
  `gamma` monotonicity (raising gamma may only shrink objects), unseen objects, rank check.
- `tests/test_flashsplat_masks.py` — 6 tests. Ids survive load verbatim (`7` stays `7`),
  nearest resize invents no intermediate ids, RGB masks use channel 0, stem matching.
- CLI argument parsing, and byte-compilation of every Python file added or changed.
- A synthetic end-to-end run: the solver recovers labels at 100% accuracy on 500 Gaussians
  with a known ground truth.
- The alpha-compositing identity the GPU test relies on, checked numerically:
  `Σ α·T == 1 − Π(1−α)`.

### NOT verified — no CUDA and no compatible Python on the machine this was written on

The dev machine is macOS with Python 3.9 and no GPU. `threedgrut/datasets/__init__.py`
uses `match`, so the entire 3DGRUT import chain fails on 3.9 before CUDA even matters.
ArtiFixer's Dockerfiles target Python 3.12. (ArtiFixer's own pre-existing
`tests/test_3dgrut_mask_resize.py` fails locally with the identical `SyntaxError`, which is
how this was confirmed to be environmental rather than a regression.)

So **the CUDA has never been compiled and the hook has never been executed.** Run these two
in the container (`docker build -f Dockerfile.cuda12 -t artifixer:cuda12 .`):

```bash
python tests/test_flashsplat_build.py     # slangc -> nvcc -> ld; asserts trace_contrib is bound

python tests/test_flashsplat_accumulation.py \
    --checkpoint <ckpt.pt> --colmap_dir <scene>
```

The second is the one that matters. It checks the hook against a physical invariant, so it
needs no reference implementation:

> For a single pixel, `Σ_hits α_i·T_i = 1 − Π(1−α_i) = 1 − T_final`. And `1 − T_final` is
> exactly what 3DGUT already writes as the alpha channel of its render.

So with an all-ones mask, `contribution[1].sum()` must equal `pred_opacity.sum()`. The test
also asserts `contribution[0] == 0` (nothing leaked into background), that an all-`-1` mask
accumulates nothing, and that **the RGB render is bit-identical with the hook enabled** —
the hook must not perturb rendering.

For non-regression, run `thirdparty/3DGRUT-ArtiFixer/tests/test_build_3dgut.py` — it must
still build the stock `lib3dgut_cc`. The submodule is untouched, so this is structurally
guaranteed, but it is cheap insurance that the two extensions do not collide.

**Do not use `tests/test_3dgrut_mask_resize.py` as a non-regression baseline — it is
already broken in this repository, independently of this work.** It imports
`load_mask_image_tensor` from `threedgrut.datasets.dataset_colmap`, but that helper does
not exist in the submodule commit artifixer pins (`62e1038`, 2026-06-09), where mask loading
is inline at `dataset_colmap.py:598` with no resize. The helper only exists on the
3DGRUT-ArtiFixer branch tip (`6e91363`, 2026-07-17), which the pin predates. So the test
raises `ImportError` on a clean `git submodule update --init` checkout. This is a
pre-existing mismatch between artifixer's test suite and its own submodule pin; nothing here
caused it and nothing here fixes it.

## 7. Known risks and gotchas

- **Blend ordering.** With `k_buffer_size: 0`, 3DGUT blends in tile-depth-key order, not
  exact per-ray sorted order. This does *not* invalidate the method: FlashSplat's linearity
  argument holds for whatever weights the renderer actually uses, and the hook accumulates
  the renderer's own `hitWeight`. Noted, not fixed.
- **Duplication.** ~7.8k lines of CUDA/slang are now copied. If the submodule pin moves,
  `diff -ru thirdparty/3DGRUT-ArtiFixer/threedgut_tracer flashsplat/threedgut_flashsplat_tracer`
  and re-apply the delta table in the tracer's README, then update the commit recorded there.
- **A latent upstream bug that had to be worked around.**
  `MixtureOfGaussians.copy_fields` (`model.py:786`) reads `feature_dim_increase_interval`
  unconditionally at `model.py:811`, but `__init__` only sets it when progressive training is on
  (`init_n_features < max_n_features`). ArtiFixer's configs always enable it
  (`base_gs.yaml`: `0 < 3`), so this never fires in practice — but it would crash every
  export path on a non-progressive config. `export.segment_model` fills in defaults.
  This is a pre-existing quirk of the submodule, not something introduced here.
- **`atomicAdd` contention** on the accumulator when many rays hit the same Gaussian with
  the same label. Expected to be minor next to the render itself. Measure before optimizing.
- **The balanced kernel path is untested.** `evalForwardNoKBufferBalanced` is only reached
  with `render.splat.fine_grained_load_balancing: true`, which no ArtiFixer config sets. The
  hook is implemented there for correctness, but nothing exercises it.

## 8. Complete file inventory

Added:

```
flashsplat/__init__.py
flashsplat/masks.py
flashsplat/solver.py
flashsplat/accumulate.py
flashsplat/export.py
flashsplat/README.md                              <- this file
flashsplat/threedgut_flashsplat_tracer/           <- vendored rasterizer copy (see its README)
data_processing/run_flashsplat_segmentation.py
tests/test_flashsplat_solver.py                   (CPU)
tests/test_flashsplat_masks.py                    (CPU)
tests/test_flashsplat_build.py                    (needs CUDA)
tests/test_flashsplat_accumulation.py             (needs CUDA + a checkpoint)
```

Modified:

```
THIRD-PARTY-NOTICES.md    <- two new entries: the rasterizer copy, and FlashSplat
```

Nothing else. In particular `thirdparty/3DGRUT-ArtiFixer` is unmodified and still on its
pinned commit.
