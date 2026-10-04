# Optional exact CUDA morphology backend

## Summary

Profiling the T3 Bubble detector showed that iterative morphological propagation,
not Hough accumulation, consumed 85.46% of aggregate `CircleIdentifier` worker
time. This change adds an optional CUDA implementation of that Boolean
fixed-point loop while leaving all floating-point image construction and all
time-dependent LPT decisions on the CPU.

The feature is disabled by default. Enable it with:

```text
-DOPENLPT_GPU_EXACT_MORPHOLOGY=ON
```

For CMake 3.24 and newer, the enabled build targets the native GPU by default.
A cluster build can select an explicit architecture, for example
`-DOPENLPT_CUDA_ARCHITECTURES=80` for an A100-class node. CPU-only builds do not
enable CUDA and retain the morphology implementation from PR #10.

## Exactness design

The CPU produces the same `Hd` double array. CUDA receives those bits unchanged
and evaluates only the original per-pixel predicate:

```text
neighbor > pixel
or
neighbor == pixel and previous_neighbor is false
```

Each output pixel is written by one CUDA thread. Neighbor conditions are joined
by Boolean OR, so their visit order cannot affect the result. The only atomic
operation is `atomicExch` on an integer “changed” flag used to decide whether
another fixed-point iteration is needed.

The CUDA path contains no floating-point addition, scatter-add, reduction,
interpolation, threshold change, FMA, or approximate comparison. In particular,
`chaccum` remains on the CPU because its floating-point accumulation order is
scientifically sensitive.

Two Boolean device buffers alternate current/next roles exactly as in PR #10.
The final Boolean image is copied back only after convergence, then the original
CPU `regionprops` and all later processing continue unchanged.

## Failure behavior and CPU compatibility

CUDA support is build-time optional and OFF by default. When enabled, every host
thread owns a reusable CUDA stream and device buffers. Allocation or execution
failure returns `false` before a result is accepted, and the caller executes the
complete CPU fixed-point loop from its original initialized state.

The backend prints its selected device and, at shutdown, call, iteration,
failure, and CPU-fallback evidence. This makes a silent partial GPU failure
visible in production logs.

## Validation dataset and conditions

- Dataset: `H:\20260612\T3`
- Inputs: original `camFile` and `imgFile` lists; VSC disabled
- Long gate: frames 0 through 500 inclusive (501 images)
- Runtime: same GCC 13.2/glibc compatibility environment on both sides
- Threads: 4
- Comparison: complete same-boundary result-file inventory and SHA-256

The CPU reference and CUDA candidate contained the same accepted optimization
stack. The only controlled difference was the optional CUDA morphology backend.

## Exactness result

- CUDA morphology calls: 277,832
- CUDA fixed-point iterations: 8,486,957
- CUDA failures / CPU fallbacks: 0
- Identical SHA-256 result files: 15 of 15
- Missing files: 0
- Unexpected files: 0
- Changed files: 0
- Numeric fallback or tolerance-based acceptance: not used

The two T3 frame-500 files retained in the completed archive also match exactly.
The archive later removed its active frame-500 snapshots, so those active files
were compared against the saved same-boundary CPU reference.

A fresh build of this PR was also replayed on frames 0 through 4 against its CPU
parent: all 15 result files matched exactly, with 3,800 CUDA calls, 255,072
fixed-point iterations, and zero failures. Its external wall time was 19.079 s
versus 37.872 s for the CPU parent (1.985x).

## Performance result

The external wall timer includes OpenLPT startup/configuration, 2,004 TIFF reads,
all computation, result writes, and shutdown.

| Frames 0--500 run | Total wall time | Wall time per frame |
| --- | ---: | ---: |
| Exact CPU compatibility reference | 1,600.813 s | 3.195 s |
| Exact CUDA morphology | 684.091 s | 1.365 s |

- Saved wall time: 916.722 seconds over 501 images
- Wall-time reduction: 57.27%
- Throughput speedup: 2.340x

An earlier 100-frame gate was also 15/15 exact and improved from 399.608 to
150.224 seconds (2.660x), with 62,440 CUDA calls, 1,822,175 iterations, and zero
fallbacks.

## Scope and limitations

- This pull request is stacked on and depends on PR #10.
- Only Boolean morphology propagation is moved to CUDA.
- Hough accumulation, input normalization, median filtering, `imhmax`,
  `regionprops`, IPR, StereoMatch, Shake, linking, and serialization remain CPU.
- One `Hd` transfer and one converged Boolean-result transfer occur per
  morphology call; intermediate iterations stay on the GPU.
- CUDA device 0 is currently selected. Multi-GPU placement is outside this PR.
- Rockfish performance and architecture selection must be confirmed on its GPU
  node type; exact-output acceptance remains mandatory there.
