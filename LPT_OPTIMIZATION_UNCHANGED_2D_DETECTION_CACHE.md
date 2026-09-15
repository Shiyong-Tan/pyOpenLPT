# Exact unchanged-residual 2D detection cache

## Summary

One IPR call performs several full- and reduced-camera iterations on a residual
image. Some iterations accept no new 3D objects, so the residual pixels entering
the next iteration are exactly unchanged. The original code nevertheless reran
the complete 2D circle detector for every active camera.

This change caches each camera's complete, ordered, unfiltered Bubble 2D
detection list while the residual image remains in the same exact generation.
An iteration that accepts at least one object conservatively advances the global
generation and invalidates every camera entry. If it accepts no objects, the
generation remains unchanged and eligible cameras reuse deep clones of their
previous detections.

## Why the output is unchanged

This is an exact reuse rule, not an approximate image-similarity test:

- a cache entry is valid only inside one `IPR::runIPR` call;
- the generation changes after every non-empty accepted object list;
- a changed active subset cannot make stale data valid after an accepted update;
- invalidating every camera is conservative, including cameras that were
  inactive during a reduced-camera iteration;
- cached detections are deep-cloned, so downstream shuffling, truncation, and
  ownership transfer cannot mutate the saved list;
- the existing deterministic object-limit step is reapplied on every iteration;
- detector order, StereoMatch tolerances, Shake, residual construction, and
  every scientific threshold are unchanged.

When an iteration accepts no objects, OpenLPT either returns before residual
construction or constructs a byte-identical residual from an empty accepted
list. Reusing the already computed 2D result therefore replaces only redundant
work on identical inputs.

The cache is limited to Bubble detection. Tracer behavior remains on the
original path.

## Validation dataset and conditions

- Dataset: `H:\20260612\T3`
- Inputs: original `camFile` and `imgFile` lists; VSC disabled
- Frames: 0 through 99 inclusive (100 frames)
- Runtime: isolated WSL build, GCC 11.4, Release configuration
- Threads: 4
- Comparison: complete result-file inventory and SHA-256 for every file

The controlled performance pair contained the same previously accepted
optimization stack. The only difference was this unchanged-residual detection
cache.

## Exactness result

Both 100-frame runs produced the same 15 result files.

- Identical SHA-256 files: 15 of 15
- Missing files: 0
- Unexpected files: 0
- Changed files: 0
- Numeric fallback or tolerance-based acceptance: not used

The pull-request branch was additionally built cleanly from upstream `main` and
replayed over frames 0--4. It completed in 25.910 seconds wall time and its
15 files matched the accepted same-range reference exactly. This fresh absolute
time is recorded as a build/replay check; the controlled 100-frame pair below is
the performance evidence.

## Performance result

| Build | Total wall time | Wall time per frame |
| --- | ---: | ---: |
| Accepted pre-2D stack | 710.941 s | 7.109 s |
| Exact unchanged-residual cache | 375.330 s | 3.753 s |

- Saved wall time: 335.611 seconds over 100 frames
- Wall-time reduction: 47.2%
- Throughput speedup: 1.894x

A preceding 20-frame gate was also exact and improved from 155.424 to 100.501
seconds (1.546x). The larger 100-frame benefit reflects how often later IPR
iterations encounter unchanged residual inputs in this dataset.

## Scope and limitations

- This cache operates within one IPR call; it does not reuse detections across
  frames.
- It does not assume that a bubble moved only slightly. Any accepted object
  invalidates the cache without comparing motion magnitude.
- It does not skip StereoMatch, Shake, linking, or residual updates.
- It does not cache changed images, use hashes, or allow hash collisions.
- It does not alter VSC, BubbleRef generation, detector geometry, morphology,
  output serialization, or GUI behavior.
- Workloads whose every IPR iteration accepts objects will see little or no
  benefit beyond small cache bookkeeping costs.
