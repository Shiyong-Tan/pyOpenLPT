# Exact halo-expanded tile detection cache

## Summary

PR #8 gives every Bubble IPR camera/tile job a stable identity and preserves its
legacy core and halo geometry. This follow-up stores each tile's complete
halo-expanded input image together with its ordered raw circle detections.

Before rerunning the generated detector, the cache compares the current tile
input with the saved snapshot. A hit reuses the ordered raw detections. A miss
runs the original `CircleIdentifier` and replaces that tile's snapshot.

Unlike the whole-camera generation cache, this optimization can reuse unaffected
parts of a residual image after accepted bubbles change pixels in other tiles.

## Exact cache key

A cache hit requires all of the following:

- identical camera and fixed tile index;
- identical halo-expanded crop bounds;
- identical image dimensions;
- bitwise-identical `radius_min`, `radius_max`, and `sense` values;
- `memcmp` equality for every row and every `double` pixel in the complete
  halo-expanded detector input.

There is no image hash, probability of collision, motion threshold, or numeric
tolerance. Even a one-bit pixel change inside the detector's halo causes a miss.

## Why the output is unchanged

On a cache miss, the original detector and merge path run unchanged. On a hit,
the detector would receive exactly the same array shape, pixel bit patterns, and
parameter bit patterns as the saved invocation. The cached list is stored before
camera-level sorting/deduplication and preserves its vector order. Results from
all jobs are still merged in fixed camera/tile order before the unchanged metric
sort and duplicate removal.

The cache entry is marked invalid while being replaced and each camera/tile has
one unique job, so workers never share a writable entry.

## Validation dataset and conditions

- Dataset: `H:\20260612\T3`
- Inputs: original `camFile` and `imgFile` lists; VSC disabled
- Frames: 0 through 4 inclusive (5 frames)
- Runtime: isolated WSL build, GCC 11.4, Release configuration
- Threads: 4
- Comparison: complete result-file inventory and SHA-256 for every file

Both executables used the fixed-geometry scheduler from PR #8. The only
controlled difference was the per-tile cache.

## Exactness result

Both runs produced the same 15 result files.

- Identical SHA-256 files: 15 of 15
- Missing files: 0
- Unexpected files: 0
- Changed files: 0
- Numeric fallback or tolerance-based acceptance: not used

A second clean branch replay was performed directly for these pull requests:
PR #8's scheduler-only branch took 56.868 seconds and this cache branch took
32.374 seconds. Their 15 files were exactly identical. That short fresh pair
observed a 1.757x speedup (43.06% less wall time); the earlier accepted-stack
measurement below is retained as the more conservative performance claim.

## Performance result

| Build | Total wall time | Wall time per frame |
| --- | ---: | ---: |
| Fixed-geometry scheduler | 33.429 s | 6.686 s |
| Exact halo-expanded tile cache | 27.006 s | 5.401 s |

- Saved wall time: 6.423 seconds over 5 frames
- Wall-time reduction: 19.21%
- Throughput speedup: 1.238x

These short-run timings were measured on the accepted experimental stack. The
structural benefit depends on how spatially localized residual changes are; a
trial that changes every halo-expanded tile on every IPR iteration will see
little benefit beyond exact-comparison overhead.

## Scope and limitations

- This pull request is stacked on and depends on PR #8.
- It affects Bubble detection through the fixed-batch path only.
- Tracer detection and VSC remain on `findObject2D` and do not use this cache.
- It does not change tile geometry, detector mathematics, residual construction,
  StereoMatch, Shake, linking, or serialization.
- It does not include the independent whole-camera generation cache, morphology
  buffer reuse, GPU work, or adaptive tiling.
