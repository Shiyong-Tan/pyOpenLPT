# Residual-image row tiling

## Summary

`Shake::calResidueImage` originally parallelized only across cameras. With four
cameras, that limited this stage to four independent tasks even when more CPU
workers were available. This change divides each active camera into fixed,
non-overlapping 64-row tiles and schedules the complete tile list with OpenMP.

Before the parallel loop, the code calculates each accepted object's projected
ROI once per camera. Each tile then applies only the objects whose ROI intersects
its row range.

## Why the output is unchanged

The original residual update for one pixel is:

```text
residual = min(residual, original - object_projection)
```

This change preserves the original object order inside every pixel. It does not
parallelize multiple object contributions to the same pixel and does not perform
a floating-point reduction across workers. Each `(camera, row)` belongs to
exactly one tile, so worker writes are disjoint.

The following behavior is unchanged:

- object and flag filtering;
- camera activation rules;
- ROI calculation and image-bound clamping;
- `project2DInt` calls and their arguments;
- per-pixel comparison and negative-value clamp;
- residual-image storage and downstream processing.

The fixed 64-row geometry is intentional. A tested adaptive partition produced
identical output but did not improve the local four-thread benchmark, so it is
not part of this change.

## Validation dataset and conditions

- Dataset: `H:\20260612\T3`
- Inputs: original `camFile` and `imgFile` lists; VSC disabled
- Frames: 0 through 199 inclusive (200 frames)
- Runtime: isolated WSL build, GCC 11.4, Release configuration
- Threads: 4
- Comparison: complete result-file inventory and SHA-256 for every file

Both benchmark executables included the previously validated BubbleRef resize
cache. The only controlled difference between them was residual-image row
tiling, which isolates the incremental effect of this change.

## Exactness result

Both runs completed frame 199 and produced the same 15 result files.

- Identical SHA-256 files: 15 of 15
- Missing files: 0
- Unexpected files: 0
- Changed files: 0
- Numeric fallback or tolerance-based acceptance: not used

## Performance result

| Build | Total wall time | Wall time per frame |
| --- | ---: | ---: |
| Camera-parallel baseline | 1516.147 s | 7.581 s |
| Fixed 64-row tiling | 1466.497 s | 7.332 s |

- Saved wall time: 49.650 seconds over 200 frames
- Wall-time reduction: 3.27%
- Throughput speedup: 1.034x

These measurements describe the tested four-thread machine. The main structural
benefit is exposing more independent work than the camera count; scaling on a
24- or 48-core Rockfish allocation still requires a representative cluster test.

## Scope and limitations

- This change affects residual-image construction only.
- It does not change IPR iteration count, 2D detection, StereoMatch, Shake,
  linking, VSC, BubbleRef generation, output files, or GUI behavior.
- It does not include adaptive tiling, per-tile detection caching, or the later
  fixed-geometry 2D scheduler.
