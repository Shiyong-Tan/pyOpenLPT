# Fixed-geometry Bubble IPR 2D scheduler

## Summary

Bubble IPR originally scheduled one OpenMP task per active camera and then ran
the generated circle detector inside that camera task. With four cameras, the
outer scheduling layer exposed at most four independent jobs. The detector also
created its own nested parallel region, making higher thread counts prone to
idle workers or oversubscription rather than useful scaling.

This change preserves the established Bubble detector geometry but flattens all
camera/tile work into one OpenMP queue. For the validated 1280-by-800 images,
each camera retains its existing 2-by-2 grid and halo, yielding 16 independent
jobs across four cameras. Each job still invokes the original
`CircleIdentifier::BubbleCenterAndSizeByCircle` implementation.

## Why the output is unchanged

The scheduler does not alter any scientific operation or detector parameter:

- core and halo dimensions use the existing normal-IPR geometry;
- tile crop bounds are unchanged;
- radius range, sensitivity, metric threshold, and duplicate thresholds are
  unchanged;
- each tile runs the same generated circle detector;
- detections are stored in fixed camera/tile slots and merged in the legacy
  tile order, independent of worker completion order;
- the original metric sort, duplicate test, and `Bubble2D` construction are
  retained.

Only ownership of the already-defined tile jobs changes. No floating-point
reduction is reordered across tiles, and no tolerance-based comparison or
numeric fallback is introduced.

## Validation dataset and conditions

- Dataset: `H:\20260612\T3`
- Inputs: original `camFile` and `imgFile` lists; VSC disabled
- Frames: 0 through 4 inclusive (5 frames)
- Runtime: isolated WSL build, GCC 11.4, Release configuration
- Configured thread counts: 4, 8, and 12
- Comparison: complete result-file inventory and SHA-256 for every file

## Exactness result

Every tested thread count produced the same 15 output files as the accepted
baseline.

| Configured threads | Result files | Exact SHA-256 files | Changed files |
| ---: | ---: | ---: | ---: |
| 4 | 15 | 15 | 0 |
| 8 | 15 | 15 | 0 |
| 12 | 15 | 15 | 0 |

Numeric fallback or tolerance-based acceptance was not used.

The pull-request branch was also built cleanly from upstream `main` and replayed
over the same five frames at four threads. Its 15 files matched both the
original-scheduler reference and the earlier fixed-geometry reference exactly.
That fresh isolated run took 56.868 seconds wall time. This absolute time is
reported for reproducibility, not compared with the table below, because those
earlier scaling executables contained the previously accepted optimization
stack.

## Scaling result

| Configured threads | Total wall time | Exact result |
| ---: | ---: | --- |
| 4 | 33.429 s | byte-for-byte identical |
| 8 | 44.449 s | byte-for-byte identical |
| 12 | 63.255 s | byte-for-byte identical |

These controlled scheduler-only scaling measurements were made on top of the
accepted optimization stack. The scheduler is not a standalone speed win on the
tested computer and does not move the local optimum above four threads.
Generated detector code and later Shake work still create nested teams, so 8-
and 12-thread runs oversubscribe this machine. Removing repeated nested teams is
a separate optimization and is intentionally not included here.

The value of this change is structural: it makes the fixed 16-job Bubble IPR
work graph explicit, removes dependence on camera-level scheduling, and provides
the deterministic tile identity required by the separately validated exact
per-tile residual cache.

## Scope and limitations

- This path is used only for Bubble 2D detection during normal IPR.
- Tracer detection and VSC continue through the existing `findObject2D` path.
- Direct callers of `findObject2D` are unchanged.
- The change does not alter tile size, halo size, image preprocessing, the
  generated circle detector, StereoMatch, Shake, linking, or serialization.
- No per-tile cache, unchanged-image cache, GPU work, or morphology change is
  included in this pull request.
- Exactness and geometry were validated on the T3 four-camera 1280-by-800
  workload. Other camera/image configurations should receive the same exact
  regression gate before production use.
