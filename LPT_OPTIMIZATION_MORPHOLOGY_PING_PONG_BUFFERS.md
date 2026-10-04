# Morphology ping-pong buffers

## Summary

The generated circle detector contains an iterative morphological propagation
loop. Each iteration previously made two full boolean-image copies:

1. copy the current result into `bwpre` for convergence testing;
2. copy the current result into a temporary parameter object so `process2D`
   could read an immutable previous state while overwriting the output.

This change preallocates two same-sized boolean arrays and alternates their
current/next roles with pointer swaps. `process2D` reads the current buffer and
writes the other buffer. At convergence, at most one final assignment is needed
when the result resides in the secondary array.

## Why the output is unchanged

The old iteration can be written as:

```text
previous = current
current = process2D(Hd, previous)
continue while previous != current
```

The new iteration is:

```text
next = process2D(Hd, current)
compare current with next
swap current and next only when another iteration is required
```

For every iteration, `process2D` receives the same immutable boolean values in
the same linear layout and produces the next state in the same pixel traversal
order. The full-array convergence comparison also traverses the same indices in
the same order. Therefore:

- the neighborhood definition is unchanged;
- border and interior handling are unchanged;
- input floating-point values are unchanged;
- the number of fixed-point iterations is unchanged;
- no floating-point expression or reduction is reordered;
- the final boolean image passed to `regionprops` is unchanged.

This is storage reuse only. It does not use an early-exit approximation or a
weaker convergence flag.

## Validation dataset and conditions

- Dataset: `H:\20260612\T3`
- Inputs: original `camFile` and `imgFile` lists; VSC disabled
- Frames: 0 through 99 inclusive (100 frames)
- Runtime: isolated WSL build, GCC 11.4, Release configuration
- Threads: 4
- Comparison: complete result-file inventory and SHA-256 for every file

Both controlled executables contained the accepted exact unchanged-residual 2D
cache. The only controlled difference was morphology buffer management.

## Exactness result

Both runs produced the same 15 result files.

- Identical SHA-256 files: 15 of 15
- Missing files: 0
- Unexpected files: 0
- Changed files: 0
- Numeric fallback or tolerance-based acceptance: not used

The pull-request branch was additionally built cleanly from upstream `main` and
replayed over frames 0--4. It completed in 37.872 seconds wall time and its
15 files matched the accepted same-range reference exactly. This fresh absolute
time is recorded as a build/replay check; the controlled cache-plus-morphology
pair below is the performance evidence.

## Performance result

| Build | Total wall time | Wall time per frame |
| --- | ---: | ---: |
| Accepted 2D cache | 375.330 s | 3.753 s |
| Ping-pong morphology buffers | 365.240 s | 3.652 s |

- Saved wall time: 10.090 seconds over 100 frames
- Wall-time reduction: 2.69%
- Throughput speedup: 1.028x

The preceding 20-frame gate was also exact and improved from 97.055 to 93.597
seconds (1.037x). Profiling found iterative morphology to be 85.46% of aggregate
`CircleIdentifier` worker time before this storage change, which is why removing
full-array copies matters despite leaving the detector mathematics untouched.

## Scope and limitations

- The change affects only the generated morphology fixed-point loop used by
  `CircleIdentifier`.
- It does not modify Hough accumulation, median filtering, `imhmax`,
  `regionprops`, detector thresholds, IPR, StereoMatch, Shake, or linking.
- The generated per-column temporary array remains. A tested direct-write
  follow-up was exact but 0.58% slower over 20 frames and was rejected.
- No GPU code, profiling instrumentation, convergence shortcut, or direct
  interior-write experiment is included in this pull request.
