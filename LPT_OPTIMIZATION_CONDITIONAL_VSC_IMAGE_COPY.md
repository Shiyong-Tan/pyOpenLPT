# Conditional VSC image copy

## Summary

At the start of every convergence frame, `STB::runConvPhase` deep-copied the
complete camera image list for possible Volume Self Calibration (VSC) use.
Those copied images are consumed only on a frame that actually performs VSC
accumulation.

This change computes the existing VSC accumulation decision once, before any
operation that can modify the input images, and copies the original images only
when that decision is true. The same Boolean controls the later VSC call.

## Why the output is unchanged

When VSC is skipped, `img_orig` has no consumer. Omitting its allocation and
copy cannot affect image processing, reconstruction, tracking, or output.

When VSC accumulation is enabled in the future, the copy remains at the start
of `runConvPhase`, before Shake or residual-image work. VSC therefore receives
the same original pixels as before. The change does not alter the existing VSC
eligibility expression, accumulation interval, calibration state, or optimizer.

The current upstream branch retains its existing `skip_vsc = true` behavior;
this change deliberately does not repair or redesign that separate policy.

## Exact-output validation

A controlled cache/copy A/B used:

- Dataset: `H:\20260612\T3`
- Inputs: original `camFile` and `imgFile` lists; VSC disabled
- Frames: 0 through 99 inclusive (100 frames)
- Runtime: isolated WSL build, GCC 11.4, Release configuration
- Threads: 4
- Comparison: complete result-file inventory and SHA-256 for every file

Both runs completed frame 99 and produced the same 15 result files.

- Identical SHA-256 files: 15 of 15
- Missing files: 0
- Unexpected files: 0
- Changed files: 0
- Numeric fallback or tolerance-based acceptance: not used

## Measured copy cost

The exact OpenLPT `Image` type was benchmarked with four 800x1280 double
images. Each deep copy moves 32,768,000 bytes. Three 500-copy trials measured:

- 9.828 ms per copy;
- 9.065 ms per copy;
- 9.097 ms per copy.

The mean direct cost was approximately 9.33 ms per convergence frame. The
100-frame whole-run pair differed by 0.31% in the opposite direction, which is
normal run-to-run noise and is not attributed to this change. The optimization
removes measured work but its overall impact is small on this dataset.

## Scope and limitations

- This change affects the convergence-frame VSC image copy and its matching
  eligibility branch only.
- VSC-on numerical equivalence was not run because VSC remains disabled in the
  current upstream control flow.
- No VSC algorithm, camera, image value, IPR, Shake, linking, serialization, or
  GUI behavior is changed.
