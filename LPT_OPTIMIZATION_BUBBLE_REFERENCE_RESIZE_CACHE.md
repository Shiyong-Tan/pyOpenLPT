# Bubble reference resize cache

## Summary

Bubble-mode shaking repeatedly resizes the same static reference image to the
same odd template sizes. This change caches each resize by `(camera, npix)` in
`BubbleShakeStrategy` and reuses the immutable result in both
`calShakeResidue` and `additionalObjectCheck`.

The cache does not include images derived from a current object ROI. In
particular, the small-radius upsample in `selectShakeCam` remains unchanged and
uncached because its input is dynamic.

## Why the output is unchanged

For a fixed `BubbleShakeStrategy`, all inputs to the cached operation are
constant for a given key:

- the camera's `BubbleRefImg` image;
- the requested odd output size, `npix`;
- the camera's maximum intensity.

The first request for a key calls the existing `BubbleResize::ResizeBubble`
implementation. Later requests receive the stored `const Image`; no
interpolation, correlation, reduction, threshold, object ordering, or output
serialization logic is changed. A shared mutex protects concurrent reads and
serializes cache misses. The double-checked lookup also prevents duplicate
first-use work when object workers request the same key concurrently.

Defining `OPENLPT_DISABLE_BUBBLE_REF_CACHE` restores the original behavior and
was used for the controlled A/B test.

## Validation dataset and conditions

- Dataset: `H:\20260612\T3`
- Inputs: original `camFile` and `imgFile` lists; VSC disabled
- Frames: 0 through 500 inclusive (501 frames)
- Runtime: isolated WSL build, GCC 11.4, Release configuration
- Threads: 4
- Resume: disabled
- Comparison: complete result-file inventory and SHA-256 for every file

Both test executables were built from the same source and settings. The only
controlled difference was whether `OPENLPT_DISABLE_BUBBLE_REF_CACHE` was
defined.

## Exactness result

Both runs completed frame 500 and produced 15 result files.

- Identical SHA-256 files: 15 of 15
- Missing files: 0
- Unexpected files: 0
- Changed files: 0
- Numeric fallback or tolerance-based acceptance: not used

The generated `BubbleRefImg_0.tif` through `BubbleRefImg_3.tif` files were also
byte-for-byte identical to the corresponding T3 reference images.

This establishes exact serialized-output equivalence between cache-disabled
and cache-enabled builds in the controlled environment. It does not claim that
an independently compiled historical cluster executable is bitwise equivalent
to the local GCC 11.4 executable.

## Performance result

| Build | Total wall time | Wall time per frame |
| --- | ---: | ---: |
| Cache disabled | 4050.481 s | 8.085 s |
| Cache enabled | 3719.600 s | 7.424 s |

- Saved wall time: 330.882 seconds over 501 frames
- Wall-time reduction: 8.17%
- Throughput speedup: 1.089x

These numbers describe this dataset and local four-thread environment; other
object-size distributions and machines may produce different gains.

## Scope and limitations

- Bubble mode only; tracer behavior is unchanged.
- Cache lifetime is one `BubbleShakeStrategy` instance.
- The number of entries is bounded by the number of cameras multiplied by the
  number of encountered odd template sizes.
- No VSC calculation, BubbleRef generation, 2D detection, residual-image
  update, tracking, or GUI behavior is changed.
