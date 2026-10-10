# Active-track repeat search with a KD-tree

## Summary

`STB::checkRepeat` rejects a newly reconstructed object when it is close enough
to the endpoint of an active long track. The original implementation scanned
every active endpoint for every new object.

This change builds one immutable 3D KD-tree over the active endpoints per call
and queries the nearest endpoint for each new object. It changes the search
complexity from a full object-by-endpoint scan to one tree build plus nearest-
neighbor queries.

## Why the decision is unchanged

For each new object, the acceptance radius is:

```text
StereoMatch 3D tolerance + that new object's bubble radius
```

The radius depends on the query object but not on the active-track endpoint.
Therefore, an object is repeated if and only if its nearest endpoint satisfies
the existing inclusive threshold.

The KD-tree is used only to select that nearest endpoint. The final displacement
and Euclidean norm are recomputed using the original `Pt3D` subtraction and
`norm()` expression before applying the unchanged `<=` comparison. This avoids
using the tree's internal squared-distance value for the scientific decision.

The endpoint vector and tree are immutable during the OpenMP query loop.
Defining `OPENLPT_DISABLE_CHECK_REPEAT_KDTREE` restores the original linear scan
and was used for controlled validation.

## Validation dataset and conditions

- Dataset: `H:\20260612\T3`
- Inputs: original `camFile` and `imgFile` lists; VSC disabled
- Frames: 0 through 199 inclusive (200 frames)
- Runtime: isolated WSL build, GCC 11.4, Release configuration
- Threads: 4
- Comparison: complete result-file inventory and SHA-256 for every file

Both executables included the previously validated BubbleRef resize cache and
fixed residual-image row tiling. The only controlled difference was the
`checkRepeat` search implementation.

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
| Linear endpoint scan | 1466.497 s | 7.332 s |
| KD-tree nearest endpoint | 1406.249 s | 7.031 s |

- Observed wall-time reduction: 4.11%
- Observed throughput speedup: 1.043x

This whole-run pair was not interleaved, and `checkRepeat` was below the
existing log's 0.01-second resolution on this relatively sparse sample. The
full timing difference must therefore not be attributed solely to this change.
The asymptotic benefit should be larger for trials with hundreds or thousands
of active endpoints, while a small-count linear path may be preferable after a
focused crossover benchmark.

## Scope and limitations

- This change affects `STB::checkRepeat` only.
- `Shake::markRepeatedObj` already used nanoflann and is not modified.
- No tracking threshold, distance expression, flag, IPR, VSC, BubbleRef, image,
  serialization, or GUI behavior is changed.
