# Connected, refraction-aware view-volume estimation

## Problem

The previous GUI estimator repeatedly resized an approximate search box and
returned the minimum and maximum of every visible grid point. On T3/T4 it could
collapse onto one thin sampled layer and round all three axes to `[-5, 5]` mm.
That box excludes valid working volume before LPT starts.

PINPLATE cameras added a second risk: when exact projection failed, the old
helper silently fell back to `cv2.projectPoints`, which ignores the calibrated
refractive model.

## Design

The replacement keeps volume selection deterministic and conservative:

1. camera poses and intrinsics define only a generous coarse search envelope;
2. every PINPLATE sample is projected through pyOpenLPT's exact
   `projectBatchStatus` implementation, in bounded chunks;
3. visibility is counted per voxel instead of immediately intersecting every
   camera;
4. the required count matches reduced-camera IPR (`nCam - nReducedCam`);
5. six-connected component labeling removes isolated/disconnected visible
   islands;
6. live reconstructed calibration points select the relevant component when
   available, followed by the robust optical center and component size;
7. collapsed coarse or fine components are rejected, preserving the user's
   existing manual values; and
8. a fine scan, half-cell safety margin, and outward 5 mm rounding produce the
   recommended GUI bounds.

Exact PINPLATE projection failures abort automatic estimation. The GUI never
silently substitutes a lower-fidelity model.

This PR is stacked on the X-span/voxel-scale synchronization PR. Once the
bounds are accepted, `Voxel to MM` is updated to `(xmax - xmin) / 1000`.

## Validation

Using the LPT Python environment and its compiled exact PINPLATE projector:

| Input cameras | Recommended bounds (mm) | Voxel to MM |
| --- | --- | ---: |
| T3 original | `[-20, 15] x [-10, 10] x [-15, 15]` | 0.035 |
| T4 original | `[-20, 15] x [-10, 10] x [-15, 15]` | 0.035 |
| T3 VSC | `[-15, 15] x [-10, 10] x [-15, 15]` | 0.030 |

All three replace the invalid `[-5, 5]` cube and maintain exact X-span/scale
consistency. A synthetic two-island grid also confirmed that a component
containing calibration seed points is selected over an otherwise equivalent
disconnected island. Python syntax compilation passed.

This changes GUI configuration generation only. It does not modify the LPT
reconstruction, Shake, linking, or serialization algorithms.
