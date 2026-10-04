# Safe BubbleRef bootstrap within the normal IPR tolerance schedule

## Problem

Bubble tracking needs one reference appearance image per camera before Shake can
start. The original code attempted to build those images only from the first
strict full-camera StereoMatch. A valid, separately checked VSC calibration can
make that first match stricter; if it yields too few large isolated bubbles,
OpenLPT aborts before the later IPR passes get the chance to use their normal
larger 2D tolerances.

## Design

The first strict match remains unchanged and remains the only scientific object
list returned by this IPR pass. If it cannot construct BubbleRef, OpenLPT:

1. follows the configured IPR `1.5x` 2D-tolerance schedule;
2. uses each relaxed match list only as an auxiliary BubbleRef candidate pool;
3. requires at least 12 radius-qualified candidates;
4. requires qualifying candidates to occupy at least two image quadrants in
   every active camera;
5. accepts the first template that passes all existing crop, intensity, and
   per-pixel coverage checks; and
6. discards every auxiliary 3D object before normal Shake and tracking resume.

The configured strict tolerance is restored after every attempt, including the
exception path. The relaxed list therefore cannot enter Shake, StereoMatch
output, track creation, linking, or serialization.

## Diagnostics

BubbleRef now retains a human-readable reason for its last failed construction
attempt. Fatal startup errors consequently identify whether the failure was
camera alignment, candidate count, radius, crop isolation, intensity quality,
or per-pixel coverage rather than reporting only `Cannot obtain bubble reference
image`.

## Correctness boundary

For datasets where the strict pass already creates BubbleRef, the generated
reference images and every later scientific operation are unchanged. For a run
that previously aborted, the fallback deliberately creates only the missing
appearance template; it does not promote relaxed matches into scientific data.

VSC quality validation remains a separate prerequisite. This fallback is not a
mechanism for accepting an unvalidated or poor calibration.

## Validation

- Clean GCC 11.4 WSL build: passed.
- T3 frames 0--4, four threads, original cameras: all 15 result files were
  byte-for-byte identical to pristine OpenLPT v2.2.4. The input signatures also
  matched; no numeric fallback was used.
- T3 frame 0 with the strict 2D tolerance deliberately reduced to 2.0 px:
  strict matching produced zero objects, the 3.0 px and 4.5 px bootstrap
  populations were rejected, and the normal 6.75 px schedule step supplied 18
  candidates and created BubbleRef. Those candidates were then discarded; the
  scientific IPR loop continued from the original zero-object strict result.
- T3 frame 0 with T3's recorded VSC cameras: the normal strict path created
  BubbleRef and completed without invoking the fallback.
