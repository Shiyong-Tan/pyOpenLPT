# Sampled VSC quality gate

## Purpose

This change adds a read-only quality gate for virtual self-calibration (VSC)
camera files before they are used by LPT.  It evaluates a small, reproducible
sample of frames rather than requiring a second full LPT run.

The gate does not modify the VSC optimizer, camera model, image detector, IPR,
Shake, tracking, or any scientific output.  It only measures an already-created
set of `vsc_cam*.txt` files and writes a JSON audit report.

## Validation modes

The preferred mode uses either a different acquisition or frames known not to
have contributed to the VSC fit.  When the fitting range is unknown or no
held-out frames are available, callers may explicitly request a current-data
self-check.  That mode is recorded as `current_data_self_check` and is always
reported as **not independent**.

The production LPT range may still start at frame zero.  Only the provenance of
the validation sample determines whether the validation is independent.

## Reproducible frame selection

The default production workflow requests 30 frames.  The random seed is derived
from the SHA-256 hashes of the VSC camera files, so unchanged VSC files select
the same frames on every run.  For same-acquisition held-out validation, all
candidate observations are filtered by frame before sampling.

Track observations are read from `LongTrackInactive_*.csv`.  Short or sparse
runs that do not yet have inactive long-track snapshots can use
`LongTrackActive_*.csv` and `ExitTrack_*.csv` instead.

Every sampled observation is re-detected in the original camera images.  Stored
2D centers are not accepted as a substitute for this image-domain check.

## Required gates

The report passes only when all of the following are true:

- VSC camera hashes match the provenance record.
- At least 50 clean multi-camera observations survive re-detection.
- Every camera covers at least 12 cells of a 4 x 4 image-space grid.
- Pooled leave-one-camera-out reprojection RMSE improves over the original
  cameras.
- Pooled leave-one-camera-out reprojection P95 improves.
- No individual camera's P95 is more than 1.10 times its original-camera P95.
- The fitted rays produce no invalid refraction-barrier intersections.

The original calibration is therefore used as a comparator, not as an assumed
ground truth.  VSC is allowed and expected to outperform it.

## Audit and invalidation

The JSON report records the exact frames used, sampling seed, dataset
relationship, eligible ranges, detection statistics, spatial coverage, all
metrics, and every gate result.  It also stores SHA-256 hashes for the VSC
camera files and the provenance manifest.

LPT preflight can call `verify_saved_validation()` before accepting VSC files.
Changing either the VSC cameras or their provenance invalidates the saved report
and requires validation again.  Log-friendly audit lines clearly distinguish
an independent validation from a current-data self-check.

## T3 verification

The implementation was exercised on `H:\\20260612\\T3` using a deterministic
30-frame current-data self-check.  It found 284 clean observations and covered
at least 15 of 16 spatial bins in every camera.

| Metric | Original cameras | VSC cameras |
| --- | ---: | ---: |
| Pooled leave-one-camera-out RMSE | 4.2536 px | 3.6297 px |
| Pooled leave-one-camera-out P95 | 7.6477 px | 6.6590 px |

All required gates passed.  The validation report used for this test was
written outside the T3 production directory.

## Relationship to the provenance change

This PR builds on the VSC provenance support introduced in the preceding PR.
For older untagged VSC results, the validator can create a legacy provenance
record without inventing a fitting frame range.  Such a result may use the
explicit current-data self-check, but it cannot be described as held-out
validation.
