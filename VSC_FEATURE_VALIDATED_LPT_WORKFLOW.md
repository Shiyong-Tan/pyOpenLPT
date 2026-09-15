# Validated VSC-to-LPT workflow

## Purpose

This change connects VSC provenance, the reproducible sampled quality gate, and
the per-run original/VSC camera selector in the Tracking view.  VSC camera files
cannot be used for LPT unless their saved quality report still matches the exact
camera files and provenance manifest.

## VSC fitting-range controls

The VSC panel now records an inclusive Source Frame Start and Source Frame End.
`All` leaves the upper bound open.  These values are passed to `VSCService`, so
the provenance manifest can record both the requested range and the actual
frames represented by accepted correspondences.  An end value below the start
is rejected before the worker begins.

Existing legacy VSC camera files remain usable, but their unknown fitting range
is never guessed.

## User flow

When `Use VSC camera files` is selected:

1. The GUI checks for a passing, hash-bound `vsc_validation.json` report.
2. If no valid report exists, it states how many contributing frames are known
   and asks whether to run the reproducible 30-frame check.
3. Selecting No returns the run to original cameras.
4. Selecting Yes runs validation in a background thread, keeping the GUI
   responsive.
5. A failed gate blocks VSC use and reports the failed checks.
6. A passed gate starts LPT automatically; no second Run click is required.

If the VSC fitting range is known, sampling excludes that inclusive range.  If
the range is unknown, validation uses the current acquisition only through the
explicit `current_data_self_check` mode and labels the result as not
independent.

The check may also be run manually from `Check VSC (30 Random Frames)`.  Manual
checks do not start LPT automatically.

## Run-time enforcement and audit

The run-config preparation step calls `verify_saved_validation()` again.  This
is the final gate even if UI state becomes stale.  Any edit to a `vsc_cam*.txt`
file or to `vsc_provenance.json` invalidates the report and stops the launch.

For an accepted VSC run, the execution log records:

- whether the result is held-out validation or a non-independent self-check;
- VSC source label and contributing-frame count;
- fitting and eligible validation ranges;
- all 30 frame IDs actually sampled;
- sampling mode and dataset fingerprint; and
- the quality-report path.

## Verification

- The GUI module compiles in the OpenLPT Python environment.
- Off-screen UI construction verified the new range, status, and 30-frame
  controls.
- A valid fixture report enabled VSC run-config creation while preserving the
  master config.
- Modifying one VSC camera byte invalidated the report and was rejected.
- A simulated successful confirmed validation scheduled LPT automatically.
- The underlying validator independently passed T3 on 30 reproducible frames,
  with 284 clean observations and improved pooled RMSE/P95.

This workflow changes selection and validation only.  It does not change VSC
optimization, IPR, Shake, tracking, or scientific numerical operations.

## Prerequisites

This commit is intentionally stacked on the VSC provenance, sampled VSC quality
gate, and per-run output/camera-selection commits.  Once those prerequisite PRs
are merged, this PR reduces to the GUI orchestration and enforcement described
above.
