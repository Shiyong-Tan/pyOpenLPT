# Refractive Wand Calibration Algorithm Documentation

This document describes the current refractive wand calibration pipeline implemented in:

- `refraction_wand_calibrator.py`
- `refraction_calibration_BA.py`

The pipeline calibrates cameras viewing through refractive interfaces (`air -> window -> water`) and exports PINPLATE camFiles and triangulation reports.

## Overview

The calibrator estimates/refines:

- Camera extrinsics (`rvec`, `tvec`)
- Camera intrinsics (`f`, `cx`, `cy`, optional `k1`, `k2`)
- Window geometry (`plane_pt`, `plane_n`)
- Window thickness

Core principles:

- C++ PINPLATE kernel is the source of truth for refracted rays
- Objective is built from ray distance and wand length residuals
- Side barrier enforces physically valid side-of-plane constraints
- Strategy can be switched per round (`sequence` or `bundle`)

## Optical Model

```
Camera (n1=air) -> Window (n2) -> Object medium (n3)
```

Per-window parameters:

- `plane_pt`: point on the closest interface (camera side)
- `plane_n`: unit normal (camera -> object direction)
- `thickness`
- `n1`, `n2`, `n3`

## Optimization Strategies

Two strategies are implemented:

- `sequence`: original staged BA (parameter deltas, point solved implicitly inside residual evaluation)
- `bundle`: explicit joint optimization of selected parameters + per-frame explicit 3D points (`A/B` endpoints)

Strategy selection supports:

- Global fallback: `optimization_strategy`
- Per-round override: `round_strategy`

Current default `round_strategy`:

- `loop_planes: bundle`
- `loop_cams: bundle`
- `joint: sequence`
- `final_refined: sequence`

### Strategy Details

#### `sequence` (staged delta-parameter BA)

`sequence` optimizes only the enabled camera/plane/media parameter deltas for the current round. 3D wand endpoints are not global optimization variables in this mode.

High-level algorithm per residual call:

1. Unpack current delta vector `x` into updated camera/plane/media parameters (starting from round relinearization state).
2. Sync updated camera/plane/media to C++ PINPLATE objects.
3. Build refracted rays (batched by camera) for endpoint A and B observations.
4. For each frame:
   - estimate endpoint A and B from rays (explicit point solve with fallback triangulation)
   - accumulate point-to-ray residuals
   - accumulate wand-length residual
   - accumulate side-barrier residuals
5. Return stacked residual vector to `least_squares`.

Characteristics:

- Lower parameter dimension than bundle for the same round.
- Uses implicit point solve every residual call.
- Often robust for `joint/final_refined` when intrinsics/thickness/distortion are enabled.

#### `bundle` (joint BA with explicit 3D points)

`bundle` augments round variables with explicit per-frame endpoint variables `X_A(fid), X_B(fid)` and optimizes them jointly with enabled camera/plane/media parameters.

High-level algorithm per round:

1. Build base round layout from enabled camera/plane/media parameters.
2. Initialize or refresh explicit points (triangulated from current state; controlled by `bundle_retriangulate_each_round`).
3. Extend layout with point deltas (`ptA`, `ptB` for each optimized frame).
4. Build sparse Jacobian pattern (`jac_sparsity`) using dependency structure:
   - ray residual depends on one camera block + one window plane block + one point block
   - wand-length residual depends on the frame's `A/B` point blocks
   - barrier residual depends on corresponding point block + window plane block
5. Run `least_squares(..., jac_sparsity=...)` with same round bounds/settings (except `xtol` handling described below).
6. Write back optimized camera/plane/media and explicit points for next round.

Characteristics:

- Much larger variable space than sequence, but sparse structure reduces finite-difference cost.
- Better coupling between camera/plane and point geometry in some rounds.
- Requires careful coordinate consistency (alignment step transforms bundle points together with cameras/planes).

### Stopping-Condition Difference Between Strategies

Both strategies use the same round-level bounds, `ftol`, `gtol`, `max_nfev` from the caller.

`xtol` handling:

- `sequence`: uses caller-provided `xtol` directly.
- `bundle`: uses caller-provided `xtol` in loop modes; for `joint/final_refined` (when those rounds run in bundle), `xtol` is intentionally disabled (`None`) to reduce premature early-stop risk.

## Residual Model

Physical objective components:

- `S_ray`: sum of squared point-to-ray distances
- `S_len`: sum of squared wand length errors
- side-barrier residuals for plane-side feasibility

Fixed weighting rule:

- `lambda = 2 * N_cams_total`

No adaptive override to `lambda=1` is applied in the current implementation.

## Coordinate Alignment in BA

Alignment is now handled inside BA:

- once before the last loop camera step (`pre-last-loop-cam`)
- once at the end before export (`final-export`)

When bundle strategy is active, explicit bundle points are transformed with the same alignment transform so camera/plane/point states remain in one coordinate frame.

## Round-by-Round Bounds and Stopping Conditions

The table below reflects current code behavior.

| Round | Default Strategy | Main variables | Bounds | Stopping conditions |
|---|---|---|---|---|
| `loop_*_planes` (Step A) | `bundle` | Plane params (`plane_d`, `plane_a`, `plane_b`) | `plane_a/plane_b: +/-2.5deg`; `plane_d: +/-500mm` global, with weak-window dynamic shrink via `plane_d_bounds` | `ftol=5e-4`, `xtol=1e-5`, `gtol=1e-5`, `max_nfev=50` |
| `loop_*_cams` (Step B) | `bundle` | Camera extrinsics (`cam_t`, `cam_r`) | `cam_r: +/-180deg`; `cam_t: +/-2000mm` | `ftol=5e-4`, `xtol=1e-5`, `gtol=1e-5`, `max_nfev=50` |
| `joint` | `sequence` | Planes + camera extrinsics | `cam_r: +/-20deg`; `cam_t: +/-50mm`; `plane_d: +/-50mm`; `plane_a/plane_b: +/-10deg` | Base solver settings: `ftol=1e-5`, `xtol=1e-5`, `gtol=1e-5`, total budget `max_nfev=50` (executed in chunks; see below) |
| `final_refined` | `sequence` | Planes + camera extrinsics + `f` + `thickness` + optional `k1/k2` | `cam_r: +/-20deg`; `cam_t: +/-50mm`; `plane_d: +/-10mm`; `plane_a/plane_b: +/-5deg`; `f: +/-bounds_f_pct`; `thickness: +/-bounds_thick_pct`; `k1/k2: +/-bounds_dist_abs` | Base solver settings: `ftol=1e-5`, `xtol=1e-5`, `gtol=1e-5`, total budget `max_nfev=100` (executed in chunks; see below) |

Notes:

- Loop outer stop: max 6 passes, or early stop when Step A plane-angle constraints are inactive.
- In `bundle`, `xtol` is applied for loop modes and disabled (`None`) for `joint/final_refined` when those rounds run in bundle mode.

## Bundle-Specific Notes

- Bundle uses sparse Jacobian structure (`jac_sparsity`) for speed.
- Bundle point state can be re-triangulated each round (`bundle_retriangulate_each_round`, default `True`).
- Point delta bound is configurable by `bundle_point_delta_mm`; `None` means unbounded point delta.

## Chunked Early-Stop Framework (Joint/Final)

`joint` and `final_refined` now use a reusable chunked execution framework. This is strategy-agnostic and works for both `sequence` and `bundle`.

Default chunk schedules:

- `joint`: `[20, 5, 5, 5, 5, 5, 5]` (sum = 50)
- `final_refined`: `[30, 5 x 14]` (sum = 100)

Decision rule after each chunk:

- Compute current `ray_rmse` and `len_rmse`.
- Continue if **either** metric improves above relative thresholds.
- Early-stop when consecutive non-improving chunks reach `chunk_patience_chunks`.

Default thresholds:

- `chunk_rel_eps_ray = 2e-3`
- `chunk_rel_eps_len = 2e-3`
- `chunk_patience_chunks = 2`

Early-stop retry behavior:

- If early-stop triggers and retry is enabled, BA performs one coordinate alignment and reruns the **same round** with the **same chunk schedule**.
- Controlled by:
  - `chunk_align_retry_enabled = True`
  - `chunk_align_retry_max = 1`

Round enable switch:

- `chunk_modes` controls which rounds use the framework.
- Default: enabled for `joint` and `final_refined` only.

## Pipeline Summary

1. Bootstrap and initialize camera/plane state
2. Alternating loop:
   - Step A planes
   - Step B cameras
   - early-stop check
3. One alignment before final loop camera step
4. `joint`
5. `final_refined`
6. Final alignment (`final-export`)
7. Export camFiles and reports

## Output Artifacts

- `camFile/` PINPLATE camera files
- `triangulation_report.json`
- cache files (bootstrap and bundle-related)

## Related Files

- `refraction_wand_calibrator.py`: orchestration and export
- `refraction_calibration_BA.py`: optimization engine (`sequence` + `bundle`)
- `refractive_bootstrap.py`: bootstrap
- `refractive_geometry.py`: refractive geometry and helpers
