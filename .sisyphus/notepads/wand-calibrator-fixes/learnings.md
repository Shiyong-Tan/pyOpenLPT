# Wand Calibrator Fixes — Learnings

## 2026-03-15 Init: Plan startup

### Key Codebase Facts
- Target files (ONLY these 4 may be modified):
  1. `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py` — B5 (~line 2224), P2 (lines 486–495)
  2. `modules/vsc/refraction_optimizer.py` — B1 (lines 458–465)
  3. `modules/camera_calibration/wand_calibration/refraction_calibration_BA.py` — B3, P1, P3, P4
  4. `modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py` (NEW — to be created)

- Test file location: `modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py`
  - NOTE: Plan says `tests/` dir but actual target should match module structure; agent must verify which `tests/` vs `modules/.../` path
  
### Mock Targets
- `build_pinplate_rays_cpp_batch` and `update_cpp_camera_state` imported at lines 44–51 of BA file
- `@patch` at module import level: `modules.camera_calibration.wand_calibration.refraction_calibration_BA.<fn>`
- `cams_cpp` values: `MagicMock()` with `projectBatchStatus.return_value = []`

### Noise Model Location
- `sigma_ray = px_target * (avg_dist_z / avg_f)` at line 561 of refraction_calibration_BA.py
- NOT in refraction_wand_calibrator.py:400–450 (that's plane init)

### Radii Validation Rules
- Must use `np.isfinite()`, NOT `is not None`
- Valid range: [0.1, 50.0] mm, positive, finite, radius_B ≥ radius_A
- Trap: `dataset.get('est_radius_small_mm', 0.0)` default of 0.0 must be caught

### Status Code Handling
- status ≥ 1 → SUCCESS, always use res.x
- status == 0 → WARNING (budget exhausted), STILL use res.x (partial progress valid)
- status == -1 → ERROR, raise RuntimeError
- DO NOT raise on status=0 (breaks chunked optimization at lines 1432–1530)

### Conda Environment
- Use `conda run -n OpenLPT python ...` for all Python commands
- Test command: `conda run -n OpenLPT python -m pytest <file> -v`

### Branch
- Currently on `Refraction` branch (detached HEAD or branch, see git status)

## 2026-03-15 B5: rs_pr4/rl_pr4 fallback hardening

### Fix Pattern
- In `refraction_wand_calibrator.py` around BA config construction, keep explicit defaults before optional estimate path:
  - `rs_pr4, rl_pr4 = 1.5, 2.0  # Defaults`
- After estimate assignment block, guard both values with `np.isfinite(...)` + positive checks and reset both to defaults on failure.

### Regression Test Pattern
- Added `TestWandCalibratorB5Fixes.test_b5_defaults_fallback` in existing `test_wand_calibrator_fixes.py`.
- For this path-specific test, patch calibrator internals to bypass unrelated bootstrap/triangulation complexity and capture BA config radii.
- Minimal dummy base must include `dist_coeff_num` in addition to camera settings/cam_params.

## 2026-03-15 T1+T2: Test infrastructure file created

### Test File
- Created `modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py` (430 lines, 16 tests)
- All 16 tests pass with both `unittest` and `pytest`

### Constructor Mocking Pattern
- `RefractiveBAOptimizer.__init__` calls `_refresh_plane_round_reference()` which uses `cv2.Rodrigues` on `cam_params`
- Zero rvec/tvec works fine with real cv2 — no need to mock cv2
- The two C++ functions that MUST be patched: `build_pinplate_rays_cpp_batch` (returns `[]`) and `update_cpp_camera_state` (returns `None`)
- Class-level `@patch` on `TestRefractiveBAOptimizerUnit` — mock args arrive as extra positional params to test methods

### Pytest Discovery
- `pytest` was NOT installed in OpenLPT conda env; installed via `pip install pytest`
- Alternative: `python -m unittest modules.camera_calibration.wand_calibration.test_wand_calibrator_fixes -v` always works
- Direct python path: `C:\Users\tan_s\miniconda3\envs\OpenLPT\python.exe` avoids conda run issues

### grep/LSP Quirks
- `grep` tool does NOT find patterns in the BA file (3281 lines, possibly encoding/size issue)
- Workaround: use `conda run -n OpenLPT python -c "import ast; ..."` to parse with AST and find class/function definitions
- LSP diagnostics show pre-existing numpy/cv2/scipy stub errors throughout the entire codebase — not actionable

### Key Architectural Decisions
- `ObsCacheBuilder.build()` at line 142 — static method, builds from `dataset['obsA']` and `dataset['obsB']`
- `RefractiveBAConfig` at line 173 — pure `@dataclass`, no need to mock (per MUST NOT rules)
- `cam_to_window` maps cam_id → window_id; `wand_length` is float in mm
- Constructor signature: `(dataset, cam_params, cams_cpp, cam_to_window, window_media, window_planes, wand_length, config=None, progress_callback=None)`

## 2026-03-15 PROJECT COMPLETE — Final Verification Wave PASSED

### Final Wave Results
- F1 Plan Compliance Audit: **APPROVE** — Oracle verified 13 commits since 2f994e5, 42 tests pass, B5/B1/B3 all grounded in source code, P2 median init at wand_calibrator.py:486-500, P3 recompute at refraction_calibration_BA.py:2901+2942-2943. AST lint passes. Smoke test passes.
- F2 Code Quality Review: **APPROVE** — B5 dataset wiring confirmed correct (validated radii written AFTER guard), all 4 files quality-reviewed.
- F3 Real Manual QA: **APPROVE** — Import smoke test passed; all key classes importable, reporter.warning() verified, sigma_schedule + tolerance_schedule present in config.
- F4 Scope Fidelity: **APPROVE** — 10 files changed, all in-scope; no unrelated refactoring.

### Key Fixes Summary
1. **B5** (`refraction_wand_calibrator.py:2225-2237`): Added validity guard + fallback defaults (rs=1.5, rl=2.0); dataset written AFTER guard so BA consumer reads safe values.
2. **B1** (`refraction_optimizer.py:458-468`): Stale normal replaced with `n_new` for correct plane-point displacement.
3. **B3** (`refraction_calibration_BA.py:70-71, 2000-2007, 2303-2310`): status==-1 → RuntimeError, status==0 → reporter.warning() (not raise). `RefractiveCalibReporter.warning()` method added.

### Key Improvements Summary
- P1: barrier_schedule parameters tuned in RefractiveBAConfig
- P2: Median aggregation for plane-point init (wand_calibrator.py:486-500)
- P3: `_compute_physical_sigmas()` called at round boundaries (refraction_calibration_BA.py:2901, 2942-2943)
- P4: tolerance_schedule in RefractiveBAConfig for parameterized ftol/xtol/gtol

### Evidence Files
- `.sisyphus/evidence/task-13-regression-final.log` — 42 passed
- `.sisyphus/evidence/f3-smoke-test.log` — F3 SMOKE TEST: PASS
- `.sisyphus/evidence/f-lint-ast-check.log` — AST LINT: ALL 4 TARGET FILES PASS

### Commits (13 since 2f994e5)
da3e377, 1c37f72, ea87a7c, 683ee9d, 32c959f, 01353e1, 8001548, 5ecd91f, 2552240, 6fcb6ad, 9f13e48, 3be7f92, ac4c883

## 2026-03-15 W1a: Phase 3 residual slicing bug (interleaved layout)

### Residual Layout Rule
- Phase 3 residuals are interleaved per frame, not grouped by residual type:
  `[wand_k, reproj_k_cam0_uvA(2), reproj_k_cam0_uvB(2), ...]`
- Correct extraction stride for wand-only residuals is:
  `residuals_per_frame = 1 + 2*n_cameras`, `wand_indices = np.arange(n_frames) * residuals_per_frame`.

### Anti-Pattern Caught
- `final_res[n_frames:]` is invalid for this layout; it slices an arbitrary tail mixing wand+reprojection terms and can hide real optimization quality.

### Test Pattern
- Use synthetic interleaved residual blocks with high reprojection terms and known wand terms to force clear RMS separation.
- Validate expected wand RMS via `extract_wand_residuals(...)` helper and assert old slicing does **not** match expected value.


## W1b - Homogeneous division guard pattern (2026-03-15)
- Triangulation outputs in homogeneous coordinates must never divide by `w` unguarded.
- Use a consistent threshold `w > 1e-8` before Euclidean conversion.
- If `w <= 1e-8`, return `np.full(3, np.nan, dtype=np.float64)` as sentinel to avoid `inf` propagation and optimizer corruption.
- Apply this guard at every triangulation conversion site (not just one phase) to prevent downstream instability in diagnostics and BA initialization.

## W1c - Phase 1 gauge anchoring pattern (2026-03-15)
- In two-camera Phase 1 BA, `cam_i` must remain outside the optimization vector to anchor gauge freedom.
- Correct state vector layout is `[cam_j(6), pts_3d(N*3)]`; `cam_i` is always reconstructed as `np.zeros(6)` in residuals and outputs.
- Jacobian sparsity must not allocate any `cam_i` columns; with one optimized camera, `pt_start` shifts from 12 to 6.
- A robust regression test pattern is monkeypatching `least_squares` to force the first six optimized parameters non-zero; this fails on old layout and passes once `cam_i` is frozen.

## W1d - Phase 3 gauge anchoring pattern (2026-03-15)
- Global BA must freeze the first calibrated camera (`min(cam_id)`) outside the state vector to remove rigid gauge freedom.
- Correct Phase 3 layout is `[free_cams(6 each), pts_3d]`; first camera is injected in residuals as `(R=I, t=0)` and output as `np.zeros(6)`.
- Jacobian sparsity for first-camera reprojection rows should include only point columns (no camera columns), while free cameras keep camera+point coupling.
- Regression test pattern: monkeypatch `least_squares` to force `x[:6]` non-zero and assert cam_0 remains exact zero while other cameras remain non-zero.

## W2b - Smooth C∞ side barrier ported to wand BA (2026-03-15)
- Replaced hard gated barrier branch (`if gap > 0 ... else 0`) with softplus/logaddexp smoothing in wand BA:
  - `tau_eff = max(tau, 1e-12)`
  - `gap_smooth = tau_eff * np.logaddexp(0.0, gap / tau_eff)`
  - residual slots preserved as 2 per barrier point:
    - `r_fix_const * (1.0 - np.exp(-gap_smooth / tau_eff))`
    - `r_grad_const * gap_smooth`
- Kept violation diagnostics behavior (`violations_count` still increments only for raw `gap > 0`).
- Preserved point-radius-aware gap definition exactly: `gap = (margin_mm + r_val) - sX`.
- Kept early/mid `soft_on` augmentation unchanged (additive soft floor on first barrier slot).
- Added `TestBarrierResiduals` to `test_refractive_bootstrap.py` covering:
  - feasible case (`gap < 0`) residuals near zero,
  - violating case (`gap > 0`) residual positivity + monotonicity with larger radius,
  - smooth no-kink behavior near `gap = 0` via left/right finite-difference derivative continuity check.

## W3d - Huber loss in Phase 3 BA (2026-03-15)

### Implementation Choice
- Phase 3 BA uses `scipy.optimize.least_squares` with `method='trf'` at line 1171 of `refractive_bootstrap.py`.
- The `trf` method natively supports `loss='huber'` — no manual residual transformation needed.
- One-line change: added `loss='huber'` to the `least_squares` call. `f_scale=1.0` was already present (controls the Huber transition threshold).

### Huber Loss Behavior (scipy)
- scipy's Huber: `rho(z) = z` if `z <= 1`, else `2*sqrt(z) - 1`, where `z = (r/f_scale)^2`.
- For small residuals (`|r| << f_scale`): identical to L2 least squares.
- For large residuals (`|r| >> f_scale`): cost grows as `sqrt(z)` instead of `z`, strongly downweighting outliers.
- `f_scale=1.0` means the transition between quadratic and linear behavior occurs at `|r| = 1.0` pixel.

### Test Pattern
- Monkeypatch `least_squares` to capture kwargs, assert `loss='huber'` and `f_scale > 0`.
- Second test validates Huber mathematical properties: identity for small residuals, strong downweighting for large residuals.
- Tests in `TestPhase3HuberLoss` class in `test_refractive_bootstrap.py`.

### Results
- 70 passed + 9 subtests, 0 failed (was 68 + 9 subtests before).

## W3e - Cheirality guard (2026-03-15)

### Problem
- `_project()` at line 602 returned `np.array([1e6, 1e6])` for behind-camera points (z <= 0).
- This magic sentinel polluted residual vectors with ~1e6 values instead of explicit NaN signaling.
- No caller-side sentinel checks existed (no `>= 1e6` or `>= 1e5` patterns found).

### Fix
- `_project()`: Changed `return np.array([1e6, 1e6])` to `return np.array([np.nan, np.nan])`.
- Phase 2 BA residuals (lines 509-525): Added `np.isnan(proj[0])` check, zero residual for NaN.
- Phase 3 BA residuals (lines 1133-1146): Same NaN guard pattern.
- Diagnostics (lines 688-694): Skip NaN projections from reproj_errors list.

### Key Insight
- NaN in residual vectors breaks `least_squares`. Cannot simply return NaN without guarding callers.
- For residual functions: NaN projections produce zero residuals (skip observation gracefully).
- For diagnostics: NaN projections are excluded from error statistics.
- Sparsity matrix expects fixed residual count per frame, so `continue` is not possible in residuals.

### Test Pattern
- `TestCheiralityCheck` class with 3 tests: behind-camera NaN, on-plane NaN, in-front finite.
- Direct unit test of `_project()` method with identity camera setup (R=I, t=0, K with f=800).
- Verifies exact projection math for in-front points.

### Results
- 73 passed + 9 subtests, 0 failed (was 70 + 9 subtests before).
