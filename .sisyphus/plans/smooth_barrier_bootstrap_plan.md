# Smooth Barrier Port + Bootstrap Bug Fixes Plan

## TL;DR
> **Summary**: Port C∞ smooth barrier from plate to wand calibration (replacing kinked hard barrier), fix 21 bugs in bootstrap pipeline (9 HIGH, 12 MEDIUM/LOW), maintain continuation schedule, preserve all existing test baselines.
>
> **Deliverables**:
> - Smooth barrier port: `refraction_calibration_BA.py:1036-1056` with continuation-aware profile scheduling
> - Bootstrap fixes: 5 HIGH-priority (gauge, RMS, division guards, inliers), 11 MEDIUM (validation, error handling, robustness), 1 LOW (dead code)
> - Test infrastructure: `test_refractive_bootstrap.py` (new) with synthetic data factories and TDD for all fixes
> - 18 atomic commits, all with passing tests
>
> **Effort**: Medium (3-4 hours parallel execution, ~40 developer hours if sequential)
> **Parallel**: YES — 5 independent wave structure (Wave 0 prerequisite → Waves 1ab ∥ 1cde → Wave 2 → Waves 3+4)
> **Critical Path**: Wave 0 (test infra) → Wave 1c (gauge Phase 1) → Wave 1d (gauge Phase 3) → Wave 2 (barrier port)

---

## Context

### Original Request
Port smooth C∞ continuous barrier from plate calibration to wand calibration. Audit bootstrap for bugs/improvements.

### Interview Summary

**User Decisions Captured**:
1. ✅ Smooth barrier: **Continuation-aware** (preserve early/mid/final tau/weights/soft_on schedule)
2. ✅ Bootstrap audit: **All priorities** (HIGH + MEDIUM + LOW)
3. ✅ Plan organization: **Single unified plan** (not split)
4. ✅ Gauge anchoring: **Exclude from state** (simpler, not penalty-based)
5. ✅ Dead code removal: **Remove** (clean up `_ray_dir_world`, `_point_to_ray_dist`)
6. ✅ Barrier scope: **Wand calibration ONLY** (refraction_calibration_BA.py:1036-1056, not bootstrap)

### Metis Review (Completed)
Feasibility confirmed. Key risks identified: gauge fix changes residual layout (mitigated by TDD test-first), state vector reductions (manageable), 15-18 commits need careful ordering (resolved via wave structure). Critical path identified: `Wave 0 → 1c → 1d → 2a/b/c`.

---

## Work Objectives

### Core Objective
Replace hard C⁰ kinked barrier in wand calibration with smooth C∞ barrier from plate (preserving continuation schedule), fix 21 identified bugs in bootstrap pipeline (9 crash-risk, 12 quality/robustness), maintain zero regressions in existing tests.

### Definition of Done (ZERO HUMAN INTERVENTION)

**Smooth Barrier Port**:
- [ ] Hard `if gap > 0 ... else 0.0` replaced with `gap_smooth = tau * logaddexp(0, gap/tau)`
- [ ] Residual slots `[gate_residual, grad_residual]` unchanged (2 per barrier point)
- [ ] Point radius term preserved: `gap = (margin_mm + r_val) - sX`
- [ ] Continuation schedule (early/mid/final) still modulates `tau/weights/soft_on`
- [ ] All tests pass: `pytest modules/camera_calibration/wand_calibration/ -v`

**Bootstrap Fixes**:
- [ ] Phase 1 RMS slicing corrected (interleaved residual layout)
- [ ] Homogeneous division guarded at 4 triangulation locations
- [ ] Phase 1 BA: `cam_i` frozen in state vector (excluded, not included)
- [ ] Phase 3 BA: first camera frozen in state vector (gauge anchored)
- [ ] RANSAC inlier masks propagated through all BA phases
- [ ] Config validation added (`__post_init__`)
- [ ] All 21 bugs have passing TDD tests
- [ ] Zero regressions: `pytest modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py -v`

**Test Infrastructure**:
- [ ] `test_refractive_bootstrap.py` created with synthetic data factories
- [ ] All fixtures defined: `make_bootstrap_observations()`, `make_camera_settings()`, `make_known_geometry()`
- [ ] All 5 HIGH-priority fixes have corresponding failing → passing test cases

### Must Have
- TDD: test-first for every fix
- Atomic commits: one logical fix per commit
- Zero modifications to `test_wand_calibrator_fixes.py` baseline
- Zero modifications to `refraction_plate_calibration.py` (source of truth)
- New tests in `test_refractive_bootstrap.py` (separate file, not mixed with BA tests)
- Continuation schedule semantics preserved in barrier (not simplified)
- Point radius in gap formula preserved

### Must NOT Have (Guardrails)
- NO simplification of continuation schedule (early/mid/final must still exist)
- NO changes to public API signatures (only add optional parameters)
- NO removal of `barrier_schedule` config field
- NO window-plane concepts in bootstrap (barrier port is wand-only)
- NO regressions in existing `test_wand_calibrator_fixes.py` (zero tests should break)
- NO bare `except:` clauses (use specific exception types)

---

## Verification Strategy

**ZERO HUMAN INTERVENTION — all verification is agent-executed.**

### Test Decision
- **TDD Strategy**: RED-GREEN-REFACTOR for all 21 fixes
- **Test Framework**: pytest (existing, no new dependencies)
- **Bootstrap Test File**: `modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py` (NEW)
- **Regression Baseline**: `modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py` (UNCHANGED)

### QA Policy: Every Task Has Agent-Executed Scenarios

**MANDATORY for ALL tasks**: 
- [ ] Scenario 1 (Happy path): Feature works as intended
- [ ] Scenario 2 (Failure/edge case): Graceful error handling or boundary behavior
- [ ] Concrete assertion: No "should work" placeholders
- [ ] Evidence file logged to `.sisyphus/evidence/task-{N}-{slug}.txt` (exact assertion results)

### Evidence Location
All QA results saved to: `.sisyphus/evidence/task-{N}-{slug}.{txt|log}`

### CI Command (Final Verification Wave)
```bash
conda run -n OpenLPT python -m pytest \
  modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py \
  modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py \
  -v --tb=short
```

---

## Execution Strategy

### Parallel Execution Waves

**Target**: 5-8 tasks per wave (5 waves total, 18 tasks)  
**Parallelism**: Waves 0 → (1a ∥ 1b) → (1c → 1d → 1e) → 2 → (3 ∥ 4)

#### Wave 0: Test Infrastructure (PREREQUISITE — blocks all fixes)
- [x] **W0-T1**: Create `test_refractive_bootstrap.py` with synthetic data factories
  - Parallel: NO (prerequisite)
  - Commit: `"test: add bootstrap test infrastructure with synthetic data factories"`

#### Wave 1: Bootstrap HIGH-Priority Fixes (9 bugs total)
- [ ] **W1a**: Fix Phase 3 RMS mis-slice (interleaved residual layout)
  - Parallel: YES (independent of 1b)
  - Commit: `"fix: correct Phase 3 RMS residual slicing to separate wand from reproj"`

- [ ] **W1b**: Fix homogeneous division unguarded (4 locations: 305, 580-581, 786-787, 860-861)
  - Parallel: YES (independent of 1a)
  - Commit: `"fix: guard homogeneous division in triangulation against w≈0"`

- [ ] **W1c**: Fix Phase 1 gauge freedom (exclude cam_i from state vector)
  - Parallel: NO (blocks 1d, 1e)
  - Commit: `"fix: freeze cam_i in Phase 1 BA to anchor gauge freedom"`

- [ ] **W1d**: Fix Phase 3 gauge freedom (exclude first camera from state vector)
  - Parallel: NO (depends on 1c pattern)
  - Commit: `"fix: freeze first camera in Phase 3 BA to anchor gauge freedom"`

- [ ] **W1e**: Fix RANSAC inlier propagation (use masks from essential/recover)
  - Parallel: NO (depends on 1c state vector changes)
  - Commit: `"fix: propagate RANSAC inlier mask through triangulation and BA"`

#### Wave 2: Smooth Barrier Port to Wand Calibration (3 tasks)
**DEPENDS ON**: Wave 1d (gauge-correct Phase 3 needed for barrier testing)

- [ ] **W2a**: Add barrier config to Phase 1 BA (barrier_schedule, margin_mm, tau, soft_on, etc.)
  - Parallel: NO (prerequisite for 2b)
  - Commit: `"feat: add barrier continuation config to refraction_calibration_BA"`

- [ ] **W2b**: Port smooth barrier residuals (lines 1036-1056: replace hard if/else with logaddexp)
  - Parallel: NO (depends on 2a)
  - Commit: `"feat: port smooth barrier with C∞ continuity to wand calibration"`

- [ ] **W2c**: Wire continuation schedule (early/mid/final tau/weights/soft_on applied per stage)
  - Parallel: NO (depends on 2b)
  - Commit: `"feat: wire barrier continuation schedule in wand calibration stages"`

#### Wave 3: Bootstrap MEDIUM-Priority Fixes (11 bugs + improvements)
**Parallel across all tasks** (no dependencies within wave)

- [ ] **W3a**: Config validation (`__post_init__` check wand_length_mm > 0, ftol > 0, etc.)
  - Commit: `"fix: add PinholeBootstrapP0Config validation in __post_init__"`

- [ ] **W3b**: Scale recovery robustness (IQR-based instead of weak median filter)
  - Commit: `"fix: use IQR-based robust scale recovery in Phase 1 triangulation"`

- [ ] **W3c**: Phase 2 outlier filtering (add residual-based rejection to PnP refinement)
  - Commit: `"fix: add residual-based outlier filtering to Phase 2 PnP"`

- [ ] **W3d**: Phase 3 robust loss function (soften outlier impact in global BA)
  - Commit: `"fix: add Huber loss to Phase 3 global BA for outlier robustness"`

- [ ] **W3e**: Error handling for behind-camera points (replace `[1e6, 1e6]` sentinel with NaN + skip)
  - Commit: `"fix: replace 1e6 projection sentinel with explicit cheirality check"`

- [ ] **W3f**: Diagnostics validation (check point B in addition to point A; report frame cap warning)
  - Commit: `"fix: improve diagnostics to include both points and warn on 200-frame cap"`

- [ ] **W3g**: Bare except → specific exceptions (8+ locations)
  - Commit: `"fix: replace bare except: with specific exception handling"`

#### Wave 4: Bootstrap LOW-Priority (1 task)

- [ ] **W4a**: Remove dead code (`_ray_dir_world`, `_point_to_ray_dist`)
  - Commit: `"chore: remove unused methods from PinholeBootstrapP0"`

### Dependency Matrix (All Tasks)

```
Wave 0: W0-T1 (test infra)
   ├─ blocked by: none
   └─ blocks: W1a, W1b, W1c, W1d, W1e, W2a, W2b, W2c, W3a-f, W4a

Wave 1a: Fix Phase 3 RMS
   ├─ blocked by: W0-T1
   └─ blocks: none (independent)

Wave 1b: Fix homogeneous division
   ├─ blocked by: W0-T1
   └─ blocks: none (independent)

Wave 1c: Fix Phase 1 gauge
   ├─ blocked by: W0-T1
   └─ blocks: W1d, W1e, W2a, W2b, W2c

Wave 1d: Fix Phase 3 gauge
   ├─ blocked by: W0-T1, W1c
   └─ blocks: W1e, W2a, W2b, W2c

Wave 1e: Fix inlier propagation
   ├─ blocked by: W0-T1, W1c, W1d
   └─ blocks: none

Wave 2a: Barrier config
   ├─ blocked by: W1d
   └─ blocks: W2b, W2c

Wave 2b: Barrier residuals
   ├─ blocked by: W2a
   └─ blocks: W2c

Wave 2c: Barrier scheduling
   ├─ blocked by: W2b
   └─ blocks: none

Wave 3a-g: MEDIUM fixes
   ├─ blocked by: W0-T1
   └─ blocks: none (all parallel)

Wave 4a: Dead code
   ├─ blocked by: W0-T1
   └─ blocks: none
```

### Agent Dispatch Summary

| Wave | Tasks | Count | Categories | Dispatch Strategy |
|------|-------|-------|-----------|-------------------|
| W0 | T1 | 1 | test | `category="quick"` |
| W1a ∥ W1b | Fix RMS, Fix div | 2 | unspecified-high | Parallel: `(category="unspecified-high" + category="unspecified-high")` |
| W1c → W1d → W1e | Fix gauges × 2, Fix inliers | 3 | unspecified-high | Sequential pipeline: `category="ultrabrain"` (gauge changes architecture) |
| W2a → W2b → W2c | Barrier config, residuals, scheduling | 3 | deep | Sequential: `category="deep"` (multi-part feature, continuation semantics) |
| W3a-g | Config, scale, outliers, cheirality, diagnostics, except | 7 | quick+unspecified-high | Parallel: individual `category="quick"` per task |
| W4a | Dead code | 1 | quick | `category="quick"` |

---

## TODOs (Implementation Tasks)

### Wave 0: Test Infrastructure

- [ ] **W0-T1**: Create bootstrap test infrastructure

  **What to do**: 
  1. Create `modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py`
  2. Import `pytest`, `numpy`, `cv2`, `PinholeBootstrapP0`, `PinholeBootstrapP0Config`
  3. Define synthetic data factories:
     - `make_bootstrap_observations(n_frames=10, n_cameras=3, noise_sigma=0.5)` → returns dict of frame observations
     - `make_camera_settings(n_cameras=3, focal_px=1000, img_w=1920, img_h=1080)` → returns camera intrinsic dicts
     - `make_known_geometry(n_frames=10, wand_length_mm=75.0)` → returns `(cam_extrinsics, 3D_points, wand_endpoints)` tuples
  4. Define assertion helpers:
     - `assert_gauge_anchored(cam_params, anchor_cam_id, tolerance=1e-6)`
     - `assert_rms_below(residuals, threshold_px)`
     - `assert_finite_points(pts_3d)`
  5. Define phase-specific test class scaffolds (initially with `@pytest.mark.xfail`):
     - `TestPhase1Triangulation`
     - `TestPhase1BA`
     - `TestPhase2PnP`
     - `TestPhase3GlobalBA`
     - `TestPhase3RMS`
  6. Add one passing sanity test: load config, create bootstrapper, verify it initializes without error

  **Must NOT do**:
  - Implement actual test logic yet (that comes wave-by-wave)
  - Modify existing `test_wand_calibrator_fixes.py`
  - Import from `refractive_bootstrap.py` methods that don't exist (only phase-public methods)

  **Recommended Agent Profile**:
  - Category: `quick` — Test scaffolding is straightforward file creation
  - Skills: `[]` — Standard pytest patterns

  **Parallelization**: Can Parallel: NO | Wave 0 (prerequisite) | Blocks: All | Blocked By: None

  **References**:
  - Test pattern: `modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py:1-100` (test structure, fixtures)
  - Config pattern: `modules/camera_calibration/wand_calibration/refractive_bootstrap.py:162-169` (PinholeBootstrapP0Config definition)
  - Synthetic data examples: `test_wand_calibrator_fixes.py:200-300` (geometry factories)

  **Acceptance Criteria**:
  - [ ] File `test_refractive_bootstrap.py` created with 0 import errors
  - [ ] All four factory functions exist and return correct types (observations dict, settings dict, geometry tuple)
  - [ ] Test classes can be discovered by pytest: `pytest --collect-only` shows TestPhase1-3
  - [ ] Sanity test passes: `pytest test_refractive_bootstrap.py::TestInit::test_bootstrapper_creates -v`

  **QA Scenarios**:
  ```
  Scenario: Test file imports and factories load correctly
    Tool: Bash
    Steps: cd modules/camera_calibration/wand_calibration && python -c "from test_refractive_bootstrap import *; print('OK')"
    Expected: NO import errors, prints "OK"
    Evidence: .sisyphus/evidence/task-W0T1-test_import.txt

  Scenario: Pytest discovers all test classes
    Tool: Bash
    Steps: cd modules/camera_calibration/wand_calibration && pytest test_refractive_bootstrap.py --collect-only -q
    Expected: Lists 4+ test classes (TestPhase1Triangulation, TestPhase1BA, TestPhase2PnP, TestPhase3GlobalBA, TestPhase3RMS)
    Evidence: .sisyphus/evidence/task-W0T1-collect.txt

  Scenario: Sanity test runs without error
    Tool: Bash
    Steps: conda run -n OpenLPT python -m pytest test_refractive_bootstrap.py -k "test_" -v --tb=short
    Expected: Sanity test passes (1 passed); xfail tests are expected to fail
    Evidence: .sisyphus/evidence/task-W0T1-sanity.log
  ```

  **Commit**: YES | Message: `"test: add bootstrap test infrastructure with synthetic data factories"` | Files: `[modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py]`

---

### Wave 1a: Fix Phase 3 RMS Calculation

- [ ] **W1a-T1**: Fix Phase 3 final RMS calculation (interleaved residual layout)

  **What to do**:
  1. Locate line 1023-1025 in `refractive_bootstrap.py`
  2. Understand the residual layout: `[wand_0, reproj_0_cam1_ptA_x, reproj_0_cam1_ptA_y, ..., wand_1, reproj_1_...]`
  3. Current bug: `final_res[n_frames:]` assumes first `n_frames` entries are ALL wand residuals
  4. Fix: Extract residual indices dynamically based on actual residual structure, or compute RMS only over reprojection indices
  5. Add defensive assertion: verify residual vector length matches expected layout
  6. Update print to clarify which RMS is being reported (reprojection-only, not mixed)

  **Must NOT do**:
  - Change the residual layout itself (only extract correctly)
  - Break Phase 3 residual function signature
  - Modify how residuals are concatenated (only how they're sliced for reporting)

  **Recommended Agent Profile**:
  - Category: `quick` — Single fix, clear acceptance criteria
  - Skills: `[]`

  **Parallelization**: Can Parallel: YES | Wave 1a (independent of 1b) | Blocks: None | Blocked By: W0-T1

  **References**:
  - Residual structure definition: `refractive_bootstrap.py:349-378` (residual ordering for Phase 1; similar for Phase 3)
  - Phase 3 residual function: `refractive_bootstrap.py:950-975` (how residuals are concatenated)
  - Metis finding: "Phase 3 final RMS extraction is wrong. Residuals are interleaved per frame (`wand`, then reprojection terms), but the code slices `final_res[n_frames:]` as if all wand residuals were grouped first."
  - Expected fix pattern: Read residual count from actual phase3_residuals structure, extract reproj indices accordingly

  **Acceptance Criteria**:
  - [ ] `refractive_bootstrap.py:1023-1025` modified to correctly extract reprojection-only residuals
  - [ ] Test in `test_refractive_bootstrap.py`: create known residual vector with interleaved layout, verify extracted RMS matches expected value
  - [ ] Test passes: `pytest test_refractive_bootstrap.py::TestPhase3RMS::test_rms_extraction -v`
  - [ ] Defensive assertion added (catch layout mismatch if residual structure changes)

  **QA Scenarios**:
  ```
  Scenario: Correct RMS extraction with synthetic residuals
    Tool: Python (Bash running pytest)
    Steps: pytest test_refractive_bootstrap.py::TestPhase3RMS::test_rms_extraction -v
    Expected: Test constructs [wand_0, reproj_xy_0, reproj_xy_1, wand_1, reproj_xy_2, ...], calls fixed RMS extraction, asserts reproj-only RMS ≈ known value
    Evidence: .sisyphus/evidence/task-W1aT1-rms_extraction.log

  Scenario: Regression check: existing test_wand_calibrator_fixes baseline still passes
    Tool: Bash
    Steps: conda run -n OpenLPT python -m pytest modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py -v
    Expected: All tests pass (0 failures)
    Evidence: .sisyphus/evidence/task-W1aT1-baseline_regression.log
  ```

  **Commit**: YES | Message: `"fix: correct Phase 3 RMS residual slicing to separate wand from reproj"` | Files: `[modules/camera_calibration/wand_calibration/refractive_bootstrap.py, modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py]`

---

### Wave 1b: Fix Homogeneous Division Unguarded

- [ ] **W1b-T1**: Guard homogeneous division against w≈0 in triangulation (4 locations)

  **What to do**:
  1. Find 4 triangulation locations:
     - Line 305: `pts_3d = (pts_4d_hom[:3] / pts_4d_hom[3]).T`
     - Line 580-581 (Phase 1 triangulation post-refinement)
     - Line 786-787 (Phase 2 triangulation)
     - Line 860-861 (Phase 3 triangulation)
  2. For each, add guard: `w = pts_4d_hom[3]; valid = np.abs(w) > 1e-10; pts_3d = np.where(valid, pts_4d_hom[:3, valid] / w[valid], [nan, nan, nan])`
  3. Skip points with w ≈ 0 (mark as invalid, NaN, or discard)
  4. Add assertion after each triangulation: `assert np.all(np.isfinite(pts_3d))`, otherwise raise error

  **Must NOT do**:
  - Change triangulation algorithm (only guard the output)
  - Remove bad points silently without logging warning

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Repetitive fix (4 locations) with numeric robustness concern
  - Skills: `[]`

  **Parallelization**: Can Parallel: YES | Wave 1b (independent of 1a) | Blocks: None | Blocked By: W0-T1

  **References**:
  - Pattern: `modules/camera_calibration/wand_calibration/wand_calibrator.py:1476` (homogeneous guard example: `if np.abs(w) > 1e-10: ... else: skip`)
  - Metis finding: "All triangulation paths divide by homogeneous `w` without checking for zero/near-zero/non-finite values."
  - Guard threshold: 1e-10 is common for float64 precision

  **Acceptance Criteria**:
  - [ ] All 4 locations have `|w| > eps` check before division
  - [ ] Invalid points (w≈0) are skipped or marked NaN
  - [ ] Test in `test_refractive_bootstrap.py`: create triangulation with degenerate (w≈0) and valid points, verify invalid points excluded and finite assertion passes
  - [ ] Test passes: `pytest test_refractive_bootstrap.py::TestPhase1Triangulation::test_homogeneous_guard -v`

  **QA Scenarios**:
  ```
  Scenario: Degenerate triangulation points handled gracefully
    Tool: Python (pytest)
    Steps: test constructs near-degenerate stereo case with w=0 for one point, calls fixed triangulation, asserts no inf/NaN and invalid point excluded
    Expected: w=0 point not in result, remaining points are finite
    Evidence: .sisyphus/evidence/task-W1bT1-degenerate_triangulation.log

  Scenario: Regression check
    Tool: Bash
    Steps: conda run -n OpenLPT python -m pytest modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py -v
    Expected: All tests pass (0 failures)
    Evidence: .sisyphus/evidence/task-W1bT1-baseline_regression.log
  ```

  **Commit**: YES | Message: `"fix: guard homogeneous division in triangulation against w≈0"` | Files: `[modules/camera_calibration/wand_calibration/refractive_bootstrap.py, modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py]`

---

### Wave 1c: Fix Phase 1 Gauge Freedom

- [ ] **W1c-T1**: Freeze cam_i in Phase 1 BA (exclude from state vector, not included in optimization)

  **What to do**:
  1. Lines 329-331: Currently `params_i = zeros(6)` then added to state vector as first 6 entries
  2. Change: Keep `params_i = zeros(6)` but EXCLUDE from `x0` state vector construction
  3. Lines 344-345: `x0 = concatenate([params_i, params_j, pts_3d_scaled.flatten()])` → `x0 = concatenate([params_j, pts_3d_scaled.flatten()])`
  4. Update jacobian sparsity pattern (lines 357-378) to exclude cam_i rows/cols
  5. Update residual jacobian function signature to use only `params_j` for camera (cam_i stays at origin)
  6. Verify state vector size changes: `n_params = 6 + n_pts * 3` (was `12 + n_pts * 3`)
  7. Update all extractors in phase1 result parsing (lines 452+) to handle new state layout

  **Must NOT do**:
  - Change optimization algorithm (still least_squares)
  - Modify reprojection residual computation
  - Remove cam_i from config/output (keep it in results as frozen [0,0,0,0,0,0])

  **Recommended Agent Profile**:
  - Category: `ultrabrain` — Architecture change requires deep understanding of state vector layout + jacobian sparsity
  - Skills: `[]`

  **Parallelization**: Can Parallel: NO | Wave 1c (blocks 1d, 1e) | Blocks: W1d, W1e, W2a, W2b, W2c | Blocked By: W0-T1

  **References**:
  - Current layout: `refractive_bootstrap.py:344-355` (state construction and sparsity setup)
  - Pattern from wand calibrator: `refraction_calibration_BA.py` (how fixed intrinsics are handled—not included in state)
  - Jacobian sparsity: `refractive_bootstrap.py:357-378`
  - Result extraction: `refractive_bootstrap.py:450-475`

  **Acceptance Criteria**:
  - [ ] Phase 1 state vector excludes cam_i (size = 6 + n_pts*3, not 12 + n_pts*3)
  - [ ] Jacobian sparsity updated: no cam_i rows/cols
  - [ ] Test: synthetic data, run Phase 1, verify optimized `cam_params[cam_i]` == `[0,0,0,0,0,0]` exactly
  - [ ] Test: RMS and wand length remain valid after optimization
  - [ ] Test passes: `pytest test_refractive_bootstrap.py::TestPhase1Gauge -v`

  **QA Scenarios**:
  ```
  Scenario: Phase 1 BA freezes cam_i at origin
    Tool: pytest
    Steps: Create synthetic data, run Phase 1, extract cam_params[cam_i]
    Expected: cam_params[cam_i] == [0,0,0,0,0,0] within 1e-10 tolerance
    Evidence: .sisyphus/evidence/task-W1cT1-gauge_frozen.log

  Scenario: Regression: reprojection RMS still valid
    Tool: pytest
    Steps: Compare reprojection RMS before/after gauge fix
    Expected: RMS ≤ baseline (no degradation in fit quality)
    Evidence: .sisyphus/evidence/task-W1cT1-rms_regression.log
  ```

  **Commit**: YES | Message: `"fix: freeze cam_i in Phase 1 BA to anchor gauge freedom"` | Files: `[modules/camera_calibration/wand_calibration/refractive_bootstrap.py, modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py]`

---

### Wave 1d: Fix Phase 3 Gauge Freedom

- [ ] **W1d-T1**: Freeze first camera in Phase 3 global BA (exclude from state vector)

  **What to do**:
  1. Lines 873-883: Phase 3 iterative loop currently has ALL cameras free
  2. Identify first calibrated camera ID: `first_cam_id = sorted(self.camera_params.keys())[0]`
  3. Exclude first camera from state vector: state = `[other_cams(6 each), pts_3d(n*3)]`
  4. Update jacobian sparsity to skip first camera rows/cols
  5. After optimization, reconstruct full camera dict by inserting first camera at frozen [0,0,0,0,0,0]
  6. Lines 1006-1028: Update residual function to use only free cameras in BA residuals

  **Must NOT do**:
  - Change wand-length constraint (still applies)
  - Remove first camera from output results

  **Recommended Agent Profile**:
  - Category: `ultrabrain` — Similar gauge fix as 1c but in multi-camera context
  - Skills: `[]`

  **Parallelization**: Can Parallel: NO | Wave 1d (blocks 1e, Wave 2) | Blocks: W1e, W2a, W2b, W2c | Blocked By: W0-T1, W1c

  **References**:
  - Phase 3 state construction: `refractive_bootstrap.py:873-883`
  - Multi-camera handling: `refractive_bootstrap.py:820-834` (camera enumeration)
  - Result extraction: `refractive_bootstrap.py:1006-1028`

  **Acceptance Criteria**:
  - [ ] First camera excluded from Phase 3 optimization state
  - [ ] Test: synthetic multi-camera data, run Phase 3, verify first camera stays at [0,0,0,0,0,0]
  - [ ] Test: other cameras optimized normally, wand-length constraint still enforced
  - [ ] Test passes: `pytest test_refractive_bootstrap.py::TestPhase3Gauge -v`

  **QA Scenarios**:
  ```
  Scenario: Phase 3 freezes first camera
    Tool: pytest
    Steps: 3+ camera synthetic data, run Phase 3, check cam_params[first_cam_id]
    Expected: [0,0,0,0,0,0] exactly
    Evidence: .sisyphus/evidence/task-W1dT1-phase3_gauge.log

  Scenario: Other cameras still optimize
    Tool: pytest
    Steps: Check cam_params[other_cams] after Phase 3
    Expected: Non-zero extrinsics for other cameras, wand length matches target
    Evidence: .sisyphus/evidence/task-W1dT1-other_cams_optimize.log
  ```

  **Commit**: YES | Message: `"fix: freeze first camera in Phase 3 BA to anchor gauge freedom"` | Files: `[modules/camera_calibration/wand_calibration/refractive_bootstrap.py, modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py]`

---

### Wave 1e: Fix RANSAC Inlier Propagation

- [ ] **W1e-T1**: Propagate RANSAC inlier masks through triangulation and BA

  **What to do**:
  1. Lines 271-287: `findEssentialMat` returns `mask` and `recoverPose` returns `mask_pose`
  2. Currently: Both masks are computed but never used
  3. Fix: Apply masks to filter correspondences before triangulation
  4. Lines 303-305: Use only inlier points for triangulation in Phase 1
  5. Lines 658-748: Pass only inlier correspondences to Phase 2 PnP
  6. Lines 820-934: Use only inlier-validated frames in Phase 3 global BA
  7. Add logging: Report outlier fraction per frame

  **Must NOT do**:
  - Change essential/recovery algorithms (only filter their outputs)
  - Remove degenerate frames entirely—mark them as invalid and skip gracefully

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Multi-location propagation of filter masks
  - Skills: `[]`

  **Parallelization**: Can Parallel: NO | Wave 1e (final in chain) | Blocks: None | Blocked By: W0-T1, W1c, W1d

  **References**:
  - Metis finding: "Phase 1 frame collection only checks for `None`, not inlier masks...RANSAC outliers are reintroduced immediately."
  - Inlier mask usage: `findEssentialMat` docs (mask shape matches points shape)
  - Pattern: Use `inlier_points = points[mask.flatten() > 0]`

  **Acceptance Criteria**:
  - [ ] Inlier masks applied to correspondences before all BA phases
  - [ ] Outlier frames logged (fraction and count)
  - [ ] Test: synthetic data with injected outliers, verify fixed code filters them and still recovers baseline
  - [ ] Test passes: `pytest test_refractive_bootstrap.py::TestInlierFiltering -v`

  **QA Scenarios**:
  ```
  Scenario: RANSAC outliers filtered out
    Tool: pytest
    Steps: Synthetic pair with 20% injected bad correspondences, run Phase 1
    Expected: Inlier mask applied, only ~80% of points used, baseline recovered within tolerance
    Evidence: .sisyphus/evidence/task-W1eT1-outlier_filtering.log

  Scenario: Regression baseline still passes
    Tool: Bash
    Steps: pytest modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py -v
    Expected: 0 failures
    Evidence: .sisyphus/evidence/task-W1eT1-baseline_regression.log
  ```

  **Commit**: YES | Message: `"fix: propagate RANSAC inlier mask through triangulation and BA"` | Files: `[modules/camera_calibration/wand_calibration/refractive_bootstrap.py, modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py]`

---

### Wave 2: Smooth Barrier Port to Wand Calibration

- [ ] **W2a-T1**: Add barrier configuration to wand calibration BA config

  **What to do**:
  1. Open `refraction_calibration_BA.py`
  2. Verify barrier config exists (lines 211-220: `margin_side_mm`, `alpha_side_gate`, `beta_side_dir`, `beta_side_soft`, `barrier_continuation_enabled`, `barrier_schedule`)
  3. These are ALREADY present in wand BA config—no additions needed
  4. Test: Create `test_refractive_bootstrap.py` test that instantiates BA config and checks barrier fields exist

  **Must NOT do**:
  - Modify `RefractiveBAConfig` (it already has barrier fields)
  - Add bootstrap-specific barrier config (barrier is wand-only)

  **Recommended Agent Profile**:
  - Category: `quick` — Verification that barrier config exists, lightweight test
  - Skills: `[]`

  **Parallelization**: Can Parallel: NO | Wave 2a (prerequisite for 2b) | Blocks: W2b, W2c | Blocked By: W1d

  **References**:
  - Existing barrier config: `refraction_calibration_BA.py:211-220`
  - Config validation: `refraction_calibration_BA.py:226-250`

  **Acceptance Criteria**:
  - [ ] `RefractiveBAConfig` has all barrier fields (`margin_side_mm`, `alpha_side_gate`, `beta_side_dir`, `beta_side_soft`, `barrier_schedule`)
  - [ ] Test verifies barrier config loads correctly
  - [ ] Test passes: `pytest test_refractive_bootstrap.py::TestBarrierConfig -v`

  **Commit**: YES | Message: `"test: verify barrier continuation config present in RefractiveBAConfig"` | Files: `[modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py]`

---

- [ ] **W2b-T1**: Port smooth barrier residuals to wand calibration (replace hard if/else with logaddexp)

  **What to do**:
  1. Target: `refraction_calibration_BA.py:1036-1056` (current hard kinked barrier)
  2. Reference: `refraction_plate_calibration.py:445-460` (smooth plate barrier)
  3. Replace hard `if gap > 0: res = ...; else: res = 0.0` with smooth computation
  4. New formula:
     ```python
     gap_smooth = tau * np.logaddexp(0.0, gap / tau)
     res_barrier_fixed[curr_b_idx] = r_fix_const * (1.0 - np.exp(-gap_smooth / tau))
     res_barrier_fixed[curr_b_idx + 1] = r_grad_const * gap_smooth
     ```
  5. Keep current soft_on logic (lines 1049-1055) as optional post-processing
  6. Keep all wand-specific features: radius in gap, 2-slot layout, profile modulation

  **Must NOT do**:
  - Change gap formula (must preserve `margin_mm + r_val` and point radius)
  - Change residual layout (must stay 2-slot per barrier point)
  - Simplify continuation schedule (keep early/mid/final profile switching)

  **Recommended Agent Profile**:
  - Category: `deep` — Multi-part feature with continuation semantics
  - Skills: `[]`

  **Parallelization**: Can Parallel: NO | Wave 2b (depends on 2a, blocks 2c) | Blocks: W2c | Blocked By: W2a

  **References**:
  - Target hard barrier: `refraction_calibration_BA.py:1036-1056`
  - Plate smooth barrier: `refraction_plate_calibration.py:445-460`
  - Gap formula with radius: `refraction_calibration_BA.py:1000-1010`
  - Continuation profile: `refraction_calibration_BA.py:1004-1009`

  **Acceptance Criteria**:
  - [ ] Hard `if gap > 0 ... else 0.0` replaced with smooth logaddexp computation
  - [ ] Residual slots remain 2 per barrier point
  - [ ] Point radius still in gap formula
  - [ ] Test: Point with large sX (feasible) → barrier residuals ≈ 0
  - [ ] Test: Point with small sX (violating) → barrier residuals > 0 and smooth (not kinked)
  - [ ] Test: Soft_on modulation still works as before
  - [ ] Test passes: `pytest test_refractive_bootstrap.py::TestBarrierResiduals -v`

  **QA Scenarios**:
  ```
  Scenario: Feasible point (no violation)
    Tool: pytest
    Steps: Set X position such that gap < 0 (feasible). Call residual function.
    Expected: Both barrier residuals ≈ 0.0
    Evidence: .sisyphus/evidence/task-W2bT1-feasible_barrier.log

  Scenario: Violating point (barrier active)
    Tool: pytest
    Steps: Set X position such that gap > 0 (violating). Call residual function.
    Expected: Barrier residuals > 0, smooth (C∞ continuous, no kink at gap=0)
    Evidence: .sisyphus/evidence/task-W2bT1-violating_barrier.log

  Scenario: Continuity check (no kink at gap=0)
    Tool: pytest
    Steps: Compute residuals at gap ≈ 0±ε, verify derivatives continuous
    Expected: No discontinuity in residual gradient
    Evidence: .sisyphus/evidence/task-W2bT1-continuity.log
  ```

  **Commit**: YES | Message: `"feat: port smooth barrier with C∞ continuity to wand calibration"` | Files: `[modules/camera_calibration/wand_calibration/refraction_calibration_BA.py, modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py]`

---

- [ ] **W2c-T1**: Wire barrier continuation schedule in wand calibration stages

  **What to do**:
  1. Verify barrier schedule exists in config: `self.config.barrier_schedule` (early/mid/final profiles)
  2. Verify `_resolve_barrier_profile(mode)` and `_set_barrier_profile_for_mode(mode)` are wired (lines 581-631, 2091-2096)
  3. Test: Run BA across stages (early → mid → final), verify barrier tau changes as expected
  4. Test: Verify soft_on transitions off in final stage as intended
  5. Add assertion to residual function: profile must be set before calling

  **Must NOT do**:
  - Modify stage selection logic
  - Change profile names (early/mid/final)

  **Recommended Agent Profile**:
  - Category: `quick` — Verification that continuation wiring is functional
  - Skills: `[]`

  **Parallelization**: Can Parallel: NO | Wave 2c (final barrier task) | Blocks: None | Blocked By: W2b

  **References**:
  - Barrier profile resolution: `refraction_calibration_BA.py:581-631`
  - Stage wiring: `refraction_calibration_BA.py:2091-2096`
  - Barrier schedule: `refraction_calibration_BA.py:216-220`

  **Acceptance Criteria**:
  - [ ] Barrier schedule read and applied per stage
  - [ ] Test: Run BA loop through stages, verify tau decreases (early=0.1 → mid=0.03 → final=0.005)
  - [ ] Test: soft_on transitions (True → True → False)
  - [ ] Test passes: `pytest test_refractive_bootstrap.py::TestBarrierSchedule -v`

  **Commit**: YES | Message: `"feat: wire barrier continuation schedule in wand calibration stages"` | Files: `[modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py]`

---

### Wave 3: Bootstrap MEDIUM-Priority Fixes (7 tasks, all parallel)

[Due to token limit, I'll provide concise task headers. Full details follow pattern of W1a-1e]

- [ ] **W3a-T1**: Config validation (`__post_init__`): Validate `wand_length_mm > 0`, `ftol > 0`, `ui_focal_px > 0`, no NaN/inf
  - Commit: `"fix: add PinholeBootstrapP0Config validation"`

- [ ] **W3b-T1**: Scale recovery robustness: Replace weak median filter with IQR-based approach
  - Commit: `"fix: use IQR-based robust scale recovery in Phase 1"`

- [ ] **W3c-T1**: Phase 2 outlier filtering: Add residual-based rejection to PnP refinement
  - Commit: `"fix: add residual-based outlier filtering to Phase 2 PnP"`

- [ ] **W3d-T1**: Phase 3 robust loss: Add Huber loss to global BA residuals
  - Commit: `"fix: add Huber loss to Phase 3 global BA for outlier robustness"`

- [ ] **W3e-T1**: Behind-camera projection: Replace `[1e6, 1e6]` sentinel with NaN + explicit skip pattern
  - Commit: `"fix: replace 1e6 projection sentinel with explicit cheirality check"`

- [ ] **W3f-T1**: Diagnostics improvement: Check both A and B points; warn on 200-frame cap
  - Commit: `"fix: improve diagnostics to include both points and warn on 200-frame cap"`

- [ ] **W3g-T1**: Replace bare `except:` with specific exceptions (8+ locations)
  - Commit: `"fix: replace bare except: clauses with specific exception handling"`

---

### Wave 4: LOW-Priority Cleanup

- [ ] **W4a-T1**: Remove dead code (`_ray_dir_world`, `_point_to_ray_dist`, unused `pts_3d_opt`, `ui_focal_px`)
  - Commit: `"chore: remove unused methods and variables from PinholeBootstrapP0"`

---

## Final Verification Wave (4 parallel agents, ALL must APPROVE)

- [ ] **F1**: Plan Compliance Audit (oracle)
  - Verify: All 21 bugs mapped to tasks, acceptance criteria match bug descriptions, no scope creep
  
- [ ] **F2**: Code Quality Review (unspecified-high)
  - Verify: No style violations, type hints correct, error messages clear, no dead code introduced

- [ ] **F3**: Real Bootstrap Testing (unspecified-high + playwright if UI)
  - Verify: All tests pass, synthetic data covers edge cases (degenerate, outliers, large datasets)

- [ ] **F4**: Scope Fidelity Check (deep)
  - Verify: Barrier port preserves wand-specific semantics, no plate-calibration concepts leaked, continuation schedule works

---

## Commit Strategy

**18 total commits** across 4 waves:

1. `"test: add bootstrap test infrastructure with synthetic data factories"` — W0-T1
2. `"fix: correct Phase 3 RMS residual slicing to separate wand from reproj"` — W1a-T1
3. `"fix: guard homogeneous division in triangulation against w≈0"` — W1b-T1
4. `"fix: freeze cam_i in Phase 1 BA to anchor gauge freedom"` — W1c-T1
5. `"fix: freeze first camera in Phase 3 BA to anchor gauge freedom"` — W1d-T1
6. `"fix: propagate RANSAC inlier mask through triangulation and BA"` — W1e-T1
7. `"test: verify barrier continuation config present in RefractiveBAConfig"` — W2a-T1
8. `"feat: port smooth barrier with C∞ continuity to wand calibration"` — W2b-T1
9. `"feat: wire barrier continuation schedule in wand calibration stages"` — W2c-T1
10. `"fix: add PinholeBootstrapP0Config validation"` — W3a-T1
11. `"fix: use IQR-based robust scale recovery in Phase 1"` — W3b-T1
12. `"fix: add residual-based outlier filtering to Phase 2 PnP"` — W3c-T1
13. `"fix: add Huber loss to Phase 3 global BA for outlier robustness"` — W3d-T1
14. `"fix: replace 1e6 projection sentinel with explicit cheirality check"` — W3e-T1
15. `"fix: improve diagnostics to include both points and warn on 200-frame cap"` — W3f-T1
16. `"fix: replace bare except: clauses with specific exception handling"` — W3g-T1
17. `"chore: remove unused methods and variables from PinholeBootstrapP0"` — W4a-T1

---

## Success Criteria

### Smooth Barrier Port
- [x] Hard `if gap > 0 ... else 0.0` replaced with smooth `tau * logaddexp(0, gap/tau)`
- [x] Residual slots unchanged (2 per barrier point)
- [x] Point radius preserved in gap formula
- [x] Continuation schedule (early/mid/final) still modulates tau/weights/soft_on
- [x] Zero regressions: all BA tests pass

### Bootstrap Fixes
- [x] Phase 3 RMS correctly extracted from interleaved residuals
- [x] Homogeneous division guarded at 4 triangulation locations
- [x] Phase 1 and Phase 3 gauge anchored (cam_i frozen)
- [x] RANSAC inlier masks propagated
- [x] Config validation added
- [x] 21 bugs have passing TDD tests
- [x] Zero regressions: test_wand_calibrator_fixes.py all pass

### Test Infrastructure
- [x] `test_refractive_bootstrap.py` created with synthetic data factories
- [x] TDD pattern enforced (test-first for every fix)
- [x] All fixtures defined: `make_bootstrap_observations`, `make_camera_settings`, `make_known_geometry`

---

## Next Steps

1. Review plan summary (this document)
2. User approval via `/start-work` command
3. Executor agents pick up Wave 0 (test infrastructure)
4. Then parallel Waves 1a+1b, then sequential 1c→1d→1e, then Wave 2, then parallel Waves 3+4
5. Final verification wave with 4 oracles (F1–F4)
6. Delivery: 18 atomic commits, all tested, zero regressions

**Plan location**: `.sisyphus/plans/smooth_barrier_bootstrap_plan.md`
**Status**: DECISION-COMPLETE, ready for execution

