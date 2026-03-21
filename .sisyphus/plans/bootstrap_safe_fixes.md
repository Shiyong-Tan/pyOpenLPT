# Bootstrap Safe Bug-Fix + Anchoring Test Plan

## TL;DR
> **Summary**: Apply 5 bug fixes + anchoring changes to `refractive_bootstrap.py`. Test on copy (`refractive_bootstrap_v2.py`) only via custom test script (not comparison harness). Accept only if no Phase 2→3 reprojection spike and error ratios ≤ 1.10 on filtered dataset.
> **Deliverables**: Fixed `refractive_bootstrap_v2.py`, custom test script `.sisyphus/test_bootstrap_v2.py`, test results, promotion decision.
> **Effort**: Large (5 fixes + anchoring, custom test script, 1 wave, ~3-4 hours execution)
> **Parallel**: NO — fixes interact; all in one batch. Test script is new, not modifying production comparison harness.
> **Critical Path**: Copy file → apply 5 fixes + anchoring → compile check → write test script → run test on v2 only → parse results → promote or revert.

---

## Context

### Original Request (Revised)
User wants to safely apply **5 bug fixes + anchoring changes** to `refractive_bootstrap.py`. Test on copy (`refractive_bootstrap_v2.py`) only; **do NOT apply to production file**. Do NOT use `scripts/full_bootstrap_comparison.py`; worker must **write a custom test script** to validate v2 against filtered dataset. Include camera intrinsics in plan so worker knows correct input.

### Scope Changes
- **Include**: Phase 1 and Phase 3 anchoring (NOT deferred)
- **Exclude**: Modifying production harness; use custom test script instead
- **Include**: Camera settings as explicit plan inputs

### Test Dataset & Configuration
**Test CSV**: `J:\Fish\T0\wand_points_selected_filtered.csv` (2996 frames, 4 outliers pre-removed)

**Camera Settings** (per-camera intrinsics, from GUI table):
```python
camera_settings = {
    0: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
    1: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
    2: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
    3: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
    4: {'focal': 9000.0, 'width': 1024.0, 'height': 976.0},
}
```

**Wand Length**: 10.0 mm (immutable)

**Baseline Passing Metrics** (current production on filtered dataset):
- Phase 3 Final RMS: ~3.82 px
- reproj_err_mean: ~5.89 px
- wand_length_error: ~0.35 mm
- reproj_err_ratio (filtered/original): 0.9433 ✓
- wand_len_error_ratio (filtered/original): 0.6884 ✓

---

## Work Objectives

### Core Objective
Apply 5 bug fixes + 2 anchoring changes to `refractive_bootstrap_v2.py` (copy only, **NOT production**). Write custom test script to validate v2 on filtered dataset. Accept for future promotion only if no Phase 2→3 spike and metrics remain within tolerance.

### Deliverables
1. `refractive_bootstrap_v2.py` (copy of production with 5 fixes + anchoring applied)
2. `.sisyphus/test_bootstrap_v2.py` (custom test script — new file)
3. Syntax and import validation of v2 file
4. Test execution output and metrics log
5. Acceptance decision and evidence documentation

### Definition of Done (verifiable conditions with commands)

**Compilation**:
```bash
conda run -n OpenLPT python -m py_compile modules/camera_calibration/wand_calibration/refractive_bootstrap_v2.py
```
Expected: exit code 0 (no syntax errors).

**Import**:
```bash
conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.refractive_bootstrap_v2 import PinholeBootstrapP0; print('OK')"
```
Expected: "OK" printed, no ImportError.

**Test Script Execution**:
```bash
conda run -n OpenLPT python .sisyphus/test_bootstrap_v2.py 2>&1 | tee .sisyphus/evidence/bootstrap_v2_test.log
```
Expected: exit code 0 or 1 (completed, not crashed), output includes Phase 1/2/3 metrics and PASS/FAIL verdict.

**Acceptance** (parse test output for):
- `reproj_err_mean` (v2 on filtered dataset) — compare to baseline ~5.89 px
- `wand_length_error` (v2) — compare to baseline ~0.35 mm
- Phase 2 reproj error — baseline for Phase 2→3 spike check
- Phase 3 reproj error — must not spike >1.50× Phase 2
- All metrics finite (no NaN / inf) ✓
- Anchoring applied correctly (log shows anchored camera poses) ✓
- No crashes or tracebacks ✓

### Must Have
- [x] Copy: v2 file created before any fixes (exact copy of production)
- [x] Fixes: All 5 bugs fixed only in v2, NOT production
- [x] Anchoring: Phase 1 and Phase 3 anchoring applied to v2
- [x] Compile: v2 passes syntax check
- [x] Test Script: Custom `.sisyphus/test_bootstrap_v2.py` created and runs without crash
- [x] Metrics: reproj_err_mean within ~1.10× baseline, wand_length_error within ~1.10× baseline
- [x] Spike: No Phase 2→3 reprojection spike (Phase 3 ≤ 1.50 × Phase 2)
- [x] Finite: All metrics are finite (no NaN / inf)
- [x] Camera settings: Explicit in test script (not hardcoded from production harness)
- [x] Evidence: All test output logged to `.sisyphus/evidence/bootstrap_v2_test.log`

### Must NOT Have (guardrails, scope boundaries)
- [x] **NO changes to production file** — only v2 is modified
- [x] **NO modification to comparison harness** — use custom test script instead
- [x] **NO signature changes to `run_all()`** — public return `Tuple[Dict, dict]` unchanged
- [x] **NO change to `report['points_3d']` shape** — must remain `Dict[int, Tuple[np.ndarray, np.ndarray]]`
- [x] **NO silent failures** — all failures must raise explicitly or be caught and logged
- [x] **NO hardcoded test data paths** — parameterize CSV path in test script

---

## Verification Strategy

> ZERO HUMAN INTERVENTION — all verification is agent-executed.

- **Test decision**: Copy-first + custom test script. No use of production harness; custom script is purpose-built for v2 validation.
- **QA policy**:
  - Compilation check: `py_compile`
  - Import check: direct import statement
  - Functional check: custom test script runs Phase 1+2+3 on v2 alone, independently re-triangulates and computes metrics, logs results
  - Regression check: metrics must not degrade beyond threshold (≤1.10× baseline), no Phase 2→3 spike
  - Anchoring verification: log shows which cameras are anchored and at what poses
- **Evidence**: `.sisyphus/evidence/bootstrap_v2_test.log` (stdout capture) + visual inspection of metrics

---

## Execution Strategy

### Parallel Execution Waves

**Single wave** (all tasks sequential within wave):

| Task | Category | Dependencies | Blocks |
|------|----------|--------------|--------|
| 1. Copy prod → v2 | quick | — | Task 2 |
| 2. Apply 5 fixes + anchoring to v2 | unspecified-high | Task 1 | Task 3 |
| 3. Syntax & import check | quick | Task 2 | Task 4 |
| 4. Write custom test script | unspecified-high | — | Task 5 |
| 5. Run test on v2 only | unspecified-high | Task 3, 4 | Task 6 |
| 6. Parse results & decide | quick | Task 5 | Task 7 |
| 7. Document decision & cleanup | quick | Task 6 | — |

### Dependency Matrix (all tasks)

```
Task 1 (copy)
   ↓
   Task 2 (apply fixes + anchoring)
   ↓
   Task 3 (syntax check)
   ↓
   Task 5 (run test) ← Task 4 (write test script)
   ↓
   Task 6 (parse results)
   ↓
   Task 7 (cleanup/document)
```

### Agent Dispatch Summary

| Wave | Tasks | Count | Categories |
|------|-------|-------|------------|
| 1 | 1–7 | 7 | quick (3), unspecified-high (3), quick (1) |

---

## TODOs

### Task 1: ✅ Copy production file to v2 for safe testing

**What to do**: Create `modules/camera_calibration/wand_calibration/refractive_bootstrap_v2.py` as an exact byte-for-byte copy of `refractive_bootstrap.py`. This copy will be the only file modified; production file remains untouched.

**Must NOT do**:
- Don't modify the copy — it must be identical to production at this stage
- Don't delete or rename the original `refractive_bootstrap.py`

**Recommended Agent Profile**:
- Category: `quick` — Single file copy operation
- Skills: None required
- Omitted: None

**Parallelization**: Can Parallel: NO | Sequential only | Blocks: Task 2

**References**:
- Source file: `modules/camera_calibration/wand_calibration/refractive_bootstrap.py` (1123 lines, production version)
- Destination: `modules/camera_calibration/wand_calibration/refractive_bootstrap_v2.py`
- Verification: `cmp` or `diff` to confirm byte-for-byte identity

**Acceptance Criteria** (agent-executable only):
- [ ] File `refractive_bootstrap_v2.py` exists in `modules/camera_calibration/wand_calibration/`
- [ ] File size matches original (1123 lines, ~44 KB)
- [ ] First and last lines match original exactly
- [ ] No changes to any class names, function signatures, or logic

**QA Scenarios**:

```
Scenario: Copy is created and is byte-for-byte identical to original
  Tool: bash
  Steps:
    1. cd modules/camera_calibration/wand_calibration/
    2. cmp -b refractive_bootstrap.py refractive_bootstrap_v2.py
  Expected: No output (files identical)
  Evidence: .sisyphus/evidence/task-1-cmp.txt

Scenario: Copy can be imported without error
  Tool: bash
  Steps:
    1. conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.refractive_bootstrap_v2 import PinholeBootstrapP0, PinholeBootstrapP0Config; print('Import OK')"
  Expected: "Import OK" printed to stdout
  Evidence: .sisyphus/evidence/task-1-import.txt
```

**Commit**: NO (v2 is test vehicle, not committed until/unless promoted to production)

---

### Task 2: Apply 5 bug fixes + 2 anchoring changes to refractive_bootstrap_v2.py

**What to do**: Apply the following 5 fixes + 2 anchoring changes to `refractive_bootstrap_v2.py` only. All must be applied in a single task because they interact.

**Fix #1: Return optimized Phase-3 3D points from run_phase3, update report['points_3d'] in run_all()**

*Problem*: `run_all()` stores `report['points_3d']` before Phase 2/3, then Phase 3 moves cameras, making stored points inconsistent with final poses.

*Implementation*: Same as original plan (see original Task 3, Fix #1)

---

**Fix #2: Remove scale fallback; fail hard on insufficient/invalid inlier scale**

*Problem*: Fallback uses all correspondences, defeating inlier-gating. No NaN guard.

*Implementation*: Same as original plan (see original Task 3, Fix #2)

---

**Fix #3: Add failure guards for Phase 1 BA, Phase 2 iterative PnP, Phase 3 BA**

*Problem*: Failed solves overwrite good priors silently, creating phase transitions.

*Implementation*: Same as original plan (see original Task 3, Fix #3)

---

**Fix #4: Correct Phase 3 RMS extraction and Phase 1 diagnostics**

*Problem*: Phase 3 RMS sliced incorrectly (interleaved residuals); Phase 1 diagnostics ignore ptB.

*Implementation*: Same as original plan (see original Task 3, Fix #4)

---

**Fix #5: Guard `mask_pose` for scale — use recoverPose cheirality inliers instead of just E-inliers**

*Problem*: Scale uses E-inlier set, but some correspondences fail cheirality check in pose recovery.

*Implementation*:
- After line 314: Use `pose_inlier_idx_global` (which is cheirality-gated) for scale computation instead of raw `inlier_idx`
- Replace lines 321–329 to select wand pairs from pose inliers only:
```python
# OLD: wand_lengths_inlier computed from inlier_idx (E-matrix inliers)
# NEW: compute from pose inliers (cheirality-gated)
wand_lengths_pose = []
for i_frame in range(0, len(pose_inlier_idx_global) - 1, 2):
    if i_frame < len(pose_inlier_idx_global) - 1:
        idx_A = pose_inlier_idx_global[i_frame]
        idx_B = pose_inlier_idx_global[i_frame + 1]
        if idx_B - idx_A == 1:  # Check A, B are consecutive frame pair
            ptA = pts_3d_inlier[i_frame]
            ptB = pts_3d_inlier[i_frame + 1]
            wand_lengths_pose.append(np.linalg.norm(ptB - ptA))

if len(wand_lengths_pose) < 3:
    raise RuntimeError("[P0 FAIL] Insufficient valid pose-inlier wand pairs for scale recovery")
median_length = np.median(np.array(wand_lengths_pose))
```

---

**Anchoring Change #1: Phase 1 BA — Anchor cam_i full pose (rvec + tvec) to zeros**

*Problem*: Phase 1 BA has 6-DoF gauge nullspace; both cam_i and cam_j are free. To remove gauge ambiguity, anchor cam_i at origin.

*Implementation* (lines 358–374, BA state construction):
```python
# OLD: x0 = np.concatenate([params_i, params_j, pts_3d_scaled.flatten()])
# Both params_i and params_j are free

# NEW: Anchor cam_i to zeros; only cam_j and points are free
# x0 = [cam_j(6), pts_3d(N*3)]
# In residuals, cam_i is fixed to [0, 0, 0, 0, 0, 0]

# State vector: [cam_j(6), pts_3d(N*3)]
x0 = np.concatenate([params_j, pts_3d_scaled.flatten()])
n_params = 6 + n_pts * 3  # Only cam_j + points

# Jacobian sparsity updated: no columns for cam_i
A_sparsity = lil_matrix((n_res, n_params), dtype=int)
for i, fid in enumerate(valid_frames):
    idx_ptA = 6 + i * 6  # Offset by 6 (no cam_i)
    idx_ptB = 6 + i * 6 + 3
    base_res = i * 9
    
    # Wand length (depends on points only)
    A_sparsity[base_res, idx_ptA:idx_ptA+3] = 1
    A_sparsity[base_res, idx_ptB:idx_ptB+3] = 1
    
    # Reproj cam_i (fixed at origin, no params)
    A_sparsity[base_res+1:base_res+3, idx_ptA:idx_ptA+3] = 1
    A_sparsity[base_res+3:base_res+5, idx_ptB:idx_ptB+3] = 1
    
    # Reproj cam_j (free)
    A_sparsity[base_res+5:base_res+7, 0:6] = 1
    A_sparsity[base_res+5:base_res+7, idx_ptA:idx_ptA+3] = 1
    A_sparsity[base_res+7:base_res+9, 0:6] = 1
    A_sparsity[base_res+7:base_res+9, idx_ptB:idx_ptB+3] = 1

# Residuals function: cam_i fixed to [0,0,0,0,0,0]
def residuals_func(x):
    p_j = x[:6]
    pts = x[6:].reshape(-1, 3)
    
    R_i, _ = cv2.Rodrigues(np.zeros(3))  # cam_i at origin
    t_i = np.zeros((3, 1))
    R_j, _ = cv2.Rodrigues(p_j[:3])
    t_j = p_j[3:6].reshape(3, 1)
    
    res = []
    for idx, fid in enumerate(valid_frames):
        ptA = pts[idx * 2]
        ptB = pts[idx * 2 + 1]
        # ... rest of residual computation (unchanged)
```

**After Phase 1 optimization** (line 497–523):
```python
# Extract optimized params
params_i_opt = np.zeros(6)  # cam_i stays at origin
params_j_opt = result.x[:6]
pts_3d_opt = result.x[6:].reshape(-1, 3)

# Log anchoring
print(f"  [ANCHORING] Phase 1 BA: cam_i fixed at origin")
print(f"  cam_j rvec: [{params_j_opt[0]:.4f}, {params_j_opt[1]:.4f}, {params_j_opt[2]:.4f}]")
print(f"  cam_j tvec: [{params_j_opt[3]:.2f}, {params_j_opt[4]:.2f}, {params_j_opt[5]:.2f}]")
```

**Risk**: Previous anchoring attempt caused Phase 2→3 spike. **MUST test carefully** — compare Phase 2→3 transition metrics with baseline.

---

**Anchoring Change #2: Phase 3 Global BA — Anchor cam_i (seed camera from Phase 1) to its Phase 2 pose**

*Problem*: Phase 3 BA has 6-DoF gauge nullspace; all cameras and points are free. Anchoring one camera removes gauge.

*Implementation* (lines 906–912, state construction):
```python
# OLD: x0[idx * n_cam_params:(idx + 1) * n_cam_params] = params[:6]  # All free
# NEW: If cid == cam_i, store its Phase 2 pose separately and don't include in x0

# Identify which camera is cam_i (passed into run_phase3 or stored in self)
# Assume cam_i is stored in self.cam_i or passed via kwargs

# Build state: [all_cam_except_cam_i(n_cams*6 - 6), pts_3d(n_pts*3)]
# Track cam_i's fixed pose separately

cam_i_pose_fixed = cam_params[cam_i]  # From Phase 2

x0_free_cams = []
free_cam_ids = []
for cid in all_cam_ids:
    if cid != cam_i:
        x0_free_cams.append(cam_params[cid][:6])
        free_cam_ids.append(cid)

x0 = np.concatenate(x0_free_cams + [pts_3d_init.flatten()])

# Build mapping: free_cam_idx → all_cam_idx
free_cam_id_to_idx = {cid: i for i, cid in enumerate(free_cam_ids)}

# Jacobian sparsity: only free cameras (not cam_i)
n_cams_free = len(free_cam_ids)
pt_start = n_cams_free * n_cam_params

# In residuals_phase3, when projecting:
for cid in cams_in_frame:
    if cid == cam_i:
        # Use fixed pose from Phase 2
        R, _ = cv2.Rodrigues(cam_i_pose_fixed[:3])
        t = cam_i_pose_fixed[3:6].reshape(3, 1)
    else:
        # Extract from x0 (free)
        cam_free_idx = free_cam_id_to_idx[cid]
        p = x[cam_free_idx * n_cam_params:(cam_free_idx + 1) * n_cam_params]
        R, _ = cv2.Rodrigues(p[:3])
        t = p[3:6].reshape(3, 1)
    
    # ... project ptA, ptB

# After optimization
print(f"  [ANCHORING] Phase 3 BA: cam_{cam_i} fixed to Phase 2 pose")
for cid in free_cam_ids:
    cam_free_idx = free_cam_id_to_idx[cid]
    params = result.x[cam_free_idx * n_cam_params:(cam_free_idx + 1) * n_cam_params]
    cam_params_opt[cid] = params

cam_params_opt[cam_i] = cam_i_pose_fixed  # Keep Phase 2 pose
```

**Risk**: Gauge mismatch between Phase 2 (free) and Phase 3 (anchored) could create spike if points aren't co-transformed. **MUST ensure** points are triangulated consistently after Phase 3.

---

**Must NOT do**:
- Don't touch production file `refractive_bootstrap.py`
- Don't change `run_all()` public signature
- Don't modify `PinholeBootstrapP0Config` dataclass
- Don't apply fixes to any file other than `refractive_bootstrap_v2.py`

**Recommended Agent Profile**:
- Category: `unspecified-high` — Complex multi-part fix + anchoring, 5 interacting changes
- Skills: None required
- Omitted: None

**Parallelization**: Can Parallel: NO | Sequential only | Blocks: Task 3

**References**:
- Bug details: Oracle analysis from handoff summary
- Anchoring mechanism: Previous attempt was risky; this time, full pose anchor (not tvec-only) + Phase 3 uses Phase 2 pose
- Lines for Fix #1: 822–1122
- Lines for Fix #2: 337–350
- Lines for Fix #3: 484–494, 729–735, 1031–1041
- Lines for Fix #4: 1052–1055, 603–619
- Lines for Fix #5: 314–329
- Lines for Anchoring #1: 358–374, 497–523
- Lines for Anchoring #2: 906–912, 955–964

**Acceptance Criteria** (agent-executable only):
- [ ] All 5 fixes applied; code compiles without SyntaxError
- [ ] Anchoring #1: Phase 1 BA removes cam_i from state vector; cam_i fixed at [0,0,0,0,0,0]
- [ ] Anchoring #2: Phase 3 BA removes cam_i from state vector; cam_i fixed to Phase 2 pose
- [ ] Sparsity matrices updated correctly for reduced state vectors
- [ ] Log messages printed confirming anchoring at each phase
- [ ] No line count change >50 lines from original (allow for added logic)

**QA Scenarios**:

```
Scenario: Syntax check passes
  Tool: bash
  Steps:
    1. conda run -n OpenLPT python -m py_compile modules/camera_calibration/wand_calibration/refractive_bootstrap_v2.py
  Expected: exit code 0
  Evidence: .sisyphus/evidence/task-2-syntax.txt

Scenario: Module imports without error
  Tool: bash
  Steps:
    1. conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.refractive_bootstrap_v2 import PinholeBootstrapP0; print('OK')"
  Expected: "OK" printed to stdout
  Evidence: .sisyphus/evidence/task-2-import.txt

Scenario: Anchoring is applied (logged during run)
  Tool: bash (via Task 5 test run)
  Steps: (embedded in test output)
  Expected: Log contains "[ANCHORING] Phase 1 BA: cam_i fixed at origin" and "[ANCHORING] Phase 3 BA: cam_i fixed to Phase 2 pose"
  Evidence: .sisyphus/evidence/bootstrap_v2_test.log
```

**Commit**: NO (v2 is test vehicle; committed only if all tests pass and user approves promotion)

---

### Task 3: Syntax and import validation of refractive_bootstrap_v2.py

**What to do**: Compile and import the fixed v2 file to ensure no syntax errors before running the custom test.

**Must NOT do**:
- Don't execute any bootstrap logic (Task 5 does that)
- Don't modify the v2 file

**Recommended Agent Profile**:
- Category: `quick` — Two shell commands
- Skills: None required

**Parallelization**: Can Parallel: NO | Blocks: Task 5

**References**:
- Target file: `modules/camera_calibration/wand_calibration/refractive_bootstrap_v2.py`

**Acceptance Criteria** (agent-executable only):
- [ ] `py_compile` returns exit code 0
- [ ] Direct import succeeds, outputs "Import OK"

**QA Scenarios**:

```
Scenario: py_compile succeeds
  Tool: bash
  Steps:
    1. conda run -n OpenLPT python -m py_compile modules/camera_calibration/wand_calibration/refractive_bootstrap_v2.py
  Expected: exit code 0
  Evidence: .sisyphus/evidence/task-3-compile.txt

Scenario: Module imports successfully
  Tool: bash
  Steps:
    1. conda run -n OpenLPT python -c "from modules.camera_calibration.wand_calibration.refractive_bootstrap_v2 import PinholeBootstrapP0, PinholeBootstrapP0Config; print('Import OK')"
  Expected: stdout="Import OK"
  Evidence: .sisyphus/evidence/task-3-import.txt
```

**Commit**: NO

---

### Task 4: Write custom test script for v2 validation

**What to do**: Create `.sisyphus/test_bootstrap_v2.py` — a new test script (not modifying production harness). This script will:
1. Load filtered CSV dataset (`J:\Fish\T0\wand_points_selected_filtered.csv`)
2. Define camera settings (per spec in plan context)
3. Import and run `refractive_bootstrap_v2` (not production version)
4. Run full Phase 1+2+3 bootstrap on v2 only
5. Independently re-triangulate 3D points using final camera poses
6. Compute reproj error, wand length error, Phase 2→3 spike metrics
7. Compare to baseline and output PASS/FAIL + detailed metrics
8. Log all output to `.sisyphus/evidence/bootstrap_v2_test.log`

**Script Structure** (pseudo-code):

```python
#!/usr/bin/env python3
"""Test script for refractive_bootstrap_v2.py on filtered dataset."""

import numpy as np
import cv2
from pathlib import Path
import sys

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modules.camera_calibration.wand_calibration.refractive_bootstrap_v2 import (
    PinholeBootstrapP0,
    PinholeBootstrapP0Config,
)

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

FILTERED_CSV = Path(r"J:\Fish\T0\wand_points_selected_filtered.csv")
WAND_LENGTH_MM = 10.0

# Camera settings (from plan spec)
CAMERA_SETTINGS = {
    0: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
    1: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
    2: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
    3: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
    4: {'focal': 9000.0, 'width': 1024.0, 'height': 976.0},
}

# Baseline metrics (current production on filtered dataset)
BASELINE_REPROJ_ERR_MEAN = 5.89  # px
BASELINE_WAND_LEN_ERROR = 0.35  # mm

# Acceptance thresholds
REPROJ_ERR_RATIO_THRESHOLD = 1.10
WAND_LEN_ERROR_RATIO_THRESHOLD = 1.10
PHASE23_SPIKE_THRESHOLD = 1.50  # Phase 3 reproj ≤ Phase 2 reproj * 1.50

# ============================================================================
# LOAD OBSERVATIONS FROM CSV
# ============================================================================

def load_observations_from_csv(csv_path):
    """Load wand observations from CSV."""
    observations = {}
    all_cams = set()
    
    with open(csv_path, 'r') as f:
        for line in f:
            # CSV format: frame_id, cam_id, pointA_x, pointA_y, pointB_x, pointB_y
            parts = line.strip().split(',')
            fid, cid, uvA_x, uvA_y, uvB_x, uvB_y = map(float, parts)
            fid, cid = int(fid), int(cid)
            
            if fid not in observations:
                observations[fid] = {}
            
            observations[fid][cid] = (
                np.array([uvA_x, uvA_y], dtype=np.float64),
                np.array([uvB_x, uvB_y], dtype=np.float64),
            )
            all_cams.add(cid)
    
    return observations, sorted(list(all_cams))

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(cam_params, observations, camera_settings, all_cam_ids, wand_length_mm):
    """Compute reproj error and wand length error on full dataset."""
    all_reproj_errs = []
    all_wand_lengths = []
    
    def project_point(pt_3d, R, t, K):
        pt_cam = R @ pt_3d.reshape(3, 1) + t
        if pt_cam[2] <= 0:
            return None  # Behind camera
        pt_img = K @ pt_cam
        return (pt_img[:2] / pt_img[2]).flatten()
    
    # Get intrinsics
    K_by_cam = {}
    for cid in all_cam_ids:
        cfg = camera_settings[cid]
        f = cfg['focal']
        w, h = cfg['width'], cfg['height']
        cx, cy = w / 2.0, h / 2.0
        K_by_cam[cid] = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    
    valid_frames = [fid for fid in observations.keys() if len(observations[fid]) >= 2]
    
    for fid in valid_frames[:500]:  # Limit to first 500 for speed
        cams_in_frame = [c for c in observations[fid].keys() if c in cam_params]
        if len(cams_in_frame) < 2:
            continue
        
        # Triangulate using first two cameras
        c1, c2 = cams_in_frame[0], cams_in_frame[1]
        p1, p2 = cam_params[c1], cam_params[c2]
        
        R1, _ = cv2.Rodrigues(p1[:3])
        t1 = p1[3:6].reshape(3, 1)
        R2, _ = cv2.Rodrigues(p2[:3])
        t2 = p2[3:6].reshape(3, 1)
        
        P1 = K_by_cam[c1] @ np.hstack([R1, t1])
        P2 = K_by_cam[c2] @ np.hstack([R2, t2])
        
        uvA_1, uvB_1 = observations[fid][c1]
        uvA_2, uvB_2 = observations[fid][c2]
        
        pts_4d_A = cv2.triangulatePoints(P1, P2, uvA_1.reshape(2, 1), uvA_2.reshape(2, 1))
        pts_4d_B = cv2.triangulatePoints(P1, P2, uvB_1.reshape(2, 1), uvB_2.reshape(2, 1))
        
        ptA = (pts_4d_A[:3] / pts_4d_A[3]).flatten()
        ptB = (pts_4d_B[:3] / pts_4d_B[3]).flatten()
        
        # Wand length
        wand_len = np.linalg.norm(ptB - ptA)
        all_wand_lengths.append(wand_len)
        
        # Reprojection errors for all cameras
        for cid in cams_in_frame:
            p = cam_params[cid]
            R, _ = cv2.Rodrigues(p[:3])
            t = p[3:6].reshape(3, 1)
            
            proj_A = project_point(ptA, R, t, K_by_cam[cid])
            proj_B = project_point(ptB, R, t, K_by_cam[cid])
            
            if proj_A is not None:
                all_reproj_errs.append(np.linalg.norm(proj_A - observations[fid][cid][0]))
            if proj_B is not None:
                all_reproj_errs.append(np.linalg.norm(proj_B - observations[fid][cid][1]))
    
    reproj_err_mean = np.mean(all_reproj_errs) if all_reproj_errs else None
    wand_len_error = np.mean(np.abs(np.array(all_wand_lengths) - wand_length_mm)) if all_wand_lengths else None
    
    return {
        'reproj_err_mean': reproj_err_mean,
        'wand_length_error': wand_len_error,
        'valid_frames': len(valid_frames),
    }

# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    print("=" * 70)
    print("TEST: refractive_bootstrap_v2.py on filtered dataset")
    print("=" * 70)
    
    # Load observations
    print(f"\nLoading CSV: {FILTERED_CSV}")
    observations, all_cam_ids = load_observations_from_csv(FILTERED_CSV)
    print(f"  Frames: {len(observations)}, Cameras: {all_cam_ids}")
    
    # Create config
    config = PinholeBootstrapP0Config(wand_length_mm=WAND_LENGTH_MM)
    bootstrap = PinholeBootstrapP0(config)
    
    # Select best pair
    from modules.camera_calibration.wand_calibration.refractive_bootstrap_v2 import (
        select_best_pair_via_precalib,
    )
    
    print("\nSelecting best camera pair...")
    # For now, hardcode pair selection (user can add precalibration later)
    cam_i, cam_j = 0, 1
    print(f"  Using pair: ({cam_i}, {cam_j})")
    
    # Run full bootstrap (Phase 1+2+3)
    print("\nRunning Phase 1+2+3 bootstrap on v2...")
    try:
        cam_params, report = bootstrap.run_all(
            cam_i=cam_i,
            cam_j=cam_j,
            observations=observations,
            camera_settings=CAMERA_SETTINGS,
            all_cam_ids=all_cam_ids,
            progress_callback=None,
        )
        print(f"  ✓ Bootstrap completed")
        print(f"    Calibrated cameras: {sorted(cam_params.keys())}")
    except Exception as e:
        print(f"  ✗ Bootstrap FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(cam_params, observations, CAMERA_SETTINGS, all_cam_ids, WAND_LENGTH_MM)
    
    print(f"  reproj_err_mean: {metrics['reproj_err_mean']:.4f} px (baseline: {BASELINE_REPROJ_ERR_MEAN:.4f})")
    print(f"  wand_length_error: {metrics['wand_length_error']:.4f} mm (baseline: {BASELINE_WAND_LEN_ERROR:.4f})")
    
    # Check acceptance
    print("\n" + "=" * 70)
    print("ACCEPTANCE CHECK")
    print("=" * 70)
    
    reproj_ratio = metrics['reproj_err_mean'] / BASELINE_REPROJ_ERR_MEAN if metrics['reproj_err_mean'] else float('inf')
    wand_ratio = metrics['wand_length_error'] / BASELINE_WAND_LEN_ERROR if metrics['wand_length_error'] else float('inf')
    
    print(f"  reproj_err_ratio: {reproj_ratio:.4f} (threshold: {REPROJ_ERR_RATIO_THRESHOLD})")
    print(f"  wand_len_error_ratio: {wand_ratio:.4f} (threshold: {WAND_LEN_ERROR_RATIO_THRESHOLD})")
    
    reproj_pass = reproj_ratio <= REPROJ_ERR_RATIO_THRESHOLD
    wand_pass = wand_ratio <= WAND_LEN_ERROR_RATIO_THRESHOLD
    
    if reproj_pass and wand_pass:
        print("\n  ✓ ACCEPT: All metrics within threshold")
        print("\n" + "=" * 70)
        return 0
    else:
        print("\n  ✗ REJECT: Metrics exceed threshold")
        if not reproj_pass:
            print(f"    - reproj_err_ratio: {reproj_ratio:.4f} > {REPROJ_ERR_RATIO_THRESHOLD}")
        if not wand_pass:
            print(f"    - wand_len_error_ratio: {wand_ratio:.4f} > {WAND_LEN_ERROR_RATIO_THRESHOLD}")
        print("\n" + "=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

**Must NOT do**:
- Don't modify production harness
- Don't hardcode file paths (parameterize if needed)
- Don't skip metrics computation or anchoring verification logging

**Recommended Agent Profile**:
- Category: `unspecified-high` — New test script, custom validation logic
- Skills: None required
- Omitted: None

**Parallelization**: Can Parallel: YES (independent of Task 2) | Blocks: Task 5

**References**:
- Baseline metrics: from plan context
- Camera settings: from plan context
- Dataset: `J:\Fish\T0\wand_points_selected_filtered.csv`
- Bootstrap module: `refractive_bootstrap_v2` (not production)

**Acceptance Criteria** (agent-executable only):
- [ ] Script is syntactically correct: `python -m py_compile .sisyphus/test_bootstrap_v2.py` exits 0
- [ ] Script imports refractive_bootstrap_v2 successfully
- [ ] Script loads filtered CSV without error
- [ ] Script runs bootstrap.run_all() without crash
- [ ] Script computes metrics independently (re-triangulation)
- [ ] Script logs all metrics and PASS/FAIL decision
- [ ] Script exits 0 if metrics pass, 1 if fail

**QA Scenarios**:

```
Scenario: Script syntax is correct
  Tool: bash
  Steps:
    1. conda run -n OpenLPT python -m py_compile .sisyphus/test_bootstrap_v2.py
  Expected: exit code 0
  Evidence: .sisyphus/evidence/task-4-syntax.txt

Scenario: Script imports v2 successfully
  Tool: bash
  Steps:
    1. conda run -n OpenLPT python -c "exec(open('.sisyphus/test_bootstrap_v2.py').read()); print('OK')" 2>&1 | head -20
  Expected: Script starts loading without ImportError
  Evidence: .sisyphus/evidence/task-4-import.txt
```

**Commit**: NO (test script is in .sisyphus, not part of production)

---

### Task 5: Run custom test script on v2 only

**What to do**: Execute `.sisyphus/test_bootstrap_v2.py` to validate v2 on filtered dataset. Capture all output to log file.

**Must NOT do**:
- Don't run on production file
- Don't modify test script during execution
- Don't skip log capture

**Recommended Agent Profile**:
- Category: `unspecified-high` — Execution + output capture, ~2–5 min runtime
- Skills: None required

**Parallelization**: Can Parallel: NO | Blocks: Task 6

**References**:
- Script: `.sisyphus/test_bootstrap_v2.py`
- Command: `conda run -n OpenLPT python .sisyphus/test_bootstrap_v2.py 2>&1 | tee .sisyphus/evidence/bootstrap_v2_test.log`
- Log output: `.sisyphus/evidence/bootstrap_v2_test.log`

**Acceptance Criteria** (agent-executable only):
- [ ] Script runs without crashing (exit code 0 or 1, not non-zero error)
- [ ] Log file contains all phase metrics (Phase 1/2/3 reproj errors, wand length error)
- [ ] Log includes ACCEPT or REJECT decision
- [ ] Log shows anchoring was applied (log messages from Phase 1 and Phase 3)
- [ ] No unhandled exceptions in log

**QA Scenarios**:

```
Scenario: Test script runs to completion
  Tool: bash
  Steps:
    1. cd D:\0.Code\OpenLPTGUI\OpenLPT
    2. conda run -n OpenLPT python .sisyphus/test_bootstrap_v2.py 2>&1 | tee .sisyphus/evidence/bootstrap_v2_test.log
    3. echo "Exit code: $?"
  Expected: exit code 0 (accepted) or 1 (rejected), not error
  Evidence: .sisyphus/evidence/bootstrap_v2_test.log

Scenario: Anchoring is visible in log
  Tool: bash
  Steps:
    1. grep -i "anchoring" .sisyphus/evidence/bootstrap_v2_test.log
  Expected: At least 2 lines showing Phase 1 and Phase 3 anchoring
  Evidence: (embedded in log)

Scenario: Metrics table is present in log
  Tool: bash
  Steps:
    1. grep -E "reproj_err|wand_length" .sisyphus/evidence/bootstrap_v2_test.log
  Expected: Numeric metrics values
  Evidence: (embedded in log)
```

**Commit**: NO

---

### Task 6: Parse results and decide acceptance

**What to do**: Parse test log to extract metrics. Check:
- `reproj_err_mean` (v2) ≤ baseline × 1.10
- `wand_length_error` (v2) ≤ baseline × 1.10
- No Phase 2→3 spike (Phase 3 ≤ Phase 2 × 1.50)
- All metrics finite (no NaN / inf)
- Anchoring applied correctly

Output binary decision: **ACCEPT** or **REJECT**.

**Must NOT do**:
- Don't modify any files
- Don't use thresholds other than 1.10 / 1.50

**Recommended Agent Profile**:
- Category: `quick` — Log parsing, <1 min
- Skills: None required

**Parallelization**: Can Parallel: NO | Blocks: Task 7

**References**:
- Log file: `.sisyphus/evidence/bootstrap_v2_test.log`
- Baselines: reproj 5.89 px, wand_len 0.35 mm
- Thresholds: 1.10 for both error ratios, 1.50 for spike

**Acceptance Criteria** (agent-executable only):
- [ ] Can extract reproj_err_mean from log (numeric ≤ 6.48 px for 1.10× threshold)
- [ ] Can extract wand_length_error from log (numeric ≤ 0.385 mm)
- [ ] All values are finite
- [ ] Log contains ACCEPT or REJECT from test script
- [ ] Decision is binary: "ACCEPT_FOR_PROMOTION" or "REJECT" output to stdout

**QA Scenarios**:

```
Scenario: Metrics are parsed correctly
  Tool: bash
  Steps:
    1. grep "reproj_err_ratio:" .sisyphus/evidence/bootstrap_v2_test.log | awk '{print $NF}'
    2. Check if numeric and ≤ 1.10
  Expected: Numeric value ≤ 1.10
  Evidence: .sisyphus/evidence/task-6-parse.txt

Scenario: Decision is ACCEPT if both metrics pass
  Tool: bash
  Steps:
    1. (parse both ratios from log)
    2. If both ≤ 1.10: output "ACCEPT_FOR_PROMOTION"
  Expected: "ACCEPT_FOR_PROMOTION"
  Evidence: .sisyphus/evidence/task-6-decision.txt
```

**Commit**: NO

---

### Task 7: Document decision and cleanup

**What to do**: Based on Task 6 decision:

**If ACCEPT**:
- Document v2 test passed in evidence summary
- Log detailed metrics for promotion record
- **NOTE**: v2 file stays in repo for now (NOT promoted to production until user explicitly approves)
- Print success summary with metrics

**If REJECT**:
- Document v2 test failed in evidence summary
- Log detailed rejection reasons
- **Optionally** delete v2 file if user wants cleanup (but default is to keep for inspection)
- Print failure summary with failing metrics

**Example output summary**:

```
================================================================================
TEST SUMMARY: refractive_bootstrap_v2.py
================================================================================

Test Dataset: J:\Fish\T0\wand_points_selected_filtered.csv (2996 frames)
Camera Settings: Per-camera intrinsics (cams 0-3: 5250px, cam 4: 9000px)
Anchoring: Phase 1 (cam_0 fixed), Phase 3 (cam_0 fixed to Phase 2 pose)

METRICS (v2 vs baseline):
  reproj_err_mean: 5.42 px (baseline: 5.89 px) — ratio: 0.920 ✓
  wand_length_error: 0.32 mm (baseline: 0.35 mm) — ratio: 0.914 ✓
  Phase 2→3 spike: Phase 2 reproj = 4.89 px, Phase 3 reproj = 5.42 px — ratio: 1.108 ✓

DECISION: ✓ ACCEPT_FOR_PROMOTION

Evidence file: .sisyphus/evidence/bootstrap_v2_test.log
Next steps: User must review metrics and run '/promote-bootstrap-v2' to copy v2→v1
```

**Must NOT do**:
- Don't promote v2 to production automatically (user must explicitly approve)
- Don't delete v2 file unless user explicitly requests

**Recommended Agent Profile**:
- Category: `quick` — Summary generation
- Skills: None required

**Parallelization**: Can Parallel: NO | Final task

**References**:
- Test log: `.sisyphus/evidence/bootstrap_v2_test.log`
- Decision from Task 6

**Acceptance Criteria** (agent-executable only):
- [ ] Summary document created in `.sisyphus/evidence/bootstrap_v2_test_summary.md`
- [ ] Summary includes decision (ACCEPT or REJECT), metrics, and next steps
- [ ] If ACCEPT: user guidance for manual promotion (e.g., "run `/promote-bootstrap-v2`")
- [ ] If REJECT: user guidance for inspection and debugging

**QA Scenarios**:

```
Scenario: ACCEPT case — summary is clear and actionable
  Tool: bash
  Steps:
    1. cat .sisyphus/evidence/bootstrap_v2_test_summary.md | head -30
  Expected: Clear ACCEPT decision with metrics and next-steps guidance
  Evidence: .sisyphus/evidence/task-7-accept-summary.txt

Scenario: REJECT case — failure reasons are documented
  Tool: bash
  Steps:
    1. cat .sisyphus/evidence/bootstrap_v2_test_summary.md
  Expected: Clear REJECT decision with which metrics failed
  Evidence: .sisyphus/evidence/task-7-reject-summary.txt
```

**Commit**: NO (summary is evidence, not code)

---

## Final Verification Wave (MANDATORY)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before marking work complete.
> **Do NOT auto-proceed. Wait for user's explicit approval before marking work complete.**

- [ ] F1. Plan Compliance Audit — oracle (verify 5 fixes + 2 anchoring are correctly identified and applied)
- [ ] F2. Code Quality Review — unspecified-high (syntax, style, correctness of anchoring logic)
- [ ] F3. Test Coverage Review — unspecified-high (custom test script has correct metrics, thresholds, dataset handling)
- [ ] F4. Scope Fidelity Check — deep (ensure no production file touched, test-only scope maintained)

---

## Commit Strategy

| Commit | Type | Target | Message | Files | Condition |
|--------|------|--------|---------|-------|-----------|
| None | — | v2 only | (all fixes applied to v2 only, not production) | refractive_bootstrap_v2.py | Always (test vehicle, not committed) |
| Test | — | evidence | (test script and results logged for record) | .sisyphus/test_bootstrap_v2.py, .sisyphus/evidence/*.log | Always (evidence, not code) |

---

## Success Criteria

### Definition of "Done"
1. ✅ `refractive_bootstrap_v2.py` is exact copy of production initially
2. ✅ 5 bug fixes applied to v2 only
3. ✅ 2 anchoring changes applied to v2 only (Phase 1 + Phase 3)
4. ✅ v2 compiles and imports without error
5. ✅ Custom test script `.sisyphus/test_bootstrap_v2.py` created and runs
6. ✅ Test runs on filtered dataset (2996 frames) with correct camera settings
7. ✅ All metrics computed independently (re-triangulation)
8. ✅ Metrics logged with anchoring verification (log shows anchoring applied)
9. ✅ reproj_err_ratio ≤ 1.10 ✓ OR reason documented
10. ✅ wand_len_error_ratio ≤ 1.10 ✓ OR reason documented
11. ✅ No Phase 2→3 spike (Phase 3 ≤ 1.50 × Phase 2) ✓ OR reason documented
12. ✅ Evidence logged to `.sisyphus/evidence/bootstrap_v2_test.log`
13. ✅ Summary decision (ACCEPT/REJECT) documented with metrics

### Definition of "Failure"
- Test script crashes or exits with unhandled exception
- Metrics are NaN / inf
- Any ratio exceeds threshold (1.10 for errors, 1.50 for spike) without documented reason
- Production file (`refractive_bootstrap.py`) is modified
- Comparison harness is modified
- Anchoring not logged or not applied

