# Auto-Debug Iterative Loop: Fix Filtered Observation Calibration

## TL;DR

> **Goal**: Achieve **≤10% degradation** on BOTH metrics:
> 1. `reproj_err_ratio = filtered_reproj_err / original_reproj_err ≤ 1.10`
> 2. `wand_length_error_ratio = filtered_wand_len_error / original_wand_len_error ≤ 1.10`
>
> **Setup**: 
> - Original dataset: 3000 frames, wand_points_selected.csv
> - Filtered dataset: 2996 frames (4 outliers removed: 2412, 2964, 2984, 2991)
>
> **Workflow**: Auto-loop until BOTH metrics pass
> - Each iteration: Oracle analysis → apply fix → run full comparison
> - Record evidence: analysis, modification, result
> - Loop stops when: reproj_err_ratio ≤ 1.10 AND wand_length_error_ratio ≤ 1.10 AND pytest passes
>
> **Deliverable**: Evidence log showing each iteration's diagnosis, fix, and validation

---

## IMMUTABLE CONSTRAINTS

### Input Parameters (MUST NOT CHANGE) — COMPREHENSIVE LIST

**EXPLICITLY FORBIDDEN** (will be verified post-iteration):

**Camera intrinsic parameters** (from calibration, hardcoded or config):
- ❌ Focal length (fx, fy) — FIXED, DO NOT CHANGE
- ❌ Principal point (cx, cy) — FIXED, DO NOT CHANGE
- ❌ Distortion coefficients (k1, k2, p1, p2, k3) — FIXED, DO NOT CHANGE
- ❌ Camera matrix K — FIXED, DO NOT CHANGE

**Wand object parameters** (from config):
- ❌ `wand_length_mm = 10.0` — FIXED, DO NOT CHANGE
- ❌ Wand diameter or detection threshold — FIXED, DO NOT CHANGE
- ❌ Wand endpoint detection method — FIXED, DO NOT CHANGE

**Refraction/optical setup parameters** (from config):
- ❌ Refractive plate thickness — FIXED, DO NOT CHANGE
- ❌ Refractive index (water/glass/air interface) — FIXED, DO NOT CHANGE
- ❌ Camera-to-window distance (air gap) — FIXED, DO NOT CHANGE
- ❌ Camera-window geometric mapping (camera frame ↔ world frame transform) — FIXED, DO NOT CHANGE
- ❌ Window surface normal vector — FIXED, DO NOT CHANGE
- ❌ Refraction correction formula or parameters — FIXED, DO NOT CHANGE

**Dataset constraints** (user-provided, immutable):
- ❌ 2996 filtered frames ONLY — DO NOT restore removed 4 frames (2412, 2964, 2984, 2991)
- ❌ Original dataset: 3000 frames — DO NOT modify, do not subsample
- ❌ Wand point coordinates — DO NOT modify or pre-process
- ❌ Camera images — DO NOT modify or filter before bootstrap

### Code Modification Rules

- ✅ Modifications ONLY to `refractive_bootstrap_debug.py` — DO NOT touch `refractive_bootstrap.py` (production)
- ✅ Allowed: Algorithm logic changes (seed selection, filtering, optimization, pose estimation)
- ✅ Allowed: Robustness improvements (better chirality detection, scale estimation)
- ✅ Allowed: Algorithm tuning parameters (RANSAC iterations, inlier threshold, BA regularization, BA tolerance)
- ✅ NOT allowed: Changing immutable config/input values (camera K, focal, wand_length_mm, refraction setup, distortion coefficients)
- ✅ NOT allowed: Pre-processing data before bootstrap
- ✅ All code changes in repo root: `D:\0.Code\OpenLPTGUI\OpenLPT`

---

## Auto-Debug Loop (Executor Follows This)

### Loop Preconditions (Setup)

- [ ] **Setup 1**: Fresh copy of production code
  - `refractive_bootstrap.py` (production, untouched) ✓
  - `refractive_bootstrap_debug.py` (copy of production, will modify) ✓
  - Verify import: `python -c "from modules.camera_calibration.wand_calibration.refractive_bootstrap_debug import PinholeBootstrapP0; print('OK')"`

- [ ] **Setup 2**: Test datasets available
  - Original: `J:\Fish\T0\wand_points_selected.csv` (3000 frames)
  - Filtered: `J:\Fish\T0\wand_points_selected_filtered.csv` (2996 frames)

- [ ] **Setup 3**: Evidence recording
  - Create directory: `.sisyphus/evidence/debug_iterations/`
  - For each iteration `N`: `iter_N_metis_analysis.md`, `iter_N_result.log`

---

### ITERATION LOOP (Repeat until stop criteria met)

#### **ITERATION N: [CYCLE_DESC]**

**Oracle ANALYSIS PHASE**

- [ ] **N.A1**: Get Oracle diagnosis
  
  Oracle reviews current code state and failure:
  - Current failure: [reproj_err_ratio / wand_len_error_ratio status]
  - Root cause analysis
  - Recommended modification to `refractive_bootstrap_debug.py`
  - Specific lines to change + before/after pseudocode
  - **CONSTRAINT CHECK**: Verify no input parameters are being changed
  - **Requested Diagnostics** (optional): Any additional outputs/data needed for next iteration's analysis (e.g., intermediate values, debug logs, visualizations)
  
  **Save to**: `.sisyphus/evidence/debug_iterations/iter_N_metis_analysis.md`
  
  **Format** (include section if Oracle requests diagnostics):
  ```markdown
  ## Requested Diagnostics for Iteration N+1
  - [diagnostic_1]: [what to log/output and why]
  - [diagnostic_2]: [specific format or location]
  ```

**IMPLEMENTATION PHASE**

- [ ] **N.B1**: Apply modification to `refractive_bootstrap_debug.py`
  
  - Read Oracle recommendation from saved file
  - Delegate a coding subagent for implementing modification
  - **CONSTRAINT CHECK**: Verify NO IMMUTABLE input parameters are being modified (compare against IMMUTABLE list above)
    - ❌ NO changes to camera K, distortion, focal length, principal point
    - ❌ NO changes to wand_length_mm, wand thresholds
    - ❌ NO changes to refraction plate, refractive index, camera-window distance
    - ❌ NO modifications to input CSV files or frame count
    - ✅ ALLOWED: Modifications to RANSAC iterations, RANSAC threshold, BA regularization, BA tolerance (algorithm tuning is permitted)
  - Identify exact lines to modify (line numbers + context)
  - Apply change (edit or write new section) — algorithm logic ONLY
  - Verify import still works: `python -c "from modules...refractive_bootstrap_debug import PinholeBootstrapP0; print('OK')"`
  - Verify no syntax errors: `python -m py_compile modules/camera_calibration/wand_calibration/refractive_bootstrap_debug.py`
  - **FINAL CHECK**: Diff debug file against production file to confirm NO immutable input parameters changed

**COMPARISON PHASE**

- [ ] **N.B2**: Run full bootstrap comparison (both datasets, all phases 1+2+3)
  
  **CRITICAL**: The comparison script **must test the full bootstrap pipeline** (`run_all()` with phases 1, 2, and 3), not just Phase 1 initialization.

  You should run full bootstrap comparison and keep tracking the running status. (Do not ask subagent to run the comparison.)
  
  `scripts/full_bootstrap_comparison.py` must:
  1. Call `PinholeBootstrapP0.run_all()` instead of `.run()` 
  2. Execute Phase 1 (extrinsic initialization with BA)
  3. Execute Phase 2 (multi-camera extension)
  4. Execute Phase 3 (full multi-camera joint BA)
  5. Return final multi-camera calibration metrics for both datasets
  
  **Rationale**: Phase 1 fixes (pose selection, scale recovery, BA regularization) are necessary but not sufficient. The full pipeline may exhibit different behavior on the filtered dataset due to:
  - Reduced frame diversity affecting Phase 2 multi-camera initialization
  - Weaker constraints in Phase 3 joint BA with fewer high-parallax frames
  - Cumulative error propagation through 3-phase optimization
  
  **Modification to comparison script**:
   ```python
   # Replace: params_i, params_j, report = bootstrap.run(...)
   # With: cam_params_all, report_all = bootstrap.run_all(
   #         cam_i=cam_i, cam_j=cam_j,
   #         observations=observations,
   #         camera_settings=camera_settings,
   #         all_cam_ids=all_cam_ids,
   #         progress_callback=None
   #       )
   ```

   **IMPORTANT — Per-Camera Intrinsics Override**:
   
   The comparison script must use the EXACT per-camera intrinsics from the GUI table.
   These values come from the calibration session that produced the 11549 px result — they
   are the ground-truth camera parameters for this dataset:
   
   ```python
   # PER-CAMERA SETTINGS (from GUI table, confirmed by user log)
   # Format: {cam_id: {'focal': float, 'width': float, 'height': float}}
   # NOTE: size=(height, width) in GUI print, but camera_settings stores (width, height)
   
   camera_settings = {
       0: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
       1: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
       2: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
       3: {'focal': 5250.0, 'width': 1280.0, 'height': 800.0},
       4: {'focal': 9000.0, 'width': 1024.0, 'height': 976.0},
   }
   ```
   
   - cams 0-3: focal=5250 px, image size=1280(W)×800(H) px
   - cam 4:     focal=9000 px, image size=1024(W)×976(H) px
   - cx = width/2, cy = height/2 for each camera (principal point = image center)
   
   **DO NOT** use the script's default `build_camera_settings()` which produces
   uniform `{focal=9000, width=4096, height=3000}` for all cameras — this is
   a different configuration and will produce different results.
   
   Replace the script's `build_camera_settings()` call with the per-camera dict above.
   The `build_camera_settings()` function at line 136 is a helper that sets uniform
   values — it must be replaced entirely for this comparison to match GUI conditions.

   **IMPORTANT — Camera-to-Window Mapping Override**:

   The comparison script must use the EXACT camera-to-window mapping from the GUI session:

   ```python
   # CAMERA-TO-WINDOW MAPPING (from GUI Mapping Snapshot)
   cam_to_window = {
       0: 0,
       1: 0,
       2: 0,
       3: 0,
       4: 1,
   }
   ```

   - Cameras 0, 1, 2, 3 belong to Window 0 (first tank/view)
   - Camera 4 belongs to Window 1 (second tank/view)
   - The mapping determines which refraction plate geometry is applied per camera

   **IMPORTANT — Refraction Plate / Media Parameters Override**:

   Both windows use identical media parameters (from GUI log):

   ```python
   # MEDIA PARAMETERS (from GUI Media Parameters section)
   window_media = {
       0: {
           'n_air': 1.000,   # Refractive index of air
           'n_win': 1.490,   # Refractive index of window (glass)
           'n_obj': 1.330,   # Refractive index of object medium (water)
           'thickness': 31.75,  # Window thickness in mm
       },
       1: {
           'n_air': 1.000,
           'n_win': 1.490,
           'n_obj': 1.330,
           'thickness': 31.75,
       },
   }
   ```

   - n_air=1.000 (air, fixed)
   - n_win=1.490 (glass/ acrylic plate)
   - n_obj=1.330 (water)
   - thickness=31.75 mm for BOTH windows
   - These values are passed to `RefractiveBAOptimizer` and the C++ camera state
  
  **Expected output format** (after full pipeline):
  ```
  Original Dataset Results (After Phase 3):
    Baseline (P0): XXX.XX mm
    Reproj Error (Phase 3 final): X.XXXX px
    Wand Length Error (Phase 3 final): X.XXXX mm
    N Cameras Calibrated: X
  
  Filtered Dataset Results (After Phase 3):
    Baseline (P0): XXX.XX mm
    Reproj Error (Phase 3 final): X.XXXX px
    Wand Length Error (Phase 3 final): X.XXXX mm
    N Cameras Calibrated: X
  
  COMPARISON (Phase 3 final metrics):
    reproj_err_ratio = filtered / original = X.XXXX
    wand_len_error_ratio = filtered / original = X.XXXX
    
    reproj_err_ratio ≤ 1.10? [PASS/FAIL]
    wand_len_error_ratio ≤ 1.10? [PASS/FAIL]
  ```
  
  **Run comparison**:
  ```powershell
  $py = "C:\Users\tan_s\miniconda3\envs\OpenLPT\python.exe"
  & $py scripts/full_bootstrap_comparison.py --debug --run-all 2>&1 | tee .sisyphus/evidence/debug_iterations/iter_N_result.log
  ```
  
  (Note: Add `--run-all` flag to comparison script to trigger full pipeline execution)

- [ ] **N.B3** (Conditional): Collect requested diagnostics (IF Oracle requested in N.A1)
  
  If Oracle analysis included "Requested Diagnostics for Iteration N+1" section:
  - Extract each diagnostic request from `.sisyphus/evidence/debug_iterations/iter_N_metis_analysis.md`
  - Add debug output/logging to `refractive_bootstrap_debug.py` to capture requested data
  - Run comparison again with debug output enabled: 
    ```powershell
    $py = "C:\Users\tan_s\miniconda3\envs\OpenLPT\python.exe"
    & $py scripts/full_bootstrap_comparison.py --debug --verbose 2>&1 | tee .sisyphus/evidence/debug_iterations/iter_N_diagnostics.log
    ```
  - Save diagnostic output to: `.sisyphus/evidence/debug_iterations/iter_N_diagnostics.log`
  - If no diagnostics requested, skip this step

**EVALUATION PHASE**

- [ ] **N.C1**: Check loop termination criteria
  
  **PASS conditions** (ALL must be true):
  - ✅ reproj_err_ratio ≤ 1.10
  - ✅ wand_len_error_ratio ≤ 1.10
  - ✅ Unit tests pass: `pytest modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py -v`
  
  **If ALL PASS**: Go to Final Verification Wave (end of plan)
  
  **If ANY FAIL**: Continue to next iteration (increment N, go back to ITERATION N+1 start)

- [ ] **N.C2**: Record iteration summary
  
  **Append to**: `.sisyphus/notepads/auto_debug_filtered_plan/iterations.md`
  
  ```markdown
  ## Iteration N: [TITLE]
  
  **Oracle Analysis**: [1-2 sentence diagnosis]
  **Recommended Fix**: [1-2 sentence description of modification]
  **Constraints Verified**: Yes (no input parameters changed)
  **Result**: 
  - reproj_err_ratio: X.XXXX [PASS/FAIL]
  - wand_len_error_ratio: X.XXXX [PASS/FAIL]
  **Diagnostics Collected**: [YES/NO - if YES, see iter_N_diagnostics.log]
  **Diagnostics for Next Iteration**: [if Oracle requested new outputs, list them here]
  **Status**: [CONTINUE / SUCCESS]
  ```

---

## AFTER LOOP: Final Verification Wave

Once loop criteria met (both metrics ≤ 1.10 AND tests pass):

- [ ] **F1**: Run unit tests (verify no regressions)
  ```powershell
  $py = "C:\Users\tan_s\miniconda3\envs\OpenLPT\python.exe"
  & $py -m pytest modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py -v | tee .sisyphus/evidence/debug_iterations/final_unit_tests.log
  ```
  **Must**: All tests PASS (exit code 0)

- [ ] **F2**: Record final evidence
  - Copy final comparison result to `.sisyphus/evidence/final_comparison.log`
  - Copy final unit test output to `.sisyphus/evidence/final_unit_tests.log`

- [ ] **F3**: Document completion
  
  **Create**: `.sisyphus/evidence/DEBUG_SUCCESS_SUMMARY.md`
  
  ```markdown
  # Debug Session: SUCCESSFUL ✓
  
  **Total Iterations**: N
  **Final Metrics**:
  - reproj_err_ratio: X.XXXX (target: ≤ 1.10)
  - wand_len_error_ratio: X.XXXX (target: ≤ 1.10)
  - Unit tests: PASS
  
  **All Constraints Verified**: YES
  - No input parameters changed
  - No config values modified
  - 2996 filtered frames used (no restoration)
  - Modifications only to debug file (production untouched)
  
  **Key Modifications to refractive_bootstrap_debug.py**:
  - [Modification 1]: [brief description]
  - [Modification 2]: [brief description]
  
  **Iteration History**: See `.sisyphus/notepads/auto_debug_filtered_plan/iterations.md`
  ```

---

## Key Files

### Reference (DO NOT MODIFY)
- `modules/camera_calibration/wand_calibration/refractive_bootstrap.py` — production code

### Working Copy (MODIFY ONLY THIS)
- `modules/camera_calibration/wand_calibration/refractive_bootstrap_debug.py` — debug copy

### Test Script
- `scripts/full_bootstrap_comparison.py` — runs both datasets, compares metrics

### Evidence Directory
- `.sisyphus/evidence/debug_iterations/` — iteration logs and analyses

### Notepad
- `.sisyphus/notepads/auto_debug_filtered_plan/iterations.md` — cumulative iteration log

---

## Important Notes

### What "Debug" Means Here (ALGORITHM LOGIC + TUNING PARAMETERS)

Allowed modifications:
- ✅ Chirality detection logic (how to pick between 4 pose candidates)
- ✅ Scale estimation method (how to compute baseline from wand length)
- ✅ Seed selection metric (how to pick best initial BA seed)
- ✅ Outlier detection/filtering in optimization
- ✅ Pose refinement algorithm
- ✅ Algorithm tuning parameters (RANSAC iterations, inlier threshold, BA regularization, BA tolerance)

NOT allowed:
- ❌ Changing IMMUTABLE config values (wand_length_mm, camera K, focal, distortion, refraction setup)
- ❌ Pre-processing dataset before bootstrap
- ❌ Modifying immutable input parameters or their sources
- ❌ Changing refractive geometry assumptions
- ❌ Adjusting camera calibration parameters
- ❌ Restoring removed frames or modifying dataset

### Oracle Role
Oracle analyzes why the current attempt failed and recommends the NEXT targeted fix. This is NOT a "try random things" loop—each iteration should build on diagnosis.

### Stopping Guarantee
The loop WILL terminate because:
1. Each iteration gets Oracle analysis (not guessing)
2. Each modification targets a real root cause
3. Each modification is constrained (algorithm logic only, inputs unchanged)
4. Maximum ~5 iterations expected
