# Learnings — axis-direction-world-alignment

## [2026-03-22] Session Start
- Plan: 10 tasks across 4 waves
- Wave 1 (Tasks 1, 2, 3): Parallel allowed for tasks 2 and 3; task 1 is independent but blocks tasks 4,7,8
- Wave 2 (Tasks 4, 5, 6): Tasks 4 and 5 can parallel; task 6 independent
- Wave 3 (Tasks 7, 8): Can parallel (pinhole and refractive hooks independent)
- Wave 4 (Tasks 9, 10): Can parallel

## Key Files
- `modules/camera_calibration/wand_calibration/wand_calibrator.py` — main calibrator (lines 2635-2786 for finalization, 2867-2937 for triangulation)
- `modules/camera_calibration/wand_calibration/refractive_geometry.py` — geometry utilities (lines 229-389 for rays, 624-709 for apply_coordinate_rotation)
- `modules/camera_calibration/wand_calibration/refraction_wand_calibrator.py` — refractive calibrator
- `modules/camera_calibration/view.py` — GUI view (lines 1553 for buttons, 3168-3229 for _save_axis_data, 5232 for _on_calibration_finished, 7480 for _on_refractive_finished)
- `.sisyphus/test_bootstrap_v2.py` — test pattern reference

## Codebase Conventions
- Python 3.10+ with numpy/scipy/cv2
- Tests go in `tests/` directory
- Evidence artifacts go in `.sisyphus/evidence/`
- Conda env: OpenLPT

## Must NOT Change
- Intrinsic camera matrix K
- Distortion coefficients dist
- Scale or shear transforms (rigid only)
- Alignment during optimization (post-calibration only)



## [2026-03-22] Task 1 Notes
- Fixed pinhole finalize state-frame mismatch by passing recentered camera slice into _parse_results via params_consistent.
- Added regression test test_pinhole_finalize_consistency in tests/test_axis_alignment.py for centroid-zero and camera-translation consistency.
- Verification evidence written to .sisyphus/evidence/task-1-finalize-fix-tests.txt and .sisyphus/evidence/task-1-centroid-preservation.txt.

## Task 3: Reference Audit of axis_direction_map (2026-03-22)

### Key Findings

**Data Flow Summary:**
1. **Definition**: `axis_direction_map = {}` initialized in `_save_axis_data()` (view.py:3169)
2. **Population**: Populated from `_compute_axis_output_for_cam(cam_idx)` (view.py:3174-3178)
3. **Storage**: 
   - In-memory: `self.axis_direction_map` (view.py:3188)
   - Disk: CSV file with 8 coordinate columns (view.py:3203-3224)
4. **Consumption**: **NONE** - Data is write-only, never loaded or used downstream

**Schema (Confirmed):**
```python
{
    cam_idx: {
        "center": [x_pixel, y_pixel],  # Mean of edge midpoints
        "+X": [x_pixel, y_pixel],      # User-selected +X axis point
        "+Y": [x_pixel, y_pixel],      # User-selected +Y axis point
        "+Z": [x_pixel, y_pixel]       # User-selected +Z axis point
    }
}
```

**CSV Format:**
```
cam_id, center_x, center_y, plus_x_x, plus_x_y, plus_y_x, plus_y_y, plus_z_x, plus_z_y
```

**Current Status:**
- ✅ Data capture fully implemented
- ✅ CSV export functional
- ❌ NO loader function exists
- ❌ NO downstream consumers (refractive calibration, world alignment, etc.)
- ❌ Data saved but never utilized

**Implementation Details:**
- UI trigger: `btn_save_axis.clicked` → `_save_axis_data()` (view.py:1198)
- Center computation: Mean of paired edge midpoints (view.py:3499)
- Validation: All axis points checked via `_find_pair_for_selected_axis()`
- Coordinates: 2D image-space pixels only

**Next Steps Required:**
1. Implement CSV loader function
2. Create world-space transformation module
3. Integrate with camera calibration pipeline
4. Add refractive correction support

## [2026-03-22] Task 2 Notes
- Added standalone synthetic axis-alignment test utilities in `tests/test_utils_axis_alignment.py` with deterministic camera/point generation, DLT N-view triangulation, orthogonality metrics, and RMS reprojection evaluation.
- Kept utility module independent from `modules/camera_calibration/*` imports to preserve pure test-scaffold behavior.
- Added baseline tests in `tests/test_axis_alignment_baseline.py` to validate setup integrity, orthogonality metric sanity, and near-zero noiseless reprojection baseline.
- Captured execution evidence in `.sisyphus/evidence/task-2-synthetic-setup.txt` using `conda run -n OpenLPT python -m pytest ...`; observed `3 passed`.

## [2026-03-22] Task 6 Notes
- Added optional calibration-tab button `Load Axis Points (Optional)` directly below `Load Wand Points (from CSV)` and connected it to `_load_axis_points_csv`.
- Implemented `_load_axis_points_csv(file_path=None)` in `CameraCalibrationView` using `QFileDialog` + `csv.DictReader` to parse: `cam_id, center_x, center_y, plus_x_x, plus_x_y, plus_y_x, plus_y_y, plus_z_x, plus_z_y`.
- Loader populates `self.axis_direction_map` with `{cam_id: {"center": [...], "+X": [...], "+Y": [...], "+Z": [...]}}`, shows `QMessageBox.critical` on parse/load failure, and disables `btn_load_axis_csv` after successful non-empty load.
- Verified syntax with `conda run -n OpenLPT python -m py_compile modules/camera_calibration/view.py`; wrote CSV happy-path evidence to `.sisyphus/evidence/task-6-csv-loader-happy.txt`.

## [2026-03-22] Task 4 Notes
- Added `align_world_to_axis_directions(...)` immediately after `apply_coordinate_rotation(...)` in `refractive_geometry.py`.
- Matched the real `apply_coordinate_rotation` signature: `(R_world, cam_params, window_planes, points_3d, t_shift=...)`, and returned a `transformed_state` dict containing transformed `cam_params`, `window_planes`, and `points_3d`.
- Added two focused tests to `tests/test_axis_alignment.py`: a happy-path orthonormality/translation check and a coverage-validation failure check.
- Verification: `py_compile` passed and `pytest tests/test_axis_alignment.py -v` passed (`3 passed`), evidence file updated at `.sisyphus/evidence/task-4-alignment-happy-path.txt`.

## [2026-03-22] Task 5 Notes
- Added two mode-specific axis landmark triangulation adapters in `refractive_geometry.py` immediately after `align_world_to_axis_directions`: `triangulate_pinhole_landmarks` (N-view DLT with `P=K[R|T]`) and `triangulate_refractive_landmarks` (ray-casting via `build_pinplate_rays_cpp_batch` + `triangulate_point`).
- Enforced per-landmark failure semantics with explicit `ValueError` on insufficient observations/rays and unsuccessful triangulation return status.
- Added `test_triangulate_pinhole_landmarks_happy` in `tests/test_axis_alignment.py` using synthetic pinhole setup utilities and validated finite `(3,)` outputs plus sub-1.0 distance to ground truth for `center/+X/+Y/+Z`.
- Captured pytest evidence in `.sisyphus/evidence/task-5-pinhole-adapter-happy.txt`; `py_compile` for `refractive_geometry.py` also succeeded with the OpenLPT interpreter.

## [2026-03-22] Task 7 Notes
- Added a post-calibration pinhole alignment hook in `CameraCalibrationView._on_calibration_finished` directly after the first `self._update_3d_viz()` call on the `if success:` path and before the pre-calibration early-return branch.
- Hook behavior: when `self.axis_direction_map` is present, it imports `align_world_to_axis_directions` + `triangulate_pinhole_landmarks` inside a try-block, builds `obs_by_landmark`, runs optional alignment, updates `self.wand_calibrator.final_params` and `self.wand_calibrator.points_3d` only on successful alignment, then refreshes 3D visualization again.
- Failure handling is non-destructive: alignment failure/skip and exceptions are surfaced via `print(...)` warnings while keeping original calibration state unchanged.
- Added `test_pinhole_alignment_roundtrip` in `tests/test_axis_alignment.py` using a synthetic 2-camera setup and real `triangulate_pinhole_landmarks` + `align_world_to_axis_directions` path; test asserts success and that aligned `cam_params` is a dict.
- Important test detail: `align_world_to_axis_directions` currently routes through `apply_coordinate_rotation`, which expects flattened pinhole camera vectors (`[rvec, tvec, f, cx, cy, k1, k2]`) rather than parsed `{"K","R","T","dist"}` dict format for the `cam_params` argument.

## [2026-03-22] Task 8 Notes
- Added a refractive post-calibration axis-alignment hook in `CameraCalibrationView._on_refractive_finished` immediately after the initial `calib_3d_view.plot_refractive(...)` visualization try/except and before `_populate_error_table()`.
- Hook is gated by `if getattr(self, 'axis_direction_map', None):` and performs local imports of `align_world_to_axis_directions` plus `triangulate_refractive_landmarks` per runtime-only integration requirement.
- Alignment path uses a lambda/closure callback (`_triangulate_fn`) that delegates to `triangulate_refractive_landmarks(obs, cams_cpp)` and uses `_refr_cams_cpp` as the source of refractive C++ camera objects.
- Because `_refr_cams_cpp` is not populated in `view.py`, the hook safely logs `[Axis Alignment] Refractive C++ camera objects not available; alignment skipped.` and preserves original calibration state without crashing.
- On alignment success, code updates `self._refr_final_cam_params` and `self._refr_window_planes`, converts aligned points for optional re-render, and calls `self.calib_3d_view.plot_refractive(...)` again with aligned state.
- Verification for this task: `py_compile` passed and `pytest tests/test_axis_alignment.py -v` passed with `5 passed`.
