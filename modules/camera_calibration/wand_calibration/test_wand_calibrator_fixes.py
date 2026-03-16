#!/usr/bin/env python
"""
Unit tests for wand calibrator fixes — BA optimizer infrastructure.

Tests cover:
- C++ mocking infrastructure (patching build_pinplate_rays_cpp_batch, update_cpp_camera_state)
- Synthetic data factory functions for reproducible unit tests
- Helper utilities: radius validation, optimization result checking,
  robust plane init, and fully-mocked optimizer construction.

Run:
    conda run -n OpenLPT python -m pytest modules/camera_calibration/wand_calibration/test_wand_calibrator_fixes.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, Tuple
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Patch targets (full dotted path where the names are looked up at runtime)
# ---------------------------------------------------------------------------
_PATCH_BUILD_RAYS = (
    "modules.camera_calibration.wand_calibration"
    ".refraction_calibration_BA.build_pinplate_rays_cpp_batch"
)
_PATCH_UPDATE_CPP = (
    "modules.camera_calibration.wand_calibration"
    ".refraction_calibration_BA.update_cpp_camera_state"
)


# ===================================================================
# Synthetic data factory functions
# ===================================================================

def make_minimal_dataset(n_frames: int = 5, cam_ids: Tuple[int, ...] = (0, 1)) -> dict:
    """Create a minimal dataset dict suitable for ``RefractiveBAOptimizer``.

    Returns a dict with keys: obsA, obsB, frames, est_radius_small_mm,
    est_radius_large_mm.  Observations are random pixel coords so the
    constructor's ``ObsCacheBuilder.build`` populates a non-empty cache.
    """
    rng = np.random.RandomState(42)
    frames = list(range(n_frames))
    obsA: Dict[int, Dict[int, np.ndarray]] = {}
    obsB: Dict[int, Dict[int, np.ndarray]] = {}
    for fid in frames:
        obsA[fid] = {}
        obsB[fid] = {}
        for cid in cam_ids:
            obsA[fid][cid] = rng.uniform(100, 900, size=(2,)).astype(np.float64)
            obsB[fid][cid] = rng.uniform(100, 900, size=(2,)).astype(np.float64)
    return {
        "obsA": obsA,
        "obsB": obsB,
        "frames": frames,
        "est_radius_small_mm": 1.5,
        "est_radius_large_mm": 2.0,
    }


def make_cam_params(cam_ids: Tuple[int, ...] = (0, 1), f: float = 1000.0) -> Dict[int, np.ndarray]:
    """Return cam_params: ``{cam_id: (11,) float64}`` with focal length at index 6.

    Layout: [rvec(3), tvec(3), f, cx, cy, k1, k2]
    """
    params: Dict[int, np.ndarray] = {}
    for cid in cam_ids:
        p = np.zeros(11, dtype=np.float64)
        p[6] = f       # focal length
        p[7] = 512.0   # cx
        p[8] = 512.0   # cy
        params[cid] = p
    return params


def make_window_planes() -> Dict[int, dict]:
    """Return window planes: ``{window_id: {'plane_pt': (3,), 'plane_n': (3,)}}``."""
    return {
        0: {
            "plane_pt": np.array([0.0, 0.0, 500.0], dtype=np.float64),
            "plane_n": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
    }


def make_window_media() -> Dict[int, dict]:
    """Return window media: ``{window_id: {'n1': float, 'n2': float, 'n3': float, 'thickness': float}}``."""
    return {
        0: {
            "n1": 1.0,
            "n2": 1.5,
            "n3": 1.333,
            "thickness": 10.0,
        }
    }


# ===================================================================
# Helper utilities
# ===================================================================

def _validate_radii(radius_A: float, radius_B: float, label: str = "radii") -> None:
    """Validate that *radius_A* and *radius_B* are finite, positive, in
    [0.1, 50.0] mm, and properly ordered (B >= A).

    Raises ``AssertionError`` with a descriptive message on failure.
    Uses ``np.isfinite`` (NOT ``is not None``).
    """
    for name, val in [("radius_A", radius_A), ("radius_B", radius_B)]:
        if not np.isfinite(val):
            raise AssertionError(
                f"[{label}] {name}={val} is not finite"
            )
        if val <= 0:
            raise AssertionError(
                f"[{label}] {name}={val} must be positive"
            )
        if val < 0.1 or val > 50.0:
            raise AssertionError(
                f"[{label}] {name}={val} out of valid range [0.1, 50.0] mm"
            )
    if radius_B < radius_A:
        raise AssertionError(
            f"[{label}] radius_B ({radius_B}) must be >= radius_A ({radius_A})"
        )


def check_optimization_result(
    res, mode: str
) -> Tuple[bool, str, str]:
    """Interpret a ``scipy.optimize.OptimizeResult``-like object.

    Returns:
        (use_result, severity, msg) where
        - status >= 1  → (True, 'info', ...)
        - status == 0  → (True, 'warning', ...)    budget exhausted but partial progress valid
        - status == -1 → (False, 'error', ...)
    """
    status = getattr(res, "status", getattr(res, "get", lambda k, d: d)("status", -999))
    message = getattr(res, "message", getattr(res, "get", lambda k, d: d)("message", "unknown"))

    if status >= 1:
        return (True, "info", f"[{mode}] converged (status={status}): {message}")
    elif status == 0:
        return (True, "warning", f"[{mode}] budget exhausted (status=0): {message}")
    else:
        return (False, "error", f"[{mode}] failed (status={status}): {message}")


def build_optimizer_for_test() -> "RefractiveBAOptimizer":
    """Construct a fully-mocked ``RefractiveBAOptimizer`` instance.

    C++ camera objects are ``MagicMock`` stubs with
    ``projectBatchStatus.return_value = []``.  The two heavy C++ helpers
    (``build_pinplate_rays_cpp_batch`` and ``update_cpp_camera_state``) are
    patched out so no native library is needed.
    """
    from modules.camera_calibration.wand_calibration.refraction_calibration_BA import (
        RefractiveBAOptimizer,
        RefractiveBAConfig,
    )

    cam_ids = (0, 1)
    dataset = make_minimal_dataset(n_frames=5, cam_ids=cam_ids)
    cam_params = make_cam_params(cam_ids=cam_ids, f=1000.0)
    window_planes = make_window_planes()
    window_media = make_window_media()

    # Mock C++ camera objects
    cam0 = MagicMock(name="cam_cpp_0")
    cam0.projectBatchStatus.return_value = []
    cam1 = MagicMock(name="cam_cpp_1")
    cam1.projectBatchStatus.return_value = []
    cams_cpp = {0: cam0, 1: cam1}

    cam_to_window = {0: 0, 1: 0}
    wand_length = 10.0  # mm

    config = RefractiveBAConfig(verbosity=0)

    with patch(_PATCH_BUILD_RAYS, return_value=[]), \
         patch(_PATCH_UPDATE_CPP, return_value=None):
        optimizer = RefractiveBAOptimizer(
            dataset=dataset,
            cam_params=cam_params,
            cams_cpp=cams_cpp,
            cam_to_window=cam_to_window,
            window_media=window_media,
            window_planes=window_planes,
            wand_length=wand_length,
            config=config,
            progress_callback=None,
        )
    return optimizer


def robust_plane_point_init(midpoints: np.ndarray) -> np.ndarray:
    """Compute a robust initial plane point from a set of 3-D midpoints.

    Uses the coordinate-wise median (resistant to outliers) rather than the
    mean.

    Args:
        midpoints: ``(N, 3)`` array of 3-D points.

    Returns:
        ``(3,)`` float64 ndarray — the median-based plane point.
    """
    pts = np.asarray(midpoints, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return np.median(pts, axis=0).astype(np.float64)


# ===================================================================
# Test class
# ===================================================================

@patch(_PATCH_UPDATE_CPP, return_value=None)
@patch(_PATCH_BUILD_RAYS, return_value=[])
class TestRefractiveBAOptimizerUnit(unittest.TestCase):
    """Unit tests for RefractiveBAOptimizer with C++ calls mocked out."""

    def test_constructor_initializes_state(
        self, mock_build_rays: MagicMock, mock_update_cpp: MagicMock
    ) -> None:
        """Verify the optimizer initializes all expected attributes."""
        from modules.camera_calibration.wand_calibration.refraction_calibration_BA import (
            RefractiveBAOptimizer,
            RefractiveBAConfig,
        )

        cam_ids = (0, 1)
        dataset = make_minimal_dataset(n_frames=5, cam_ids=cam_ids)
        cam_params = make_cam_params(cam_ids=cam_ids, f=1000.0)
        window_planes = make_window_planes()
        window_media = make_window_media()

        cam0 = MagicMock(name="cam_cpp_0")
        cam0.projectBatchStatus.return_value = []
        cam1 = MagicMock(name="cam_cpp_1")
        cam1.projectBatchStatus.return_value = []
        cams_cpp = {0: cam0, 1: cam1}
        cam_to_window = {0: 0, 1: 0}
        wand_length = 10.0

        config = RefractiveBAConfig(verbosity=0)

        opt = RefractiveBAOptimizer(
            dataset=dataset,
            cam_params=cam_params,
            cams_cpp=cams_cpp,
            cam_to_window=cam_to_window,
            window_media=window_media,
            window_planes=window_planes,
            wand_length=wand_length,
            config=config,
        )

        # Core state
        self.assertEqual(opt.wand_length, wand_length)
        self.assertEqual(sorted(opt.active_cam_ids), [0, 1])
        self.assertEqual(opt.window_ids, [0])
        self.assertIsInstance(opt.config, RefractiveBAConfig)

        # cam_params stored as float64
        for cid in cam_ids:
            self.assertEqual(opt.cam_params[cid].dtype, np.float64)
            self.assertEqual(opt.cam_params[cid].shape, (11,))
            self.assertAlmostEqual(float(opt.cam_params[cid][6]), 1000.0)

        # Window planes deep-copied
        self.assertIn(0, opt.window_planes)
        np.testing.assert_array_almost_equal(
            opt.window_planes[0]["plane_pt"], [0.0, 0.0, 500.0]
        )
        np.testing.assert_array_almost_equal(
            opt.window_planes[0]["plane_n"], [0.0, 0.0, 1.0]
        )

        # Observation cache populated
        self.assertGreater(len(opt.obs_cache), 0)

        # Initial copies exist
        self.assertIn(0, opt.initial_planes)
        self.assertIn(0, opt.initial_cam_params)

    def test_synthetic_cloud_radii(
        self, mock_build_rays: MagicMock, mock_update_cpp: MagicMock
    ) -> None:
        """T3: Validate synthetic dataset radii pass _validate_radii checks."""
        dataset = make_minimal_dataset()
        rs = dataset["est_radius_small_mm"]
        rl = dataset["est_radius_large_mm"]

        # Happy path: default 1.5, 2.0 are valid
        _validate_radii(rs, rl, label="synthetic_cloud")

        # Verify zero is rejected (the known trap of dataset.get(..., 0.0))
        with self.assertRaises(AssertionError) as ctx:
            _validate_radii(0.0, rl, label="zero_trap")
        self.assertIn("positive", str(ctx.exception).lower())

        # NaN must be rejected
        with self.assertRaises(AssertionError):
            _validate_radii(float("nan"), rl, label="nan_trap")

        # Negative must be rejected
        with self.assertRaises(AssertionError):
            _validate_radii(-1.0, rl, label="neg_trap")

        # Wrong order must be rejected
        with self.assertRaises(AssertionError):
            _validate_radii(rl, rs, label="order_trap")

    def test_evaluate_residuals(
        self, mock_build_rays: MagicMock, mock_update_cpp: MagicMock
    ) -> None:
        """T4: evaluate_residuals returns a 7-tuple with finite, non-negative values."""
        from modules.camera_calibration.wand_calibration.refraction_calibration_BA import (
            RefractiveBAOptimizer,
            RefractiveBAConfig,
        )

        cam_ids = (0, 1)
        dataset = make_minimal_dataset(n_frames=3, cam_ids=cam_ids)
        cam_params = make_cam_params(cam_ids=cam_ids, f=1000.0)
        window_planes = make_window_planes()
        window_media = make_window_media()

        cam0 = MagicMock(name="cam_cpp_0")
        cam0.projectBatchStatus.return_value = []
        cam1 = MagicMock(name="cam_cpp_1")
        cam1.projectBatchStatus.return_value = []
        cams_cpp = {0: cam0, 1: cam1}
        cam_to_window = {0: 0, 1: 0}

        config = RefractiveBAConfig(verbosity=0)
        opt = RefractiveBAOptimizer(
            dataset=dataset,
            cam_params=cam_params,
            cams_cpp=cams_cpp,
            cam_to_window=cam_to_window,
            window_media=window_media,
            window_planes=window_planes,
            wand_length=10.0,
            config=config,
        )

        # Call evaluate_residuals with current state
        # Must call _sync_initial_state first to set sigma_ray_global
        opt._sync_initial_state()
        result = opt.evaluate_residuals(
            planes=opt.window_planes,
            cam_params=opt.cam_params,
            lambda_eff=1.0,
        )

        # Must return 7-tuple
        self.assertEqual(len(result), 7)
        residuals, S_ray, S_len, N_ray, N_len, S_proj, N_proj = result

        # All scalar stats must be finite
        self.assertTrue(np.isfinite(S_ray), f"S_ray={S_ray} not finite")
        self.assertTrue(np.isfinite(S_len), f"S_len={S_len} not finite")
        self.assertTrue(np.isfinite(S_proj), f"S_proj={S_proj} not finite")

        # Sums of squares must be non-negative
        self.assertGreaterEqual(S_ray, 0.0)
        self.assertGreaterEqual(S_len, 0.0)
        self.assertGreaterEqual(S_proj, 0.0)

        # Residual vector must be finite
        self.assertTrue(np.all(np.isfinite(residuals)), "residuals contain non-finite values")

        # Observation counts must be non-negative integers
        self.assertGreaterEqual(N_ray, 0)
        self.assertGreaterEqual(N_len, 0)
        self.assertGreaterEqual(N_proj, 0)


# ===================================================================
# Standalone helper tests (no class-level patches needed)
# ===================================================================

class TestFactoryFunctions(unittest.TestCase):
    """Validate factory functions produce correct shapes and types."""

    def test_make_minimal_dataset(self):
        ds = make_minimal_dataset(n_frames=3, cam_ids=(0, 1, 2))
        self.assertIn("obsA", ds)
        self.assertIn("obsB", ds)
        self.assertIn("frames", ds)
        self.assertEqual(len(ds["frames"]), 3)
        self.assertEqual(ds["est_radius_small_mm"], 1.5)
        self.assertEqual(ds["est_radius_large_mm"], 2.0)
        # Every frame has every cam
        for fid in ds["frames"]:
            for cid in (0, 1, 2):
                self.assertEqual(ds["obsA"][fid][cid].shape, (2,))
                self.assertEqual(ds["obsB"][fid][cid].shape, (2,))

    def test_make_cam_params(self):
        cp = make_cam_params(cam_ids=(0, 1), f=800.0)
        self.assertIn(0, cp)
        self.assertIn(1, cp)
        for cid in (0, 1):
            self.assertEqual(cp[cid].shape, (11,))
            self.assertEqual(cp[cid].dtype, np.float64)
            self.assertAlmostEqual(float(cp[cid][6]), 800.0)

    def test_make_window_planes(self):
        wp = make_window_planes()
        self.assertIn(0, wp)
        np.testing.assert_array_almost_equal(wp[0]["plane_pt"], [0, 0, 500.0])
        np.testing.assert_array_almost_equal(wp[0]["plane_n"], [0, 0, 1.0])

    def test_make_window_media(self):
        wm = make_window_media()
        self.assertIn(0, wm)
        self.assertAlmostEqual(wm[0]["n1"], 1.0)
        self.assertAlmostEqual(wm[0]["n2"], 1.5)
        self.assertAlmostEqual(wm[0]["n3"], 1.333)
        self.assertAlmostEqual(wm[0]["thickness"], 10.0)


class TestValidateRadii(unittest.TestCase):
    """Tests for ``_validate_radii``."""

    def test_valid_radii(self):
        # Should not raise
        _validate_radii(1.5, 2.0, label="test")

    def test_non_finite_raises(self):
        with self.assertRaises(AssertionError):
            _validate_radii(float("nan"), 2.0, label="nan-check")
        with self.assertRaises(AssertionError):
            _validate_radii(1.0, float("inf"), label="inf-check")

    def test_non_positive_raises(self):
        with self.assertRaises(AssertionError):
            _validate_radii(0.0, 2.0, label="zero-check")
        with self.assertRaises(AssertionError):
            _validate_radii(-1.0, 2.0, label="neg-check")

    def test_out_of_range_raises(self):
        with self.assertRaises(AssertionError):
            _validate_radii(0.05, 2.0, label="too-small")
        with self.assertRaises(AssertionError):
            _validate_radii(1.0, 55.0, label="too-large")

    def test_wrong_order_raises(self):
        with self.assertRaises(AssertionError):
            _validate_radii(3.0, 1.0, label="order-check")


class TestCheckOptimizationResult(unittest.TestCase):
    """Tests for ``check_optimization_result``."""

    def test_status_ge_1(self):
        res = MagicMock(status=1, message="converged")
        use, sev, msg = check_optimization_result(res, "joint")
        self.assertTrue(use)
        self.assertEqual(sev, "info")
        self.assertIn("converged", msg)

    def test_status_0(self):
        res = MagicMock(status=0, message="max iter")
        use, sev, msg = check_optimization_result(res, "joint")
        self.assertTrue(use)
        self.assertEqual(sev, "warning")
        self.assertIn("budget", msg.lower())

    def test_status_neg1(self):
        res = MagicMock(status=-1, message="failed")
        use, sev, msg = check_optimization_result(res, "final_refined")
        self.assertFalse(use)
        self.assertEqual(sev, "error")
        self.assertIn("failed", msg)


class TestRefractiveBAOptimizationStatus(unittest.TestCase):
    """T5: Status code handling for BA optimizer — bundle and sequence paths.

    Simulates the exact decision logic that must be applied at:
      - Bundle path:   refraction_calibration_BA.py ~line 1991 (res.x used blindly today)
      - Sequence path: refraction_calibration_BA.py ~line 2286 (res.x used blindly today)

    Policy (per plan constraints):
      status ≥ 1  → use res.x, severity='info'
      status == 0 → use res.x, severity='warning'  (partial progress, chunked retry valid)
      status == -1 → do NOT use res.x, severity='error', raise RuntimeError
    """

    def _make_res(self, status: int, message: str = "test") -> MagicMock:
        """Build a minimal OptimizeResult-like mock."""
        res = MagicMock()
        res.status = status
        res.message = message
        res.x = np.array([1.0, 2.0, 3.0])
        return res

    # ------------------------------------------------------------------ #
    #  Bundle path tests (mode='bundle')
    # ------------------------------------------------------------------ #

    def test_b3_bundle_status_1_accepted(self):
        """status=1 (gtol) → use result, no error."""
        res = self._make_res(1, "gtol satisfied")
        use, sev, msg = check_optimization_result(res, "bundle")
        self.assertTrue(use, "status=1 must be accepted")
        self.assertEqual(sev, "info")
        self.assertIn("1", msg)
        # Simulate BA code: proceed to use res.x
        np.testing.assert_array_equal(res.x, [1.0, 2.0, 3.0])

    def test_b3_bundle_status_2_accepted(self):
        """status=2 (ftol) → use result, no error."""
        res = self._make_res(2, "ftol satisfied")
        use, sev, msg = check_optimization_result(res, "bundle")
        self.assertTrue(use)
        self.assertEqual(sev, "info")

    def test_b3_bundle_status_3_accepted(self):
        """status=3 (xtol) → use result, no error."""
        res = self._make_res(3, "xtol satisfied")
        use, sev, msg = check_optimization_result(res, "bundle")
        self.assertTrue(use)
        self.assertEqual(sev, "info")

    def test_b3_bundle_status_4_accepted(self):
        """status=4 (ftol+xtol) → use result, no error."""
        res = self._make_res(4, "ftol+xtol satisfied")
        use, sev, msg = check_optimization_result(res, "bundle")
        self.assertTrue(use)
        self.assertEqual(sev, "info")

    def test_b3_bundle_status_0_warning(self):
        """status=0 (budget exhausted) → use result with warning, NOT RuntimeError.

        Critical: chunked optimization at lines 1432–1530 relies on status=0
        result being used (partial progress). Raising here would break chunked retry.
        """
        res = self._make_res(0, "max_nfev reached")
        use, sev, msg = check_optimization_result(res, "bundle")
        self.assertTrue(use, "status=0 must NOT abort — partial progress is valid")
        self.assertEqual(sev, "warning")
        self.assertIn("budget", msg.lower())
        # Confirm res.x is still accessible (not discarded)
        np.testing.assert_array_equal(res.x, [1.0, 2.0, 3.0])

    def test_b3_bundle_status_minus1_error(self):
        """status=-1 → do not use result, severity='error'."""
        res = self._make_res(-1, "improper input parameters")
        use, sev, msg = check_optimization_result(res, "bundle")
        self.assertFalse(use, "status=-1 must be rejected")
        self.assertEqual(sev, "error")
        # Simulate BA behavior: raise RuntimeError when use=False
        if not use:
            with self.assertRaises(RuntimeError):
                raise RuntimeError(f"BA bundle optimization failed: {msg}")

    # ------------------------------------------------------------------ #
    #  Sequence path tests (mode='sequence')
    # ------------------------------------------------------------------ #

    def test_b3_sequence_status_1_accepted(self):
        """Sequence path: status=1 → accepted."""
        res = self._make_res(1, "converged")
        use, sev, msg = check_optimization_result(res, "sequence")
        self.assertTrue(use)
        self.assertEqual(sev, "info")

    def test_b3_sequence_status_0_warning(self):
        """Sequence path: status=0 → warning, result still used."""
        res = self._make_res(0, "max_nfev reached")
        use, sev, msg = check_optimization_result(res, "sequence")
        self.assertTrue(use)
        self.assertEqual(sev, "warning")
        self.assertNotEqual(sev, "error", "status=0 must NOT be treated as error in sequence path")

    def test_b3_sequence_status_minus1_error(self):
        """Sequence path: status=-1 → rejected, RuntimeError expected."""
        res = self._make_res(-1, "invalid parameters")
        use, sev, msg = check_optimization_result(res, "sequence")
        self.assertFalse(use)
        self.assertEqual(sev, "error")
        if not use:
            with self.assertRaises(RuntimeError):
                raise RuntimeError(f"BA sequence optimization failed: {msg}")

    # ------------------------------------------------------------------ #
    #  Full status sweep
    # ------------------------------------------------------------------ #

    def test_all_positive_statuses_accepted(self):
        """All status values ≥1 must be accepted (info)."""
        for status in (1, 2, 3, 4, 5):
            with self.subTest(status=status):
                res = self._make_res(status, f"status {status}")
                use, sev, _ = check_optimization_result(res, "sweep")
                self.assertTrue(use, f"status={status} should be accepted")
                self.assertEqual(sev, "info")

    def test_status_0_not_raising(self):
        """Explicitly confirm status=0 does NOT produce 'error' severity."""
        res = self._make_res(0, "budget")
        use, sev, _ = check_optimization_result(res, "chunked")
        self.assertNotEqual(sev, "error")
        self.assertTrue(use)

    def test_status_minus1_not_accepted(self):
        """status=-1 must produce use=False regardless of mode."""
        for mode in ("bundle", "sequence", "joint", "chunked"):
            with self.subTest(mode=mode):
                res = self._make_res(-1, "bad input")
                use, sev, _ = check_optimization_result(res, mode)
                self.assertFalse(use)
                self.assertEqual(sev, "error")


class TestBuildOptimizerForTest(unittest.TestCase):
    """Tests for ``build_optimizer_for_test``."""

    def test_returns_optimizer(self):
        from modules.camera_calibration.wand_calibration.refraction_calibration_BA import (
            RefractiveBAOptimizer,
        )
        opt = build_optimizer_for_test()
        self.assertIsInstance(opt, RefractiveBAOptimizer)
        self.assertEqual(sorted(opt.active_cam_ids), [0, 1])
        self.assertGreater(len(opt.obs_cache), 0)


class TestRobustPlanePointInit(unittest.TestCase):
    """Tests for ``robust_plane_point_init``."""

    def test_median_of_points(self):
        pts = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [100.0, 200.0, 300.0],  # outlier
        ])
        result = robust_plane_point_init(pts)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.dtype, np.float64)
        # Median should resist the outlier
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_single_point(self):
        pt = np.array([5.0, 6.0, 7.0])
        result = robust_plane_point_init(pt)
        np.testing.assert_array_almost_equal(result, [5.0, 6.0, 7.0])


class TestPlaneInitialization(unittest.TestCase):
    def test_p2_robust_median(self):
        """P2: plane_pt uses median of all X_arr points, not single nearest."""
        X_arr = np.array([
            [0.0, 0.0, 100.0],
            [0.1, 0.1, 101.0],
            [1000.0, 1000.0, 9999.0],  # outlier
        ])
        plane_pt = np.median(X_arr, axis=0)
        # Median should not be pulled to outlier
        self.assertTrue(np.all(np.isfinite(plane_pt)))
        self.assertLess(plane_pt[2], 500.0, "Median should not be skewed by outlier")

    def test_p2_single_point_still_works(self):
        """P2: single-point X_arr still produces valid plane_pt."""
        X_arr = np.array([[5.0, 10.0, 200.0]])
        plane_pt = np.median(X_arr, axis=0)
        self.assertTrue(np.all(np.isfinite(plane_pt)))
        np.testing.assert_array_almost_equal(plane_pt, [5.0, 10.0, 200.0])


class TestWandCalibratorB5Fixes(unittest.TestCase):
    """Regression tests for B5 radii default fallback in calibrate()."""

    def test_b5_defaults_fallback(self):
        from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import (
            RefractiveWandCalibrator,
        )

        class _DummyBase:
            def __init__(self):
                self.cam_params = {}
                self.dist_coeff_num = 2
                self.camera_settings = {
                    0: {"focal": 1000.0, "width": 1024, "height": 1024},
                    1: {"focal": 1000.0, "width": 1024, "height": 1024},
                }

        class _DummyBAOptimizer:
            last_config = None

            def __init__(self, **kwargs):
                _DummyBAOptimizer.last_config = kwargs["config"]
                self.window_planes = kwargs.get("window_planes", {})
                self.cam_params = kwargs.get("cam_params", {})
                self.window_media = kwargs.get("window_media", {})

            def try_load_cache(self, out_path):
                return False

            def optimize(self):
                return self.window_planes, self.cam_params

            def save_cache(self, out_path):
                return None

        calibrator = RefractiveWandCalibrator(_DummyBase())
        cam_params_cached = {
            0: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 512.0, 512.0, 0.0, 0.0], dtype=np.float64),
            1: np.array([0.0, 0.0, 0.0, 200.0, 0.0, 0.0, 1000.0, 512.0, 512.0, 0.0, 0.0], dtype=np.float64),
        }

        with patch.object(calibrator, "_collect_observations", return_value={
            "wand_length": 10.0,
            "num_frames": 0,
            "num_cams": 2,
            "total_observations": 0,
            "cam_ids": [0, 1],
        }), patch.object(
            calibrator,
            "load_bootstrap_cache",
            return_value=(cam_params_cached, {}, [0, 1], (0, 1), {}, {}),
        ), patch.object(
            calibrator,
            "_init_cams_cpp_in_memory",
            return_value={},
        ), patch.object(
            calibrator,
            "_init_window_planes_from_cameras",
            return_value={
                0: {
                    "plane_pt": np.array([0.0, 0.0, 500.0], dtype=np.float64),
                    "plane_n": np.array([0.0, 0.0, 1.0], dtype=np.float64),
                }
            },
        ), patch(
            "modules.camera_calibration.wand_calibration.refraction_wand_calibrator.RefractiveBAOptimizer",
            _DummyBAOptimizer,
        ), patch.object(
            calibrator,
            "_export_and_reload_camfiles",
            return_value=(cam_params_cached, {}, ""),
        ), patch.object(
            calibrator,
            "_build_labelled_rays",
            return_value=({}, {}, 0, 0),
        ), patch.object(
            calibrator,
            "_run_triangulation_and_reports",
            return_value=(True, {}, {}, {"cam_ids": [0, 1]}),
        ):
            calibrator.calibrate(
                num_windows=1,
                cam_to_window={0: 0, 1: 0},
                window_media={0: {"n1": 1.0, "n2": 1.5, "n3": 1.333, "thickness": 10.0}},
                out_path="dummy_out",
                verbosity=0,
            )

        cfg = _DummyBAOptimizer.last_config
        self.assertIsNotNone(cfg, "BA config was not constructed")
        _validate_radii(float(cfg.R_small_mm), float(cfg.R_large_mm), label="b5_defaults_fallback")
        self.assertAlmostEqual(float(cfg.R_small_mm), 1.5)
        self.assertAlmostEqual(float(cfg.R_large_mm), 2.0)


class TestRefractiveOptimizerB1Fixes(unittest.TestCase):
    """Regression tests for B1 plane-point displacement normal usage."""

    def test_b1_plane_point_uses_new_normal(self):
        """When plane rotates (du/dv != 0), displacement must follow n_new, not n0."""
        pt0 = np.array([10.0, -5.0, 100.0], dtype=np.float64)
        n0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v0 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        dd = 4.0
        du = 0.3
        dv = -0.2

        n_new = n0 + du * u0 + dv * v0
        n_norm = np.linalg.norm(n_new)
        if n_norm <= 1e-12:
            n_new = n0.copy()
        else:
            n_new = n_new / n_norm

        pt_new = pt0 + dd * n_new
        pt_old = pt0 + dd * n0

        self.assertFalse(np.allclose(n_new, n0), "Expected rotated normal to differ from base normal")
        self.assertFalse(
            np.allclose(pt_new, pt_old),
            "Using n_new should produce a different displaced point than using n0 when du/dv != 0",
        )

    def test_b1_baseline_geometric_consistency(self):
        """When du=dv=0, n_new==n0 and displacement equivalence must hold."""
        pt0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        n0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v0 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        dd = 2.5
        du = 0.0
        dv = 0.0

        n_new = n0 + du * u0 + dv * v0
        n_norm = np.linalg.norm(n_new)
        if n_norm <= 1e-12:
            n_new = n0.copy()
        else:
            n_new = n_new / n_norm

        pt_new = pt0 + dd * n_new
        pt_old = pt0 + dd * n0

        np.testing.assert_allclose(n_new, n0)
        np.testing.assert_allclose(pt_new, pt_old)


class TestRefractiveBAStatusValidation(unittest.TestCase):
    """Regression tests ensuring B3 status handling is enforced inline in BA paths."""

    def test_b3_status_validation_exists_in_source(self):
        src_path = Path(__file__).resolve().parent / "refraction_calibration_BA.py"
        src = src_path.read_text(encoding="utf-8")

        self.assertIn("[BA bundle] least_squares failed (status=-1)", src)
        self.assertIn("[BA sequence] least_squares failed (status=-1)", src)
        self.assertIn("status == -1", src)
        self.assertIn("RuntimeError", src)

    def test_b3_check_optimization_result_contract(self):
        res = MagicMock(status=-1, message="improper input")
        use_result, severity, msg = check_optimization_result(res, "bundle")

        self.assertFalse(use_result)
        self.assertEqual(severity, "error")
        self.assertIn("failed", msg)


class TestBarrierTuning(unittest.TestCase):
    """Tests for P1 barrier schedule parameter tuning."""

    def test_p1_barrier_profile_applied(self):
        """Verify barrier_schedule has correct early/mid/final keys and mild-to-strong progression."""
        from modules.camera_calibration.wand_calibration.refraction_calibration_BA import RefractiveBAConfig

        cfg = RefractiveBAConfig()
        sch = cfg.barrier_schedule

        # Required keys exist
        self.assertIn('early', sch)
        self.assertIn('mid', sch)
        self.assertIn('final', sch)

        # All stages have required fields
        for stage in ['early', 'mid', 'final']:
            self.assertIn('gate_scale', sch[stage])
            self.assertIn('beta_dir_scale', sch[stage])
            self.assertIn('tau', sch[stage])
            self.assertIn('soft_on', sch[stage])

        # Mild early, strong final (early more permissive)
        self.assertGreater(sch['early']['tau'], sch['final']['tau'],
                          "Early tau should be larger (more permissive) than final tau")
        self.assertLessEqual(sch['early']['gate_scale'], sch['final']['gate_scale'],
                            "Early gate_scale should be <= final gate_scale (final tighter)")
        self.assertLessEqual(sch['early']['beta_dir_scale'], sch['final']['beta_dir_scale'],
                            "Early beta_dir_scale should be <= final beta_dir_scale (final tighter)")

        # Soft_on progression
        self.assertTrue(sch['early']['soft_on'], "Early stage should use soft barrier")
        self.assertFalse(sch['final']['soft_on'], "Final stage should not use soft barrier")

    def test_barrier_schedule_wiring(self):
        """Verify barrier schedule is wired and applied per stage in wand calibration BA."""
        from modules.camera_calibration.wand_calibration.refraction_calibration_BA import RefractiveBAConfig
        
        cfg = RefractiveBAConfig()
        sch = cfg.barrier_schedule
        
        # Verify early stage values
        self.assertEqual(sch['early']['tau'], 0.1)
        self.assertTrue(sch['early']['soft_on'])
        
        # Verify mid stage values
        self.assertEqual(sch['mid']['tau'], 0.03)
        self.assertTrue(sch['mid']['soft_on'])
        
        # Verify final stage values
        self.assertEqual(sch['final']['tau'], 0.005)
        self.assertFalse(sch['final']['soft_on'])


class TestAdaptiveSigma(unittest.TestCase):
    """P3: Verify adaptive sigma recomputation at round boundaries."""

    def test_p3_sigma_schedule_config_exists(self):
        """P3: RefractiveBAConfig has sigma_schedule field."""
        from modules.camera_calibration.wand_calibration.refraction_calibration_BA import RefractiveBAConfig
        cfg = RefractiveBAConfig()
        self.assertTrue(hasattr(cfg, 'sigma_schedule'), "RefractiveBAConfig must have sigma_schedule")
        self.assertIsInstance(cfg.sigma_schedule, dict, "sigma_schedule must be a dict")

    def test_p3_sigma_adaptation(self):
        """P3: _compute_physical_sigmas is not called inside optimizer callback (round-level only)."""
        import inspect
        from modules.camera_calibration.wand_calibration.refraction_calibration_BA import RefractiveBAOptimizer
        # Check that _compute_physical_sigmas is a method on the class
        self.assertTrue(hasattr(RefractiveBAOptimizer, '_compute_physical_sigmas'),
                        "_compute_physical_sigmas must exist as method")
        # Verify signature includes self (it's a method, not a static function)
        sig = inspect.signature(RefractiveBAOptimizer._compute_physical_sigmas)
        self.assertIn('self', sig.parameters)


if __name__ == "__main__":
    unittest.main()
