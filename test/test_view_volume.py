"""Dedicated common-FOV regression tests.

Run: conda run -n OpenLPT python -m unittest discover -s test -p test_view_volume.py -v

The refractive fixture checks the real binding with injected solver failures.
It is not the original user dataset that exhibited the collapsed volume.
"""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from gui.utils.view_volume import (
    CameraVisibility, OUTSIDE, UNKNOWN, VISIBLE, VolumeEstimate,
    estimate_volume, scan_grid,
)


def box_visibility(lower, upper):
    return lambda pts: np.where(((pts >= lower) & (pts <= upper)).all(axis=1), VISIBLE, OUTSIDE)


def pinhole_cameras(scale=1.0):
    """Six inward cameras whose common FOV has analytic bounds +/-2.5*scale."""
    cams = []
    for axis in np.eye(3):
        for sign in (-1, 1):
            center = axis * sign * 10 * scale
            forward = -axis * sign
            up = np.array([0., 1., 0.]) if axis[1] == 0 else np.array([0., 0., 1.])
            right = np.cross(up, forward)
            R = np.array([right, np.cross(forward, right), forward])
            rvec, _ = cv2.Rodrigues(R)
            cams.append(dict(model="PINHOLE", h=1000, w=1000,
                             K=np.array([[2000., 0, 500], [0, 2000., 500], [0, 0, 1]]),
                             rvec=rvec.ravel(), tvec=-R @ center, t_inv=center))
    return cams


class EstimatorTests(unittest.TestCase):
    def test_pinhole_geometry_at_5_50_500_mm_scales(self):
        for scale in (1., 10., 100.):
            with self.subTest(width_mm=5 * scale):
                result = estimate_volume(CameraVisibility(pinhole_cameras(scale)),
                                         np.full(3, -6 * scale), np.full(3, 6 * scale))
                self.assertTrue(result.confident, result.reason)
                np.testing.assert_allclose(result.minimum / scale, -2.5, atol=0.15)
                np.testing.assert_allclose(result.maximum / scale, 2.5, atol=0.15)

    def test_legitimate_thin_layer(self):
        for scale in (.1, 1., 10.):
            with self.subTest(width_mm=50 * scale, thickness_mm=.2 * scale):
                lower = np.array([-25., -25., -.1]) * scale
                upper = -lower
                result = estimate_volume(box_visibility(lower, upper),
                                         np.full(3, -50. * scale), np.full(3, 50. * scale))
                self.assertTrue(result.confident, result.reason)
                thickness = (result.maximum[2] - result.minimum[2]) / scale
                self.assertAlmostEqual(thickness, .2, delta=.02)
                self.assertTrue(np.all(abs(result.minimum - lower) <= 2 * result.step))
                self.assertTrue(np.all(abs(result.maximum - upper) <= 2 * result.step))

    def test_partial_failures_at_boundary_are_not_accepted(self):
        def classify(pts):
            states = box_visibility(-np.ones(3), np.ones(3))(pts)
            states[(pts[:, 0] > .8) & (pts[:, 0] < 1.3)] = UNKNOWN
            return states
        result = estimate_volume(classify, np.full(3, -2.), np.full(3, 2.))
        self.assertFalse(result.confident)
        self.assertIn("Projection failures", result.reason)

    def test_sparse_interior_failures_do_not_enlarge_volume(self):
        def classify(pts):
            states = box_visibility(-np.ones(3), np.ones(3))(pts)
            states[(abs(pts) < .1).all(axis=1)] = UNKNOWN
            return states
        result = estimate_volume(classify, np.full(3, -2.), np.full(3, 2.))
        self.assertTrue(result.confident, result.reason)
        np.testing.assert_allclose(result.maximum, 1., atol=.07)

    def test_high_interior_failure_rate_is_not_accepted(self):
        def classify(pts):
            states = box_visibility(-np.ones(3), np.ones(3))(pts)
            states[(abs(pts) < .8).all(axis=1)] = UNKNOWN
            return states
        result = estimate_volume(classify, np.full(3, -2.), np.full(3, 2.))
        self.assertFalse(result.confident)
        self.assertIn("Projection failures", result.reason)

    def test_expand_only_touched_face(self):
        boxes = []
        def record(*args, **kwargs):
            boxes.append((args[1].copy(), args[2].copy()))
            return scan_grid(*args, **kwargs)
        with patch("gui.utils.view_volume.scan_grid", side_effect=record):
            estimate_volume(box_visibility(np.array([-.5, -.5, -.5]), np.array([3., .5, .5])),
                            -np.ones(3), np.ones(3), max_iterations=2)
        np.testing.assert_array_equal(boxes[1][0], boxes[0][0])
        np.testing.assert_array_equal(boxes[1][1][1:], boxes[0][1][1:])
        self.assertGreater(boxes[1][1][0], boxes[0][1][0])

    def test_empty_scan_refines_without_expanding(self):
        boxes = []
        def record(*args, **kwargs):
            boxes.append((args[1].copy(), args[2].copy()))
            return scan_grid(*args, **kwargs)
        with patch("gui.utils.view_volume.scan_grid", side_effect=record):
            result = estimate_volume(lambda pts: np.full(len(pts), OUTSIDE),
                                     -np.ones(3), np.ones(3), max_iterations=2)
        self.assertFalse(result.confident)
        np.testing.assert_array_equal(boxes[0], boxes[1])

    def test_unbounded_overlap_and_iteration_limit(self):
        result = estimate_volume(lambda pts: np.full(len(pts), VISIBLE),
                                 -np.ones(3), np.ones(3), max_iterations=3)
        self.assertFalse(result.confident)
        self.assertIn("unbounded", result.reason)

    def test_grid_honors_spacing_beyond_old_60_interval_cap(self):
        scan = scan_grid(lambda pts: np.full(len(pts), VISIBLE), np.zeros(3),
                         np.array([70., 1., 1.]), np.full(3, .5), 100_000)
        np.testing.assert_array_equal(scan.step, [.5, .5, .5])
        self.assertEqual(scan.layers[0], 141)

    def test_budget_exhaustion_is_not_success(self):
        result = estimate_volume(box_visibility(-np.ones(3), np.ones(3)),
                                 np.full(3, -2.), np.full(3, 2.), max_samples=16_000)
        self.assertFalse(result.confident)
        self.assertIn("budget", result.reason)


class CameraStatusTests(unittest.TestCase):
    def fake_lpt(self, statuses=None, error=None):
        def project(points, detail):
            if error:
                raise error
            return statuses[:len(points)]
        return SimpleNamespace(Camera=lambda path: SimpleNamespace(projectBatchStatus=project),
                               Pt3D=lambda *coords: coords)

    def test_pinplate_partial_projection_failures(self):
        cam = dict(model="PINPLATE", h=100, w=100, cam_file_path="synthetic")
        lpt = self.fake_lpt([(True, (50, 50), ""), (False, (0, 0), "solver failed"),
                             (True, (150, 50), ""), (True, (np.nan, 50), "")])
        with patch.dict("sys.modules", pyopenlpt=lpt):
            states = CameraVisibility([cam, cam])(np.zeros((4, 3)))
        np.testing.assert_array_equal(states, [VISIBLE, UNKNOWN, OUTSIDE, UNKNOWN])

    def test_pinplate_batch_exception_never_uses_pinhole(self):
        cam = pinhole_cameras()[0] | dict(model="PINPLATE", cam_file_path="synthetic")
        with patch.dict("sys.modules", pyopenlpt=self.fake_lpt(error=RuntimeError("solver failed"))):
            with patch("gui.utils.view_volume.cv2.projectPoints", side_effect=AssertionError("fallback")):
                states = CameraVisibility([cam, cam])(np.zeros((3, 3)))
        np.testing.assert_array_equal(states, [UNKNOWN] * 3)

    def test_one_definite_exclusion_overrides_unknown(self):
        refr = dict(model="PINPLATE", h=100, w=100, cam_file_path="synthetic")
        # The pinhole camera sees the origin but excludes a far off-axis point.
        pin = pinhole_cameras()[0]
        points = np.array([[0., 0., 0.], [0., 1000., 0.]])
        for cams in ([refr, pin], [pin, refr]):
            with patch.dict("sys.modules", pyopenlpt=self.fake_lpt(error=RuntimeError("failed"))):
                states = CameraVisibility(cams)(points)
            np.testing.assert_array_equal(states, [UNKNOWN, OUTSIDE])

    def test_invalid_camera_is_not_silently_skipped(self):
        with self.assertRaises(ValueError):
            CameraVisibility([pinhole_cameras()[0], {}])

    def test_installed_refractive_binding(self):
        try:
            import pyopenlpt as lpt
        except ImportError:
            self.skipTest("pyopenlpt is not installed")
        path = Path(__file__).parent / "inputs/test_Camera/cam1_refract.txt"
        cam = dict(model="PINPLATE", h=1024, w=1024, cam_file_path=str(path))
        classifier = CameraVisibility([cam, cam])
        points = np.array([[0., 0., 0.], [10., 10., 10.]])
        statuses = lpt.Camera(str(path)).projectBatchStatus([lpt.Pt3D(*p) for p in points], False)
        expected = []
        for ok, uv, message in statuses:
            expected.append((VISIBLE if 0 <= uv[0] < 1024 and 0 <= uv[1] < 1024 else OUTSIDE)
                            if ok else UNKNOWN)
        np.testing.assert_array_equal(classifier(points), expected)

    def test_real_refractive_camera_with_injected_partial_failures(self):
        try:
            import pyopenlpt
        except ImportError:
            self.skipTest("pyopenlpt is not installed")
        path = Path(__file__).parent / "inputs/test_Camera/cam1_refract.txt"
        refr = dict(model="PINPLATE", h=1024, w=1024, cam_file_path=str(path))
        classifier = CameraVisibility(pinhole_cameras(.1) + [refr])
        model, h, w, actual_camera = classifier.cameras[-1]

        def failing_projection(points, detail):
            statuses = actual_camera.projectBatchStatus(points, detail)
            return [(False, (0., 0.), "injected solver failure") if p[0] > .15 else status
                    for p, status in zip(points, statuses)]

        classifier.cameras[-1] = (model, h, w, SimpleNamespace(projectBatchStatus=failing_projection))
        np.testing.assert_array_equal(classifier(np.array([[0., 0., 0.], [.2, 0., 0.]])),
                                      [VISIBLE, UNKNOWN])
        result = estimate_volume(classifier, np.full(3, -.6), np.full(3, .6))
        self.assertFalse(result.confident)
        self.assertIn("Projection failures", result.reason)


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_low_confidence_keeps_volume_and_voxel(self):
        from gui.views.tracking_settings_view import TrackingSettingsView
        view = TrackingSettingsView()
        cams = pinhole_cameras()
        view.detected_cam_files = [str(i) for i in range(len(cams))]
        widgets = [view.vol_x_min, view.vol_x_max, view.vol_y_min, view.vol_y_max,
                   view.vol_z_min, view.vol_z_max, view.voxel_spin]
        before = [w.value() for w in widgets]
        with patch("gui.utils.view_volume.estimate_volume", return_value=VolumeEstimate(reason="Not converged")):
            with patch("gui.views.tracking_settings_view.QMessageBox.warning") as warning:
                view._estimate_volume_from_cameras(cams)
        self.assertEqual(before, [w.value() for w in widgets])
        warning.assert_called_once()
        view.close()

    def test_success_keeps_thin_volume_without_5_mm_rounding(self):
        from gui.views.tracking_settings_view import TrackingSettingsView
        view = TrackingSettingsView()
        cams = pinhole_cameras()
        view.detected_cam_files = [str(i) for i in range(len(cams))]
        result = VolumeEstimate(np.array([-2.5, -2.5, -.1]), np.array([2.5, 2.5, .1]),
                                np.full(3, .001), True, "Stable")
        with patch("gui.utils.view_volume.estimate_volume", return_value=result):
            view._estimate_volume_from_cameras(cams)
        self.assertAlmostEqual(view.vol_z_max.value() - view.vol_z_min.value(), .2)
        self.assertAlmostEqual(view.voxel_spin.value(), .005)
        view.close()

    def test_camera_seed_to_gui_at_multiple_scales(self):
        from gui.views.tracking_settings_view import TrackingSettingsView
        for scale in (1., 10., 100.):
            with self.subTest(scale=scale):
                view = TrackingSettingsView()
                cams = pinhole_cameras(scale)
                view.detected_cam_files = [str(i) for i in range(len(cams))]
                with patch("gui.views.tracking_settings_view.QMessageBox.warning") as warning:
                    view._estimate_volume_from_cameras(cams)
                warning.assert_not_called()
                for lo, hi in ((view.vol_x_min, view.vol_x_max), (view.vol_y_min, view.vol_y_max),
                               (view.vol_z_min, view.vol_z_max)):
                    self.assertAlmostEqual((hi.value() - lo.value()) / scale, 5., delta=.3)
                view.close()


if __name__ == "__main__":
    unittest.main()
