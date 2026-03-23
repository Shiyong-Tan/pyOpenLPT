"""
Regression tests for list/float TypeError fixes in plot_calibration and
calculate_per_frame_errors.
"""

import numpy as np
import cv2
from modules.camera_calibration.wand_calibration.wand_calibrator import (
    WandCalibrator
)


def test_plot_calibration_handles_list_input():
    """
    Regression test: plot_calibration() should NOT crash when points_3d
    is a Python list.

    Root cause: After axis alignment, wand_calibrator.points_3d was
    stored as list via .tolist(). plot_calibration() then attempted
    `list / float` → TypeError. Fix: np.asarray() guards all division
    sites.
    """
    from unittest.mock import MagicMock
    import sys

    # Mock PySide6 modules to prevent Qt import errors in headless test
    sys.modules['PySide6'] = MagicMock()
    sys.modules['PySide6.QtWidgets'] = MagicMock()
    sys.modules['PySide6.QtCore'] = MagicMock()
    sys.modules['PySide6.QtGui'] = MagicMock()

    from modules.camera_calibration.view import Calibration3DViewer

    # Create a minimal mock `self` that satisfies plot_calibration()
    # attribute accesses
    mock_self = MagicMock()
    mock_self.ax = MagicMock()
    mock_self.canvas = MagicMock()

    cameras = {
        0: {
            "R": np.eye(3),
            "T": np.array([[0.0], [0.0], [1000.0]]),
            "K": np.eye(3),
        }
    }

    # The bug scenario: pass a PYTHON LIST (not numpy array) for
    # points_3d
    points_3d_list = [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]

    # This should NOT raise TypeError: unsupported operand type(s)
    # for /: 'list' and 'float'
    # The fix: np.asarray(points_3d) at line 628 converts list to
    # array before /
    Calibration3DViewer.plot_calibration(
        mock_self, cameras=cameras, points_3d=points_3d_list
    )

    # If we reach here, no TypeError was raised — the fix worked


def test_calculate_per_frame_errors_with_numpy_points():
    """
    Confirm points_3d stays numpy after fix;
    calculate_per_frame_errors() works.
    """

    wand = WandCalibrator()

    # 2 frames → 4 points (2 points per frame)
    wand.points_3d = np.array([
        [0.0, 0.0, 500.0],   # frame 0, point A
        [50.0, 0.0, 500.0],  # frame 0, point B
        [0.0, 0.0, 510.0],   # frame 1, point A
        [50.0, 0.0, 510.0],  # frame 1, point B
    ], dtype=np.float64)
    wand.wand_length = 50.0

    # Build camera params for 1 camera
    K = np.array(
        [[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=np.float64
    )
    R = np.eye(3, dtype=np.float64)
    T = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    wand.final_params = {
        0: {"R": R, "T": T, "K": K, "dist": dist}
    }

    # Wand observations for 2 frames (project the 3D points to get 2D
    # observations)
    def project(pt3d):
        rvec = np.zeros(3, dtype=np.float64)
        uv, _ = cv2.projectPoints(pt3d.reshape(1, 3), rvec, T, K, dist)
        return uv.flatten()[:2].tolist()

    wand.wand_points = {
        0: {0: [project(wand.points_3d[0]), project(wand.points_3d[1])]},
        1: {0: [project(wand.points_3d[2]), project(wand.points_3d[3])]},
    }
    wand.wand_points_filtered = None
    # Must clear cache so calculate_per_frame_errors() runs
    wand.per_frame_errors = {}

    # Verify points_3d is numpy
    assert isinstance(
        wand.points_3d, np.ndarray
    ), f"Expected numpy, got {type(wand.points_3d)}"

    # This should NOT raise TypeError if points_3d is numpy
    # The bug was: if points_3d was a list, then
    # `pt3d_A = self.points_3d[idx_A]` gives a list,
    # and cv2.projectPoints(pt3d_A.reshape(1,3), ...) fails because
    # list has no reshape method.
    errors = wand.calculate_per_frame_errors()

    assert errors is not None
    assert len(errors) == 2, \
        f"Expected 2 frame errors, got {len(errors)}"
    assert 0 in errors
    assert 1 in errors

    # Verify structure
    for fid in [0, 1]:
        assert 'cam_errors' in errors[fid]
        assert 'len_error' in errors[fid]
        assert 0 in errors[fid]['cam_errors']  # cam_id=0

        # Length error should be near 0 since we constructed perfect
        # wand points
        assert errors[fid]['len_error'] < 1.0, \
            f"Frame {fid} len_error too high: {errors[fid]['len_error']}"
