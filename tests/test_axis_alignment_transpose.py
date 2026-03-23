import numpy as np
import math

from modules.camera_calibration.wand_calibration.refractive_geometry import (
    align_world_to_axis_directions,
)


def _mock_triangulate_fn(landmark_3d):
    """Return a fixed landmark dict, ignoring the observations argument."""
    def _fn(obs_by_landmark):
        return landmark_3d
    return _fn


def test_axis_alignment_non_aligned_landmarks():
    """Regression: 45-degree rotated axis landmarks must produce correct R_world.

    dir_X points at 45 degrees in the XY plane of the old frame.
    After alignment, R_world @ dir_X should equal [1, 0, 0] (canonical X).
    Similarly for dir_Y and dir_Z.
    """
    s = 1.0 / math.sqrt(2.0)

    # Landmarks in old (world) frame where the user-detected axes are NOT aligned
    # with canonical X/Y/Z.
    #
    #  detected +X direction: (1/sqrt2, 1/sqrt2, 0)
    #  detected +Y direction: (-1/sqrt2, 1/sqrt2, 0)
    #  detected +Z direction: (0, 0, 1)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    pt_X   = np.array([s,   s,  0.0], dtype=np.float64)  # 45 deg in XY
    pt_Y   = np.array([-s,  s,  0.0], dtype=np.float64)  # 135 deg in XY
    pt_Z   = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    landmark_3d = {
        "center": center,
        "+X":    pt_X,
        "+Y":    pt_Y,
        "+Z":    pt_Z,
    }

    # axis_direction_map needs >= 2 cameras so coverage check passes
    axis_direction_map = {
        0: {"center": [0.0, 0.0], "+X": [1.0, 0.0], "+Y": [0.0, 1.0], "+Z": [0.5, 0.5]},
        1: {"center": [0.0, 0.0], "+X": [1.0, 0.0], "+Y": [0.0, 1.0], "+Z": [0.5, 0.5]},
    }

    success, R_world, t_shift, state = align_world_to_axis_directions(
        axis_direction_map=axis_direction_map,
        triangulate_fn=_mock_triangulate_fn(landmark_3d),
        cam_params={},
        points_3d=np.zeros((4, 3)),
    )

    assert success is True, "align_world_to_axis_directions returned failure"
    assert R_world is not None

    # R_world must be a proper rotation (orthonormal, det=+1)
    np.testing.assert_allclose(
        R_world @ R_world.T, np.eye(3), atol=1e-10,
        err_msg="R_world is not orthonormal"
    )
    assert abs(np.linalg.det(R_world) - 1.0) < 1e-10, "det(R_world) != +1"

    # Core invariant: rotating the detected direction vectors by R_world
    # should yield the canonical basis vectors.
    dir_X = (pt_X - center) / np.linalg.norm(pt_X - center)
    dir_Y = (pt_Y - center) / np.linalg.norm(pt_Y - center)
    dir_Z = (pt_Z - center) / np.linalg.norm(pt_Z - center)

    np.testing.assert_allclose(
        R_world @ dir_X, np.array([1.0, 0.0, 0.0]), atol=1e-10,
        err_msg="R_world @ dir_X != [1,0,0]"
    )
    np.testing.assert_allclose(
        R_world @ dir_Y, np.array([0.0, 1.0, 0.0]), atol=1e-10,
        err_msg="R_world @ dir_Y != [0,1,0]"
    )
    np.testing.assert_allclose(
        R_world @ dir_Z, np.array([0.0, 0.0, 1.0]), atol=1e-10,
        err_msg="R_world @ dir_Z != [0,0,1]"
    )

    # t_shift should equal -center (which is zero here)
    np.testing.assert_allclose(t_shift, -center, atol=1e-10)


def test_axis_alignment_left_handed_triad():
    """Regression: left-handed input triad exercises the det<0 correction branch.

    If we give dir_X, dir_Y, dir_Z that form a left-handed system, the SVD
    det-check must flip one column to restore det=+1. The resulting R_world
    must still be a proper rotation (det=+1) and orthonormal.
    """
    # Left-handed triad: swap X and Y relative to a right-handed system.
    # dir_X = [0,1,0], dir_Y = [1,0,0], dir_Z = [0,0,1]  -> det(M) = -1
    center = np.array([5.0, 3.0, 2.0], dtype=np.float64)
    d = 2.0  # distance from center to axis point
    pt_X = center + d * np.array([0.0, 1.0, 0.0], dtype=np.float64)  # points along old Y
    pt_Y = center + d * np.array([1.0, 0.0, 0.0], dtype=np.float64)  # points along old X
    pt_Z = center + d * np.array([0.0, 0.0, 1.0], dtype=np.float64)  # points along old Z

    # Verify this is indeed left-handed (det(M) = -1)
    dir_X_raw = pt_X - center
    dir_Y_raw = pt_Y - center
    dir_Z_raw = pt_Z - center
    M_raw = np.column_stack([
        dir_X_raw / np.linalg.norm(dir_X_raw),
        dir_Y_raw / np.linalg.norm(dir_Y_raw),
        dir_Z_raw / np.linalg.norm(dir_Z_raw),
    ])
    assert np.linalg.det(M_raw) < 0, "Test setup error: expected left-handed triad (det<0)"

    landmark_3d = {
        "center": center,
        "+X":    pt_X,
        "+Y":    pt_Y,
        "+Z":    pt_Z,
    }

    axis_direction_map = {
        0: {"center": [0.0, 0.0], "+X": [1.0, 0.0], "+Y": [0.0, 1.0], "+Z": [0.5, 0.5]},
        1: {"center": [0.0, 0.0], "+X": [1.0, 0.0], "+Y": [0.0, 1.0], "+Z": [0.5, 0.5]},
    }

    success, R_world, t_shift, state = align_world_to_axis_directions(
        axis_direction_map=axis_direction_map,
        triangulate_fn=_mock_triangulate_fn(landmark_3d),
        cam_params={},
        points_3d=np.zeros((4, 3)),
    )

    assert success is True, "align_world_to_axis_directions returned failure for left-handed triad"
    assert R_world is not None

    # R_world must be a proper rotation
    np.testing.assert_allclose(
        R_world @ R_world.T, np.eye(3), atol=1e-10,
        err_msg="R_world is not orthonormal (left-handed triad)"
    )
    det_R = np.linalg.det(R_world)
    assert abs(det_R - 1.0) < 1e-10, f"det(R_world)={det_R:.8f} != +1 for left-handed triad"

    # t_shift should equal -center
    np.testing.assert_allclose(t_shift, -center, atol=1e-10)


def test_buggy_formula_would_fail():
    """Documents the regression: the old formula (U @ Vt without .T) is wrong.

    This test directly demonstrates that R = U @ Vt gives the WRONG mapping
    for non-identity axis directions, while R = (U @ Vt).T gives the correct one.
    This serves as an in-code proof of why the fix was necessary.
    """
    s = 1.0 / math.sqrt(2.0)

    dir_X = np.array([s,   s,  0.0])
    dir_Y = np.array([-s,  s,  0.0])
    dir_Z = np.array([0.0, 0.0, 1.0])

    M = np.column_stack([dir_X, dir_Y, dir_Z])
    U, _S, Vt = np.linalg.svd(M)

    R_buggy = U @ Vt        # old (wrong) formula
    R_fixed = (U @ Vt).T    # correct formula

    # Buggy formula does NOT map dir_X -> [1,0,0]
    assert not np.allclose(R_buggy @ dir_X, [1.0, 0.0, 0.0], atol=1e-6), (
        "Buggy formula unexpectedly maps dir_X correctly -- test assumptions are wrong"
    )

    # Fixed formula DOES map dir_X -> [1,0,0]
    np.testing.assert_allclose(
        R_fixed @ dir_X, np.array([1.0, 0.0, 0.0]), atol=1e-10,
        err_msg="Fixed formula does not map dir_X to [1,0,0]"
    )
    np.testing.assert_allclose(
        R_fixed @ dir_Y, np.array([0.0, 1.0, 0.0]), atol=1e-10,
        err_msg="Fixed formula does not map dir_Y to [0,1,0]"
    )
    np.testing.assert_allclose(
        R_fixed @ dir_Z, np.array([0.0, 0.0, 1.0]), atol=1e-10,
        err_msg="Fixed formula does not map dir_Z to [0,0,1]"
    )
