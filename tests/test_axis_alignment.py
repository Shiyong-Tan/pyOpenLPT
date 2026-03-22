import numpy as np
import cv2
from scipy.optimize import OptimizeResult

from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator
from tests.test_utils_axis_alignment import (
    generate_synthetic_pinhole_setup,
    project_points,
    compute_orthogonality_metrics,
)
from modules.camera_calibration.wand_calibration.refractive_geometry import (
    align_world_to_axis_directions,
    triangulate_pinhole_landmarks,
)


def _build_pinhole_finalize_fixture():
    calibrator = WandCalibrator()
    calibrator.image_size = (1000, 1000)
    calibrator.wand_length = 50.0

    # Keep IDs as 0/1 to match current _parse_results lookup behavior.
    cam_id_map = {0: 0, 1: 1}

    # 11 params/cam: [rvec(3), tvec(3), f, cx, cy, k1, k2]
    cam_params = np.array(
        [
            [0.0, 0.0, 0.0, -40.0, 5.0, 1200.0, 900.0, 500.0, 500.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 35.0, -10.0, 1180.0, 920.0, 500.0, 500.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Two frames -> four 3D points. Centroid intentionally not zero.
    points_3d = np.array(
        [
            [110.0, 25.0, 700.0],
            [160.0, 25.0, 700.0],
            [130.0, -20.0, 760.0],
            [180.0, -20.0, 760.0],
        ],
        dtype=np.float64,
    )

    frame_list = [0, 1]
    wand_data = {}

    for i, fid in enumerate(frame_list):
        pt3d_A = points_3d[i * 2]
        pt3d_B = points_3d[i * 2 + 1]
        frame_obs = {}

        for cam_id in cam_id_map:
            cp = cam_params[cam_id]
            uv_A = calibrator._project_point(pt3d_A, cp, calibrator.image_size)
            uv_B = calibrator._project_point(pt3d_B, cp, calibrator.image_size)
            frame_obs[cam_id] = (uv_A, uv_B)

        wand_data[fid] = frame_obs

    calibrator.wand_points = wand_data

    full_params = np.hstack([cam_params.reshape(-1), points_3d.reshape(-1)])
    res = OptimizeResult(x=full_params, message="Optimization converged")

    # Expected camera translation after centroid recentering in _finalize_calibration.
    centroid = np.mean(points_3d, axis=0)
    expected_cam_params = cam_params.copy()
    for i in range(len(cam_id_map)):
        rvec = expected_cam_params[i, 0:3]
        tvec = expected_cam_params[i, 3:6]
        R, _ = cv2.Rodrigues(rvec)
        expected_cam_params[i, 3:6] = tvec + R @ centroid

    return calibrator, res, cam_id_map, wand_data, frame_list, expected_cam_params


def test_pinhole_finalize_consistency():
    (
        calibrator,
        res,
        cam_id_map,
        wand_data,
        frame_list,
        expected_cam_params,
    ) = _build_pinhole_finalize_fixture()

    success, _, _ = calibrator._finalize_calibration(
        res=res,
        cam_id_map=cam_id_map,
        wand_data=wand_data,
        wand_length_mm=calibrator.wand_length,
        frame_list=frame_list,
    )

    assert success is True
    assert calibrator.points_3d is not None
    assert np.isfinite(calibrator.points_3d).all()

    # Recentered world-frame check: point cloud centroid should be at origin.
    np.testing.assert_allclose(np.mean(calibrator.points_3d, axis=0), np.zeros(3), atol=1e-10, rtol=0)

    # Final parsed camera translations must match recentered translations.
    for cam_id in sorted(cam_id_map.keys()):
        got_t = calibrator.final_params[cam_id]["T"].reshape(3)
        expected_t = expected_cam_params[cam_id, 3:6]
        np.testing.assert_allclose(got_t, expected_t, atol=1e-10, rtol=0)

    # Ensure camera centers are finite and consistent with final parsed extrinsics.
    for cam_id in sorted(cam_id_map.keys()):
        R = calibrator.final_params[cam_id]["R"]
        T = calibrator.final_params[cam_id]["T"]
        center = (-R.T @ T).reshape(3)
        assert np.isfinite(center).all()


def test_align_world_to_axis_directions_happy():
    """Happy path: alignment computes valid orthonormal R and correct t_shift."""
    K_list, R_list, T_list, points_3d = generate_synthetic_pinhole_setup(num_cams=2, num_points=20)

    # Known landmarks (3D)
    axis_center_3d = np.array([0.0, 0.0, 600.0])
    axis_x_3d = np.array([10.0, 0.0, 600.0])
    axis_y_3d = np.array([0.0, 10.0, 600.0])
    axis_z_3d = np.array([0.0, 0.0, 610.0])
    landmark_3d = {
        "center": axis_center_3d,
        "+X": axis_x_3d,
        "+Y": axis_y_3d,
        "+Z": axis_z_3d,
    }

    # Create 2D observations by projecting landmarks
    axis_direction_map = {}
    for cam_idx in range(2):
        cam_landmarks = {}
        for lm_name, pt3d in landmark_3d.items():
            obs_2d = project_points(K_list[cam_idx], R_list[cam_idx], T_list[cam_idx], pt3d.reshape(1, 3))
            cam_landmarks[lm_name] = obs_2d[0].tolist()
        axis_direction_map[cam_idx] = cam_landmarks

    # Mock triangulate_fn that returns known landmarks (simulating perfect triangulation)
    def mock_triangulate_fn(obs_by_landmark):
        return landmark_3d

    success, R_new, t_shift, state = align_world_to_axis_directions(
        axis_direction_map,
        mock_triangulate_fn,
        {},
        points_3d,
    )

    assert success is True
    assert R_new is not None
    metrics = compute_orthogonality_metrics(R_new)
    assert metrics["frob_norm_error"] < 1e-10
    assert abs(metrics["det"] - 1.0) < 1e-10
    # t_shift should be -center
    np.testing.assert_allclose(t_shift, -axis_center_3d, atol=1e-10)
    assert isinstance(state, dict)


def test_align_world_to_axis_directions_coverage_fail():
    """Failure case: only 1 camera per landmark -> should fail with success=False."""
    # axis_direction_map with only 1 camera
    axis_direction_map = {
        0: {
            "center": [100.0, 200.0],
            "+X": [150.0, 200.0],
            "+Y": [100.0, 250.0],
            "+Z": [100.0, 200.0],
        }
    }

    def mock_triangulate_fn(obs_by_landmark):
        return {}

    success, R_new, t_shift, state = align_world_to_axis_directions(
        axis_direction_map,
        mock_triangulate_fn,
        {},
        np.zeros((4, 3)),
        validate_coverage=True,
    )

    assert success is False
    assert R_new is None


def test_triangulate_pinhole_landmarks_happy():
    K_list, R_list, T_list, _ = generate_synthetic_pinhole_setup(num_cams=2, num_points=4)

    landmark_names = ["center", "+X", "+Y", "+Z"]
    gt_landmarks = {
        "center": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "+X": np.array([0.25, 0.0, 0.0], dtype=np.float64),
        "+Y": np.array([0.0, 0.25, 0.0], dtype=np.float64),
        "+Z": np.array([0.0, 0.0, 0.25], dtype=np.float64),
    }

    obs_by_landmark = {lm: {} for lm in landmark_names}
    for cam_id in range(2):
        cam_pts = np.vstack([gt_landmarks[lm] for lm in landmark_names])
        uv = project_points(K_list[cam_id], R_list[cam_id], T_list[cam_id], cam_pts)
        for i, lm in enumerate(landmark_names):
            obs_by_landmark[lm][cam_id] = uv[i].tolist()

    cam_params_dict = {
        cam_id: {
            "K": K_list[cam_id],
            "R": R_list[cam_id],
            "T": T_list[cam_id],
            "dist": np.zeros(2, dtype=np.float64),
        }
        for cam_id in range(2)
    }

    out = triangulate_pinhole_landmarks(obs_by_landmark, cam_params_dict)

    assert list(out.keys()) == landmark_names
    for lm in landmark_names:
        pt = np.asarray(out[lm], dtype=np.float64).reshape(3)
        assert pt.shape == (3,)
        assert np.isfinite(pt).all()
        assert np.linalg.norm(pt - gt_landmarks[lm]) < 1.0


def test_pinhole_alignment_roundtrip():
    """Round-trip pinhole alignment path: triangulate landmarks then align world frame."""
    K_list, R_list, T_list, points_3d = generate_synthetic_pinhole_setup(num_cams=2, num_points=12)

    landmark_3d = {
        "center": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "+X": np.array([0.2, 0.0, 0.0], dtype=np.float64),
        "+Y": np.array([0.0, 0.2, 0.0], dtype=np.float64),
        "+Z": np.array([0.0, 0.0, 0.2], dtype=np.float64),
    }

    axis_direction_map = {}
    for cam_id in range(2):
        cam_landmarks = {}
        for lm_name, pt3d in landmark_3d.items():
            uv = project_points(K_list[cam_id], R_list[cam_id], T_list[cam_id], pt3d.reshape(1, 3))
            cam_landmarks[lm_name] = uv[0].tolist()
        axis_direction_map[cam_id] = cam_landmarks

    cam_params_dict = {
        cam_id: {
            "K": K_list[cam_id],
            "R": R_list[cam_id],
            "T": T_list[cam_id],
            "dist": np.zeros(2, dtype=np.float64),
        }
        for cam_id in range(2)
    }

    # align_world_to_axis_directions -> apply_coordinate_rotation expects flattened
    # pinhole camera vectors: [rvec(3), tvec(3), f, cx, cy, k1, k2]
    cam_params_for_align = {}
    for cam_id in range(2):
        rvec, _ = cv2.Rodrigues(R_list[cam_id])
        tvec = T_list[cam_id].reshape(3)
        fx = float(K_list[cam_id][0, 0])
        cx = float(K_list[cam_id][0, 2])
        cy = float(K_list[cam_id][1, 2])
        cam_params_for_align[cam_id] = np.array(
            [rvec[0, 0], rvec[1, 0], rvec[2, 0], tvec[0], tvec[1], tvec[2], fx, cx, cy, 0.0, 0.0],
            dtype=np.float64,
        )

    obs_by_landmark = {"center": {}, "+X": {}, "+Y": {}, "+Z": {}}
    for cam_id, cam_data in axis_direction_map.items():
        for lm in ["center", "+X", "+Y", "+Z"]:
            obs_by_landmark[lm][cam_id] = cam_data[lm]

    triangulated = triangulate_pinhole_landmarks(obs_by_landmark, cam_params_dict)
    assert set(triangulated.keys()) == {"center", "+X", "+Y", "+Z"}

    def _triangulate_fn(obs):
        return triangulate_pinhole_landmarks(obs, cam_params_dict)

    success, R_new, t_shift, aligned_state = align_world_to_axis_directions(
        axis_direction_map=axis_direction_map,
        triangulate_fn=_triangulate_fn,
        cam_params=cam_params_for_align,
        points_3d=points_3d,
    )

    assert success is True
    assert R_new is not None
    assert t_shift is not None
    assert isinstance(aligned_state, dict)
    assert isinstance(aligned_state.get("cam_params"), dict)
