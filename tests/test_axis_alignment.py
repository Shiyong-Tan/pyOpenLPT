import numpy as np
import cv2
from scipy.optimize import OptimizeResult

from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator


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
