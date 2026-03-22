# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false
import numpy as np

from tests.test_utils_axis_alignment import (
    generate_synthetic_pinhole_setup,
    project_points,
    triangulate_synthetic_landmarks,
    compute_orthogonality_metrics,
    compute_reprojection_error,
)


def test_synthetic_pinhole_setup():
    """Confirms synthetic data generation works."""
    K_list, R_list, T_list, points_3d = generate_synthetic_pinhole_setup(num_cams=2, num_points=4)

    assert len(K_list) == 2
    assert len(R_list) == 2
    assert len(T_list) == 2
    assert points_3d.shape == (4, 3)

    obs_by_cam = {
        i: project_points(K_list[i], R_list[i], T_list[i], points_3d)
        for i in range(2)
    }

    for obs in obs_by_cam.values():
        assert obs.shape == (4, 2)
        assert np.all(np.isfinite(obs))

    triangulated = triangulate_synthetic_landmarks(obs_by_cam, K_list, R_list, T_list)
    assert triangulated.shape == (4, 3)
    assert np.all(np.isfinite(triangulated))


def test_orthogonality_metrics():
    """Verifies metric computation: identity R → metric ≈ 0; random R → metric >> 0."""
    rng = np.random.default_rng(7)

    I = np.eye(3)
    metrics = compute_orthogonality_metrics(I)
    assert metrics["frob_norm_error"] < 1e-10
    assert abs(metrics["det"] - 1.0) < 1e-10

    M = rng.normal(size=(3, 3))
    metrics_bad = compute_orthogonality_metrics(M)
    assert metrics_bad["frob_norm_error"] > 0.1


def test_reprojection_baseline():
    """Establishes pre-alignment accuracy baseline."""
    K_list, R_list, T_list, points_3d = generate_synthetic_pinhole_setup(num_cams=2, num_points=10)

    obs_by_cam = {
        i: project_points(K_list[i], R_list[i], T_list[i], points_3d)
        for i in range(2)
    }

    rms = compute_reprojection_error(points_3d, obs_by_cam, K_list, R_list, T_list)
    assert rms < 1e-6  # Near-zero for exact (noiseless) setup
