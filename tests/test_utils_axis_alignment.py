# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false
import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Return unit-norm vector."""
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n <= 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def _look_at_rotation(camera_center: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Build world->camera rotation for a camera looking at target."""
    if target is None:
        target = np.zeros(3, dtype=np.float64)

    c = np.asarray(camera_center, dtype=np.float64).reshape(3)
    t = np.asarray(target, dtype=np.float64).reshape(3)

    forward = _normalize(t - c)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(forward, world_up)) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = _normalize(np.cross(forward, world_up))
    down = np.cross(forward, right)

    return np.vstack([right, down, forward])


def _projection_matrix(K: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Construct 3x4 projection matrix P = K [R|T]."""
    return np.asarray(K, dtype=np.float64) @ np.hstack([
        np.asarray(R, dtype=np.float64),
        np.asarray(T, dtype=np.float64).reshape(3, 1),
    ])


def generate_synthetic_pinhole_setup(num_cams=2, num_points=4):
    """Returns (K_list, R_list, T_list, points_3d) for a synthetic 2-camera setup."""
    if num_cams < 2:
        raise ValueError("num_cams must be >= 2")
    if num_points < 1:
        raise ValueError("num_points must be >= 1")

    rng = np.random.default_rng(0)

    fx = fy = 900.0
    cx, cy = 640.0, 400.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    K_list, R_list, T_list = [], [], []
    radius = 2.5

    for i in range(num_cams):
        theta = 2.0 * np.pi * i / num_cams
        z = 0.35 * np.sin(theta)
        camera_center = np.array([radius * np.cos(theta), radius * np.sin(theta), z], dtype=np.float64)

        R = _look_at_rotation(camera_center, target=np.zeros(3, dtype=np.float64))
        T = (-R @ camera_center.reshape(3, 1)).astype(np.float64)

        K_list.append(K.copy())
        R_list.append(R)
        T_list.append(T)

    points_3d = rng.uniform(low=[-0.4, -0.3, -0.25], high=[0.4, 0.3, 0.25], size=(num_points, 3))
    return K_list, R_list, T_list, points_3d.astype(np.float64)


def project_points(K, R, T, points_3d, noise_std=0.0):
    """Project 3D points to 2D image using pinhole model. Returns (N,2) array."""
    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3)")

    P = _projection_matrix(K, R, T)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    img_h = (P @ pts_h.T).T
    uv = img_h[:, :2] / img_h[:, 2:3]

    if noise_std > 0.0:
        rng = np.random.default_rng(12345)
        uv = uv + rng.normal(0.0, noise_std, size=uv.shape)

    return uv.astype(np.float64)


def _triangulate_point_dlt(obs_2d: list, proj_mats: list) -> np.ndarray:
    """Triangulate one 3D point via N-view DLT/SVD."""
    if len(obs_2d) < 2:
        raise ValueError("Need at least 2 observations to triangulate")

    A_rows = []
    for (u, v), p_mat in zip(obs_2d, proj_mats):
        p_arr = np.asarray(p_mat, dtype=np.float64)
        A_rows.append(u * p_arr[2, :] - p_arr[0, :])
        A_rows.append(v * p_arr[2, :] - p_arr[1, :])

    A = np.asarray(A_rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    X_h = vt[-1, :]
    X = X_h[:3] / X_h[3]
    return X


def triangulate_synthetic_landmarks(obs_by_cam, K_list, R_list, T_list):
    """Triangulate 3D points from multi-view 2D observations.
    obs_by_cam: dict {cam_idx: (N,2) array of 2D points}
    Returns (N,3) array of 3D points
    """
    if len(obs_by_cam) < 2:
        raise ValueError("Need observations from at least 2 cameras")

    cam_ids = sorted(obs_by_cam.keys())
    n_points = np.asarray(obs_by_cam[cam_ids[0]], dtype=np.float64).shape[0]
    proj_mats = [_projection_matrix(K_list[c], R_list[c], T_list[c]) for c in cam_ids]

    points = []
    for i in range(n_points):
        obs_2d = [np.asarray(obs_by_cam[c], dtype=np.float64)[i] for c in cam_ids]
        points.append(_triangulate_point_dlt(obs_2d, proj_mats))

    return np.asarray(points, dtype=np.float64)


def compute_orthogonality_metrics(R):
    """Compute rotation matrix quality metrics.
    Returns dict with keys: 'frob_norm_error', 'det', 'singular_values', 'condition_number'
    """
    M = np.asarray(R, dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError("R must be 3x3")

    gram = M.T @ M
    frob_norm_error = float(np.linalg.norm(gram - np.eye(3), ord="fro"))
    det_val = float(np.linalg.det(M))
    singular_values = np.linalg.svd(M, compute_uv=False)
    condition_number = float(singular_values[0] / max(singular_values[-1], 1e-15))

    return {
        "frob_norm_error": frob_norm_error,
        "det": det_val,
        "singular_values": singular_values,
        "condition_number": condition_number,
    }


def compute_reprojection_error(points_3d, obs_by_cam, K_list, R_list, T_list):
    """Compute RMS reprojection error across all cameras.
    Returns float (RMS pixel error)
    """
    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3)")

    residuals = []
    for cam_idx, obs in obs_by_cam.items():
        obs_arr = np.asarray(obs, dtype=np.float64)
        if obs_arr.shape != (pts.shape[0], 2):
            raise ValueError(f"obs for cam {cam_idx} must have shape ({pts.shape[0]}, 2)")

        proj = project_points(K_list[cam_idx], R_list[cam_idx], T_list[cam_idx], pts)
        diff = proj - obs_arr
        residuals.append(np.sum(diff * diff, axis=1))

    if not residuals:
        raise ValueError("obs_by_cam is empty")

    sq_errors = np.concatenate(residuals)
    return float(np.sqrt(np.mean(sq_errors)))
