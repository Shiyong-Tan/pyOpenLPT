import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Ray:
    o: np.ndarray      # (3,) world coord
    d: np.ndarray      # (3,) normalized, world coord
    valid: bool
    reason: str = ""
    cam_id: int = -1
    window_id: int = -1
    frame_id: int = -1
    endpoint: str = "" # "A" or "B"
    uv: Tuple[float, float] = (np.nan, np.nan)

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm

def _extract_line3d(line):
    """
    Robustly extract pt and unit_vector from a C++ Line3D object.
    Supports multiple binding styles.
    """
    # Try .pt and .unit_vector (found in pyMatrix.cpp)
    if hasattr(line, "pt") and hasattr(line, "unit_vector"):
        o = np.array([line.pt[0], line.pt[1], line.pt[2]])
        d = np.array([line.unit_vector[0], line.unit_vector[1], line.unit_vector[2]])
        return o, d
    
    # Try .p0 and .p1 (alternative style)
    if hasattr(line, "p0") and hasattr(line, "p1"):
        o = np.array([line.p0[0], line.p0[1], line.p0[2]])
        p1 = np.array([line.p1[0], line.p1[1], line.p1[2]])
        return o, normalize(p1 - o)
    
    # Try getter methods
    try:
        pt = line.getPoint()
        vec = line.getVector()
        return np.array([pt[0], pt[1], pt[2]]), np.array([vec[0], vec[1], vec[2]])
    except AttributeError:
        pass

    # If all fail, provide diagnostic info
    attrs = dir(line)
    raise AttributeError(f"Could not extract geometry from Line3D object. Available attributes: {attrs}")

# Removed _pixel_to_undist_normalized as pinplateLine expects pixels when cam_mtx is set

_PROBE_LOGGED = False
_RAY_ORIGIN_AUDIT_COUNT = 0
_RAY_ORIGIN_AUDIT_PRINTED = False

def build_pinplate_ray_cpp(cam, uv, *, cam_id=-1, window_id=-1, frame_id=-1, endpoint="") -> Ray:
    """
    Build object-side refracted ray using C++ lineOfSight.
    
    Input 'uv' should be PIXEL coordinates.
    The C++ Camera.lineOfSight() handles undistortion and normalization internally.
    """
    global _PROBE_LOGGED, _RAY_ORIGIN_AUDIT_COUNT, _RAY_ORIGIN_AUDIT_PRINTED
    import pyopenlpt as lpt
    
    u, v = float(uv[0]), float(uv[1])
    
    try:
        # Use lineOfSight which accepts pixel coordinates directly
        # It calls undistort() internally, then passes to pinplateLine()
        line = cam.lineOfSight(lpt.Pt2D(u, v))
        o, d = _extract_line3d(line)
        
        if np.all(np.isfinite(o)) and np.all(np.isfinite(d)):
            if not _PROBE_LOGGED:
                # print(f"[Refractive] lineOfSight: Input ({u:.1f},{v:.1f}) pixel coords")
                _PROBE_LOGGED = True
            
            # === [C] RAY ORIGIN AUDIT (first 10 rays) ===
            _RAY_ORIGIN_AUDIT_COUNT += 1
            if _RAY_ORIGIN_AUDIT_COUNT <= 10:
                # Get camera center from t_vec_inv
                try:
                    pp = cam._pinplate_param
                    t_vec_inv = pp.t_vec_inv
                    C_cam = np.array([t_vec_inv[0], t_vec_inv[1], t_vec_inv[2]])
                    dist_O_C = np.linalg.norm(o - C_cam)
                    
                    if not _RAY_ORIGIN_AUDIT_PRINTED:
                        print("\n=== RAY ORIGIN AUDIT (first 10 rays) ===")
                        _RAY_ORIGIN_AUDIT_PRINTED = True
                    
                    print(f"Ray {_RAY_ORIGIN_AUDIT_COUNT}: O=[{o[0]:.2f},{o[1]:.2f},{o[2]:.2f}], C_cam=[{C_cam[0]:.2f},{C_cam[1]:.2f},{C_cam[2]:.2f}], ||O-C||={dist_O_C:.3f}mm")
                    
                    if _RAY_ORIGIN_AUDIT_COUNT == 10:
                        print("=== END RAY ORIGIN AUDIT ===\n")
                except:
                    pass  # Silently skip if C_cam extraction fails
            
            # Sanity check on direction z (should be >> 0 for forward facing)
            d_norm = normalize(d)
            if abs(d_norm[2]) < 0.1:
                # Diagnostics for the "collapsed ray" issue
                return Ray(o=o, d=d_norm, valid=False, reason=f"collapsed_ray_z={d_norm[2]:.4f}", 
                           cam_id=cam_id, window_id=window_id, frame_id=frame_id, endpoint=endpoint, uv=(u, v))
                
            return Ray(o=o, d=d_norm, valid=True, cam_id=cam_id, window_id=window_id, frame_id=frame_id, endpoint=endpoint, uv=(u, v))
            
    except Exception as e:
        reason = f"C++ lineOfSight failed: {str(e)}"
        if "total internal reflection" in reason.lower():
             reason = "total_internal_reflection"
        return Ray(o=np.zeros(3), d=np.array([0, 0, 1.0]), valid=False, reason=reason, 
                   cam_id=cam_id, window_id=window_id, frame_id=frame_id, endpoint=endpoint, uv=(u, v))
    
    # Fallback
    return Ray(o=np.zeros(3), d=np.array([0, 0, 1.0]), valid=False, reason="extraction_failed", 
               cam_id=cam_id, window_id=window_id, frame_id=frame_id, endpoint=endpoint, uv=(u, v))

def triangulate_point(rays_list: list) -> Tuple[np.ndarray, float, bool, str]:
    """
    Intersection of N rays in 3D using least squares.
    Formula: Σ (I - di*di.T) * X = Σ (I - di*di.T) * oi
    """
    valid_rays = [r for r in rays_list if r.valid]
    if len(valid_rays) < 2:
        return np.zeros(3), 0.0, False, "insufficient_valid_rays"
    
    A = np.zeros((3, 3))
    b = np.zeros(3)
    I = np.eye(3)
    
    for ray in valid_rays:
        d = ray.d.reshape(3, 1)
        # (I - d*d.T)
        proj_perp = I - d @ d.T
        A += proj_perp
        b += proj_perp @ ray.o
    
    try:
        # Check conditioning
        cond = np.linalg.cond(A)
        # Using solve for better accuracy if conditioned well
        if cond < 1e12:
            X = np.linalg.solve(A, b)
            return X, cond, True, ""
        else:
            # Ill-conditioned: Fallback to lstsq but mark as success=False for wand triangulation
            X, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            # For wand calib, we usually want 3 rays and full rank
            return X, cond, False, f"ill_conditioned: cond={cond:.2e}, rank={rank}"
    except (np.linalg.LinAlgError, ValueError) as e:
        return np.zeros(3), np.inf, False, f"linalg_error: {e}"

def point_to_ray_dist(X: np.ndarray, o: np.ndarray, d: np.ndarray) -> float:
    """
    Distance from point X to ray (o, d) treated as a HALF-LINE.
    If projection is backward (t < 0), clamp to ray origin.
    
    This enforces forward-only ray geometry to prevent non-physical
    solutions where objects appear on the camera-side of refractive planes.
    """
    d = d / (np.linalg.norm(d) + 1e-12)  # safety normalize
    v = X - o
    t = float(np.dot(v, d))

    if t >= 0.0:
        # Forward projection: use perpendicular distance
        v_perp = v - t * d
        return float(np.linalg.norm(v_perp))
    else:
        # Backward projection: clamp to origin, return distance to origin
        return float(np.linalg.norm(v))


def point_to_ray_dist_vec(X: np.ndarray, O: np.ndarray, D: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Vectorized INFINITE LINE distance from point X to rays (O, D).
    
    Ray distance residual measures only geometric deviation from the ray line
    and is invariant to ray origin translation. Physical feasibility 
    (point ordering / plane side) is enforced separately via a soft side penalty.
    
    Args:
        X: (3,) single 3D point
        O: (N, 3) ray origins
        D: (N, 3) ray directions (will be normalized)
        eps: small value for numerical stability
    
    Returns:
        (N,) array of perpendicular distances from X to each ray (infinite line)
    """
    # Normalize directions
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + eps)
    
    # Vector from each origin to point X
    V = X - O  # (N, 3)
    
    # Projection parameter t for each ray (can be negative, that's OK)
    t = np.sum(V * Dn, axis=1)  # (N,)
    
    # Perpendicular component (always non-negative distance)
    perp = V - t[:, None] * Dn  # (N, 3)
    
    # Distance = perpendicular distance (infinite line, no clamp)
    return np.linalg.norm(perp, axis=1)  # (N,)

def closest_distance_rays(ray1: Ray, ray2: Ray) -> float:
    """
    Compute minimal distance between two 3D lines.
    L1: P1 + s1*d1
    L2: P2 + s2*d2
    """
    p1, d1 = ray1.o, ray1.d
    p2, d2 = ray2.o, ray2.d
    
    w0 = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        # Parallel lines
        return np.linalg.norm(np.cross(w0, d1)) / np.sqrt(a)
    
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    
    closest1 = p1 + s * d1
    closest2 = p2 + t * d2
    
    return np.linalg.norm(closest1 - closest2)


def compute_tangent_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct an orthonormal basis (t1, t2) tangent to unit normal n.
    n must be normalized.
    """
    # Pick a helper vector not parallel to n
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
        
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    
    return t1, t2

def update_normal_tangent(n_current: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Update normal using tangent space parameters (alpha, beta).
    Returns normalized new normal.
    """
    t1, t2 = compute_tangent_basis(n_current)
    n_new = n_current + alpha * t1 + beta * t2
    return normalize(n_new)


def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    """
    Convert Rodrigues vector (3,) to rotation matrix (3,3).
    Uses cv2.Rodrigues if available, otherwise manual implementation.
    """
    rvec = np.asarray(rvec).flatten()
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute camera center in world coordinates.
    C = -R^T @ t
    """
    return -R.T @ np.asarray(t).flatten()


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute angle between two vectors in DEGREES.
    """
    v1 = np.asarray(v1).flatten()
    v2 = np.asarray(v2).flatten()
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def optical_axis_world(R: np.ndarray) -> np.ndarray:
    """
    Get camera optical axis (z-axis) in world coordinates.
    z_world = R^T @ [0, 0, 1]
    """
    return R.T @ np.array([0.0, 0.0, 1.0])


def compute_plane_intersection_line(n0: np.ndarray, pt0: np.ndarray,
                                     n1: np.ndarray, pt1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute intersection line of two planes.
    
    Returns:
        line_dir: Unit direction vector of intersection line
        line_pt: A point on the intersection line
    """
    n0 = n0 / (np.linalg.norm(n0) + 1e-12)
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)
    
    # Direction is cross product of normals
    line_dir = np.cross(n0, n1)
    norm_dir = np.linalg.norm(line_dir)
    
    if norm_dir < 1e-9:
        # Planes are parallel, no intersection line
        return np.array([0.0, 1.0, 0.0]), np.zeros(3)
    
    line_dir = line_dir / norm_dir
    
    # Find a point on the intersection line
    # Solve: dot(n0, p - pt0) = 0, dot(n1, p - pt1) = 0
    # Use least squares with constraint that p lies on both planes
    d0 = np.dot(n0, pt0)
    d1 = np.dot(n1, pt1)
    
    A = np.vstack([n0, n1, line_dir])
    b = np.array([d0, d1, 0.0])
    
    try:
        line_pt = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        line_pt = (pt0 + pt1) / 2
    
    return line_dir, line_pt


def build_rotation_align_y_to_dir(target_dir: np.ndarray) -> np.ndarray:
    """
    Build 3x3 rotation matrix that aligns world Y-axis [0,1,0] to target_dir.
    
    Uses Rodrigues formula for rotation about axis perpendicular to both vectors.
    """
    target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-12)
    y_axis = np.array([0.0, 1.0, 0.0])
    
    # If already aligned (or anti-aligned)
    cos_theta = np.dot(y_axis, target_dir)
    if abs(cos_theta) > 0.9999:
        if cos_theta > 0:
            return np.eye(3)
        else:
            # 180 degree rotation about X axis
            return np.diag([1.0, -1.0, -1.0])
    
    # Rotation axis (perpendicular to both)
    axis = np.cross(y_axis, target_dir)
    axis = axis / np.linalg.norm(axis)
    
    # Rotation angle
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    
    return R


def apply_coordinate_rotation(
    R_world: np.ndarray,
    cam_params: dict,
    window_planes: dict,
    points_3d: Optional[list] = None
) -> Tuple[dict, dict, Optional[list]]:
    """
    Apply world coordinate rotation to all calibration data.
    
    Transforms all coordinates so that Y-axis aligns with a specified direction.
    
    Args:
        R_world: 3x3 rotation matrix to apply to world coordinates
        cam_params: Dict of cam_id -> [rvec(3), tvec(3), ...]
        window_planes: Dict of window_id -> {'plane_pt': [...], 'plane_n': [...]}
        points_3d: Optional list of 3D points
        
    Returns:
        (new_cam_params, new_window_planes, new_points_3d)
    """
    import cv2
    
    new_cam_params = {}
    new_window_planes = {}
    
    # Transform camera extrinsics
    # If R_old transforms world->camera, new is: R_new = R_old @ R_world.T
    # t_new = R_world @ t_old (since t is in camera frame pointing to origin)
    for cid, params in cam_params.items():
        params = np.array(params).flatten()
        rvec_old = params[0:3]
        tvec_old = params[3:6]
        
        R_old = cv2.Rodrigues(rvec_old.reshape(3, 1))[0]
        
        # C_world = -R_old.T @ t_old (camera center in world)
        # C_new = R_world @ C_world
        # t_new = -R_new @ C_new = -R_new @ R_world @ (-R_old.T @ t_old)
        
        # Simpler: R_new = R_old @ R_world.T, t_new stays same in new world
        R_new = R_old @ R_world.T
        
        # Camera center transforms: C_new = R_world @ C_old
        C_old = -R_old.T @ tvec_old
        C_new = R_world @ C_old
        
        # t_new = -R_new @ C_new
        t_new = -R_new @ C_new
        
        rvec_new = cv2.Rodrigues(R_new)[0].flatten()
        
        new_params = params.copy()
        new_params[0:3] = rvec_new
        new_params[3:6] = t_new
        new_cam_params[cid] = new_params
    
    # Transform window planes
    for wid, pl in window_planes.items():
        pt_old = np.array(pl['plane_pt'])
        n_old = np.array(pl['plane_n'])
        
        pt_new = R_world @ pt_old
        n_new = R_world @ n_old
        
        new_window_planes[wid] = {
            **pl,
            'plane_pt': pt_new.tolist(),
            'plane_n': n_new.tolist()
        }
    
    # Transform 3D points
    new_points_3d = None
    if points_3d is not None and len(points_3d) > 0:
        pts = np.array(points_3d).reshape(-1, 3)
        pts_new = (R_world @ pts.T).T
        new_points_3d = pts_new.tolist()
    
    return new_cam_params, new_window_planes, new_points_3d


def align_world_y_to_plane_intersection(
    window_planes: dict,
    cam_params: dict,
    points_3d: Optional[list] = None
) -> Tuple[dict, dict, Optional[list], np.ndarray]:
    """
    Align world Y-axis to the intersection line of two window planes.
    
    Returns:
        (new_cam_params, new_window_planes, new_points_3d, R_world)
    """
    wids = list(window_planes.keys())
    
    if len(wids) < 2:
        # Only one plane, no intersection line
        print("[Coordinate Alignment] Only one window plane, skipping Y-axis alignment.")
        return cam_params, window_planes, points_3d, np.eye(3)
    
    # Use first two planes
    pl0 = window_planes[wids[0]]
    pl1 = window_planes[wids[1]]
    
    n0 = np.array(pl0['plane_n'])
    pt0 = np.array(pl0['plane_pt'])
    n1 = np.array(pl1['plane_n'])
    pt1 = np.array(pl1['plane_pt'])
    
    # Compute intersection line
    line_dir, line_pt = compute_plane_intersection_line(n0, pt0, n1, pt1)
    
    # Sign stabilization: ensure line_dir points towards +Y hemisphere
    line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-12)
    if np.dot(line_dir, np.array([0.0, 1.0, 0.0])) < 0:
        line_dir = -line_dir
    
    print(f"[Coordinate Alignment] Intersection line direction: [{line_dir[0]:.4f}, {line_dir[1]:.4f}, {line_dir[2]:.4f}]")
    
    # Build rotation: R_y2dir rotates Y-axis to line_dir
    # We need R_world that rotates line_dir to Y-axis (inverse rotation)
    R_y2dir = build_rotation_align_y_to_dir(line_dir)
    R_world = R_y2dir.T  # Inverse rotation: line_dir -> +Y
    
    # Verification: R_world @ line_dir should be [0, 1, 0]
    line_dir_new = R_world @ line_dir
    print(f"[ALIGN CHECK] R_world @ line_dir = [{line_dir_new[0]:.6f}, {line_dir_new[1]:.6f}, {line_dir_new[2]:.6f}]")
    
    # Apply transformation
    new_cam_params, new_window_planes, new_points_3d = apply_coordinate_rotation(
        R_world, cam_params, window_planes, points_3d
    )
    
    print(f"[Coordinate Alignment] Applied rotation to align Y-axis with plane intersection.")
    
    return new_cam_params, new_window_planes, new_points_3d, R_world

