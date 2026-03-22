import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import traceback

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

# --- Ray Validity Statistics Tracking ---
_RAY_TRACKING_ENABLED = False
_RAY_STATS = {
    "total": 0,
    "valid": 0,
    "invalid": 0,
    "reasons": {}
}

_CAMERA_UPDATE_STATS = {
    "total": 0,
    "success": 0,
    "failed": 0,
    "reasons": {}
}


def reset_camera_update_stats():
    global _CAMERA_UPDATE_STATS
    _CAMERA_UPDATE_STATS = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "reasons": {}
    }


def get_camera_update_stats() -> dict:
    return {
        "total": _CAMERA_UPDATE_STATS["total"],
        "success": _CAMERA_UPDATE_STATS["success"],
        "failed": _CAMERA_UPDATE_STATS["failed"],
        "reasons": dict(_CAMERA_UPDATE_STATS["reasons"]),
    }


def print_camera_update_report(tag: str = "Summary"):
    stats = _CAMERA_UPDATE_STATS
    if stats["total"] == 0:
        return
    print(f"\n[{tag}] Camera Update Path Report:")
    print(f"  Total Calls      : {stats['total']:,}")
    print(f"  Success          : {stats['success']:,}")
    print(f"  Failed           : {stats['failed']:,}")
    if stats["reasons"]:
        print("  Failure Reasons:")
        for r, count in sorted(stats["reasons"].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {r:<30}: {count:,}")


def _record_camera_update(success: bool, reason: Optional[str] = None):
    _CAMERA_UPDATE_STATS["total"] += 1
    if success:
        _CAMERA_UPDATE_STATS["success"] += 1
        return
    _CAMERA_UPDATE_STATS["failed"] += 1
    if reason:
        _CAMERA_UPDATE_STATS["reasons"][reason] = _CAMERA_UPDATE_STATS["reasons"].get(reason, 0) + 1

def enable_ray_tracking(enabled: bool = True, reset: bool = True):
    """Enable or disable global ray statistics tracking."""
    global _RAY_TRACKING_ENABLED
    _RAY_TRACKING_ENABLED = enabled
    if reset:
        reset_ray_stats()

def reset_ray_stats():
    """Reset the global ray statistics counter."""
    global _RAY_STATS
    _RAY_STATS = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "reasons": {}
    }

def get_ray_stats() -> dict:
    """Return a copy of the current ray statistics."""
    return _RAY_STATS.copy()

def print_ray_stats_report(tag: str = "Summary"):
    """Print a concise summary of recorded ray statistics."""
    stats = _RAY_STATS
    if stats["total"] == 0:
        return
        
    print(f"\n[{tag}] Ray Validity Report:")
    print(f"  Total Rays : {stats['total']:,}")
    print(f"  Valid      : {stats['valid']:,} ({stats['valid']/stats['total']*100:.2f}%)")
    print(f"  Invalid    : {stats['invalid']:,} ({stats['invalid']/stats['total']*100:.2f}%)")
    
    if stats["reasons"]:
        print("  Invalid Reasons:")
        for r, count in sorted(stats["reasons"].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {r:<25}: {count:,}")

def build_pinplate_ray_cpp(cam, uv, *, cam_id=-1, window_id=-1, frame_id=-1, endpoint="") -> Ray:
    """
    Build object-side refracted ray using C++ lineOfSight.
    
    Input 'uv' should be PIXEL coordinates.
    The C++ Camera.lineOfSight() handles undistortion and normalization internally.
    """
    global _PROBE_LOGGED, _RAY_ORIGIN_AUDIT_COUNT, _RAY_ORIGIN_AUDIT_PRINTED
    import pyopenlpt as lpt
    
    u, v = float(uv[0]), float(uv[1])

    def _line_of_sight(c, pt2):
        if not hasattr(c, "lineOfSightStatus"):
            raise RuntimeError("lineOfSightStatus is required in hard migration mode")
        ok, line, err = c.lineOfSightStatus(pt2)
        if not ok:
            raise RuntimeError(f"lineOfSightStatus failed: {err}")
        return line
    
    res = None
    try:
        # Use lineOfSight which accepts pixel coordinates directly
        # It calls undistort() internally, then passes to pinplateLine()
        line = _line_of_sight(cam, lpt.Pt2D(u, v))
        o, d = _extract_line3d(line)
        
        if np.all(np.isfinite(o)) and np.all(np.isfinite(d)):
            if not _PROBE_LOGGED:
                _PROBE_LOGGED = True
            
            d_norm = normalize(d)
            
            res = Ray(
                o=o,
                d=d_norm,
                valid=True,
                cam_id=cam_id,
                window_id=window_id,
                frame_id=frame_id,
                endpoint=endpoint,
                uv=(u, v),
            )
        else:
            res = Ray(o=np.zeros(3), d=np.array([0, 0, 1.0]), valid=False, reason="non_finite_outputs", 
                       cam_id=cam_id, window_id=window_id, frame_id=frame_id, endpoint=endpoint, uv=(u, v))
            
    except Exception as e:
        reason = f"C++ lineOfSight failed: {str(e)}"
        if "total internal reflection" in reason.lower():
             reason = "total_internal_reflection"
             
        res = Ray(o=np.zeros(3), d=np.array([0, 0, 1.0]), valid=False, reason=reason, 
                   cam_id=cam_id, window_id=window_id, frame_id=frame_id, endpoint=endpoint, uv=(u, v))
        
    # --- Tracking Logic ---
    if _RAY_TRACKING_ENABLED and res is not None:
        _RAY_STATS["total"] += 1
        if res.valid:
            _RAY_STATS["valid"] += 1
        else:
            _RAY_STATS["invalid"] += 1
            r = res.reason or "unknown"
            # Strip numeric details from reason for cleaner categorical grouping
            if "=" in r: r = r.split("=")[0]
            _RAY_STATS["reasons"][r] = _RAY_STATS["reasons"].get(r, 0) + 1
            
    return res
    
    # Fallback
    return Ray(o=np.zeros(3), d=np.array([0, 0, 1.0]), valid=False, reason="extraction_failed", 
               cam_id=cam_id, window_id=window_id, frame_id=frame_id, endpoint=endpoint, uv=(u, v))


def build_pinplate_rays_cpp_batch(cam, uv_list, *, meta_list=None) -> list:
    """
    Build object-side refracted rays in batch using C++ lineOfSightBatchStatus.

    Args:
        cam: pyopenlpt Camera object
        uv_list: list of (u, v) pixel coordinates
        meta_list: optional list of dicts with keys cam_id/window_id/frame_id/endpoint

    Returns:
        list[Ray] in the same order as uv_list
    """
    global _PROBE_LOGGED
    import pyopenlpt as lpt

    if not uv_list:
        return []

    if meta_list is None:
        meta_list = [{} for _ in uv_list]

    # Fallback path for old pyopenlpt builds.
    if not hasattr(cam, "lineOfSightBatchStatus"):
        out = []
        for uv, meta in zip(uv_list, meta_list):
            out.append(
                build_pinplate_ray_cpp(
                    cam,
                    uv,
                    cam_id=int(meta.get("cam_id", -1)),
                    window_id=int(meta.get("window_id", -1)),
                    frame_id=int(meta.get("frame_id", -1)),
                    endpoint=str(meta.get("endpoint", "")),
                )
            )
        return out

    pt2_list = [lpt.Pt2D(float(uv[0]), float(uv[1])) for uv in uv_list]
    try:
        batch = cam.lineOfSightBatchStatus(pt2_list)
    except Exception as e:
        uv0 = uv_list[0] if uv_list else None
        uv1 = uv_list[-1] if uv_list else None
        print(
            "[CRASH-LOC][lineOfSightBatchStatus] "
            f"cam={getattr(cam, '_cam_id', 'unknown')} n={len(pt2_list)} "
            f"uv_first={uv0} uv_last={uv1} err={repr(e)}",
            flush=True,
        )
        print(traceback.format_exc(), flush=True)
        raise

    rays = []
    for idx, (ok, line, err) in enumerate(batch):
        uv = uv_list[idx]
        u, v = float(uv[0]), float(uv[1])
        meta = meta_list[idx]
        cam_id = int(meta.get("cam_id", -1))
        window_id = int(meta.get("window_id", -1))
        frame_id = int(meta.get("frame_id", -1))
        endpoint = str(meta.get("endpoint", ""))

        res = None
        try:
            if not ok:
                raise RuntimeError(err)

            o, d = _extract_line3d(line)
            if np.all(np.isfinite(o)) and np.all(np.isfinite(d)):
                d_norm = normalize(d)
                if not _PROBE_LOGGED:
                    _PROBE_LOGGED = True

                res = Ray(
                    o=o,
                    d=d_norm,
                    valid=True,
                    cam_id=cam_id,
                    window_id=window_id,
                    frame_id=frame_id,
                    endpoint=endpoint,
                    uv=(u, v),
                )
            else:
                res = Ray(
                    o=np.zeros(3),
                    d=np.array([0, 0, 1.0]),
                    valid=False,
                    reason="non_finite_outputs",
                    cam_id=cam_id,
                    window_id=window_id,
                    frame_id=frame_id,
                    endpoint=endpoint,
                    uv=(u, v),
                )
        except Exception as e:
            reason = f"C++ lineOfSight failed: {str(e)}"
            if "total internal reflection" in reason.lower():
                reason = "total_internal_reflection"
            res = Ray(
                o=np.zeros(3),
                d=np.array([0, 0, 1.0]),
                valid=False,
                reason=reason,
                cam_id=cam_id,
                window_id=window_id,
                frame_id=frame_id,
                endpoint=endpoint,
                uv=(u, v),
            )

        if _RAY_TRACKING_ENABLED and res is not None:
            _RAY_STATS["total"] += 1
            if res.valid:
                _RAY_STATS["valid"] += 1
            else:
                _RAY_STATS["invalid"] += 1
                r = res.reason or "unknown"
                if "=" in r:
                    r = r.split("=")[0]
                _RAY_STATS["reasons"][r] = _RAY_STATS["reasons"].get(r, 0) + 1

        rays.append(res)

    return rays

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
    points_3d: Optional[list] = None,
    t_shift: Optional[np.ndarray] = None
) -> Tuple[dict, dict, Optional[list]]:
    """
    Apply world coordinate rotation AND translation to all calibration data.
    
    Transforms all coordinates: P_new = R_world @ (P_old + t_shift)
    
    Args:
        R_world: 3x3 rotation matrix
        cam_params: Dict of cam_id -> [rvec(3), tvec(3), ...]
        window_planes: Dict of window_id -> {'plane_pt': [...], 'plane_n': [...]}
        points_3d: Optional list of 3D points
        t_shift: Optional (3,) vector to shift world origin BEFORE rotation.
        
    Returns:
        (new_cam_params, new_window_planes, new_points_3d)
    """
    import cv2
    
    if t_shift is None:
        t_shift = np.zeros(3)
    else:
        t_shift = np.array(t_shift).flatten()
    
    new_cam_params = {}
    new_window_planes = {}
    
    # Transform camera extrinsics
    for cid, params in cam_params.items():
        params = np.array(params).flatten()
        rvec_old = params[0:3]
        tvec_old = params[3:6]
        
        R_old = cv2.Rodrigues(rvec_old.reshape(3, 1))[0]
        
        # C_world = -R_old.T @ t_old (camera center in world)
        # C_new = R_world @ (C_world + t_shift)
        
        # Rotations just compose: R_new = R_old @ R_world.T
        R_new = R_old @ R_world.T
        
        # Camera center transforms
        C_old = -R_old.T @ tvec_old
        C_new = R_world @ (C_old + t_shift)
        
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
        
        # Plane point shifts and rotates
        pt_new = R_world @ (pt_old + t_shift)
        # Normal vector only rotates
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
        # Shift then Rotate
        pts_shifted = pts + t_shift
        pts_new = (R_world @ pts_shifted.T).T
        new_points_3d = pts_new.tolist()
    
    return new_cam_params, new_window_planes, new_points_3d


def align_world_to_axis_directions(
    axis_direction_map,
    triangulate_fn,
    cam_params,
    points_3d,
    window_planes=None,
    validate_coverage=True,
):
    """
    Compute and apply rigid world-coordinate alignment to user-detected axis directions.

    Triangulates 4 landmarks (center, +X, +Y, +Z), computes closest orthonormal basis
    via SVD polar decomposition, and applies a rigid transform.

    Args:
        axis_direction_map: {cam_idx: {"center":[x,y], "+X":[x,y], "+Y":[x,y], "+Z":[x,y]}}
        triangulate_fn: callable(obs_by_landmark) -> {"center": arr(3), "+X": arr(3), "+Y": arr(3), "+Z": arr(3)}
            obs_by_landmark format: {landmark_name: {cam_idx: [px, py]}}
        cam_params: camera parameters (passed to apply_coordinate_rotation)
        points_3d: (N,3) array of 3D wand points to transform
        window_planes: optional refraction planes dict
        validate_coverage: if True, require 2+ cameras per landmark

    Returns:
        (True, R_world, t_shift, transformed_state_dict) on success
        (False, None, None, None) on any failure
    """
    import logging

    logger = logging.getLogger(__name__)

    if window_planes is None:
        window_planes = {}

    landmark_names = ["center", "+X", "+Y", "+Z"]

    # Reorganize axis_direction_map into per-landmark observation dict
    obs_by_landmark = {lm: {} for lm in landmark_names}
    for cam_idx, cam_data in axis_direction_map.items():
        for lm in landmark_names:
            if lm in cam_data:
                obs_by_landmark[lm][cam_idx] = cam_data[lm]

    # Validate coverage (need 2+ cameras per landmark)
    if validate_coverage:
        for lm in landmark_names:
            n_cams = len(obs_by_landmark[lm])
            if n_cams < 2:
                logger.warning(
                    "Axis alignment failed: landmark '%s' has only %d camera obs (need 2+)", lm, n_cams
                )
                return False, None, None, None

    # Triangulate landmarks
    try:
        landmark_3d = triangulate_fn(obs_by_landmark)
    except Exception as exc:
        logger.warning("Axis alignment failed during triangulation: %s", exc)
        return False, None, None, None

    # Validate triangulated results
    for lm in landmark_names:
        if lm not in landmark_3d or landmark_3d[lm] is None:
            logger.warning("Axis alignment failed: triangulation returned None for '%s'", lm)
            return False, None, None, None
        pt = np.asarray(landmark_3d[lm]).reshape(3)
        if not np.isfinite(pt).all():
            logger.warning("Axis alignment failed: non-finite point for landmark '%s'", lm)
            return False, None, None, None

    center = np.asarray(landmark_3d["center"]).reshape(3)
    pt_x = np.asarray(landmark_3d["+X"]).reshape(3)
    pt_y = np.asarray(landmark_3d["+Y"]).reshape(3)
    pt_z = np.asarray(landmark_3d["+Z"]).reshape(3)

    # Compute direction vectors
    dir_X = pt_x - center
    dir_Y = pt_y - center
    dir_Z = pt_z - center

    norms = np.array([np.linalg.norm(d) for d in [dir_X, dir_Y, dir_Z]])
    if np.any(norms < 1e-10):
        logger.warning("Axis alignment failed: degenerate direction vectors (near-zero norm: %s)", norms)
        return False, None, None, None

    dir_X /= norms[0]
    dir_Y /= norms[1]
    dir_Z /= norms[2]

    # Build orientation matrix M with direction vectors as columns
    M = np.column_stack([dir_X, dir_Y, dir_Z])

    # SVD polar decomposition: R_new = U @ Vt is closest orthonormal matrix to M
    U, S, Vt = np.linalg.svd(M)
    logger.debug("SVD singular values: %s, condition: %.4e", S, S[0] / max(S[-1], 1e-20))

    if S[-1] < 1e-12 or (S[0] / max(S[-1], 1e-20)) > 1e12:
        logger.warning("Axis alignment failed: SVD ill-conditioned (S=%s)", S)
        return False, None, None, None

    R_new = U @ Vt

    # Ensure right-handed system (det = +1)
    det = np.linalg.det(R_new)
    logger.debug("det(R_new) = %.8f", det)
    if det < 0:
        U_fix = U.copy()
        U_fix[:, -1] *= -1
        R_new = U_fix @ Vt
        det = np.linalg.det(R_new)

    if abs(det - 1.0) > 1e-6:
        logger.warning("Axis alignment failed: det(R)=%.8f (expected +1)", det)
        return False, None, None, None

    # Translation: shift world origin to axis center
    t_shift = -center

    # Apply rigid transform
    logger.debug("Applying axis alignment: |t_shift|=%.4f mm", np.linalg.norm(t_shift))
    new_cam_params, new_window_planes, new_points_3d = apply_coordinate_rotation(
        R_new, cam_params, window_planes, points_3d, t_shift=t_shift
    )

    transformed_state = {
        "cam_params": new_cam_params,
        "window_planes": new_window_planes,
        "points_3d": new_points_3d,
    }

    return True, R_new, t_shift, transformed_state


def align_world_y_to_plane_intersection(
    window_planes: dict,
    cam_params: dict,
    points_3d: Optional[list] = None,
    align_mode: str = "yz",
) -> Tuple[dict, dict, Optional[list], np.ndarray, np.ndarray]:
    """
    Align world axes using window plane geometry.

    - Two-window mode: one axis aligns to the plane intersection line and one axis
      aligns to first plane normal, controlled by `align_mode`:
        * "yz": line->Y, normal->Z (default)
        * "xz": line->X, normal->Z
        * "xy": line->X, normal->Y
    - Single-window mode: plane normal aligns to normal-axis from `align_mode`
      (2nd char), and in-plane axis is chosen from projected line-axis
      (1st char) with robust fallback.
    
    Returns:
        (new_cam_params, new_window_planes, new_points_3d, R_world, t_shift)
    """
    mode = str(align_mode).lower()
    if mode not in ("yz", "xz", "xy"):
        mode = "yz"

    axis_line = mode[0]
    axis_normal = mode[1]

    axis_basis = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0]),
    }

    def _build_basis_from_axes(ax_map: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = ax_map.get('x')
        y = ax_map.get('y')
        z = ax_map.get('z')

        if x is not None and y is not None:
            z = np.cross(x, y)
            z = z / (np.linalg.norm(z) + 1e-12)
            y = np.cross(z, x)
            y = y / (np.linalg.norm(y) + 1e-12)
            x = np.cross(y, z)
            x = x / (np.linalg.norm(x) + 1e-12)
        elif y is not None and z is not None:
            x = np.cross(y, z)
            x = x / (np.linalg.norm(x) + 1e-12)
            z = np.cross(x, y)
            z = z / (np.linalg.norm(z) + 1e-12)
            y = np.cross(z, x)
            y = y / (np.linalg.norm(y) + 1e-12)
        elif x is not None and z is not None:
            y = np.cross(z, x)
            y = y / (np.linalg.norm(y) + 1e-12)
            x = np.cross(y, z)
            x = x / (np.linalg.norm(x) + 1e-12)
            z = np.cross(x, y)
            z = z / (np.linalg.norm(z) + 1e-12)
        else:
            x = np.array([1.0, 0.0, 0.0])
            y = np.array([0.0, 1.0, 0.0])
            z = np.array([0.0, 0.0, 1.0])

        return x, y, z

    wids = list(window_planes.keys())
    if len(wids) < 1:
        print("[Coordinate Alignment] No window planes, skipping alignment.")
        return cam_params, window_planes, points_3d, np.eye(3), np.zeros(3)

    pl0 = window_planes[wids[0]]
    n0 = np.array(pl0['plane_n'], dtype=float)
    n0 = n0 / (np.linalg.norm(n0) + 1e-12)

    if n0[2] < 0:
        n0 = -n0

    if len(wids) >= 2:
        pl1 = window_planes[wids[1]]
        n1 = np.array(pl1['plane_n'])
        pt0 = np.array(pl0['plane_pt'])
        pt1 = np.array(pl1['plane_pt'])

        # Compute intersection line
        line_dir, _ = compute_plane_intersection_line(n0, pt0, n1, pt1)

        # Sign stabilization: ensure line_dir points towards +Y hemisphere
        line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-12)
        if np.dot(line_dir, np.array([0.0, 1.0, 0.0])) < 0:
            line_dir = -line_dir

        print(f"[Coordinate Alignment] Intersection line direction: [{line_dir[0]:.4f}, {line_dir[1]:.4f}, {line_dir[2]:.4f}]")

        # Two-plane mode
        line_vec = line_dir
        normal_vec = n0 - np.dot(n0, line_vec) * line_vec
        normal_vec = normal_vec / (np.linalg.norm(normal_vec) + 1e-12)

        ax_map = {
            axis_line: line_vec,
            axis_normal: normal_vec,
        }
        x_new, y_new, z_new = _build_basis_from_axes(ax_map)

        print(f"[Coordinate Alignment] Two-window mode ({mode}): {axis_line.upper()}<-intersection, {axis_normal.upper()}<-plane0 normal")
    else:
        # Single-plane mode
        normal_vec = n0
        preferred = axis_basis[axis_line]
        in_plane = preferred - np.dot(preferred, normal_vec) * normal_vec
        if np.linalg.norm(in_plane) < 1e-8:
            for k in ('x', 'y', 'z'):
                if k == axis_normal:
                    continue
                cand = axis_basis[k]
                in_plane = cand - np.dot(cand, normal_vec) * normal_vec
                if np.linalg.norm(in_plane) >= 1e-8:
                    break
        in_plane = in_plane / (np.linalg.norm(in_plane) + 1e-12)

        ax_map = {
            axis_line: in_plane,
            axis_normal: normal_vec,
        }
        x_new, y_new, z_new = _build_basis_from_axes(ax_map)

        print(f"[Coordinate Alignment] Single-window mode ({mode}): {axis_normal.upper()}<-plane normal")

    # Construct Rotation Matrix R_world (Old -> New)
    # Rows are the new basis vectors expressed in old frame
    R_world = np.vstack([x_new, y_new, z_new])

    # Verification
    n0_new = R_world @ n0
    print(f"[ALIGN CHECK] Z-Axis (Plane 0 Normal): {n0_new}")
    if len(wids) >= 2:
        line_dir_new = R_world @ line_dir
        print(f"[ALIGN CHECK] Y-Axis (Intersection): {line_dir_new}")
    
    # [USER REQUEST] Translation: Center World Origin at 3D Cloud Centroid
    t_shift = np.zeros(3)
    if points_3d is not None and len(points_3d) > 0:
        pts = np.array(points_3d).reshape(-1, 3)
        centroid = np.mean(pts, axis=0)
        t_shift = -centroid
        print(f"[Coordinate Alignment] Centering World at Cloud Centroid: {centroid}")
    else:
        print("[Coordinate Alignment] No 3D points provided, skipping Centering (Translation = 0).")
    
    # Apply transformation
    new_cam_params, new_window_planes, new_points_3d = apply_coordinate_rotation(
        R_world, cam_params, window_planes, points_3d, t_shift=t_shift
    )
    
    print(f"[Coordinate Alignment] Applied rotation and translation to align coordinates.")
    
    return new_cam_params, new_window_planes, new_points_3d, R_world, t_shift


# ---------------------------------------------------------------------------
# Shared native-safety validation
# ---------------------------------------------------------------------------

_MIN_SIGNED_DEPTH_MM = 1.0


class NativeSafetyValidationError(RuntimeError):
    """Raised when Python-side validation rejects state before native C++ update."""


def validate_native_safety(
    image_size: Optional[Tuple[int, int]] = None,
    intrinsics: Optional[dict] = None,
    extrinsics: Optional[dict] = None,
    plane_geom: Optional[dict] = None,
    media_props: Optional[dict] = None,
    *,
    min_signed_depth_mm: float = _MIN_SIGNED_DEPTH_MM,
) -> List[str]:
    """Validate camera/plane/media state before entering C++ native code.

    Returns a list of machine-sortable error strings.  An empty list means the
    state is safe.  Every tag has the form ``<category>:<detail>`` so that logs
    can be sorted and filtered programmatically.

    Checks performed:
      1. image_size positive and finite
      2. intrinsics fx,fy > 0 and finite; cx,cy finite; dist coeffs finite
      3. extrinsics rvec/tvec length-3 and all-finite
      4. plane_geom pt/n finite, ||n|| > eps, normalised for subsequent depth check
      5. media refractive indices finite and > 0; thickness finite and > 0
      6. camera-to-plane signed depth >= min_signed_depth_mm
      7. global finiteness guard on all float conversions
    """
    errors: List[str] = []

    # -- 1. image size --
    if image_size is not None:
        try:
            nr, nc = int(image_size[0]), int(image_size[1])
            if nr <= 0 or nc <= 0:
                errors.append(f"image_size:non_positive:{nr}x{nc}")
            if not (np.isfinite(nr) and np.isfinite(nc)):
                errors.append(f"image_size:non_finite:{nr}x{nc}")
        except (TypeError, ValueError, IndexError) as exc:
            errors.append(f"image_size:conversion_error:{exc}")

    # -- 2. intrinsics --
    intr = intrinsics or {}
    try:
        fx = float(intr.get('f', 1.0))
        fy = fx  # same source key in current codebase
        cx = float(intr.get('cx', 0.0))
        cy = float(intr.get('cy', 0.0))
        if not np.isfinite(fx) or fx <= 0:
            errors.append(f"intrinsics:fx_invalid:{fx}")
        if not np.isfinite(fy) or fy <= 0:
            errors.append(f"intrinsics:fy_invalid:{fy}")
        if not np.isfinite(cx):
            errors.append(f"intrinsics:cx_non_finite:{cx}")
        if not np.isfinite(cy):
            errors.append(f"intrinsics:cy_non_finite:{cy}")
        dist_raw = intr.get('dist', None)
        if dist_raw is not None:
            dist_arr = np.asarray(dist_raw, dtype=np.float64).ravel()
            if not np.all(np.isfinite(dist_arr)):
                errors.append(f"intrinsics:dist_non_finite:{dist_arr.tolist()}")
    except (TypeError, ValueError) as exc:
        errors.append(f"intrinsics:conversion_error:{exc}")

    # -- 3. extrinsics --
    ext = extrinsics or {}
    try:
        rvec_np = np.asarray(ext.get('rvec', [0.0, 0.0, 0.0]), dtype=np.float64).flatten()
        tvec_np = np.asarray(ext.get('tvec', [0.0, 0.0, 0.0]), dtype=np.float64).flatten()
        if rvec_np.shape != (3,):
            errors.append(f"extrinsics:rvec_wrong_length:{rvec_np.shape}")
        elif not np.all(np.isfinite(rvec_np)):
            errors.append(f"extrinsics:rvec_non_finite:{rvec_np.tolist()}")
        if tvec_np.shape != (3,):
            errors.append(f"extrinsics:tvec_wrong_length:{tvec_np.shape}")
        elif not np.all(np.isfinite(tvec_np)):
            errors.append(f"extrinsics:tvec_non_finite:{tvec_np.tolist()}")
    except (TypeError, ValueError) as exc:
        errors.append(f"extrinsics:conversion_error:{exc}")

    # -- 4. plane geometry --
    plane_pt_np: Optional[np.ndarray] = None
    plane_normal: Optional[np.ndarray] = None
    plane_normal_unit: Optional[np.ndarray] = None
    thickness_for_cpp: Optional[float] = None
    if plane_geom is not None:
        try:
            plane_pt_np = np.asarray(plane_geom.get('pt', [0.0, 0.0, 0.0]), dtype=np.float64).flatten()
            plane_normal = np.asarray(plane_geom.get('n', [0.0, 0.0, 1.0]), dtype=np.float64).flatten()
            if plane_pt_np.shape != (3,):
                errors.append(f"plane_geom:pt_wrong_length:{plane_pt_np.shape}")
            elif not np.all(np.isfinite(plane_pt_np)):
                errors.append(f"plane_geom:pt_non_finite:{plane_pt_np.tolist()}")
            if plane_normal.shape != (3,):
                errors.append(f"plane_geom:n_wrong_length:{plane_normal.shape}")
            else:
                n_norm = float(np.linalg.norm(plane_normal))
                if not np.isfinite(n_norm) or n_norm < 1e-9:
                    errors.append(f"plane_geom:n_degenerate:norm={n_norm}")
                elif not np.all(np.isfinite(plane_normal)):
                    errors.append(f"plane_geom:n_non_finite:{plane_normal.tolist()}")
                else:
                    plane_normal_unit = plane_normal / n_norm
        except (TypeError, ValueError) as exc:
            errors.append(f"plane_geom:conversion_error:{exc}")

    # -- 5. media properties --
    med = media_props or {}
    try:
        thickness = float(med.get('thickness', 10.0))
        thickness_for_cpp = thickness
        if not np.isfinite(thickness) or thickness <= 0:
            errors.append(f"media:thickness_invalid:{thickness}")
        n1 = float(med.get('n1', med.get('n_air', 1.0)))
        n2 = float(med.get('n2', med.get('n_window', med.get('n_glass', 1.49))))
        n3 = float(med.get('n3', med.get('n_object', med.get('n_medium', 1.33))))
        for label, val in [('n1', n1), ('n2', n2), ('n3', n3)]:
            if not np.isfinite(val) or val <= 0:
                errors.append(f"media:{label}_invalid:{val}")
    except (TypeError, ValueError) as exc:
        errors.append(f"media:conversion_error:{exc}")

    # -- 6. camera-to-plane signed depth --
    if (plane_pt_np is not None and plane_normal_unit is not None
            and 'rvec' in ext and 'tvec' in ext
            and len(errors) == 0):  # only if no prior errors on these inputs
        try:
            R = rodrigues_to_R(rvec_np)
            C = camera_center(R, tvec_np)
            signed_depth = float(np.dot(plane_pt_np - C, plane_normal_unit))
            if not np.isfinite(signed_depth) or signed_depth < min_signed_depth_mm:
                errors.append(
                    f"depth:plane_too_close_to_camera:{signed_depth:.4f}<{min_signed_depth_mm}"
                )
            if thickness_for_cpp is not None:
                p_farthest = plane_pt_np + plane_normal_unit * thickness_for_cpp
                if not np.all(np.isfinite(p_farthest)):
                    errors.append(f"plane_geom:p_farthest_non_finite:{p_farthest.tolist()}")
                farthest_depth = float(np.dot(p_farthest - C, plane_normal_unit))
                if not np.isfinite(farthest_depth) or farthest_depth < min_signed_depth_mm:
                    errors.append(
                        f"depth:farthest_plane_too_close_to_camera:{farthest_depth:.4f}<{min_signed_depth_mm}"
                    )
        except Exception as exc:
            errors.append(f"depth:computation_error:{exc}")

    return errors

def update_cpp_camera_state(cam_obj, 
                            extrinsics: Optional[dict] = None,
                            intrinsics: Optional[dict] = None,
                            plane_geom: Optional[dict] = None,
                            media_props: Optional[dict] = None,
                            image_size: Optional[Tuple[int, int]] = None,
                            solver_opts: Optional[dict] = None,
                            validate_native: bool = True) -> None:
    """
    Unified function to update C++ Camera parameters (PinPlate model).
    
    Handles:
    1. Intrinsics (K matrix, distortion)
    2. Extrinsics (R, T, and their inverses)
    3. Plane Geometry (Coordinate Shift: Closest -> Farthest)
    4. Refractive Media Properties
    5. Internal State Refresh (updatePt3dClosest)
    
    Args:
        cam_obj: lpt.Camera instance
        extrinsics: Dict with keys 'rvec' (3,) and 'tvec' (3,)
        intrinsics: Dict with optional keys 'f', 'cx', 'cy', 'dist'
        plane_geom: Dict with keys 'pt', 'n' (point and normal at Closest Interface)
        media_props: Dict with keys 'thickness' (and optional 'n1','n2','n3' or 'n_air'...)
    """
    try:
        import pyopenlpt as lpt
    except ImportError:
        return # Cannot update if lpt not available

    required_methods = (
        'setPinplateImageSize',
        'setPinplateIntrinsics',
        'setPinplateExtrinsics',
        'setPinplateRefraction',
        'commitPinplateUpdate',
    )
    missing = [name for name in required_methods if not hasattr(cam_obj, name)]
    if missing:
        _record_camera_update(success=False, reason="missing_grouped_api")
        raise RuntimeError("Hard migration mode requires grouped camera API: missing " + ", ".join(missing))

    # ---- Pre-flight native-safety validation ----
    if validate_native:
        safety_errors = validate_native_safety(
            image_size=image_size,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            plane_geom=plane_geom,
            media_props=media_props,
        )
        if safety_errors:
            _record_camera_update(success=False, reason="native_safety_rejected")
            raise NativeSafetyValidationError(
                "update_cpp_camera_state blocked by native-safety validation: "
                + "; ".join(safety_errors)
            )

    try:
        if image_size is not None:
            n_row, n_col = int(image_size[0]), int(image_size[1])
        else:
            try:
                n_row = int(cam_obj.getNRow())
                n_col = int(cam_obj.getNCol())
            except Exception:
                n_row, n_col = 800, 1280
            if n_row <= 0 or n_col <= 0:
                n_row, n_col = 800, 1280

        if intrinsics is None:
            intrinsics = {}
        fx = float(intrinsics.get('f', 1.0))
        fy = float(intrinsics.get('f', 1.0))
        cx = float(intrinsics.get('cx', 0.0))
        cy = float(intrinsics.get('cy', 0.0))
        if ('dist' in intrinsics) and (intrinsics['dist'] is not None):
            dist_coeff = [float(x) for x in intrinsics['dist']]
        else:
            dist_coeff = [0.0, 0.0, 0.0, 0.0, 0.0]

        if extrinsics is None:
            extrinsics = {}
        rvec_np = np.asarray(extrinsics.get('rvec', [0.0, 0.0, 0.0]), dtype=float).flatten()
        tvec_np = np.asarray(extrinsics.get('tvec', [0.0, 0.0, 0.0]), dtype=float).flatten()
        rvec = lpt.Pt3D(float(rvec_np[0]), float(rvec_np[1]), float(rvec_np[2]))
        tvec = lpt.Pt3D(float(tvec_np[0]), float(tvec_np[1]), float(tvec_np[2]))

        if media_props is None:
            media_props = {}
        current_thick = float(media_props.get('thickness', 10.0))
        n1 = media_props.get('n1', media_props.get('n_air', 1.0))
        n2 = media_props.get('n2', media_props.get('n_window', media_props.get('n_glass', 1.49)))
        n3 = media_props.get('n3', media_props.get('n_object', media_props.get('n_medium', 1.33)))
        refract_array = [float(n3), float(n2), float(n1)]
        w_array = [current_thick]

        proj_tol = 1e-6
        proj_nmax = 1000
        lr = 0.1
        if solver_opts:
            proj_tol = float(solver_opts.get('proj_tol', proj_tol))
            proj_nmax = int(solver_opts.get('proj_nmax', proj_nmax))
            lr = float(solver_opts.get('lr', lr))

        if plane_geom is None:
            plane_pt_np = np.array([0.0, 0.0, 0.0], dtype=float)
            plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            plane_pt_np = np.asarray(plane_geom.get('pt', [0.0, 0.0, 0.0]), dtype=float).flatten()
            plane_normal = np.asarray(plane_geom.get('n', [0.0, 0.0, 1.0]), dtype=float).flatten()
        plane_normal = normalize(plane_normal)

        # Python-side plane point is at the closest interface; C++ pinplate expects farthest interface.
        p_farthest = plane_pt_np + plane_normal * current_thick
        plane_pt = lpt.Pt3D(float(p_farthest[0]), float(p_farthest[1]), float(p_farthest[2]))
        plane_n = lpt.Pt3D(float(plane_normal[0]), float(plane_normal[1]), float(plane_normal[2]))

        cam_obj.setPinplateImageSize(n_row, n_col)
        cam_obj.setPinplateIntrinsics(fx, fy, cx, cy, dist_coeff)
        cam_obj.setPinplateExtrinsics(rvec, tvec)
        cam_obj.setPinplateRefraction(plane_pt, plane_n, refract_array, w_array,
                                      proj_tol, proj_nmax, lr)
        cam_obj.commitPinplateUpdate(bool(getattr(cam_obj, '_is_active', True)),
                                     float(getattr(cam_obj, '_max_intensity', 255.0)))
        _record_camera_update(success=True)
    except Exception as exc:
        _record_camera_update(success=False, reason="grouped_update_failed")
        raise RuntimeError(f"update_cpp_camera_state failed in hard migration mode: {exc}") from exc
