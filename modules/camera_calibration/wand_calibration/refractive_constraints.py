# -*- coding: utf-8 -*-
"""
Refractive Constraints - Physical Feasibility Utilities

Provides soft penalty functions and diagnostics to enforce physical ordering:
Camera → Refractive Plane → Object

Conventions:
- Plane: (P_plane, n) where n points camera → object
- Signed side: s(P) = dot(n, P - P_plane)
  - Camera side: s(C) < 0
  - Object side: s(X) > 0
- Ray: R(t) = O + t*v where v is unit direction camera → object
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import cv2

def softplus(x):
    return np.log(1 + np.exp(x))


@dataclass
class PlaneOrderConfig:
    """Stage-dependent weights for plane order constraints."""
    w_cam: float = 1.0       # Camera-side penalty weight
    w_obj: float = 1.0       # Object-side penalty weight
    w_order: float = 2.0     # t_point > t_plane penalty weight
    w_plane: float = 1.0     # t_plane > 0 penalty weight
    w_graze: float = 0.5     # Anti-grazing penalty weight
    
    margin_side: float = 1e-3   # mm, margin for s(C) and s(X)
    margin_t: float = 1e-3      # mm, margin for t_plane and t_point order
    den_min: float = 1e-3       # Minimum denominator for valid t_plane
    
    @staticmethod
    def P1():
        """Weak constraints for P1 (plane-only optimization)."""
        return PlaneOrderConfig(
            w_cam=1.0, w_obj=1.0, w_order=2.0, w_plane=1.0, w_graze=0.5
        )
    
    @staticmethod
    def PR4():
        """Medium constraints for PR4 (bundle adjustment)."""
        return PlaneOrderConfig(
            w_cam=10.0, w_obj=10.0, w_order=20.0, w_plane=10.0, w_graze=5.0
        )
    
    @staticmethod
    def PR5():
        """Strong constraints for PR5 (robust BA)."""
        return PlaneOrderConfig(
            w_cam=100.0, w_obj=100.0, w_order=200.0, w_plane=100.0, w_graze=20.0
        )


@dataclass
class PointSideConfig:
    """
    Adaptive point-side constraint configuration.
    
    Enforces triangulated 3D points X lie on object-side of window planes.
    Uses adaptive near-hard constraint with hardening factor.
    
    Constraint logic:
        sX = dot(n, X - plane_pt)  # signed distance (mm)
        v = max(0, eps_side_mm - sX)  # violation amount
        
        Base weight from RayRMSE:
            w_base = (k_side * ray_rmse_mm)^2
            
        Hardening (stronger penalty for larger violations):
            hard = 1 + (v / v0_mm)^p
            
        Residual = sqrt(w_base) * v * hard
    """
    # Margin
    eps_side_mm: float = 0.5      # Safety margin for sX > eps_side_mm
    eps_cam_mm: float = 0.5       # Margin for camera-side check
    
    # Adaptive weight parameters
    ray_rmse_mm: float = 0.1      # Current RayRMSE (updated each PR5 round)
    k_side: float = 20.0          # Strength relative to RayRMSE
    
    # Hardening parameters
    v0_mm: float = 0.2            # Hardening threshold
    p: float = 2.0                # Hardening exponent
    
    # Weight clamps
    w_min: float = 1e-6
    w_max: float = 1e6
    
    # Camera-side weight (fixed, not adaptive)
    w_cam: float = 500.0
    
    @staticmethod
    def P1():
        """Weak constraints for P1."""
        return PointSideConfig(k_side=5.0, w_cam=20.0)
    
    @staticmethod
    def PR4():
        """Medium constraints for PR4."""
        return PointSideConfig(k_side=15.0, w_cam=500.0)
    
    @staticmethod
    def PR5(ray_rmse_mm: float = 0.1):
        """Strong adaptive constraints for PR5."""
        return PointSideConfig(
            ray_rmse_mm=ray_rmse_mm,
            k_side=20.0,
            w_cam=500.0
        )


def compute_point_side_penalty(
    X: np.ndarray,           # Triangulated 3D point (3,) in mm
    plane_pt: np.ndarray,    # Plane reference point (3,) in mm  
    plane_n: np.ndarray,     # Plane unit normal (3,), points camera → object
    cfg: PointSideConfig
) -> float:
    """
    Adaptive near-hard object-side constraint.
    
    sX = dot(n, X - plane_pt)  (mm)
    Enforces: sX >= eps_side_mm
    
    Violation:
        v = max(0, eps_side_mm - sX)
    
    Base weight adapts to current RayRMSE:
        sqrt(w_base) * 1mm ≈ k_side * RayRMSE
        => w_base = (k_side * ray_rmse_mm)^2
    
    Hardening:
        hard = 1 + (v / v0_mm)^p
        
    Returns: sqrt(w_base) * v * hard
    """
    n = plane_n / (np.linalg.norm(plane_n) + 1e-12)
    sX = float(np.dot(n, X - plane_pt))      # mm
    v = max(0.0, cfg.eps_side_mm - sX)       # violation in mm
    
    if v <= 0:
        return 0.0
    
    # Adaptive weight from current geometric noise
    ray_rmse = max(1e-6, float(cfg.ray_rmse_mm))
    k_side = cfg.k_side
    w_base = (k_side * ray_rmse) ** 2
    
    # Safety clamps
    w_base = min(max(w_base, cfg.w_min), cfg.w_max)
    
    # Hardening (violation grows → constraint becomes very stiff)
    hard = 1.0 + (v / cfg.v0_mm) ** cfg.p
    
    return np.sqrt(w_base) * v * hard



def compute_camera_side_penalty(
    C: np.ndarray,           # Camera center (3,) in mm
    plane_pt: np.ndarray,    # Plane reference point (3,) in mm
    plane_n: np.ndarray,     # Plane unit normal (3,), points camera → object
    cfg: PointSideConfig
) -> float:
    """
    Compute hinge penalty for camera on camera-side.
    
    penalty = max(0, sC + eps_cam_mm) where sC = dot(n, C - plane_pt)
    Camera should be on negative side: sC < -eps_cam_mm
    Returns residual term: sqrt(w_cam) * penalty
    """
    plane_n = plane_n / (np.linalg.norm(plane_n) + 1e-12)
    sC = float(np.dot(plane_n, C - plane_pt))
    pC = max(0.0, sC + cfg.eps_cam_mm)  # require sC < -eps_cam_mm
    return np.sqrt(cfg.w_cam) * pC


def compute_soft_barrier_penalty(
    X: np.ndarray,
    plane_pt: np.ndarray,
    plane_n: np.ndarray,
    w_side: float,
    sigma: float = 0.01,
    R_mm: float = 0.0,
    margin_mm: float = 0.02
) -> float:
    """
    Compute smooth softplus barrier penalty for object-side constraint.
    
    sX = dot(n, X - plane_pt)
    gap = (R_mm + margin_mm) - sX
    r = w_side * softplus(gap / sigma)
    
    This is smooth and always differentiable, pushing sX > R + margin.
    """
    n = plane_n / (np.linalg.norm(plane_n) + 1e-12)
    sX = float(np.dot(n, X - plane_pt))
    gap = (R_mm + margin_mm) - sX
    
    # Use stable softplus
    # if gap/sigma is very large, softplus(x) ~= x
    val = gap / sigma
    if val > 50:
        sp = val
    else:
        sp = np.log(1 + np.exp(val))
        
    return w_side * sp, sX


def compute_plane_order_penalties(
    C: np.ndarray,          # Camera center (3,) in mm
    O: np.ndarray,          # Ray origin (3,) in mm
    v: np.ndarray,          # Ray unit direction (3,)
    P_plane: np.ndarray,    # Plane point (3,) in mm
    n: np.ndarray,          # Plane normal (3,), points camera → object
    X: np.ndarray,          # 3D point (3,) in mm
    cfg: PlaneOrderConfig
) -> Tuple[List[float], Dict[str, float]]:
    """
    Compute soft penalty residuals for physical plane ordering.
    
    Returns:
        penalties: List of residual values [r_cam, r_obj, r_plane, r_order, r_graze]
        diagnostics: Dict with raw values for debugging
    """
    # Signed distances
    sC = np.dot(n, C - P_plane)
    sX = np.dot(n, X - P_plane)
    
    # Ray parameters
    t_point = np.dot(X - O, v)
    den = np.dot(v, n)
    
    # B2: Side constraints (always active)
    p_cam = max(0.0, sC + cfg.margin_side)        # require sC < 0
    p_obj = max(0.0, cfg.margin_side - sX)        # require sX > 0
    
    # B3 & B4: Plane order constraints
    t_plane = None
    p_plane = 0.0
    p_order = 0.0
    p_graze = 0.0
    
    if abs(den) > cfg.den_min:
        # Valid denominator - compute plane order
        t_plane = np.dot(P_plane - O, n) / den
        p_plane = max(0.0, cfg.margin_t - t_plane)               # t_plane > 0
        p_order = max(0.0, (t_plane + cfg.margin_t) - t_point)   # t_point > t_plane + margin
    else:
        # Grazing ray - add anti-grazing penalty (B4)
        p_graze = max(0.0, cfg.den_min - abs(den))
    
    # Convert to residuals (sqrt(w) * p for least-squares)
    r_cam = np.sqrt(cfg.w_cam) * p_cam
    r_obj = np.sqrt(cfg.w_obj) * p_obj
    r_plane = np.sqrt(cfg.w_plane) * p_plane
    r_order = np.sqrt(cfg.w_order) * p_order
    r_graze = np.sqrt(cfg.w_graze) * p_graze
    
    penalties = [r_cam, r_obj, r_plane, r_order, r_graze]
    
    diagnostics = {
        'sC': sC,
        'sX': sX,
        't_point': t_point,
        't_plane': t_plane,
        'den': den,
        'p_cam': p_cam,
        'p_obj': p_obj,
        'p_plane': p_plane,
        'p_order': p_order,
        'p_graze': p_graze,
    }
    
    return penalties, diagnostics


def clamp_ray_parameter(t: float, t_min: float = 0.0) -> float:
    """
    Clamp ray parameter to enforce half-line constraint.
    
    Part A1: Rays are half-lines, no backward extension.
    """
    return max(t, t_min)


def clamp_ray_ray_parameters(t1: float, t2: float, t_min: float = 0.0) -> Tuple[float, float]:
    """Clamp both ray-ray intersection parameters."""
    return max(t1, t_min), max(t2, t_min)


def print_plane_side_verification(
    stage_name: str,
    window_planes: Dict[int, Dict],
    cam_params: Dict[int, np.ndarray],
    cam_to_window: Dict[int, int],
    points_3d: np.ndarray,
    rays_data: Optional[Dict] = None  # Optional: for t_negative and order stats
):
    """
    Print plane side verification diagnostics.
    
    Format:
    [STAGE] Plane Side Verification:
      Win w Cam c:
        s(C)=XXX.X mm
        pct_object_side(sX>0)=YY.Y%
        pct_same_side_as_cam=ZZ.Z%
    """
    if points_3d is None or len(points_3d) == 0:
        print(f"\n[{stage_name}] Plane Side Verification: No 3D points available")
        return
    
    pts = np.array(points_3d).reshape(-1, 3)  # N x 3
    
    print(f"\n[{stage_name}] Plane Side Verification:")
    
    for wid, pl in window_planes.items():
        plane_pt = np.array(pl['plane_pt'])
        plane_n = np.array(pl['plane_n'])
        plane_n = plane_n / np.linalg.norm(plane_n)
        
        # Find cameras for this window
        cams_for_win = [cid for cid, w in cam_to_window.items() if w == wid]
        
        for cid in cams_for_win:
            if cid not in cam_params:
                continue
            
            # Get camera center
            cp = cam_params[cid]
            if isinstance(cp, np.ndarray):
                rvec = cp[0:3]
                tvec = cp[3:6]
            else:
                rvec = np.array(cp.get('rvec', cp[:3]))
                tvec = np.array(cp.get('tvec', cp[3:6]))
            
            R = cv2.Rodrigues(rvec.reshape(3, 1))[0]
            C = -R.T @ tvec  # Camera center in world coords (mm)
            
            # Signed distances
            sC = np.dot(plane_n, C - plane_pt)
            sX_arr = np.dot(pts - plane_pt, plane_n)
            
            # Statistics
            pct_object_side = np.mean(sX_arr > 0) * 100
            pct_same_side = np.mean(np.sign(sX_arr) == np.sign(sC)) * 100
            
            # Print with fixed format
            print(f"  Win {wid} Cam {cid}:")
            print(f"    s(C)={sC:.1f} mm")
            print(f"    pct_object_side(sX>0)={pct_object_side:.1f}%")
            print(f"    pct_same_side_as_cam={pct_same_side:.1f}%")
    
    print("")


def check_and_flip_normal_if_needed(
    plane_n: np.ndarray,
    plane_pt: np.ndarray,
    cam_centers: List[np.ndarray]
) -> np.ndarray:
    """
    Ensure cameras are on camera-side (s(C) < 0).
    If mean s(C) is positive, flip the normal.
    
    Part E: Plane orientation stability.
    """
    plane_n = plane_n / np.linalg.norm(plane_n)
    
    sC_values = [np.dot(plane_n, C - plane_pt) for C in cam_centers]
    mean_sC = np.mean(sC_values)
    
    if mean_sC > 0:
        # Flip normal so cameras are on negative side
        return -plane_n
    
    return plane_n
