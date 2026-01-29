"""
Refractive PR4 Bundle Adjustment Optimizer

This module implements Phase PR4 optimization: Selective Bundle Adjustment
that refines both window planes AND selected camera extrinsics (when observable).

Key Design Principles:
- Observability-based freezing: N_cam, baseline, view-angle diversity
- Staged optimization: PR4.1 → PR4.2 → PR4.3
- Intrinsics ALWAYS fixed
- Safe defaults: freeze rvec unless geometry is strong

Freeze Semantics (OPTIMIZE = move, FREEZE = fixed):
- N_cam = 1: Plane OPTIMIZE (strong reg), tvec FREEZE, rvec FREEZE
- N_cam ≥ 2, weak: Plane OPTIMIZE, tvec OPTIMIZE (small TR), rvec FREEZE
- N_cam ≥ 2, good: Plane OPTIMIZE, tvec OPTIMIZE, rvec OPTIMIZE

Rotation Thresholds:
- theta_enable_rot = 20°: Below → FREEZE rvec
- theta_strong_rot = 35°: Above → full rvec, weaker reg
- Baseline guard: If baseline < 10mm, keep rvec heavily damped
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import cv2
from scipy.optimize import least_squares
import json
import os
from pathlib import Path



try:
    import pyopenlpt as lpt
except ImportError:
    lpt = None

from .refractive_geometry import (
    Ray, normalize, build_pinplate_ray_cpp, triangulate_point, point_to_ray_dist,
    point_to_ray_dist_vec,  # Vectorized version for performance
    update_normal_tangent, rodrigues_to_R, camera_center, angle_between_vectors,
    optical_axis_world, update_cpp_camera_state,
    enable_ray_tracking, reset_ray_stats, print_ray_stats_report
)
from datetime import datetime
from pathlib import Path
import os

def evaluate_plane_side_constraints(dataset, window_planes, cam_params, cams_cpp, cam_to_window, R_small_mm, R_large_mm, margin_mm):
    """
    Unified feasibility evaluator for PR5 / Polish.
    Returns: min_sX, violations_list, worst_viol, frame_details, min_dist
    """
    # Requires re-tracing logic. Can we reuse a simplified trace loop?
    # Yes, we need a simplified version of _trace_rays_all but standalone.
    # Or we can make this a method of a base class?
    # Standalone is cleaner but duplicates trace logic.
    # Let's rely on cams_cpp being UP-TO-DATE with cam_params/planes.
    # CALLER MUST ENSURE cams_cpp IS SYNCED.
    
    min_sX = 999.0
    min_dist = 999.0
    violations = []
    worst_viol = None
    frame_details = {} # fid -> {max_gap, min_g, best_g?}
    
    # Pre-fetch plane data
    w_data = {}
    for wid, pl in window_planes.items():
        w_data[wid] = {
            'n': np.array(pl['plane_n']),
            'pt': np.array(pl['plane_pt'])
        }
        
    for fid in dataset['frames']:
        # For efficiency, we can group rays? No, just iter.
        # k=0 (A), k=1 (B)
        for k, (key, radius_base) in enumerate([('obsA', R_small_mm), ('obsB', R_large_mm)]):
             if fid not in dataset[key]: continue
             
             limit = radius_base + margin_mm
             
             # Group observations by wid to triangulate?
             # No, standard triangulation needs mult-cam.
             # We need to collect rays for this point (fid, k).
             
             rays = []
             obs_map = dataset[key][fid]
             
             for cid, uv in obs_map.items():
                 if cid not in cam_to_window: continue
                 wid = cam_to_window[cid]
                 cam = cams_cpp[cid]
                 
                 # Trace
                 label = "A" if k == 0 else "B"
                 # Ensure build_pinplate_ray_cpp is available
                 r = build_pinplate_ray_cpp(
                     cam, uv, 
                     cam_id=cid, window_id=wid, frame_id=fid, endpoint=label
                 )
                 if r.valid:
                     rays.append(r)
             
             if len(rays) < 2: continue
             
             # Triangulate
             X, _, success, _ = triangulate_point(rays)
             if not success: continue
             
             # Check constraints for ALL involved windows
             involved_wids = set(r.window_id for r in rays)
             
             for wid in involved_wids:
                 if wid not in w_data: continue
                 pl = w_data[wid]
                 sX = np.dot(pl['n'], X - pl['pt'])
                 
                 gap = limit - sX # positive means violation
                 # User defined: g = sX - limit. g < 0 is violation.
                 # Let's stick to g = sX - limit.
                 g = sX - limit
                 
                 if sX < min_dist: min_dist = sX
                 if g < min_sX: min_sX = g
                 
                 if g < -0.001: # Violation threshold (hardcoded small epsilon for reporting equality)
                     v_entry = {
                         'frame': fid,
                         'cam': -1, # Point-based
                         'window': wid,
                         'endpoint': 'A' if k==0 else 'B',
                         'sX': sX,
                         'limit': limit,
                         'g': g,
                         'gap': -g
                     }
                     violations.append(v_entry)
                     if worst_viol is None or g < worst_viol['g']:
                         worst_viol = v_entry
                          
                 # Collect per-frame stats for injection
                 gap_val = gap # limit - sX
                 g_val = g     # sX - limit
                 
                 if fid not in frame_details:
                     frame_details[fid] = {'frame': fid, 'max_gap': -999.0, 'min_g': 999.0}
                 
                 if gap_val > frame_details[fid]['max_gap']:
                     frame_details[fid]['max_gap'] = gap_val
                     frame_details[fid]['worst_info'] = f"W{wid} {label}"
                     
                 if g_val < frame_details[fid]['min_g']:
                     frame_details[fid]['min_g'] = g_val

    return min_sX, violations, worst_viol, frame_details, min_dist

def analyze_worst_frames(frame_details):
    """
    Process frame_details to get sorted worst and near frames.
    """
    if not frame_details:
        return [], []
        
    # Sort by gap (descending) -> Most violating first
    # gap = limit - sX. larger gap = worse. 
    # frame_details is dict: fid -> {max_gap, min_g, ...}
    
    worst_list = [] # Frames with gap > 0 (violation)
    near_list = []  # All frames, sorted by g (closest to violating)
    
    for fid, stats in frame_details.items():
        if stats['max_gap'] > 0.001:
             worst_list.append(stats)
        
        near_list.append(stats)
        
    worst_sorted = sorted(worst_list, key=lambda x: x['max_gap'], reverse=True)
    # Near: Sort by g ascending. Smallest g is most negative (worst violation) or smallest positive (closest safety).
    near_sorted = sorted(near_list, key=lambda x: x['min_g'])
    
    return worst_sorted, near_sorted


class FreezeStatus(Enum):
    """Status for each parameter DOF."""
    OPTIMIZE = "OPTIMIZE"
    FREEZE = "FREEZE"
    OPTIMIZE_REGULARIZED = "OPTIMIZE_REG"
    OPTIMIZE_STRONG = "OPT_STRONG"


@dataclass
class ObservabilityInfo:
    """Per-window observability analysis."""
    window_id: int
    n_cam: int  # Number of cameras viewing this window
    camera_ids: List[int] = field(default_factory=list)
    baseline_max_mm: float = 0.0  # Max baseline between camera pairs
    baseline_median_mm: float = 0.0
    angle_diversity_p50: float = 0.0  # degrees, median pairwise view angle
    angle_diversity_p90: float = 0.0  # degrees, 90th percentile
    
    # Freeze decisions
    plane_status: FreezeStatus = FreezeStatus.OPTIMIZE
    tvec_status: FreezeStatus = FreezeStatus.FREEZE
    rvec_status: FreezeStatus = FreezeStatus.FREEZE
    freeze_reason: str = ""


@dataclass
class PR4Config:
    """Configuration for PR4 Bundle Adjustment."""
    # Regularization
    lambda_reg_plane: float = 10.0  # Normal drift penalty
    lambda_reg_tvec: float = 1.0    # Translation drift penalty
    lambda_reg_rvec: float = 50.0   # Rotation drift penalty (standard)
    
    # Sampling
    max_frames: int = 50000  # Default to all (high limit)
    random_seed: int = 42
    
    # Unit Normalization
    px_target: float = 0.5            # 投影误差目标 (px)
    wand_tol_pct: float = 0.02        # 棒长容忍度 (2% 即 0.2mm)
    
    # Sphere Radii (Estimated or Config)
    R_small_mm: float = 1.5
    R_large_mm: float = 2.0

    # Stage control
    skip_pr4: bool = False
    pr4_stage: int = 3
    verbosity: int = 1
    margin_side_mm: float = 0.05    # Margin for side constraint (mm)
    alpha_side_gate: float = 10.0   # Gate magnitude: C_gate = alpha * J_ref
    beta_side_dir: float = 1e4      # Directional weight when gate is active
    beta_side_soft: float = 100.0   # Soft floor weight when gate is NOT active (though PR4 defaults to ON)
    
    
@dataclass
class PR5Config:
    """
    Configuration for PR5 Robust Bundle Adjustment.
    Geometric-only optimization (Ray + Wand) with Strong Priors.
    """
    pr5_stage: int = 2  # 1=BA, 2=Final Joint
    verbosity: int = 1
    
    # Robust Loss (Huber)
    delta_ray: float = 0.30
    delta_len: float = 0.15
    
    # Lambda Weights
    lambda_ray: float = 1.0     # Base weight for ray residuals
    lambda_len_init: float = 100.0  # Initial weight for length residuals
    lambda_len_min: float = 10.0
    lambda_len_max: float = 1000.0
    
    # Side Gate (Hysteresis-Based Feasibility Constraint)
    # Stable round-level gate with zero cost when feasible
    margin_side_mm: float = 0.05     # Margin for side constraint (mm); sX < margin is violation
    v_on_side_gate: float = 0.010    # Turn gate ON if v > 0.010
    v_off_side_gate: float = 0.0     # Turn gate OFF if v <= 0.0 (strict feasibility)
    alpha_side_gate: float = 10.0    # Gate magnitude: C_gate = alpha * J_ref
    beta_side_dir: float = 1e4       # Directional weight when gate ON
    beta_side_soft: float = 100.0    # Soft floor weight when gate OFF
    scale_len_gate_active: float = 0.1 # Multiply lambda_len by this when gate is ON (reduce dominance)
    
    # Sampling
    max_frames: int = 50000  # Default to all (high limit)
    random_seed: int = 42
    
    # Unit Normalization
    # Unit Normalization
    px_target: float = 0.5            # 投影误差目标 (px)
    wand_tol_pct: float = 0.05        # 棒长容忍度 (5%)

    bounds_thick_pct: float = 0.05
    bounds_f_pct: float = 0.05        # 5%
    bounds_alpha_beta_deg: float = 5.0
    bounds_d_delta_mm: float = 10.0   # 10mm
    
    # Uniform Sigma Priors (User Specified)
    # 10 deg, 10 mm, 20 mm, 10 deg
    sigma_rvec: float = 0.1745      # 10.0 degrees
    sigma_tvec: float = 10.0        # 10.0 mm
    sigma_d: float = 10.0           # 10.0 mm
    sigma_plane_ang: float = 0.0873 # 5.0 degrees
    pr4_stage: int = 3
    
    # Logging
    verbosity: int = 1  # 0=clean, 1=summary+tables, 2=full audit  # Max stage to run (1, 2, or 3)


class RefractiveBAOptimizerPR4:
    """
    PR4 Bundle Adjustment Optimizer.
    
    Refines window planes AND selected camera extrinsics based on observability.
    """
    
    def __init__(self,
                 dataset: Dict,
                 cam_params: Dict[int, np.ndarray],
                 cams_cpp: Dict,
                 cam_to_window: Dict[int, int],
                 window_media: Dict[int, Dict],
                 window_planes: Dict[int, Dict],
                 wand_length: float,
                 config: Optional[PR4Config] = None,
                 progress_callback: Optional[callable] = None):
        """
        Initialize PR4 optimizer.
        
        Args:
            dataset: Observation data with 'obsA', 'obsB', 'frames' keys
            cam_params: Dict mapping cam_id to parameter array [rvec(3), tvec(3), ...]
            cams_cpp: Dict mapping cam_id to C++ Camera objects
            cam_to_window: Dict mapping cam_id to window_id
            window_media: Dict with window properties (thickness, n_obj, etc.)
            window_planes: Dict with plane parameters (plane_n, plane_pt)
            wand_length: Target wand length in mm
            config: Optimization configuration
            progress_callback: Optional callback(phase, ray_rmse, len_rmse, cost) for UI updates
        """
        self.dataset = dataset
        self.cam_params = {int(k): np.array(v, dtype=np.float64) for k, v in cam_params.items()}
        self.cams_cpp = cams_cpp
        self.cam_to_window = {int(k): int(v) for k, v in cam_to_window.items()}
        self.window_media = window_media
        self.wand_length = wand_length
        self.config = config or PR4Config()
        self.progress_callback = progress_callback  # For UI progress updates

        
        # Deep copy window_planes for modification
        self.window_planes = {}
        for wid, pl in window_planes.items():
            self.window_planes[int(wid)] = {
                'plane_pt': np.array(pl['plane_pt'], dtype=np.float64),
                'plane_n': np.array(pl['plane_n'], dtype=np.float64)
            }
        
        # Store initial values for regularization
        self.initial_planes = {wid: {
            k: (v.copy() if hasattr(v, 'copy') else v) for k, v in pl.items()
        } for wid, pl in self.window_planes.items()}
        
        self.initial_cam_params = {cid: p.copy() for cid, p in self.cam_params.items()}
        self._j_ref = 1.0 # Reference cost for barrier scaling
        
        # Derived data
        self.window_ids = sorted(self.window_planes.keys())
        self.active_cam_ids = sorted(self.cam_params.keys())
        
        # Build per-window camera lists
        self.window_to_cams = {wid: [] for wid in self.window_ids}
        for cid, wid in self.cam_to_window.items():
            if cid in self.active_cam_ids and wid in self.window_to_cams:
                self.window_to_cams[wid].append(cid)
        
        # Build observation cache
        self._build_obs_cache()
        
        # Frame sampling
        all_frames = sorted(self.dataset.get('frames', []))
        if self.config.max_frames > 0 and len(all_frames) > self.config.max_frames:
             # Random sample
             import random
             rnd = random.Random(self.config.random_seed)
             self.fids_optim = sorted(rnd.sample(all_frames, self.config.max_frames))
        else:
             self.fids_optim = all_frames
        
    def _sync_initial_state(self):
        """Update initial_planes/cams from current state (Relinearization)."""
        self.initial_planes = {wid: {
            k: (v.copy() if hasattr(v, 'copy') else v) for k, v in pl.items()
        } for wid, pl in self.window_planes.items()}
        
        self.initial_cam_params = {cid: p.copy() for cid, p in self.cam_params.items()}
        
        self._last_ray_rmse = -1.0
        self._last_len_rmse = -1.0
        
        self.sigma_ray_global = 0.04  # Default, will be recalculated
        self.sigma_wand = 0.1        # Default, will be recalculated
    
    
    def _build_obs_cache(self):
        """Build observation cache from dataset."""
        self.obs_cache = {}
        obsA = self.dataset.get('obsA', {})
        obsB = self.dataset.get('obsB', {})
        
        all_fids = set(obsA.keys()) | set(obsB.keys())
        
        for fid in all_fids:
            self.obs_cache[fid] = {}
            for cid in self.active_cam_ids:
                uvA = None
                uvB = None
                if fid in obsA and cid in obsA[fid]:
                    pt = obsA[fid][cid]
                    if pt is not None and len(pt) >= 2 and np.all(np.isfinite(pt[:2])):
                        uvA = pt[:2]
                if fid in obsB and cid in obsB[fid]:
                    pt = obsB[fid][cid]
                    if pt is not None and len(pt) >= 2 and np.all(np.isfinite(pt[:2])):
                        uvB = pt[:2]
                if uvA is not None or uvB is not None:
                    self.obs_cache[fid][cid] = (uvA, uvB)
    

    def _compute_physical_sigmas(self):
        """Estimate global sigma values across all optimize stages."""
        cfg = self.config
        
        all_f = [p[6] for p in self.cam_params.values() if len(p) > 6]
        avg_f = np.mean(all_f) if all_f else 1000.0
        
        sample_dists = []
        fids_to_sample = self.fids_optim[::max(1, len(self.fids_optim) // 100)]
        for fid in fids_to_sample:
            # Reconstruct centers
            obs = self.dataset['obsA'].get(fid, {})
            cids = sorted(obs.keys())
            rays = []
            for cid in cids:
                if cid in self.cams_cpp:
                    r = build_pinplate_ray_cpp(self.cams_cpp[cid], obs[cid])
                    if r.valid: rays.append(r)
            if len(rays) >= 2:
                X, _, ok, _ = triangulate_point(rays)
                if ok:
                    for cid in cids:
                        rv = self.cam_params[cid][0:3]
                        tv = self.cam_params[cid][3:6]
                        R = cv2.Rodrigues(rv)[0]
                        C = -R.T @ tv
                        sample_dists.append(np.linalg.norm(X - C))
        
        avg_dist_z = np.mean(sample_dists) if sample_dists else 600.0
        
        self.sigma_ray_global = cfg.px_target * (avg_dist_z / avg_f)
        self.sigma_wand = self.wand_length * cfg.wand_tol_pct
        
        if cfg.verbosity >= 1:
            print(f"  [PR4] Unit Normalization: sigma_ray={self.sigma_ray_global:.4f}mm ({cfg.px_target}px at Z={avg_dist_z:.1f}mm), sigma_wand={self.sigma_wand:.4f}mm ({cfg.wand_tol_pct*100}%)")

    def _print_plane_diagnostics(self, stage_name: str):
        """Print current plane normals and angles between them."""
        print(f"\n  [{stage_name}] Plane Diagnostics:")
        wids = sorted(self.window_planes.keys())
        normals = []
        for wid in wids:
            n = self.window_planes[wid]['plane_n']
            pt = self.window_planes[wid]['plane_pt']
            normals.append(n)
            print(f"    Win {wid}: n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}], pt=[{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}]")
        
        if len(normals) == 2:
            ang = angle_between_vectors(normals[0], normals[1])
            print(f"    Angle between Win 0 and Win 1: {ang:.2f}°")
    

    def print_diagnostics(self, current_planes: Dict, current_cam_params: Dict):
        """Print final comparison of parameters."""
        cfg = self.config
        
        if cfg.verbosity >= 1:
             print("\n  [PR4] Final Parameter Diagnostics (Detailed Diff skipped for brevity)")
        
        print("\n  [PR4] Final Window State (Delta Invariant d_key):")
        for wid, pl_new in current_planes.items():
            pl_init = self.initial_planes[wid]
            
            # New normal/pt
            n_new = pl_new['plane_n']
            pt_new = pl_new['plane_pt']
            
            # Compute d_key: dot(n, pt - C_mean)
            # Need to re-compute C_mean from *optimized* cams
            cams = self.window_to_cams.get(wid, [])
            centers = []
            for cid in cams:
                if cid in current_cam_params:
                    p = current_cam_params[cid]
                    R, _ = cv2.Rodrigues(p[0:3])
                    C = camera_center(R, p[3:6])
                    centers.append(C)
            
            d_key = 0.0
            if centers:
                C_mean = np.mean(centers, axis=0)
                d_key = np.dot(n_new, pt_new - C_mean)
                
            delta_n_deg = np.degrees(angle_between_vectors(pl_init['plane_n'], n_new))
            
            print(f"    Win {wid}: d_key={d_key:.2f} mm (from opt C_mean), normal_shift={delta_n_deg:.2f} deg")







    def _build_rays_frame(self, fid: int) -> Tuple[List[Ray], List[Ray]]:
        """Build rays for a frame using current C++ camera state."""
        rays_A, rays_B = [], []
        
        if fid not in self.obs_cache:
            return rays_A, rays_B
        
        for cid, (uvA, uvB) in self.obs_cache[fid].items():
            if cid not in self.cams_cpp:
                continue
            
            wid = self.cam_to_window.get(cid, -1)
            cam_obj = self.cams_cpp[cid]
            
            if uvA is not None:
                r = build_pinplate_ray_cpp(cam_obj, uvA, cam_id=cid, 
                                           window_id=wid, frame_id=fid, endpoint="A")
                if r.valid:
                    rays_A.append(r)
            
            if uvB is not None:
                r = build_pinplate_ray_cpp(cam_obj, uvB, cam_id=cid,
                                           window_id=wid, frame_id=fid, endpoint="B")
                if r.valid:
                    rays_B.append(r)
        
        return rays_A, rays_B

    def evaluate_residuals(self, planes: Dict[int, Dict], cam_params: Dict[int, np.ndarray],
                           lambda_eff: float) -> Tuple[np.ndarray, float, float, int, int]:
        """
        Evaluate residuals with fixed-size padding for Scipy least_squares compatibility.
        """
        # Apply planes and extrinsics
        # Apply to C++ objects (Consolidated Update)
        for cid in self.active_cam_ids:
            if cid not in self.cams_cpp: continue
            
            # Prepare update arguments
            update_kwargs = {}
            
            # 1. Extrinsics
            if cid in cam_params:
                p = cam_params[cid]
                update_kwargs['extrinsics'] = {'rvec': p[0:3], 'tvec': p[3:6]}
            
            # 2. Plane Geometry
            wid = self.cam_to_window.get(cid)
            if wid in planes:
                pl = planes[wid]
                update_kwargs['plane_geom'] = {
                    'pt': pl['plane_pt'].tolist(), 
                    'n': pl['plane_n'].tolist()
                }
            
            if update_kwargs:
                update_cpp_camera_state(self.cams_cpp[cid], **update_kwargs)
        
        # 1. Pre-calculate total possible counts for FIXED size
        total_ray_slots = 0
        for fid in self.fids_optim:
            for key in ['obsA', 'obsB']:
                obs = self.dataset[key].get(fid, {})
                total_ray_slots += len(obs)
        
        total_len_slots = len(self.fids_optim)
        
        # Barrier slots: each point (max 2 per frame) can collide with its window's planes
        total_barrier_slots = 0
        for fid in self.fids_optim:
            for key in ['obsA', 'obsB']:
                obs = self.dataset[key].get(fid, {})
                if obs:
                    # Point sees these windows. Use cam_to_window mapping to define slots.
                    wids = set()
                    for cid in obs.keys():
                        wid = self.cam_to_window.get(cid)
                        if wid is not None and wid != -1:
                            wids.add(wid)
                    # 2 residuals per window (Step + Gradient)
                    total_barrier_slots += 2 * len(wids)
                    
        # Pre-allocate
        res_ray_fixed = np.zeros(total_ray_slots)
        res_len_fixed = np.zeros(total_len_slots)
        res_barrier_fixed = np.zeros(total_barrier_slots)
        
        PENALTY_RAY = 100.0   # mm
        PENALTY_LEN = self.wand_length
        
        idx_ray = 0
        idx_len = 0
        idx_barrier = 0
        
        S_ray = 0.0
        S_len = 0.0
        N_ray_actual = 0
        N_len_actual = 0
        num_triangulated_points = 0
        valid_points_data = [] # For barrier computation
        
        radius_A = self.dataset.get('est_radius_small_mm', 0.0)
        radius_B = self.dataset.get('est_radius_large_mm', 0.0)
        
        # --- Computation Loop ---
        for fid in self.fids_optim:
            # Get observations for this frame
            obs_A = self.dataset['obsA'].get(fid, {})
            obs_B = self.dataset['obsB'].get(fid, {})
            
            # --- Endpoint A ---
            n_slots_A = len(obs_A)
            # Build rays in STABLE order (sorted cam_id)
            cids_A = sorted(obs_A.keys())
            rays_A_all = []
            for cid in cids_A:
                cam_obj = self.cams_cpp.get(cid)
                if cam_obj:
                    r = build_pinplate_ray_cpp(cam_obj, obs_A[cid], cam_id=cid, 
                                               window_id=self.cam_to_window.get(cid, -1), 
                                               frame_id=fid, endpoint="A")
                    rays_A_all.append(r)
                else:
                    # Fallback for missing camera object
                    rays_A_all.append(Ray(o=np.zeros(3), d=np.array([0,0,1]), valid=False, reason="missing_cam", cam_id=cid))
            
            validA = False
            XA = None
            rays_A_valid = [r for r in rays_A_all if r.valid]
            if len(rays_A_valid) >= 2:
                XA, _, validA, _ = triangulate_point(rays_A_valid)
            
            # Use fixed iteration to fill slots
            start_idx_barrier_A = idx_barrier
            # Determine expected windows for this point
            wids_A_expected = sorted([w for w in set(self.cam_to_window.get(cid) for cid in cids_A) if w is not None and w != -1])
            n_barrier_A = len(wids_A_expected)
            
            if validA:
                num_triangulated_points += 1
                valid_points_data.append((XA, wids_A_expected, 'A', start_idx_barrier_A))
                for r in rays_A_all:
                    if r.valid:
                        d = point_to_ray_dist(XA, r.o, r.d)
                        res_ray_fixed[idx_ray] = d / self.sigma_ray_global
                        S_ray += d**2
                        N_ray_actual += 1
                    else:
                        res_ray_fixed[idx_ray] = PENALTY_RAY / self.sigma_ray_global
                    idx_ray += 1
            else:
                # Triangulation failed, fill all rays with penalty
                for _ in range(n_slots_A):
                    res_ray_fixed[idx_ray] = PENALTY_RAY / self.sigma_ray_global
                    idx_ray += 1
            
            # Advance barrier index by fixed amount
            idx_barrier += 2 * n_barrier_A

            # --- Endpoint B ---
            n_slots_B = len(obs_B)
            cids_B = sorted(obs_B.keys())
            rays_B_all = []
            for cid in cids_B:
                cam_obj = self.cams_cpp.get(cid)
                if cam_obj:
                    r = build_pinplate_ray_cpp(cam_obj, obs_B[cid], cam_id=cid, 
                                               window_id=self.cam_to_window.get(cid, -1), 
                                               frame_id=fid, endpoint="B")
                    rays_B_all.append(r)
                else:
                    rays_B_all.append(Ray(o=np.zeros(3), d=np.array([0,0,1]), valid=False, reason="missing_cam", cam_id=cid))
            
            validB = False
            XB = None
            rays_B_valid = [r for r in rays_B_all if r.valid]
            if len(rays_B_valid) >= 2:
                XB, _, validB, _ = triangulate_point(rays_B_valid)
            
            start_idx_barrier_B = idx_barrier
            wids_B_expected = sorted([w for w in set(self.cam_to_window.get(cid) for cid in cids_B) if w is not None and w != -1])
            n_barrier_B = len(wids_B_expected)
            
            if validB:
                num_triangulated_points += 1
                valid_points_data.append((XB, wids_B_expected, 'B', start_idx_barrier_B))
                for r in rays_B_all:
                    if r.valid:
                        d = point_to_ray_dist(XB, r.o, r.d)
                        res_ray_fixed[idx_ray] = d / self.sigma_ray_global
                        S_ray += d**2
                        N_ray_actual += 1
                    else:
                        res_ray_fixed[idx_ray] = PENALTY_RAY / self.sigma_ray_global
                    idx_ray += 1
            else:
                for _ in range(n_slots_B):
                    res_ray_fixed[idx_ray] = PENALTY_RAY / self.sigma_ray_global
                    idx_ray += 1
            
            idx_barrier += 2 * n_barrier_B

            # --- Wand Length ---
            if validA and validB:
                wand_len = np.linalg.norm(XA - XB)
                err = wand_len - self.wand_length
                res_len_fixed[idx_len] = err / self.sigma_wand
                S_len += err**2
                N_len_actual += 1
            else:
                res_len_fixed[idx_len] = PENALTY_LEN / self.sigma_wand
            idx_len += 1

        # 2. Side Barrier (Adaptive)
        J_data = S_ray + lambda_eff * S_len
        margin_mm = self.config.margin_side_mm
        sX_vals = []
        
        # Hard Barrier Constants (PR5-style)
        C_gate = self.config.alpha_side_gate * self._j_ref
        r_fix_const = np.sqrt(2.0 * C_gate)
        r_grad_const = np.sqrt(2.0 * self.config.beta_side_dir)
        tau = 0.01

        for (X, wids, endpoint, b_start_idx) in valid_points_data:
            r_val = radius_A if endpoint == 'A' else radius_B
            curr_b_idx = b_start_idx
            for wid in wids: # Already sorted at creation
                if wid not in planes: 
                    curr_b_idx += 2
                    continue
                pl = planes[wid]
                n = pl['plane_n']
                P_plane = pl['plane_pt']
                # Signed distance sX (Positive = correct side)
                sX = np.dot(n, X - P_plane)
                sX_vals.append(sX)

                # PR5-style Strong Constraint (Gate ON by default)
                gap = (margin_mm + r_val) - sX
                if gap > 0:
                    # Violation: Smooth Step + Gradient
                    res_barrier_fixed[curr_b_idx] = r_fix_const * (1.0 - np.exp(-gap / tau))
                    res_barrier_fixed[curr_b_idx + 1] = r_grad_const * gap
                else:
                    # Feasible
                    res_barrier_fixed[curr_b_idx] = 0.0
                    res_barrier_fixed[curr_b_idx + 1] = 0.0
                
                curr_b_idx += 2

        # Diagnostics
        if sX_vals:
            sX_arr = np.array(sX_vals)
            self._last_barrier_stats = {
                'min_sX': np.min(sX_arr),
                'pct_near': np.mean(sX_arr < margin_mm) * 100,
                'ratio': np.sum(res_barrier_fixed**2) / max(1e-9, J_data)
            }
        else:
            self._last_barrier_stats = {}

        # Update RMSE for diagnostics based on physical units
        self._last_ray_rmse = np.sqrt(S_ray / max(1, N_ray_actual))
        self._last_len_rmse = np.sqrt(S_len / max(1, N_len_actual)) if N_len_actual > 0 else 0.0

        # 3. Combine
        weighted_len = np.sqrt(lambda_eff) * res_len_fixed
        residuals = np.concatenate([res_ray_fixed, weighted_len, res_barrier_fixed])
            
        return residuals, S_ray, S_len, N_ray_actual, N_len_actual

    def _get_param_layout(self, enable_planes: bool, enable_cam_t: bool, enable_cam_r: bool) -> List[Tuple]:
        """
        Get layout of parameter vector x based on enabled flags.
        Returns list of (type, id, subparam_idx).
        """
        layout = []
        
        # 1. Planes
        if enable_planes:
            for wid in self.window_ids:
                layout.append(('plane_d', wid, 0))
                layout.append(('plane_a', wid, 0))
                layout.append(('plane_b', wid, 0))
        
        # 2. Cameras
        if enable_cam_t or enable_cam_r:
            for cid in self.active_cam_ids:
                if enable_cam_t:
                    layout.append(('cam_t', cid, 0)) # tx
                    layout.append(('cam_t', cid, 1)) # ty
                    layout.append(('cam_t', cid, 2)) # tz
                
                if enable_cam_r:
                    layout.append(('cam_r', cid, 0)) # rx
                    layout.append(('cam_r', cid, 1)) # ry
                    layout.append(('cam_r', cid, 2)) # rz
        
        return layout
    
    def _rvec_step_cap_deg(self, status: int) -> float:
        """Get rotation step cap in degrees based on optimization status."""
        config = self.config
        if status == FreezeStatus.OPTIMIZE_STRONG:
            return config.step_cap_rot_deg_strong
        elif status == FreezeStatus.OPTIMIZE_REGULARIZED:
            return config.step_cap_rot_deg_weak
        else: # Normal OPTIMIZE
            return config.step_cap_rot_deg_normal
    
    def _rvec_step_cap_deg(self, status: int) -> float:
        """Get rotation step cap in degrees based on optimization status."""
        config = self.config
        if status == FreezeStatus.OPTIMIZE_STRONG:
            return config.step_cap_rot_deg_strong
        elif status == FreezeStatus.OPTIMIZE_REGULARIZED:
            return config.step_cap_rot_deg_weak
        else: # Normal OPTIMIZE
            return config.step_cap_rot_deg_normal


    def _unpack_params_delta(self, x: np.ndarray, layout: List[Tuple]) -> Tuple[Dict, Dict]:
        """
        Unpack x (deltas) into updated planes and cam_params.
        
        Returns:
            (new_planes, new_cam_params)
        """
        # Start from INITIAL state
        current_planes = {}
        for wid, pl in self.initial_planes.items():
            current_planes[wid] = {
                'plane_n': pl['plane_n'].copy(),
                'plane_pt': pl['plane_pt'].copy(),
                'initialized': pl.get('initialized', True)
            }
        
        current_cam_params = {cid: p.copy() for cid, p in self.initial_cam_params.items()}
        
        idx = 0
        for (ptype, pid, subidx) in layout:
            val = x[idx]
            idx += 1
            
            if ptype.startswith('plane'):
                # Plane update logic
                # We need to collect d, a, b for each window
                # This unpacking is slightly inefficient (repeatedly accessing), but safe
                pass 
        
        # Better approach: Group by ID first
        # But 'layout' defines the order in 'x'.
        # Let's iterate 'x' and accumulate updates
        
        plane_deltas = {wid: {'d': 0.0, 'a': 0.0, 'b': 0.0} for wid in self.window_ids}
        cam_deltas = {cid: {'t': np.zeros(3), 'r': np.zeros(3)} for cid in self.active_cam_ids}
        
        idx = 0
        for (ptype, pid, subidx) in layout:
            val = x[idx]
            idx += 1
            
            if ptype == 'plane_d':
                plane_deltas[pid]['d'] = val
            elif ptype == 'plane_a':
                plane_deltas[pid]['a'] = val
            elif ptype == 'plane_b':
                plane_deltas[pid]['b'] = val
            elif ptype == 'cam_t':
                cam_deltas[pid]['t'][subidx] = val
            elif ptype == 'cam_r':
                cam_deltas[pid]['r'][subidx] = val
        
        # Apply Plane Deltas
        for wid, deltas in plane_deltas.items():
            if wid not in current_planes: continue
            
            n0 = self.initial_planes[wid]['plane_n']
            pt0 = self.initial_planes[wid]['plane_pt']
            
            # 1. Update distance (d)
            # d_new = d_old + delta_d
            # pt_new = pt_old + delta_d * n0  (approximation of shift along normal)
            d_shift = deltas['d']
            pt_shifted = pt0 + d_shift * n0
            
            # 2. Update normal (alpha, beta) using tangent space
            alpha, beta = deltas['a'], deltas['b']
            n_new = update_normal_tangent(n0, alpha, beta)
            
            current_planes[wid]['plane_n'] = n_new
            current_planes[wid]['plane_pt'] = pt_shifted
            current_planes[wid]['initialized'] = True
            
        # Apply Camera Deltas
        for cid, deltas in cam_deltas.items():
            if cid not in current_cam_params: continue
            
            # Apply tvec delta
            current_cam_params[cid][3:6] += deltas['t']
            
            # Apply rvec delta
            # R_new = R_delta * R_old  (global perturbation? or local?)
            # Usually optimization finds a delta-rvec.
            # Local perturbation: R_new = R(delta) * R_old
            # Global perturbation: R_new = R_old * R(delta)
            # Let's use Local (perturbation intrisic to camera frame?)
            # Actually standard bundle adjustment often uses: R_new = exp(w) * R_old
            # where w is rotation vector update.
            
            r_old = current_cam_params[cid][0:3]
            dr = deltas['r']
            
            if np.linalg.norm(dr) > 1e-8:
                R_old, _ = cv2.Rodrigues(r_old)
                dR_mat, _ = cv2.Rodrigues(dr)
                # Left multiplication (global) vs Right multiplication (local)
                # Camera projects: X_c = R * X_w + T
                # If we perturb R: (I + [w]x) * R * X_w
                # This corresponds to left multiplication.
                R_new = dR_mat @ R_old
                r_new, _ = cv2.Rodrigues(R_new)
                current_cam_params[cid][0:3] = r_new.flatten()
            
        return current_planes, current_cam_params

    def _residuals_pr4(self, x: np.ndarray, layout: List[Tuple], mode: str, lambda_eff: float) -> np.ndarray:
        """Residual function for generic PR4 optimization."""
        # Unpack
        curr_planes, curr_cams = self._unpack_params_delta(x, layout)
        
        # Data Residuals
        # Note: evaluate_residuals handles applying to CPP internally
        residuals, S_ray, S_len, N_ray, N_len = self.evaluate_residuals(curr_planes, curr_cams, lambda_eff)
        
        # [Fix] Update live stats for logging
        self._last_ray_rmse = np.sqrt(S_ray / max(N_ray, 1))
        self._last_len_rmse = np.sqrt(S_len / max(N_len, 1)) if N_len > 0 else 0.0
        
        # Regularization
        reg_residuals = []
        cfg = self.config
        
        idx = 0
        for (ptype, pid, subidx) in layout:
            val = x[idx]
            idx += 1
            
            if ptype.startswith('plane'):
                # Plane regularization
                # Penalty on deviation from initial (d, alpha, beta)
                reg_residuals.append(val * np.sqrt(cfg.lambda_reg_plane))
            elif ptype == 'cam_t':
                reg_residuals.append(val * np.sqrt(cfg.lambda_reg_tvec))
            elif ptype == 'cam_r':
                weight = cfg.lambda_reg_rvec
                reg_residuals.append(val * np.sqrt(weight))
        
        if len(reg_residuals) > 0:
            residuals = np.concatenate([residuals, np.array(reg_residuals)])

        return np.array(residuals)

    def _optimize_generic(self, mode: str, description: str, 
                          enable_planes: bool, enable_cam_t: bool, enable_cam_r: bool,
                          limit_rot_rad: float, limit_trans_mm: float, 
                          limit_plane_d_mm: float, limit_plane_angle_rad: float,
                          plane_d_bounds: Dict[int, float] = None,
                          ftol: float = 1e-6):
        """
        Generic optimization loop with explicit bounds and parameter selection.
        """
        layout = self._get_param_layout(enable_planes, enable_cam_t, enable_cam_r)
        
        if not layout:
            print(f"  [{description}] No parameters to optimize.")
            return

        x0 = np.zeros(len(layout), dtype=np.float64)
        cfg = self.config
        
        print(f"  [{description}] optimizing {len(x0)} parameters ({len(layout)//3} blocks)...")
        # Calc initial RMSE for rollback reference
        planes0, cams0 = self._unpack_params_delta(x0, layout)
        
        # [USER REQUEST] Fixed Weighting Strategy
        # Lambda = 2.0 * N_active_cams
        n_cams = max(1, len(self.active_cam_ids))
        lambda_fixed = 2.0 * n_cams
        
        # Initial evaluation
        _, S_ray0, S_len0, N_ray, N_len = self.evaluate_residuals(planes0, cams0, lambda_fixed)
        
        # [Constraint] Force lambda=1 if S_ray / S_len < 10 (User Request)
        if S_len0 > 1e-9:
             ratio_s = S_ray0 / S_len0
             if ratio_s < 10.0:
                 print(f"    [Constraint] S_ray/S_len ({ratio_s:.2f}) < 10. Forcing lambda=1.0 (was {lambda_fixed:.1f})")
                 lambda_fixed = 1.0
        
        rmse_ray0 = np.sqrt(S_ray0 / max(N_ray, 1))
        rmse_len0 = np.sqrt(S_len0 / max(N_len, 1)) if N_len > 0 else 0.0
        
        # Initial normalized cost J0
        J0 = (rmse_ray0**2) + lambda_fixed * (rmse_len0**2)
        self._j_ref = J0 if J0 > 1e-6 else 1.0
        
        print(f"    Global Fixed Weighting: lambda={lambda_fixed:.1f} (N_cams={n_cams})")
        print(f"    Initial: S_ray={S_ray0:.2f}, S_len={S_len0:.2f} (J0={J0:.4f})")
        
        # Build Bounds
        lb = []
        ub = []
        for (ptype, pid, subidx) in layout:
            if ptype == 'plane_d':
                limit = limit_plane_d_mm
                if plane_d_bounds and pid in plane_d_bounds:
                    limit = plane_d_bounds[pid]
                lb.append(-limit)
                ub.append(limit)
            elif ptype == 'plane_a' or ptype == 'plane_b':
                lb.append(-limit_plane_angle_rad)
                ub.append(limit_plane_angle_rad)
            elif ptype == 'cam_r':
                lb.append(-limit_rot_rad)
                ub.append(limit_rot_rad)
            elif ptype == 'cam_t':
                lb.append(-limit_trans_mm)
                ub.append(limit_trans_mm)
            else:
                lb.append(-1.0)
                ub.append(1.0)
        
        bounds = (np.array(lb), np.array(ub))
        
         # Residual wrapper for event pumping
        self._res_call_count = 0
        def residuals_wrapper(x, *args, **kwargs):
            res = self._residuals_pr4(x, *args, **kwargs)
            self._res_call_count += 1
            if self.progress_callback and self._res_call_count % 30 == 0:
                try:
                    c_approx = 0.5 * np.sum(res**2)
                    
                    # [DEBUG] Print to terminal instead of UI
                    if hasattr(self, '_last_ratio_info'):
                         # Calculate percentage
                         j_ratio = getattr(self, '_last_ratio_cost', 0.0)
                         pct = (j_ratio / c_approx * 100) if c_approx > 0 else 0
                         print(f"  [PR4 DEBUG] J_tot={c_approx:.1e}, J_ratio={j_ratio:.1e} ({pct:.1f}%) | {self._last_ratio_info}")
                         
                    if self.progress_callback:
                        self.progress_callback(f"{description}", self._last_ray_rmse, self._last_len_rmse, c_approx)
                except Exception as e:
                    print(f"[Warning] Progress callback failed: {e}")
            return res


        # Single Pass Optimization
        res = least_squares(
            residuals_wrapper, 
            x0, 
            args=(layout, mode, lambda_fixed),
            method='trf', 
            bounds=bounds,
            verbose=0,
            x_scale='jac',
            ftol=ftol,
            xtol=1e-6,
            gtol=1e-6,
            max_nfev=50
        )

        # Print Barrier Stats
        if cfg.verbosity >= 1 and hasattr(self, '_last_barrier_stats') and self._last_barrier_stats:
            s = self._last_barrier_stats
            print(f"    [PR4][SIDE-BARRIER] min(sX)={s['min_sX']:.4f}mm, near(<20um)={s['pct_near']:.1f}%, cost/J={s['ratio']:.1e}")

        # Final evaluation
        planes_final, cams_final = self._unpack_params_delta(res.x, layout)
        _, S_rayF, S_lenF, _, _ = self.evaluate_residuals(planes_final, cams_final, lambda_fixed)
        
        rmse_rayF = np.sqrt(S_rayF / max(N_ray, 1))
        rmse_lenF = np.sqrt(S_lenF / max(N_len, 1)) if N_len > 0 else 0.0
        JF = (rmse_rayF**2) + lambda_fixed * (rmse_lenF**2)
        
        print(f"    Final:   S_ray={S_rayF:.2f}, S_len={S_lenF:.2f} (JF={JF:.4f})")
        print(f"      RMSE Ray: {rmse_ray0:.4f} -> {rmse_rayF:.4f}")
        print(f"      RMSE Len: {rmse_len0:.4f} -> {rmse_lenF:.4f}")
        
        # Rollback check if degraded (Safety)
        # However, pure geometric optimization shouldn't degrade unless local minima.
        # We accept result.
        
        # Update Initial State for next stage
        self.initial_planes = planes_final
        self.initial_cam_params = cams_final
        
        # Update Public State
        self.window_planes = planes_final
        self.cam_params = cams_final
        
        return res, layout

    def _print_plane_diagnostics(self, stage_name: str):
        """Print current plane normals and angles between them."""
        print(f"\n  [{stage_name}] Plane Diagnostics:")
        wids = sorted(self.window_planes.keys())
        normals = []
        for wid in wids:
            n = self.window_planes[wid]['plane_n']
            pt = self.window_planes[wid]['plane_pt']
            normals.append(n)
            print(f"    Win {wid}: n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}], pt=[{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}]")
            
            # [LOGGING] Print distances from associated cameras
            cams = self.window_to_cams.get(wid, [])
            for cid in cams:
                if cid in self.cam_params:
                     p = self.cam_params[cid]
                     R, _ = cv2.Rodrigues(p[0:3])
                     C = camera_center(R, p[3:6])
                     dist = abs(np.dot(n, pt - C))
                     print(f"      -> Cam {cid}: Dist {dist:.2f} mm")
        
        if len(normals) == 2:
            from .refractive_geometry import angle_between_vectors
            ang = angle_between_vectors(normals[0], normals[1])
            print(f"    Angle between Win 0 and Win 1: {ang:.2f}°")

    def _perform_geometric_initialization(self, wid: int, cid: int):
        """
        Hard geometric reset for weak window.
        Move plane so closest 3D point is at exactly 'gap' distance.
        gap = R_ball + 0.2 * d_cam_point + 0.05
        """
        print(f"  [GeoInit] Performing geometric initialization for Win {wid} (Cam {cid})...")
        
        # 1. Get Camera Center
        p = self.cam_params[cid]
        R, _ = cv2.Rodrigues(p[0:3])
        C_A = -R.T @ p[3:6]
        
        # 2. Find closest 3D point optimized by *other* views
        # We need points that are triangulatable.
        min_dist = 1e9
        X_min = None
        R_min = 0.0 # Radius of ball at that point
        
        # Helper to triangulate single frame without `cid` (to avoid circular dependency on this plane)
        count_3d = 0
        
        # Iterate cache (subset for speed?)
        fids = sorted(list(self.obs_cache.keys()))
        step = max(1, len(fids) // 2000) # Check up to 2000 frames
        
        for fid in fids[::step]:
            obs = self.obs_cache[fid]
            if cid not in obs: continue
            
            # Check for other cams
            uvA_self, uvB_self = obs[cid]
            
            for endpoint, uv_self, radius_val in [('A', uvA_self, self.config.R_small_mm), ('B', uvB_self, self.config.R_large_mm)]:
                if uv_self is None: continue
                
                # Build rays from OTHER cameras
                rays_other = []
                for other_cid, (o_uvA, o_uvB) in obs.items():
                    if other_cid == cid: continue
                    val = o_uvA if endpoint == 'A' else o_uvB
                    if val is not None:
                        # Build ray
                        r = build_pinplate_ray_cpp(
                            self.cams_cpp[other_cid], val, 
                            cam_id=other_cid, 
                            window_id=self.cam_to_window.get(other_cid, -1),
                            frame_id=fid, endpoint=endpoint
                        )
                        if r.valid:
                            rays_other.append(r)
                
                if len(rays_other) >= 2:
                    X, _, ok, _ = triangulate_point(rays_other)
                    if ok:
                        d = np.linalg.norm(X - C_A)
                        if d < min_dist:
                            min_dist = d
                            X_min = X
                            R_min = radius_val
                            count_3d += 1
        
        if X_min is None:
            print(f"  [GeoInit] Failed: No triangulatable 3D points found for Win {wid} (Need overlap with other cams).")
            return

        # 3. Calculate target gap
        margin = 0.05
        # User formula: gap = R_min + 0.1 * d_min + margin
        gap = R_min + 0.1 * min_dist + margin
        
        # 4. Move Plane
        pl = self.window_planes[wid]
        n = pl['plane_n']
        pt = pl['plane_pt']
        
        # Current s0 = dot(n, X_min - pt)
        # We need s_new = gap
        # s_new = dot(n, X_min - (pt + t*n)) = s0 - t
        # => t = s0 - gap
        
        s0 = np.dot(n, X_min - pt)
        t = s0 - gap
        
        pt_new = pt + t * n
        
        # 5. Safety Check
        # Check 1: Point on object side?
        # s(X_min)_new = dot(n, X_min - pt_new)
        s_X_new = np.dot(n, X_min - pt_new)
        # Check 2: Camera on camera side?
        # s(C_A)_new = dot(n, C_A - pt_new)
        s_C_new = np.dot(n, C_A - pt_new)
        
        print(f"  [GeoInit] Found X_min at d={min_dist:.2f}mm. s0={s0:.2f}mm -> Target gap={gap:.2f}mm.")
        print(f"            Shift t={t:.2f}mm. Check: s(X)={s_X_new:.4f} (>0), s(C)={s_C_new:.4f} (<0)")
        
        if s_X_new > -1e-3 and s_C_new < 1e-3:
            # Apply
            pl['plane_pt'] = pt_new
            print(f"  [GeoInit] APPLIED. New pt=[{pt_new[0]:.2f}, {pt_new[1]:.2f}, {pt_new[2]:.2f}]")
            
            return min_dist
        else:
            print(f"  [GeoInit] REJECTED. Geometric violation. (C on wrong side or X on wrong side)")
            return None

    def _detect_weak_windows(self):
        """
        Identify weak windows (Single Camera + Angle < 5 deg) vs Strong windows.
        Computes reference d1_avg from strong windows.
        
        Stores:
          self._weak_windows = {wid: {'cam_id': cid, 'd0_init': float, 'n_init': vec}}
          self._d1_avg_ref = float (mean distance of strong windows)
        """
        self._weak_windows = {}
        self._strong_windows_list = []
        strong_dists = []
        
        print("\n[PR4] Detecting Weak Windows (Dist-Ratio Constraint)...")
        
        for wid in self.window_planes:
            cams = self.window_to_cams.get(wid, [])
            cams_active = [c for c in cams if c in self.active_cam_ids]
            
            pl = self.window_planes[wid]
            n = pl['plane_n']
            pt = pl['plane_pt']
            
            # Compute distance for this window (mean of cameras)
            d_win_vals = []
            for cid in cams_active:
                p = self.cam_params[cid]
                R, _ = cv2.Rodrigues(p[0:3])
                C = camera_center(R, p[3:6])
                # Dist = dot(n, pt - C_cam) ? No, C++ convention: s < 0
                # Distance = |dot(n, pt - C)|
                dist = abs(np.dot(n, pt - C))
                d_win_vals.append(dist)
            
            d_win_mean = np.mean(d_win_vals) if d_win_vals else 0.0
            
            # Classification
            # Weak if: 1 active camera AND angle(n, optical_axis) < 5 deg
            is_weak = False
            angle_deg = 90.0
            
            if len(cams_active) == 1:
                cid = cams_active[0]
                # Optical axis: R.T @ [0,0,1] = [r_31, r_32, r_33] (3rd row of R?)
                # Actually Z-axis of camera frame in world coords.
                # R maps World->Cam. Z_cam = [0,0,1].
                # in World: R.T @ [0,0,1] = 3rd row of R (since R is orthogonal)? No, 3rd column of R.T = 3rd row of R.
                # Yes, R = [r1; r2; r3]. Z_cam_in_world = r3 (3rd row of R).
                p = self.cam_params[cid]
                R, _ = cv2.Rodrigues(p[0:3])
                opt_axis = R[2, :] # 3rd row
                
                # Angle between n and opt_axis
                # dot
                costh = abs(np.dot(n, opt_axis))
                angle_deg = np.degrees(np.arccos(min(1.0, costh)))
                
                if angle_deg < 5.0:
                    is_weak = True
            
            if is_weak:
                # Count observations (frames) for this camera
                # obs_count calculation... (keep existing logic)
                obs_count = 0
                cid = cams_active[0]
                for fid in self.obs_cache:
                    if cid in self.obs_cache[fid]:
                        uvA, uvB = self.obs_cache[fid][cid]
                        if uvA is not None or uvB is not None:
                             obs_count += 1
                
                # Check geometric init status
                if not hasattr(self, '_weak_window_refs'):
                    self._weak_window_refs = {}
                
                d_min_ref = 0.0
                if wid not in self._weak_window_refs:
                    # Execute ONCE
                    d_min = self._perform_geometric_initialization(wid, cid)
                    if d_min is not None:
                        self._weak_window_refs[wid] = d_min
                        d_min_ref = d_min
                        
                        # [Fix] Refresh d0 for log since plane moved
                        p = self.cam_params[cid]
                        R, _ = cv2.Rodrigues(p[0:3])
                        C = -R.T @ p[3:6]
                        pl_new = self.window_planes[wid]
                        d_win_mean = abs(np.dot(pl_new['plane_n'], pl_new['plane_pt'] - C))
                else:
                    d_min_ref = self._weak_window_refs[wid]
                
                self._weak_windows[wid] = {
                    'cam_id': cams_active[0], 
                    'd0_init': d_win_mean, 
                    'angle_deg': angle_deg,
                    'obs_count': obs_count,
                    'd_min_ref': d_min_ref
                }
                print(f"  [WEAK] Win {wid}: 1 Cam ({cid}), Ang={angle_deg:.2f}° (<5°). d0={d_win_mean:.2f}mm, d_min={d_min_ref:.2f}")
            elif len(cams_active) > 0:
                self._strong_windows_list.append(wid)
                strong_dists.append(d_win_mean)
                print(f"  [STRONG] Win {wid}: {len(cams_active)} Cams, Ang={angle_deg:.2f}°. d={d_win_mean:.2f}mm")
        
        # Initial Reference (for logging/fallback)
        if strong_dists:
            self._d1_avg_ref = np.mean(strong_dists)
        else:
            self._d1_avg_ref = 600.0 # Fallback
            
        print(f"  Ref Distance d1 (Initial Strong Avg): {self._d1_avg_ref:.2f} mm")

    def optimize(self, skip_pr4: bool = False, pr4_stage: int = 3) -> Tuple[Dict[int, Dict], Dict[int, np.ndarray]]:
        """
        Execute PR4 Optimization (Alternating Refinement).
        
        Strategy:
        1. Alternating Loop (Max 6 iterations):
           - A: Optimize Planes (Fixed Cams). Bounds: Angle +/- 2.5 deg.
           - B: Optimize Cams (Fixed Planes). Bounds: Free.
           - Check: If Plane optimization (A) did NOT hit angle boundary, terminate loop early.
        2. Final Joint Optimization (PR4.3 / Round 3).
        """
        self._compute_physical_sigmas()
        
        # [MOVED per user request] Weak window detection now inside loop.
        # self._detect_weak_windows()
        
        # [NEW] Persistent store for geometric init state (d_min)
        self._weak_window_refs = {}
        
        enable_ray_tracking(True, reset=True)
        print(f"\n[PR4] Optimization Start ({len(self.active_cam_ids)} cameras, {len(self.window_ids)} windows)")
        for wid, pl in sorted(self.window_planes.items()):
            pt = pl['plane_pt']
            n = pl['plane_n']
            print(f"  [PR4][INIT] Win {wid}: pt=[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}], n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")

        if skip_pr4 or self.config.skip_pr4:
            print("[PR4] Skipped (config.skip_pr4=True).")
            return self.window_planes, self.cam_params

        # --- Alternating Loop ---
        max_loop_iters = 6
        loop_iter = 0
        hit_boundary = True # Assume hit to start
        
        print(f"\n[PR4] Starting Alternating Loop (Max {max_loop_iters} passes)")
        
        while loop_iter < max_loop_iters:
            loop_iter += 1
            
            # [NEW] Dynamic Detection per loop
            self._detect_weak_windows()
            
            # [Fix] Sync initial state because _detect_weak_windows might have moved planes (GeoInit)
            # This ensures the optimization starts from the correct new position (delta=0 -> new_pt)
            self._sync_initial_state()
            
            # [NEW] Calculate Shrinking Bounds for Weak Windows
            # Loop 1: +/- 10% of d_min
            # Loop 2: +/- 5%
            # Loop 3: +/- 2.5%
            # Formula: 0.1 * (0.5 ^ (loop_iter - 1))
            plane_d_bounds = {}
            if hasattr(self, '_weak_windows'):
                print(f"  [PR4][LOOP {loop_iter}] Bounds Configuration:")
                print(f"    Global Plane Angle: +/- 2.5 deg")
                
                factor = 0.1 * (0.5 ** (loop_iter - 1))
                for wid, info in self._weak_windows.items():
                    d_ref = info.get('d_min_ref', 0.0)
                    if d_ref > 0:
                        limit = d_ref * factor
                        plane_d_bounds[wid] = limit
                        print(f"    [WEAK BOUNDS] Win {wid}: +/- {limit:.2f} mm ({factor*100:.2f}% of {d_ref:.1f}mm)")
            
            print(f"\n[PR4][LOOP {loop_iter}] Step A: Optimize Planes (Bounds: +/- 2.5 deg)")
            self._print_plane_diagnostics(f"Pre-Loop {loop_iter} Planes")
            
            # Step A: Optimize Planes (Fixed Cams) - Strict Angle Bound (2.5 deg)
            # Bounds: Angle +/- 2.5 deg, Distance +/- 500mm (effectively free)
            limit_angle_rad = np.radians(2.5)
            b_plane_strict = (limit_angle_rad, 500.0)
            
            res_planes, layout_planes = self._optimize_generic(
                mode=f'pr4_loop_{loop_iter}_planes', 
                description=f"Adjusting plane parameters ...",
                enable_planes=True,
                enable_cam_t=False,
                enable_cam_r=False,
                limit_rot_rad=0.0,
                limit_trans_mm=0.0,
                limit_plane_d_mm=b_plane_strict[1],
                limit_plane_angle_rad=b_plane_strict[0],
                plane_d_bounds=plane_d_bounds, # [NEW] Pass bounds
                ftol=5e-4
            )
            self._print_plane_diagnostics(f"PR4 Loop {loop_iter} Planes")
            
            # Check for Boundary Hit in Plane Angles
            # active_mask: 0 = interior, -1/1 = hit bound
            # Inspect layout to find plane_a/plane_b indices
            active_mask = res_planes.active_mask
            hit_boundary = False
            
            idx = 0
            for (ptype, pid, subidx) in layout_planes:
                if (ptype == 'plane_a' or ptype == 'plane_b') and active_mask[idx] != 0:
                    hit_boundary = True
                    # print(f"  [DEBUG] Hit boundary on {ptype} (Win {pid})")
                idx += 1
            
            if hit_boundary:
                print(f"  [PR4][LOOP {loop_iter}] Plane constraints ACTIVE (hit 2.5 deg bound). Continuing loop.")
            else:
                print(f"  [PR4][LOOP {loop_iter}] Plane constraints INACTIVE (all within 2.5 deg). Loop condition satisfied.")

            # Step B: Optimize Cameras (Fixed Planes) - Free Bounds
            print(f"\n[PR4][LOOP {loop_iter}] Step B: Optimize Cameras (Free Bounds)")
            b_cam_free = (np.deg2rad(180.0), 2000.0)
            
            self._optimize_generic(
                mode=f'pr4_loop_{loop_iter}_cams', 
                description=f"Optimizing camera extrinsic parameters ...",
                enable_planes=False,
                enable_cam_t=True,
                enable_cam_r=True,
                limit_rot_rad=b_cam_free[0],
                limit_trans_mm=b_cam_free[1],
                limit_plane_d_mm=0.0,
                limit_plane_angle_rad=0.0,
                ftol=5e-4
            )
            self._print_plane_diagnostics(f"PR4 Loop {loop_iter} Cams")

            # Check termination
            if not hit_boundary:
                print(f"  [PR4] Converged early at Loop {loop_iter} (Planes inside 2.5 deg). Stopping loop.")
                break
        
        if hit_boundary and loop_iter == max_loop_iters:
             print(f"  [PR4] Loop reached max iterations ({max_loop_iters}). Proceeding to Joint.")

        # --- Final Joint Optimization (Round 3) ---
        if pr4_stage >= 3:
            print("\n[PR4][FINAL] Joint Optimization (Round 3 Rules).")
            
            # [NEW] Re-detect before final joint
            self._detect_weak_windows()
            # Bounds: 20 deg, 50 mm d, 10 mm tvec
            limit_rvec = np.radians(20.0)
            limit_plane_d = 50.0
            limit_plane_ang = np.radians(20.0)
            limit_tvec = 10.0
            
            print(f"  Bounds: rvec < 20deg, plane_d < 50mm, plane_ang < 20deg, tvec < 10mm")
            
            self._optimize_generic(
                mode='pr4_final_joint', 
                description="Optimizing plane and camera extrinsic parameters ...",
                enable_planes=True,
                enable_cam_t=True,
                enable_cam_r=True,
                limit_rot_rad=limit_rvec,
                limit_trans_mm=limit_tvec,
                limit_plane_d_mm=limit_plane_d,
                limit_plane_angle_rad=limit_plane_ang,
                ftol=1e-5
            )
            self._print_plane_diagnostics("PR4 Final Joint End")
        
        # Explicit sync call to be safe for returning
        n_cams = max(1, len(self.active_cam_ids))
        lambda_fixed = 2.0 * n_cams
        self.evaluate_residuals(self.window_planes, self.cam_params, lambda_fixed)

        self.print_diagnostics()
        print("\n[PR4] Optimization Complete.")
        print_ray_stats_report("PR4 Bundle")
        enable_ray_tracking(False)
        
        return self.window_planes, self.cam_params

    def _optimize_pr4_1(self): pass
    def _optimize_pr4_2(self): pass
    def _optimize_pr4_3(self): pass
    
    def print_diagnostics(self):
        """Print comprehensive diagnostics after optimization."""
        print("\n[PR4] Final Diagnostics:")
        print("-" * 40)
        
        # Evaluate final residuals
        n_cams = max(1, len(self.active_cam_ids))
        lambda_fixed = 2.0 * n_cams
        residuals, S_ray, S_len, N_ray, N_len = self.evaluate_residuals(
            self.window_planes, self.cam_params, lambda_fixed
        )
        
        # Ray stats
        if N_ray > 0:
            ray_rmse = np.sqrt(S_ray / N_ray)
            print(f"  Ray Distance RMSE: {ray_rmse:.4f} mm ({N_ray} rays)")
        
        # Wand stats
        if N_len > 0:
            wand_rmse = np.sqrt(S_len / N_len)
            print(f"  Wand Length RMSE: {wand_rmse:.4f} mm ({N_len} pairs)")
            print(f"  Wand Length Target: {self.wand_length:.2f} mm")
        
        # Per-window summary
        print("\n  Per-Window Summary (d_internal=dot(n,pt), d_key_phys=dot(n,pt-Cmean)):")
        for wid in self.window_ids:
            pl = self.window_planes[wid]
            n = pl['plane_n']
            pt = pl['plane_pt']
            
            # d_internal: legacy/optimization-internal value
            d_internal = np.dot(n, pt)
            
            # d_key_phys: canonical distance from cameras (P1 invariant)
            cams = self.window_to_cams.get(wid, [])
            centers = []
            for cid in cams:
                if cid in self.cam_params:
                    p = self.cam_params[cid]
                    R, _ = cv2.Rodrigues(p[0:3])
                    C = camera_center(R, p[3:6])
                    centers.append(C)
            
            d_key_phys = 0.0
            if centers:
                C_mean = np.mean(centers, axis=0)
                d_key_phys = np.dot(n, pt - C_mean)
            
            print(f"    Window {wid}: d_internal={d_internal:.2f}mm, d_key_phys={d_key_phys:.2f}mm, n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")
        
        print("-" * 40)

    def _get_cache_path(self, dataset_path: str) -> str:
        """Get path to PR4 cache file."""
        return str(Path(dataset_path).parent / "pr4_bundle_cache.json")

    def try_load_cache(self, out_path: str) -> bool:
        """
        Try to load PR4 results from cache.
        Returns True if loaded successfully.
        """
        cache_path = self._get_cache_path(out_path)
        if not Path(cache_path).exists():
            return False
            
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                
            # Verify version matching or simple existance
            # PR4 cache structure is simple: planes, cam_params
            
            # Load Params (Params / Windows) - NO DATASET
            cached_cams = data.get('cam_params', {})
            for cid_str, p_list in cached_cams.items():
                cid = int(cid_str)
                if cid in self.cam_params:
                    self.cam_params[cid] = np.array(p_list)
                    
            # Load Planes
            planes_data = data.get('planes', {}) # Current PR4 format uses 'planes'
            for wid_str, pl in planes_data.items():
                wid = int(wid_str)
                if wid in self.window_planes:
                     self.window_planes[wid]['plane_pt'] = np.array(pl['plane_pt'])
                     self.window_planes[wid]['plane_n'] = np.array(pl['plane_n'])

            # Apply to C++ objects (Consolidated Update)
            for cid in self.active_cam_ids:
                if cid not in self.cams_cpp: continue
                
                # Prepare update arguments
                update_kwargs = {}
                
                # 1. Extrinsics AND Intrinsics
                if cid in self.cam_params:
                    p = self.cam_params[cid]
                    update_kwargs['extrinsics'] = {'rvec': p[0:3], 'tvec': p[3:6]}
                    
                    # [CRITICAL] Pass full intrinsics to ensure update_cpp_camera_state 
                    # doesn't zero them out if C++ state is not fully initialized.
                    # This prevents PR4 cache load from corrupting state for PR5.
                    update_kwargs['intrinsics'] = {
                        'f': p[6],
                        'cx': p[7],
                        'cy': p[8],
                        'dist': [p[9], p[10], 0, 0, 0]
                    }
                
                # 2. Plane Geometry
                wid = self.cam_to_window.get(cid)
                if wid in self.window_planes:
                    pl = self.window_planes[wid]
                    update_kwargs['plane_geom'] = {
                        'pt': pl['plane_pt'].tolist(), 
                        'n': pl['plane_n'].tolist()
                    }
                
                if update_kwargs:
                    update_cpp_camera_state(self.cams_cpp[cid], **update_kwargs)
                
            
            print(f"[PR4][CACHE] Loaded parameters successfully from {cache_path}")
            print(f"  Note: Using cached parameters with FRESH dataset observations.")
            return True
        except Exception as e:
            print(f"[PR4][CACHE] Load failed (ignored): {e}")
            return False

    def save_cache(self, out_path: str):
        """Save PR4 results to cache."""
        try:
            cache_path = self._get_cache_path(out_path)
            
            data = {
                'timestamp': str(datetime.now()),
                'cam_ids': self.active_cam_ids,
                'window_ids': self.window_ids,
                'planes': {
                    str(w): {
                        'plane_pt': np.asarray(pl['plane_pt']).tolist(),
                        'plane_n': np.asarray(pl['plane_n']).tolist()
                    } for w, pl in self.window_planes.items()
                },
                'cam_params': {
                    str(c): np.asarray(p).tolist() for c, p in self.cam_params.items()
                }
            }
            
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"[PR4][CACHE] Saved results to {cache_path}")
            
        except Exception as e:
            print(f"[PR4][CACHE] Save failed: {e}")


class RefractiveBAOptimizerPR5:
    """
    PR5: Robust Bundle Adjustment for Refractive Calibration.
    Optimizes:
    - Window Planes (d, alpha, beta)
    - Window Thickness (t)
    - Intrinsics (f)
    - Extrinsics (rvec, tvec)
    
    Uses purely geometric residuals (Ray Distance + Wand Length) with Strong Priors.
    """
    
    def __init__(self,
                 dataset: Dict,
                 cam_params: Dict[int, np.ndarray],
                 cams_cpp: Dict[int, 'lpt.Camera'],
                 cam_to_window: Dict[int, int],
                 window_media: Dict[int, Dict],
                 window_planes: Dict[int, Dict],
                 wand_length: float,
                 config: PR5Config,
                 progress_callback: Optional[callable] = None,
                 freeze_table: Dict = None):
        
        self.dataset = dataset
        self.cam_params = {int(cid): np.array(p, dtype=np.float64) for cid, p in cam_params.items()}
        self.freeze_table = freeze_table
        self.cams_cpp = cams_cpp
        self.cam_to_window = cam_to_window
        
        # Deep copy window data to avoid polluting previous stages until commit
        self.window_media = {int(w): m.copy() for w, m in window_media.items()}
        self.window_planes = {}
        for wid, pl in window_planes.items():
            self.window_planes[int(wid)] = {
                'plane_pt': np.array(pl['plane_pt'], dtype=np.float64),
                'plane_n': np.array(pl['plane_n'], dtype=np.float64),
                **{k: v for k, v in pl.items() if k not in ['plane_pt', 'plane_n']}
            }
        
        self.wand_length = wand_length
        self.config = config
        self.progress_callback = progress_callback  # For UI progress updates
        
        self._last_ray_rmse = -1.0
        self._last_len_rmse = -1.0
        self._last_s_ray = 0.0
        self._last_n_ray = 0
        self._last_s_len = 0.0
        self._last_n_len = 0
        
        self.sigma_ray_global = 0.04
        self.sigma_wand = 0.1

        
        self.active_cam_ids = sorted(list(self.cams_cpp.keys()))
        self.window_ids = sorted(list(self.window_planes.keys()))
        
        # Initial State storage (for priors)
        self.initial_cam_params = {cid: p.copy() for cid, p in self.cam_params.items()}
        self.initial_planes = {w: {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in p.items()} for w, p in self.window_planes.items()}
        self.initial_media = {w: (m.copy() if hasattr(m, 'copy') else m) for w, m in self.window_media.items()}
        
        # Extract initial intrinsics (f)
        self.initial_f = {}
        for cid in self.active_cam_ids:
            # Assuming pinplate param structure
            pp = self.cams_cpp[cid]._pinplate_param
            # f = (fx + fy) / 2
            fx = pp.cam_mtx[0, 0]
            fy = pp.cam_mtx[1, 1]
            self.initial_f[cid] = (fx + fy) / 2.0

        # Build Observation Cache (same as PR4)
        self._build_obs_cache()
        
        # Build Camera->Window reverse map logic
        self.window_to_cams = {}
        for cid in self.active_cam_ids:
            wid = self.cam_to_window.get(cid)
            if wid not in self.window_to_cams: self.window_to_cams[wid] = []
            self.window_to_cams[wid].append(cid)
        
    def _build_obs_cache(self, frames: Optional[List[int]] = None):
        """Build observation cache from dataset (optionally subset of frames)."""
        self.obs_cache = {} # {fid: {cid: (uvA, uvB)}}
        obsA = self.dataset.get('obsA', {})
        obsB = self.dataset.get('obsB', {})
        
        target_frames = frames
        if target_frames is None:
            # Default to all frames if not specified
            target_frames = sorted(list(set(obsA.keys()) | set(obsB.keys())))
        
        for fid in target_frames:
            self.obs_cache[fid] = {}
            for cid in self.active_cam_ids:
                uvA = None
                uvB = None
                
                if cid in obsA.get(fid, {}):
                    pt = obsA[fid][cid]
                    if pt is not None and len(pt) >= 2 and np.all(np.isfinite(pt[:2])):
                        uvA = pt[:2]
                
                if cid in obsB.get(fid, {}):
                    pt = obsB[fid][cid]
                    if pt is not None and len(pt) >= 2 and np.all(np.isfinite(pt[:2])):
                        uvB = pt[:2]
                        
                if uvA is not None or uvB is not None:
                    self.obs_cache[fid][cid] = (uvA, uvB)


    def build_pr5_frame_subset(self, all_frame_ids, max_frames, injected_frames_prev=None, rng=None):
        """
        Returns a list of frame_ids to use for this PR5 round.
        Strategy:
          - K_cap = ceil(0.10 * max_frames)
          - injected = first min(len(injected_frames_prev), K_cap) frames
          - remaining = max_frames - len(injected)
          - random sample remaining frames from (all_frame_ids - injected)
        """
        if injected_frames_prev is None:
            injected_frames_prev = []
            
        K_cap = int(np.ceil(0.10 * max_frames))
        
        # 1. Select Injected (Preserve order: worst first)
        injected = []
        for fid in injected_frames_prev:
            if len(injected) >= K_cap:
                break
            if fid in all_frame_ids:
                injected.append(fid)
                
        # 2. Random Sample Remaining
        needed = max_frames - len(injected)
        pool = sorted(list(set(all_frame_ids) - set(injected)))
        
        if rng is None:
            import random
            rng = random.Random(self.config.random_seed if hasattr(self.config, 'random_seed') else 42)
            
        if len(pool) <= needed:
            subset = injected + pool
        else:
            selected_random = sorted(rng.sample(pool, max(0, needed)))
            subset = injected + selected_random
            
        # Log sampling
        if self.config.verbosity >= 1:
            print(f"[PR5][SAMPLE] max={max_frames}, K_cap={K_cap}, injected={len(injected)}, random={len(subset)-len(injected)}")
                
        return sorted(subset)

    def _compute_physical_sigmas(self):
        """Estimate global sigma values across all optimize stages."""
        cfg = self.config
        all_f = [p[6] for p in self.cam_params.values() if len(p) > 6]
        avg_f = np.mean(all_f) if all_f else 1000.0
        
        sample_dists = []
        # Use first few frames of obs_cache
        fids_all = sorted(list(self.obs_cache.keys()))
        fids_to_sample = fids_all[::max(1, len(fids_all) // 100)]
        for fid in fids_to_sample:
            obs = self.obs_cache.get(fid, {})
            rays = []
            for cid in obs:
                uvA, _ = obs[cid]
                if uvA is not None:
                    # cam_id=cid, window_id=self.cam_to_window.get(cid, -1), frame_id=fid, endpoint="A"
                    r = build_pinplate_ray_cpp(self.cams_cpp[cid], uvA, cam_id=cid, window_id=self.cam_to_window.get(cid, -1), frame_id=fid, endpoint="A")
                    if r.valid: rays.append(r)
            if len(rays) >= 2:
                X, _, ok, _ = triangulate_point(rays)
                if ok:
                    for cid in obs:
                        rv = self.cam_params[cid][0:3]
                        tv = self.cam_params[cid][3:6]
                        R = cv2.Rodrigues(rv)[0]
                        C = -R.T @ tv
                        sample_dists.append(np.linalg.norm(X - C))
        
        avg_dist_z = np.mean(sample_dists) if sample_dists else 600.0
        
        self.sigma_ray_global = cfg.px_target * (avg_dist_z / avg_f)
        self.sigma_wand = self.wand_length * cfg.wand_tol_pct
        
        if cfg.verbosity >= 1:
            print(f"  [PR5] Unit Normalization: sigma_ray={self.sigma_ray_global:.4f}mm ({cfg.px_target}px at Z={avg_dist_z:.1f}mm), sigma_wand={self.sigma_wand:.4f}mm ({cfg.wand_tol_pct*100}%)")

    def _print_plane_diagnostics(self, stage_name: str):
        """Print current plane normals and angles between them."""
        print(f"\n  [{stage_name}] Plane Diagnostics:")
        wids = sorted(self.window_planes.keys())
        normals = []
        for wid in wids:
            n = self.window_planes[wid]['plane_n']
            pt = self.window_planes[wid]['plane_pt']
            normals.append(n)
            print(f"    Win {wid}: n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}], pt=[{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}]")
            
            # [LOGGING] Print distances from associated cameras
            cams = self.window_to_cams.get(wid, [])
            for cid in cams:
                if cid in self.cam_params:
                     p = self.cam_params[cid]
                     R, _ = cv2.Rodrigues(p[0:3])
                     C = camera_center(R, p[3:6])
                     dist = abs(np.dot(n, pt - C))
                     print(f"      -> Cam {cid}: Dist {dist:.2f} mm")
        
        if len(normals) == 2:
            from .refractive_geometry import angle_between_vectors
            ang = angle_between_vectors(normals[0], normals[1])
            print(f"    Angle between Win 0 and Win 1: {ang:.2f}°")


    def optimize(self) -> Tuple[Dict, Dict, Dict]:
        """Run PR5 optimization (Single Round)."""
        enable_ray_tracking(True, reset=True)
        verbosity = self.config.verbosity
        print("\n" + "="*60)
        print("PR5: ROBUST BUNDLE ADJUSTMENT (Geometric Only - Simplified)")
        print("="*60)
        
        print(f"\n[PR5] Optimization Start ({len(self.active_cam_ids)} cameras, {len(self.window_ids)} windows)")
        for wid, pl in sorted(self.window_planes.items()):
            pt = pl['plane_pt']
            n = pl['plane_n']
            print(f"  [PR5][INIT] Win {wid}: pt=[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}], n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")

        # Calculate initial RMSE to avoid -1.0 display at start
        try:
             x0, _, _ = self._pack_parameters()
             if len(x0) > 0:
                 res = self._residuals_pr5(x0, self.config.lambda_len_init)
                 # Values are now set in self._last_s_ray/len 
                 self._last_ray_rmse = np.sqrt(self._last_s_ray / max(self._last_n_ray, 1))
                 self._last_len_rmse = np.sqrt(self._last_s_len / max(self._last_n_len, 1)) if self._last_n_len > 0 else 0.0
        except:
             pass
        
        # PR5 Main BA (Single Stage)
        self._optimize_stage(stage_name="Optimizing plane and all camera parameters ...", max_rounds=1)
        
        for wid, pl in sorted(self.window_planes.items()):
            pt = pl['plane_pt']
            n = pl['plane_n']
            print(f"  [PR5][FINAL] Win {wid}: pt=[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}], n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")
        
        self._print_plane_diagnostics("PR5 Final End")

        print_ray_stats_report("PR5 Bundle")
        enable_ray_tracking(False)
            
        return self.window_planes, self.cam_params, self.window_media

    def _optimize_stage(self, stage_name: str, max_rounds: int, tight: bool = False):
        """
        Generic PR5 stage optimization. 
        Single round with uniform priors.
        """
        print(f"\n  [{stage_name}] Optimization...")
        
        cfg = self.config
        # Fixed Weighting
        n_cams = max(1, len(self.active_cam_ids))
        lambda_fixed = 2.0 * n_cams
        
        # Initial pack
        x0, bounds, names = self._pack_parameters()
        
        if len(x0) == 0:
            print("    No parameters to optimize.")
            return

        print(f"    Num Params: {len(x0)}")
        
        # Force Gate Enable
        self._round_gate_enabled = True 
        
        # Full Dataset Cache
        self._build_obs_cache(None) # None = All frames
        
        # --- ROUND 1: Optimization ---
        print(f"\n[PR5] Optimization (Full Dataset, lambda={lambda_fixed}, Gate=ON)")
        
        # Residual wrapper
        self._res_call_count_pr5 = 0
        def residuals_wrapper_pr5(x):
             # Fixed weighting
             res = self._residuals_pr5(x, lambda_fixed) # Pass fixed lambda
             self._res_call_count_pr5 += 1
             if self.progress_callback and self._res_call_count_pr5 % 30 == 0:
                 try:
                     c_approx = 0.5 * np.sum(res**2)
                     # Report actual tracked RMSEs (Full)
                     r_rmse = np.sqrt(self._last_s_ray / max(self._last_n_ray, 1))
                     l_rmse = np.sqrt(self._last_s_len / max(self._last_n_len, 1)) if self._last_n_len > 0 else 0.0
                     
                     
                     # [RESTORED] Log Barrier Stats
                     # [USER REQUEST] Print to terminal, NOT UI
                     if hasattr(self, '_last_barrier_stats') and self._last_barrier_stats:
                         s = self._last_barrier_stats
                         print(f"  [PR5 BARRIER] min(sX)={s['min_sX']:.4f}mm vio={s['num_violations']}")
                     
                     if self.progress_callback:
                        self.progress_callback(stage_name, r_rmse, l_rmse, c_approx)
                 except:
                     pass
             return res

        res = least_squares(
            residuals_wrapper_pr5,
            x0,
            bounds=bounds,
            loss='linear',
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
            verbose=0,
            max_nfev=50
        )
        x0 = res.x
        
        if cfg.verbosity >= 1:
            print(f"    Res: Cost={res.cost:.4f}, nfev={res.nfev}, msg={res.message}")

        # Commit Final State
        planes_up, thick_map, f_map, rvec_map, tvec_map = self._unpack_parameters(x0)
        
        # 1. Update C++ and get final planes
        final_planes = self._apply_params_to_cpp(planes_up, thick_map, f_map, rvec_map, tvec_map)
        
        # 2. Update Python state
        for wid, pl in final_planes.items():
            self.window_planes[wid] = pl
            self.window_media[wid]['thickness'] = thick_map[wid]
            
        for cid in self.active_cam_ids:
            if cid in rvec_map:
                self.cam_params[cid][0:3] = rvec_map[cid]
            if cid in tvec_map:
                self.cam_params[cid][3:6] = tvec_map[cid]
            if cid in f_map:
                self.cam_params[cid][6] = f_map[cid]
        
        # 3. Stage Report
        print(f"  [{stage_name}] Complete.")
        if cfg.verbosity >= 1:
            mean_f = np.mean([f_map[c] for c in self.active_cam_ids])
            mean_t = np.mean([thick_map[w] for w in self.window_ids])
            f_delta = mean_f - np.mean(list(self.initial_f.values()))
            t_delta = mean_t - np.mean([self.initial_media[w]['thickness'] for w in self.window_ids])
            print(f"    Avg F: {mean_f:.2f} (diff: {f_delta:+.2f})")
            print(f"    Avg Thick: {mean_t:.4f} (diff: {t_delta:+.4f} mm)")

        # [DEBUG] Verify F persistence
        if len(self.active_cam_ids) > 0:
             cid0 = self.active_cam_ids[0]
             print(f"    [DEBUG] Stage End: Cam {cid0} F={self.cam_params[cid0][6]:.4f}")

    def _get_cache_path(self, dataset_path: str) -> str:
        """Get path to PR5 cache file."""
        return str(Path(dataset_path).parent / "pr5_bundle_cache.json")

    def try_load_cache(self, out_path: str) -> bool:
        """
        Try to load PR5 results from cache.
        Returns True if loaded successfully.
        """
        cache_path = self._get_cache_path(out_path)
        if not os.path.exists(cache_path):
            return False
            
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                
            # Validation
            # Check camera IDs match
            cached_cams = set(data.get('cam_ids', []))
            current_cams = set(self.active_cam_ids)
            if cached_cams != current_cams:
                print(f"[PR5][CACHE] Mismatch cam IDs: {cached_cams} vs {current_cams}")
                # We can accept mismatch if dataset is fresh? Maybe risky for now.
                return False
                
            # Check Window IDs
            cached_wins = set(map(int, data.get('window_ids', [])))
            current_wins = set(self.window_ids)
            if cached_wins != current_wins:
                print(f"[PR5][CACHE] Mismatch Window IDs")
                return False
                
            # Load Data (Parameters Only)
            print(f"[PR5][CACHE] Loading PR5 params from {cache_path}")
            
            # 1. Window Planes
            # Check format ('window_planes' vs 'planes')
            # The save_cache code uses 'planes' in PR4 but let's check PR5 save code.
            # Assuming typical JSON structure.
            
            # Robust load for planes
            planes_src = data.get('window_planes', data.get('planes', {}))
            for wid_str, pl in planes_src.items():
                wid = int(wid_str)
                if wid in self.window_planes:
                    self.window_planes[wid]['plane_pt'] = np.array(pl['plane_pt'])
                    self.window_planes[wid]['plane_n'] = np.array(pl['plane_n'])
                    if 'd' in pl: # Also support d/alpha/beta if stored
                        pass 
            
            # 2. Window Media
            media_src = data.get('window_media', data.get('media', {}))
            for wid_str, m in media_src.items():
                wid = int(wid_str)
                if wid in self.window_media:
                    # Update thickness
                    if 'thickness' in m:
                        self.window_media[wid]['thickness'] = float(m['thickness'])
                
            # 3. Camera Params
            cam_src = data.get('cam_params', {})
            for cid_str, p in cam_src.items():
                cid = int(cid_str)
                if cid in self.cam_params:
                    self.cam_params[cid] = np.array(p)
                
            # 4. Update C++ objects
            # Use internal helper to push everything to C++
            # 4. Update C++ objects
            for cid in self.active_cam_ids:
                if cid not in self.cams_cpp: continue
                
                # Prepare update arguments
                update_kwargs = {}
                
                # 1. Extrinsics AND Intrinsics
                if cid in self.cam_params:
                    p = self.cam_params[cid]
                    update_kwargs['extrinsics'] = {'rvec': p[0:3], 'tvec': p[3:6]}
                    
                    # [CRITICAL] Pass full intrinsics to ensure update_cpp_camera_state 
                    # doesn't zero them out if C++ state is not fully initialized
                    # p = [rvec(3), tvec(3), f, cx, cy, k1, k2]
                    update_kwargs['intrinsics'] = {
                        'f': p[6],
                        'cx': p[7],
                        'cy': p[8],
                        'dist': [p[9], p[10], 0, 0, 0]
                    }
                
                # 2. Plane Geometry
                wid = self.cam_to_window.get(cid)
                if wid in self.window_planes:
                    pl = self.window_planes[wid]
                    update_kwargs['plane_geom'] = {
                        'pt': pl['plane_pt'].tolist(), 
                        'n': pl['plane_n'].tolist()
                    }
                    if wid in self.window_media:
                        update_kwargs['media_props'] = self.window_media[wid]
                
                if update_kwargs:
                    update_cpp_camera_state(self.cams_cpp[cid], **update_kwargs)
            
            # 5. RE-BUILD OBSERVATION CACHE (Important!)
            # Ensure obs_cache matches the FRESH dataset, not whatever was implicitly loaded
            # or left over. This ensures 'frames' list and 'obsA/B' coincide.
            # self.dataset was UNTOUCHED by this load function.
            self._build_obs_cache()
            
            print(f"[PR5][CACHE] Success. Params loaded, used FREH dataset for obs.")
            return True
            
        except Exception as e:
            print(f"[PR5][CACHE] Load failed: {e}")
            return False

        except Exception as e:
            print(f"[PR5][CACHE] Save failed: {e}")



    def _pack_parameters(self) -> Tuple[np.ndarray, List[Tuple[float, float]], List[str]]:
        """
        Pack optimization parameters into vector x.
        Always packs all parameters (Simplified PR5).
        """
        x = []
        bounds_lower = []
        bounds_upper = []
        names = []
        
        cfg = self.config
        
        # 1. Windows
        for wid in self.window_ids:
            media = self.window_media[wid]

            # Plane (d, alpha, beta)
            x.append(0.0)
            names.append(f"w{wid}_d")
            d_bound = cfg.bounds_d_delta_mm
            bounds_lower.append(-d_bound)
            bounds_upper.append(d_bound)
            
            # alpha, beta
            x.extend([0.0, 0.0])
            names.append(f"w{wid}_a"); names.append(f"w{wid}_b")
            ab_bound = np.radians(cfg.bounds_alpha_beta_deg)
            bounds_lower.extend([-ab_bound, -ab_bound])
            bounds_upper.extend([ab_bound, ab_bound])
            
            # Thickness
            t_curr = media['thickness']
            x.append(t_curr)
            names.append(f"w{wid}_t")
            # Bounds +/- 5%
            t_min = t_curr * (1.0 - cfg.bounds_thick_pct)
            t_max = t_curr * (1.0 + cfg.bounds_thick_pct)
            bounds_lower.append(t_min)
            bounds_upper.append(t_max)

        # 2. Cameras
        for cid in self.active_cam_ids:
            # Focal (f)
            f_val = self.cam_params[cid][6]
            x.append(f_val)
            names.append(f"c{cid}_f")
            # Bounds +/- 2% from absolute initial
            f0 = self.initial_f[cid]
            f_min = f0 * (1.0 - cfg.bounds_f_pct)
            f_max = f0 * (1.0 + cfg.bounds_f_pct)
            bounds_lower.append(f_min)
            bounds_upper.append(f_max)
            
            # Rvec
            r_curr = self.cam_params[cid][0:3]
            x.extend(r_curr)
            names.extend([f"c{cid}_r0", f"c{cid}_r1", f"c{cid}_r2"])
            inf = np.inf
            bounds_lower.extend([-inf, -inf, -inf])
            bounds_upper.extend([inf, inf, inf])
            
            # Tvec
            t_curr = self.cam_params[cid][3:6]
            x.extend(t_curr)
            names.extend([f"c{cid}_tx", f"c{cid}_ty", f"c{cid}_tz"])
            bounds_lower.extend([-inf, -inf, -inf])
            bounds_upper.extend([inf, inf, inf])
            
        return np.array(x), (bounds_lower, bounds_upper), names

    def _unpack_parameters(self, x: np.ndarray) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Unpack x into readable dicts.
        Returns (dt_planes, dict_thick, dict_f, dict_rvec, dict_tvec)
        """
        idx = 0
        
        # Temp storage
        planes_update = {} # {wid: (d_delta, alpha, beta)}
        thick_new = {}
        f_new = {}
        rvec_new = {}
        tvec_new = {}
        
        # 1. Windows (must match pack order exactly)
        for wid in self.window_ids:
            d_delta = x[idx]; idx += 1
            alpha = x[idx]; idx += 1
            beta = x[idx]; idx += 1
            planes_update[wid] = (d_delta, alpha, beta)
            
            thick_new[wid] = x[idx]; idx += 1
            
        # 2. Cameras
        for cid in self.active_cam_ids:
            # F
            f_new[cid] = x[idx]; idx += 1
                
            # Rvec
            rvec_new[cid] = x[idx:idx+3]; idx += 3
                
            # Tvec
            tvec_new[cid] = x[idx:idx+3]; idx += 3
        
        return planes_update, thick_new, f_new, rvec_new, tvec_new


    def _apply_params_to_cpp(self, planes_update, thick_map, f_map, rvec_map, tvec_map):
        """
        Update C++ objects with temporary parameters.
        Returns: updated_planes_dict (for residual calc)
        """
        updated_planes = {}
        
        # Update Windows
        for wid, (d_delta, alpha, beta) in planes_update.items():
            # Update thickness in window_media (not CPP directly? CPP Camera holds thickness in PinPlate)
            # Actually C++ Camera has _pinplate_param.thickness and .plane
            
            # Compute new plane
            pl_init = self.window_planes[wid] # Current base
            n_init = pl_init['plane_n']
            pt_init = pl_init['plane_pt']
            
            # Alpha/Beta update logic
            # Use 'initial_planes' as basis for tangent space or current?
            # PR5 spec: "local incremental parameterization around CURRENT normal"
            # But here x[alpha] is optimized from 0. So it's relative to the start of 'optimize' call (or step).
            # If we call this inside residuals, and x is from solver, it's relative to self.window_planes.
            
            n_new = update_normal_tangent(n_init, alpha, beta)
            
            # d update: pt_new = pt_init + n_new * d_delta
            # PR5 spec: d_delta along n_new
            pt_new = pt_init + n_new * d_delta
            
            # Thickness from thick_map
            t_val = thick_map[wid]
            
            updated_planes[wid] = {
                'plane_pt': pt_new, 
                'plane_n': n_new,
                'initialized': True,
                'thick_mm': t_val
            }

            # Update all cameras attached to this window
            for cid in self.window_to_cams.get(wid, []):
                if cid not in self.cams_cpp: continue
                cam = self.cams_cpp[cid]
                
                # Prepare Args (Explicit Intrinsics to prevent zeroing if C++ state is bad)
                rv = rvec_map[cid]
                tv = tvec_map[cid]
                f = f_map[cid]
                
                # Fetch stable intrinsics from params
                # Note: 'dist' might be in cam_params or initial_cam_params?
                # cam_params: [rvec(3), tvec(3), f, cx, cy, k1, k2]
                cp = self.cam_params[cid]
                intrinsics_full = {
                    'f': f,
                    'cx': cp[7],
                    'cy': cp[8],
                    'dist': [cp[9], cp[10], 0, 0, 0]
                }
                
                update_cpp_camera_state(cam,
                                        extrinsics={'rvec': rv, 'tvec': tv},
                                        intrinsics=intrinsics_full,
                                        plane_geom={
                                            'pt': pt_new.tolist(),
                                            'n': n_new.tolist()
                                        },
                                        media_props={
                                           'thickness': t_val,
                                           # n1/n2/n3 assumed static in window_media (not optimized)
                                           # We can re-fetch them from self.window_media if needed,
                                           # but update_cpp_camera_state defaults are 'None' -> preserve.
                                           # BUT if we initialized from memory, maybe we should pass them?
                                           # Let's trust they are preserved or handled by update logic.
                                           # Re-passing them is safer against corruption.
                                            'n_air': self.window_media[wid].get('n_air', 1.0),
                                            'n_window': self.window_media[wid].get('n_window', 1.49),
                                            'n_object': self.window_media[wid].get('n_object', 1.33)
                                        })
                    
        return updated_planes

    def _residuals_pr5(self, x: np.ndarray, lambda_len: float) -> np.ndarray:
        """
        PR5 Cost Function:
        Ray Dist + Wand Length + Strong Priors
        """
        # Unpack
        planes_scalars, thick_map, f_map, rvec_map, tvec_map = self._unpack_parameters(x)
        
        # Apply to C++ and get updated plane geometry
        planes_up = self._apply_params_to_cpp(planes_scalars, thick_map, f_map, rvec_map, tvec_map)
        
        # 1. Pre-calculate total possible counts for FIXED size
        total_ray_slots = 0
        total_len_slots = len(self.obs_cache)
        total_barrier_slots = 0
        
        for fid in self.obs_cache:
            # Ray slots: count observations in cache
            cids = sorted(self.obs_cache[fid].keys())
            for cid in cids:
                uvA, uvB = self.obs_cache[fid][cid]
                if uvA is not None: total_ray_slots += 1
                if uvB is not None: total_ray_slots += 1
            
            # Barrier slots: each point (A and B) can collide with its window's planes
            for endpoint in ['A', 'B']:
                wids = set()
                for cid in cids:
                    uvA, uvB = self.obs_cache[fid][cid]
                    uv = uvA if endpoint == 'A' else uvB
                    if uv is not None:
                        wid = self.cam_to_window.get(cid)
                        if wid is not None and wid != -1:
                            wids.add(wid)
                total_barrier_slots += 2 * len(wids) # Step + Gradient

        # Camera-side Barrier slots: Each active camera can have one
        total_barrier_slots += 2 * len(self.active_cam_ids)

        # Pre-allocate
        res_ray_fixed = np.zeros(total_ray_slots)
        res_len_fixed = np.zeros(total_len_slots)
        res_barrier_fixed = np.zeros(total_barrier_slots)
        priors = []

        PENALTY_RAY = 100.0
        PENALTY_LEN = self.wand_length
        
        idx_ray = 0
        idx_len = 0
        idx_barrier = 0
        
        S_ray = 0.0
        S_len = 0.0
        N_ray_actual = 0
        N_len_actual = 0
        num_triangulated_points = 0
        
        cfg = self.config
        radius_A = self.dataset.get('est_radius_small_mm', 0.0)
        radius_B = self.dataset.get('est_radius_large_mm', 0.0)
        
        # Track side violations for logging
        all_sX = []
        all_sC = []
        
        # 1. Observations (Ray + Wand)
        for fid in self.obs_cache:
            cids = sorted(self.obs_cache[fid].keys())
            
            # --- Endpoint A ---
            rays_A_all = []
            cids_with_A = []
            for cid in cids:
                uvA, _ = self.obs_cache[fid][cid]
                if uvA is not None:
                    cids_with_A.append(cid)
                    cam = self.cams_cpp[cid]
                    wid = self.cam_to_window.get(cid, -1)
                    rA = build_pinplate_ray_cpp(cam, uvA, cam_id=cid, window_id=wid, frame_id=fid, endpoint="A")
                    rays_A_all.append(rA)

            validA = False
            XA = None
            rays_A_valid = [r for r in rays_A_all if r.valid]
            if len(rays_A_valid) >= 2:
                XA, _, validA, _ = triangulate_point(rays_A_valid)
            
            wids_A_expected = sorted([w for w in set(self.cam_to_window.get(cid) for cid in cids_with_A) if w is not None and w != -1])
            
            if validA:
                num_triangulated_points += 1
                for r in rays_A_all:
                    if r.valid:
                        d = point_to_ray_dist(XA, r.o, r.d)
                        res_ray_fixed[idx_ray] = d / self.sigma_ray_global
                        S_ray += d**2
                        N_ray_actual += 1
                    else:
                        res_ray_fixed[idx_ray] = PENALTY_RAY / self.sigma_ray_global
                    idx_ray += 1
                # Barrier residuals
                for wid in wids_A_expected:
                    if wid in planes_up:
                        pl = planes_up[wid]
                        sX = np.dot(pl['plane_n'], XA - pl['plane_pt'])
                        all_sX.append((sX, fid, wid, 'A', idx_barrier)) # index for reference if needed
                        # Penalty logic follows below in aggregate section
                    idx_barrier += 2
            else:
                for _ in range(len(rays_A_all)):
                    res_ray_fixed[idx_ray] = PENALTY_RAY / self.sigma_ray_global
                    idx_ray += 1
                idx_barrier += 2 * len(wids_A_expected)
            
            # --- Endpoint B ---
            rays_B_all = []
            cids_with_B = []
            for cid in cids:
                _, uvB = self.obs_cache[fid][cid]
                if uvB is not None:
                    cids_with_B.append(cid)
                    cam = self.cams_cpp[cid]
                    wid = self.cam_to_window.get(cid, -1)
                    rB = build_pinplate_ray_cpp(cam, uvB, cam_id=cid, window_id=wid, frame_id=fid, endpoint="B")
                    rays_B_all.append(rB)

            validB = False
            XB = None
            rays_B_valid = [r for r in rays_B_all if r.valid]
            if len(rays_B_valid) >= 2:
                XB, _, validB, _ = triangulate_point(rays_B_valid)
            
            wids_B_expected = sorted([w for w in set(self.cam_to_window.get(cid) for cid in cids_with_B) if w is not None and w != -1])

            if validB:
                num_triangulated_points += 1
                for r in rays_B_all:
                    if r.valid:
                        d = point_to_ray_dist(XB, r.o, r.d)
                        res_ray_fixed[idx_ray] = d / self.sigma_ray_global
                        S_ray += d**2
                        N_ray_actual += 1
                    else:
                        res_ray_fixed[idx_ray] = PENALTY_RAY / self.sigma_ray_global
                    idx_ray += 1
                # Barrier
                for wid in wids_B_expected:
                    if wid in planes_up:
                        pl = planes_up[wid]
                        sX = np.dot(pl['plane_n'], XB - pl['plane_pt'])
                        all_sX.append((sX, fid, wid, 'B', idx_barrier))
                    idx_barrier += 2
            else:
                for _ in range(len(rays_B_all)):
                    res_ray_fixed[idx_ray] = PENALTY_RAY / self.sigma_ray_global
                    idx_ray += 1
                idx_barrier += 2 * len(wids_B_expected)
            
            # --- Wand Length ---
            if validA and validB:
                L = np.linalg.norm(XA - XB)
                err = L - self.wand_length
                res_len_fixed[idx_len] = err / self.sigma_wand
                S_len += err**2
                N_len_actual += 1
            else:
                res_len_fixed[idx_len] = PENALTY_LEN / self.sigma_wand
            idx_len += 1
        
        # Camera-side: Collect signed distances
        for cid in sorted(self.active_cam_ids):
            wid = self.cam_to_window.get(cid)
            if wid not in planes_up: 
                idx_barrier += 2 # Maintain slot
                continue
            
            # Reconstruct C from C++ object
            cam = self.cams_cpp[cid]
            pp = cam._pinplate_param
            R_np = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    R_np[i, j] = pp.r_mtx[i, j]
            t_np = np.array([pp.t_vec[0], pp.t_vec[1], pp.t_vec[2]])
            C = -R_np.T @ t_np
            
            pl = planes_up[wid]
            sC = np.dot(pl['plane_n'], C - pl['plane_pt'])
            all_sC.append((sC, cid, wid, idx_barrier))
            idx_barrier += 2

        # 2. Priors (STABLE ORDER)
        priors = []
        # Plane update priors
        for wid in self.window_ids:
            d_delta, alpha, beta = planes_scalars[wid]
            priors.append(alpha / cfg.sigma_plane_ang)
            priors.append(beta / cfg.sigma_plane_ang)
            priors.append(d_delta / cfg.sigma_d)
            
            # Thickness Prior
            t_ui = self.initial_media[wid]['thickness']
            t_curr = thick_map[wid]
            sigma_t = (cfg.bounds_thick_pct * t_ui) / 3.0 if t_ui > 0 else 1.0
            priors.append((t_curr - t_ui) / sigma_t)
            
        # Camera Priors
        for cid in self.active_cam_ids:
            # F Prior
            f_curr = f_map[cid]
            f_init = self.initial_f[cid]
            sigma_f = (cfg.bounds_f_pct * f_init) / 3.0 if f_init > 0 else 1.0
            priors.append((f_curr - f_init) / sigma_f)
            
            # Rvec/Tvec Priors
            r_curr = rvec_map[cid]
            r_init = self.initial_cam_params[cid][0:3]
            priors.extend((r_curr - r_init) / cfg.sigma_rvec)
            
            t_curr = tvec_map[cid]
            t_init = self.initial_cam_params[cid][3:6]
            priors.extend((t_curr - t_init) / cfg.sigma_tvec)

        # 3. Huber & Aggregate
        def apply_huber(r_array, delta):
            if len(r_array) == 0: return r_array
            u = r_array / delta
            abs_u = np.abs(u)
            w = np.where(abs_u <= 1.0, 1.0, 1.0 / abs_u)
            return np.sqrt(w) * r_array

        # 4. Barrier Penalties & Stats
        # Process Camera-side (sC)
        all_sC_vals = []
        for (sC, cid, wid, idx) in all_sC:
            all_sC_vals.append(sC)
            # Expect sC < 0. Violation if sC >= -margin
            gap = (cfg.margin_side_mm) - (-sC) # sC should be negative. gap > 0 if sC > -margin
            # Simplified: Violation if sC > -margin
            # Let's say sC must be < -10mm. if sC = -5, gap = 5.
            
            # Use PR5 logic: sC < 0 is strict.
            # Barrier at 0.
            val = sC
            if val > -cfg.margin_side_mm:
                # Violation
                # Linear penalty
                res_barrier_fixed[idx] = (val + cfg.margin_side_mm) * 10.0 # Weight?
            
        # Process Point-side (sX)
        all_sX_vals = []
        violations_count = 0
        
        for (sX, fid, wid, endpoint, idx) in all_sX:
            all_sX_vals.append(sX)
            # Expect sX > 0.
            # Barrier at 0.
            val = sX
            if val < cfg.margin_side_mm:
                violations_count += 1
                gap = cfg.margin_side_mm - val
                res_barrier_fixed[idx] = gap * 10.0
                
        # Store stats for logging
        min_sX = np.min(all_sX_vals) if all_sX_vals else 0.0
        min_sC = np.min(all_sC_vals) if all_sC_vals else 0.0 # sC is negative usually
        max_sC = np.max(all_sC_vals) if all_sC_vals else 0.0 # crucial one
        
        self._last_barrier_stats = {
            'min_sX': min_sX,
            'max_sC': max_sC,
            'num_violations': violations_count
        }

        # 5. Store Stats for wrapper
        self._last_s_ray = S_ray
        self._last_s_len = S_len
        self._last_n_ray = N_ray_actual
        self._last_n_len = N_len_actual
        
        # Concatenate all
        # Huber on Ray/Len (Removed per user request -> Linear Loss)
        # res_ray_huber = apply_huber(res_ray_fixed, 2.0)
        # res_len_huber = apply_huber(res_len_fixed, 2.0)
        
        return np.concatenate([
            res_ray_fixed, 
            res_len_fixed,
            res_barrier_fixed, 
            np.array(priors)
        ])
        arr_pri = np.array(priors)
        
        # 4. Barriers Logic
        gate_enabled = getattr(self, '_round_gate_enabled', False)
        J_ref = getattr(self, '_j_ref_for_round', 1.0)
        C_gate = cfg.alpha_side_gate * J_ref
        r_fix_const = np.sqrt(2.0 * C_gate)
        r_grad_const = np.sqrt(2.0 * cfg.beta_side_dir)
        r_soft_const = np.sqrt(2.0 * cfg.beta_side_soft)
        tau = 0.01

        # Point Side Barriers
        for (sX, fid, wid, endpoint, b_idx) in all_sX:
            r_val = radius_A if endpoint == 'A' else radius_B
            gap = (cfg.margin_side_mm + r_val) - sX
            if gap > 0:
                if gate_enabled:
                    res_barrier_fixed[b_idx] = r_fix_const * (1.0 - np.exp(-gap / tau))
                    res_barrier_fixed[b_idx+1] = r_grad_const * gap
                else:
                    res_barrier_fixed[b_idx] = r_soft_const * gap
                    res_barrier_fixed[b_idx+1] = 0.0
            else:
                res_barrier_fixed[b_idx] = 0.0
                res_barrier_fixed[b_idx+1] = 0.0

        # Camera Side Barriers
        for (sC, cid, wid, b_idx) in all_sC:
            gap = sC + cfg.margin_side_mm
            if gap > 0:
                if gate_enabled:
                    res_barrier_fixed[b_idx] = r_fix_const * (1.0 - np.exp(-gap / tau))
                    res_barrier_fixed[b_idx+1] = r_grad_const * gap
                else:
                    res_barrier_fixed[b_idx] = r_soft_const * gap
                    res_barrier_fixed[b_idx+1] = 0.0
            else:
                res_barrier_fixed[b_idx] = 0.0
                res_barrier_fixed[b_idx+1] = 0.0

        # Logging
        self._last_s_ray = S_ray
        self._last_n_ray = N_ray_actual
        self._last_s_len = S_len
        self._last_n_len = N_len_actual
        self._res_call_count_pr5 += 1
        
        if self._res_call_count_pr5 % 20 == 0 and cfg.verbosity >= 1:
            sX_vals = [val[0] for val in all_sX]
            min_sx = min(sX_vals) if sX_vals else 0.0
            print(f"      [Step {self._res_call_count_pr5}] min(sX)={min_sx:.5f} mm")

        # Store current J_data
        J_data_current = self._last_s_ray + lambda_len * self._last_s_len
        self._last_j_data = J_data_current

        # Update RMSE for diagnostics
        self._last_ray_rmse = np.sqrt(S_ray / max(1, N_ray_actual))
        self._last_len_rmse = np.sqrt(S_len / max(1, N_len_actual)) if N_len_actual > 0 else 0.0

        return np.concatenate([arr_ray, arr_len, arr_pri, res_barrier_fixed])

    def save_cache(self, out_path: str, points_3d: list = None):
        """Save PR5 results to cache (with Feasibility Audit)."""
        try:

            Rs = self.dataset.get('est_radius_small_mm', 2.0)
            Rl = self.dataset.get('est_radius_large_mm', 5.0)
            
            min_sX, viols, worst, _, _ = evaluate_plane_side_constraints(
                self.dataset, self.window_planes, self.cam_params, self.cams_cpp,
                self.cam_to_window, Rs, Rl, self.config.margin_side_mm
            )
            
            is_feasible = (len(viols) == 0)
            if not is_feasible:
                print(f"[PR5][WARNING] Final solution is NOT feasible: violations={len(viols)}")
                if worst:
                    print(f"  Worst: {worst['gap']:.4f}mm violation at W{worst['window']} F{worst['frame']}")

            cache_path = str(Path(out_path).parent / "pr5_bundle_cache.json")
            
            data = {
                'timestamp': str(datetime.now()),
                'feasible': is_feasible,
                'min_sX_gap': min_sX,
                'violations_count': len(viols),
                'cam_ids': self.active_cam_ids,
                'window_ids': self.window_ids,
                'planes': {
                    str(w): {
                        'plane_pt': np.asarray(pl['plane_pt']).tolist(),
                        'plane_n': np.asarray(pl['plane_n']).tolist(),
                        'd_delta': pl.get('d_delta', 0.0),
                        'alpha': pl.get('alpha', 0.0),
                        'beta': pl.get('beta', 0.0)
                    } for w, pl in self.window_planes.items()
                },
                'cam_params': {
                    str(c): np.asarray(p).tolist() for c, p in self.cam_params.items()
                },
                'window_media': {
                   str(w): m for w, m in self.window_media.items()
                }
            }
            
            # Add 3D points if provided
            if points_3d is not None:
                data['points_3d'] = points_3d
            
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"[PR5][CACHE] Saved results to {cache_path} (feasible={is_feasible})")
            
        except Exception as e:
            print(f"[PR5][CACHE] Save failed: {e}")



