"""
Refractive Bundle Adjustment Optimizer

This module implements selective bundle adjustment
that refines both window planes AND selected camera extrinsics (when observable).

Key Design Principles:
- Observability-based freezing: N_cam, baseline, view-angle diversity
- Staged optimization: Stage 1 → Stage 2 → Stage 3
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
from pathlib import Path
from datetime import datetime



try:
    import pyopenlpt as lpt
except ImportError:
    lpt = None

from .refractive_geometry import (
    Ray, build_pinplate_ray_cpp, triangulate_point, point_to_ray_dist,
    update_normal_tangent, camera_center, angle_between_vectors,
    update_cpp_camera_state,
    enable_ray_tracking, reset_ray_stats, print_ray_stats_report
)


class RefractiveCalibReporter:
    def section(self, title: str, width: int = 60):
        line = "=" * width
        print(f"\n{line}")
        print(title)
        print(line)

    def header(self, title: str):
        print(f"\n{title}")

    def info(self, message: str):
        print(message)

    def detail(self, message: str):
        print(message)


class CppSyncAdapter:
    @staticmethod
    def build_update_kwargs(
        cam_params: Dict[int, np.ndarray],
        window_planes: Dict[int, Dict],
        window_media: Dict[int, Dict],
        cam_to_window: Dict[int, int],
        cam_id: int,
    ) -> Dict:
        update_kwargs = {}
        if cam_id in cam_params:
            p = cam_params[cam_id]
            update_kwargs['extrinsics'] = {'rvec': p[0:3], 'tvec': p[3:6]}
            update_kwargs['intrinsics'] = {
                'f': p[6],
                'cx': p[7],
                'cy': p[8],
                'dist': [p[9], p[10], 0, 0, 0]
            }

        wid = cam_to_window.get(cam_id)
        if wid in window_planes:
            pl = window_planes[wid]
            update_kwargs['plane_geom'] = {
                'pt': np.asarray(pl['plane_pt'], dtype=float).tolist(),
                'n': np.asarray(pl['plane_n'], dtype=float).tolist()
            }
        if wid in window_media:
            update_kwargs['media_props'] = window_media[wid]

        return update_kwargs

    @staticmethod
    def apply(cams_cpp: Dict[int, 'lpt.Camera'], cam_id: int, update_kwargs: Dict):
        if not update_kwargs:
            return
        update_cpp_camera_state(cams_cpp[cam_id], **update_kwargs)


class CacheStore:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path

    def load(self) -> Optional[Dict]:
        if not Path(self.cache_path).exists():
            return None
        with open(self.cache_path, 'r') as f:
            return json.load(f)

    def save(self, data: Dict):
        with open(self.cache_path, 'w') as f:
            json.dump(data, f, indent=2)


class ObsCacheBuilder:
    @staticmethod
    def build(dataset: Dict, active_cam_ids: List[int]) -> Dict[int, Dict[int, Tuple]]:
        obs_cache = {}
        obsA = dataset.get('obsA', {})
        obsB = dataset.get('obsB', {})
        all_fids = set(obsA.keys()) | set(obsB.keys())

        for fid in all_fids:
            obs_cache[fid] = {}
            for cid in active_cam_ids:
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
                    obs_cache[fid][cid] = (uvA, uvB)

        return obs_cache





@dataclass
class RefractiveBAConfig:
    """Configuration for bundle adjustment."""
    # Regularization
    lambda_reg_plane: float = 10.0  # Normal drift penalty
    lambda_reg_tvec: float = 1.0    # Translation drift penalty
    lambda_reg_rvec: float = 50.0   # Rotation drift penalty (standard)
    lambda_reg_f: float = 10.0      # Focal drift penalty
    lambda_reg_thick: float = 10.0  # Thickness drift penalty
    lambda_reg_dist: float = 10.0   # Distortion drift penalty
    use_regularization: bool = False
    
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
    skip_optimization: bool = False
    stage: int = 4
    verbosity: int = 1
    margin_side_mm: float = 0.05    # Margin for side constraint (mm)
    alpha_side_gate: float = 10.0   # Gate magnitude: C_gate = alpha * J_ref
    beta_side_dir: float = 1e4      # Directional weight when gate is active
    beta_side_soft: float = 100.0   # Soft floor weight when gate is NOT active (defaults to ON)

    # Bounds for Round 4 refinement
    bounds_thick_pct: float = 0.05
    bounds_f_pct: float = 0.05
    bounds_dist_abs: float = 0.2

    # Distortion optimization
    dist_coeff_num: int = 0

    # Explicit point solver
    explicit_point_min_rays: int = 3

    # Robust loss settings
    loss_plane: str = "linear"
    loss_cam: str = "soft_l1"
    loss_joint: str = "huber"
    loss_round4: str = "huber"
    loss_f_scale: float = 1.0

    # Weak-window control
    skip_weak_round3: bool = True
    skip_weak_round4: bool = True

    # Finite difference step sizes (per-parameter)
    diff_step_rvec: float = 1e-4
    diff_step_tvec: float = 1e-2
    diff_step_plane_d: float = 1e-2
    diff_step_plane_ang: float = 1e-4
    diff_step_f: float = 1e-1
    diff_step_thick: float = 1e-3
    diff_step_k: float = 1e-6
    diff_step_default: float = 1e-4
    
    
class RefractiveBAOptimizer:
    """
    Bundle Adjustment Optimizer.

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
                 config: Optional[RefractiveBAConfig] = None,
                 progress_callback: Optional[callable] = None):
        """
        Initialize bundle adjustment optimizer.
        
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
        self.window_media = {int(w): m.copy() for w, m in window_media.items()}
        self.wand_length = wand_length
        self.config = config or RefractiveBAConfig()
        self.progress_callback = progress_callback  # For UI progress updates
        self.reporter = RefractiveCalibReporter()

        self._explicit_points = {}

        
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
        self.initial_media = {w: m.copy() for w, m in self.window_media.items()}
        self.initial_f = {cid: float(p[6]) for cid, p in self.cam_params.items() if len(p) > 6}
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
        self.obs_cache = ObsCacheBuilder.build(self.dataset, self.active_cam_ids)
        
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
        self.initial_media = {w: m.copy() for w, m in self.window_media.items()}
        self.initial_f = {cid: float(p[6]) for cid, p in self.cam_params.items() if len(p) > 6}
        
        self._last_ray_rmse = -1.0
        self._last_len_rmse = -1.0
        
        self.sigma_ray_global = 0.04  # Default, will be recalculated
        self.sigma_wand = 0.1        # Default, will be recalculated
    
    
    def _build_obs_cache(self):
        """Build observation cache from dataset."""
        self.obs_cache = ObsCacheBuilder.build(self.dataset, self.active_cam_ids)
    

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
            self.reporter.info(
                f"Unit Normalization: sigma_ray={self.sigma_ray_global:.4f}mm "
                f"({cfg.px_target}px at Z={avg_dist_z:.1f}mm), "
                f"sigma_wand={self.sigma_wand:.4f}mm ({cfg.wand_tol_pct*100}%)"
            )

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

    def evaluate_residuals(self, planes: Dict[int, Dict], cam_params: Dict[int, np.ndarray],
                           lambda_eff: float, window_media: Optional[Dict[int, Dict]] = None) -> Tuple[np.ndarray, float, float, int, int]:
        """
        Evaluate residuals with fixed-size padding for Scipy least_squares compatibility.
        """
        # Apply planes and extrinsics
        # Apply to C++ objects (Consolidated Update)
        media = window_media or self.window_media

        for cid in self.active_cam_ids:
            if cid not in self.cams_cpp: continue

            update_kwargs = CppSyncAdapter.build_update_kwargs(
                cam_params=cam_params,
                window_planes=planes,
                window_media=media,
                cam_to_window=self.cam_to_window,
                cam_id=cid,
            )
            CppSyncAdapter.apply(self.cams_cpp, cid, update_kwargs)

        self._explicit_points = {}
        
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

        explicit_total = 0
        explicit_ok = 0
        fallback_used = 0
        fallback_ok = 0
        
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

            XA_explicit, okA, _ = self._compute_explicit_point(rays_A_all)

            if n_slots_A > 0:
                explicit_total += 1
                if okA:
                    explicit_ok += 1

            validA = False
            XA = None
            rays_A_valid = [r for r in rays_A_all if r.valid]
            if okA and XA_explicit is not None:
                XA = XA_explicit
                validA = True
            elif len(rays_A_valid) >= 2:
                XA, _, validA, _ = triangulate_point(rays_A_valid)
                if n_slots_A > 0:
                    fallback_used += 1
                    if validA:
                        fallback_ok += 1
            
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

            XB_explicit, okB, _ = self._compute_explicit_point(rays_B_all)

            if n_slots_B > 0:
                explicit_total += 1
                if okB:
                    explicit_ok += 1
            
            validB = False
            XB = None
            rays_B_valid = [r for r in rays_B_all if r.valid]
            if okB and XB_explicit is not None:
                XB = XB_explicit
                validB = True
            elif len(rays_B_valid) >= 2:
                XB, _, validB, _ = triangulate_point(rays_B_valid)
                if n_slots_B > 0:
                    fallback_used += 1
                    if validB:
                        fallback_ok += 1
            
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

            self._explicit_points[fid] = {
                'A': XA_explicit,
                'B': XB_explicit,
                'validA': okA,
                'validB': okB
            }

        # 2. Side Barrier (Adaptive)
        J_data = S_ray + lambda_eff * S_len
        margin_mm = self.config.margin_side_mm
        sX_vals = []
        
        # Hard Barrier Constants (PR5-style)
        C_gate = self.config.alpha_side_gate * self._j_ref
        r_fix_const = np.sqrt(2.0 * C_gate)
        r_grad_const = np.sqrt(2.0 * self.config.beta_side_dir)
        tau = 0.01

        violations_count = 0
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
                    violations_count += 1
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
                'ratio': np.sum(res_barrier_fixed**2) / max(1e-9, J_data),
                'violations': violations_count
            }
        else:
            self._last_barrier_stats = {}

        if explicit_total > 0:
            self._last_explicit_stats = {
                'total': explicit_total,
                'explicit_ok': explicit_ok,
                'fallback_used': fallback_used,
                'fallback_ok': fallback_ok
            }
        else:
            self._last_explicit_stats = {}

        # Update RMSE for diagnostics based on physical units
        self._last_ray_rmse = np.sqrt(S_ray / max(1, N_ray_actual))
        self._last_len_rmse = np.sqrt(S_len / max(1, N_len_actual)) if N_len_actual > 0 else 0.0

        # 3. Combine
        weighted_len = np.sqrt(lambda_eff) * res_len_fixed
        residuals = np.concatenate([res_ray_fixed, weighted_len, res_barrier_fixed])
            
        return residuals, S_ray, S_len, N_ray_actual, N_len_actual

    def _compute_explicit_point(self, rays_list: List[Ray]) -> Tuple[Optional[np.ndarray], bool, str]:
        valid_rays = [r for r in rays_list if r.valid]
        min_rays = max(1, int(self.config.explicit_point_min_rays))
        if len(valid_rays) < min_rays:
            return None, False, "insufficient_valid_rays"

        A = np.zeros((3, 3))
        b = np.zeros(3)
        I = np.eye(3)

        for ray in valid_rays:
            d = ray.d.reshape(3, 1)
            proj_perp = I - d @ d.T
            A += proj_perp
            b += proj_perp @ ray.o

        try:
            cond = np.linalg.cond(A)
            if not np.isfinite(cond):
                return None, False, "ill_conditioned"
            if cond < 1e12:
                X = np.linalg.solve(A, b)
                return X, True, ""
            X, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            return X, False, f"ill_conditioned: cond={cond:.2e}, rank={rank}"
        except (np.linalg.LinAlgError, ValueError) as e:
            return None, False, f"linalg_error: {e}"

    def _get_param_layout(self, enable_planes: bool, enable_cam_t: bool, enable_cam_r: bool,
                          enable_cam_f: bool = False, enable_win_t: bool = False,
                          enable_cam_k1: bool = False, enable_cam_k2: bool = False) -> List[Tuple]:
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
                if enable_win_t:
                    layout.append(('win_t', wid, 0))
        
        # 2. Cameras
        if enable_cam_t or enable_cam_r or enable_cam_f or enable_cam_k1 or enable_cam_k2:
            for cid in self.active_cam_ids:
                if enable_cam_f:
                    layout.append(('cam_f', cid, 0))
                if enable_cam_k1:
                    layout.append(('cam_k1', cid, 0))
                if enable_cam_k2:
                    layout.append(('cam_k2', cid, 0))
                if enable_cam_t:
                    layout.append(('cam_t', cid, 0)) # tx
                    layout.append(('cam_t', cid, 1)) # ty
                    layout.append(('cam_t', cid, 2)) # tz
                
                if enable_cam_r:
                    layout.append(('cam_r', cid, 0)) # rx
                    layout.append(('cam_r', cid, 1)) # ry
                    layout.append(('cam_r', cid, 2)) # rz
        
        return layout
    


    def _unpack_params_delta(self, x: np.ndarray, layout: List[Tuple]) -> Tuple[Dict, Dict, Dict]:
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
        current_media = {w: m.copy() for w, m in self.initial_media.items()}
        
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
        
        plane_deltas = {wid: {'d': 0.0, 'a': 0.0, 'b': 0.0, 't': 0.0} for wid in self.window_ids}
        cam_deltas = {
            cid: {'t': np.zeros(3), 'r': np.zeros(3), 'f': 0.0, 'k1': 0.0, 'k2': 0.0}
            for cid in self.active_cam_ids
        }
        
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
            elif ptype == 'win_t':
                plane_deltas[pid]['t'] = val
            elif ptype == 'cam_f':
                cam_deltas[pid]['f'] = val
            elif ptype == 'cam_t':
                cam_deltas[pid]['t'][subidx] = val
            elif ptype == 'cam_r':
                cam_deltas[pid]['r'][subidx] = val
            elif ptype == 'cam_k1':
                cam_deltas[pid]['k1'] = val
            elif ptype == 'cam_k2':
                cam_deltas[pid]['k2'] = val
        
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

            # Thickness delta (if present in media)
            if wid in current_media and 'thickness' in current_media[wid]:
                current_media[wid]['thickness'] = float(current_media[wid]['thickness']) + deltas['t']
            
        # Apply Camera Deltas
        for cid, deltas in cam_deltas.items():
            if cid not in current_cam_params: continue
            
            # Apply tvec delta
            current_cam_params[cid][3:6] += deltas['t']

            # Apply focal delta
            if len(current_cam_params[cid]) > 6:
                current_cam_params[cid][6] = float(current_cam_params[cid][6]) + deltas['f']

            # Apply distortion deltas
            if len(current_cam_params[cid]) > 9:
                current_cam_params[cid][9] = float(current_cam_params[cid][9]) + deltas['k1']
            if len(current_cam_params[cid]) > 10:
                current_cam_params[cid][10] = float(current_cam_params[cid][10]) + deltas['k2']
            
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
            
        return current_planes, current_cam_params, current_media

    def _residuals(self, x: np.ndarray, layout: List[Tuple], mode: str, lambda_eff: float) -> np.ndarray:
        """Residual function for generic optimization."""
        # Unpack
        curr_planes, curr_cams, curr_media = self._unpack_params_delta(x, layout)
        
        # Data Residuals
        # Note: evaluate_residuals handles applying to CPP internally
        residuals, S_ray, S_len, N_ray, N_len = self.evaluate_residuals(curr_planes, curr_cams, lambda_eff, window_media=curr_media)
        
        # [Fix] Update live stats for logging
        self._last_ray_rmse = np.sqrt(S_ray / max(N_ray, 1))
        self._last_len_rmse = np.sqrt(S_len / max(N_len, 1)) if N_len > 0 else 0.0
        
        # Regularization
        cfg = self.config
        if cfg.use_regularization:
            reg_residuals = []

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
                elif ptype == 'cam_f':
                    reg_residuals.append(val * np.sqrt(cfg.lambda_reg_f))
                elif ptype == 'cam_k1' or ptype == 'cam_k2':
                    reg_residuals.append(val * np.sqrt(cfg.lambda_reg_dist))
                elif ptype == 'win_t':
                    reg_residuals.append(val * np.sqrt(cfg.lambda_reg_thick))

            if len(reg_residuals) > 0:
                residuals = np.concatenate([residuals, np.array(reg_residuals)])

        return np.array(residuals)

    def _optimize_generic(self, mode: str, description: str, 
                          enable_planes: bool, enable_cam_t: bool, enable_cam_r: bool,
                          limit_rot_rad: float, limit_trans_mm: float, 
                          limit_plane_d_mm: float, limit_plane_angle_rad: float,
                          enable_cam_f: bool = False, enable_win_t: bool = False,
                          enable_cam_k1: bool = False, enable_cam_k2: bool = False,
                          plane_d_bounds: Dict[int, float] = None,
                          ftol: float = 1e-6,
                          loss: Optional[str] = None,
                          f_scale: Optional[float] = None):
        """
        Generic optimization loop with explicit bounds and parameter selection.
        """
        layout = self._get_param_layout(
            enable_planes, enable_cam_t, enable_cam_r,
            enable_cam_f, enable_win_t, enable_cam_k1, enable_cam_k2
        )
        
        if not layout:
            print(f"  [{description}] No parameters to optimize.")
            return

        x0 = np.zeros(len(layout), dtype=np.float64)
        cfg = self.config

        diff_step = []
        for (ptype, _, _) in layout:
            if ptype == 'cam_r':
                diff_step.append(cfg.diff_step_rvec)
            elif ptype == 'cam_t':
                diff_step.append(cfg.diff_step_tvec)
            elif ptype == 'plane_d':
                diff_step.append(cfg.diff_step_plane_d)
            elif ptype == 'plane_a' or ptype == 'plane_b':
                diff_step.append(cfg.diff_step_plane_ang)
            elif ptype == 'cam_f':
                diff_step.append(cfg.diff_step_f)
            elif ptype == 'win_t':
                diff_step.append(cfg.diff_step_thick)
            elif ptype == 'cam_k1' or ptype == 'cam_k2':
                diff_step.append(cfg.diff_step_k)
            else:
                diff_step.append(cfg.diff_step_default)

        diff_step = np.array(diff_step, dtype=np.float64)
        if cfg.verbosity >= 1:
            self.reporter.detail(
                f"  diff_step: rvec={cfg.diff_step_rvec:g}, tvec={cfg.diff_step_tvec:g}, "
                f"plane_d={cfg.diff_step_plane_d:g}, plane_ang={cfg.diff_step_plane_ang:g}, "
                f"f={cfg.diff_step_f:g}, thick={cfg.diff_step_thick:g}, k={cfg.diff_step_k:g}"
            )
        
        self.reporter.detail(f"  [{description}] optimizing {len(x0)} parameters ({len(layout)//3} blocks)...")
        # Calc initial RMSE for rollback reference
        planes0, cams0, media0 = self._unpack_params_delta(x0, layout)
        
        # [USER REQUEST] Fixed Weighting Strategy
        # Lambda = 2.0 * N_active_cams
        n_cams = max(1, len(self.active_cam_ids))
        lambda_fixed = 2.0 * n_cams
        
        # Initial evaluation
        _, S_ray0, S_len0, N_ray, N_len = self.evaluate_residuals(planes0, cams0, lambda_fixed, window_media=media0)
        
        # [Constraint] Force lambda=1 if S_ray / S_len < 10 (User Request)
        if S_len0 > 1e-9:
            ratio_s = S_ray0 / S_len0
            if ratio_s < 10.0:
                self.reporter.detail(
                    f"    [Constraint] S_ray/S_len ({ratio_s:.2f}) < 10. "
                    f"Forcing lambda=1.0 (was {lambda_fixed:.1f})"
                )
                lambda_fixed = 1.0
        
        rmse_ray0 = np.sqrt(S_ray0 / max(N_ray, 1))
        rmse_len0 = np.sqrt(S_len0 / max(N_len, 1)) if N_len > 0 else 0.0
        
        # Initial normalized cost J0
        J0 = (rmse_ray0**2) + lambda_fixed * (rmse_len0**2)
        self._j_ref = J0 if J0 > 1e-6 else 1.0
        
        self.reporter.detail(f"    Global Fixed Weighting: lambda={lambda_fixed:.1f} (N_cams={n_cams})")
        self.reporter.detail(f"    Initial: S_ray={S_ray0:.2f}, S_len={S_len0:.2f} (J0={J0:.4f})")
        
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
            elif ptype == 'win_t':
                t0 = self.initial_media.get(pid, {}).get('thickness', 0.0)
                limit = abs(t0) * self.config.bounds_thick_pct
                lb.append(-limit)
                ub.append(limit)
            elif ptype == 'cam_r':
                lb.append(-limit_rot_rad)
                ub.append(limit_rot_rad)
            elif ptype == 'cam_t':
                lb.append(-limit_trans_mm)
                ub.append(limit_trans_mm)
            elif ptype == 'cam_f':
                f0 = self.initial_f.get(pid, self.cam_params.get(pid, [0,0,0,0,0,0,0])[6] if pid in self.cam_params else 0.0)
                limit = abs(f0) * self.config.bounds_f_pct
                lb.append(-limit)
                ub.append(limit)
            elif ptype == 'cam_k1' or ptype == 'cam_k2':
                limit = self.config.bounds_dist_abs
                lb.append(-limit)
                ub.append(limit)
            else:
                lb.append(-1.0)
                ub.append(1.0)
        
        bounds = (np.array(lb), np.array(ub))
        
        # Residual wrapper for event pumping
        self._res_call_count = 0
        def residuals_wrapper(x, *args, **kwargs):
            res = self._residuals(x, *args, **kwargs)
            self._res_call_count += 1
            if self.progress_callback and self._res_call_count % 30 == 0:
                try:
                    c_approx = 0.5 * np.sum(res**2)

                    # [DEBUG] Print to terminal instead of UI
                    if hasattr(self, '_last_ratio_info'):
                        # Calculate percentage
                        j_ratio = getattr(self, '_last_ratio_cost', 0.0)
                        pct = (j_ratio / c_approx * 100) if c_approx > 0 else 0
                        self.reporter.detail(
                            f"  [RefractiveCalib DEBUG] J_tot={c_approx:.1e}, J_ratio={j_ratio:.1e} "
                            f"({pct:.1f}%) | {self._last_ratio_info}"
                        )

                    if self.progress_callback:
                        self.progress_callback(
                            f"{description}", self._last_ray_rmse, self._last_len_rmse, c_approx
                        )
                except Exception as e:
                    self.reporter.detail(f"[Warning] Progress callback failed: {e}")
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
            loss=loss or 'linear',
            f_scale=self.config.loss_f_scale if f_scale is None else f_scale,
            xtol=1e-6,
            gtol=1e-6,
            max_nfev=50,
            diff_step=diff_step
        )

        if res is not None:
            status = getattr(res, 'status', None)
            message = getattr(res, 'message', '')
            nfev = getattr(res, 'nfev', None)
            reason_map = {
                1: 'gtol',
                2: 'ftol',
                3: 'xtol',
                4: 'ftol+xtol'
            }
            reason = reason_map.get(status, f'status_{status}')
            nfev_str = f"{nfev}" if nfev is not None else "?"
            self.reporter.detail(f"  Termination: {reason} (nfev={nfev_str})")
            if message:
                self.reporter.detail(f"  Message: {message}")

            if layout and hasattr(res, 'x'):
                cam_deltas = {}
                plane_deltas = {}

                for (ptype, pid, subidx), val in zip(layout, res.x):
                    if ptype == 'cam_r':
                        cam_deltas.setdefault(pid, {}).setdefault('r', np.zeros(3))
                        cam_deltas[pid]['r'][subidx] = val
                    elif ptype == 'cam_t':
                        cam_deltas.setdefault(pid, {}).setdefault('t', np.zeros(3))
                        cam_deltas[pid]['t'][subidx] = val
                    elif ptype == 'cam_f':
                        cam_deltas.setdefault(pid, {})['f'] = val
                    elif ptype == 'cam_k1':
                        cam_deltas.setdefault(pid, {})['k1'] = val
                    elif ptype == 'cam_k2':
                        cam_deltas.setdefault(pid, {})['k2'] = val
                    elif ptype == 'win_t':
                        plane_deltas.setdefault(pid, {})['t'] = val
                    elif ptype == 'plane_d':
                        plane_deltas.setdefault(pid, {})['d'] = val
                    elif ptype == 'plane_a':
                        plane_deltas.setdefault(pid, {})['a'] = val
                    elif ptype == 'plane_b':
                        plane_deltas.setdefault(pid, {})['b'] = val

                if cam_deltas:
                    self.reporter.detail("  Camera Δ:")
                    for cid in sorted(cam_deltas.keys()):
                        cd = cam_deltas[cid]
                        parts = []
                        if 'r' in cd:
                            rot_deg = np.linalg.norm(cd['r']) * 180.0 / np.pi
                            parts.append(f"Δr={rot_deg:.3f}deg")
                        if 't' in cd:
                            trans = np.linalg.norm(cd['t'])
                            parts.append(f"Δt={trans:.3f}mm")
                        if 'f' in cd:
                            parts.append(f"Δf={cd['f']:.4f}")
                        if 'k1' in cd:
                            parts.append(f"Δk1={cd['k1']:.6f}")
                        if 'k2' in cd:
                            parts.append(f"Δk2={cd['k2']:.6f}")
                        if parts:
                            self.reporter.detail(f"    Cam {cid}: " + ", ".join(parts))

                if plane_deltas:
                    self.reporter.detail("  Window Δ:")
                    for wid in sorted(plane_deltas.keys()):
                        pd = plane_deltas[wid]
                        parts = []
                        if 'd' in pd:
                            parts.append(f"Δd={pd['d']:.4f}mm")
                        if 'a' in pd:
                            parts.append(f"Δa={pd['a'] * 180.0 / np.pi:.4f}deg")
                        if 'b' in pd:
                            parts.append(f"Δb={pd['b'] * 180.0 / np.pi:.4f}deg")
                        if 't' in pd:
                            parts.append(f"Δt={pd['t']:.4f}mm")
                        if parts:
                            self.reporter.detail(f"    Win {wid}: " + ", ".join(parts))

        # Print Barrier Stats
        if cfg.verbosity >= 1 and hasattr(self, '_last_barrier_stats') and self._last_barrier_stats:
            s = self._last_barrier_stats
            self.reporter.detail(f"    [RefractiveCalib][SIDE-BARRIER] min(sX)={s['min_sX']:.4f}mm, near(<20um)={s['pct_near']:.1f}%, vio={s.get('violations', 0)}, cost/J={s['ratio']:.1e}")

        # Final evaluation
        planes_final, cams_final, media_final = self._unpack_params_delta(res.x, layout)
        _, S_rayF, S_lenF, _, _ = self.evaluate_residuals(planes_final, cams_final, lambda_fixed, window_media=media_final)
        
        rmse_rayF = np.sqrt(S_rayF / max(N_ray, 1))
        rmse_lenF = np.sqrt(S_lenF / max(N_len, 1)) if N_len > 0 else 0.0
        JF = (rmse_rayF**2) + lambda_fixed * (rmse_lenF**2)
        
        self.reporter.detail(f"    Final:   S_ray={S_rayF:.2f}, S_len={S_lenF:.2f} (JF={JF:.4f})")
        self.reporter.detail(f"      RMSE Ray: {rmse_ray0:.4f} -> {rmse_rayF:.4f}")
        self.reporter.detail(f"      RMSE Len: {rmse_len0:.4f} -> {rmse_lenF:.4f}")
        
        # Rollback check if degraded (Safety)
        # However, pure geometric optimization shouldn't degrade unless local minima.
        # We accept result.
        
        # Update Initial State for next stage
        self.initial_planes = planes_final
        self.initial_cam_params = cams_final
        self.initial_media = media_final
        
        # Update Public State
        self.window_planes = planes_final
        self.cam_params = cams_final
        self.window_media = media_final
        
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
        
        self.reporter.header("Detecting Weak Windows (Dist-Ratio Constraint)")
        
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

    def optimize(self, skip_optimization: bool = False, stage: Optional[int] = None) -> Tuple[Dict[int, Dict], Dict[int, np.ndarray]]:
        """
        Execute bundle adjustment (Alternating Refinement).
        
        Strategy:
        1. Alternating Loop (Max 6 iterations):
           - A: Optimize Planes (Fixed Cams). Bounds: Angle +/- 2.5 deg.
           - B: Optimize Cams (Fixed Planes). Bounds: Free.
           - Check: If Plane optimization (A) did NOT hit angle boundary, terminate loop early.
        2. Final Joint Optimization (Round 3).
        """
        self._compute_physical_sigmas()
        if stage is None:
            stage = self.config.stage
        
        # [MOVED per user request] Weak window detection now inside loop.
        # self._detect_weak_windows()
        
        # [NEW] Persistent store for geometric init state (d_min)
        self._weak_window_refs = {}
        
        enable_ray_tracking(True, reset=True)
        self.reporter.section(f"Bundle Adjustment Start ({len(self.active_cam_ids)} cameras, {len(self.window_ids)} windows)")
        for wid, pl in sorted(self.window_planes.items()):
            pt = pl['plane_pt']
            n = pl['plane_n']
            self.reporter.detail(f"  [INIT] Win {wid}: pt=[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}], n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")

        if skip_optimization or self.config.skip_optimization:
            self.reporter.info("Skipped (config.skip_optimization=True).")
            return self.window_planes, self.cam_params

        # --- Alternating Loop ---
        max_loop_iters = 6
        loop_iter = 0
        hit_boundary = True # Assume hit to start
        
        self.reporter.section(f"Alternating Loop (Max {max_loop_iters} passes)")
        
        while loop_iter < max_loop_iters:
            loop_iter += 1

            loss_loop = 'linear' if loop_iter == 1 else 'soft_l1'
            
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
                self.reporter.detail(f"  [LOOP {loop_iter}] Bounds Configuration:")
                print(f"    Global Plane Angle: +/- 2.5 deg")
                
                factor = 0.1 * (0.5 ** (loop_iter - 1))
                for wid, info in self._weak_windows.items():
                    d_ref = info.get('d_min_ref', 0.0)
                    if d_ref > 0:
                        limit = d_ref * factor
                        plane_d_bounds[wid] = limit
                        print(f"    [WEAK BOUNDS] Win {wid}: +/- {limit:.2f} mm ({factor*100:.2f}% of {d_ref:.1f}mm)")
            
            self.reporter.header(f"Loop {loop_iter} - Step A: Optimize Planes (Bounds: +/- 2.5 deg)")
            self._print_plane_diagnostics(f"Pre-Loop {loop_iter} Planes")
            
            # Step A: Optimize Planes (Fixed Cams) - Strict Angle Bound (2.5 deg)
            # Bounds: Angle +/- 2.5 deg, Distance +/- 500mm (effectively free)
            limit_angle_rad = np.radians(2.5)
            b_plane_strict = (limit_angle_rad, 500.0)
            
            res_planes, layout_planes = self._optimize_generic(
                mode=f'loop_{loop_iter}_planes', 
                description=f"Adjusting plane parameters ...",
                enable_planes=True,
                enable_cam_t=False,
                enable_cam_r=False,
                limit_rot_rad=0.0,
                limit_trans_mm=0.0,
                limit_plane_d_mm=b_plane_strict[1],
                limit_plane_angle_rad=b_plane_strict[0],
                plane_d_bounds=plane_d_bounds, # [NEW] Pass bounds
                ftol=5e-4,
                loss=loss_loop
            )
            self._print_plane_diagnostics(f"Loop {loop_iter} Planes")
            
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
                self.reporter.detail(f"  [LOOP {loop_iter}] Plane constraints ACTIVE (hit 2.5 deg bound). Continuing loop.")
            else:
                self.reporter.detail(f"  [LOOP {loop_iter}] Plane constraints INACTIVE (all within 2.5 deg). Loop condition satisfied.")

            # Step B: Optimize Cameras (Fixed Planes) - Free Bounds
            self.reporter.header(f"Loop {loop_iter} - Step B: Optimize Cameras (Free Bounds)")
            b_cam_free = (np.deg2rad(180.0), 2000.0)
            
            self._optimize_generic(
                mode=f'loop_{loop_iter}_cams', 
                description=f"Optimizing camera extrinsic parameters ...",
                enable_planes=False,
                enable_cam_t=True,
                enable_cam_r=True,
                limit_rot_rad=b_cam_free[0],
                limit_trans_mm=b_cam_free[1],
                limit_plane_d_mm=0.0,
                limit_plane_angle_rad=0.0,
                ftol=5e-4,
                loss=loss_loop
            )
            self._print_plane_diagnostics(f"Loop {loop_iter} Cams")

            # Check termination
            if not hit_boundary:
                self.reporter.info(f"Converged early at Loop {loop_iter} (Planes inside 2.5 deg). Stopping loop.")
                break
        
        if hit_boundary and loop_iter == max_loop_iters:
             self.reporter.info(f"Loop reached max iterations ({max_loop_iters}). Proceeding to Joint.")

        # --- Final Joint Optimization (Round 3) ---
        if stage >= 3:
            self.reporter.section("Final Joint Optimization (Round 3 Rules)")
            
            # [NEW] Re-detect before final joint (optional)
            if not self.config.skip_weak_round3:
                self._detect_weak_windows()
            # Bounds: 20 deg, 50 mm d, 10 deg plane_ang, 50 mm tvec
            limit_rvec = np.radians(20.0)
            limit_plane_d = 50.0
            limit_plane_ang = np.radians(10.0)
            limit_tvec = 50.0
            
            print("  Bounds: rvec < 20deg, plane_d < 50mm, plane_ang < 10deg, tvec < 50mm")
            
            self._optimize_generic(
                mode='final_joint', 
                description="Optimizing plane and camera extrinsic parameters ...",
                enable_planes=True,
                enable_cam_t=True,
                enable_cam_r=True,
                limit_rot_rad=limit_rvec,
                limit_trans_mm=limit_tvec,
                limit_plane_d_mm=limit_plane_d,
                limit_plane_angle_rad=limit_plane_ang,
                ftol=5e-4,
                loss=self.config.loss_joint
            )
            self._print_plane_diagnostics("Final Joint End")

        # --- Round 4: Joint + Intrinsics + Thickness ---
        if stage >= 4:
            self.reporter.section("Round 4: Joint Optimization + Intrinsics/Thickness")
            limit_rvec = np.radians(20.0)
            limit_plane_d = 10.0
            limit_plane_ang = np.radians(5.0)
            limit_tvec = 50.0

            print(
                f"  Bounds: rvec < 20deg, plane_d < 10mm, plane_ang < 5deg, tvec < 50mm, "
                f"f/thickness within {self.config.bounds_f_pct*100:.1f}%/{self.config.bounds_thick_pct*100:.1f}%"
            )

            enable_k1 = self.config.dist_coeff_num >= 1
            enable_k2 = self.config.dist_coeff_num >= 2
            if enable_k1 or enable_k2:
                print(f"  Distortion: optimize k1={enable_k1}, k2={enable_k2} (|k| <= {self.config.bounds_dist_abs:.3f})")

            self._optimize_generic(
                mode='round4_full',
                description="Optimizing plane and all camera parameters ...",
                enable_planes=True,
                enable_cam_t=True,
                enable_cam_r=True,
                enable_cam_f=True,
                enable_win_t=True,
                enable_cam_k1=enable_k1,
                enable_cam_k2=enable_k2,
                limit_rot_rad=limit_rvec,
                limit_trans_mm=limit_tvec,
                limit_plane_d_mm=limit_plane_d,
                limit_plane_angle_rad=limit_plane_ang,
                ftol=1e-5,
                loss=self.config.loss_round4
            )
            self._print_plane_diagnostics("Round4 End")
        
        # Explicit sync call to be safe for returning
        n_cams = max(1, len(self.active_cam_ids))
        lambda_fixed = 2.0 * n_cams
        self.evaluate_residuals(self.window_planes, self.cam_params, lambda_fixed, window_media=self.window_media)

        self.print_diagnostics()
        self.reporter.section("Bundle Adjustment Complete")
        print_ray_stats_report("Bundle")
        enable_ray_tracking(False)
        
        return self.window_planes, self.cam_params

    
    def print_diagnostics(self):
        """Print comprehensive diagnostics after optimization."""
        self.reporter.section("Final Diagnostics")
        
        # Evaluate final residuals
        n_cams = max(1, len(self.active_cam_ids))
        lambda_fixed = 2.0 * n_cams
        residuals, S_ray, S_len, N_ray, N_len = self.evaluate_residuals(
            self.window_planes, self.cam_params, lambda_fixed, window_media=self.window_media
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
        
        self.reporter.detail("-" * 40)

    def _get_cache_path(self, dataset_path: str) -> str:
        """Get path to cache file."""
        return str(Path(dataset_path).parent / "bundle_cache.json")

    def try_load_cache(self, out_path: str) -> bool:
        """
        Try to load results from cache.
        Returns True if loaded successfully.
        """
        cache_path = self._get_cache_path(out_path)
        store = CacheStore(cache_path)
        data = store.load()
        if not data:
            return False

        try:
            cached_cams = data.get('cam_params', {})
            for cid_str, p_list in cached_cams.items():
                cid = int(cid_str)
                if cid in self.cam_params:
                    self.cam_params[cid] = np.array(p_list)

            planes_data = data.get('planes', {})
            for wid_str, pl in planes_data.items():
                wid = int(wid_str)
                if wid in self.window_planes:
                    self.window_planes[wid]['plane_pt'] = np.array(pl['plane_pt'])
                    self.window_planes[wid]['plane_n'] = np.array(pl['plane_n'])

            media_data = data.get('window_media', {})
            for wid_str, media in media_data.items():
                wid = int(wid_str)
                if wid in self.window_media and isinstance(media, dict):
                    self.window_media[wid].update(media)

            for cid in self.active_cam_ids:
                if cid not in self.cams_cpp:
                    continue
                update_kwargs = CppSyncAdapter.build_update_kwargs(
                    cam_params=self.cam_params,
                    window_planes=self.window_planes,
                    window_media=self.window_media,
                    cam_to_window=self.cam_to_window,
                    cam_id=cid,
                )
                CppSyncAdapter.apply(self.cams_cpp, cid, update_kwargs)

            self.reporter.info(f"Cache loaded: {cache_path}")
            self.reporter.detail("  Note: Using cached parameters with FRESH dataset observations.")
            return True
        except Exception as e:
            self.reporter.info(f"Cache load failed (ignored): {e}")
            return False

    def sync_cpp_state(self, cam_params: Optional[Dict[int, np.ndarray]] = None,
                       window_planes: Optional[Dict[int, Dict]] = None,
                       window_media: Optional[Dict[int, Dict]] = None):
        """Push current parameters to C++ camera objects."""
        cam_params = cam_params or self.cam_params
        window_planes = window_planes or self.window_planes
        window_media = window_media or self.window_media

        self.cam_params = cam_params
        self.window_planes = window_planes
        self.window_media = window_media

        for cid in self.active_cam_ids:
            if cid not in self.cams_cpp or cid not in cam_params:
                continue
            p = cam_params[cid]
            wid = self.cam_to_window.get(cid)
            update_kwargs = CppSyncAdapter.build_update_kwargs(
                cam_params=cam_params,
                window_planes=window_planes,
                window_media=window_media,
                cam_to_window=self.cam_to_window,
                cam_id=cid,
            )
            CppSyncAdapter.apply(self.cams_cpp, cid, update_kwargs)

    def save_cache(self, out_path: str, points_3d: Optional[List[float]] = None):
        """Save results to cache."""
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
                },
                'window_media': {
                    str(w): m for w, m in self.window_media.items()
                }
            }

            if points_3d is not None:
                data['points_3d'] = points_3d
            
            CacheStore(cache_path).save(data)
            self.reporter.info(f"Cache saved: {cache_path}")
            
        except Exception as e:
            self.reporter.info(f"Cache save failed: {e}")


