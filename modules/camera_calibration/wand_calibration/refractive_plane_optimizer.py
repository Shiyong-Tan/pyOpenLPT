"""
Refractive Plane Optimizer (P1)

This module implements Phase P1 optimization for refractive window planes.
All camera parameters (intrinsics, extrinsics) remain FIXED.

Key Design Decisions:
1. Outer-Loop Lambda Adaptation: Lambda is constant within each least_squares call
   to ensure stable Jacobian computation. Lambda is updated only between solver
   calls to target an energy ratio of ~1.0 between ray and length residuals.

2. Joint Triangulation: For each frame and endpoint, a single global X is
   triangulated using rays from ALL active cameras (each using its own window
   plane). This ensures proper constraints even for single-camera windows.

3. Staged Optimization:
   - P1.1: 1D optimization of d per window
   - P1.2: 3D optimization of (d, alpha, beta) per window
   - P1.3: Joint optimization of all windows simultaneously
"""

import numpy as np
import time
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pyopenlpt as lpt

from .refractive_geometry import (
    Ray, normalize, build_pinplate_ray_cpp, triangulate_point, point_to_ray_dist,
    update_normal_tangent, rodrigues_to_R, camera_center, angle_between_vectors,
    optical_axis_world
)


@dataclass
class OptimizationConfig:
    """Configuration for P1 optimization."""
    # Lambda adaptation parameters
    lambda0_init: float = 200.0  # Initial base lambda
    lambda_min: float = 10.0
    lambda_max: float = 5000.0
    target_ratio: float = 1.0  # Target S_len / S_ray ratio
    adaptation_eta: float = 0.30  # Damping factor
    deadband_low: float = 0.7
    deadband_high: float = 1.5
    change_limit_low: float = 0.8
    change_limit_high: float = 1.25
    outer_rounds: int = 3
    
    # Regularization
    lambda_reg: float = 10.0  # Normal drift penalty weight
    
    # Bounds
    alpha_beta_bound: float = 0.5
    
    # Subset
    max_frames: int = 300
    
    # Logging
    verbosity: int = 1  # 0=clean, 1=summary, 2=full audit


class RefractivePlaneOptimizer:
    """
    Optimizer for refractive window plane parameters.
    
    Optimizes plane distance (d) and normal (n) for each window while keeping
    camera parameters fixed. Uses joint triangulation across all cameras.
    """
    
    def __init__(self, 
                 dataset: Dict,
                 cam_params: Dict[int, np.ndarray],
                 cams_cpp: Dict[int, 'lpt.Camera'],
                 cam_to_window: Dict[int, int],
                 window_media: Dict[int, Dict],
                 window_planes: Dict[int, Dict],
                 wand_length: float,
                 config: Optional[OptimizationConfig] = None,
                 progress_callback=None):
        """
        Initialize the optimizer.
        
        Args:
            dataset: Observation data with 'obsA', 'obsB', 'frames' keys
            cam_params: Dict mapping cam_id to parameter array [rvec(3), tvec(3), ...]
            cams_cpp: Dict mapping cam_id to pyopenlpt.Camera objects
            cam_to_window: Dict mapping cam_id to window_id
            window_media: Dict mapping window_id to media properties
            window_planes: Dict mapping window_id to {'plane_pt', 'plane_n'}
            wand_length: Target wand length in mm
            config: Optimization configuration
            progress_callback: Optional callback(phase, ray, len, cost)
        """
        self.dataset = dataset
        self.cam_params = cam_params
        self.cams_cpp = cams_cpp
        self.cam_to_window = cam_to_window
        self.window_media = window_media
        self.window_planes = {wid: {
            'plane_pt': pl['plane_pt'].copy(),
            'plane_n': pl['plane_n'].copy()
        } for wid, pl in window_planes.items()}
        self.wand_length = wand_length
        self.config = config or OptimizationConfig()
        self.progress_callback = progress_callback

        
        # Derived data
        self.active_cam_ids = list(cam_params.keys())
        self.window_ids = sorted(set(self.window_planes.keys()))
        self.n_windows = len(self.window_ids)
        
        # Build observation cache: {frame_id: {cam_id: (uvA, uvB)}}
        self._build_obs_cache()
        
        # Build window -> cameras mapping
        self.win_cams = {wid: [] for wid in self.window_ids}
        for cid in self.active_cam_ids:
            wid = cam_to_window.get(cid, 0)
            if wid in self.win_cams:
                self.win_cams[wid].append(cid)
        
        # Compute anchors (mean camera center per window)
        self.win_anchors = {}
        for wid in self.window_ids:
            centers = []
            for cid in self.win_cams[wid]:
                p = cam_params[cid]
                R = rodrigues_to_R(p[0:3])
                C = camera_center(R, p[3:6])
                centers.append(C)
            if centers:
                self.win_anchors[wid] = np.mean(centers, axis=0)
            else:
                self.win_anchors[wid] = np.zeros(3)
        
        # Store initial normals for regularization
        self.initial_normals = {wid: pl['plane_n'].copy() 
                                for wid, pl in self.window_planes.items()}
        
        # Select frame subset
        import random
        rng = random.Random(42)
        all_frames = list(dataset.get('frames', range(len(dataset.get('obsA', [])))))
        subset_size = min(len(all_frames), self.config.max_frames)
        self.fids_optim = sorted(rng.sample(all_frames, subset_size))
        
    def _build_obs_cache(self):
        """Build observation cache from dataset."""
        self.obs_cache = {}
        obsA = self.dataset.get('obsA', {})
        obsB = self.dataset.get('obsB', {})
        
        # obsA and obsB are dicts: {fid: {cid: (u, v, ...)}}
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
    
    def _update_cpp_camera(self, cam_obj, plane_pt: List[float], plane_n: List[float]):
        """
        Update C++ camera's plane parameters using daisy-chain assignment.
        This handles pybind11's value-copy behavior for nested structs.
        """
        try:
            pp = cam_obj._pinplate_param
            pl = pp.plane
            pl.pt = lpt.Pt3D(plane_pt[0], plane_pt[1], plane_pt[2])
            pl.norm_vector = lpt.Pt3D(plane_n[0], plane_n[1], plane_n[2])
            pp.plane = pl
            cam_obj._pinplate_param = pp
            cam_obj.updatePt3dClosest()
        except Exception as e:
            print(f"  [Warning] C++ update failed: {e}")
    
    def _apply_planes_to_cpp(self, planes: Dict[int, Dict]):
        """Apply plane parameters to all C++ camera objects."""
        for wid, pl in planes.items():
            pt_list = pl['plane_pt'].tolist()
            n_list = pl['plane_n'].tolist()
            for cid in self.win_cams.get(wid, []):
                if cid in self.cams_cpp:
                    self._update_cpp_camera(self.cams_cpp[cid], pt_list, n_list)
    
    def _check_plane_sanity(self, planes: Dict[int, Dict], radii: Dict[str, float]) -> bool:
        """
        Check if any bootstrap point violates the plane-side constraint s(X) > -epsilon - R.
        
        Args:
            planes: Dictionary of plane parameters to validate/fix in-place.
            radii: Dictionary with 'A' and 'B' keys for wand radii.
            
        Returns:
            True if valid (after potential flips), False if rejected.
        """
        # 1. Camera Side Sanity (Flip normal if needed)
        for wid, pl in planes.items():
            n = pl['plane_n']
            pt = pl['plane_pt']
            
            # Get camera centers for this window
            centers = []
            for cid in self.win_cams.get(wid, []):
                if cid in self.cam_params:
                    p = self.cam_params[cid]
                    R = rodrigues_to_R(p[0:3])
                    C = camera_center(R, p[3:6])
                    centers.append(C)
            
            if centers:
                # s(C) should be negative
                s_vals = [np.dot(n, C - pt) for C in centers]
                mean_s = np.mean(s_vals)
                
                if mean_s > 0:
                    # Flip normal
                    if self.config.verbosity >= 1:
                        print(f"    [SANITY] Window {wid}: Cameras on positive side (s={mean_s:.1f}mm). FLIPPING normal.")
                    pl['plane_n'] = -n
                    # Re-normalize just in case
                    pl['plane_n'] /= np.linalg.norm(pl['plane_n'])
        
        # 2. Point-side Sanity (using triangulated points and wand radii)
        # Apply planes to C++ objects for ray building
        self._apply_planes_to_cpp(planes)
        
        eps = 0.05 # 50 microns margin
        
        # Check a subset of frames for speed
        frames_to_check = self.fids_optim[:min(len(self.fids_optim), 20)]
        
        for fid in frames_to_check:
            rays_A, rays_B = self._build_rays_frame(fid)
            
            # Helper to check for a given set of rays and radius
            def check_endpoint_rays(rays: List[Ray], radius: float) -> bool:
                if len(rays) >= 2:
                    X, _, valid, _ = triangulate_point(rays)
                    if valid:
                        for r in rays:
                            # Check against the plane associated with the camera that observed this ray
                            wid = r.window_id
                            if wid in planes:
                                n = planes[wid]['plane_n']
                                pt = planes[wid]['plane_pt']
                                
                                # s = n . (X - pt)
                                # The center of the sphere X must be at least (radius + eps) away from the plane
                                # in the direction of the normal.
                                # So, s(X) = n . (X - pt) must be >= (radius + eps)
                                # Violation if s(X) < (radius + eps)
                                
                                s = np.dot(X - pt, n)
                                limit = radius + eps
                                if s < limit:
                                    if self.config.verbosity >= 0:
                                        print(f"    [SANITY] Window {wid}: Point-side violation (s={s:.2f}mm < {limit:.2f}mm). REJECT update.")
                                    return False
                return True

            if not check_endpoint_rays(rays_A, radii.get('A', 0.0)):
                return False
            if not check_endpoint_rays(rays_B, radii.get('B', 0.0)):
                return False
            
        return True

    def _build_rays_frame(self, fid: int) -> Tuple[List[Ray], List[Ray]]:
        """
        Build rays for a frame using JOINT triangulation approach.
        Returns (rays_A, rays_B) where each list contains rays from ALL cameras.
        """
        rays_A = []
        rays_B = []
        
        if fid not in self.obs_cache:
            return rays_A, rays_B
        
        for cid, (uvA, uvB) in self.obs_cache[fid].items():
            if cid not in self.cams_cpp:
                continue
            cam_ref = self.cams_cpp[cid]
            wid = self.cam_to_window.get(cid, -1)
            
            if uvA is not None:
                rA = build_pinplate_ray_cpp(cam_ref, uvA, cam_id=cid, 
                                            window_id=wid, frame_id=fid, endpoint="A")
                if rA.valid:
                    rays_A.append(rA)
            
            if uvB is not None:
                rB = build_pinplate_ray_cpp(cam_ref, uvB, cam_id=cid,
                                            window_id=wid, frame_id=fid, endpoint="B")
                if rB.valid:
                    rays_B.append(rB)
        
        return rays_A, rays_B
    
    def evaluate_residuals(self, planes: Dict[int, Dict], lambda_eff: float,
                           include_reg: bool = True, mode: str = 'joint',
                           status_desc: Optional[str] = None
                           ) -> Tuple[np.ndarray, float, float, int, int]:
        """
        Evaluate residuals for given plane parameters.
        
        Returns:
            (residuals, S_ray, S_len_unweighted, N_ray, N_len)
        """
        # Apply planes to C++ objects
        # print(f"[DEBUG] Entering evaluate_residuals (mode={mode})")
        self._apply_planes_to_cpp(planes)
        # print(f"[DEBUG] _apply_planes_to_cpp done")
        
        res_ray = []
        res_len = []
        
        for fid in self.fids_optim:
            # Pump events deep inside loop (critical for responsiveness)
            if hasattr(self, 'progress_callback') and self.progress_callback:
                now = time.time()
                last_pump = getattr(self, '_last_pump_time', 0.0)
                if now - last_pump > 0.1:
                    self._last_pump_time = now
                    # Yield GIL to let Main Thread process GUI events
                    time.sleep(0)
                    try:
                        # Use provided status description or default
                        desc = status_desc if status_desc else "Calculating Errors..."
                        self.progress_callback(desc, -1.0, -1.0, -1.0)
                    except:
                        pass
            
            rays_A, rays_B = self._build_rays_frame(fid)
            
            # JOINT triangulation: single X per endpoint using ALL cameras
            validA, validB = False, False
            XA, XB = None, None
            
            if len(rays_A) >= 2:
                XA, _, validA, _ = triangulate_point(rays_A)
                if validA:
                    for r in rays_A:
                        res_ray.append(point_to_ray_dist(XA, r.o, r.d))
            
            if len(rays_B) >= 2:
                XB, _, validB, _ = triangulate_point(rays_B)
                if validB:
                    for r in rays_B:
                        res_ray.append(point_to_ray_dist(XB, r.o, r.d))
            
            if validA and validB:
                dist = np.linalg.norm(XA - XB)
                res_len.append(dist - self.wand_length)
        
        # Compute statistics
        arr_ray = np.array(res_ray) if res_ray else np.array([0.0])
        arr_len = np.array(res_len) if res_len else np.array([0.0])
        
        S_ray = np.sum(arr_ray**2)
        S_len = np.sum(arr_len**2)  # Unweighted
        N_ray = len(arr_ray)
        N_len = len(arr_len)
        
        # Build weighted residual vector
        residuals = arr_ray.copy()
        
        if len(res_len) > 0:
            weighted_len = np.sqrt(lambda_eff) * arr_len
            residuals = np.concatenate([residuals, weighted_len])
        
        # Add regularization if in 3D mode
        if include_reg and mode in ['3D_full', 'joint']:
            reg_residuals = []
            for wid in self.window_ids:
                n_curr = planes[wid]['plane_n']
                n_init = self.initial_normals[wid]
                diff = n_curr - n_init
                reg_residuals.extend(np.sqrt(self.config.lambda_reg) * diff)
            if reg_residuals:
                residuals = np.concatenate([residuals, np.array(reg_residuals)])
        
        return residuals, S_ray, S_len, N_ray, N_len
    
    def _joint_residual_func(self, x: np.ndarray, lambda_eff: float) -> np.ndarray:
        """
        Residual function for joint optimization.
        x = [d0, a0, b0, d1, a1, b1, ...] for all windows
        """
        # Unpack parameters
        planes = {}
        for i, wid in enumerate(self.window_ids):
            d_val = x[3*i]
            alpha = x[3*i + 1]
            beta = x[3*i + 2]
            
            n_base = self.initial_normals[wid]
            n_new = update_normal_tangent(n_base, alpha, beta)
            anchor = self.win_anchors[wid]
            plane_pt = anchor + n_new * d_val
            
            planes[wid] = {'plane_pt': plane_pt, 'plane_n': n_new}
        
        residuals, S_ray, S_len, N_ray, N_len = self.evaluate_residuals(planes, lambda_eff, 
                                                         include_reg=True, mode='joint',
                                                         status_desc="Optimizing All Window parameters...")
        
        # Pump events and report progress
        if hasattr(self, 'progress_callback') and self.progress_callback:
            now = time.time()
            last_time = getattr(self, '_last_update_time', 0.0)
            if now - last_time > 0.1:
                self._last_update_time = now
                try:
                    # Compute metrics
                    rmse_ray = np.sqrt(S_ray / max(1, N_ray))
                    rmse_len = np.sqrt(S_len / max(1, N_len)) if N_len > 0 else 0.0
                    cost = 0.5 * (S_ray + lambda_eff * S_len)
                    
                    self.progress_callback(f"P1.3: Optimizing All Window parameters...", rmse_ray, rmse_len, cost)
                except Exception as e:
                    print(f"[CRITICAL] Callback Error (Joint): {e}", flush=True)
                
        return residuals

    
    def _adapt_lambda(self, lambda_old: float, S_ray: float, S_len: float) -> float:
        """
        Adapt lambda using damped update rule.
        Target: (lambda * S_len) / S_ray ≈ target_ratio
        """
        cfg = self.config
        eps = 1e-12
        
        if S_len < eps:
            return lambda_old
        
        current_ratio = (lambda_old * S_len) / max(S_ray, eps)
        
        # Check deadband
        if cfg.deadband_low <= current_ratio <= cfg.deadband_high:
            return lambda_old
        
        # Compute update
        lambda_new = lambda_old * ((cfg.target_ratio * S_ray) / max(lambda_old * S_len, eps)) ** cfg.adaptation_eta
        
        # Apply per-update change limit
        lambda_new = np.clip(lambda_new, 
                             cfg.change_limit_low * lambda_old,
                             cfg.change_limit_high * lambda_old)
        
        # Apply global clamp
        lambda_new = np.clip(lambda_new, cfg.lambda_min, cfg.lambda_max)
        
        return lambda_new
    
    def _single_window_residual_func(self, x: np.ndarray, target_wid: int, 
                                      mode: str, lambda_eff: float) -> np.ndarray:
        """
        Residual function for single-window optimization.
        x = [d] for 1D mode, [d, alpha, beta] for 3D mode
        """
        d_val = x[0]
        anchor = self.win_anchors[target_wid]
        n_base = self.initial_normals[target_wid]
        
        if mode == '1D_d':
            n_new = n_base
        else:  # 3D_full
            alpha, beta = x[1], x[2]
            n_new = update_normal_tangent(n_base, alpha, beta)
        
        plane_pt = anchor + n_new * d_val
        
        # Create temporary planes dict (update only target window)
        planes = {wid: {'plane_pt': pl['plane_pt'].copy(), 'plane_n': pl['plane_n'].copy()}
                  for wid, pl in self.window_planes.items()}
        planes[target_wid] = {'plane_pt': plane_pt, 'plane_n': n_new}
        
        # Include regularization only for 3D mode
        include_reg = (mode == '3D_full')
        
        # Prepare status string
        if mode == '1D_d':
            status = f"Adjusting Window {target_wid} Distance..."
        else:
            status = f"Adjusting Window {target_wid} Angle..."

        residuals, S_ray, S_len, N_ray, N_len = self.evaluate_residuals(planes, lambda_eff, 
                                                         include_reg=include_reg, mode=mode,
                                                         status_desc=status)
        
        # Pump events and report progress
        if hasattr(self, 'progress_callback') and self.progress_callback:
            now = time.time()
            last_time = getattr(self, '_last_update_time', 0.0)
            if now - last_time > 0.1:
                self._last_update_time = now
                try:
                    # Compute metrics
                    rmse_ray = np.sqrt(S_ray / max(1, N_ray))
                    rmse_len = np.sqrt(S_len / max(1, N_len)) if N_len > 0 else 0.0
                    cost = 0.5 * (S_ray + lambda_eff * S_len)
                    
                    if mode == '1D_d':
                        self.progress_callback(f"Adjusting Window {target_wid} Distance...", rmse_ray, rmse_len, cost)
                    else:
                        self.progress_callback(f"Adjusting Window {target_wid} Angle...", rmse_ray, rmse_len, cost)
                except Exception as e:
                    print(f"[CRITICAL] Callback Error (Single): {e}", flush=True)

        return residuals

    
    def optimize_per_window(self) -> Dict[int, Dict]:
        """
        P1.1 and P1.2: Per-window optimization with outer-loop lambda adaptation.
        
        For each window:
        - P1.1: 1D optimization of d only
        - P1.2: 3D optimization of (d, alpha, beta)
        
        Both stages use outer-loop lambda adaptation.
        """
        print(f"\n  [P1.1/P1.2] Per-Window Optimization")
        
        cfg = self.config
        
        # Initial lambda estimation using current planes
        _, S_ray_init, S_len_init, N_ray, N_len = self.evaluate_residuals(
            self.window_planes, 1.0, include_reg=False, status_desc="Initializing...")
        
        if N_len > 0:
            lambda_eff = cfg.lambda0_init * (N_ray / N_len)
        else:
            lambda_eff = cfg.lambda0_init
        lambda_eff = np.clip(lambda_eff, cfg.lambda_min, cfg.lambda_max)
        
        print(f"    Global Init: N_ray={N_ray}, N_len={N_len}, lambda_eff={lambda_eff:.2f}")

        # Get wand radii for sanity checks
        radii = {
            'A': self.dataset.get('est_radius_small_mm', 0.0),
            'B': self.dataset.get('est_radius_large_mm', 0.0)
        }
        
        for wid in self.window_ids:
            if not self.win_cams.get(wid):
                print(f"\n    Window {wid}: [SKIP] No cameras")
                continue
            
            print(f"\n    === Window {wid} ===")
            
            # Get initial parameters
            pl = self.window_planes[wid]
            n_init = pl['plane_n']
            anchor = self.win_anchors[wid]
            d_init = np.dot(n_init, pl['plane_pt'] - anchor)
            
            # Bounds
            thick = self.window_media.get(wid, {}).get('thickness', 10.0)
            d_min = max(1.0 * thick, 20.0)
            d_max = 2500.0
            
            # ===== P1.1: 1D Optimization (d only) =====
            print(f"    [P1.1] 1D Optimize d (init={d_init:.1f}, bounds=[{d_min:.1f}, {d_max:.1f}])")
            
            x_1d = np.array([d_init])
            # Ensure start is within bounds (float tolerance)
            x_1d = np.clip(x_1d, d_min, d_max)
            lambda_local = lambda_eff  # Start with global estimate
            
            for outer_round in range(cfg.outer_rounds):
                # Pump events
                if self.progress_callback:
                    try:
                        self.progress_callback(f"Adjusting Window {wid} Distance...", 0, 0, 0)
                    except:
                        pass

                result = least_squares(
                    lambda x: self._single_window_residual_func(x, wid, '1D_d', lambda_local),
                    x_1d,
                    bounds=(np.array([d_min]), np.array([d_max])),
                    loss='huber', f_scale=1.0,
                    verbose=0,
                    max_nfev=100
                )
                x_1d = result.x.copy()
                
                # Evaluate at solution
                d_opt = x_1d[0]
                temp_planes = {w: {'plane_pt': self.window_planes[w]['plane_pt'].copy(),
                                   'plane_n': self.window_planes[w]['plane_n'].copy()}
                               for w in self.window_ids}
                temp_planes[wid]['plane_pt'] = anchor + n_init * d_opt
                
                _, S_ray, S_len, _, _ = self.evaluate_residuals(temp_planes, lambda_local, include_reg=False,
                                                              status_desc=f"Adjusting Window {wid} Distance...")
                ratio = (lambda_local * S_len) / max(S_ray, 1e-12)
                
                if outer_round == 0:
                    print(f"      [AUDIT] lambda={lambda_local:.2f}, S_ray={S_ray:.2f}, S_len={S_len:.4f}, ratio={ratio:.3f}")
                
                # Report progress
                if self.progress_callback:
                    rmse_ray_audit = np.sqrt(S_ray / max(1, N_ray))
                    rmse_len_audit = np.sqrt(S_len / max(1, N_len)) if N_len > 0 else 0.0
                    self.progress_callback(f"Adjusting Window {wid} Distance...", rmse_ray_audit, rmse_len_audit, result.cost)
                
                # Adapt lambda
                lambda_old = lambda_local
                lambda_local = self._adapt_lambda(lambda_old, S_ray, S_len)
                
                if abs(lambda_local - lambda_old) < 0.1:
                    break
            
            d_opt_1d = x_1d[0]
            print(f"      -> d_opt: {d_opt_1d:.2f} mm (cost: {result.cost:.4f})")
            
            # Update planes with 1D result
            old_pt = self.window_planes[wid]['plane_pt'].copy()
            old_n = self.window_planes[wid]['plane_n'].copy()
            
            self.window_planes[wid]['plane_pt'] = anchor + n_init * d_opt_1d
            
            # Sanity Check
            if not self._check_plane_sanity(self.window_planes, radii):
                 print(f"      [P1.1] Reverting Window {wid} (Bad Geometry)")
                 self.window_planes[wid]['plane_pt'] = old_pt
                 self.window_planes[wid]['plane_n'] = old_n
            
            # ===== P1.2: 3D Optimization [d, alpha, beta] =====
            print(f"    [P1.2] 3D Optimize [d, alpha, beta]")
            
            lb = np.array([d_min, -cfg.alpha_beta_bound, -cfg.alpha_beta_bound])
            ub = np.array([d_max, cfg.alpha_beta_bound, cfg.alpha_beta_bound])
            x_3d = np.array([d_opt_1d, 0.0, 0.0])
            x_3d = np.clip(x_3d, lb, ub)
            
            for outer_round in range(cfg.outer_rounds):
                # Pump events
                if self.progress_callback:
                    try:
                        self.progress_callback(f"Adjusting Window {wid} Angle...", 0, 0, 0)
                    except:
                        pass
                result = least_squares(

                    lambda x: self._single_window_residual_func(x, wid, '3D_full', lambda_local),
                    x_3d,
                    bounds=(lb, ub),
                    loss='huber', f_scale=1.0,
                    verbose=0,
                    max_nfev=150
                )
                x_3d = result.x.copy()
                
                # Evaluate at solution
                d_opt, alpha, beta = x_3d
                n_new = update_normal_tangent(n_init, alpha, beta)
                temp_planes = {w: {'plane_pt': self.window_planes[w]['plane_pt'].copy(),
                                   'plane_n': self.window_planes[w]['plane_n'].copy()}
                               for w in self.window_ids}
                temp_planes[wid] = {'plane_pt': anchor + n_new * d_opt, 'plane_n': n_new}
                
                _, S_ray, S_len, _, _ = self.evaluate_residuals(temp_planes, lambda_local, include_reg=False,
                                                              status_desc=f"Adjusting Window {wid} Angle...")
                ratio = (lambda_local * S_len) / max(S_ray, 1e-12)
                
                if outer_round == 0:
                    print(f"      [AUDIT] lambda={lambda_local:.2f}, S_ray={S_ray:.2f}, S_len={S_len:.4f}, ratio={ratio:.3f}")

                # Report progress
                if self.progress_callback:
                    rmse_ray_audit = np.sqrt(S_ray / max(1, N_ray))
                    rmse_len_audit = np.sqrt(S_len / max(1, N_len)) if N_len > 0 else 0.0
                    self.progress_callback(f"Adjusting Window {wid} Angle...", rmse_ray_audit, rmse_len_audit, result.cost)
                
                # Adapt lambda
                lambda_old = lambda_local
                lambda_local = self._adapt_lambda(lambda_old, S_ray, S_len)
                
                if abs(lambda_local - lambda_old) < 0.1:
                    break
            
            d_final, a_final, b_final = x_3d
            n_final = update_normal_tangent(n_init, a_final, b_final)
            pt_final = anchor + n_final * d_final
            
            # Report
            angle = angle_between_vectors(n_final, n_init)
            print(f"      -> d={d_final:.2f}, alpha={a_final:.4f}, beta={b_final:.4f}")
            print(f"      -> n_new: {n_final.round(4)}, angle_change: {angle:.2f}°")
            
            # Commit
            old_pt = self.window_planes[wid]['plane_pt'].copy()
            old_n = self.window_planes[wid]['plane_n'].copy()
            
            self.window_planes[wid] = {'plane_pt': pt_final, 'plane_n': n_final}
            
            # Sanity Check
            radii = {
                'A': self.dataset.get('est_radius_small_mm', 0.0),
                'B': self.dataset.get('est_radius_large_mm', 0.0)
            }
            if not self._check_plane_sanity(self.window_planes, radii):
                 print(f"      [P1.2] Reverting Window {wid} (Bad Geometry)")
                 self.window_planes[wid]['plane_pt'] = old_pt
                 self.window_planes[wid]['plane_n'] = old_n
            
            # Update initial_normals for P1.3 regularization baseline
            self.initial_normals[wid] = n_final.copy()
            
            # Carry forward lambda for next window
            lambda_eff = lambda_local
        
        return self.window_planes
    
    def optimize_joint(self) -> Dict[int, Dict]:
        """
        P1.3: Joint optimization of all windows with outer-loop lambda adaptation.
        """
        print(f"\n  [P1.3] Joint Optimization (All {self.n_windows} Windows)")
        
        cfg = self.config
        
        # Initialize parameters
        x0 = []
        lb = []
        ub = []
        
        for wid in self.window_ids:
            pl = self.window_planes[wid]
            n_init = pl['plane_n']
            anchor = self.win_anchors[wid]
            d_init = np.dot(n_init, pl['plane_pt'] - anchor)
            
            # Bounds
            thick = self.window_media.get(wid, {}).get('thickness', 10.0)
            d_min = max(1.0 * thick, 20.0)
            d_max = 2500.0
            
            x0.extend([d_init, 0.0, 0.0])
            lb.extend([d_min, -cfg.alpha_beta_bound, -cfg.alpha_beta_bound])
            ub.extend([d_max, cfg.alpha_beta_bound, cfg.alpha_beta_bound])
        
        x0 = np.array(x0)
        bounds = (np.array(lb), np.array(ub))
        
        if self.progress_callback:
             try:
                 self.progress_callback("P1.3: Optimizing All Window parameters...", 0, 0, 0)
             except:
                 pass

        # Initial lambda estimation
        # First, evaluate with current planes to get N_ray, N_len
        _, S_ray_init, S_len_init, N_ray, N_len = self.evaluate_residuals(
            self.window_planes, 1.0, include_reg=False, status_desc="Optimizing All Window parameters...")
        
        if N_len > 0:
            lambda_eff = cfg.lambda0_init * (N_ray / N_len)
        else:
            lambda_eff = cfg.lambda0_init
        
        lambda_eff = np.clip(lambda_eff, cfg.lambda_min, cfg.lambda_max)
        
        print(f"    Initial: N_ray={N_ray}, N_len={N_len}, lambda_eff={lambda_eff:.2f}")
        
        # Outer-loop lambda adaptation
        x_current = x0.copy()
        
        for outer_round in range(cfg.outer_rounds):
            # Pump events
            if self.progress_callback:
                try:
                    self.progress_callback(f"Optimizing All Window parameters...", 0, 0, 0)
                except:
                    pass

            print(f"\n    --- Outer Round {outer_round + 1}/{cfg.outer_rounds} ---")

            
            # Run optimizer with fixed lambda
            result = least_squares(
                lambda x: self._joint_residual_func(x, lambda_eff),
                x_current,
                bounds=bounds,
                loss='huber', f_scale=1.0,
                verbose=0,
                max_nfev=200
            )
            
            x_current = result.x.copy()
            
            # Reconstruct planes from solution
            planes = {}
            for i, wid in enumerate(self.window_ids):
                d_val = x_current[3*i]
                alpha = x_current[3*i + 1]
                beta = x_current[3*i + 2]
                
                n_base = self.initial_normals[wid]
                n_new = update_normal_tangent(n_base, alpha, beta)
                anchor = self.win_anchors[wid]
                plane_pt = anchor + n_new * d_val
                
                planes[wid] = {'plane_pt': plane_pt, 'plane_n': n_new}
            
            # Evaluate S_ray and S_len at solution
            _, S_ray, S_len, _, _ = self.evaluate_residuals(planes, lambda_eff, include_reg=False,
                                                          status_desc="Optimizing All Window parameters...")
            
            current_ratio = (lambda_eff * S_len) / max(S_ray, 1e-12)
            
            # AUDIT log
            print(f"    [AUDIT] N_ray={N_ray}, N_len={N_len}, lambda_eff={lambda_eff:.2f}")
            print(f"            S_ray={S_ray:.4f}, S_len={S_len:.4f}, ratio={current_ratio:.4f}")
            print(f"            cost={result.cost:.4f}")
            
            # Push AUDIT results to UI
            if self.progress_callback:
                rmse_ray_audit = np.sqrt(S_ray / max(1, N_ray))
                rmse_len_audit = np.sqrt(S_len / max(1, N_len)) if N_len > 0 else 0.0
                self.progress_callback(f"Optimizing All Window parameters...", rmse_ray_audit, rmse_len_audit, result.cost)
            
            # Adapt lambda
            lambda_old = lambda_eff
            lambda_eff = self._adapt_lambda(lambda_old, S_ray, S_len)
            
            if abs(lambda_eff - lambda_old) < 0.1:
                print(f"    Lambda converged, stopping early.")
                break
            else:
                print(f"    Lambda update: {lambda_old:.2f} -> {lambda_eff:.2f}")
        
        # Final result
        final_planes = {}
        for i, wid in enumerate(self.window_ids):
            d_val = x_current[3*i]
            alpha = x_current[3*i + 1]
            beta = x_current[3*i + 2]
            
            n_base = self.initial_normals[wid]
            n_new = update_normal_tangent(n_base, alpha, beta)
            anchor = self.win_anchors[wid]
            plane_pt = anchor + n_new * d_val
            
            final_planes[wid] = {'plane_pt': plane_pt, 'plane_n': n_new}
            
            # Report per-window results
            angle = angle_between_vectors(n_new, self.initial_normals[wid])
            print(f"\n    Window {wid}: d={d_val:.2f}mm, alpha={alpha:.4f}, beta={beta:.4f}")
            print(f"      Normal change: {angle:.2f}°")
            print(f"      n_new: {n_new.round(4)}")
        
        # Update internal state
        old_planes = {w: {'plane_pt': self.window_planes[w]['plane_pt'].copy(), 
                          'plane_n': self.window_planes[w]['plane_n'].copy()} 
                      for w in self.window_ids}
                      
        self.window_planes = final_planes
        
        # Sanity Check
        radii = {
            'A': self.dataset.get('est_radius_small_mm', 0.0),
            'B': self.dataset.get('est_radius_large_mm', 0.0)
        }
        if not self._check_plane_sanity(self.window_planes, radii):
             print(f"    [P1.3] Joint Optimization rejected (Bad Geometry). Reverting.")
             self.window_planes = old_planes
             return old_planes
        
        return final_planes
    
    def print_diagnostics(self):
        """Print comprehensive diagnostics after optimization."""
        print("\n  === P1 Optimization Diagnostics ===")
        
        # Apply final planes
        self._apply_planes_to_cpp(self.window_planes)
        
        # Collect per-camera and per-window residuals
        cam_residuals_A = {cid: [] for cid in self.active_cam_ids}
        cam_residuals_B = {cid: [] for cid in self.active_cam_ids}
        wand_lengths = []
        
        for fid in self.fids_optim:
            rays_A, rays_B = self._build_rays_frame(fid)
            
            validA, validB = False, False
            XA, XB = None, None
            
            if len(rays_A) >= 2:
                XA, _, validA, _ = triangulate_point(rays_A)
                if validA:
                    for r in rays_A:
                        d = point_to_ray_dist(XA, r.o, r.d)
                        cam_residuals_A[r.cam_id].append(d)
            
            if len(rays_B) >= 2:
                XB, _, validB, _ = triangulate_point(rays_B)
                if validB:
                    for r in rays_B:
                        d = point_to_ray_dist(XB, r.o, r.d)
                        cam_residuals_B[r.cam_id].append(d)
            
            if validA and validB:
                wand_lengths.append(np.linalg.norm(XA - XB))
        
        # Per-camera statistics
        print("\n  Per-Camera Ray Residuals (mm):")
        for cid in sorted(self.active_cam_ids):
            resA = cam_residuals_A[cid]
            resB = cam_residuals_B[cid]
            wid = self.cam_to_window.get(cid, 0)
            
            if resA:
                med_A = np.median(resA)
                mean_A = np.mean(resA)
                rmse_A = np.sqrt(np.mean(np.array(resA)**2))
            else:
                med_A = mean_A = rmse_A = 0.0
            
            if resB:
                med_B = np.median(resB)
                mean_B = np.mean(resB)
                rmse_B = np.sqrt(np.mean(np.array(resB)**2))
            else:
                med_B = mean_B = rmse_B = 0.0
            
            print(f"    Cam {cid} (Win {wid}): A[med={med_A:.3f}, rmse={rmse_A:.3f}] "
                  f"B[med={med_B:.3f}, rmse={rmse_B:.3f}]")
        
        # Per-window aggregate
        print("\n  Per-Window Ray Residuals (mm):")
        for wid in self.window_ids:
            win_res = []
            for cid in self.win_cams.get(wid, []):
                win_res.extend(cam_residuals_A.get(cid, []))
                win_res.extend(cam_residuals_B.get(cid, []))
            if win_res:
                med = np.median(win_res)
                p90 = np.percentile(win_res, 90)
                rmse = np.sqrt(np.mean(np.array(win_res)**2))
                print(f"    Window {wid}: median={med:.3f}, p90={p90:.3f}, rmse={rmse:.3f}")
        
        # Wand length statistics
        if wand_lengths:
            arr = np.array(wand_lengths)
            err = arr - self.wand_length
            print(f"\n  Wand Length Statistics (mm):")
            print(f"    Mean:   {np.mean(arr):.3f} (target: {self.wand_length:.2f})")
            print(f"    Median: {np.median(arr):.3f}")
            print(f"    RMSE:   {np.sqrt(np.mean(err**2)):.3f}")
            print(f"    Bias:   {np.mean(err):.3f}")
            print(f"    p90 err: {np.percentile(np.abs(err), 90):.3f}")
        
        # Angle diagnostics
        print("\n  Angle Diagnostics:")
        for wid in self.window_ids:
            pl = self.window_planes[wid]
            n_w = pl['plane_n']
            plane_pt = pl['plane_pt']
            
            print(f"    Window {wid}:")
            for cid in self.win_cams.get(wid, []):
                p = self.cam_params[cid]
                R = rodrigues_to_R(p[0:3])
                C = camera_center(R, p[3:6])
                
                # View direction to plane
                v_to_plane = normalize(plane_pt - C)
                angle_view = angle_between_vectors(n_w, v_to_plane)
                
                # Optical axis angle
                z_world = optical_axis_world(R)
                angle_optical = angle_between_vectors(n_w, z_world)
                
                print(f"      Cam {cid}: n vs view_to_plane={angle_view:.1f}°, "
                      f"n vs optical_axis={angle_optical:.1f}°")
