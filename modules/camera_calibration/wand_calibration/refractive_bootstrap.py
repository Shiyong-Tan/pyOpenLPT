"""
Refractive Bootstrap (P0 Stage)
===============================
Frozen-Intrinsics Pinhole Bootstrap for Refractive Calibration.

This provides a physically reasonable extrinsic initialization for later stages.
It is NOT calibration - it ONLY initializes extrinsics.

KEY RULES:
- Intrinsics (fx, fy, cx, cy) are FROZEN to UI values
- NO distortion parameters
- NO camFile output (in-memory only)
- Uses 8-Point Algorithm for initialization (same as pinhole Phase 1)
- Only optimizes extrinsics (Phase 1 BA with frozen intrinsics)
- Wand length is the ONLY scale constraint
- Pair selection is handled externally via precalibrate
"""

import numpy as np
from scipy.optimize import least_squares
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


def select_best_pair_via_precalib(base_calibrator, wand_len_mm: float, initial_focal_px: float) -> Optional[Tuple[int, int]]:
    """
    Run pinhole precalibration to determine the best camera pair based on
    reprojection errors. Useful baseline even for refractive setups.
    
    Args:
        base_calibrator: WandCalibrator instance
        wand_len_mm: Target wand length in mm
        initial_focal_px: Initial focal length in pixels
        
    Returns:
        (cam_i, cam_j) tuple of best camera pair, or None if failed
    """
    print("\n[BOOT] Running Precalibration Check to select best pair...")
    
    try:
        if not hasattr(base_calibrator, 'run_precalibration_check'):
            raise AttributeError("run_precalibration_check is unavailable on base calibrator")
        ret, msg, precalib_result = base_calibrator.run_precalibration_check(
            wand_length=wand_len_mm,
            init_focal_length=initial_focal_px
        )
    except Exception as e:
        print(f"  [WARN] Precalibration failed: {e}. Falling back to shared count.")
        
        # Fallback: Select pair with most common frames
        counts = {}
        points = getattr(base_calibrator, 'wand_points_filtered', None) or getattr(base_calibrator, 'wand_points', {})
        if not points:
             return None
        
        all_cams = set()
        for fid, cams in points.items():
            cam_list = list(cams.keys())
            for i in range(len(cam_list)):
                for j in range(i+1, len(cam_list)):
                    c1, c2 = sorted((cam_list[i], cam_list[j]))
                    counts[(c1, c2)] = counts.get((c1, c2), 0) + 1
                    all_cams.add(c1)
                    all_cams.add(c2)
        
        if not counts:
             # Just pick first two cams
             cams = sorted(list(all_cams))
             if len(cams) >= 2:
                 return (cams[0], cams[1])
             return None

        # Sort by shared count desc; tie-break by median pixel disparity (larger is better).
        sorted_pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        def _extract_uv(pts):
            if pts is None:
                return None
            arr = np.asarray(pts)
            if arr.ndim == 1 and arr.shape[0] >= 2:
                return np.array(arr[:2], dtype=np.float64)
            if arr.ndim >= 2 and arr.shape[0] >= 1 and arr.shape[1] >= 2:
                # Use centroid of first two detected points (small/large) when available.
                n = min(2, arr.shape[0])
                return np.mean(arr[:n, :2].astype(np.float64), axis=0)
            return None

        def _pair_disparity(pair):
            c1, c2 = pair
            d = []
            for _, cams in points.items():
                if c1 not in cams or c2 not in cams:
                    continue
                uv1 = _extract_uv(cams[c1])
                uv2 = _extract_uv(cams[c2])
                if uv1 is None or uv2 is None:
                    continue
                disp = np.linalg.norm(uv1 - uv2)
                if np.isfinite(disp):
                    d.append(float(disp))
            if not d:
                return -1.0
            return float(np.median(d))

        top_count = sorted_pairs[0][1]
        top_pairs = [p for p, cnt in sorted_pairs if cnt == top_count]
        best_pair = max(top_pairs, key=_pair_disparity)
        print(
            f"[BOOT] Fallback: Selected pair {best_pair} with {top_count} shared frames "
            f"(median disparity={_pair_disparity(best_pair):.2f}px)."
        )
        return best_pair

    if not ret:
        print(f"  [WARN] Precalibration returned False: {msg}")
        # Try to parse errors anyway
        pass

    # Extract errors
    wand_data = base_calibrator.wand_points_filtered or base_calibrator.wand_points
    all_cam_ids = sorted(list(set(cid for f in wand_data.values() for cid in f)))
    
    per_cam_error = {}
    # Try to get from internal state first
    if hasattr(base_calibrator, 'per_frame_errors') and base_calibrator.per_frame_errors:
        cam_errors_list = {cid: [] for cid in all_cam_ids}
        for fid, frame_data in base_calibrator.per_frame_errors.items():
            if 'cam_errors' in frame_data:
                for cid, err in frame_data['cam_errors'].items():
                    if cid in cam_errors_list: 
                        cam_errors_list[cid].append(err)
        for cid in all_cam_ids:
            if cam_errors_list[cid]:
                per_cam_error[cid] = np.sqrt(np.mean(np.array(cam_errors_list[cid])**2))
    
    # Fallback to parsing message
    if not per_cam_error:
        for line in msg.split('\n'):
            match = re.search(r'Cam\s*(\d+):\s*([\d.]+)\s*px', line)
            if match:
                per_cam_error[int(match.group(1))] = float(match.group(2))
    
    if not per_cam_error:
        print("  [WARN] Could not determine per-camera errors.")
        return None
        
    print("\n[BOOT] Per-camera reprojection errors (Pinhole approx):")
    for cid in sorted(per_cam_error.keys()):
        print(f"  Cam {cid}: {per_cam_error[cid]:.2f}px")
        
    sorted_cams = sorted(per_cam_error.keys(), key=lambda c: per_cam_error[c])
    if len(sorted_cams) < 2:
        return None
        
    best_pair = (min(sorted_cams[0], sorted_cams[1]), max(sorted_cams[0], sorted_cams[1]))
    print(f"[BOOT] Selected best pair: {best_pair}")
    return best_pair


@dataclass
class PinholeBootstrapP0Config:
    """Configuration for P0 bootstrap."""
    wand_length_mm: float = 10.0
    ui_focal_px: float = 9000.0  # UI-provided focal length (FROZEN)
    ftol: float = 1e-6
    xtol: float = 1e-6


class PinholeBootstrapP0:
    """
    Stage P0: Two-camera pinhole initialization with frozen intrinsics.
    
    Uses 8-Point Algorithm (same as original pinhole Phase 1) but with frozen intrinsics.
    Optimizes only extrinsics.
    
    Pair selection is handled externally (via precalibrate).
    """
    
    def __init__(self, config: PinholeBootstrapP0Config):
        self.config = config

    @staticmethod
    def _get_camera_intrinsics(cam_id: int, camera_settings: Dict[int, dict]) -> Tuple[np.ndarray, float, float, float]:
        if cam_id not in camera_settings:
            raise ValueError(f"[P0] Missing camera_settings for cam {cam_id}")
        cfg = camera_settings[cam_id]
        f = float(cfg.get('focal', 0.0))
        w = float(cfg.get('width', 0.0))
        h = float(cfg.get('height', 0.0))
        if f <= 0 or w <= 0 or h <= 0:
            raise ValueError(
                f"[P0] Invalid camera_settings for cam {cam_id}: focal={f}, width={w}, height={h}"
            )
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        return K, f, cx, cy
        
    def run(
        self,
        cam_i: int,
        cam_j: int,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Run P0 pinhole bootstrap for camera pair using 8-Point Algorithm.
        
        This is identical to pinhole Phase 1, but with frozen intrinsics.
        
        Args:
            cam_i, cam_j: Camera IDs (cam_i fixed at origin)
            observations: {fid: {cid: (uvA, uvB)}}
            camera_settings: per-camera intrinsics source from UI table
            progress_callback: Optional callback(phase, ray, len, cost)
            
        Returns:
            params_i: [rvec(3), tvec(3)] for cam_i (zeros)
            params_j: [rvec(3), tvec(3)] for cam_j (from 8-Point + refinement)
            report: diagnostics dict
        """
        K_i, f_i, cx_i, cy_i = self._get_camera_intrinsics(cam_i, camera_settings)
        K_j, f_j, cx_j, cy_j = self._get_camera_intrinsics(cam_j, camera_settings)
        
        print(f"\n{'='*60}")
        print(f"[P0] Pinhole Bootstrap - Frozen Intrinsics (8-Point)")
        print(f"{'='*60}")
        print(f"  Camera pair: ({cam_i}, {cam_j})")
        
        if progress_callback:
            try:
                progress_callback("P0 Pair Init", -1, 0, 0, 0)
            except:
                pass
        
        # Collect valid frames and points
        valid_frames = self._collect_valid_frames(observations, cam_i, cam_j)
        print(f"  Valid frames: {len(valid_frames)}")
        
        if len(valid_frames) < 10:
            raise ValueError(f"[P0] Insufficient frames: {len(valid_frames)} < 10")
        
        # Collect point correspondences for 8-Point Algorithm
        pts_i = []  # Points in cam_i
        pts_j = []  # Points in cam_j
        
        for fid in valid_frames:
            uvA_i, uvB_i = observations[fid][cam_i]
            uvA_j, uvB_j = observations[fid][cam_j]
            
            pts_i.append(uvA_i)
            pts_i.append(uvB_i)
            pts_j.append(uvA_j)
            pts_j.append(uvB_j)
        
        pts_i = np.array(pts_i, dtype=np.float64)
        pts_j = np.array(pts_j, dtype=np.float64)
        
        print(f"\n[P0] Step 1: Essential Matrix (8-Point Algorithm)...")
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except:
                pass

        # Step 1: Essential Matrix (normalized rays, supports different intrinsics)
        pts_i_norm = cv2.undistortPoints(pts_i.reshape(-1, 1, 2), K_i, None).reshape(-1, 2)
        pts_j_norm = cv2.undistortPoints(pts_j.reshape(-1, 1, 2), K_j, None).reshape(-1, 2)
        E, mask = cv2.findEssentialMat(
            pts_i_norm,
            pts_j_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1e-3,
        )
        
        if E is None or E.shape != (3, 3):
            raise RuntimeError("[P0 FAIL] Essential Matrix computation failed")
        
        # Step 2: Recover Pose (R, t)
        n_inliers, R_rel, t_rel, mask_pose = cv2.recoverPose(
            E, pts_i_norm, pts_j_norm, focal=1.0, pp=(0.0, 0.0)
        )
        
        print(f"  Inliers: {n_inliers} / {len(pts_i)}")
        
        # Step 3: Triangulate to compute scale
        print(f"\n[P0] Step 2: Triangulation & Scale Recovery...")
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except:
                pass
        
        # Projection matrices (cam_i at origin)
        P_i = np.hstack([np.eye(3), np.zeros((3, 1))])
        P_j = np.hstack([R_rel, t_rel])
        
        # Triangulate all points
        pts_4d_hom = cv2.triangulatePoints(P_i, P_j, pts_i_norm.T, pts_j_norm.T)
        pts_3d = (pts_4d_hom[:3] / pts_4d_hom[3]).T  # (N, 3)
        
        # Compute wand lengths
        wand_lengths = []
        for i in range(0, len(pts_3d), 2):
            ptA = pts_3d[i]
            ptB = pts_3d[i + 1]
            wand_lengths.append(np.linalg.norm(ptB - ptA))
        
        wand_lengths = np.array(wand_lengths)
        valid_lengths = wand_lengths[(wand_lengths > 0.001) & (wand_lengths < 1000)]
        
        if len(valid_lengths) < 5:
            raise RuntimeError("[P0 FAIL] Triangulation failed to produce valid structure")
        
        median_length = np.median(valid_lengths)
        scale_factor = self.config.wand_length_mm / median_length
        
        # Apply scale to translation
        t_scaled = t_rel * scale_factor
        
        # Convert R to rvec
        rvec_j, _ = cv2.Rodrigues(R_rel)
        
        # Build params arrays (both cameras, like original Phase 1)
        params_i = np.zeros(6)  # cam_i at origin initially
        params_j = np.concatenate([rvec_j.flatten(), t_scaled.flatten()])
        
        print(f"\n[P0] Step 3: Extrinsic Refinement (frozen intrinsics)...")
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except:
                pass
        
        # Build initial 3D points (scaled)
        pts_3d_scaled = pts_3d * scale_factor
        n_pts = len(pts_3d_scaled)
        
        # State vector: [cam_i(6), cam_j(6), pts_3d(N*3)]
        x0 = np.concatenate([params_i, params_j, pts_3d_scaled.flatten()])
        
        from scipy.sparse import lil_matrix
        
        # Residuals: for each frame: wand(1) + reproj(8) = 9
        n_frames = len(valid_frames)
        n_res = n_frames * 9
        n_cams = 2
        n_cam_params = 6
        pt_start = n_cams * n_cam_params  # = 12
        n_params = pt_start + n_pts * 3
        
        A_sparsity = lil_matrix((n_res, n_params), dtype=int)
        
        for i, fid in enumerate(valid_frames):
            idx_ptA = pt_start + i * 6
            idx_ptB = pt_start + i * 6 + 3
            base_res = i * 9
            
            # Wand length
            A_sparsity[base_res, idx_ptA:idx_ptA+3] = 1
            A_sparsity[base_res, idx_ptB:idx_ptB+3] = 1
            
            # Reproj cam_i ptA/ptB
            A_sparsity[base_res+1:base_res+3, 0:6] = 1
            A_sparsity[base_res+1:base_res+3, idx_ptA:idx_ptA+3] = 1
            A_sparsity[base_res+3:base_res+5, 0:6] = 1
            A_sparsity[base_res+3:base_res+5, idx_ptB:idx_ptB+3] = 1
            
            # Reproj cam_j ptA/ptB
            A_sparsity[base_res+5:base_res+7, 6:12] = 1
            A_sparsity[base_res+5:base_res+7, idx_ptA:idx_ptA+3] = 1
            A_sparsity[base_res+7:base_res+9, 6:12] = 1
            A_sparsity[base_res+7:base_res+9, idx_ptB:idx_ptB+3] = 1
        
        # Residuals function (frozen intrinsics)
        self._res_call_count = 0 
        def residuals_func(x):
            p_i = x[:6]
            p_j = x[6:12]
            pts = x[12:].reshape(-1, 3)
            
            R_i, _ = cv2.Rodrigues(p_i[:3])
            t_i = p_i[3:6].reshape(3, 1)
            R_j, _ = cv2.Rodrigues(p_j[:3])
            t_j = p_j[3:6].reshape(3, 1)
            
            res = []
            sq_err_len = 0.0
            n_len = 0
            sq_err_proj = 0.0
            n_proj = 0
            for idx, fid in enumerate(valid_frames):
                uvA_i, uvB_i = observations[fid][cam_i]
                uvA_j, uvB_j = observations[fid][cam_j]
                
                ptA = pts[idx * 2]
                ptB = pts[idx * 2 + 1]
                
                # Wand length
                wand_len = np.linalg.norm(ptB - ptA)
                d_len = wand_len - self.config.wand_length_mm
                res.append(d_len)
                sq_err_len += d_len * d_len
                n_len += 1
                
                # Reprojections with frozen K
                proj_Ai = self._project(ptA, R_i, t_i, K_i)
                proj_Bi = self._project(ptB, R_i, t_i, K_i)
                diff_Ai = (proj_Ai - uvA_i)
                diff_Bi = (proj_Bi - uvB_i)
                res.extend(diff_Ai.tolist())
                res.extend(diff_Bi.tolist())
                sq_err_proj += float(np.sum(diff_Ai**2) + np.sum(diff_Bi**2))
                n_proj += 4

                proj_Aj = self._project(ptA, R_j, t_j, K_j)
                proj_Bj = self._project(ptB, R_j, t_j, K_j)
                diff_Aj = (proj_Aj - uvA_j)
                diff_Bj = (proj_Bj - uvB_j)
                res.extend(diff_Aj.tolist())
                res.extend(diff_Bj.tolist())
                sq_err_proj += float(np.sum(diff_Aj**2) + np.sum(diff_Bj**2))
                n_proj += 4

            # To numpy
            res_arr = np.array(res)

            # Report progress with metrics
            self._res_call_count += 1
            if progress_callback and self._res_call_count % 5 == 0:
                try:
                    rmse_len = np.sqrt(sq_err_len / max(1, n_len))
                    rmse_proj = np.sqrt(sq_err_proj / max(1, n_proj))
                    rmse_ray = -1.0
                    cost = 0.5 * np.sum(res_arr**2)

                    progress_callback(
                        "Use PinHole model to initialize camera parameters...",
                        rmse_ray,
                        rmse_len,
                        rmse_proj,
                        cost,
                    )
                except:
                    pass
            
            return res_arr

        
        result = least_squares(
            residuals_func, x0,
            jac_sparsity=A_sparsity,
            method='trf',
            x_scale='jac',
            f_scale=1.0,
            verbose=1,
            ftol=self.config.ftol,
            xtol=self.config.xtol,
            max_nfev=100,
        )

        
        params_i_opt = result.x[:6]
        params_j_opt = result.x[6:12]
        pts_3d_opt = result.x[12:].reshape(-1, 3)
        
        print(f"  BA cost: {result.cost:.2e}")
        print(f"  cam_i moved: |t| = {np.linalg.norm(params_i_opt[3:6]):.4f} mm")
        
        # Compute diagnostics
        report = self._compute_diagnostics(
            cam_i, cam_j, params_i_opt, params_j_opt,
            observations, valid_frames, K_i, K_j
        )
        report['scale_factor'] = scale_factor
        report['n_inliers'] = n_inliers
        
        # Sanity checks
        self._validate(report)
        
        print(f"\n[P0] Phase 1 Complete:")
        print(f"  cam_{cam_i} rvec: [{params_i_opt[0]:.4f}, {params_i_opt[1]:.4f}, {params_i_opt[2]:.4f}]")
        print(f"  cam_{cam_i} tvec: [{params_i_opt[3]:.2f}, {params_i_opt[4]:.2f}, {params_i_opt[5]:.2f}]")
        print(f"  cam_{cam_j} rvec: [{params_j_opt[0]:.4f}, {params_j_opt[1]:.4f}, {params_j_opt[2]:.4f}]")
        print(f"  cam_{cam_j} tvec: [{params_j_opt[3]:.2f}, {params_j_opt[4]:.2f}, {params_j_opt[5]:.2f}]")
        print(f"  Baseline: {report['baseline_mm']:.2f} mm")
        print(f"  Wand length: {report['wand_length_median']:.4f} mm")
        
        return params_i_opt, params_j_opt, report
    

    
    def _project(self, pt3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D using pinhole model."""
        pt_cam = R @ pt3d.reshape(3, 1) + t
        pt_cam = pt_cam.flatten()
        if pt_cam[2] <= 0:
            return np.array([1e6, 1e6])  # Behind camera
        pt_norm = pt_cam[:2] / pt_cam[2]
        pt_px = K[:2, :2] @ pt_norm + K[:2, 2]
        return pt_px

    def _ray_dir_world(self, uv: np.ndarray, K: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Build world-space pinhole ray direction from pixel coordinate."""
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        x = (float(uv[0]) - cx) / max(abs(fx), 1e-12)
        y = (float(uv[1]) - cy) / max(abs(fy), 1e-12)
        d_cam = np.array([x, y, 1.0], dtype=np.float64)
        d_cam /= (np.linalg.norm(d_cam) + 1e-12)
        d_world = R.T @ d_cam
        d_world /= (np.linalg.norm(d_world) + 1e-12)
        return d_world

    def _point_to_ray_dist(self, X: np.ndarray, C: np.ndarray, d: np.ndarray) -> float:
        """Distance from 3D point to 3D ray (half-line), in mm."""
        v = X - C
        t = float(np.dot(v, d))
        if t < 0.0:
            return float(np.linalg.norm(v))
        return float(np.linalg.norm(v - t * d))
    
    def _collect_valid_frames(
        self,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        cam_i: int,
        cam_j: int
    ) -> List[int]:
        """Collect frames where both cameras see both A and B."""
        valid = []
        for fid, cam_obs in observations.items():
            if cam_i in cam_obs and cam_j in cam_obs:
                uvA_i, uvB_i = cam_obs[cam_i]
                uvA_j, uvB_j = cam_obs[cam_j]
                if all(x is not None for x in [uvA_i, uvB_i, uvA_j, uvB_j]):
                    valid.append(fid)
        return valid
    
    def _compute_diagnostics(
        self,
        cam_i: int,
        cam_j: int,
        params_i: np.ndarray,
        params_j: np.ndarray,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        valid_frames: List[int],
        K_i: np.ndarray,
        K_j: np.ndarray,
    ) -> dict:
        """Compute diagnostics after optimization."""
        # Build projection matrices
        R_i, _ = cv2.Rodrigues(params_i[:3])
        t_i = params_i[3:6].reshape(3, 1)
        R_j, _ = cv2.Rodrigues(params_j[:3])
        t_j = params_j[3:6].reshape(3, 1)
        
        P_i = K_i @ np.hstack([R_i, t_i])
        P_j = K_j @ np.hstack([R_j, t_j])
        
        wand_lengths = []
        reproj_errors = []
        
        for fid in valid_frames[:200]:
            uvA_i, uvB_i = observations[fid][cam_i]
            uvA_j, uvB_j = observations[fid][cam_j]
            
            # Triangulate
            pts_4d_A = cv2.triangulatePoints(P_i, P_j, 
                                             uvA_i.reshape(2, 1), uvA_j.reshape(2, 1))
            pts_4d_B = cv2.triangulatePoints(P_i, P_j, 
                                             uvB_i.reshape(2, 1), uvB_j.reshape(2, 1))
            
            ptA = (pts_4d_A[:3] / pts_4d_A[3]).flatten()
            ptB = (pts_4d_B[:3] / pts_4d_B[3]).flatten()
            
            wand_lengths.append(np.linalg.norm(ptB - ptA))
            
            # Reprojection error
            proj_Ai = self._project(ptA, R_i, t_i, K_i)
            proj_Aj = self._project(ptA, R_j, t_j, K_j)
            
            reproj_errors.append(np.linalg.norm(proj_Ai - uvA_i))
            reproj_errors.append(np.linalg.norm(proj_Aj - uvA_j))
        
        return {
            'baseline_mm': np.linalg.norm(params_j[3:6]),
            'wand_length_median': np.median(wand_lengths) if wand_lengths else 0,
            'wand_length_std': np.std(wand_lengths) if wand_lengths else 0,
            'wand_length_error': abs(np.median(wand_lengths) - self.config.wand_length_mm) if wand_lengths else float('inf'),
            'reproj_err_mean': np.mean(reproj_errors) if reproj_errors else 0,
            'reproj_err_max': np.max(reproj_errors) if reproj_errors else 0,
            'valid_frames': len(valid_frames),
        }
    
    def _validate(self, report: dict):
        """Validate P0 results. FAIL if constraints violated."""
        print(f"\n{'-'*60}")
        print("[P0 VALIDATION]")
        print(f"{'-'*60}")
        
        b = report['baseline_mm']
        print(f"  Baseline: {b:.2f} mm (recommended min: 50 mm)")
        if b < 50.0:
            print(f"  [WARN] Baseline is below recommended minimum: {b:.2f} mm < 50 mm")
        
        reproj = report.get('reproj_err_mean', 0)
        print(f"  Reproj error mean: {reproj:.2f} px")
        if reproj > 50.0:
            raise RuntimeError(f"[P0 FAIL] Reprojection error too high: {reproj:.2f} px")
        
        wand_err = report.get('wand_length_error', float('inf'))
        print(f"  Wand length error: {wand_err:.4f} mm")
        
        print("[P0 VALIDATION] PASSED")
        print(f"{'-'*60}")
    
    # =========================================================================
    # Phase 2: Calibrate remaining cameras using 3D points from Phase 1
    # =========================================================================
    
    def run_phase2(
        self,
        cam_params: Dict[int, np.ndarray],
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        points_3d: Dict[int, Tuple[np.ndarray, np.ndarray]],
        camera_settings: Dict[int, dict],
        all_cam_ids: List[int],
    ) -> Dict[int, np.ndarray]:
        """
        Phase 2: Calibrate remaining cameras using 3D points from Phase 1.
        
        For each camera not in cam_params:
        - Collect 2D-3D correspondences from points_3d
        - Solve PnP with frozen intrinsics
        """
        dist_coeffs = np.zeros(5)
        
        calibrated_cams = set(cam_params.keys())
        remaining_cams = [c for c in all_cam_ids if c not in calibrated_cams]
        
        if not remaining_cams:
            print("[P0 Phase 2] No remaining cameras to calibrate.")
            return cam_params
        
        print(f"\n{'='*60}")
        print(f"[P0 Phase 2] Calibrating {len(remaining_cams)} remaining cameras")
        print(f"{'='*60}")
        print(f"  Already calibrated: {sorted(calibrated_cams)}")
        print(f"  To calibrate: {remaining_cams}")
        
        for cid in remaining_cams:
            print(f"\n  --- Calibrating cam_{cid} ---")
            
            # Collect 2D-3D correspondences
            pts_2d = []
            pts_3d_list = []
            
            for fid, (XA, XB) in points_3d.items():
                if fid not in observations:
                    continue
                if cid not in observations[fid]:
                    continue
                    
                uvA, uvB = observations[fid][cid]
                if uvA is not None:
                    pts_2d.append(uvA)
                    pts_3d_list.append(XA)
                if uvB is not None:
                    pts_2d.append(uvB)
                    pts_3d_list.append(XB)
            
            if len(pts_2d) < 6:
                print(f"    [WARN] Insufficient correspondences: {len(pts_2d)} < 6. Skipping.")
                continue
            
            pts_2d = np.array(pts_2d, dtype=np.float64)
            pts_3d_arr = np.array(pts_3d_list, dtype=np.float64)
            
            print(f"    Correspondences: {len(pts_2d)}")
            
            K, _, _, _ = self._get_camera_intrinsics(cid, camera_settings)
            # Solve PnP with frozen intrinsics (EPNP + ITERATIVE, like original)
            success, rvec, tvec = cv2.solvePnP(
                pts_3d_arr, pts_2d, K, dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if not success:
                print(f"    [WARN] PnP (EPNP) failed for cam_{cid}. Skipping.")
                continue
            
            # Refine with ITERATIVE
            success, rvec, tvec = cv2.solvePnP(
                pts_3d_arr, pts_2d, K, dist_coeffs,
                rvec, tvec, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            rvec = rvec.flatten()
            tvec = tvec.flatten()
            
            # Compute initial reprojection error
            pts_reproj, _ = cv2.projectPoints(pts_3d_arr, rvec, tvec, K, dist_coeffs)
            pts_reproj = pts_reproj.reshape(-1, 2)
            reproj_err_init = np.sqrt(np.mean(np.sum((pts_2d - pts_reproj)**2, axis=1)))
            
            print(f"    PnP result: RMS = {reproj_err_init:.2f}px")
            
            # Per-camera extrinsic-only optimization (like original Phase 2, but frozen f)
            print(f"    Optimizing extrinsics (frozen intrinsics)...")
            
            x0_cam = np.concatenate([rvec, tvec])  # [rvec(3), tvec(3)]
            
            def residuals_cam(x):
                r = x[:3].reshape(3, 1)
                t = x[3:6].reshape(3, 1)
                pts_proj, _ = cv2.projectPoints(pts_3d_arr, r, t, K, dist_coeffs)
                pts_proj = pts_proj.reshape(-1, 2)
                return (pts_2d - pts_proj).flatten()
            
            result = least_squares(
                residuals_cam, x0_cam,
                method='lm',
                ftol=self.config.ftol,
                xtol=self.config.xtol,
                max_nfev=100,
            )
            
            rvec_opt = result.x[:3]
            tvec_opt = result.x[3:6]
            
            # Compute final reprojection error
            pts_reproj_opt, _ = cv2.projectPoints(pts_3d_arr, rvec_opt, tvec_opt, K, dist_coeffs)
            pts_reproj_opt = pts_reproj_opt.reshape(-1, 2)
            reproj_err_final = np.sqrt(np.mean(np.sum((pts_2d - pts_reproj_opt)**2, axis=1)))
            
            print(f"    rvec: [{rvec_opt[0]:.4f}, {rvec_opt[1]:.4f}, {rvec_opt[2]:.4f}]")
            print(f"    tvec: [{tvec_opt[0]:.2f}, {tvec_opt[1]:.2f}, {tvec_opt[2]:.2f}]")
            print(f"    Reproj RMS: {reproj_err_init:.2f} -> {reproj_err_final:.2f}px")
            
            cam_params[cid] = np.concatenate([rvec_opt, tvec_opt])
        
        print(f"\n[P0 Phase 2] Calibrated {len(cam_params)} cameras total.")
        return cam_params
    
    def triangulate_all_points(
        self,
        cam_i: int,
        cam_j: int,
        params_i: np.ndarray,
        params_j: np.ndarray,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Triangulate all 3D wand points using Phase 1 cameras."""
        K_i, _, _, _ = self._get_camera_intrinsics(cam_i, camera_settings)
        K_j, _, _, _ = self._get_camera_intrinsics(cam_j, camera_settings)
        
        R_i, _ = cv2.Rodrigues(params_i[:3])
        t_i = params_i[3:6].reshape(3, 1)
        R_j, _ = cv2.Rodrigues(params_j[:3])
        t_j = params_j[3:6].reshape(3, 1)
        
        P_i = K_i @ np.hstack([R_i, t_i])
        P_j = K_j @ np.hstack([R_j, t_j])
        
        points_3d = {}
        valid_frames = self._collect_valid_frames(observations, cam_i, cam_j)
        
        for fid in valid_frames:
            uvA_i, uvB_i = observations[fid][cam_i]
            uvA_j, uvB_j = observations[fid][cam_j]
            
            pts_4d_A = cv2.triangulatePoints(P_i, P_j, 
                                             uvA_i.reshape(2, 1), uvA_j.reshape(2, 1))
            pts_4d_B = cv2.triangulatePoints(P_i, P_j, 
                                             uvB_i.reshape(2, 1), uvB_j.reshape(2, 1))
            
            XA = (pts_4d_A[:3] / pts_4d_A[3]).flatten()
            XB = (pts_4d_B[:3] / pts_4d_B[3]).flatten()
            
            points_3d[fid] = (XA, XB)
        
        return points_3d
    
    def run_phase3(
        self,
        cam_params: Dict[int, np.ndarray],
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
        progress_callback=None
    ) -> Dict[int, np.ndarray]:
        """
        Phase 3: Global BA with all cameras and frozen intrinsics.
        
        Joint optimization of all camera extrinsics + 3D points.
        """
        K_by_cam = {}
        for cid in cam_params.keys():
            K_by_cam[cid], _, _, _ = self._get_camera_intrinsics(cid, camera_settings)
        
        all_cam_ids = sorted(cam_params.keys())
        n_cams = len(all_cam_ids)
        cam_id_to_idx = {cid: i for i, cid in enumerate(all_cam_ids)}
        
        print(f"\n{'='*60}")
        print(f"[P0 Phase 3] Global BA with frozen intrinsics")
        print(f"{'='*60}")
        print(f"  Cameras: {all_cam_ids}")
        print("  Frozen intrinsics: per-camera table values")
        
        # Collect valid frames (seen by at least 2 calibrated cameras)
        valid_frames = []
        for fid, cams in observations.items():
            calibrated_in_frame = [c for c in cams.keys() if c in cam_params]
            if len(calibrated_in_frame) >= 2:
                valid_frames.append(fid)
        
        print(f"  Valid frames: {len(valid_frames)}")
        
        if len(valid_frames) < 10:
            print("  [WARN] Not enough frames for Phase 3, skipping.")
            return cam_params
        
        # Triangulate initial 3D points using first available pair per frame
        print("  Triangulating initial points for global BA...")
        pts_3d_init = []
        frame_cams = []  # [(fid, [cams that see this frame])]
        
        for fid in valid_frames:
            cams_in_frame = [c for c in observations[fid].keys() if c in cam_params]
            if len(cams_in_frame) < 2:
                continue
                
            # Use first two cameras for triangulation
            c1, c2 = cams_in_frame[0], cams_in_frame[1]
            p1, p2 = cam_params[c1], cam_params[c2]
            
            R1, _ = cv2.Rodrigues(p1[:3])
            t1 = p1[3:6].reshape(3, 1)
            R2, _ = cv2.Rodrigues(p2[:3])
            t2 = p2[3:6].reshape(3, 1)
            
            P1 = K_by_cam[c1] @ np.hstack([R1, t1])
            P2 = K_by_cam[c2] @ np.hstack([R2, t2])
            
            uvA_1, uvB_1 = observations[fid][c1]
            uvA_2, uvB_2 = observations[fid][c2]
            
            pts_4d_A = cv2.triangulatePoints(P1, P2, uvA_1.reshape(2, 1), uvA_2.reshape(2, 1))
            pts_4d_B = cv2.triangulatePoints(P1, P2, uvB_1.reshape(2, 1), uvB_2.reshape(2, 1))
            
            ptA = (pts_4d_A[:3] / pts_4d_A[3]).flatten()
            ptB = (pts_4d_B[:3] / pts_4d_B[3]).flatten()
            
            pts_3d_init.append(ptA)
            pts_3d_init.append(ptB)
            frame_cams.append((fid, cams_in_frame))
        
        pts_3d_init = np.array(pts_3d_init)
        n_pts = len(pts_3d_init)
        n_frames = len(frame_cams)
        
        print(f"  Initial points: {n_pts}")
        
        # Build state vector: [all_cams(n_cams*6), pts_3d(n_pts*3)]
        n_cam_params = 6  # Only extrinsics
        pt_start = n_cams * n_cam_params
        
        x0 = np.zeros(pt_start + n_pts * 3)
        
        for cid, params in cam_params.items():
            idx = cam_id_to_idx[cid]
            x0[idx * n_cam_params:(idx + 1) * n_cam_params] = params[:6]
        
        x0[pt_start:] = pts_3d_init.flatten()
        
        # Build sparse Jacobian
        from scipy.sparse import lil_matrix
        
        # Count residuals
        n_res = 0
        for fid, cams_in_frame in frame_cams:
            n_res += 1  # wand length
            n_res += len(cams_in_frame) * 4  # 2 points × 2 coords per camera
        
        n_params = len(x0)
        A_sparsity = lil_matrix((n_res, n_params), dtype=int)
        
        ridx = 0
        for frame_idx, (fid, cams_in_frame) in enumerate(frame_cams):
            idx_ptA = pt_start + frame_idx * 6
            idx_ptB = pt_start + frame_idx * 6 + 3
            
            # Wand length
            A_sparsity[ridx, idx_ptA:idx_ptA+3] = 1
            A_sparsity[ridx, idx_ptB:idx_ptB+3] = 1
            ridx += 1
            
            # Reprojection for each camera
            for cid in cams_in_frame:
                cam_idx = cam_id_to_idx[cid]
                cam_start = cam_idx * n_cam_params
                
                # ptA projection (2 residuals)
                A_sparsity[ridx:ridx+2, cam_start:cam_start+6] = 1
                A_sparsity[ridx:ridx+2, idx_ptA:idx_ptA+3] = 1
                ridx += 2
                
                # ptB projection (2 residuals)
                A_sparsity[ridx:ridx+2, cam_start:cam_start+6] = 1
                A_sparsity[ridx:ridx+2, idx_ptB:idx_ptB+3] = 1
                ridx += 2
        
        print(f"  Residuals: {n_res}, Params: {n_params}")
        
        # Residuals function
        self._phase3_res_count = 0
        def residuals_phase3(x):
            # Extract camera params
            cams = {}
            for cid in all_cam_ids:
                idx = cam_id_to_idx[cid]
                p = x[idx * n_cam_params:(idx + 1) * n_cam_params]
                R, _ = cv2.Rodrigues(p[:3])
                t = p[3:6].reshape(3, 1)
                cams[cid] = (R, t)
            
            pts = x[pt_start:].reshape(-1, 3)
            
            res = []
            
            # Track stats for progress reporting
            sq_err_len = 0.0
            n_len = 0
            sq_err_proj = 0.0
            n_proj = 0
            
            for frame_idx, (fid, cams_in_frame) in enumerate(frame_cams):
                ptA = pts[frame_idx * 2]
                ptB = pts[frame_idx * 2 + 1]
                
                # Wand length
                wand_len = np.linalg.norm(ptB - ptA)
                d_len = wand_len - self.config.wand_length_mm
                res.append(d_len)
                
                sq_err_len += d_len**2
                n_len += 1
                
                # Reprojection for each camera
                for cid in cams_in_frame:
                    R, t = cams[cid]
                    uvA, uvB = observations[fid][cid]
                    
                    # Project ptA
                    proj_A = self._project(ptA, R, t, K_by_cam[cid])
                    diffA = proj_A - uvA
                    res.extend(diffA.tolist())
                    sq_err_proj += float(np.sum(diffA**2))
                    n_proj += 2
                    
                    # Project ptB
                    proj_B = self._project(ptB, R, t, K_by_cam[cid])
                    diffB = proj_B - uvB
                    res.extend(diffB.tolist())
                    sq_err_proj += float(np.sum(diffB**2))
                    n_proj += 2

            res_arr = np.array(res)

            # Report progress
            self._phase3_res_count += 1
            if progress_callback and self._phase3_res_count % 5 == 0:
                try:
                    rmse_len = np.sqrt(sq_err_len / max(1, n_len))
                    rmse_ray = -1.0
                    rmse_proj = np.sqrt(sq_err_proj / max(1, n_proj))
                    cost = 0.5 * float(np.sum(res_arr**2))
                    progress_callback(
                        "Use PinHole model to initialize camera parameters...",
                        rmse_ray,
                        rmse_len,
                        rmse_proj,
                        cost,
                    )
                except:
                    pass
            
            return res_arr

        
        # Run global BA
        print("  Running global BA...")
        result = least_squares(
            residuals_phase3, x0,
            jac_sparsity=A_sparsity,
            method='trf',
            x_scale='jac',
            f_scale=1.0,
            verbose=1,
            ftol=self.config.ftol,
            xtol=self.config.xtol,
            max_nfev=100,
        )
        
        print(f"  Phase 3 cost: {result.cost:.2e}")
        
        # Extract optimized params
        cam_params_opt = {}
        for cid in all_cam_ids:
            idx = cam_id_to_idx[cid]
            cam_params_opt[cid] = result.x[idx * n_cam_params:(idx + 1) * n_cam_params]
        
        # Compute final reprojection error
        final_res = residuals_phase3(result.x)
        reproj_res = final_res[n_frames:]  # Skip wand residuals
        rms = np.sqrt(np.mean(reproj_res**2))
        print(f"  Final RMS: {rms:.2f}px")
        
        return cam_params_opt
    
    def run_all(
        self,
        cam_i: int,
        cam_j: int,
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
        camera_settings: Dict[int, dict],
        all_cam_ids: List[int],
        progress_callback=None
    ) -> Tuple[Dict[int, np.ndarray], dict]:
        """
        Run full P0 bootstrap: Phase 1 (8-Point + BA) + Phase 2 (PnP) + Phase 3 (Global BA).
        
        All phases use frozen intrinsics (fixed focal length).
        """
        # Phase 1: Calibrate best pair
        params_i, params_j, report = self.run(
            cam_i, cam_j, observations, camera_settings, progress_callback=progress_callback
        )
        
        cam_params = {
            cam_i: params_i,
            cam_j: params_j,
        }
        
        # Triangulate 3D points
        print(f"\n[P0] Triangulating 3D points for Phase 2...")
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except:
                pass

        points_3d = self.triangulate_all_points(
            cam_i, cam_j, params_i, params_j, observations, camera_settings
        )
        report['points_3d'] = points_3d
        print(f"  Triangulated {len(points_3d)} frames")
        
        # Phase 2: Calibrate remaining cameras
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except:
                pass
        
        cam_params = self.run_phase2(
            cam_params, observations, points_3d, camera_settings, all_cam_ids
        )
        
        # Phase 3: Global BA with all cameras
        if progress_callback:
            try:
                progress_callback("Use PinHole model to initialize camera parameters...", -1, 0, 0, 0)
            except:
                pass

        cam_params = self.run_phase3(
            cam_params, observations, camera_settings, progress_callback=progress_callback
        )

        
        report['all_cam_ids'] = list(cam_params.keys())
        
        return cam_params, report

