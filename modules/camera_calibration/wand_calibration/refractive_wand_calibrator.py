import os
import numpy as np
import cv2
import logging
import json
from datetime import datetime
from pathlib import Path

try:
    import pyopenlpt as lpt
except ImportError:
    lpt = None

from .refractive_geometry import (
    build_pinplate_ray_cpp, closest_distance_rays, triangulate_point, 
    point_to_ray_dist, update_normal_tangent, camera_center,
    align_world_y_to_plane_intersection
)

from .refractive_bootstrap import PinholeBootstrapP0, PinholeBootstrapP0Config, select_best_pair_via_precalib
from .refractive_plane_optimizer import RefractivePlaneOptimizer, OptimizationConfig
from .refractive_bundle_adjustment import RefractiveBAOptimizerPR4, PR4Config, RefractiveBAOptimizerPR5, PR5Config
from scipy.optimize import least_squares


class RefractiveWandCalibrator:
    """
    PR1: Refractive Wand Calibration Framework (MVP)
    #     Handles data collection, strict validation, and refraction-aware I/O.
    #     
    """
    
    def __init__(self, base_calibrator):
        """
        Wrap an existing WandCalibrator to reuse detection data and base params.
        base_calibrator: instance of WandCalibrator
        """
        self.base = base_calibrator
        self.verbose = False
        self.logger = logging.getLogger("RefractiveCalibrator")
    
    # ==========================================
    # Bootstrap Cache Functions
    # ==========================================
    
    def save_bootstrap_cache(self, path, cam_params_by_id, err_px_by_id, active_cam_ids, 
                             chosen_pair, X_A_list, X_B_list, wand_len_mm, cam_ids, num_frames):
        """
        Save bootstrap results to disk to skip Phase1+Phase2 on subsequent runs.
        
        Args:
            path: Path to save cache file (.json)
            cam_params_by_id: Dict[int, np.ndarray(11,)]
            err_px_by_id: Dict[int, float]
            active_cam_ids: List[int]
            chosen_pair: Tuple[int, int]
            X_A_list: Dict[int, np.ndarray(3,)] - fid -> 3D point A
            X_B_list: Dict[int, np.ndarray(3,)] - fid -> 3D point B
            wand_len_mm: float - target wand length used
            cam_ids: List[int] - all camera IDs in dataset
            num_frames: int - number of frames in dataset
        """
        import json
        from pathlib import Path
        
        cache = {
            "version": 1,
            "wand_len_mm": float(wand_len_mm),
            "cam_ids": sorted([int(c) for c in cam_ids]),
            "num_frames": int(num_frames),
            "chosen_pair": [int(chosen_pair[0]), int(chosen_pair[1])],
            "active_cam_ids": [int(c) for c in active_cam_ids],
            "err_px_by_id": {str(k): float(v) for k, v in err_px_by_id.items()},
            "cam_params_by_id": {str(k): v.tolist() for k, v in cam_params_by_id.items()},
            "X_A_list": {str(k): v.tolist() for k, v in X_A_list.items()},
            "X_B_list": {str(k): v.tolist() for k, v in X_B_list.items()}
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(cache, f, indent=2)
        
        print(f"[BOOT][CACHE] Saved to {path}")
    
    def load_bootstrap_cache(self, path, wand_len_target, cam_ids_current, num_frames_current):
        """
        Load bootstrap cache from disk if valid.
        
        Args:
            path: Path to cache file (.json)
            wand_len_target: Expected wand length for validation
            cam_ids_current: Current camera IDs for validation
            num_frames_current: Current frame count for validation
        
        Returns:
            Tuple of (cam_params_by_id, err_px_by_id, active_cam_ids, chosen_pair, X_A_list, X_B_list)
            or None if cache is invalid/missing
        """
        import json
        from pathlib import Path
        
        path = Path(path)
        if not path.exists():
            print(f"[BOOT][CACHE] No cache found at {path}")
            return None
        
        try:
            with open(path, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"[BOOT][CACHE] Failed to load cache: {e}")
            return None
        
        # Validate version
        if cache.get("version") != 1:
            print(f"[BOOT][CACHE] Cache version mismatch (expected 1, got {cache.get('version')})")
            return None
        
        # Validate cam_ids
        cached_cam_ids = sorted(cache.get("cam_ids", []))
        current_cam_ids = sorted([int(c) for c in cam_ids_current])
        if cached_cam_ids != current_cam_ids:
            print(f"[BOOT][CACHE] Cache mismatch: cam_ids differ")
            print(f"  Cached: {cached_cam_ids}")
            print(f"  Current: {current_cam_ids}")
            return None
        
        # Validate num_frames
        cached_num_frames = cache.get("num_frames", 0)
        if cached_num_frames != num_frames_current:
            print(f"[BOOT][CACHE] Cache mismatch: num_frames ({cached_num_frames} != {num_frames_current})")
            return None
        
        # Validate wand_len_mm
        cached_wand_len = cache.get("wand_len_mm", 0)
        if abs(cached_wand_len - wand_len_target) > 1e-6:
            print(f"[BOOT][CACHE] Cache mismatch: wand_len ({cached_wand_len} != {wand_len_target})")
            return None
        
        # Restore types
        try:
            cam_params_by_id = {int(k): np.array(v, dtype=np.float64) for k, v in cache["cam_params_by_id"].items()}
            err_px_by_id = {int(k): float(v) for k, v in cache["err_px_by_id"].items()}
            active_cam_ids = [int(c) for c in cache["active_cam_ids"]]
            chosen_pair = (int(cache["chosen_pair"][0]), int(cache["chosen_pair"][1]))
            X_A_list = {int(k): np.array(v, dtype=np.float64) for k, v in cache["X_A_list"].items()}
            X_B_list = {int(k): np.array(v, dtype=np.float64) for k, v in cache["X_B_list"].items()}
        except Exception as e:
            print(f"[BOOT][CACHE] Failed to restore types: {e}")
            return None
        
        # Validate chosen_pair exists in cam_ids
        if chosen_pair[0] not in cam_params_by_id or chosen_pair[1] not in cam_params_by_id:
            print(f"[BOOT][CACHE] Cache mismatch: chosen_pair {chosen_pair} not in cam_params")
            return None
        
        # Validate X_A_list and X_B_list
        if len(X_A_list) == 0 or len(X_B_list) == 0:
            print(f"[BOOT][CACHE] Cache mismatch: empty X_A_list or X_B_list")
            return None
        
        if len(X_A_list) != len(X_B_list):
            print(f"[BOOT][CACHE] Cache mismatch: X_A_list size ({len(X_A_list)}) != X_B_list size ({len(X_B_list)})")
            return None
        
        # Quick sanity check: median wand length
        sample_fids = list(X_A_list.keys()) # Use ALL frames for consistent stats
        sample_lens = []
        for fid in sample_fids:
            if fid in X_B_list:
                dist = np.linalg.norm(X_B_list[fid] - X_A_list[fid])
                sample_lens.append(dist)
        
        if sample_lens:
            median_len = np.median(sample_lens)
            if median_len < wand_len_target / 2 or median_len > wand_len_target * 2:
                print(f"[BOOT][CACHE] Cache sanity failed: median wand length {median_len:.2f}mm (expected ~{wand_len_target}mm)")
                return None
            print(f"[BOOT][CACHE] Sanity check: median wand length = {median_len:.4f}mm (target: {wand_len_target}mm)")
        
        print(f"[BOOT][CACHE] Loaded OK from {path}")
        print(f"  Cameras: {list(cam_params_by_id.keys())}")
        print(f"  Frames: {len(X_A_list)}")
        print(f"  Best pair: {chosen_pair}")
        
        return (cam_params_by_id, err_px_by_id, active_cam_ids, chosen_pair, X_A_list, X_B_list)
    
    # ==========================================
    # P1 Plane Cache Functions
    # ==========================================
    
    def _compute_config_hash(self, cam_params, cam_to_window, window_media):
        """Compute a short hash of config for cache validation."""
        import hashlib
        # Include rounded camera intrinsics/extrinsics + window config
        parts = []
        for cid in sorted(cam_params.keys()):
            p = cam_params[cid]
            # Round to avoid float precision issues
            rounded = [round(float(x), 3) for x in p[:6]]  # rvec + tvec
            parts.append(f"c{cid}:{rounded}")
        for wid in sorted(window_media.keys()):
            wm = window_media[wid]
            parts.append(f"w{wid}:t{wm.get('thickness', 0):.2f}")
        for cid, wid in sorted(cam_to_window.items()):
            parts.append(f"m{cid}->{wid}")
        
        data = "|".join(parts)
        return hashlib.md5(data.encode()).hexdigest()[:8]
    
    def save_p1_cache(self, path, window_planes, window_media, cam_to_window, 
                      dataset, cam_params, diagnostics=None):
        """
        Save P1 plane optimization results to disk for reuse.
        
        Args:
            path: Path to save cache file (.json)
            window_planes: Dict[wid, {'plane_n': ndarray, 'plane_pt': ndarray}]
            window_media: Dict[wid, {'thickness': float, 'n_obj': float, ...}]
            cam_to_window: Dict[cid, wid]
            dataset: Dataset dict with 'num_frames', 'cam_ids', etc.
            cam_params: Dict[cid, ndarray] - current camera params
            diagnostics: Optional dict with ray_rmse, wand_rmse, etc.
        """
        import json
        from datetime import datetime
        from pathlib import Path
        
        # Build windows dict
        windows = {}
        for wid, pl in window_planes.items():
            wm = window_media.get(wid, {})
            windows[str(wid)] = {
                "plane_normal": pl['plane_n'].tolist(),
                "plane_distance": float(np.dot(pl['plane_n'], pl['plane_pt'])),
                "plane_pt_closest": pl['plane_pt'].tolist(),
                "thickness": float(wm.get('thickness', 10.0)),
                "refraction_indices": [
                    float(wm.get('n_obj', 1.33)),
                    float(wm.get('n_win', 1.52)),
                    float(wm.get('n_air', 1.0))
                ]
            }
        
        cache = {
            "cache_version": "P1_v1",
            "timestamp": datetime.now().isoformat(),
            "conventions": {
                "plane_point_convention": "closest_interface",
                "normal_convention": "camera_to_object",
                "sign_rules": {"s_C": "<0", "s_X": ">0"}
            },
            "dataset_fingerprint": {
                "num_frames": int(dataset.get('num_frames', 0)),
                "camera_ids": sorted([int(c) for c in dataset.get('cam_ids', [])]),
                "cam_to_window": {str(k): int(v) for k, v in cam_to_window.items()},
                "config_hash": self._compute_config_hash(cam_params, cam_to_window, window_media)
            },
            "windows": windows,
            "optimization_summary": diagnostics or {}
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(cache, f, indent=2)
        
        print(f"[P1][CACHE] Plane cache exported to: {path}")
    
    def load_p1_cache(self, path, window_media, cam_to_window, dataset, cam_params):
        """
        Load P1 cache from disk if valid.
        
        Args:
            path: Path to cache file (.json)
            window_media: Current window media config
            cam_to_window: Current camera-to-window mapping
            dataset: Current dataset info
            cam_params: Current camera params (for hash validation)
            
        Returns:
            (window_planes, True) if cache is valid
            (None, False) if cache is invalid/missing
        """
        import json
        from pathlib import Path
        
        if not Path(path).exists():
            return None, False
        
        try:
            with open(path, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"[P1][CACHE] Failed to load: {e}")
            return None, False
        
        # Version check
        if cache.get("cache_version") != "P1_v1":
            print(f"[P1][CACHE] Version mismatch: {cache.get('cache_version')} != P1_v1")
            return None, False
        
        # Convention check
        conventions = cache.get("conventions", {})
        if conventions.get("plane_point_convention") != "closest_interface":
            print(f"[P1][CACHE] Convention mismatch: plane_point_convention")
            return None, False
        if conventions.get("normal_convention") != "camera_to_object":
            print(f"[P1][CACHE] Convention mismatch: normal_convention")
            return None, False
        
        fingerprint = cache.get("dataset_fingerprint", {})
        
        # Camera ID check
        cached_cams = set(fingerprint.get("camera_ids", []))
        current_cams = set(int(c) for c in dataset.get('cam_ids', []))
        if cached_cams != current_cams:
            print(f"[P1][CACHE] Camera mismatch: {cached_cams} != {current_cams}")
            return None, False
        
        # cam_to_window check
        cached_c2w = {int(k): int(v) for k, v in fingerprint.get("cam_to_window", {}).items()}
        current_c2w = {int(k): int(v) for k, v in cam_to_window.items()}
        if cached_c2w != current_c2w:
            print(f"[P1][CACHE] cam_to_window mismatch")
            return None, False
        
        # Config hash check
        current_hash = self._compute_config_hash(cam_params, cam_to_window, window_media)
        cached_hash = fingerprint.get("config_hash", "")
        if current_hash != cached_hash:
            print(f"[P1][CACHE] Config hash mismatch: {cached_hash} != {current_hash}")
            return None, False
        
        # Load windows
        cached_windows = cache.get("windows", {})
        window_planes = {}
        
        for wid_str, wdata in cached_windows.items():
            wid = int(wid_str)
            wm = window_media.get(wid, {})
            
            # Validate thickness
            cached_thick = wdata.get("thickness", 10.0)
            current_thick = wm.get('thickness', 10.0)
            if abs(cached_thick - current_thick) > 0.01:
                print(f"[P1][CACHE] Thickness mismatch for window {wid}: {cached_thick} != {current_thick}")
                return None, False
            
            # Validate RI
            cached_ri = wdata.get("refraction_indices", [1.33, 1.52, 1.0])
            current_ri = [wm.get('n_obj', 1.33), wm.get('n_win', 1.52), wm.get('n_air', 1.0)]
            for i, (cr, curr) in enumerate(zip(cached_ri, current_ri)):
                if abs(cr - curr) > 0.001:
                    print(f"[P1][CACHE] RI mismatch for window {wid}: {cached_ri} != {current_ri}")
                    return None, False
            
            # Validate normal is unit length
            plane_n = np.array(wdata["plane_normal"], dtype=np.float64)
            if abs(np.linalg.norm(plane_n) - 1.0) > 0.01:
                print(f"[P1][CACHE] Invalid normal for window {wid}: not unit length")
                return None, False
            
            plane_pt = np.array(wdata["plane_pt_closest"], dtype=np.float64)
            
            window_planes[wid] = {
                'plane_n': plane_n,
                'plane_pt': plane_pt,
                'initialized': True,
                'thick_mm': current_thick
            }
        
        # Verify we have all required windows
        required_wids = set(cam_to_window.values())
        loaded_wids = set(window_planes.keys())
        if required_wids != loaded_wids:
            print(f"[P1][CACHE] Window mismatch: expected {required_wids}, got {loaded_wids}")
            return None, False
        
        # Print summary
        summary = cache.get("optimization_summary", {})
        print(f"[P1][CACHE] Loaded plane cache successfully from: {path}")
        print(f"  Windows: {list(window_planes.keys())}")
        print(f"  Timestamp: {cache.get('timestamp', 'unknown')}")
        if summary:
            print(f"  Ray RMSE: {summary.get('ray_residual_rmse', 'N/A'):.4f}")
            print(f"  Wand RMSE: {summary.get('wand_length_rmse', 'N/A'):.4f}")
        
        return window_planes, True

    def _collect_observations(self, cam_to_window):
        """
        Aggregate filtered wand points into a structured dict.
        Enforces strict 'fail-fast' consistency (Engineering Guardrail #5).
        """
        # 1. Physical Sanity & Config Validation (Engineering Guardrail #5.2 & #5.4)
        wand_len = getattr(self.base, 'wand_length', 0)
        if wand_len <= 0:
            raise ValueError(f"CRITICAL: Invalid wand length: {wand_len}mm. Must be > 0.")
            
        dist_coeff_num = getattr(self.base, 'dist_coeff_num', 2)
        if dist_coeff_num not in [0, 1, 2]:
            raise ValueError(f"CRITICAL: Invalid distortion coefficient count: {dist_coeff_num}. Expected 0, 1, or 2.")

        # 2. Derive Active Cameras from common visibility (PR1/PR2 requirement)
        if not self.base.wand_points:
            raise ValueError("No wand points detected or loaded.")
        
        # Intersection over all frames to find cameras consistently seeing the wand
        frames_list = sorted(self.base.wand_points.keys())
        active_cams = None
        for fid in frames_list:
            cams_in_frame = set(self.base.wand_points[fid].keys())
            if active_cams is None:
                active_cams = cams_in_frame
            else:
                active_cams = active_cams.intersection(cams_in_frame)
        
        if not active_cams:
            raise ValueError("CRITICAL: No cameras found that consistently see the wand in all frames.")
        
        active_cams = sorted(list(active_cams))

        # 3. Mandatory Mapping Check (Engineering Guardrail #2 & #5.1)
        # MUST cover all active cams found in the data
        for cid in active_cams:
            if cid not in cam_to_window:
                raise ValueError(f"CRITICAL: Camera {cid} is active in data but missing from Window Mapping table.")
            
        # 4. Data Collection (Label-Based: Engineering Guardrail #3 & #5.3)
        obs_data_A = {} # A[frame][cam] = (u,v)
        obs_data_B = {} # B[frame][cam] = (u,v)
        radii_small = {} # small[frame][cam] = r_px
        radii_large = {} # large[frame][cam] = r_px
        mask_A = {}     # mask[frame][cam] = bool
        mask_B = {}
        
        for fid in frames_list:
            obs_data_A[fid] = {}
            obs_data_B[fid] = {}
            radii_small[fid] = {}
            radii_large[fid] = {}
            mask_A[fid] = {}
            mask_B[fid] = {}
            frame_data = self.base.wand_points[fid]
            
            for cid in active_cams:
                pts = frame_data[cid]
                
                uvA = None
                uvB = None
                rA = None
                rB = None
                
                for pt in pts:
                    # Point format from modified WandCalibrator: [x, y, r, metric, status, pt_idx]
                    if len(pt) < 5:
                        # Fallback for old data or direct detection without labels
                        continue
                    
                    label = pt[4]
                    pt_idx = pt[5] if len(pt) >= 6 else -1
                    radius_px = float(pt[2]) if len(pt) >= 3 else 0.0
                    
                    # PointIdx Consistency Check (Recommended)
                    if pt_idx != -1:
                        target_id = 0 if label == "Filtered_Small" else 1
                        if pt_idx != target_id:
                            print(f"  [Warning] Consistency Mismatch: Frame {fid} Cam {cid} {label} has PointIdx={pt_idx} (Expected {target_id}). Priority: Label.")
                    
                    if label == "Filtered_Small":
                        if uvA is not None:
                            raise ValueError(f"CRITICAL: Duplicate 'Filtered_Small' in Frame {fid} Cam {cid}.")
                        uvA = pt[:2]
                        rA = radius_px
                    elif label == "Filtered_Large":
                        if uvB is not None:
                            raise ValueError(f"CRITICAL: Duplicate 'Filtered_Large' in Frame {fid} Cam {cid}.")
                        uvB = pt[:2]
                        rB = radius_px

                # Rule: Allow A or B to exist independently
                if uvA is not None:
                    obs_data_A[fid][cid] = uvA
                    radii_small[fid][cid] = rA
                    mask_A[fid][cid] = True
                else:
                    mask_A[fid][cid] = False
                
                if uvB is not None:
                    obs_data_B[fid][cid] = uvB
                    radii_large[fid][cid] = rB
                    mask_B[fid][cid] = True
                else:
                    mask_B[fid][cid] = False

                # Finite check if they exist
                if uvA is not None and not np.all(np.isfinite(uvA)):
                    raise ValueError(f"CRITICAL: Infinite A coordinates in Frame {fid} Cam {cid}.")
                if uvB is not None and not np.all(np.isfinite(uvB)):
                    raise ValueError(f"CRITICAL: Infinite B coordinates in Frame {fid} Cam {cid}.")

        dataset = {
            "frames": frames_list,
            "cam_ids": active_cams,
            "obsA": obs_data_A,
            "obsB": obs_data_B,
            "radii_small": radii_small,
            "radii_large": radii_large,
            "maskA": mask_A,
            "maskB": mask_B,
            "num_frames": len(frames_list),
            "num_cams": len(active_cams),
            "wand_length": wand_len,
            "dist_coeff_num": dist_coeff_num,
            "total_observations": len(frames_list) * len(active_cams)
        }
        return dataset


    def _prepare_observations_for_bootstrap(self, cam_to_window: dict) -> dict:
        """
        Prepare observations in format for P0 bootstrap: {fid: {cid: (uvA, uvB)}}.
        
        Args:
            cam_to_window: Camera to window mapping (used for context)
            
        Returns:
            observations: {fid: {cid: (uvA, uvB)}}
        """
        dataset = self._collect_observations(cam_to_window)
        
        observations = {}
        for fid in dataset['frames']:
            observations[fid] = {}
            for cid in dataset['cam_ids']:
                uvA = dataset['obsA'][fid].get(cid)
                uvB = dataset['obsB'][fid].get(cid)
                if uvA is not None and uvB is not None:
                    observations[fid][cid] = (np.array(uvA), np.array(uvB))
        
        return observations


    def _init_window_planes_from_cameras(self, cam_params, cam_to_window, window_media, err_px, 
                                          X_A_list=None, X_B_list=None, active_cam_ids=None):
        """
        Initialize window planes using cameras assigned to each window.
        
        Properly handles:
        - cams_all: All cameras mapped to this window (from cam_to_window)
        - cams_used: Active cameras that will be used for ray-building (constraints)
        
        Normal is computed from cams_used centers to object mean.
        Inactive cameras are logged as warnings, not hard errors.
        
        Args:
            cam_params: dict[cid] -> [rvec(3), tvec(3), f, cx, cy, k1, k2]
            cam_to_window: dict[cid] -> window_id (FULL mapping, not filtered)
            window_media: dict[wid] -> {n1, n2, n3, thickness}
            err_px: dict[cid] -> per-camera pinhole reprojection error (px)
            X_A_list: dict[fid] -> 3D point A from bootstrap (scaled, in mm)
            X_B_list: dict[fid] -> 3D point B from bootstrap (scaled, in mm)
            active_cam_ids: list of camera IDs that are active (will be used for ray-building)
        
        Returns:
            window_planes: dict[wid] -> {'plane_pt': np.array, 'plane_n': np.array}
        """
        # Default: all cameras are active if not specified
        if active_cam_ids is None:
            active_cam_ids = list(cam_params.keys())
        active_cam_set = set(active_cam_ids)
        
        # Collect bootstrap midpoints (X_mid = 0.5*(XA + XB))
        X_mids = []
        if X_A_list and X_B_list:
            sample_fids = list(X_A_list.keys())[:500]
            for fid in sample_fids:
                if fid in X_B_list:
                    X_mid = 0.5 * (np.array(X_A_list[fid]) + np.array(X_B_list[fid]))
                    X_mids.append(X_mid)
        
        # Group cameras by window: cams_all (from mapping) and cams_used (active only)
        window_cams_all = {}
        window_cams_used = {}
        for cid, wid in cam_to_window.items():
            if wid not in window_cams_all:
                window_cams_all[wid] = []
                window_cams_used[wid] = []
            window_cams_all[wid].append(cid)
            if cid in active_cam_set and cid in cam_params:
                window_cams_used[wid].append(cid)
        
        window_planes = {}
        
        for wid in sorted(window_cams_all.keys()):
            cams_all = window_cams_all[wid]
            cams_used = window_cams_used.get(wid, [])
            cams_inactive = [c for c in cams_all if c not in cams_used]
            
            print(f"\n[WIN_INIT] Win {wid}:")
            print(f"  cams_all (from cam_to_window) = {cams_all}")
            print(f"  cams_used (active, for constraints) = {cams_used}")
            
            if cams_inactive:
                for c in cams_inactive:
                    print(f"  [WARN] Cam {c} inactive -> excluded from plane constraints")
            
            if not cams_used:
                print(f"  [ERROR] Win {wid}: No active cameras! Skipping plane init.")
                continue
            
            # === 1. Compute camera centers for cams_used only ===
            centers = {}
            for cid in cams_used:
                p = cam_params[cid]
                rvec, tvec = p[0:3], p[3:6]
                R, _ = cv2.Rodrigues(rvec)
                centers[cid] = -R.T @ tvec
            
            C_mean = np.mean([centers[c] for c in cams_used], axis=0)
            if self.verbose:
                print(f"  C_mean (cams_used) = {C_mean.round(2)}")
            
            # === 2. Compute mean object point ===
            if X_mids:
                X_mean = np.mean(X_mids, axis=0)
                if self.verbose:
                    print(f"  X_mean (bootstrap midpoints) = {X_mean.round(2)}")
                
                # === 3. Compute normal: Robust Mean Optical Axis (Weighted, Safe) ===
                # C++ convention: plane.norm_vector points AWAY from camera (Camera -> Object)
                optical_axes = []
                # Use legacy direction (X-C) as a guide for hemisphere check
                ref_dir_guide = X_mean - C_mean
                ref_dir_guide /= (np.linalg.norm(ref_dir_guide) + 1e-12)

                for cid in cams_used:
                    p = cam_params[cid]
                    R, _ = cv2.Rodrigues(p[0:3])
                    # Axis in world = R.T * [0,0,1]
                    axis = (R.T @ np.array([0, 0, 1.0])).reshape(3,)
                    axis /= (np.linalg.norm(axis) + 1e-12) # Safe normalize
                    
                    optical_axes.append(axis)
                
                optical_axes = np.array(optical_axes)
                
                # Double Flip Check:
                # 1. Align all to their own rough mean (internal consistency)
                mean_internal = np.mean(optical_axes, axis=0)
                if np.linalg.norm(mean_internal) < 1e-6: mean_internal = optical_axes[0]
                mean_internal /= (np.linalg.norm(mean_internal) + 1e-12)

                aligned_axes = []
                for axis in optical_axes:
                    if np.dot(axis, mean_internal) < 0: axis *= -1.0 # Internal flip
                    aligned_axes.append(axis)
                aligned_axes = np.array(aligned_axes)

                # 2. Align the group trend to scene direction (global consistency)
                mean_aligned = np.mean(aligned_axes, axis=0)
                na = np.linalg.norm(mean_aligned)
                if na < 1e-6:
                    mean_aligned = mean_internal # fallback to stable direction
                else:
                    mean_aligned /= (na + 1e-12)

                if np.dot(mean_aligned, ref_dir_guide) < 0:
                     aligned_axes *= -1.0 # Global flip ALL
                
                # Robust Weighted Mean (Power 2 weighting against rough mean)
                mean_rough = np.mean(aligned_axes, axis=0)
                norm_mr = np.linalg.norm(mean_rough)
                if norm_mr < 1e-6: mean_rough = aligned_axes[0]
                else: mean_rough /= norm_mr
                
                weights = []
                for axis in aligned_axes:
                    w = max(0.0, np.dot(axis, mean_rough))**2
                    weights.append(w)
                
                weights = np.array(weights)
                w_sum = np.sum(weights)
                if w_sum < 1e-12:
                     # Fallback to uniform mean
                     n0 = np.mean(aligned_axes, axis=0)
                else:
                     n0 = np.average(aligned_axes, axis=0, weights=weights)

                n_win = n0 / (np.linalg.norm(n0) + 1e-12)
                
                # Diagnostic
                dot_val = np.dot(n_win, ref_dir_guide)
                angle_deg = np.degrees(np.arccos(np.clip(dot_val, -1.0, 1.0)))
                
                if self.verbose:
                    print(f"  n_win (Robust Weighted) = {n_win.round(4)}")
                    print(f"  [DIAG] Angle vs (X-C): {angle_deg:.2f} deg")
                    print(f"  [CONVENTION] n_win points: camera -> object (away from camera)")
                
                # === 4. Compute depths for d0 calculation ===
                # depth = dot(n_win, X - C)
                # Also log side-check stats here for robustness
                s_vals_c = []
                depths = []
                for cid in cams_used:
                    C_cam = centers[cid]
                    s_c = np.dot(n_win, C_cam - (C_mean + n_win * 100)) # illustrative, will check later properly
                    # Real side check relies on plane_pt which needs d0 first.
                    
                    for X_mid in X_mids[:200]:
                        depth = np.dot(n_win, X_mid - C_cam)
                        if depth > 0:
                            depths.append(depth)
                
                depth_med = np.median(depths) if depths else np.linalg.norm(X_mean - C_mean)
                
                # Side Check Logic (using C_mean as anchor stats)
                s_X_vals = [np.dot(n_win, X - C_mean) for X in X_mids[:100]]
                obj_pct = np.mean(np.array(s_X_vals) > 0) * 100.0 if s_X_vals else 0.0
                if self.verbose:
                    print(f"  [DIR] Using C_mean anchor (NOT plane-side check): {obj_pct:.1f}% points have dot(n, X-C_mean)>0.")
                    print(f"  depth_med = {depth_med:.1f} mm (from {len(depths)} samples)")
                
                # === 5. Compute d0 with data-driven bounds ===
                thick_mm = window_media.get(wid, {}).get('thickness', 10.0)
                d0_min = max(10.0 * thick_mm, 0.02 * depth_med, 20.0)
                d0_max = min(1.2 * depth_med, 2500.0)
                d0_init = 0.4 * depth_med
                d0_mm = np.clip(d0_init, d0_min, d0_max)
                
                if self.verbose:
                    print(f"  d0_init = 0.4 * {depth_med:.1f} = {d0_init:.1f} mm")
                    print(f"  d0 range = [{d0_min:.1f}, {d0_max:.1f}], d0_final = {d0_mm:.1f} mm")
                
                # === 6. Compute plane point (farthest plane) ===
                # plane_pt = C_mean + n_win * d0 (advance from camera toward object)
                plane_pt = C_mean + n_win * d0_mm
                if self.verbose:
                    print(f"  plane_pt = C_mean + n*d0 = {plane_pt.round(2)}")
                
            else:
                print(f"  [WARNING] No bootstrap points - using fallback")
                n_win = np.array([0.0, 0.0, 1.0])
                d0_mm = 100.0
                depth_med = 200.0
                plane_pt = C_mean + n_win * d0_mm
            
            # === 7. Orientation check: try both n and -n ===
            # C++ convention: 
            #   - signed_dist(C) = dot(n, C - plane_pt) < 0 for cameras (camera-side)
            #   - signed_dist(X) = dot(n, X - plane_pt) > 0 for objects (object-side)
            def compute_score(n_test):
                """Compute orientation score using C++ convention."""
                cams_ok = 0
                for cid in cams_used:
                    s = np.dot(n_test, centers[cid] - plane_pt)
                    if s < 0:  # Camera should be on camera-side (s < 0)
                        cams_ok += 1
                
                obj_ok = 0
                if X_mids:
                    for X_mid in X_mids[:200]:
                        sX = np.dot(n_test, X_mid - plane_pt)
                        if sX > 0:  # Objects should be on object-side (s > 0)
                            obj_ok += 1
                    obj_pct = 100.0 * obj_ok / min(len(X_mids), 200)
                else:
                    obj_pct = 100.0
                
                return cams_ok, obj_pct
            
            cams_ok_pos, obj_pct_pos = compute_score(n_win)
            cams_ok_neg, obj_pct_neg = compute_score(-n_win)
            
            print(f"\n[WIN_ORIENT] Win {wid}: Checking orientations (C++ convention: s(C)<0, s(X)>0)...")
            if self.verbose:
                print(f"  n_win:  cams_ok={cams_ok_pos}/{len(cams_used)}, obj_pct={obj_pct_pos:.1f}%")
                print(f"  -n_win: cams_ok={cams_ok_neg}/{len(cams_used)}, obj_pct={obj_pct_neg:.1f}%")
            
            # Choose orientation with all cams OK and best object percentage
            if cams_ok_pos == len(cams_used) and obj_pct_pos >= 50.0:
                print(f"  [OK] Using n_win (all cams camera-side, {obj_pct_pos:.1f}% objects object-side)")
            elif cams_ok_neg == len(cams_used) and obj_pct_neg >= 50.0:
                n_win = -n_win
                print(f"  [OK] Using -n_win (all cams camera-side, {obj_pct_neg:.1f}% objects object-side)")
            elif cams_ok_pos == len(cams_used):
                print(f"  [WARN] Using n_win (all cams OK, but only {obj_pct_pos:.1f}% objects object-side)")
            elif cams_ok_neg == len(cams_used):
                n_win = -n_win
                if self.verbose:
                    print(f"  [WARN] Using -n_win (all cams OK, but only {obj_pct_neg:.1f}% objects object-side)")
            else:
                # Neither orientation satisfies all cameras
                if cams_ok_pos >= cams_ok_neg:
                    n_choice, cams_ok_choice = n_win, cams_ok_pos
                else:
                    n_win = -n_win
                    cams_ok_choice = cams_ok_neg
                print(f"  [ERROR] Cannot satisfy all cams! Best: {cams_ok_choice}/{len(cams_used)} cams camera-side")
                print(f"  This may indicate: (1) cam_to_window mapping error, or (2) cameras on opposite sides of window")
            
            # === 8. Final side check statistics (A3) ===
            print(f"\n[WIN_SANITY] Win {wid}: s = dot(n_win, P - plane_pt)")
            
            
            # Warn individually if wrong
            if self.verbose:
                for cid in cams_used:
                    s = np.dot(n_win, centers[cid] - plane_pt)
                    if s >= 0:
                        print(f"  [WARN] Cam {cid} s={s:.2f} mm >= 0 (WRONG side)")

            # Stats for cams
            s_cams = [np.dot(n_win, centers[c] - plane_pt) for c in cams_used]
            if s_cams:
                min_s_cam = min(s_cams)
                max_s_cam = max(s_cams)
                print(f"  [WIN_SANITY][STATS] cams_used s(C): min={min_s_cam:.2f} mm, max={max_s_cam:.2f} mm (expect <0)")
            
            # Stats for objects
            obj_pct_plane = 0.0
            if X_mids:
                s_objs = [np.dot(n_win, X - plane_pt) for X in X_mids[:200]]
                obj_pct_plane = np.mean(np.array(s_objs) > 0) * 100.0
                print(f"  [WIN_SANITY][STATS] objects on object-side (s>0): {obj_pct_plane:.1f}%")
            
            # === 9. Key invariant check ===
            dot_n_to_plane = np.dot(n_win, plane_pt - C_mean)
            print(f"\n  [KEY INVARIANT] dot(n, plane_pt - C_mean) = {dot_n_to_plane:.2f} (MUST be > 0)")
            if dot_n_to_plane <= 0:
                print(f"  [CRITICAL] KEY INVARIANT VIOLATED! Normal direction is WRONG!")
            
            # Store result
            window_planes[wid] = {
                'plane_pt': plane_pt,
                'plane_n': n_win,
                'thick_mm': thick_mm,
                'initialized': True
            }
            
            print(f"\n  n_win (final) = {n_win.round(4)}")
            print(f"  plane_pt (final) = {plane_pt.round(2)}")
            if self.verbose:
                print(f"  Plane at {d0_mm:.1f}mm from cameras, objects at ~{depth_med:.1f}mm")
        
        return window_planes
        
    def _estimate_and_log_sphere_radii(self, dataset, cam_params, points_3d_A, points_3d_B, tag="P1"):
        """
        Step 1: Estimate and Log sphere radii (mm)
        Model: R_mm = r_px * Zc / f
        """
        print(f"\n[SPHERE_RADIUS_ESTIMATION]")
        print(f"  Method: R_mm = r_px * Zc / fx")
        print(f"  Computed at: {tag} initialization (once per round)")
        
        radii_small = dataset.get('radii_small', {})
        radii_large = dataset.get('radii_large', {})
        
        # Returns median (or 0.0 if fail)
        def compute_stats(name, r_map, pts_3d):
            vals_global = []
            vals_per_cam = {cid: [] for cid in cam_params.keys()}
            
            for fid, cam_dict in r_map.items():
                if fid not in pts_3d or pts_3d[fid] is None:
                    continue
                X_world = pts_3d[fid] # (3,)
                
                for cid, r_px in cam_dict.items():
                    if cid not in cam_params: continue
                    if r_px <= 0: continue
                    
                    # Project X_world to Cam
                    # C = -R.T @ t -> this is center.
                    # We need X_cam = R @ X_world + t
                    p = cam_params[cid]
                    rvec, tvec = p[0:3], p[3:6]
                    f = p[6]
                    
                    R, _ = cv2.Rodrigues(rvec)
                    X_cam = R @ X_world + tvec
                    Zc = X_cam[2]
                    
                    if Zc <= 10.0: continue # Sanity check (too close or behind)
                    
                    R_mm = r_px * Zc / f
                    vals_global.append(R_mm)
                    vals_per_cam[cid].append(R_mm)
            
            print(f"\n  Endpoint {name}:")
            if not vals_global:
                print(f"    [WARNING] Insufficient samples for endpoint {name} (N=0)")
                return 0.0
            
            # Global Stats
            v = np.array(vals_global)
            med = np.median(v)
            p10 = np.percentile(v, 10)
            p90 = np.percentile(v, 90)
            vmin = np.min(v)
            vmax = np.max(v)
            
            print(f"    Global:")
            print(f"      median = {med:.2f} mm")
            print(f"      p10 / p90 = {p10:.2f} / {p90:.2f} mm")
            print(f"      min / max = {vmin:.2f} / {vmax:.2f} mm")
            
            print(f"\n    Per-camera medians:")
            for cid in sorted(vals_per_cam.keys()):
                vc = vals_per_cam[cid]
                if vc:
                    print(f"      Cam {cid}: {np.median(vc):.2f} mm")
                else:
                    print(f"      Cam {cid}: N/A")
            return float(med)
                    
        r_small = compute_stats("A (Small Sphere)", radii_small, points_3d_A)
        r_large = compute_stats("B (Large Sphere)", radii_large, points_3d_B)
        print("")
        return r_small, r_large

    def export_camfile_with_refraction(self, out_dir, cam_params, window_media, cam_to_window, window_planes=None):
        """
        Export PINPLATE camFiles per camera in a directory.
        Strictly follows Camera::loadParameters (PINPLATE branch) from Camera.cpp.
        
        Format rules:
        - Comments are whole lines starting with #
        - Single-token fields use comma separation (no spaces): n_row,n_col
        - Multi-value fields use space separation
        """
        from pathlib import Path
        
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        print(f"\n[Refractive][CAMFILE] Exporting PINPLATE files to {out_dir}")
        print(f"[Refractive][MAP] cam_to_window: {dict(cam_to_window)}")

        for cid in sorted(cam_params.keys()):
            # Vector format: rvec(3), tvec(3), f, cx, cy, k1, k2
            p = cam_params[cid] 
            rvec, tvec = p[0:3], p[3:6]
            f, cx, cy = p[6], p[7], p[8]
            k1, k2 = p[9], p[10]
            
            # Setup PINPLATE specific data
            wid = cam_to_window[cid]
            media = window_media[wid]
            
            thick_mm = media.get('thickness', 0.0)
            
            # Use initialized window planes if provided
            if window_planes and wid in window_planes:
                wp = window_planes[wid]
                plane_pt = wp.get('plane_pt')
                plane_norm = wp.get('plane_n')
                
                # Minimum Validity Condition: keys exist, shape(3), non-NaN, unit normal, positive thickness
                data_valid = (plane_pt is not None and plane_norm is not None and 
                              np.shape(plane_pt) == (3,) and np.shape(plane_norm) == (3,) and
                              np.all(np.isfinite(plane_pt)) and np.all(np.isfinite(plane_norm)) and
                              0.99 <= np.linalg.norm(plane_norm) <= 1.01 and
                              thick_mm > 0.0)
                
                if data_valid:
                    plane_pt = np.array(plane_pt)
                    plane_norm = np.array(plane_norm)
                    # Auto-repair flag if missing in valid data
                    if not wp.get('initialized', False):
                        wp['initialized'] = True
                else:
                    # Fallback if corrupt
                    plane_pt = np.array([0.0, 0.0, 300.0])
                    plane_norm = np.array([0.0, 0.0, 1.0])
                    thick_mm = 10.0 # Force fallback thickness if <= 0
                    print(f"  [FALLBACK] Cam {cid} (Win {wid}): Corrupt plane data (NaN/thick<=0/non-unit n)! Using Z=300, t=10.")
            else:
                # Fallback: default plane at z=300mm (debug only)
                plane_pt = np.array([0.0, 0.0, 300.0])
                plane_norm = np.array([0.0, 0.0, 1.0])
                if thick_mm <= 0.0: thick_mm = 10.0
                print(f"  [FALLBACK] Cam {cid} (Win {wid}): Missing plane data! Using default Z=300, t={thick_mm}.")
            
            n_air = media.get('n_air', media.get('n1', 1.0))
            n_win = media.get('n_window', media.get('n2', 1.49))
            n_obj = media.get('n_object', media.get('n3', 1.33))

            file_name = f"cam{cid}.txt"
            file_path = out_path / file_name
            
            # Rotation matrices
            R, _ = cv2.Rodrigues(rvec)
            R_inv = R.T
            t_vec = tvec.flatten()
            t_vec_inv = (-R_inv @ tvec).flatten()
            
            # =========================================================================
            # Consistency Check #1: t_vec_inv = -R_inv * T
            # =========================================================================
            expected_t_inv = (-R_inv @ tvec).flatten()
            diff_t_inv = np.linalg.norm(t_vec_inv - expected_t_inv)
            assert diff_t_inv < 1e-6, f"Cam {cid}: t_vec_inv inconsistent! diff={diff_t_inv}"
            
            # =========================================================================
            # Consistency Check #2: refract_array and w_array lengths
            # Route B Convention: Farthest -> Nearest
            # Physical order (ray): Camera(Air) -> Window -> Object
            # Storage order (camFile): [n_ref_obj, n_ref_win, n_ref_air] Farthest -> Nearest
            refract_array = [n_obj, n_win, n_air]
            w_array = [thick_mm]
            n_plate = len(w_array)
            
            # --- Convert Plane Point to Farthest Interface (Route B Contract) ---
            # Current 'plane_pt' is initialized as the closest interface (Air->Glass)
            P_closest = plane_pt.copy()
            
            # P_farthest = P_closest + n_geom * total_thickness (along away-from-cam normal)
            # Assuming n_plate=1 for MVP, thickness is just thick_mm.
            P_farthest = P_closest + plane_norm * thick_mm
            
            # Use P_farthest for export
            plane_pt_export = P_farthest
            
            # Sanity Check
            dot_check = np.dot(plane_norm, P_farthest - P_closest)
            print(f"  [CAMFILE][SANITY] Cam {cid}: Shifted plane point to farthest interface.")
            print(f"    P_closest: {P_closest}")
            print(f"    P_farthest (Export): {P_farthest}")
            print(f"    Shift along n: {dot_check:.4f} mm (Expected: {thick_mm:.4f} mm)")
            
            assert len(refract_array) == len(w_array) + 2, \
                f"Cam {cid}: len(refract_array)={len(refract_array)} != len(w_array)+2={len(w_array)+2}"
            assert len(refract_array) >= 3, f"Cam {cid}: refract_array too short"
            
            # Build single-token strings
            h, w = self.base.image_size if hasattr(self.base, 'image_size') else (800, 1280)
            img_size_str = f"{h},{w}"  # n_row,n_col as ONE token
            dist_coeff_str = f"{k1:.8g},{k2:.8g},0,0,0"  # ONE token
            rvec_placeholder = "0,0,0"  # ONE token (unused)
            refract_str = ",".join([f"{n:.4f}" for n in refract_array])  # ONE token
            w_str = ",".join([f"{w:.4f}" for w in w_array])  # ONE token
            
            with open(file_path, 'w') as f_out:
                # --- Header ---
                f_out.write("PINPLATE\n")
                
                # --- Error placeholders (2 tokens, unused) ---
                f_out.write("# (unused) error placeholders\n")
                f_out.write("0.0 0.0\n")
                f_out.write("\n")
                
                # --- Image Size ---
                f_out.write("# Image Size: (n_row,n_col) as ONE token\n")
                f_out.write(f"{img_size_str}\n")
                f_out.write("\n")
                
                # --- Camera Matrix K (3x3) ---
                f_out.write("# Camera Matrix K (3x3)\n")
                f_out.write(f"{f:.8g} 0 {cx:.8g}\n")
                f_out.write(f"0 {f:.8g} {cy:.8g}\n")
                f_out.write("0 0 1\n")
                f_out.write("\n")
                
                # --- Distortion coefficients ---
                f_out.write("# Distortion coefficients (ONE token, comma-separated)\n")
                f_out.write(f"{dist_coeff_str}\n")
                f_out.write("\n")
                
                # --- Rotation vector placeholder (unused by PINPLATE) ---
                f_out.write("# Rotation vector (unused placeholder, ONE token)\n")
                f_out.write(f"{rvec_placeholder}\n")
                f_out.write("\n")
                
                # --- Rotation Matrix R (world -> camera) ---
                f_out.write("# Rotation Matrix R (world->camera)\n")
                for row in R:
                    f_out.write(f"{row[0]:.8g} {row[1]:.8g} {row[2]:.8g}\n")
                f_out.write("\n")
                
                # --- Inverse Rotation Matrix R_inv ---
                f_out.write("# Inverse Rotation Matrix R_inv\n")
                for row in R_inv:
                    f_out.write(f"{row[0]:.8g} {row[1]:.8g} {row[2]:.8g}\n")
                f_out.write("\n")
                
                # --- Translation Vector T (world -> camera) ---
                f_out.write("# Translation Vector T (world->camera)\n")
                f_out.write(f"{t_vec[0]:.8g} {t_vec[1]:.8g} {t_vec[2]:.8g}\n")
                f_out.write("\n")
                
                # --- Inverse Translation Vector = camera center in world ---
                f_out.write("# Inverse Translation Vector (-R_inv*T) = camera center in world\n")
                f_out.write(f"{t_vec_inv[0]:.8g} {t_vec_inv[1]:.8g} {t_vec_inv[2]:.8g}\n")
                f_out.write("\n")
                
                f_out.write("# Refractive plane reference point plane.pt (Farthest Interface)\n")
                f_out.write(f"{plane_pt_export[0]:.8g} {plane_pt_export[1]:.8g} {plane_pt_export[2]:.8g}\n")
                f_out.write("\n")
                
                # --- Refractive Plane Normal ---
                f_out.write("# Refractive plane normal plane.norm_vector (camera->object direction)\n")
                f_out.write(f"{plane_norm[0]:.8g} {plane_norm[1]:.8g} {plane_norm[2]:.8g}\n")
                f_out.write("\n")
                
                # --- Refractive Index Array ---
                f_out.write(f"# refract_array (ONE token, comma-separated, farthest->nearest: obj->win->air)\n")
                f_out.write(f"# n_plate = {n_plate}\n")
                f_out.write(f"{refract_str}\n")
                f_out.write("\n")
                
                # --- Plate Thickness Array ---
                f_out.write("# w_array (ONE token, comma-separated, plate thicknesses in mm)\n")
                f_out.write(f"{w_str}\n")
                f_out.write("\n")
                
                # --- Projection Parameters ---
                f_out.write("# proj_tol\n")
                f_out.write("1e-6\n")
                f_out.write("# proj_nmax\n")
                f_out.write("50\n")
                f_out.write("# lr (learning rate)\n")
                f_out.write("0.1\n")
                f_out.write("\n")
                
                # --- Metadata ---
                f_out.write("# --- BEGIN_REFRACTION_META ---\n")
                f_out.write(f"# VERSION=2\n")
                f_out.write(f"# CAM_ID={cid}\n")
                f_out.write(f"# WINDOW_ID={wid}\n")
                f_out.write(f"# PLANE_PT_EXPORT=[{plane_pt_export[0]:.4f},{plane_pt_export[1]:.4f},{plane_pt_export[2]:.4f}]\n")
                f_out.write(f"# PLANE_N=[{plane_norm[0]:.6f},{plane_norm[1]:.6f},{plane_norm[2]:.6f}]\n")
                f_out.write("# --- END_REFRACTION_META ---\n")
            
            # --- PER-CAMERA LOGGING ---
            print(f"  [CAMFILE] Cam {cid} -> Win {wid}")
            print(f"    plane_pt (Farthest): [{plane_pt_export[0]:.2f}, {plane_pt_export[1]:.2f}, {plane_pt_export[2]:.2f}]")
            print(f"    plane_n:  [{plane_norm[0]:.6f}, {plane_norm[1]:.6f}, {plane_norm[2]:.6f}]")
            print(f"    refract:  [{n_obj:.4f}, {n_win:.4f}, {n_air:.4f}]")
            print(f"    thick:    {thick_mm:.4f} mm")
            print(f"    file:     {file_path}")
                
            exported_files.append(str(file_path))
        
        # --- VERIFICATION ASSERTION ---
        used_wids = sorted(set(cam_to_window[cid] for cid in cam_params.keys()))
        num_windows = len(window_media)
        print(f"\n[Refractive][MAP] used window ids: {used_wids}, num_windows={num_windows}")
        assert max(used_wids) < num_windows, f"Window ID {max(used_wids)} exceeds num_windows {num_windows}"
        assert len(used_wids) >= 1, "No windows used"
            
        return out_dir

    def _update_camera_cpp_via_assign(self, cam_obj, pinplate_params_dict):
        """
        Force update a C++ Camera object by explicitly daisy-chaining assignments.
        This handles pybind11's value-copy behavior for nested structs.
        
        Logic:
          pp = cam.param         (Copy A)
          pl = pp.plane          (Copy B)
          pl.pt = ...            (Modify B)
          pp.plane = pl          (Update A with B)
          cam.param = pp         (Update Camera with A)
        """
        # 1. Get Main Param Copy
        pp = cam_obj._pinplate_param
        
        # 2. Get Plane Copy (Nested Struct)
        pl = pp.plane
        
        # 3. Modify Plane Copy
        # Inputs are lists [x,y,z], need to match pybind type (likely list/array compatible)
        # ERROR FIX: Must convert list to lpt.Pt3D explicit object if pybind doesn't auto-convert
        # Try lpt.Pt3D(x, y, z) constructor if available, or just verify type.
        # Given error: (self: pyopenlpt.Plane3D, arg0: pyopenlpt.Pt3D) -> None
        # We must instantiate lpt.Pt3D.
        
        pt_list = pinplate_params_dict['plane_pt']
        n_list = pinplate_params_dict['plane_n']
        
        # Check if lpt is imported available? It is at top level.
        # Use * unpacking if constructor takes 3 args, or list if 1 arg.
        # Usually Pt3D(x, y, z).
        pl.pt = lpt.Pt3D(pt_list[0], pt_list[1], pt_list[2])
        pl.norm_vector = lpt.Pt3D(n_list[0], n_list[1], n_list[2])
        
        # 4. Assign Plane Back to Param (Crucial Step!)
        pp.plane = pl
        
        # 5. Assign Param Back to Camera (Crucial Step!)
        cam_obj._pinplate_param = pp
        
        # 6. Trigger internal update if needed
        cam_obj.updatePt3dClosest()

    def optimize_planes(self, dataset, cam_to_window, window_media, window_planes, cam_params, cams_cpp, active_cam_ids, verbosity: int = 0, progress_callback=None):
        """
        PR4: Refractive Plane Optimization.
        Returns updated window_planes.
        """
        if verbosity >= 0:
            print("\n" + "="*60)
            print("[Refractive] Phase P1: Plane Optimization (Fixed Extrinsics)")
            print("="*60)
        
        wand_len_target = dataset.get('wand_length', 10.0)
        
        # Inject bootstrap points for P1 sanity Check
        if hasattr(self, 'X_B_list') and self.X_B_list:
            dataset['bootstrap_points'] = [x for x in self.X_B_list.values() if x is not None]
        
        # Create optimizer with all necessary data
        optimizer = RefractivePlaneOptimizer(
            dataset=dataset,
            cam_params=cam_params,
            cams_cpp=cams_cpp,
            cam_to_window=cam_to_window,
            window_media=window_media,
            window_planes=window_planes,
            wand_length=wand_len_target,
            config=OptimizationConfig(
                lambda0_init=200.0,
                lambda_min=10.0,
                lambda_max=5000.0,
                outer_rounds=3,
                lambda_reg=10.0,
                alpha_beta_bound=0.5,
                max_frames=300,
                verbosity=verbosity
            ),
            progress_callback=progress_callback
        )
        
        # P1.1 / P1.2: Per-Window Optimization
        if progress_callback:
            try:
                progress_callback("Optimizing Window Parameters...", 0.0, 0.0, 0.0)
            except:
                pass
        opt_planes = optimizer.optimize_per_window()
        
        # P1.3: Joint Optimization (All Windows)
        if verbosity >= 1:
            print("\n" + "="*60)
            print("[Refractive] Phase P1.3: Joint Optimization")
            print("="*60)
        
        if progress_callback:
            try:
                progress_callback("Refining All Windows Jointly...", 0.0, 0.0, 0.0)
            except:
                pass
        opt_planes = optimizer.optimize_joint()

        
        # P1 Summary
        print("\n[Refractive] P1 Optimization Summary (d_key = dist from mean cam center):")
        # Build cam centers for d_key calc
        import cv2
        center_cache = {}
        for cid, p in cam_params.items():
             R, _ = cv2.Rodrigues(p[0:3])
             center_cache[cid] = -R.T @ p[3:6] # Recompute or assume helper avail? Use canonical helper if possible?
             # Actually I have camera_center helper imported
             # from .refractive_geometry import camera_center
             # Let's use it if available in self scope? 
             # No, this is inside method. Let's use helper as I imported it at top of file.
             
        for wid, pl in opt_planes.items():
             n = pl['plane_n']
             pt = pl['plane_pt']
             
             # Compute d_key
             cams = [c for c, w in cam_to_window.items() if w == wid and c in cam_params]
             d_key = 0.0
             if cams:
                 centers = [camera_center(cv2.Rodrigues(cam_params[c][0:3])[0], cam_params[c][3:6]) for c in cams]
                 C_mean = np.mean(centers, axis=0)
                 d_key = np.dot(n, pt - C_mean)
             
             thick = window_media.get(wid, {}).get('thickness', 0)
             print(f"  Win {wid}: d_key={d_key:.2f}mm, thick={thick:.1f}mm, n={n.round(4)}")

        if verbosity >= 1:
            optimizer.print_diagnostics()
            print("\n[Refractive] P1 Optimization Complete.")
        
        return opt_planes

    def calibrate(self, num_windows, cam_to_window, window_media, out_path, verbosity: int = 1, progress_callback=None):

        """
        PR1 MVP Stub: Validate, Log, Export Snapshot, and Exit (Engineering Guardrail #5).
        """
        # Progress callback: report initialization phase
        if progress_callback:
            try:
                progress_callback("Initializing", 0.0, 0.0, 0.0)
            except:
                pass
        
        if verbosity >= 0:
            print("\n" + "="*50)
            print("[Refractive] Phase 1: Validating Inputs & Config")
        self.verbose = (verbosity >= 2) # Legacy compat
        print(f"  Model: Pinhole+Refraction")
        print(f"  Num Windows: {num_windows}")

        
        # 1. Validate and Structure Observations
        try:
            dataset = self._collect_observations(cam_to_window)
            wand_len_target = dataset.get('wand_length', 10.0)
            print(f"  Frames: {dataset['num_frames']}, Cameras: {dataset['num_cams']}")
            print(f"  Total Valid Observations (2 dots per view): {dataset['total_observations']}")
        except Exception as e:
            print(f"  [Error] Validation failed: {e}")
            raise
            
        # 2. Log Config Snapshot (Engineering Guardrail #2)
        print("\n[Refractive] Mapping Snapshot:")
        for cid in dataset['cam_ids']:
            wid = cam_to_window[cid]
            print(f"  Cam {cid} -> Window {wid}")
            
        print("\n[Refractive] Media Parameters:")
        for wid, media in sorted(window_media.items()):
             print(f"  Win {wid}: n_air={media['n1']:.3f}, n_win={media['n2']:.3f}, n_obj={media['n3']:.3f}, thick={media['thickness']:.2f}mm")

        # 3. Log Phase Stubs (Engineering Guardrail #2)

        
        # === STAGE 0: Pinhole Bootstrap Initialization ===
        self.window_planes = {} # Single source of truth
        cam_params_raw = getattr(self.base, 'cam_params', {})
        
        # Cache options - store in PARENT directory so it survives across timestamped runs
        FORCE_REBUILD_PINHOLE_BOOTSTRAP = False # Task 3: Force Rebuild (User requested False)
        use_bootstrap_cache = True
        from pathlib import Path
        import os
        if out_path:
            parent_dir = Path(out_path).parent  # e.g., H:/20260106/T0/Refraction/
            bootstrap_cache_path = parent_dir / "bootstrap_cache.json"
        else:
            bootstrap_cache_path = None
        
        if not cam_params_raw:
            # Get initial focal length
            initial_focal = getattr(self.base, 'initial_focal', 5000.0)
            
            # Try to load from cache first
            cache_loaded = False
            print(f"[BOOT] Using cache = {not FORCE_REBUILD_PINHOLE_BOOTSTRAP}")
            if not FORCE_REBUILD_PINHOLE_BOOTSTRAP and use_bootstrap_cache and bootstrap_cache_path:
                print(f"\n[BOOT][CACHE] Loading bootstrap cache from: {bootstrap_cache_path}")
                cache_result = self.load_bootstrap_cache(
                    path=bootstrap_cache_path,
                    wand_len_target=wand_len_target,
                    cam_ids_current=dataset['cam_ids'],
                    num_frames_current=dataset['num_frames']
                )
                
                if cache_result is not None:
                    cam_params, err_px, active_cam_ids, best_pair, X_A_scaled, X_B_scaled = cache_result
                    cache_loaded = True
                    print(f"[BOOT][CACHE] Loaded OK: cams={list(cam_params.keys())}, frames={len(X_A_scaled)}")
                    print("[BOOT][CACHE] Skipping Phase1+Phase2 (using cached pinhole bootstrap)")
            
            if not cache_loaded:
                # Run P0 Bootstrap (Frozen-Intrinsics Pinhole)
                print("\n" + "="*60)
                print("[Refractive][BOOT] Running Stage P0 (Frozen-Intrinsics Pinhole Bootstrap)")
                print("="*60)
                print("[BOOT] Using frozen intrinsics from UI.")
                
                # Progress callback: report P0 Bootstrap phase
                if progress_callback:
                    try:
                        progress_callback("Use PinHole model to initialize camera parameters...", 0.0, 0.0, 0.0)
                    except:
                        pass

                
                # Ensure UI focal is valid
                if initial_focal < 100:
                     raise ValueError(f"CRITICAL: UI focal length {initial_focal} seems invalid.")
                
                # --- Pre-Step: Select Best Pair ---
                best_pair = select_best_pair_via_precalib(self.base, wand_len_target, initial_focal)
                
                if best_pair is None:
                    raise ValueError("[BOOT] Could not select best pair")
                print(f"[BOOT] Using reliable pair from precalib: {best_pair}") 

                # Prepare observations for P0
                observations = self._prepare_observations_for_bootstrap(cam_to_window)
                # Task 1: Fix W/H definition. image_size is usually (H, W) from numpy shape.
                image_size = self.base.image_size if hasattr(self.base, 'image_size') else (800, 1280)
                img_h, img_w = image_size
                print(f"[BOOT] image_size (H, W) = {image_size}. Using W={img_w}, H={img_h}")
                
                all_cam_ids = dataset['cam_ids']
                
                # Run P0 (Phase 1 + Phase 2 + Phase 3)
                config = PinholeBootstrapP0Config(
                    wand_length_mm=wand_len_target,
                    ui_focal_px=initial_focal,
                )
                bootstrap = PinholeBootstrapP0(config)
                
                cam_i, cam_j = best_pair
                cam_params_p0, report = bootstrap.run_all(
                    cam_i=cam_i,
                    cam_j=cam_j,
                    observations=observations,
                    image_size=image_size,
                    all_cam_ids=all_cam_ids,
                    progress_callback=progress_callback
                )

                
                # Convert P0 output to expected format: {cid: [rvec, tvec, f, cx, cy, k1, k2]}
                # Task 1: Correct cx, cy
                cx = img_w * 0.5
                cy = img_h * 0.5
                
                # Task 4: [K_CHECK] 
                print(f"[K_CHECK] img_w={img_w}, img_h={img_h}, cx={cx}, cy={cy}, (img_w/2, img_h/2)=({img_w/2}, {img_h/2})")
                
                cam_params = {}
                for cid, params in cam_params_p0.items():
                    # params is [rvec(3), tvec(3)]
                    cam_params[cid] = np.concatenate([params, [initial_focal, cx, cy, 0.0, 0.0]])
                
                # Extract 3D points from report
                points_3d = report.get('points_3d', {})
                X_A_scaled = {fid: XA for fid, (XA, XB) in points_3d.items()}
                X_B_scaled = {fid: XB for fid, (XA, XB) in points_3d.items()}
                active_cam_ids = list(cam_params.keys())
                err_px = {}  # P0 doesn't compute per-camera pixel error
                
                # Save to cache
                if use_bootstrap_cache and bootstrap_cache_path:
                    try:
                        self.save_bootstrap_cache(
                            path=bootstrap_cache_path,
                            cam_params_by_id=cam_params,
                            err_px_by_id=err_px,
                            active_cam_ids=active_cam_ids,
                            chosen_pair=best_pair,
                            X_A_list=X_A_scaled,
                            X_B_list=X_B_scaled,
                            wand_len_mm=wand_len_target,
                            cam_ids=dataset['cam_ids'],
                            num_frames=dataset['num_frames']
                        )
                        print(f"[BOOT][CACHE] Saved bootstrap cache to: {bootstrap_cache_path}")
                    except Exception as e:
                        print(f"[BOOT][CACHE] Warning: Failed to save cache: {e}")
            
            inactive_cam_ids = [cid for cid in dataset['cam_ids'] if cid not in active_cam_ids]
            
            # --- UNIFIED Sanity check: Use bootstrap's already-scaled 3D points (single source of truth) ---
            print("[Refractive][STAGE 0] Unified Sanity Check (using bootstrap 3D points):")
            print(f"  Scale source: bootstrap_pinhole_for_refractive")
            print(f"  Frames with scaled 3D: {len(X_A_scaled)}")
            
            # Show sample values for debug (Random)
            all_fids = list(X_A_scaled.keys())
            sample_count = min(3, len(all_fids))
            sample_fids = np.random.choice(all_fids, size=sample_count, replace=False)
            sample_fids.sort() # Keep sorted for nicer logs
            
            print(f"  Random Sample values (should match bootstrap sample):")
            for fid in sample_fids:
                if fid in X_B_scaled:
                    len_sample = np.linalg.norm(X_B_scaled[fid] - X_A_scaled[fid])
                    print(f"    Frame {fid}: ||XB-XA|| = {len_sample:.4f} mm")
            
            sanity_lens = []
            for fid in X_A_scaled.keys():  # Use ALL frames, not just first 20
                if fid in X_B_scaled:
                    len_est = np.linalg.norm(X_B_scaled[fid] - X_A_scaled[fid])
                    sanity_lens.append(len_est)
            
            if sanity_lens:
                median_sanity = np.median(sanity_lens)
                std_sanity = np.std(sanity_lens)
                print(f"  Median wand length (from {len(sanity_lens)} frames): {median_sanity:.4f} mm (target: {wand_len_target:.2f} mm)")
                print(f"  Std: {std_sanity:.4f} mm")
                
                # Verify scale propagation is consistent
                if abs(median_sanity - wand_len_target) > 0.5:
                    print(f"  [WARNING] Scale mismatch detected!")
                    print(f"    Expected: {wand_len_target:.2f} mm, Got: {median_sanity:.4f} mm")
                    print(f"    This indicates bootstrap scale was not applied correctly.")
                
                # Hard abort if > 2Ã— off
                if median_sanity < wand_len_target / 2.0 or median_sanity > wand_len_target * 2.0:
                    raise RuntimeError(
                        f"[ABORT] Scale sanity failed: median={median_sanity:.2f}mm, target={wand_len_target:.2f}mm."
                    )
            else:
                print("  [WARNING] No valid frames for sanity check.")
            
            # --- Verify cam_params scale propagation ---
            c1, c2 = best_pair
            t1 = cam_params[c1][3:6]
            t2 = cam_params[c2][3:6]
            baseline = np.linalg.norm(t2 - t1)
            
            if verbosity >= 1:
                print(f"\n  Scale propagation check:")
                print(f"    Cam {c1} tvec: {t1.round(2)} (Origin of P0 gauge)")
                print(f"    Cam {c2} tvec: {t2.round(2)} (||t||={baseline:.2f}mm)")
            
            # --- Initialize window planes (pass FULL cam_to_window, function handles active vs all) ---
            print("\n[Refractive][STAGE 0] Window Plane Initialization")
            self.window_planes = self._init_window_planes_from_cameras(
                cam_params, cam_to_window, window_media, err_px,
                X_A_list=X_A_scaled, X_B_list=X_B_scaled, active_cam_ids=dataset['cam_ids']
            )

            # [AUDIT] Estimate Radii from Bootstrap (Step 1)
            est_r_small = 0.0
            est_r_large = 0.0
            if X_A_scaled and X_B_scaled:
                 est_r_small, est_r_large = self._estimate_and_log_sphere_radii(dataset, cam_params, X_A_scaled, X_B_scaled, tag="P1")
            
            # Store in dataset for later stages (Step 2)
            dataset['est_radius_small_mm'] = est_r_small
            dataset['est_radius_large_mm'] = est_r_large
            print(f"[P1] Stored estimated radii in dataset: Small={est_r_small:.3f}mm, Large={est_r_large:.3f}mm")


            if self.window_planes is None:
                print("[P1] Plane initialization failed.")
                if progress_callback: progress_callback("Plane initialization failed.", 0, 0, 0)
                return None, False
                
            # [DEBUG] Inspect window_planes keys
            print(f"[DEBUG] P1 returned window_planes keys: {list(self.window_planes.keys())}")
            if self.window_planes:
                k0 = list(self.window_planes.keys())[0]
                print(f"[DEBUG] Key type: {type(k0)}")

        
        # === SIDE-SANITY CHECK (read-only, no modification) ===
        # Distinguish between active cameras (used for ray-building) and inactive cameras
        # C++ convention: s(C) = dot(n, C - plane_pt) < 0 for cameras (camera-side)
        active_cam_set = set(active_cam_ids)
        print(f"\n[SANITY][PLANE_SIDE] Camera side w.r.t. window planes (C++ convention: s<0 = camera-side):")
        orientation_errors = []
        for cid in sorted(cam_to_window.keys()):
            wid = cam_to_window.get(cid, 0)
            if wid not in self.window_planes:
                print(f"  Cam {cid} (Win {wid}): [SKIP] Window plane not initialized")
                continue
            
            if cid not in cam_params:
                print(f"  [WARN] Cam {cid} (Win {wid}): No cam_params available (inactive)")
                continue
            
            p = cam_params[cid]
            R, _ = cv2.Rodrigues(p[0:3])
            C_cam = -R.T @ p[3:6]
            
            plane_pt = np.array(self.window_planes[wid]['plane_pt'])
            plane_n = np.array(self.window_planes[wid]['plane_n'])
            s = np.dot(plane_n, C_cam - plane_pt)
            
            is_active = cid in active_cam_set
            
            if s < 0:  # C++ convention: camera-side is s < 0
                print(f"  Cam {cid} (Win {wid}): s = {s:.2f} mm -> camera-side (OK)")
            elif is_active:
                print(f"  [ERROR] Cam {cid} (Win {wid}): s = {s:.2f} mm -> object-side (ACTIVE cam WRONG)")
                orientation_errors.append((cid, wid, s))
            else:
                print(f"  [WARN] Cam {cid} (Win {wid}): s = {s:.2f} mm -> object-side (inactive, excluded from constraints)")
        
        if orientation_errors:
            print(f"[ERROR] Window normal orientation failed for {len(orientation_errors)} ACTIVE camera(s)!")
            print(f"  Check cam_to_window mapping or geometry configuration.")
        elif verbosity > 0:
            print(f"  All camera-to-plane orientations valid (s < 0).")

        # --- GEOMETRY SANITY REPORT ---
        print("\n[Refractive][SANITY] Geometry & Baselines")
        
        tvecs_final = []
        cams_reported = []
        cam_centers_final = []
        
        for cid in dataset['cam_ids']:
            if cid not in cam_params:
                continue
                
            p = cam_params[cid]
            rvec = p[0:3]
            tvec = p[3:6]
            
            # Compute Camera Center explicitly
            R, _ = cv2.Rodrigues(rvec)
            C = camera_center(R, tvec)
            
            tvecs_final.append(tvec)
            cam_centers_final.append(C)
            cams_reported.append(cid)
            
            # Print detailed geometry ONLY if verbosity is high
            if verbosity >= 1:
                print(f"  Cam {cid:2}: tvec=[{tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}], Center=[{C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f}]")
            
        print("  Baseline Definition: ||C_i - C_j|| using world camera centers.")
        
        max_b = 0.0
        baselines = []
        for i in range(len(cam_centers_final)):
            for j in range(i + 1, len(cam_centers_final)):
                d = np.linalg.norm(cam_centers_final[i] - cam_centers_final[j])
                max_b = max(max_b, d)
                baselines.append(d)
                if verbosity >= 1:
                    print(f"    (Cam{cams_reported[i]}, Cam{cams_reported[j]}) = {d:.2f} mm")
        
        print(f"  Max Baseline: {max_b:.2f} mm (N_cam={len(cams_reported)})")
        assert max_b > 10.0 or len(dataset['cam_ids']) < 2, f"Degenerate Max Baseline: {max_b:.2f} mm"

        print("\n[Refractive][SANITY] Window Parameters (P1 Initial)")
        for wid, media in window_media.items():
            wp = self.window_planes.get(wid)
            if wp is None:
                print(f"  [WARN] Win {wid}: Not found in window_planes! Using default.")
                pt = np.array([0, 0, 300])
                n = np.array([0, 0, 1])
            else:
                pt = np.array(wp['plane_pt'])
                n = np.array(wp['plane_n'])
            
            # Compute d_key: Distance from mean camera center
            # Identify cameras for this window
            win_cams_centers = [cam_centers_final[i] for i, cid in enumerate(cams_reported) if cam_to_window.get(cid) == wid]
            d_key = 0.0
            if win_cams_centers:
                C_mean = np.mean(win_cams_centers, axis=0)
                d_key = np.dot(n, pt - C_mean)
            
            if verbosity >= 1:
                print(f"  Win {wid}: plane_pt={pt.round(2)}, plane_n={n.round(4)}")
                print(f"          thick={media['thickness']:.2f}mm, n_air={media['n1']:.2f}, n_glass={media['n2']:.2f}, n_water={media['n3']:.2f}")
            else:
                 print(f"  Win {wid}: d_key={d_key:.2f}mm (dist from mean cam), thick={media['thickness']:.1f}mm")
                 
        print("-" * 30 + "\n")

        try:
            stored_dir = self.export_camfile_with_refraction(out_path, cam_params, window_media, cam_to_window, self.window_planes)
            print(f"[Refractive] Configuration snapshot exported to: {stored_dir}")
        except Exception as e:
            print(f"  [Error] Export failed: {e}")
            raise

        # === Phase 2/3: Build Refracted Rays (C++ Authority) ===
        print("\n" + "-"*30)
        print("[Refractive] Phase 2/3: Building Rays (C++ Kernel)")
        
        # Progress callback: report Phase 2/3
        if progress_callback:
            try:
                progress_callback("Phase 2/3: Building Rays", 0.0, 0.0, 0.0)
            except:
                pass

        
        cams_cpp = {}
        for cid in dataset['cam_ids']:
            cam_path = os.path.join(stored_dir, f"cam{cid}.txt")
            if lpt and os.path.exists(cam_path):
                try:
                    cams_cpp[cid] = lpt.Camera(cam_path)
                except Exception as e:
                    print(f"  [Warning] Cam {cid} C++ load failed: {e}")


        # === Phase P1: Refractive Plane Optimization ===
        # Save cache in PARENT directory (dataset root) to share across timestamped runs
        p1_cache_path = os.path.join(str(Path(out_path).parent), "p1_plane_cache.json")
        
        # Try to load P1 cache
        cached_planes, cache_valid = self.load_p1_cache(
            p1_cache_path, window_media, cam_to_window, dataset, cam_params
        )
        
        if cache_valid:
            print("[P1][CACHE] Loaded plane cache successfully, skipping P1 optimization")
            self.window_planes = cached_planes
        else:
            # Run P1 optimization
            # Progress callback: report P1 Plane Optimization
            if progress_callback:
                try:
                    progress_callback("Initializing Window Optimization...", 0.0, 0.0, 0.0)
                except:
                    pass
            
            optimized_planes = self.optimize_planes(
                dataset, cam_to_window, window_media, self.window_planes, 
                cam_params, cams_cpp, active_cam_ids, verbosity=verbosity,
                progress_callback=progress_callback
            )
            
            # [CRITICAL] Update internal self.window_planes with optimized result
            if optimized_planes:
                self.window_planes = optimized_planes



            
            # Collect diagnostics for cache
            p1_diagnostics = {
                "ray_residual_rmse": getattr(self, '_last_ray_rmse', 0.0),
                "wand_length_rmse": getattr(self, '_last_wand_rmse', 0.0),
                "wand_length_bias": getattr(self, '_last_wand_bias', 0.0)
            }
            
            # Save P1 cache
            self.save_p1_cache(
                p1_cache_path, self.window_planes, window_media, cam_to_window, 
                dataset, cam_params, p1_diagnostics
            )
        
        # Re-export camfiles with optimized planes and reload C++ objects
        if verbosity >= 0:
            print("\n[Refractive] Exporting P1-optimized parameters to camFiles...")
            
        try:
            stored_dir = self.export_camfile_with_refraction(
                out_path, 
                cam_params, 
                window_media, 
                cam_to_window, 
                window_planes=self.window_planes
            )
            if verbosity >= 1:
                print(f"  Updated camFiles in: {stored_dir}")
            
            # Reload C++ cameras with updated planes
            for cid in dataset['cam_ids']:
                cam_path = os.path.join(stored_dir, f"cam{cid}.txt")
                if lpt and os.path.exists(cam_path):
                    try:
                        cams_cpp[cid] = lpt.Camera(cam_path)
                    except Exception as e:
                        print(f"  [Warning] Cam {cid} reload failed: {e}")
        except Exception as e:
            print(f"  [Error] Plane export failed: {e}")

        # === Phase PR4: Bundle Adjustment (Selective BA) ===
        # Pass verbosity to config
        pr4_config = PR4Config(
            skip_pr4=False,
            pr4_stage=3,
            verbosity=verbosity
        )
        
        pr4_optimizer = RefractiveBAOptimizerPR4(
            dataset=dataset,
            cam_params=cam_params,
            cams_cpp=cams_cpp,
            cam_to_window=cam_to_window,
            window_media=window_media,
            window_planes=self.window_planes,
            wand_length=wand_len_target,
            config=pr4_config,
            progress_callback=progress_callback
        )

        # Try to load PR4 cache
        loaded_pr4 = pr4_optimizer.try_load_cache(out_path)
        
        if loaded_pr4:
            # Cache loaded successfully, extract updated state
            self.window_planes = pr4_optimizer.window_planes
            cam_params = pr4_optimizer.cam_params
            print("[PR4][CACHE] Using cached PR4 results, skipping optimization")
        else:
            # Run PR4 optimization
            self.window_planes, cam_params = pr4_optimizer.optimize()
            # Save PR4 cache
            pr4_optimizer.save_cache(out_path)
        
        # [Refractive][CAMFILE] Export CamFiles (Initial / P1)
        if verbosity >= 0:
             print("\n[Refractive][CAMFILE] Exporting PR4-optimized parameters to camFiles...")
             
        try:
            # Pass self.window_planes explicitly
            stored_dir = self.export_camfile_with_refraction(
                out_path, 
                cam_params, 
                window_media, 
                cam_to_window, 
                window_planes=self.window_planes
            )
            if verbosity >= 1:
                print(f"  Updated camFiles in: {stored_dir}")
            
            # Reload C++ cameras with PR4-optimized planes AND VERIFY
            for cid in dataset['cam_ids']:
                cam_path = os.path.join(stored_dir, f"cam{cid}.txt")
                if lpt and os.path.exists(cam_path):
                    try:
                        cams_cpp[cid] = lpt.Camera(cam_path)
                        
                        # --- CONSISTENCY CHECK ---
                        # Verify plane_pt shift and unit normal
                        pp = cams_cpp[cid]._pinplate_param
                        pl = pp.plane
                        
                        n_cpp = np.array([pl.norm_vector[0], pl.norm_vector[1], pl.norm_vector[2]])
                        pt_cpp = np.array([pl.pt[0], pl.pt[1], pl.pt[2]])
                        
                        # Fix: PinPlateParam uses w_array for thickness
                        w_arr = pp.w_array
                        thick_cpp = w_arr[0] if w_arr else 0.0
                        
                        wid = cam_to_window.get(cid, 0)
                        wp = self.window_planes.get(wid)
                        
                        if wp:
                            n_py = np.array(wp['plane_n'])
                            pt_py = np.array(wp['plane_pt'])
                            
                            # Shift along n
                            # s = dot(n, pt_cpp - pt_py)
                            shift = np.dot(n_cpp, pt_cpp - pt_py)
                            norm_n = np.linalg.norm(n_cpp)
                            
                            if verbosity >= 0: # Always check, print summary per cam
                                print(f"[CAMFILE][CHECK] Cam {cid}: shift={shift:.3f}mm (exp. {thick_cpp:.3f}mm); ||n||={norm_n:.6f} ok")
                                
                    except Exception as e:
                        print(f"  [Warning] Cam {cid} reload/check failed: {e}")
        except Exception as e:
            print(f"  [Error] PR4 export failed: {e}")

        # === Phase PR5: Robust Bundle Adjustment (Geometric) ===
        # Optimize planes, thickness, intrinsics, extrinsics with strong priors
        pr5_config = PR5Config(
            pr5_stage=2,
            verbosity=verbosity,
            allow_weak_rvec=True
        )
        
        pr5_optimizer = RefractiveBAOptimizerPR5(
            dataset=dataset,
            cam_params=cam_params,
            cams_cpp=cams_cpp,
            cam_to_window=cam_to_window,
            window_media=window_media,
            window_planes=self.window_planes,
            wand_length=wand_len_target,
            config=pr5_config,
            progress_callback=progress_callback
        )

        
        # Check Cache
        loaded_pr5 = pr5_optimizer.try_load_cache(out_path)
        
        if loaded_pr5:
            # Output loaded values updated in init ? No, try_load_cache updates internal state of optimizer.
            # We need to extract them back to local variables?
            # pr5_optimizer.window_planes is updated.
            self.window_planes = pr5_optimizer.window_planes
            cam_params = pr5_optimizer.cam_params
            window_media = pr5_optimizer.window_media
            
            # Since C++ objects updated inside try_load_cache, we are good?
            # Yes, try_load_cache calls _apply_planes_to_cpp.
        else:
            # Run Optimization
            self.window_planes, cam_params, window_media = pr5_optimizer.optimize()
            # Save Cache
            pr5_optimizer.save_cache(out_path)
            
        # Re-export camFiles with PR5 results
        if verbosity >= 0:
             print("\n[Refractive] Exporting PR5-optimized parameters to camFiles...")
        
        # Coordinate Alignment: Align Y-axis with plane intersection line
        if len(self.window_planes) >= 2:
            try:
                cam_params, self.window_planes, _, R_align = align_world_y_to_plane_intersection(
                    self.window_planes, cam_params, points_3d=None
                )
                if verbosity >= 1:
                    print("[Coordinate Alignment] Y-axis aligned with plane intersection line.")
            except Exception as e:
                print(f"[Coordinate Alignment] Warning: alignment failed: {e}")
        
        try:
            stored_dir = self.export_camfile_with_refraction(out_path, cam_params, window_media, cam_to_window, self.window_planes)

            if verbosity >= 1:
                print(f"  Updated camFiles in: {stored_dir}")
            
            # Reload and Verify
            for cid in dataset['cam_ids']:
                cam_path = os.path.join(stored_dir, f"cam{cid}.txt")
                if lpt and os.path.exists(cam_path):
                    try:
                        cams_cpp[cid] = lpt.Camera(cam_path)
                    except Exception as e:
                        print(f"  [Warning] Cam {cid} reload failed: {e}")
        except Exception as e:
            print(f"  [Error] PR5 export failed: {e}")


        # rays_db[fid][0=A, 1=B] = [Ray, Ray, ...]
        rays_db = {fid: {0: [], 1: []} for fid in dataset['frames']}
        invalid_reasons = {}
        total_obs = 0
        invalid_obs = 0

        print("\n[Refractive] Phase 2/3: Building Labelled Rays (C++ Kernel)")
        print(f"  Using active cameras only: {active_cam_ids}")
        
        for fid in dataset['frames']:
            for cid in active_cam_ids:  # <-- Only use active cameras
                wid = cam_to_window.get(cid)
                cam_obj = cams_cpp.get(cid)
                if not cam_obj: continue
                
                # STRICT LABEL-BASED FETCH
                uvA = dataset['obsA'][fid].get(cid)   # A = Filtered_Small
                uvB = dataset['obsB'][fid].get(cid)   # B = Filtered_Large
                
                # Assertion (a): Points must differ significantly in image space (only if both exist)
                if uvA is not None and uvB is not None:
                    dist_2d = np.linalg.norm(np.array(uvA) - np.array(uvB))
                    assert dist_2d > 1e-3, f"Frame {fid} Cam {cid}: Endpoints collapse in 2D (dist={dist_2d})"

                # Build Independent Labelled Rays (allow partial presence)
                if uvA is not None:
                    total_obs += 1
                    
                    # Task 2: UV Norm Check (Log first 10 obs)
                    if self.verbose and total_obs <= 10 and cid in cam_params:
                        u, v = uvA
                        cp_check = cam_params[cid]
                        f_check = cp_check[6]
                        cx_check = cp_check[7]
                        cy_check = cp_check[8]
                        u_norm_calc = (u - cx_check) / f_check
                        v_norm_calc = (v - cy_check) / f_check
                        print(f"[UV_NORM_CHECK] f={f_check:.1f}, cx={cx_check:.1f}, cy={cy_check:.1f}")
                        print(f"  uv_px=({u:.1f},{v:.1f}) -> uv_norm_calc=({u_norm_calc:.4f}, {v_norm_calc:.4f})")

                    rayA = build_pinplate_ray_cpp(cam_obj, uvA, cam_id=cid, window_id=wid, frame_id=fid, endpoint="A")
                    if rayA.valid:
                        rays_db[fid][0].append(rayA)
                    else:
                        invalid_obs += 1
                        invalid_reasons[rayA.reason or "unknown"] = invalid_reasons.get(rayA.reason or "unknown", 0) + 1
                
                if uvB is not None:
                    total_obs += 1
                    rayB = build_pinplate_ray_cpp(cam_obj, uvB, cam_id=cid, window_id=wid, frame_id=fid, endpoint="B")
                    if rayB.valid:
                        rays_db[fid][1].append(rayB)
                    else:
                        invalid_obs += 1
                        invalid_reasons[rayB.reason or "unknown"] = invalid_reasons.get(rayB.reason or "unknown", 0) + 1

        print(f"  Total Observations: {total_obs}, Invalid Rays: {invalid_obs}")
        if inactive_cam_ids:
            print(f"  [Note] Skipped inactive cameras: {inactive_cam_ids}")

        # --- Ray Direction Diagnostic: Check if rays collapsed to plane normal ---
        # --- Ray Direction Diagnostic: Angle Spread by (Cam, Endpoint) (Metric A) ---
        if verbosity >= 0:
            print("\n[RAY_DIAG] Checking ray angle spread (Summary):")
        
        # Group: rays[cid][k] = [d1, d2...]
        rays_grouped = {} # (cid, k) -> list of d
        
        for fid in list(rays_db.keys())[:200]: # Sample more frames
            for k in [0, 1]:
                for r in rays_db[fid][k]:
                    key = (r.cam_id, k)
                    if key not in rays_grouped:
                         rays_grouped[key] = []
                    rays_grouped[key].append(r.d)

        total_suspicious = 0
        if verbosity >= 1:
             print(f"  {'Cam':<4} {'Pt':<2} {'N_rays':<8} {'AlignN':<8} {'P90_deg':<8} {'Status'}")
             print("  " + "-"*50)
             
        for (cid, k), ds in sorted(rays_grouped.items()):
            if len(ds) < 10: continue
            
            ds = np.asarray(ds, dtype=float)
            # Robust mean direction
            mean_d = np.mean(ds, axis=0)
            norm_m = np.linalg.norm(mean_d)
            if norm_m < 1e-9: 
                 mean_d = ds[0] # Degenerate case fallback
            else:
                 mean_d /= norm_m

            # Align to N (A4)
            wid = cam_to_window.get(cid, 0)
            wp = self.window_planes.get(wid)
            n_win_r = np.array(wp['plane_n'], dtype=np.float64) if wp else np.array([0.,0.,1.], dtype=np.float64)
            n_win_r /= (np.linalg.norm(n_win_r) + 1e-12)
            align_to_n = abs(np.dot(mean_d, n_win_r))
            
            # Angles (Safe Clamped)
            dots = np.clip(np.dot(ds, mean_d), -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(dots))
            
            ang_med = np.median(angles_deg)
            ang_p90 = np.percentile(angles_deg, 90)
            ang_p99 = np.percentile(angles_deg, 99)
            ang_max = np.max(angles_deg)
            
            label = "A" if k==0 else "B"
            
            status = "OK"
            if ang_p90 < 0.05:
                status = "COLLAPSE"
                total_suspicious += 1
                if align_to_n > 0.999 and ang_p90 < 0.2:
                    status = "GLUED_N"
            
            if verbosity >= 1:
                print(f"  {cid:<4} {label:<2} {len(ds):<8} {align_to_n:<8.4f} {ang_p90:<8.4f} {status}")
            
            # Legacy verbose block (gated by verbosity >= 2)
            if verbosity >= 2:
                print(f"    Details: med={ang_med:.4f}, p99={ang_p99:.4f}, max={ang_max:.4f}")

        if total_suspicious > 0:
            print(f"  [SUMMARY] {total_suspicious} groups show strong signs of Ray Collapse. (Enable verbosity=1 for table)")
        elif verbosity >= 0:
            print(f"  [SUMMARY] All groups OK (p90 >= 0.05 deg)")
        
        # --- Detailed Ray Probe: 10 sample rays ---
        if verbosity >= 2:
            print("\n[RAY_PROBE] Sample rays (first 10):")
            probe_count = 0
            for fid in sorted(rays_db.keys()):
                if probe_count >= 10:
                    break
            for k, label in [(0, "A"), (1, "B")]:
                for r in rays_db[fid][k]:
                    if probe_count >= 10:
                        break
                    wid_r = cam_to_window.get(r.cam_id, 0)
                    n_win_r = np.array(self.window_planes[wid_r]['plane_n'])
                    dot_dn = np.dot(r.d, n_win_r)
                    print(f"  Frame {fid} Cam {r.cam_id} {label}: uv=({r.uv[0]:.1f}, {r.uv[1]:.1f}), o={r.o.round(1)}, d={r.d.round(4)}, dot(d,n)={dot_dn:.4f}")
                    probe_count += 1

        # --- Ray Scale Diagnostics (Frame 0 Only) ---
        if 0 in rays_db and self.verbose:
            print("\n[Refractive][DEBUG] Frame 0 Ray Scale Diagnostics:")
            for k, label in [(0, "A"), (1, "B")]:
                print(f"  Endpoint {label}:")
                for r in rays_db[0][k]:
                    norm_o = np.linalg.norm(r.o)
                    wid_ray = cam_to_window.get(r.cam_id, 0)
                    n_win_ray = np.array(self.window_planes[wid_ray]['plane_n'])
                    dot_dn_final = np.dot(r.d, n_win_ray)
                    print(f"    Cam {r.cam_id}: o=[{r.o[0]:.2f}, {r.o[1]:.2f}, {r.o[2]:.2f}], ||o||={norm_o:.1f}, d=[{r.d[0]:.4f}, {r.d[1]:.4f}, {r.d[2]:.4f}], dot(d,n)={dot_dn_final:.4f}")
        
        # === Phase 4: Triangulation ===
        print("\n[Refractive] Phase 4: Independent Multi-Camera Triangulation")
        
        tri_data = {}
        frames_valid_both = 0
        frames_valid_A = 0
        frames_valid_B = 0
        bad_frames = []  # Frames that collapse or fail sanity
        px_min_2d_sep = 5.0  # Minimum 2D separation for valid endpoints

        for fid in dataset['frames']:
            # Triangulate A and B using strictly separated ray containers
            res_A = triangulate_point(rays_db[fid][0])
            res_B = triangulate_point(rays_db[fid][1])
            
            XA, XB = res_A[0], res_B[0]
            validA, validB = res_A[2], res_B[2]
            
            # --- Check 2D separation before 3D validation ---
            skip_2d_check = False
            for cid in active_cam_ids:
                uvA = dataset['obsA'][fid].get(cid)
                uvB = dataset['obsB'][fid].get(cid)
                if uvA is not None and uvB is not None:
                    dist_2d = np.linalg.norm(np.array(uvA) - np.array(uvB))
                    if dist_2d < px_min_2d_sep:
                        skip_2d_check = True
                        bad_frames.append({
                            'fid': fid, 'reason': f'2D_sep<{px_min_2d_sep}px in Cam{cid}',
                            'dist_2d': dist_2d
                        })
                        # break # Don't break, check all cams? Actually break is fine for marking bad frame
                        break

            # --- Triangulated endpoint collapse check (with outlier handling) ---
            is_collapse = False
            coord_check_fails = 0
            
            if validA and validB and not skip_2d_check:
                dist_3d = np.linalg.norm(XA - XB)
                if fid < 5 and self.verbose:
                    print(f"  Frame {fid:<3}: XA={XA.round(3)}, XB={XB.round(3)}, len_est={dist_3d:.4f}mm")
                
                    # Coordinate-frame consistency check for first few frames
                    for k, (X_k, label) in enumerate([(XA, "A"), (XB, "B")]):
                        for ray in rays_db[fid][k]:
                            # Check: point should be in front of ray origin (dot(X-o, d) > 0)
                            vec_to_X = X_k - ray.o
                            dot_fwd = np.dot(vec_to_X, ray.d)
                            if dot_fwd < 0:
                                coord_check_fails += 1
                                if self.verbose:
                                    print(f"    [COORD-CHECK FAIL] {label} Cam{ray.cam_id}: X.z={X_k[2]:.1f}, o.z={ray.o[2]:.1f}, d.z={ray.d[2]:.4f}, dot={dot_fwd:.2f}")

                if coord_check_fails > 0:
                     print(f"  [WARNING] Frame {fid}: {coord_check_fails} rays failed coordinate check (dot < 0). Enable verbose for details.")
                
                if dist_3d < 0.2 * wand_len_target:
                    is_collapse = True
                    bad_frames.append({
                        'fid': fid, 'reason': 'collapse',
                        'dist_3d': dist_3d, 'XA': XA.tolist(), 'XB': XB.tolist(),
                        'num_rays_A': len(rays_db[fid][0]), 'num_rays_B': len(rays_db[fid][1])
                    })

            # Mark invalid if collapsed
            if is_collapse or skip_2d_check:
                validA = False
                validB = False

            tri_data[fid] = {
                "XA": XA, "condA": res_A[1], "validA": validA,
                "XB": XB, "condB": res_B[1], "validB": validB
            }
            if validA: frames_valid_A += 1
            if validB: frames_valid_B += 1
            if validA and validB: frames_valid_both += 1

        print(f"  Frames Valid (A/B/Both): {frames_valid_A} / {frames_valid_B} / {frames_valid_both}")
        
        # Report bad frames
        total_frames = len(dataset['frames'])
        num_bad = len(bad_frames)
        bad_ratio = num_bad / total_frames if total_frames > 0 else 0
        
        if num_bad > 0:
            print(f"\n  [Outlier Report] Bad frames: {num_bad}/{total_frames} ({bad_ratio*100:.1f}%)")
            for bf in bad_frames[:5]:  # Show first 5
                print(f"    Frame {bf['fid']}: {bf['reason']}")
            if num_bad > 5:
                print(f"    ... and {num_bad - 5} more")
            
            # Only abort if bad ratio > 10%
            if bad_ratio > 0.10:
                print(f"\n  [WARNING] High bad frame ratio ({bad_ratio*100:.1f}% > 10%). Geometry may be degenerate.")

        # === Phase 5: Residual Report ===
        print("\n[Refractive] Phase 5: Residual Analysis")
        
        r_ray_all = []
        r_len_all = []
        len_est_all = []
        per_cam_r_ray = {cid: [] for cid in active_cam_ids}
        per_win_r_ray = {wid: [] for wid in window_media.keys()}
        frame_metrics = [] 

        for fid in dataset['frames']:
            f_tri = tri_data[fid]
            frame_r_rays = []
            
            for k in [0, 1]:
                X_k = f_tri["XA" if k==0 else "XB"]
                if not f_tri["validA" if k==0 else "validB"]: continue
                for ray in rays_db[fid][k]:
                    dist = point_to_ray_dist(X_k, ray.o, ray.d)
                    r_ray_all.append(dist)
                    frame_r_rays.append(dist)
                    per_cam_r_ray[ray.cam_id].append(dist)
                    per_win_r_ray[ray.window_id].append(dist)
            
            r_len = np.nan
            len_est = np.nan
            if f_tri["validA"] and f_tri["validB"]:
                len_est = np.linalg.norm(f_tri["XB"] - f_tri["XA"])
                r_len = len_est - wand_len_target
                r_len_all.append(r_len)
                len_est_all.append(len_est)
            
            if frame_r_rays:
                med_r = np.median(frame_r_rays)
                score = med_r + (abs(r_len) if not np.isnan(r_len) else 1.0)
                frame_metrics.append({
                    "fid": fid, "score": score, "r_ray_med": med_r, 
                    "r_len": r_len, "len_est": len_est,
                    "num_rays": len(rays_db[fid][0]) + len(rays_db[fid][1]),
                    "condA": f_tri["condA"], "condB": f_tri["condB"]
                })

        def get_stats(arr):
            if not arr: return {"median": 0.0, "mean": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
            arr = np.abs(np.array(arr))
            return {
                "median": float(np.median(arr)), "mean": float(np.mean(arr)),
                "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99)),
                "max": float(np.max(arr))
            }

        stats_ray = get_stats(r_ray_all)
        stats_len = get_stats(r_len_all)
        stats_len_est = get_stats(len_est_all)


        print(f"  Point-to-Ray Dist (mm): med={stats_ray['median']:.4f}, p90={stats_ray['p90']:.4f}, max={stats_ray['max']:.4f}")
        print(f"  Wand Length Error (mm): med={stats_len['median']:.4f}, p90={stats_len['p90']:.4f}, max={stats_len['max']:.4f}")

        frame_metrics.sort(key=lambda x: x['score'], reverse=True)
        print("\n  Worst 5 Frames (by Score):")
        for f in frame_metrics[:5]:
            print(f"    Frame {f['fid']:<5}: Score={f['score']:<8.4f} RayMed={f['r_ray_med']:<8.4f} LenErr={f['r_len']:<8.4f}")

        suggestions = [f['fid'] for f in frame_metrics if f['r_ray_med'] > stats_ray['p99'] or abs(f['r_len']) > stats_len['p99']]

        import random
        sample_fids = random.sample(dataset['frames'], min(10, len(dataset['frames'])))
        samples = []
        for fid in sample_fids:
            f_tri = tri_data[fid]
            samples.append({
                "fid": fid, "XA": f_tri["XA"].tolist(), "XB": f_tri["XB"].tolist(),
                "raysA": [{"cam": r.cam_id, "uv": r.uv, "o": r.o.tolist(), "d": r.d.tolist()} for r in rays_db[fid][0]],
                "raysB": [{"cam": r.cam_id, "uv": r.uv, "o": r.o.tolist(), "d": r.d.tolist()} for r in rays_db[fid][1]]
            })
            
        report_path = Path(stored_dir).parent / "triangulation_report.json"
        
        def safe_json(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.float64, np.float32, np.float16)): return float(obj)
            if isinstance(obj, (np.int64, np.int32, np.int16)): return int(obj)
            if isinstance(obj, dict): return {k: safe_json(v) for k, v in obj.items()}
            if isinstance(obj, list): return [safe_json(x) for x in obj]
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return None
            return obj

        report = {
            "global_counts": {
                "total_frames": len(dataset['frames']), "frames_valid_both": frames_valid_both,
                "total_obs": total_obs, "invalid_obs": invalid_obs, "invalid_reasons": invalid_reasons
            },
            "residuals_mm": {"r_ray": stats_ray, "r_len": stats_len, "len_est": stats_len_est},
            "breakdown": {
                "per_camera": {cid: get_stats(vals) for cid, vals in per_cam_r_ray.items()},
                "per_window": {wid: get_stats(vals) for wid, vals in per_win_r_ray.items()}
            },
            "worst_frames": frame_metrics[:50],
            "sample_frames": samples,
            "suggestions": {
                "removal_threshold_ray": stats_ray['p99'],
                "removal_threshold_len": stats_len['p99'],
                "suggested_remove_fids": suggestions
            }
        }
        
        with open(report_path, 'w') as f_json:
            json.dump(safe_json(report), f_json, indent=2)
            
        print(f"\n[Refractive] Report exported to: {report_path}")
        print("[Refractive] Status: PR5 Intrinsic Refinement Completed.")
        print("="*50 + "\n")

        # Return signature matching view.py: success, cam_params, report, dataset
        # Inject window_planes into report for visualization
        report['window_planes'] = {wid: {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                         for k, v in pl.items()} 
                                   for wid, pl in self.window_planes.items()}
        
        # Inject all 3D points into dataset for visualization
        # Flatten simple list of points
        all_points_3d = []
        for fid, res in tri_data.items():
            if res['XA'] is not None: all_points_3d.extend(res['XA'].tolist())
            if res['XB'] is not None: all_points_3d.extend(res['XB'].tolist())
        dataset['points_3d'] = all_points_3d
        
        # === PLANE SIDE STATISTICS ===
        # Verify 3D points are on correct side of window planes relative to cameras
        # Also identify specific frame IDs with points on camera side
        if tri_data and self.window_planes and cam_params:
            print("\n[PR5] Plane Side Verification (eps=0.05mm):")
            eps_mm = 0.05
            
            for wid, pl in self.window_planes.items():
                plane_pt = np.array(pl['plane_pt'])
                plane_n = np.array(pl['plane_n'])
                plane_n = plane_n / np.linalg.norm(plane_n)
                
                # Find cameras that view through this window
                cams_for_win = [cid for cid, wid2 in cam_to_window.items() if wid2 == wid]
                
                for cid in cams_for_win:
                    if cid not in cam_params:
                        continue
                    
                    # Get camera center C
                    cp = cam_params[cid]
                    rvec = cp[0:3]
                    tvec = cp[3:6]
                    R = cv2.Rodrigues(rvec)[0]
                    C = -R.T @ tvec
                    s_C = np.dot(C - plane_pt, plane_n)
                    
                    # Check each frame's triangulated points
                    bad_fids = []
                    total_pts = 0
                    pts_on_cam_side = 0
                    
                    for fid, res in tri_data.items():
                        for label, X in [('A', res.get('XA')), ('B', res.get('XB'))]:
                            if X is not None:
                                total_pts += 1
                                sX = np.dot(X - plane_pt, plane_n)
                                # Check if on same side as camera (bad)
                                if sX * s_C > 0 or sX < eps_mm:
                                    pts_on_cam_side += 1
                                    bad_fids.append(fid)
                    
                    # Remove duplicates and sort
                    bad_fids = sorted(set(bad_fids))
                    pct_good = (total_pts - pts_on_cam_side) / max(total_pts, 1) * 100
                    
                    print(f"  Win {wid} Cam {cid}: s(C)={s_C:.1f}mm | pct_object_side={pct_good:.1f}%")
                    
                    if bad_fids:
                        print(f"    Frames with points on camera side ({len(bad_fids)}): {bad_fids[:20]}" + 
                              ("..." if len(bad_fids) > 20 else ""))
            print("")

        
        # Update cache with 3D points (for visualization when loading from cache)
        if not loaded_pr5:  # Only if we ran optimization (not loaded from cache)
            pr5_optimizer.save_cache(out_path, points_3d=all_points_3d)

        # === FINAL COORDINATE ALIGNMENT FOR VISUALIZATION ===
        # Align Y-axis with plane intersection line for 3D view
        if len(self.window_planes) >= 2:
            try:
                cam_params, self.window_planes, aligned_pts, R_align = align_world_y_to_plane_intersection(
                    self.window_planes, cam_params, points_3d=all_points_3d
                )
                # Update report and dataset with aligned data
                report['window_planes'] = {wid: {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                                 for k, v in pl.items()} 
                                           for wid, pl in self.window_planes.items()}
                if aligned_pts is not None:
                    dataset['points_3d'] = aligned_pts
                print("[Coordinate Alignment] Y-axis aligned with plane intersection for visualization.")
            except Exception as e:
                print(f"[Coordinate Alignment] Warning: final alignment failed: {e}")

        return True, cam_params, report, dataset



