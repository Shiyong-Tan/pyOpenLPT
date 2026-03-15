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
import faulthandler
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import json
from pathlib import Path
from datetime import datetime
import traceback



try:
    import pyopenlpt as lpt
except ImportError:
    lpt = None

from .refractive_geometry import (
    Ray, build_pinplate_rays_cpp_batch, triangulate_point, point_to_ray_dist,
    update_normal_tangent, camera_center, angle_between_vectors,
    update_cpp_camera_state, align_world_y_to_plane_intersection,
    enable_ray_tracking, reset_ray_stats, print_ray_stats_report,
    reset_camera_update_stats, print_camera_update_report,
    NativeSafetyValidationError,
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
        try:
            update_cpp_camera_state(cams_cpp[cam_id], validate_native=False, **update_kwargs)
        except Exception as e:
            intr = update_kwargs.get('intrinsics', {})
            ext = update_kwargs.get('extrinsics', {})
            pl = update_kwargs.get('plane_geom', {})
            med = update_kwargs.get('media_props', {})
            print(
                "[CRASH-LOC][CppSyncAdapter.apply] "
                f"cam={cam_id} err={repr(e)} "
                f"f={intr.get('f', None)} cx={intr.get('cx', None)} cy={intr.get('cy', None)} "
                f"rvec={ext.get('rvec', None)} tvec={ext.get('tvec', None)} "
                f"plane_pt={pl.get('pt', None)} plane_n={pl.get('n', None)} "
                f"thickness={med.get('thickness', None)}",
                flush=True,
            )
            print(traceback.format_exc(), flush=True)
            raise


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
    px_target: float = 1.0            # 投影误差目标 (px)
    wand_tol_pct: float = 0.02        # 棒长容忍度 (2% 即 0.2mm)
    lambda_base_per_cam: float = 2.0
    lambda_scale_by_mode: Dict[str, float] = field(default_factory=lambda: {
        "joint": 1.0,
        "final_refined": 1.0,
    })
    use_proj_residuals: bool = False
    sigma_proj_px: float = 1.0
    penalty_proj_px: float = 50.0
    
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
    barrier_continuation_enabled: bool = True
    barrier_schedule: Dict[str, Dict[str, object]] = field(default_factory=lambda: {
        'early': {'gate_scale': 0.05, 'beta_dir_scale': 0.05, 'tau': 0.1, 'soft_on': True},
        'mid': {'gate_scale': 0.3, 'beta_dir_scale': 0.3, 'tau': 0.03, 'soft_on': True},
        'final': {'gate_scale': 1.0, 'beta_dir_scale': 1.0, 'tau': 0.005, 'soft_on': False},
    })
    sigma_schedule: Dict[str, float] = field(default_factory=dict)
    tolerance_schedule: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'loop_planes': {'ftol': 5e-4, 'xtol': 1e-5, 'gtol': 1e-5},
        'loop_cams': {'ftol': 5e-4, 'xtol': 1e-5, 'gtol': 1e-5},
        'joint': {'ftol': 1e-5, 'xtol': 1e-5, 'gtol': 1e-5},
        'final_refined': {'ftol': 1e-5, 'xtol': 1e-5, 'gtol': 1e-5},
    })

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
    loss_cam: str = "linear"
    loss_joint: str = "linear"
    loss_round4: str = "linear"
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
    diff_step_mode_joint: str = "auto"     # manual|auto
    diff_step_mode_final: str = "auto"     # manual|auto
    optimization_strategy: str = "bundle"    # sequence|bundle
    bundle_point_delta_mm: Optional[float] = None
    bundle_retriangulate_each_round: bool = True
    # Per-round strategy override. Keys: loop_planes, loop_cams, joint, final_refined.
    # If key missing, fallback to `optimization_strategy`.
    round_strategy: Dict[str, str] = field(default_factory=lambda: {
        "loop_planes": "bundle",
        "loop_cams": "bundle",
        "joint": "sequence",
        "final_refined": "sequence",
    })

    # Chunked early-stop framework (reusable across rounds)
    chunk_modes: Dict[str, bool] = field(default_factory=lambda: {
        "joint": True,
        "final_refined": True,
    })
    chunk_joint_schedule: List[int] = field(default_factory=lambda: [20, 5, 5, 5, 5, 5, 5])
    chunk_final_refined_schedule: List[int] = field(default_factory=lambda: [30] + [5] * 14)
    chunk_patience_chunks: int = 2
    chunk_rel_eps_ray: float = 5e-3
    chunk_rel_eps_len: float = 5e-3
    chunk_align_retry_enabled: bool = True
    chunk_align_retry_max: int = 1
    chunk_align_retry_modes: List[str] = field(default_factory=lambda: ["yz", "xz", "xy"])
    
    
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
        self._barrier_profile = {
            'name': 'final',
            'gate_scale': 1.0,
            'beta_dir_scale': 1.0,
            'tau': 0.01,
            'soft_on': False,
        }
        
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

        # Bundle-strategy point state: fid -> {'A': (3,), 'B': (3,)}
        self._bundle_points: Dict[int, Dict[str, np.ndarray]] = {}
        self._bundle_points_ref: Dict[int, Dict[str, np.ndarray]] = {}

        # Per-round plane references for anchor parameterization.
        self._plane_anchor = {}
        self._plane_d0 = {}
        self._refresh_plane_round_reference()

        # Global alignment-mode counter: consumed by pre-last alignment and all
        # chunk retry alignments across joint/final_refined.
        self._align_mode_counter = 0
        
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
        self._last_proj_rmse = -1.0
        
        self.sigma_ray_global = 0.04  # Default, will be recalculated
        self.sigma_wand = 0.1        # Default, will be recalculated

        # Recompute per-round plane references after relinearization.
        self._refresh_plane_round_reference()

    def _refresh_plane_round_reference(self):
        """Rebuild plane anchors and signed-distance bases for current round.

        Anchor A is the mean camera center of cameras mapped to each window
        using current initial camera parameters. If unavailable, fall back to
        the current plane point.
        """
        self._plane_anchor = {}
        self._plane_d0 = {}

        for wid in self.window_ids:
            pl = self.initial_planes.get(wid, self.window_planes.get(wid, {}))
            if not pl:
                continue

            n0 = np.asarray(pl['plane_n'], dtype=np.float64)
            pt0 = np.asarray(pl['plane_pt'], dtype=np.float64)

            centers = []
            for cid in self.window_to_cams.get(wid, []):
                p = self.initial_cam_params.get(cid)
                if p is None:
                    continue
                R, _ = cv2.Rodrigues(p[0:3])
                C = camera_center(R, p[3:6])
                centers.append(C)

            if centers:
                A = np.mean(np.asarray(centers), axis=0)
            else:
                A = pt0.copy()

            self._plane_anchor[wid] = A
            self._plane_d0[wid] = float(np.dot(n0, pt0 - A))
    
    
    def _build_obs_cache(self):
        """Build observation cache from dataset."""
        self.obs_cache = ObsCacheBuilder.build(self.dataset, self.active_cam_ids)

    def _diag_log(self, msg: str):
        print(msg, flush=True)

    def _diag_ctx(self) -> str:
        mode = getattr(self, '_diag_current_mode', '?')
        return f"mode={mode}"

    def _make_invalid_ray(self, cam_id: int, frame_id: int, endpoint: str, reason: str = "missing_cam", uv=None) -> Ray:
        """Create a consistent invalid ray placeholder for missing/failed paths."""
        if uv is None:
            uv = (np.nan, np.nan)
        return Ray(
            o=np.zeros(3),
            d=np.array([0, 0, 1.0]),
            valid=False,
            reason=reason,
            cam_id=cam_id,
            window_id=self.cam_to_window.get(cam_id, -1),
            frame_id=frame_id,
            endpoint=endpoint,
            uv=(float(uv[0]), float(uv[1])),
        )

    def _build_batched_ray_lookup(self, per_cam_items: Dict[int, List[Tuple[int, np.ndarray]]], endpoint: str) -> Dict[Tuple[int, int], Ray]:
        """
        Batch-build rays grouped by camera.

        Args:
            per_cam_items: {cam_id: [(frame_id, uv_px), ...]}
            endpoint: endpoint label used for diagnostics metadata

        Returns:
            Dict[(frame_id, cam_id)] -> Ray
        """
        ray_lookup: Dict[Tuple[int, int], Ray] = {}

        for cam_id, items in per_cam_items.items():
            cam_obj = self.cams_cpp.get(cam_id)
            if cam_obj is None:
                for frame_id, uv in items:
                    ray_lookup[(frame_id, cam_id)] = self._make_invalid_ray(
                        cam_id=cam_id,
                        frame_id=frame_id,
                        endpoint=endpoint,
                        reason="missing_cam",
                        uv=uv,
                    )
                continue

            uv_list = [uv for _, uv in items]
            meta_list = [
                {
                    "cam_id": cam_id,
                    "window_id": self.cam_to_window.get(cam_id, -1),
                    "frame_id": frame_id,
                    "endpoint": endpoint,
                }
                for frame_id, _ in items
            ]
            try:
                rays = build_pinplate_rays_cpp_batch(cam_obj, uv_list, meta_list=meta_list)
            except Exception as e:
                uv0 = uv_list[0] if uv_list else None
                uv1 = uv_list[-1] if uv_list else None
                self._diag_log(
                    "[CRASH-LOC][build_pinplate_rays_cpp_batch] "
                    f"{self._diag_ctx()} endpoint={endpoint} cam={cam_id} n={len(uv_list)} "
                    f"uv_first={uv0} uv_last={uv1} err={repr(e)}"
                )
                self._diag_log(traceback.format_exc())
                raise

            for (frame_id, _), ray in zip(items, rays):
                ray_lookup[(frame_id, cam_id)] = ray

        return ray_lookup
    

    def _compute_physical_sigmas(self):
        """Estimate global sigma values across all optimize stages."""
        cfg = self.config
        
        all_f = [p[6] for p in self.cam_params.values() if len(p) > 6]
        avg_f = np.mean(all_f) if all_f else 1000.0
        
        sample_dists = []
        fids_to_sample = self.fids_optim[::max(1, len(self.fids_optim) // 100)]

        # Batch-build endpoint A rays once per camera for sampled frames.
        per_cam_items_A: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for fid in fids_to_sample:
            obs = self.dataset['obsA'].get(fid, {})
            for cid, uv in obs.items():
                per_cam_items_A.setdefault(cid, []).append((fid, uv))
        ray_lookup_A = self._build_batched_ray_lookup(per_cam_items_A, endpoint="A")

        for fid in fids_to_sample:
            # Reconstruct centers
            obs = self.dataset['obsA'].get(fid, {})
            cids = sorted(obs.keys())
            rays = []
            for cid in cids:
                r = ray_lookup_A.get((fid, cid))
                if r is not None and r.valid:
                    rays.append(r)
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

    def _resolve_barrier_profile(self, mode: str) -> Dict[str, object]:
        cfg = self.config
        schedule = cfg.barrier_schedule or {}
        default_profile = {
            'name': 'final',
            'gate_scale': 1.0,
            'beta_dir_scale': 1.0,
            'tau': 0.01,
            'soft_on': False,
        }

        if not cfg.barrier_continuation_enabled:
            return default_profile

        profile_name = 'final'
        if mode.startswith('loop_'):
            loop_idx = None
            parts = mode.split('_')
            if len(parts) >= 2:
                try:
                    loop_idx = int(parts[1])
                except ValueError:
                    loop_idx = None
            if loop_idx is not None and loop_idx <= 2:
                profile_name = 'early'
            else:
                profile_name = 'mid'
        elif mode == 'joint':
            profile_name = 'mid'
        elif mode == 'final_refined' or mode == 'final':
            profile_name = 'final'

        base = schedule.get(profile_name, {})
        profile = {
            'name': profile_name,
            'gate_scale': float(base.get('gate_scale', default_profile['gate_scale'])),
            'beta_dir_scale': float(base.get('beta_dir_scale', default_profile['beta_dir_scale'])),
            'tau': float(base.get('tau', default_profile['tau'])),
            'soft_on': bool(base.get('soft_on', default_profile['soft_on'])),
        }
        profile['tau'] = max(profile['tau'], 1e-6)
        return profile

    def _set_barrier_profile_for_mode(self, mode: str, log: bool = False) -> None:
        self._barrier_profile = self._resolve_barrier_profile(mode)
        if log and self.config.verbosity >= 1:
            p = self._barrier_profile
            self.reporter.detail(
                f"  [Barrier] profile={p['name']}, gate_scale={p['gate_scale']:.3f}, "
                f"beta_dir_scale={p['beta_dir_scale']:.3f}, tau={p['tau']:.3f}, soft_on={p['soft_on']}"
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

    def _compute_slot_counts(self) -> Tuple[int, int, int, int]:
        """Return (ray_slots, len_slots, proj_slots, barrier_slots) for penalty vector sizing."""
        total_ray = 0
        total_barrier = 0
        for fid in self.fids_optim:
            for key in ['obsA', 'obsB']:
                obs = self.dataset[key].get(fid, {})
                total_ray += len(obs)
                if obs:
                    wids = set()
                    for cid in obs.keys():
                        wid = self.cam_to_window.get(cid)
                        if wid is not None and wid != -1:
                            wids.add(wid)
                    total_barrier += 2 * len(wids)
        total_len = len(self.fids_optim)
        total_proj = 2 * total_ray if self.config.use_proj_residuals else 0
        return total_ray, total_len, total_proj, total_barrier

    def _make_full_penalty_residuals(self) -> Tuple[np.ndarray, float, float, int, int, float, int]:
        """Return a fixed-shape finite penalty residual package for invalid BA states."""
        pen_ray, pen_len, pen_proj, pen_barrier = self._compute_slot_counts()
        pen_total = pen_ray + pen_len + pen_proj + pen_barrier
        penalty_vec = np.full(pen_total, 1e6, dtype=np.float64)
        return penalty_vec, 1e12, 1e12, max(pen_ray, 1), max(pen_len, 1), 1e12, max(pen_proj, 1)

    def evaluate_residuals(self, planes: Dict[int, Dict], cam_params: Dict[int, np.ndarray],
                           lambda_eff: float, window_media: Optional[Dict[int, Dict]] = None,
                           explicit_points: Optional[Dict[int, Dict[str, np.ndarray]]] = None
                           ) -> Tuple[np.ndarray, float, float, int, int, float, int]:
        """
        Evaluate residuals with fixed-size padding for Scipy least_squares compatibility.
        """
        # Apply planes and extrinsics
        # Apply to C++ objects (Consolidated Update)
        media = window_media or self.window_media

        update_plan: List[Tuple[int, Dict]] = []
        for cid in self.active_cam_ids:
            if cid not in self.cams_cpp:
                continue
            update_kwargs = CppSyncAdapter.build_update_kwargs(
                cam_params=cam_params,
                window_planes=planes,
                window_media=media,
                cam_to_window=self.cam_to_window,
                cam_id=cid,
            )
            update_plan.append((cid, update_kwargs))

        for cid, update_kwargs in update_plan:
            try:
                CppSyncAdapter.apply(self.cams_cpp, cid, update_kwargs)
            except Exception as e:
                intr = update_kwargs.get('intrinsics', {})
                ext = update_kwargs.get('extrinsics', {})
                pl = update_kwargs.get('plane_geom', {})
                med = update_kwargs.get('media_props', {})
                err_msg = repr(e)
                self._diag_log(
                    f"[CRASH-LOC][update_cpp_camera_state] "
                    f"{self._diag_ctx()} cam={cid} err={err_msg} "
                    f"f={intr.get('f', None)} cx={intr.get('cx', None)} cy={intr.get('cy', None)} "
                    f"rvec={ext.get('rvec', None)} tvec={ext.get('tvec', None)} "
                    f"plane_pt={pl.get('pt', None)} plane_n={pl.get('n', None)} "
                    f"thickness={med.get('thickness', None)}"
                )
                self._diag_log(traceback.format_exc())
                raise

        self._explicit_points = {}
        
        # 1. Pre-calculate total possible counts for FIXED size
        total_ray_slots = 0
        for fid in self.fids_optim:
            for key in ['obsA', 'obsB']:
                obs = self.dataset[key].get(fid, {})
                total_ray_slots += len(obs)
        
        total_len_slots = len(self.fids_optim)
        total_proj_slots = 2 * total_ray_slots if self.config.use_proj_residuals else 0

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
        res_proj_fixed = np.zeros(total_proj_slots)
        res_len_fixed = np.zeros(total_len_slots)
        res_barrier_fixed = np.zeros(total_barrier_slots)
        
        PENALTY_RAY = 100.0   # mm
        PENALTY_LEN = self.wand_length
        PENALTY_PROJ = float(self.config.penalty_proj_px)
        sigma_proj = max(float(self.config.sigma_proj_px), 1e-9)
        
        idx_ray = 0
        idx_proj = 0
        idx_len = 0
        idx_barrier = 0
        
        S_ray = 0.0
        S_len = 0.0
        S_proj = 0.0
        N_ray_actual = 0
        N_len_actual = 0
        N_proj_actual = 0
        num_triangulated_points = 0
        valid_points_data = [] # For barrier computation

        explicit_total = 0
        explicit_ok = 0
        fallback_used = 0
        fallback_ok = 0

        # Batch-build all rays up front, grouped by camera (largest runtime hotspot).
        per_cam_items_A: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        per_cam_items_B: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for fid in self.fids_optim:
            obs_A_f = self.dataset['obsA'].get(fid, {})
            for cid, uv in obs_A_f.items():
                per_cam_items_A.setdefault(cid, []).append((fid, uv))

            obs_B_f = self.dataset['obsB'].get(fid, {})
            for cid, uv in obs_B_f.items():
                per_cam_items_B.setdefault(cid, []).append((fid, uv))

        ray_lookup_A = self._build_batched_ray_lookup(per_cam_items_A, endpoint="A")
        ray_lookup_B = self._build_batched_ray_lookup(per_cam_items_B, endpoint="B")
        per_cam_proj_items: Dict[int, List[Tuple[int, str, np.ndarray, int]]] = {}
        points_by_fid: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
        
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
                r = ray_lookup_A.get((fid, cid))
                if r is None:
                    r = self._make_invalid_ray(cam_id=cid, frame_id=fid, endpoint="A", reason="missing_ray", uv=obs_A[cid])
                rays_A_all.append(r)

            if explicit_points is not None and fid in explicit_points and explicit_points[fid].get('A') is not None:
                XA_explicit = np.asarray(explicit_points[fid]['A'], dtype=np.float64)
                okA = True
            else:
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

            if self.config.use_proj_residuals:
                for cid in cids_A:
                    per_cam_proj_items.setdefault(cid, []).append((fid, 'A', obs_A[cid], idx_proj))
                    idx_proj += 2
            
            # Advance barrier index by fixed amount
            idx_barrier += 2 * n_barrier_A

            # --- Endpoint B ---
            n_slots_B = len(obs_B)
            cids_B = sorted(obs_B.keys())
            rays_B_all = []
            for cid in cids_B:
                r = ray_lookup_B.get((fid, cid))
                if r is None:
                    r = self._make_invalid_ray(cam_id=cid, frame_id=fid, endpoint="B", reason="missing_ray", uv=obs_B[cid])
                rays_B_all.append(r)

            if explicit_points is not None and fid in explicit_points and explicit_points[fid].get('B') is not None:
                XB_explicit = np.asarray(explicit_points[fid]['B'], dtype=np.float64)
                okB = True
            else:
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

            if self.config.use_proj_residuals:
                for cid in cids_B:
                    per_cam_proj_items.setdefault(cid, []).append((fid, 'B', obs_B[cid], idx_proj))
                    idx_proj += 2

            points_by_fid[fid] = {
                'A': np.asarray(XA, dtype=np.float64) if validA else None,
                'B': np.asarray(XB, dtype=np.float64) if validB else None,
            }
            
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

        if self.config.use_proj_residuals and per_cam_proj_items:
            import pyopenlpt as lpt
            for cid, items in per_cam_proj_items.items():
                cam = self.cams_cpp.get(cid, None)
                if cam is None:
                    for _, _, _, idx in items:
                        res_proj_fixed[idx] = PENALTY_PROJ / sigma_proj
                        res_proj_fixed[idx + 1] = PENALTY_PROJ / sigma_proj
                    continue

                pts3d = []
                idx_map = []
                uv_obs_map = []
                for fid, endpoint, uv_obs, idx in items:
                    pt = points_by_fid.get(fid, {}).get(endpoint, None)
                    if pt is None or not np.all(np.isfinite(pt)):
                        res_proj_fixed[idx] = PENALTY_PROJ / sigma_proj
                        res_proj_fixed[idx + 1] = PENALTY_PROJ / sigma_proj
                        continue
                    pts3d.append(lpt.Pt3D(float(pt[0]), float(pt[1]), float(pt[2])))
                    idx_map.append(idx)
                    uv_obs_map.append(uv_obs)

                if not pts3d:
                    continue

                try:
                    proj_results = cam.projectBatchStatus(pts3d, False)
                except Exception:
                    for idx in idx_map:
                        res_proj_fixed[idx] = PENALTY_PROJ / sigma_proj
                        res_proj_fixed[idx + 1] = PENALTY_PROJ / sigma_proj
                    continue

                for (idx, uv_obs, proj) in zip(idx_map, uv_obs_map, proj_results):
                    okp, uvp, _ = proj
                    if not okp:
                        res_proj_fixed[idx] = PENALTY_PROJ / sigma_proj
                        res_proj_fixed[idx + 1] = PENALTY_PROJ / sigma_proj
                        continue
                    dx = float(uvp[0] - uv_obs[0])
                    dy = float(uvp[1] - uv_obs[1])
                    res_proj_fixed[idx] = dx / sigma_proj
                    res_proj_fixed[idx + 1] = dy / sigma_proj
                    S_proj += dx * dx + dy * dy
                    N_proj_actual += 2

        # 2. Side Barrier (Adaptive)
        J_data = S_ray + lambda_eff * S_len
        margin_mm = self.config.margin_side_mm
        sX_vals = []
        profile = self._barrier_profile if hasattr(self, '_barrier_profile') else {
            'gate_scale': 1.0,
            'beta_dir_scale': 1.0,
            'tau': 0.01,
            'soft_on': False,
        }

        # Hard Barrier Constants (continuation-aware)
        C_gate = self.config.alpha_side_gate * float(profile['gate_scale']) * self._j_ref
        r_fix_const = np.sqrt(2.0 * C_gate)
        beta_dir_eff = self.config.beta_side_dir * float(profile['beta_dir_scale'])
        r_grad_const = np.sqrt(2.0 * beta_dir_eff)
        tau = float(profile['tau'])
        soft_on = bool(profile['soft_on'])
        beta_soft_eff = self.config.beta_side_soft * float(profile['beta_dir_scale'])
        r_soft_const = np.sqrt(2.0 * beta_soft_eff) if soft_on else 0.0

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

                # Soft floor guidance in early/mid stages.
                if soft_on:
                    # Numerically stable softplus: tau*log(1+exp(gap/tau))
                    z = gap / max(tau, 1e-12)
                    softplus_z = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
                    soft_step = r_soft_const * tau * softplus_z
                    res_barrier_fixed[curr_b_idx] += soft_step
                
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
        self._last_proj_rmse = np.sqrt(S_proj / max(1, N_proj_actual)) if N_proj_actual > 0 else 0.0

        # 3. Combine
        weighted_len = np.sqrt(lambda_eff) * res_len_fixed
        residuals = np.concatenate([res_ray_fixed, res_proj_fixed, weighted_len, res_barrier_fixed])
            
        return residuals, S_ray, S_len, N_ray_actual, N_len_actual, S_proj, N_proj_actual

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
            A = self._plane_anchor.get(wid, self.initial_planes[wid]['plane_pt'])
            d0 = self._plane_d0.get(
                wid,
                float(np.dot(self.initial_planes[wid]['plane_n'], self.initial_planes[wid]['plane_pt'] - A))
            )
            
            # 1. Update normal (alpha, beta) using tangent space
            alpha, beta = deltas['a'], deltas['b']
            n_new = update_normal_tangent(n0, alpha, beta)

            # 2. Update signed distance along UPDATED normal in anchor coordinates
            d_new = d0 + deltas['d']
            pt_new = A + d_new * n_new
            
            current_planes[wid]['plane_n'] = n_new
            current_planes[wid]['plane_pt'] = pt_new
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
        self._diag_current_mode = mode
        # Unpack
        curr_planes, curr_cams, curr_media = self._unpack_params_delta(x, layout)
        
        # Data Residuals
        # Note: evaluate_residuals handles applying to CPP internally
        residuals, S_ray, S_len, N_ray, N_len, S_proj, N_proj = self.evaluate_residuals(
            curr_planes, curr_cams, lambda_eff, window_media=curr_media
        )
        
        # [Fix] Update live stats for logging
        self._last_ray_rmse = np.sqrt(S_ray / max(N_ray, 1))
        self._last_len_rmse = np.sqrt(S_len / max(N_len, 1)) if N_len > 0 else 0.0
        self._last_proj_rmse = np.sqrt(S_proj / max(N_proj, 1)) if N_proj > 0 else 0.0
        
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

        res_arr = np.asarray(residuals, dtype=np.float64)
        if not np.all(np.isfinite(res_arr)):
            # Keep optimizer alive under occasional geometric singularities.
            res_arr = np.nan_to_num(res_arr, nan=1e6, posinf=1e6, neginf=-1e6)
        return res_arr

    def _mode_strategy_key(self, mode: str) -> str:
        """Map internal mode string to round_strategy key."""
        if mode.startswith('loop_') and mode.endswith('_planes'):
            return 'loop_planes'
        if mode.startswith('loop_') and mode.endswith('_cams'):
            return 'loop_cams'
        if mode == 'joint':
            return 'joint'
        if mode == 'final_refined':
            return 'final_refined'
        return mode

    def _resolve_strategy(self, mode: str, strategy_override: Optional[str] = None) -> str:
        """Resolve optimization strategy with override > per-round map > global fallback."""
        if strategy_override is not None:
            strategy = str(strategy_override).lower()
        else:
            key = self._mode_strategy_key(mode)
            per_round = getattr(self.config, 'round_strategy', {}) or {}
            strategy = str(per_round.get(key, getattr(self.config, 'optimization_strategy', 'sequence'))).lower()

        if strategy not in ('sequence', 'bundle'):
            self.reporter.detail(f"[RoundConfig] invalid strategy={strategy}, fallback to sequence")
            strategy = 'sequence'
        return strategy

    def _get_chunk_schedule_for_mode(self, mode: str) -> List[int]:
        cfg = self.config
        enabled = bool((cfg.chunk_modes or {}).get(mode, False))
        if not enabled:
            return []
        if mode == 'joint':
            return [int(max(1, x)) for x in (cfg.chunk_joint_schedule or [])]
        if mode == 'final_refined':
            return [int(max(1, x)) for x in (cfg.chunk_final_refined_schedule or [])]
        return []

    def _compute_lambda_fixed(self, mode: str) -> float:
        n_cams = max(1, len(self.cam_params))
        base = float(getattr(self.config, 'lambda_base_per_cam', 2.0)) * n_cams
        key = self._mode_strategy_key(mode)
        scale_map = getattr(self.config, 'lambda_scale_by_mode', {}) or {}
        scale = float(scale_map.get(key, 1.0))
        return base * scale

    def _compute_current_rmse_for_chunk(self, mode: str) -> Tuple[float, float]:
        lambda_fixed = self._compute_lambda_fixed(mode)
        _, S_ray, S_len, N_ray, N_len, _, _ = self.evaluate_residuals(
            self.window_planes, self.cam_params, lambda_fixed, window_media=self.window_media
        )
        ray_rmse = float(np.sqrt(S_ray / max(1, N_ray)))
        len_rmse = float(np.sqrt(S_len / max(1, N_len))) if N_len > 0 else 0.0
        return ray_rmse, len_rmse

    def _snapshot_reference_state(self) -> Dict[str, Dict]:
        return {
            'planes': {
                int(wid): {
                    'plane_n': np.asarray(pl['plane_n'], dtype=np.float64).copy(),
                    'plane_pt': np.asarray(pl['plane_pt'], dtype=np.float64).copy(),
                    'initialized': bool(pl.get('initialized', True)),
                }
                for wid, pl in self.initial_planes.items()
            },
            'cams': {int(cid): np.asarray(p, dtype=np.float64).copy() for cid, p in self.initial_cam_params.items()},
            'media': {int(wid): dict(m) for wid, m in self.initial_media.items()},
        }

    def _apply_reference_state(self, ref_state: Dict[str, Dict]) -> None:
        self.initial_planes = {
            int(wid): {
                'plane_n': np.asarray(pl['plane_n'], dtype=np.float64).copy(),
                'plane_pt': np.asarray(pl['plane_pt'], dtype=np.float64).copy(),
                'initialized': bool(pl.get('initialized', True)),
            }
            for wid, pl in (ref_state.get('planes', {}) or {}).items()
        }
        self.initial_cam_params = {
            int(cid): np.asarray(p, dtype=np.float64).copy()
            for cid, p in (ref_state.get('cams', {}) or {}).items()
        }
        self.initial_media = {
            int(wid): dict(m)
            for wid, m in (ref_state.get('media', {}) or {}).items()
        }
        self._refresh_plane_round_reference()

    def _get_retry_alignment_mode(self, retry_count: int = 0) -> str:
        modes = [str(m).lower() for m in (getattr(self.config, 'chunk_align_retry_modes', []) or [])]
        modes = [m for m in modes if m in ('yz', 'xz', 'xy')]
        if not modes:
            return 'yz'
        idx = int(self._align_mode_counter) % len(modes)
        mode = modes[idx]
        self._align_mode_counter += 1
        return mode

    def _run_round_chunked(
        self,
        round_name: str,
        base_kwargs: Dict,
        chunk_schedule: List[int],
        retry_count: int = 0,
        freeze_bounds_reference: bool = False,
        bounds_ref_state: Optional[Dict[str, Dict]] = None,
        chunk_x_prev: Optional[np.ndarray] = None,
    ):
        cfg = self.config
        patience = max(1, int(cfg.chunk_patience_chunks))
        eps_ray = float(cfg.chunk_rel_eps_ray)
        eps_len = float(cfg.chunk_rel_eps_len)

        prev_best_ray, prev_best_len = self._compute_current_rmse_for_chunk(round_name)
        no_improve_chunks = 0
        final_result = None

        self.reporter.detail(
            f"  [ChunkPlan] {round_name}: schedule={chunk_schedule}, patience={patience}, "
            f"eps_ray={eps_ray:g}, eps_len={eps_len:g}, retry={retry_count}/{cfg.chunk_align_retry_max}"
        )

        if freeze_bounds_reference and bounds_ref_state is None:
            bounds_ref_state = self._snapshot_reference_state()
            self.reporter.detail(f"  [ChunkPlan] {round_name}: freeze bounds to first-chunk reference")

        for i, chunk_nfev in enumerate(chunk_schedule, start=1):
            kwargs = dict(base_kwargs)
            kwargs['max_nfev'] = int(chunk_nfev)
            if freeze_bounds_reference and bounds_ref_state is not None:
                self._apply_reference_state(bounds_ref_state)
                kwargs['freeze_bounds_reference'] = True
                kwargs['bounds_ref_state'] = bounds_ref_state
                if chunk_x_prev is not None:
                    kwargs['x0_override'] = np.asarray(chunk_x_prev, dtype=np.float64).copy()
            self.reporter.detail(f"  [Chunk] {round_name} {i}/{len(chunk_schedule)} max_nfev={chunk_nfev}")
            final_result = self._optimize_generic(**kwargs)

            # _optimize_generic/_optimize_bundle_generic return (res, layout)
            # while legacy paths may return res directly.
            res_obj = None
            if isinstance(final_result, tuple):
                if len(final_result) >= 1:
                    res_obj = final_result[0]
            else:
                res_obj = final_result

            if res_obj is not None and hasattr(res_obj, 'x'):
                chunk_x_prev = np.asarray(res_obj.x, dtype=np.float64).copy()

            cur_ray, cur_len = self._compute_current_rmse_for_chunk(round_name)
            imp_ray = (prev_best_ray - cur_ray) / max(abs(prev_best_ray), 1e-12)
            imp_len = (prev_best_len - cur_len) / max(abs(prev_best_len), 1e-12)
            improved = (imp_ray > eps_ray) or (imp_len > eps_len)

            self.reporter.detail(
                f"    [ChunkResult] ray={cur_ray:.6f}mm len={cur_len:.6f}mm "
                f"improve_ray={imp_ray:.4%} improve_len={imp_len:.4%} improved={improved}"
            )

            prev_best_ray = min(prev_best_ray, cur_ray)
            prev_best_len = min(prev_best_len, cur_len)

            if improved:
                no_improve_chunks = 0
            else:
                no_improve_chunks += 1

            if no_improve_chunks >= patience:
                self.reporter.detail(
                    f"  [EarlyStop] {round_name}: no improvement for {no_improve_chunks} chunks"
                )
                can_retry = (
                    bool(cfg.chunk_align_retry_enabled)
                    and retry_count < int(cfg.chunk_align_retry_max)
                )
                if can_retry:
                    self.reporter.section(f"EarlyStop Retry: {round_name}")
                    retry_mode = self._get_retry_alignment_mode(retry_count)
                    self.reporter.detail(f"  [Retry] {round_name}: alignment mode={retry_mode}")
                    aligned = self._apply_coordinate_alignment(
                        tag=f"{round_name}-earlystop-retry-{retry_count+1}",
                        refresh_initial=True,
                        align_mode=retry_mode,
                    )
                    if aligned:
                        self.reporter.detail(
                            f"  [Retry] {round_name}: alignment applied, rerun same chunk schedule"
                        )
                        return self._run_round_chunked(
                            round_name=round_name,
                            base_kwargs=base_kwargs,
                            chunk_schedule=chunk_schedule,
                            retry_count=retry_count + 1,
                            freeze_bounds_reference=freeze_bounds_reference,
                            bounds_ref_state=None if freeze_bounds_reference else bounds_ref_state,
                            chunk_x_prev=None,
                        )
                break

        return final_result

    def _ensure_bundle_points_initialized(self, cam_params: Dict[int, np.ndarray], planes: Dict[int, Dict], media: Dict[int, Dict]) -> None:
        """Initialize SBA explicit 3D points once from current camera/plane state."""
        retri_each_round = bool(getattr(self.config, 'bundle_retriangulate_each_round', True))
        if self._bundle_points and (not retri_each_round):
            return

        self.sync_cpp_state(cam_params=cam_params, window_planes=planes, window_media=media)

        points: Dict[int, Dict[str, np.ndarray]] = {}

        per_cam_items_A: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        per_cam_items_B: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for fid in self.fids_optim:
            for cid, uv in self.dataset.get('obsA', {}).get(fid, {}).items():
                per_cam_items_A.setdefault(cid, []).append((fid, uv))
            for cid, uv in self.dataset.get('obsB', {}).get(fid, {}).items():
                per_cam_items_B.setdefault(cid, []).append((fid, uv))

        ray_lookup_A = self._build_batched_ray_lookup(per_cam_items_A, endpoint="A")
        ray_lookup_B = self._build_batched_ray_lookup(per_cam_items_B, endpoint="B")

        for fid in self.fids_optim:
            obsA = self.dataset.get('obsA', {}).get(fid, {})
            obsB = self.dataset.get('obsB', {}).get(fid, {})

            raysA = []
            for cid in sorted(obsA.keys()):
                r = ray_lookup_A.get((fid, cid))
                if r is not None and r.valid:
                    raysA.append(r)
            XA = None
            if len(raysA) >= 2:
                XA, _, okA, _ = triangulate_point(raysA)
                if not okA:
                    XA = None

            raysB = []
            for cid in sorted(obsB.keys()):
                r = ray_lookup_B.get((fid, cid))
                if r is not None and r.valid:
                    raysB.append(r)
            XB = None
            if len(raysB) >= 2:
                XB, _, okB, _ = triangulate_point(raysB)
                if not okB:
                    XB = None

            if XA is None and XB is None:
                XA = np.zeros(3, dtype=np.float64)
                XB = np.array([0.0, 0.0, self.wand_length], dtype=np.float64)
            elif XA is None:
                XA = np.asarray(XB, dtype=np.float64) + np.array([0.0, 0.0, -self.wand_length], dtype=np.float64)
            elif XB is None:
                XB = np.asarray(XA, dtype=np.float64) + np.array([0.0, 0.0, self.wand_length], dtype=np.float64)

            points[fid] = {
                'A': np.asarray(XA, dtype=np.float64),
                'B': np.asarray(XB, dtype=np.float64),
            }

        self._bundle_points = points
        self._bundle_points_ref = {fid: {'A': p['A'].copy(), 'B': p['B'].copy()} for fid, p in points.items()}

    def _transform_bundle_points(self, R_world: np.ndarray, t_shift: np.ndarray) -> None:
        """Apply same coordinate transform to bundle point state as cameras/planes."""
        if not self._bundle_points:
            return

        t_shift = np.asarray(t_shift, dtype=np.float64).reshape(3)

        def _xfm(pt: np.ndarray) -> np.ndarray:
            p = np.asarray(pt, dtype=np.float64).reshape(3)
            return R_world @ (p + t_shift)

        for fid, ep in self._bundle_points.items():
            if ep.get('A') is not None:
                ep['A'] = _xfm(ep['A'])
            if ep.get('B') is not None:
                ep['B'] = _xfm(ep['B'])

        self._bundle_points_ref = {
            fid: {
                'A': ep['A'].copy() if ep.get('A') is not None else None,
                'B': ep['B'].copy() if ep.get('B') is not None else None,
            }
            for fid, ep in self._bundle_points.items()
        }

    def _build_bundle_layout(self, base_layout: List[Tuple]) -> List[Tuple]:
        layout = list(base_layout)
        for fid in self.fids_optim:
            for k in range(3):
                layout.append(('ptA', fid, k))
            for k in range(3):
                layout.append(('ptB', fid, k))
        return layout

    def _build_bundle_jac_sparsity(self, full_layout: List[Tuple]):
        """Build jacobian sparsity for bundle residual structure."""
        n_params = len(full_layout)

        # Residual slot counts must match evaluate_residuals()
        total_ray_slots = 0
        total_len_slots = len(self.fids_optim)
        total_barrier_slots = 0
        for fid in self.fids_optim:
            obsA = self.dataset.get('obsA', {}).get(fid, {})
            obsB = self.dataset.get('obsB', {}).get(fid, {})
            total_ray_slots += len(obsA) + len(obsB)
            if obsA:
                wids = {self.cam_to_window.get(cid) for cid in obsA.keys()}
                wids = {w for w in wids if w is not None and w != -1}
                total_barrier_slots += 2 * len(wids)
            if obsB:
                wids = {self.cam_to_window.get(cid) for cid in obsB.keys()}
                wids = {w for w in wids if w is not None and w != -1}
                total_barrier_slots += 2 * len(wids)

        total_proj_slots = 2 * total_ray_slots if self.config.use_proj_residuals else 0

        n_res = total_ray_slots + total_proj_slots + total_len_slots + total_barrier_slots
        J = lil_matrix((n_res, n_params), dtype=np.int8)

        cam_var_idx: Dict[int, List[int]] = {}
        plane_var_idx: Dict[int, List[int]] = {}
        ptA_var_idx: Dict[int, List[int]] = {}
        ptB_var_idx: Dict[int, List[int]] = {}

        for i, (ptype, pid, subidx) in enumerate(full_layout):
            if ptype.startswith('cam_'):
                cam_var_idx.setdefault(pid, []).append(i)
            elif ptype in ('plane_d', 'plane_a', 'plane_b', 'win_t'):
                plane_var_idx.setdefault(pid, []).append(i)
            elif ptype == 'ptA':
                ptA_var_idx.setdefault(pid, []).append(i)
            elif ptype == 'ptB':
                ptB_var_idx.setdefault(pid, []).append(i)

        row_ray = 0
        row_proj = total_ray_slots
        row_len = total_ray_slots + total_proj_slots
        row_bar = total_ray_slots + total_proj_slots + total_len_slots

        for fid in self.fids_optim:
            obsA = self.dataset.get('obsA', {}).get(fid, {})
            obsB = self.dataset.get('obsB', {}).get(fid, {})

            # Endpoint A ray residuals
            for cid in sorted(obsA.keys()):
                for vi in cam_var_idx.get(cid, []):
                    J[row_ray, vi] = 1
                wid = self.cam_to_window.get(cid)
                for vi in plane_var_idx.get(wid, []):
                    J[row_ray, vi] = 1
                for vi in ptA_var_idx.get(fid, []):
                    J[row_ray, vi] = 1
                row_ray += 1

            # Endpoint A projection residuals (u/v)
            if self.config.use_proj_residuals:
                for cid in sorted(obsA.keys()):
                    for vi in cam_var_idx.get(cid, []):
                        J[row_proj, vi] = 1
                        J[row_proj + 1, vi] = 1
                    wid = self.cam_to_window.get(cid)
                    for vi in plane_var_idx.get(wid, []):
                        J[row_proj, vi] = 1
                        J[row_proj + 1, vi] = 1
                    for vi in ptA_var_idx.get(fid, []):
                        J[row_proj, vi] = 1
                        J[row_proj + 1, vi] = 1
                    row_proj += 2

            # Endpoint B ray residuals
            for cid in sorted(obsB.keys()):
                for vi in cam_var_idx.get(cid, []):
                    J[row_ray, vi] = 1
                wid = self.cam_to_window.get(cid)
                for vi in plane_var_idx.get(wid, []):
                    J[row_ray, vi] = 1
                for vi in ptB_var_idx.get(fid, []):
                    J[row_ray, vi] = 1
                row_ray += 1

            # Endpoint B projection residuals (u/v)
            if self.config.use_proj_residuals:
                for cid in sorted(obsB.keys()):
                    for vi in cam_var_idx.get(cid, []):
                        J[row_proj, vi] = 1
                        J[row_proj + 1, vi] = 1
                    wid = self.cam_to_window.get(cid)
                    for vi in plane_var_idx.get(wid, []):
                        J[row_proj, vi] = 1
                        J[row_proj + 1, vi] = 1
                    for vi in ptB_var_idx.get(fid, []):
                        J[row_proj, vi] = 1
                        J[row_proj + 1, vi] = 1
                    row_proj += 2

            # Wand length residual
            for vi in ptA_var_idx.get(fid, []):
                J[row_len, vi] = 1
            for vi in ptB_var_idx.get(fid, []):
                J[row_len, vi] = 1
            row_len += 1

            # Barrier residuals (2 per window per endpoint)
            if obsA:
                widsA = sorted({self.cam_to_window.get(cid) for cid in obsA.keys() if self.cam_to_window.get(cid) not in (None, -1)})
                for wid in widsA:
                    for vi in plane_var_idx.get(wid, []):
                        J[row_bar, vi] = 1
                        J[row_bar + 1, vi] = 1
                    for vi in ptA_var_idx.get(fid, []):
                        J[row_bar, vi] = 1
                        J[row_bar + 1, vi] = 1
                    row_bar += 2

            if obsB:
                widsB = sorted({self.cam_to_window.get(cid) for cid in obsB.keys() if self.cam_to_window.get(cid) not in (None, -1)})
                for wid in widsB:
                    for vi in plane_var_idx.get(wid, []):
                        J[row_bar, vi] = 1
                        J[row_bar + 1, vi] = 1
                    for vi in ptB_var_idx.get(fid, []):
                        J[row_bar, vi] = 1
                        J[row_bar + 1, vi] = 1
                    row_bar += 2

        return J.tocsr()

    def _unpack_bundle_params(self, x: np.ndarray, base_layout: List[Tuple], full_layout: List[Tuple]) -> Tuple[Dict, Dict, Dict, Dict[int, Dict[str, np.ndarray]]]:
        x_base = x[:len(base_layout)]
        planes, cams, media = self._unpack_params_delta(x_base, base_layout)

        points = {fid: {'A': self._bundle_points_ref[fid]['A'].copy(), 'B': self._bundle_points_ref[fid]['B'].copy()} for fid in self.fids_optim}
        idx = len(base_layout)
        for (ptype, fid, subidx) in full_layout[len(base_layout):]:
            if ptype == 'ptA':
                points[fid]['A'][subidx] += x[idx]
            elif ptype == 'ptB':
                points[fid]['B'][subidx] += x[idx]
            idx += 1
        return planes, cams, media, points

    def _residuals_bundle(self, x: np.ndarray, base_layout: List[Tuple], full_layout: List[Tuple], mode: str, lambda_eff: float) -> np.ndarray:
        planes, cams, media, points = self._unpack_bundle_params(x, base_layout, full_layout)
        residuals, S_ray, S_len, N_ray, N_len, S_proj, N_proj = self.evaluate_residuals(
            planes, cams, lambda_eff, window_media=media, explicit_points=points
        )
        self._last_ray_rmse = np.sqrt(S_ray / max(N_ray, 1))
        self._last_len_rmse = np.sqrt(S_len / max(N_len, 1)) if N_len > 0 else 0.0
        self._last_proj_rmse = np.sqrt(S_proj / max(N_proj, 1)) if N_proj > 0 else 0.0
        res_arr = np.asarray(residuals, dtype=np.float64)
        if not np.all(np.isfinite(res_arr)):
            res_arr = np.nan_to_num(res_arr, nan=1e6, posinf=1e6, neginf=-1e6)
        return res_arr

    def _optimize_bundle_generic(self, mode: str, description: str,
                                 enable_planes: bool, enable_cam_t: bool, enable_cam_r: bool,
                                 limit_rot_rad: float, limit_trans_mm: float,
                                 limit_plane_d_mm: float, limit_plane_angle_rad: float,
                                 enable_cam_f: bool = False, enable_win_t: bool = False,
                                 enable_cam_k1: bool = False, enable_cam_k2: bool = False,
                                 plane_d_bounds: Dict[int, object] = None,
                                 ftol: float = 1e-6,
                                 xtol: float = 1e-6,
                                 gtol: float = 1e-6,
                                 loss: Optional[str] = None,
                                 f_scale: Optional[float] = None,
                                 max_nfev: int = 50,
                                 strategy_override: Optional[str] = None,
                                 x0_override: Optional[np.ndarray] = None,
                                 freeze_bounds_reference: bool = False,
                                 bounds_ref_state: Optional[Dict[str, Dict]] = None):
        """Bundle strategy: optimize enabled camera/plane params and explicit 3D points jointly."""
        # Keep loop behavior consistent across strategies; disable xtol only for joint/final bundle rounds.
        xtol_effective = None if mode in ('joint', 'final_refined', 'final') else xtol

        if freeze_bounds_reference and bounds_ref_state is not None:
            self._apply_reference_state(bounds_ref_state)
        self._refresh_plane_round_reference()
        self._compute_physical_sigmas()
        self._set_barrier_profile_for_mode(mode, log=True)

        effective_loss = loss or 'linear'
        if self.config.verbosity >= 1:
            self.reporter.detail(
                f"  [RoundConfig] mode={mode}, strategy=bundle, loss={effective_loss}, "
                f"ftol={ftol:g}, xtol={xtol_effective}, gtol={gtol:g}, max_nfev={max_nfev}"
            )

        base_layout = self._get_param_layout(
            enable_planes, enable_cam_t, enable_cam_r,
            enable_cam_f, enable_win_t, enable_cam_k1, enable_cam_k2
        )
        self._ensure_bundle_points_initialized(self.initial_cam_params, self.initial_planes, self.initial_media)
        self._bundle_points_ref = {fid: {'A': p['A'].copy(), 'B': p['B'].copy()} for fid, p in self._bundle_points.items()}

        full_layout = self._build_bundle_layout(base_layout)
        if not full_layout:
            self.reporter.detail(f"  [{description}] No parameters to optimize.")
            return

        jac_sparsity = self._build_bundle_jac_sparsity(full_layout)
        if self.config.verbosity >= 1:
            self.reporter.detail(f"  bundle jac_sparsity: shape={jac_sparsity.shape}, nnz={jac_sparsity.nnz}")

        x0 = np.zeros(len(full_layout), dtype=np.float64)
        if x0_override is not None and len(x0_override) == len(full_layout):
            x0 = np.asarray(x0_override, dtype=np.float64).copy()
        cfg = self.config

        diff_step_mode = 'manual'
        if mode == 'joint':
            diff_step_mode = str(getattr(self.config, 'diff_step_mode_joint', 'manual')).lower()
        elif mode == 'final_refined' or mode == 'final':
            diff_step_mode = str(getattr(self.config, 'diff_step_mode_final', 'manual')).lower()
        if diff_step_mode not in ('manual', 'auto'):
            diff_step_mode = 'manual'

        diff_step = None
        if diff_step_mode == 'manual':
            vals = []
            for (ptype, _, _) in full_layout:
                if ptype == 'cam_r':
                    vals.append(cfg.diff_step_rvec)
                elif ptype == 'cam_t':
                    vals.append(cfg.diff_step_tvec)
                elif ptype == 'plane_d':
                    vals.append(cfg.diff_step_plane_d)
                elif ptype == 'plane_a' or ptype == 'plane_b':
                    vals.append(cfg.diff_step_plane_ang)
                elif ptype == 'cam_f':
                    vals.append(cfg.diff_step_f)
                elif ptype == 'win_t':
                    vals.append(cfg.diff_step_thick)
                elif ptype == 'cam_k1' or ptype == 'cam_k2':
                    vals.append(cfg.diff_step_k)
                elif ptype == 'ptA' or ptype == 'ptB':
                    vals.append(cfg.diff_step_tvec)
                else:
                    vals.append(cfg.diff_step_default)
            diff_step = np.asarray(vals, dtype=np.float64)

        n_cams = max(1, len(self.cam_params))
        lambda_fixed = self._compute_lambda_fixed(mode)

        planes0, cams0, media0, points0 = self._unpack_bundle_params(x0, base_layout, full_layout)
        _, S_ray0, S_len0, N_ray, N_len, S_proj0, N_proj0 = self.evaluate_residuals(
            planes0, cams0, lambda_fixed, window_media=media0, explicit_points=points0
        )
        rmse_ray0 = np.sqrt(S_ray0 / max(N_ray, 1))
        rmse_len0 = np.sqrt(S_len0 / max(N_len, 1)) if N_len > 0 else 0.0
        rmse_proj0 = np.sqrt(S_proj0 / max(N_proj0, 1)) if N_proj0 > 0 else 0.0
        J0 = (rmse_ray0**2) + lambda_fixed * (rmse_len0**2)
        self._j_ref = J0 if J0 > 1e-6 else 1.0
        self.reporter.detail(f"    Global Fixed Weighting: lambda={lambda_fixed:.1f} (N_cams_total={n_cams})")
        if self.config.use_proj_residuals:
            self.reporter.detail(
                f"    Initial: S_ray={S_ray0:.2f}, S_len={S_len0:.2f}, S_proj={S_proj0:.2f} (J0={J0:.4f})"
            )
        else:
            self.reporter.detail(f"    Initial: S_ray={S_ray0:.2f}, S_len={S_len0:.2f} (J0={J0:.4f})")

        lb, ub = [], []
        for (ptype, pid, subidx) in full_layout:
            if ptype == 'plane_d':
                if plane_d_bounds and pid in plane_d_bounds:
                    b = plane_d_bounds[pid]
                    if isinstance(b, (tuple, list)) and len(b) == 2:
                        blo = float(b[0])
                        bhi = float(b[1])
                        if bhi < blo:
                            blo, bhi = bhi, blo
                        lb.append(blo); ub.append(bhi)
                    else:
                        limit = float(abs(b))
                        lb.append(-limit); ub.append(limit)
                else:
                    limit = float(limit_plane_d_mm)
                    lb.append(-limit); ub.append(limit)
            elif ptype == 'plane_a' or ptype == 'plane_b':
                lb.append(-limit_plane_angle_rad); ub.append(limit_plane_angle_rad)
            elif ptype == 'win_t':
                t0 = self.initial_media.get(pid, {}).get('thickness', 0.0)
                limit = abs(t0) * self.config.bounds_thick_pct
                lb.append(-limit); ub.append(limit)
            elif ptype == 'cam_r':
                lb.append(-limit_rot_rad); ub.append(limit_rot_rad)
            elif ptype == 'cam_t':
                lb.append(-limit_trans_mm); ub.append(limit_trans_mm)
            elif ptype == 'cam_f':
                f0 = self.initial_f.get(pid, self.cam_params.get(pid, [0, 0, 0, 0, 0, 0, 0])[6] if pid in self.cam_params else 0.0)
                limit = abs(f0) * self.config.bounds_f_pct
                lb.append(-limit); ub.append(limit)
            elif ptype == 'cam_k1' or ptype == 'cam_k2':
                limit = self.config.bounds_dist_abs
                lb.append(-limit); ub.append(limit)
            elif ptype == 'ptA' or ptype == 'ptB':
                dlim_cfg = self.config.bundle_point_delta_mm
                if dlim_cfg is None:
                    lb.append(-np.inf); ub.append(np.inf)
                else:
                    dlim = float(dlim_cfg)
                    lb.append(-dlim); ub.append(dlim)
            else:
                lb.append(-1.0); ub.append(1.0)
        bounds = (np.asarray(lb), np.asarray(ub))

        self._res_call_count = 0
        def residuals_wrapper(x, *args, **kwargs):
            res = self._residuals_bundle(x, *args, **kwargs)
            self._res_call_count += 1
            if self.progress_callback and self._res_call_count % 10 == 0:
                c_approx = 0.5 * np.sum(res**2)
                self.progress_callback(
                    f"{description}", self._last_ray_rmse, self._last_len_rmse, self._last_proj_rmse, c_approx
                )
            return res

        res = least_squares(
            residuals_wrapper,
            x0,
            args=(base_layout, full_layout, mode, lambda_fixed),
            method='trf',
            bounds=bounds,
            verbose=0,
            x_scale='jac',
            jac_sparsity=jac_sparsity,
            ftol=ftol,
            xtol=xtol_effective,
            gtol=gtol,
            loss=effective_loss,
            f_scale=self.config.loss_f_scale if f_scale is None else f_scale,
            max_nfev=max_nfev,
            diff_step=diff_step,
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
            if status == -1:
                raise RuntimeError(
                    f"[BA bundle] least_squares failed (status=-1): {message}"
                )
            elif status == 0:
                self.reporter.warning(
                    f"[BA bundle] least_squares budget exhausted (status=0): {message}. Using partial result."
                )

        planes_final, cams_final, media_final, points_final = self._unpack_bundle_params(res.x, base_layout, full_layout)
        _, S_rayF, S_lenF, _, _, S_projF, N_projF = self.evaluate_residuals(
            planes_final, cams_final, lambda_fixed, window_media=media_final, explicit_points=points_final
        )
        rmse_rayF = np.sqrt(S_rayF / max(N_ray, 1))
        rmse_lenF = np.sqrt(S_lenF / max(N_len, 1)) if N_len > 0 else 0.0
        rmse_projF = np.sqrt(S_projF / max(N_projF, 1)) if N_projF > 0 else 0.0
        if self.config.use_proj_residuals:
            self.reporter.detail(f"    Final:   S_ray={S_rayF:.2f}, S_len={S_lenF:.2f}, S_proj={S_projF:.2f}")
        else:
            self.reporter.detail(f"    Final:   S_ray={S_rayF:.2f}, S_len={S_lenF:.2f}")
        self.reporter.detail(f"      RMSE Ray: {rmse_ray0:.4f} -> {rmse_rayF:.4f}")
        self.reporter.detail(f"      RMSE Len: {rmse_len0:.4f} -> {rmse_lenF:.4f}")
        if self.config.use_proj_residuals:
            self.reporter.detail(f"      RMSE Proj: {rmse_proj0:.4f} -> {rmse_projF:.4f} px")

        self.initial_planes = planes_final
        self.initial_cam_params = cams_final
        self.initial_media = media_final
        self.window_planes = planes_final
        self.cam_params = cams_final
        self.window_media = media_final
        self._bundle_points = {fid: {'A': p['A'].copy(), 'B': p['B'].copy()} for fid, p in points_final.items()}
        self._bundle_points_ref = {fid: {'A': p['A'].copy(), 'B': p['B'].copy()} for fid, p in points_final.items()}

        self._refresh_plane_round_reference()
        self.sync_cpp_state(cam_params=self.cam_params, window_planes=self.window_planes, window_media=self.window_media)
        return res, full_layout

    def _optimize_generic(self, mode: str, description: str, 
                          enable_planes: bool, enable_cam_t: bool, enable_cam_r: bool,
                          limit_rot_rad: float, limit_trans_mm: float, 
                          limit_plane_d_mm: float, limit_plane_angle_rad: float,
                          enable_cam_f: bool = False, enable_win_t: bool = False,
                          enable_cam_k1: bool = False, enable_cam_k2: bool = False,
                          plane_d_bounds: Dict[int, object] = None,
                          ftol: float = 1e-6,
                          xtol: float = 1e-6,
                          gtol: float = 1e-6,
                          loss: Optional[str] = None,
                          f_scale: Optional[float] = None,
                          max_nfev: int = 50,
                          strategy_override: Optional[str] = None,
                          x0_override: Optional[np.ndarray] = None,
                          freeze_bounds_reference: bool = False,
                          bounds_ref_state: Optional[Dict[str, Dict]] = None):
        """
        Generic optimization loop with explicit bounds and parameter selection.
        """
        strategy = self._resolve_strategy(mode, strategy_override=strategy_override)

        if strategy == 'bundle':
            return self._optimize_bundle_generic(
                mode=mode,
                description=description,
                enable_planes=enable_planes,
                enable_cam_t=enable_cam_t,
                enable_cam_r=enable_cam_r,
                limit_rot_rad=limit_rot_rad,
                limit_trans_mm=limit_trans_mm,
                limit_plane_d_mm=limit_plane_d_mm,
                limit_plane_angle_rad=limit_plane_angle_rad,
                enable_cam_f=enable_cam_f,
                enable_win_t=enable_win_t,
                enable_cam_k1=enable_cam_k1,
                enable_cam_k2=enable_cam_k2,
                plane_d_bounds=plane_d_bounds,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                loss=loss,
                f_scale=f_scale,
                max_nfev=max_nfev,
                strategy_override=strategy_override,
                x0_override=x0_override,
                freeze_bounds_reference=freeze_bounds_reference,
                bounds_ref_state=bounds_ref_state,
            )

        if freeze_bounds_reference and bounds_ref_state is not None:
            self._apply_reference_state(bounds_ref_state)

        # Recompute round-wise plane anchors/d0 from current relinearization state.
        self._refresh_plane_round_reference()
        # Recompute physical sigmas for current round state.
        self._compute_physical_sigmas()
        self._set_barrier_profile_for_mode(mode, log=True)

        effective_loss = loss or 'linear'

        diff_step_mode = 'manual'
        if mode == 'joint':
            diff_step_mode = str(getattr(self.config, 'diff_step_mode_joint', 'manual')).lower()
        elif mode == 'final_refined' or mode == 'final':
            diff_step_mode = str(getattr(self.config, 'diff_step_mode_final', 'manual')).lower()

        if diff_step_mode not in ('manual', 'auto'):
            self.reporter.detail(f"  [RoundConfig] invalid diff_step_mode={diff_step_mode}, fallback to manual")
            diff_step_mode = 'manual'

        if self.config.verbosity >= 1:
            self.reporter.detail(
                f"  [RoundConfig] mode={mode}, strategy=sequence, loss={effective_loss}, "
                f"ftol={ftol:g}, xtol={xtol:g}, gtol={gtol:g}, max_nfev={max_nfev}, "
                f"diff_step_mode={diff_step_mode}"
            )

        layout = self._get_param_layout(
            enable_planes, enable_cam_t, enable_cam_r,
            enable_cam_f, enable_win_t, enable_cam_k1, enable_cam_k2
        )
        
        if not layout:
            print(f"  [{description}] No parameters to optimize.")
            return

        x0 = np.zeros(len(layout), dtype=np.float64)
        if x0_override is not None and len(x0_override) == len(layout):
            x0 = np.asarray(x0_override, dtype=np.float64).copy()
        cfg = self.config

        diff_step = None
        if diff_step_mode == 'manual':
            diff_step_vals = []
            for (ptype, _, _) in layout:
                if ptype == 'cam_r':
                    diff_step_vals.append(cfg.diff_step_rvec)
                elif ptype == 'cam_t':
                    diff_step_vals.append(cfg.diff_step_tvec)
                elif ptype == 'plane_d':
                    diff_step_vals.append(cfg.diff_step_plane_d)
                elif ptype == 'plane_a' or ptype == 'plane_b':
                    diff_step_vals.append(cfg.diff_step_plane_ang)
                elif ptype == 'cam_f':
                    diff_step_vals.append(cfg.diff_step_f)
                elif ptype == 'win_t':
                    diff_step_vals.append(cfg.diff_step_thick)
                elif ptype == 'cam_k1' or ptype == 'cam_k2':
                    diff_step_vals.append(cfg.diff_step_k)
                else:
                    diff_step_vals.append(cfg.diff_step_default)

            diff_step = np.array(diff_step_vals, dtype=np.float64)
            if cfg.verbosity >= 1:
                self.reporter.detail(
                    f"  diff_step: rvec={cfg.diff_step_rvec:g}, tvec={cfg.diff_step_tvec:g}, "
                    f"plane_d={cfg.diff_step_plane_d:g}, plane_ang={cfg.diff_step_plane_ang:g}, "
                    f"f={cfg.diff_step_f:g}, thick={cfg.diff_step_thick:g}, k={cfg.diff_step_k:g}"
                )
        elif cfg.verbosity >= 1:
            self.reporter.detail("  diff_step: auto (scipy default)")
        
        self.reporter.detail(f"  [{description}] optimizing {len(x0)} parameters ({len(layout)//3} blocks)...")
        # Calc initial RMSE for rollback reference
        planes0, cams0, media0 = self._unpack_params_delta(x0, layout)
        
        # [USER REQUEST] Fixed Weighting Strategy
        # Lambda base = lambda_base_per_cam * N_total_cams, optional per-round scaling.
        n_cams = max(1, len(self.cam_params))
        lambda_fixed = self._compute_lambda_fixed(mode)
        
        # Initial evaluation
        _, S_ray0, S_len0, N_ray, N_len, S_proj0, N_proj0 = self.evaluate_residuals(
            planes0, cams0, lambda_fixed, window_media=media0
        )

        rmse_ray0 = np.sqrt(S_ray0 / max(N_ray, 1))
        rmse_len0 = np.sqrt(S_len0 / max(N_len, 1)) if N_len > 0 else 0.0
        rmse_proj0 = np.sqrt(S_proj0 / max(N_proj0, 1)) if N_proj0 > 0 else 0.0
        
        # Initial normalized cost J0
        J0 = (rmse_ray0**2) + lambda_fixed * (rmse_len0**2)
        self._j_ref = J0 if J0 > 1e-6 else 1.0
        
        self.reporter.detail(f"    Global Fixed Weighting: lambda={lambda_fixed:.1f} (N_cams_total={n_cams})")
        if self.config.use_proj_residuals:
            self.reporter.detail(
                f"    Initial: S_ray={S_ray0:.2f}, S_len={S_len0:.2f}, S_proj={S_proj0:.2f} (J0={J0:.4f})"
            )
        else:
            self.reporter.detail(f"    Initial: S_ray={S_ray0:.2f}, S_len={S_len0:.2f} (J0={J0:.4f})")
        
        # Build Bounds
        lb = []
        ub = []
        for (ptype, pid, subidx) in layout:
            if ptype == 'plane_d':
                if plane_d_bounds and pid in plane_d_bounds:
                    b = plane_d_bounds[pid]
                    if isinstance(b, (tuple, list)) and len(b) == 2:
                        blo = float(b[0])
                        bhi = float(b[1])
                        if bhi < blo:
                            blo, bhi = bhi, blo
                        lb.append(blo)
                        ub.append(bhi)
                    else:
                        limit = float(abs(b))
                        lb.append(-limit)
                        ub.append(limit)
                else:
                    limit = float(limit_plane_d_mm)
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
            if self.progress_callback and self._res_call_count % 10 == 0:
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
                            f"{description}", self._last_ray_rmse, self._last_len_rmse, self._last_proj_rmse, c_approx
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
            loss=effective_loss,
            f_scale=self.config.loss_f_scale if f_scale is None else f_scale,
            xtol=xtol,
            gtol=gtol,
            max_nfev=max_nfev,
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
            if status == -1:
                raise RuntimeError(
                    f"[BA sequence] least_squares failed (status=-1): {message}"
                )
            elif status == 0:
                self.reporter.warning(
                    f"[BA sequence] least_squares budget exhausted (status=0): {message}. Using partial result."
                )

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
                    self.reporter.detail("  Camera delta:")
                    for cid in sorted(cam_deltas.keys()):
                        cd = cam_deltas[cid]
                        parts = []
                        if 'r' in cd:
                            rot_deg = np.linalg.norm(cd['r']) * 180.0 / np.pi
                            parts.append(f"dr={rot_deg:.3f}deg")
                        if 't' in cd:
                            trans = np.linalg.norm(cd['t'])
                            parts.append(f"dt={trans:.3f}mm")
                        if 'f' in cd:
                            parts.append(f"df={cd['f']:.4f}")
                        if 'k1' in cd:
                            parts.append(f"dk1={cd['k1']:.6f}")
                        if 'k2' in cd:
                            parts.append(f"dk2={cd['k2']:.6f}")
                        if parts:
                            self.reporter.detail(f"    Cam {cid}: " + ", ".join(parts))

                if plane_deltas:
                    self.reporter.detail("  Window delta:")
                    for wid in sorted(plane_deltas.keys()):
                        pd = plane_deltas[wid]
                        parts = []
                        if 'd' in pd:
                            parts.append(f"dd={pd['d']:.4f}mm")
                        if 'a' in pd:
                            parts.append(f"da={pd['a'] * 180.0 / np.pi:.4f}deg")
                        if 'b' in pd:
                            parts.append(f"db={pd['b'] * 180.0 / np.pi:.4f}deg")
                        if 't' in pd:
                            parts.append(f"dt={pd['t']:.4f}mm")
                        if parts:
                            self.reporter.detail(f"    Win {wid}: " + ", ".join(parts))

        # Print Barrier Stats
        if cfg.verbosity >= 1 and hasattr(self, '_last_barrier_stats') and self._last_barrier_stats:
            s = self._last_barrier_stats
            self.reporter.detail(f"    [RefractiveCalib][SIDE-BARRIER] min(sX)={s['min_sX']:.4f}mm, near(<20um)={s['pct_near']:.1f}%, vio={s.get('violations', 0)}, cost/J={s['ratio']:.1e}")

        # Final evaluation
        planes_final, cams_final, media_final = self._unpack_params_delta(res.x, layout)
        _, S_rayF, S_lenF, _, _, S_projF, N_projF = self.evaluate_residuals(
            planes_final, cams_final, lambda_fixed, window_media=media_final
        )

        rmse_rayF = np.sqrt(S_rayF / max(N_ray, 1))
        rmse_lenF = np.sqrt(S_lenF / max(N_len, 1)) if N_len > 0 else 0.0
        rmse_projF = np.sqrt(S_projF / max(N_projF, 1)) if N_projF > 0 else 0.0
        JF = (rmse_rayF**2) + lambda_fixed * (rmse_lenF**2)

        if self.config.use_proj_residuals:
            self.reporter.detail(f"    Final:   S_ray={S_rayF:.2f}, S_len={S_lenF:.2f}, S_proj={S_projF:.2f} (JF={JF:.4f})")
        else:
            self.reporter.detail(f"    Final:   S_ray={S_rayF:.2f}, S_len={S_lenF:.2f} (JF={JF:.4f})")
        self.reporter.detail(f"      RMSE Ray: {rmse_ray0:.4f} -> {rmse_rayF:.4f}")
        self.reporter.detail(f"      RMSE Len: {rmse_len0:.4f} -> {rmse_lenF:.4f}")
        if self.config.use_proj_residuals:
            self.reporter.detail(f"      RMSE Proj: {rmse_proj0:.4f} -> {rmse_projF:.4f} px")
        
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

        # Refresh references for next round and explicitly sync to C++ state.
        self._refresh_plane_round_reference()
        self.sync_cpp_state(cam_params=self.cam_params, window_planes=self.window_planes, window_media=self.window_media)
        
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

        selected_fids = fids[::step]

        # Pre-batch rays from OTHER cameras for both endpoints on selected frames.
        per_cam_items_A: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        per_cam_items_B: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for fid in selected_fids:
            obs = self.obs_cache[fid]
            if cid not in obs:
                continue

            for other_cid, (o_uvA, o_uvB) in obs.items():
                if other_cid == cid:
                    continue
                if o_uvA is not None:
                    per_cam_items_A.setdefault(other_cid, []).append((fid, o_uvA))
                if o_uvB is not None:
                    per_cam_items_B.setdefault(other_cid, []).append((fid, o_uvB))

        ray_lookup_A = self._build_batched_ray_lookup(per_cam_items_A, endpoint="A")
        ray_lookup_B = self._build_batched_ray_lookup(per_cam_items_B, endpoint="B")
        
        for fid in selected_fids:
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
                        lookup = ray_lookup_A if endpoint == 'A' else ray_lookup_B
                        r = lookup.get((fid, other_cid))
                        if r is None:
                            r = self._make_invalid_ray(
                                cam_id=other_cid,
                                frame_id=fid,
                                endpoint=endpoint,
                                reason="missing_ray",
                                uv=val,
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

    def _collect_points_for_alignment(self) -> List[np.ndarray]:
        """Collect triangulated points from obsA/obsB for coordinate alignment."""
        points_3d: List[np.ndarray] = []
        obsA = self.dataset.get('obsA', {})
        obsB = self.dataset.get('obsB', {})

        def _tri_all(obs_dict: Dict[int, Dict[int, np.ndarray]]) -> List[np.ndarray]:
            pts: List[np.ndarray] = []
            rays_by_fid: Dict[int, List[Ray]] = {fid: [] for fid in obs_dict.keys()}

            per_cam: Dict[int, List[Tuple[int, np.ndarray]]] = {}
            for fid, cam_obs in obs_dict.items():
                for cid, uv in cam_obs.items():
                    if cid not in self.cams_cpp or cid not in self.cam_params:
                        continue
                    per_cam.setdefault(cid, []).append((fid, uv))

            for cid, items in per_cam.items():
                wid = self.cam_to_window.get(cid)
                uv_list = [uv for _, uv in items]
                meta_list = [
                    {
                        "cam_id": cid,
                        "window_id": wid,
                        "frame_id": fid,
                        "endpoint": "?",
                    }
                    for fid, _ in items
                ]
                rays = build_pinplate_rays_cpp_batch(self.cams_cpp[cid], uv_list, meta_list=meta_list)
                for (fid, _), ray in zip(items, rays):
                    if ray.valid:
                        rays_by_fid[fid].append(ray)

            for fid in obs_dict.keys():
                rays = rays_by_fid[fid]
                if len(rays) >= 2:
                    X, _, valid, _ = triangulate_point(rays)
                    if valid:
                        pts.append(X)
            return pts

        points_3d.extend(_tri_all(obsA))
        points_3d.extend(_tri_all(obsB))
        return points_3d

    def _build_step_a_plane_d_bounds(self, loop_iter: int) -> Dict[int, Tuple[float, float]]:
        """Recompute per-window plane_d bounds using current cameras and triangulated 3D points.

        Bounds enforce that plane point update `pt = pt0 + d * n0` remains between camera
        and closest 3D point, then weak-window tightening is intersected on top.
        """
        pts = self._collect_points_for_alignment()
        if not pts:
            self.reporter.detail(f"  [LOOP {loop_iter}] No triangulated points for geometric d-bounds; fallback to global bounds")
            return {}

        pts_arr = np.asarray(pts, dtype=np.float64)
        eps = 0.05
        factor = 0.1 * (0.5 ** (loop_iter - 1))
        out: Dict[int, Tuple[float, float]] = {}

        self.reporter.detail(f"  [LOOP {loop_iter}] Recomputing plane_d bounds from current cameras/points")
        for wid in self.window_ids:
            pl0 = self.initial_planes.get(wid, self.window_planes.get(wid))
            if pl0 is None:
                continue
            pt0 = np.asarray(pl0['plane_pt'], dtype=np.float64)
            n0 = np.asarray(pl0['plane_n'], dtype=np.float64)
            nn = np.linalg.norm(n0)
            if nn < 1e-12:
                continue
            n0 = n0 / nn

            cams = [cid for cid in self.window_to_cams.get(wid, []) if cid in self.active_cam_ids and cid in self.cam_params]
            if not cams:
                continue

            lo = -np.inf
            hi = np.inf
            cam_terms = []
            min_d_ref = None
            for cid in cams:
                p = self.cam_params[cid]
                R, _ = cv2.Rodrigues(p[0:3])
                C = camera_center(R, p[3:6])
                dists = np.linalg.norm(pts_arr - C.reshape(1, 3), axis=1)
                if dists.size == 0:
                    continue
                idx = int(np.argmin(dists))
                X_min = pts_arr[idx]
                d_min = float(dists[idx])
                if min_d_ref is None or d_min < min_d_ref:
                    min_d_ref = d_min

                s_c0 = float(np.dot(n0, C - pt0))
                s_x0 = float(np.dot(n0, X_min - pt0))
                lo_i = min(s_c0, s_x0) + eps
                hi_i = max(s_c0, s_x0) - eps
                if hi_i <= lo_i:
                    continue
                lo = max(lo, lo_i)
                hi = min(hi, hi_i)
                cam_terms.append((cid, lo_i, hi_i, d_min))

            if not cam_terms:
                continue

            geo_lo, geo_hi = lo, hi
            weak_ref = None
            weak_lo, weak_hi = None, None
            if hasattr(self, '_weak_windows') and wid in self._weak_windows:
                weak_ref = float(self._weak_windows[wid].get('d_min_ref', 0.0))
                if weak_ref <= 1e-9 and min_d_ref is not None:
                    weak_ref = float(min_d_ref)
                if weak_ref > 1e-9:
                    tight = weak_ref * factor
                    weak_lo, weak_hi = -tight, tight
                    lo = max(lo, weak_lo)
                    hi = min(hi, weak_hi)

            if hi <= lo:
                mid = 0.5 * (lo + hi)
                lo = mid - 1e-3
                hi = mid + 1e-3

            # Delta-parameterization safety: plane_d bounds are for delta and must include x0=0.
            if not (lo <= 0.0 <= hi):
                raw_lo, raw_hi = lo, hi
                lo = min(lo, -1e-6)
                hi = max(hi, 1e-6)
                self.reporter.detail(
                    f"    [d-bound-fix] Win {wid}: raw [{raw_lo:.3f}, {raw_hi:.3f}] excluded 0; "
                    f"adjusted to [{lo:.3f}, {hi:.3f}] for delta x0=0"
                )

            out[int(wid)] = (float(lo), float(hi))
            cam_msg = ", ".join([f"cam{cid}:[{c_lo:.2f},{c_hi:.2f}] dmin={dmin:.1f}" for cid, c_lo, c_hi, dmin in cam_terms])
            if weak_ref is not None and weak_ref > 1e-9:
                if weak_lo is not None and weak_hi is not None:
                    self.reporter.detail(
                        f"    [d-bound] Win {wid} WEAK -> [{lo:.3f}, {hi:.3f}] mm = "
                        f"intersect(geo:[{geo_lo:.3f},{geo_hi:.3f}], weak:[{weak_lo:.3f},{weak_hi:.3f}]); {cam_msg}"
                    )
                else:
                    self.reporter.detail(
                        f"    [d-bound] Win {wid} WEAK -> [{lo:.3f}, {hi:.3f}] mm; weak_ref={weak_ref:.2f}, factor={factor:.4f}; {cam_msg}"
                    )
            else:
                self.reporter.detail(f"    [d-bound] Win {wid} STRONG -> [{lo:.3f}, {hi:.3f}] mm; {cam_msg}")

        return out

    def _apply_coordinate_alignment(self, tag: str, refresh_initial: bool = True, align_mode: str = 'yz') -> bool:
        """Apply world-frame alignment and sync aligned parameters to C++ state."""
        if len(self.window_planes) < 1:
            if self.config.verbosity >= 1:
                self.reporter.detail(f"[Coordinate Alignment] Skip ({tag}): no window planes")
            return False

        points_3d = self._collect_points_for_alignment()
        if self.config.verbosity >= 1:
            self.reporter.detail(f"[Coordinate Alignment] {tag}: collected {len(points_3d)} points")

        try:
            new_cam_params, new_window_planes, _, R_align, t_shift = align_world_y_to_plane_intersection(
                self.window_planes, self.cam_params, points_3d=points_3d, align_mode=align_mode
            )
        except Exception as e:
            self.reporter.detail(f"[Coordinate Alignment] {tag}: failed ({e})")
            return False

        self.cam_params = {int(cid): np.asarray(p, dtype=np.float64) for cid, p in new_cam_params.items()}

        converted_planes: Dict[int, Dict] = {}
        for wid, pl in new_window_planes.items():
            pl_new = dict(pl)
            pl_new['plane_pt'] = np.asarray(pl['plane_pt'], dtype=np.float64)
            pl_new['plane_n'] = np.asarray(pl['plane_n'], dtype=np.float64)
            converted_planes[int(wid)] = pl_new
        self.window_planes = converted_planes

        # Keep bundle explicit points in the same coordinate frame.
        self._transform_bundle_points(R_align, t_shift)

        self.sync_cpp_state(cam_params=self.cam_params, window_planes=self.window_planes, window_media=self.window_media)

        if refresh_initial:
            self._sync_initial_state()
            self._compute_physical_sigmas()

        if self.config.verbosity >= 1:
            self.reporter.detail(f"[Coordinate Alignment] {tag}: applied and synced (mode={align_mode})")
        return True

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
        if not faulthandler.is_enabled():
            try:
                faulthandler.enable(all_threads=True)
            except Exception as e:
                self._diag_log(f"[DIAG] faulthandler enable failed: {e}")

        self._compute_physical_sigmas()
        if stage is None:
            stage = self.config.stage
        
        # [MOVED per user request] Weak window detection now inside loop.
        # self._detect_weak_windows()
        
        # [NEW] Persistent store for geometric init state (d_min)
        self._weak_window_refs = {}
        
        enable_ray_tracking(True, reset=True)
        reset_camera_update_stats()
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
        pre_last_loop_align_done = False
        
        self.reporter.section(f"Alternating Loop (Max {max_loop_iters} passes)")
        
        while loop_iter < max_loop_iters:
            loop_iter += 1

            loss_plane = self.config.loss_plane or 'linear'
            loss_cam = self.config.loss_cam or 'linear'
            
            # [NEW] Dynamic Detection per loop
            self._detect_weak_windows()
            
            # [Fix] Sync initial state because _detect_weak_windows might have moved planes (GeoInit)
            # This ensures the optimization starts from the correct new position (delta=0 -> new_pt)
            self._sync_initial_state()
            self._compute_physical_sigmas()
            
            plane_d_bounds = self._build_step_a_plane_d_bounds(loop_iter)
            loop_planes_tol = self.config.tolerance_schedule.get(
                'loop_planes', {'ftol': 5e-4, 'xtol': 1e-5, 'gtol': 1e-5}
            )
            
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
                ftol=loop_planes_tol['ftol'],
                xtol=loop_planes_tol['xtol'],
                gtol=loop_planes_tol['gtol'],
                loss=loss_plane
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

            # Apply coordinate alignment once before the last camera-extrinsic step.
            # Last step means either convergence-triggered final loop or max-iter final loop.
            is_last_loop_for_cam = (not hit_boundary) or (loop_iter == max_loop_iters)
            if (not pre_last_loop_align_done) and is_last_loop_for_cam:
                self.reporter.section("Coordinate Alignment Before Last Loop Camera Step")
                pre_mode = self._get_retry_alignment_mode()
                self.reporter.detail(f"[Coordinate Alignment] pre-last-loop-cam mode={pre_mode}")
                self._apply_coordinate_alignment(tag="pre-last-loop-cam", refresh_initial=True, align_mode=pre_mode)
                pre_last_loop_align_done = True

            # Step B: Optimize Cameras (Fixed Planes) - Free Bounds
            self.reporter.header(f"Loop {loop_iter} - Step B: Optimize Cameras (Free Bounds)")
            b_cam_free = (np.deg2rad(180.0), 2000.0)
            loop_cams_tol = self.config.tolerance_schedule.get(
                'loop_cams', {'ftol': 5e-4, 'xtol': 1e-5, 'gtol': 1e-5}
            )
            
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
                ftol=loop_cams_tol['ftol'],
                xtol=loop_cams_tol['xtol'],
                gtol=loop_cams_tol['gtol'],
                loss=loss_cam
            )
            self._print_plane_diagnostics(f"Loop {loop_iter} Cams")

            # Check termination
            if not hit_boundary:
                self.reporter.info(f"Converged early at Loop {loop_iter} (Planes inside 2.5 deg). Stopping loop.")
                break
        
        if hit_boundary and loop_iter == max_loop_iters:
             self.reporter.info(f"Loop reached max iterations ({max_loop_iters}). Proceeding to Joint.")

        # --- Joint Optimization (Round 3) ---
        if stage >= 3:
            self.reporter.section("Joint Optimization (Round 3 Rules)")

            # Skip weak-window checks for final joint (ExpB-style).
            # Bounds: 20 deg, 50 mm d, 10 deg plane_ang, 50 mm tvec
            limit_rvec = np.radians(20.0)
            limit_plane_d = 50.0
            limit_plane_ang = np.radians(10.0)
            limit_tvec = 50.0

            print("  Bounds: rvec < 20deg, plane_d < 50mm, plane_ang < 10deg, tvec < 50mm")
            joint_tol = self.config.tolerance_schedule.get(
                'joint', {'ftol': 1e-5, 'xtol': 1e-5, 'gtol': 1e-5}
            )

            joint_kwargs = dict(
                mode='joint',
                description="Optimizing plane and camera extrinsic parameters ...",
                enable_planes=True,
                enable_cam_t=True,
                enable_cam_r=True,
                limit_rot_rad=limit_rvec,
                limit_trans_mm=limit_tvec,
                limit_plane_d_mm=limit_plane_d,
                limit_plane_angle_rad=limit_plane_ang,
                ftol=joint_tol['ftol'],
                xtol=joint_tol['xtol'],
                gtol=joint_tol['gtol'],
                loss=self.config.loss_joint,
                max_nfev=50,
            )
            joint_chunks = self._get_chunk_schedule_for_mode('joint')
            if joint_chunks:
                self._run_round_chunked(
                    'joint',
                    joint_kwargs,
                    joint_chunks,
                    freeze_bounds_reference=True,
                )
            else:
                self._optimize_generic(**joint_kwargs)
            self._print_plane_diagnostics("Joint End")

        # --- Final Refined: Joint + Intrinsics + Thickness ---
        if stage >= 4:
            self.reporter.section("Final Refined: Joint Optimization + Intrinsics/Thickness")
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

            final_refined_tol = self.config.tolerance_schedule.get(
                'final_refined', {'ftol': 1e-5, 'xtol': 1e-5, 'gtol': 1e-5}
            )

            final_kwargs = dict(
                mode='final_refined',
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
                ftol=final_refined_tol['ftol'],
                xtol=final_refined_tol['xtol'],
                gtol=final_refined_tol['gtol'],
                loss=self.config.loss_round4,
                max_nfev=100,
            )
            final_chunks = self._get_chunk_schedule_for_mode('final_refined')
            if final_chunks:
                self._run_round_chunked(
                    'final_refined',
                    final_kwargs,
                    final_chunks,
                    freeze_bounds_reference=True,
                )
            else:
                self._optimize_generic(**final_kwargs)
            self._print_plane_diagnostics("Final Refined End")

            self.reporter.section("Coordinate Alignment Final Export")
            self._apply_coordinate_alignment(tag="final-export", refresh_initial=True)
        
        # Explicit sync call to be safe for returning
        n_cams = max(1, len(self.cam_params))
        lambda_fixed = 2.0 * n_cams
        self._set_barrier_profile_for_mode('final', log=False)
        self.evaluate_residuals(self.window_planes, self.cam_params, lambda_fixed, window_media=self.window_media)

        self.print_diagnostics()
        self.reporter.section("Bundle Adjustment Complete")
        print_ray_stats_report("Bundle")
        print_camera_update_report("Bundle")
        enable_ray_tracking(False)
        
        return self.window_planes, self.cam_params

    
    def print_diagnostics(self):
        """Print comprehensive diagnostics after optimization."""
        self.reporter.section("Final Diagnostics")
        
        # Evaluate final residuals
        n_cams = max(1, len(self.cam_params))
        lambda_fixed = 2.0 * n_cams
        residuals, S_ray, S_len, N_ray, N_len, S_proj, N_proj = self.evaluate_residuals(
            self.window_planes, self.cam_params, lambda_fixed, window_media=self.window_media
        )
        
        # Ray stats
        if N_ray > 0:
            ray_rmse = np.sqrt(S_ray / N_ray)
            print(f"  Ray Distance RMSE: {ray_rmse:.4f} mm ({N_ray} rays)")

        # Projection stats
        if self.config.use_proj_residuals and N_proj > 0:
            proj_rmse = np.sqrt(S_proj / N_proj)
            print(f"  Projection RMSE: {proj_rmse:.4f} px ({N_proj} comps)")
        
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


