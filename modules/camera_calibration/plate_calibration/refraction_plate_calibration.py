import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

try:
    import pyopenlpt as lpt
except ImportError:
    lpt = None

from PySide6.QtCore import QObject, Signal

from modules.camera_calibration.wand_calibration.refractive_geometry import (
    point_to_ray_dist,
    update_cpp_camera_state,
    update_normal_tangent,
    align_world_y_to_plane_intersection,
)


def _summarize_errors(errs):
    """Compute comprehensive error statistics from residual array.
    
    Args:
        errs: List or array of error values
        
    Returns:
        dict with keys: mean, std, median, p90, p95, max, count
        or None if errs is empty
    """
    if not errs:
        return None
    arr = np.asarray(errs, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }


@dataclass
class RefractionPlateConfig:
    method: str = "trf"
    loss: str = "linear"
    max_nfev_joint: int = 50
    max_nfev_final: int = 100
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8

    max_loop_iters: int = 6
    loop_ftol: float = 1e-6
    loop_xtol: float = 1e-6
    loop_gtol: float = 1e-6

    sigma_proj_px: float = 1.0
    penalty_proj_px: float = 50.0

    limit_rot_rad: float = float(np.deg2rad(20.0))
    limit_trans_mm: float = 50.0
    limit_plane_d_mm: float = 50.0
    limit_plane_angle_rad: float = float(np.deg2rad(10.0))
    # Match wand final-refined intrinsic/distortion bounds.
    limit_focal_rel: float = 0.05
    limit_k1: float = 0.2
    limit_k2: float = 0.2

    barrier_enabled: bool = True
    margin_side_mm: float = 0.05
    alpha_side_gate: float = 10.0
    beta_side_dir: float = 1e4
    tau: float = 0.02

    optimize_focal_stage_b: bool = True
    optimize_k1_stage_b: bool = False
    optimize_k2_stage_b: bool = False

    verbosity: int = 1


def _camera_center(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return (-R.T @ tvec.reshape(3, 1)).reshape(3)


class RefractionPlateCalibrator:
    def __init__(
        self,
        cfg: RefractionPlateConfig,
        observations: List[Dict[str, Any]],
        cam_intrinsics: Dict[int, Dict[str, float]],
        cam_to_window: Dict[int, int],
        window_media: Dict[int, Dict[str, float]],
        dist_mode: int = 0,
        progress_callback: Optional[Callable[[str, float, float], None]] = None,
    ):
        if lpt is None:
            raise RuntimeError("pyopenlpt is required for refraction plate calibration")

        self.cfg = cfg
        self.observations = observations
        self.cam_intrinsics = cam_intrinsics
        self.cam_to_window = cam_to_window
        self.window_media = window_media
        self.dist_mode = int(dist_mode)
        self.progress_callback = progress_callback

        self.cam_ids = sorted(cam_intrinsics.keys())
        self.window_ids = sorted({cam_to_window[c] for c in self.cam_ids if c in cam_to_window})

        self.cam_ref: Dict[int, np.ndarray] = {}
        self.cam_cur: Dict[int, np.ndarray] = {}
        self.win_ref: Dict[int, Dict[str, np.ndarray]] = {}
        self.win_cur: Dict[int, Dict[str, np.ndarray]] = {}

        self.cams_cpp: Dict[int, Any] = {cid: lpt.Camera() for cid in self.cam_ids}

        self._eval_count = 0
        self._last_proj_rmse = 0.0
        self._last_cost = 0.0

    @staticmethod
    def _stop_reason_text(status: int, message: str, nfev: int, max_nfev: int) -> str:
        if int(status) == 0:
            return f"max_nfev reached ({nfev}/{max_nfev}): {message}"
        if int(status) == 1:
            return f"gtol reached: {message}"
        if int(status) == 2:
            return f"ftol reached: {message}"
        if int(status) == 3:
            return f"xtol reached: {message}"
        if int(status) == 4:
            return f"ftol/xtol reached: {message}"
        if int(status) < 0:
            return f"solver reported failure (status={status}): {message}"
        return f"status={status}: {message}"

    # -------------------------------
    # Initialization
    # -------------------------------
    def _init_pinhole_per_camera(self):
        for cid in self.cam_ids:
            obj_pts = []
            img_pts = []
            for obs in self.observations:
                uv = obs["uv_by_cam"].get(cid)
                if uv is None:
                    continue
                obj_pts.append(np.asarray(obs["X_world"], dtype=np.float64))
                img_pts.append(np.asarray(uv, dtype=np.float64))

            if len(obj_pts) < 6:
                raise RuntimeError(f"Camera {cid}: insufficient points for pinhole init ({len(obj_pts)})")

            obj_pts_np = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 1, 3)
            img_pts_np = np.asarray(img_pts, dtype=np.float32).reshape(-1, 1, 2)

            intr = self.cam_intrinsics[cid]
            w = int(intr["width"])
            h = int(intr["height"])
            f = float(intr["focal_px"])
            cx = w * 0.5
            cy = h * 0.5
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist = np.zeros(5, dtype=np.float64)

            flags = cv2.CALIB_USE_INTRINSIC_GUESS
            if self.dist_mode == 0:
                flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST
            elif self.dist_mode == 1:
                flags |= cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST
            elif self.dist_mode == 2:
                flags |= cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST

            _, K_opt, d_opt, rvecs, tvecs = cv2.calibrateCamera(
                [obj_pts_np], [img_pts_np], (w, h), K, dist, flags=flags
            )

            rv = np.asarray(rvecs[0], dtype=np.float64).reshape(3)
            tv = np.asarray(tvecs[0], dtype=np.float64).reshape(3)
            f_opt = float(K_opt[0, 0])
            cx_opt = float(K_opt[0, 2])
            cy_opt = float(K_opt[1, 2])
            k1_opt = float(d_opt.reshape(-1)[0]) if d_opt.size > 0 else 0.0
            k2_opt = float(d_opt.reshape(-1)[1]) if d_opt.size > 1 else 0.0

            vec = np.array([rv[0], rv[1], rv[2], tv[0], tv[1], tv[2], f_opt, cx_opt, cy_opt, k1_opt, k2_opt], dtype=np.float64)
            self.cam_ref[cid] = vec.copy()
            self.cam_cur[cid] = vec.copy()

    def _init_windows(self):
        all_points = np.asarray([obs["X_world"] for obs in self.observations], dtype=np.float64)
        if all_points.size == 0:
            raise RuntimeError("No observations for window initialization")

        for wid in self.window_ids:
            cams_w = [cid for cid in self.cam_ids if self.cam_to_window.get(cid) == wid]
            if not cams_w:
                continue

            centers = np.asarray([_camera_center(self.cam_ref[c][:3], self.cam_ref[c][3:6]) for c in cams_w], dtype=np.float64)
            C_mean = centers.mean(axis=0)

            # Normal parallel to optical axis: single-cam axis or sign-aligned mean axis.
            axes = []
            for cid in cams_w:
                R, _ = cv2.Rodrigues(self.cam_ref[cid][0:3].reshape(3, 1))
                axis = (R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)).reshape(3,)
                na = np.linalg.norm(axis)
                if na > 1e-12:
                    axes.append(axis / na)
            if not axes:
                raise RuntimeError(f"Window {wid}: failed to compute optical axis from mapped cameras")

            ref_axis = np.asarray(axes[0], dtype=np.float64)
            aligned_axes = []
            for axis in axes:
                a = np.asarray(axis, dtype=np.float64)
                if float(np.dot(a, ref_axis)) < 0.0:
                    a = -a
                aligned_axes.append(a)
            n = np.mean(np.asarray(aligned_axes, dtype=np.float64), axis=0)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                raise RuntimeError(f"Window {wid}: degenerate optical-axis average")
            n = n / n_norm

            dists = np.linalg.norm(all_points - C_mean.reshape(1, 3), axis=1)
            if dists.size == 0:
                raise RuntimeError(f"Window {wid}: no valid 3D points for initialization")
            idx_min = int(np.argmin(dists))
            X_min = all_points[idx_min]

            # PREC-1 (Line 223): Window plane initialization using 50% Cartesian midpoint formula.
            # Verified: pt = 0.5 * (C_mean + X_min) is CORRECT and matches wand calibrator pattern.
            # This maximizes barrier constraint margins and is physically reasonable.
            # Midpoint initialization (no clamp).
            pt = 0.5 * (C_mean + X_min)

            # Enforce C++ side convention: camera side s(C)<0.
            s_cam = float(np.dot(n, C_mean - pt))
            if s_cam >= 0.0:
                n = -n

            self.win_ref[wid] = {
                "plane_pt": np.asarray(pt, dtype=np.float64),
                "plane_n": np.asarray(n, dtype=np.float64),
            }
            self.win_cur[wid] = {
                "plane_pt": np.asarray(pt, dtype=np.float64),
                "plane_n": np.asarray(n, dtype=np.float64),
            }

    # -------------------------------
    # Layout / bounds
    # -------------------------------
    def _layout(
        self,
        enable_planes: bool = True,
        enable_cam_t: bool = True,
        enable_cam_r: bool = True,
        enable_cam_f: bool = False,
        enable_cam_k1: bool = False,
        enable_cam_k2: bool = False,
    ) -> List[Tuple[str, int, int]]:
        layout = []
        for cid in self.cam_ids:
            if enable_cam_r:
                for k in range(3):
                    layout.append(("cam_r", cid, k))
            if enable_cam_t:
                for k in range(3):
                    layout.append(("cam_t", cid, k))
            if enable_cam_f:
                layout.append(("cam_f", cid, 0))
            if enable_cam_k1:
                layout.append(("cam_k1", cid, 0))
            if enable_cam_k2:
                layout.append(("cam_k2", cid, 0))

        if enable_planes:
            for wid in self.window_ids:
                layout.append(("plane_d", wid, 0))
                layout.append(("plane_a", wid, 0))
                layout.append(("plane_b", wid, 0))

        return layout

    def _bounds(
        self,
        layout: List[Tuple[str, int, int]],
        limit_rot_rad: float,
        limit_trans_mm: float,
        limit_plane_d_mm: float,
        limit_plane_angle_rad: float,
        plane_d_bounds: Optional[Dict[int, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(len(layout), dtype=np.float64)
        ub = np.zeros(len(layout), dtype=np.float64)
        for i, (t, pid, _) in enumerate(layout):
            if t == "cam_r":
                lb[i], ub[i] = -float(limit_rot_rad), float(limit_rot_rad)
            elif t == "cam_t":
                lb[i], ub[i] = -float(limit_trans_mm), float(limit_trans_mm)
            elif t == "cam_f":
                lb[i], ub[i] = -self.cfg.limit_focal_rel, self.cfg.limit_focal_rel
            elif t == "cam_k1":
                lb[i], ub[i] = -self.cfg.limit_k1, self.cfg.limit_k1
            elif t == "cam_k2":
                lb[i], ub[i] = -self.cfg.limit_k2, self.cfg.limit_k2
            elif t == "plane_d":
                if plane_d_bounds and int(pid) in plane_d_bounds:
                    bound_val = plane_d_bounds[int(pid)]
                    if isinstance(bound_val, (tuple, list)) and len(bound_val) == 2:
                        lo = float(bound_val[0])
                        hi = float(bound_val[1])
                        if hi < lo:
                            lo, hi = hi, lo
                        lb[i], ub[i] = lo, hi
                    else:
                        limit = float(abs(bound_val))
                        lb[i], ub[i] = -limit, limit
                else:
                    lb[i], ub[i] = -float(limit_plane_d_mm), float(limit_plane_d_mm)
            elif t in ("plane_a", "plane_b"):
                lb[i], ub[i] = -float(limit_plane_angle_rad), float(limit_plane_angle_rad)
        return lb, ub

    # -------------------------------
    # Apply delta and sync cpp
    # -------------------------------
    def _apply_x(self, x: np.ndarray, layout: List[Tuple[str, int, int]]):
        self.cam_cur = {cid: self.cam_ref[cid].copy() for cid in self.cam_ids}
        self.win_cur = {
            wid: {"plane_pt": self.win_ref[wid]["plane_pt"].copy(), "plane_n": self.win_ref[wid]["plane_n"].copy()}
            for wid in self.window_ids
        }

        win_delta = {wid: {"d": 0.0, "a": 0.0, "b": 0.0} for wid in self.window_ids}

        for val, (t, pid, sub) in zip(x, layout):
            if t == "cam_r":
                self.cam_cur[pid][sub] = self.cam_ref[pid][sub] + val
            elif t == "cam_t":
                self.cam_cur[pid][3 + sub] = self.cam_ref[pid][3 + sub] + val
            elif t == "cam_f":
                self.cam_cur[pid][6] = self.cam_ref[pid][6] * (1.0 + val)
            elif t == "cam_k1":
                self.cam_cur[pid][9] = self.cam_ref[pid][9] + val
            elif t == "cam_k2":
                self.cam_cur[pid][10] = self.cam_ref[pid][10] + val
            elif t == "plane_d":
                win_delta[pid]["d"] = float(val)
            elif t == "plane_a":
                win_delta[pid]["a"] = float(val)
            elif t == "plane_b":
                win_delta[pid]["b"] = float(val)

        for wid in self.window_ids:
            n0 = self.win_ref[wid]["plane_n"]
            pt0 = self.win_ref[wid]["plane_pt"]
            d = win_delta[wid]["d"]
            a = win_delta[wid]["a"]
            b = win_delta[wid]["b"]
            # BUG-2 (Line 351): Stale normal used in plane displacement.
            # Should use updated normal 'n' instead of reference 'n0' for geometric correctness.
            # Fix: Change 'pt = pt0 + d * n0' to 'pt = pt0 + d * n'
            n = update_normal_tangent(n0, a, b)
            pt = pt0 + d * n
            self.win_cur[wid]["plane_n"] = n
            self.win_cur[wid]["plane_pt"] = pt

    def _sync_cpp(self):
        for cid in self.cam_ids:
            p = self.cam_cur[cid]
            wid = self.cam_to_window[cid]
            w = self.win_cur[wid]
            media = self.window_media[wid]
            intr = self.cam_intrinsics[cid]

            update_cpp_camera_state(
                self.cams_cpp[cid],
                extrinsics={"rvec": p[0:3], "tvec": p[3:6]},
                intrinsics={"f": float(p[6]), "cx": float(p[7]), "cy": float(p[8]), "dist": [float(p[9]), float(p[10]), 0.0, 0.0, 0.0]},
                plane_geom={"pt": w["plane_pt"].tolist(), "n": w["plane_n"].tolist()},
                media_props={
                    "n_air": float(media["n1"]),
                    "n_window": float(media["n2"]),
                    "n_object": float(media["n3"]),
                    "thickness": float(media["thickness"]),
                },
                image_size=(int(intr["height"]), int(intr["width"])),
                solver_opts={"proj_tol": float(media.get("proj_tol", 1e-6)), "proj_nmax": int(media.get("proj_nmax", 1000)), "lr": float(media.get("lr", 0.1))},
            )

    # -------------------------------
    # Residuals
    # -------------------------------
    def _residuals(self, x: np.ndarray, layout: List[Tuple[str, int, int]]) -> np.ndarray:
        self._apply_x(x, layout)
        self._sync_cpp()

        sigma = max(self.cfg.sigma_proj_px, 1e-9)
        pen = self.cfg.penalty_proj_px / sigma

        by_cam: Dict[int, List[Tuple[np.ndarray, np.ndarray, int]]] = {cid: [] for cid in self.cam_ids}
        for i, obs in enumerate(self.observations):
            X = np.asarray(obs["X_world"], dtype=np.float64)
            for cid, uv in obs["uv_by_cam"].items():
                if cid in by_cam:
                    by_cam[cid].append((X, np.asarray(uv, dtype=np.float64), i))

        residuals = []
        proj_sq = 0.0
        proj_n = 0
        barrier_viol = 0

        for cid, items in by_cam.items():
            if not items:
                continue
            pts = [lpt.Pt3D(float(x[0]), float(x[1]), float(x[2])) for x, _, _ in items]
            results = self.cams_cpp[cid].projectBatchStatus(pts, False)
            wid = self.cam_to_window[cid]
            n = self.win_cur[wid]["plane_n"]
            P = self.win_cur[wid]["plane_pt"]

            for (X, uv_obs, _), rr in zip(items, results):
                ok, uv_pred, _ = rr
                if ok:
                    du = float(uv_pred[0] - uv_obs[0])
                    dv = float(uv_pred[1] - uv_obs[1])
                    residuals.extend([du / sigma, dv / sigma])
                    proj_sq += du * du + dv * dv
                    proj_n += 2
                else:
                    residuals.extend([pen, pen])

                # PREC-2: Smooth softplus barrier (C∞ continuous, no kink at gap=0)
                # Replaces hard if/else switch with smooth approximation: softplus(gap) ≈ max(gap, 0)
                if self.cfg.barrier_enabled:
                    sX = float(np.dot(n, X - P))
                    gap = self.cfg.margin_side_mm - sX
                    
                    # Smooth barrier using softplus: tau * log1p(exp(gap/tau))
                    tau = max(self.cfg.tau, 1e-9)
                    gap_smooth = tau * np.logaddexp(0.0, gap / tau)
                    
                    c_gate = self.cfg.alpha_side_gate
                    r_fix_const = np.sqrt(2.0 * c_gate)
                    r_grad_const = np.sqrt(2.0 * self.cfg.beta_side_dir)
                    
                    residuals.append(r_fix_const * (1.0 - np.exp(-gap_smooth / tau)))
                    residuals.append(r_grad_const * gap_smooth)
                    
                    # Track violations for diagnostics (when gap > 0)
                    if gap > 0:
                        barrier_viol += 1

        # FIXED: RMSE denominator now correctly divides by N observations instead of 2N.
        # proj_sq = sum(du² + dv²) per observation (correct)
        # proj_n = 2 per observation (one for du, one for dv) → divide by proj_n // 2 to get N
        self._eval_count += 1
        self._last_proj_rmse = float(np.sqrt(proj_sq / max(proj_n // 2, 1)))
        arr = np.asarray(residuals, dtype=np.float64)
        self._last_cost = 0.5 * float(np.sum(arr * arr))

        if self.progress_callback and (self._eval_count % 10 == 0):
            phase = getattr(self, "_phase_name", "Stage")
            self.progress_callback(phase, self._last_proj_rmse, self._last_cost)

        return arr if arr.size > 0 else np.array([1e6], dtype=np.float64)

    def _sync_refs_from_current(self):
        self.cam_ref = {cid: self.cam_cur[cid].copy() for cid in self.cam_ids}
        self.win_ref = {
            wid: {"plane_pt": self.win_cur[wid]["plane_pt"].copy(), "plane_n": self.win_cur[wid]["plane_n"].copy()}
            for wid in self.window_ids
        }

    def _run_stage(
        self,
        stage_name: str,
        *,
        enable_planes: bool,
        enable_cam_t: bool,
        enable_cam_r: bool,
        enable_cam_f: bool,
        enable_cam_k1: bool,
        enable_cam_k2: bool,
        limit_rot_rad: float,
        limit_trans_mm: float,
        limit_plane_d_mm: float,
        limit_plane_angle_rad: float,
        max_nfev: int,
        ftol: float,
        xtol: float,
        gtol: float,
        loss: str,
        plane_d_bounds: Optional[Dict[int, Any]] = None,
    ):
        self._phase_name = stage_name
        self._eval_count = 0
        layout = self._layout(
            enable_planes=enable_planes,
            enable_cam_t=enable_cam_t,
            enable_cam_r=enable_cam_r,
            enable_cam_f=enable_cam_f,
            enable_cam_k1=enable_cam_k1,
            enable_cam_k2=enable_cam_k2,
        )
        x0 = np.zeros(len(layout), dtype=np.float64)
        lb, ub = self._bounds(
            layout,
            limit_rot_rad=limit_rot_rad,
            limit_trans_mm=limit_trans_mm,
            limit_plane_d_mm=limit_plane_d_mm,
            limit_plane_angle_rad=limit_plane_angle_rad,
            plane_d_bounds=plane_d_bounds,
        )

        _ = self._residuals(x0, layout)
        proj_before = float(self._last_proj_rmse)
        cost_before = float(self._last_cost)
        if self.progress_callback:
            self.progress_callback(stage_name, proj_before, cost_before)
        if self.cfg.verbosity >= 1:
            print(
                f"[PlateRefr][{stage_name}] start: proj_rmse={proj_before:.6f} px, "
                f"max_nfev={int(max_nfev)}, ftol={ftol:.1e}, xtol={xtol:.1e}, gtol={gtol:.1e}"
            )

        # PREC-3 (Line 503): Missing parameter scaling for heterogeneous parameter space.
        # Optimizer mixes mm, radians, relative deltas without x_scale or diff_step.
        # Fix: Add x_scale='jac' (Jacobian-adaptive) and diff_step array per parameter type.
        diff_step = np.ones(len(layout), dtype=np.float64)
        
        for i, (t, pid, sub) in enumerate(layout):
            if t == "cam_r":
                diff_step[i] = 1e-4
            elif t == "cam_t":
                diff_step[i] = 1e-2
            elif t == "cam_f":
                diff_step[i] = 1e-3
            elif t == "cam_k1" or t == "cam_k2":
                diff_step[i] = 1e-6
            elif t == "plane_d":
                diff_step[i] = 1e-2
            elif t == "plane_a" or t == "plane_b":
                diff_step[i] = 1e-4
        
        res = least_squares(
            lambda x: self._residuals(x, layout),
            x0,
            method=self.cfg.method,
            loss=loss,
            bounds=(lb, ub),
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            verbose=0,
            x_scale='jac',
            diff_step=diff_step,
        )

        self._apply_x(res.x, layout)
        self._sync_cpp()
        _ = self._residuals(res.x, layout)
        proj_after = float(self._last_proj_rmse)
        cost_after = float(self._last_cost)
        stop_reason = self._stop_reason_text(int(res.status), str(res.message), int(res.nfev), int(max_nfev))

        if self.cfg.verbosity >= 1:
            print(
                f"[PlateRefr][{stage_name}] done: proj_rmse [{proj_before:.6f}] -> [{proj_after:.6f}] px; "
                f"nfev={int(res.nfev)}; {stop_reason}"
            )

        self._sync_refs_from_current()

        hit_plane_angle_boundary = False
        try:
            active_mask = np.asarray(getattr(res, "active_mask", []), dtype=np.int8)
            if active_mask.size == len(layout):
                for idx, (ptype, _pid, _sub) in enumerate(layout):
                    if ptype in ("plane_a", "plane_b") and active_mask[idx] != 0:
                        hit_plane_angle_boundary = True
                        break
        except Exception:
            hit_plane_angle_boundary = False

        # BUG-3 (Line 543): Success flag treats status=0 (max_nfev reached) as success.
        # Scipy status=0 means iteration limit hit WITHOUT convergence.
        # Fix: Use 'res.status > 0' or 'res.success' (which is False for status <= 0)
        return {
            "success": bool(int(res.status) > 0),  # FIXED: Only status > 0 indicates convergence
            "converged": bool(res.success),
            "message": str(res.message),
            "status": int(res.status),
            "nfev": int(res.nfev),
            "max_nfev": int(max_nfev),
            "optimality": float(getattr(res, "optimality", np.nan)),
            "stop_reason": stop_reason,
            "cost": float(cost_after),
            "cost_before": float(cost_before),
            "cost_after": float(cost_after),
            "proj_rmse": float(proj_after),
            "proj_rmse_before": float(proj_before),
            "proj_rmse_after": float(proj_after),
            "hit_plane_angle_boundary": bool(hit_plane_angle_boundary),
        }

    def _build_step_a_plane_d_bounds(self, loop_iter: int) -> Dict[int, Tuple[float, float]]:
        if not hasattr(self, "_weak_window_refs"):
            self._weak_window_refs = {}

        all_points = np.asarray([obs["X_world"] for obs in self.observations], dtype=np.float64)
        if all_points.size == 0:
            return {}

        eps = max(float(self.cfg.margin_side_mm), 1e-3)
        factor = 0.1 * (0.5 ** max(0, int(loop_iter) - 1))
        out: Dict[int, Tuple[float, float]] = {}

        if self.cfg.verbosity >= 1:
            print(f"[PlateRefr][Loop {loop_iter}] plane_d bounds (Step A)")

        for wid in self.window_ids:
            if wid not in self.win_ref:
                continue

            pt0 = np.asarray(self.win_ref[wid]["plane_pt"], dtype=np.float64)
            n0 = np.asarray(self.win_ref[wid]["plane_n"], dtype=np.float64)
            nn = np.linalg.norm(n0)
            if nn < 1e-12:
                continue
            n0 = n0 / nn

            cams_w = [cid for cid in self.cam_ids if self.cam_to_window.get(cid) == wid]
            if not cams_w:
                continue

            lo_all = -np.inf
            hi_all = np.inf
            cam_ranges = []
            min_dist_for_weak = None

            for cid in cams_w:
                c = _camera_center(self.cam_ref[cid][:3], self.cam_ref[cid][3:6])
                dists = np.linalg.norm(all_points - c.reshape(1, 3), axis=1)
                if dists.size == 0:
                    continue
                idx = int(np.argmin(dists))
                x_min = all_points[idx]
                d_min = float(dists[idx])
                if min_dist_for_weak is None or d_min < min_dist_for_weak:
                    min_dist_for_weak = d_min

                s_c0 = float(np.dot(n0, c - pt0))
                s_x0 = float(np.dot(n0, x_min - pt0))
                lo = min(s_c0, s_x0) + eps
                hi = max(s_c0, s_x0) - eps
                if hi <= lo:
                    continue
                cam_ranges.append((cid, lo, hi, d_min))
                lo_all = max(lo_all, lo)
                hi_all = min(hi_all, hi)

            if not cam_ranges:
                continue

            weak = False
            if len(cams_w) == 1:
                cid = cams_w[0]
                R, _ = cv2.Rodrigues(self.cam_ref[cid][0:3].reshape(3, 1))
                opt_axis = np.asarray(R[2, :], dtype=np.float64)
                costh = abs(float(np.dot(n0, opt_axis) / (np.linalg.norm(opt_axis) + 1e-12)))
                angle_deg = float(np.degrees(np.arccos(min(1.0, max(-1.0, costh)))))
                weak = angle_deg < 5.0

            geo_lo, geo_hi = lo_all, hi_all
            weak_lo, weak_hi = None, None
            if weak and min_dist_for_weak is not None and min_dist_for_weak > 1e-6:
                if wid not in self._weak_window_refs:
                    self._weak_window_refs[wid] = float(min_dist_for_weak)
                d_ref = float(self._weak_window_refs[wid])
                weak_lo = -d_ref * factor
                weak_hi = d_ref * factor
                lo_all = max(lo_all, weak_lo)
                hi_all = min(hi_all, weak_hi)

            if hi_all <= lo_all:
                mid = 0.5 * (lo_all + hi_all)
                lo_all = mid - 1e-3
                hi_all = mid + 1e-3

            # Delta-parameterization safety: plane_d bounds are for delta and must include x0=0.
            if not (lo_all <= 0.0 <= hi_all):
                raw_lo, raw_hi = lo_all, hi_all
                lo_all = min(lo_all, -1e-6)
                hi_all = max(hi_all, 1e-6)
                if self.cfg.verbosity >= 1:
                    print(
                        f"  [d-bound-fix] Win {wid}: raw [{raw_lo:.3f}, {raw_hi:.3f}] excluded 0; "
                        f"adjusted to [{lo_all:.3f}, {hi_all:.3f}] for delta x0=0"
                    )

            out[int(wid)] = (float(lo_all), float(hi_all))

            if self.cfg.verbosity >= 1:
                kind = "WEAK" if weak else "STRONG"
                cam_str = ", ".join([f"cam{cid}: [{lo:.2f},{hi:.2f}] dmin={dmin:.1f}" for cid, lo, hi, dmin in cam_ranges])
                if weak and weak_lo is not None and weak_hi is not None:
                    print(
                        f"  Win {wid} ({kind}) -> d in [{lo_all:.3f}, {hi_all:.3f}] mm "
                        f"= intersect(geo:[{geo_lo:.3f},{geo_hi:.3f}], weak:[{weak_lo:.3f},{weak_hi:.3f}]) | {cam_str}"
                    )
                else:
                    print(f"  Win {wid} ({kind}) -> d in [{lo_all:.3f}, {hi_all:.3f}] mm | {cam_str}")

        return out

    def _run_alternating_loop(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        max_loop_iters = int(self.cfg.max_loop_iters)
        for loop_iter in range(1, max_loop_iters + 1):
            plane_d_bounds = self._build_step_a_plane_d_bounds(loop_iter)

            step_a = self._run_stage(
                f"Loop {loop_iter} - Step A: Planes",
                enable_planes=True,
                enable_cam_t=False,
                enable_cam_r=False,
                enable_cam_f=False,
                enable_cam_k1=False,
                enable_cam_k2=False,
                limit_rot_rad=0.0,
                limit_trans_mm=0.0,
                limit_plane_d_mm=500.0,
                limit_plane_angle_rad=float(np.deg2rad(2.5)),
                max_nfev=int(self.cfg.max_nfev_joint),
                ftol=float(self.cfg.loop_ftol),
                xtol=float(self.cfg.loop_xtol),
                gtol=float(self.cfg.loop_gtol),
                loss=self.cfg.loss,
                plane_d_bounds=plane_d_bounds,
            )
            self._log_two_plane_angle(f"Loop {loop_iter} Step A")

            step_b = self._run_stage(
                f"Loop {loop_iter} - Step B: Cameras",
                enable_planes=False,
                enable_cam_t=True,
                enable_cam_r=True,
                enable_cam_f=False,
                enable_cam_k1=False,
                enable_cam_k2=False,
                limit_rot_rad=float(np.deg2rad(180.0)),
                limit_trans_mm=2000.0,
                limit_plane_d_mm=0.0,
                limit_plane_angle_rad=0.0,
                max_nfev=int(self.cfg.max_nfev_joint),
                ftol=float(self.cfg.loop_ftol),
                xtol=float(self.cfg.loop_xtol),
                gtol=float(self.cfg.loop_gtol),
                loss=self.cfg.loss,
                plane_d_bounds=None,
            )

            summaries.append({
                "loop": int(loop_iter),
                "step_a": step_a,
                "step_b": step_b,
            })

            if not bool(step_a.get("hit_plane_angle_boundary", False)):
                if self.cfg.verbosity >= 1:
                    print(f"[PlateRefr] Alternating loop early stop at pass {loop_iter} (plane angle bounds inactive)")
                break
        return summaries

    def _log_two_plane_angle(self, tag: str):
        if len(self.window_ids) != 2:
            return
        try:
            w0, w1 = self.window_ids[0], self.window_ids[1]
            n0 = np.asarray(self.win_cur[w0]["plane_n"], dtype=np.float64)
            n1 = np.asarray(self.win_cur[w1]["plane_n"], dtype=np.float64)
            n0n = float(np.linalg.norm(n0))
            n1n = float(np.linalg.norm(n1))
            if n0n < 1e-12 or n1n < 1e-12:
                return
            c = float(np.dot(n0, n1) / (n0n * n1n))
            c = float(np.clip(c, -1.0, 1.0))
            ang = float(np.degrees(np.arccos(c)))
            print(f"[PlateRefr][{tag}] plane-angle(Win {w0}, Win {w1}) = {ang:.6f} deg")
        except Exception:
            return

    # -------------------------------
    # Final metrics / alignment
    # -------------------------------
    def _align_final(self):
        points_3d = [np.asarray(obs["X_world"], dtype=np.float64) for obs in self.observations]
        new_cam, new_win, new_pts, _, _ = align_world_y_to_plane_intersection(
            {wid: {"plane_n": self.win_cur[wid]["plane_n"], "plane_pt": self.win_cur[wid]["plane_pt"]} for wid in self.window_ids},
            {cid: self.cam_cur[cid].copy() for cid in self.cam_ids},
            points_3d=points_3d,
            align_mode="yz",
        )
        self.cam_cur = {cid: np.asarray(new_cam[cid], dtype=np.float64) for cid in self.cam_ids}
        self.win_cur = {
            wid: {
                "plane_n": np.asarray(new_win[wid]["plane_n"], dtype=np.float64),
                "plane_pt": np.asarray(new_win[wid]["plane_pt"], dtype=np.float64),
            }
            for wid in self.window_ids
        }
        aligned = new_pts if new_pts is not None else points_3d
        for i, X in enumerate(aligned):
            if i < len(self.observations):
                self.observations[i]["X_world"] = np.asarray(X, dtype=np.float64)
        return aligned

    def _compute_error_stats(self):
        per_cam_proj: Dict[int, List[float]] = {cid: [] for cid in self.cam_ids}
        per_cam_tri: Dict[int, List[float]] = {cid: [] for cid in self.cam_ids}

        by_cam: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {cid: [] for cid in self.cam_ids}
        for obs in self.observations:
            X = np.asarray(obs["X_world"], dtype=np.float64)
            for cid, uv in obs["uv_by_cam"].items():
                if cid in by_cam:
                    by_cam[cid].append((X, np.asarray(uv, dtype=np.float64)))

        for cid, items in by_cam.items():
            if not items:
                continue
            try:
                pts = [lpt.Pt3D(float(X[0]), float(X[1]), float(X[2])) for X, _ in items]
                proj_res = self.cams_cpp[cid].projectBatchStatus(pts, False)
                los_res = self.cams_cpp[cid].lineOfSightBatchStatus([lpt.Pt2D(float(uv[0]), float(uv[1])) for _, uv in items])
            except Exception as exc:
                if self.cfg.verbosity >= 1:
                    print(f"[PlateRefr][Stats] Cam {cid}: batch eval failed: {exc}")
                continue

            for (X, uv), pp, rr in zip(items, proj_res, los_res):
                try:
                    okp = bool(pp[0])
                    uvp = pp[1]
                    if okp:
                        e = float(np.linalg.norm(np.asarray([uvp[0], uvp[1]]) - uv))
                        if np.isfinite(e):
                            per_cam_proj[cid].append(e)
                except Exception:
                    pass
                try:
                    okr = bool(rr[0])
                    line = rr[1]
                    if okr:
                        o = np.array([line.pt[0], line.pt[1], line.pt[2]], dtype=np.float64)
                        d = np.array([line.unit_vector[0], line.unit_vector[1], line.unit_vector[2]], dtype=np.float64)
                        dn = np.linalg.norm(d)
                        if dn > 1e-12:
                            d = d / dn
                            dist = float(point_to_ray_dist(X, o, d))
                            if np.isfinite(dist):
                                per_cam_tri[cid].append(dist)
                except Exception:
                    pass

        # Compute detailed statistics for each camera
        proj_detail = {}
        tri_detail = {}
        for cid in self.cam_ids:
            vp = per_cam_proj[cid]
            vt = per_cam_tri[cid]
            
            # Compute detailed stats
            proj_detail[cid] = _summarize_errors(vp) if vp else {}
            tri_detail[cid] = _summarize_errors(vt) if vt else {}
        
        # Build legacy tuple format from detailed stats for backward compatibility
        proj_stats = {}
        tri_stats = {}
        for cid in self.cam_ids:
            pd = proj_detail[cid]
            td = tri_detail[cid]
            proj_stats[cid] = (pd.get("mean", 0.0), pd.get("std", 0.0)) if pd else (0.0, 0.0)
            tri_stats[cid] = (td.get("mean", 0.0), td.get("std", 0.0)) if td else (0.0, 0.0)
        
        return proj_stats, tri_stats, proj_detail, tri_detail

    def run(self) -> Dict[str, Any]:
        self._init_pinhole_per_camera()
        self._init_windows()
        self._sync_cpp()

        self.cfg.optimize_k1_stage_b = bool(self.dist_mode >= 1)
        self.cfg.optimize_k2_stage_b = bool(self.dist_mode >= 2)

        loop_summaries = self._run_alternating_loop()

        if self.progress_callback:
            self.progress_callback("Joint: Planes + Extrinsics", 0.0, 0.0)
        stage_a = self._run_stage(
            "Joint: Planes + Extrinsics",
            enable_planes=True,
            enable_cam_t=True,
            enable_cam_r=True,
            enable_cam_f=False,
            enable_cam_k1=False,
            enable_cam_k2=False,
            limit_rot_rad=float(np.deg2rad(20.0)),
            limit_trans_mm=50.0,
            limit_plane_d_mm=50.0,
            limit_plane_angle_rad=float(np.deg2rad(10.0)),
            max_nfev=int(self.cfg.max_nfev_joint),
            ftol=float(self.cfg.ftol),
            xtol=float(self.cfg.xtol),
            gtol=float(self.cfg.gtol),
            loss=self.cfg.loss,
            plane_d_bounds=None,
        )
        self._log_two_plane_angle("Joint")

        if self.progress_callback:
            self.progress_callback("Final Refined: Joint + Intrinsics", stage_a["proj_rmse"], stage_a["cost"])
        stage_b = self._run_stage(
            "Final Refined: Joint + Intrinsics",
            enable_planes=True,
            enable_cam_t=True,
            enable_cam_r=True,
            enable_cam_f=bool(self.cfg.optimize_focal_stage_b),
            enable_cam_k1=bool(self.cfg.optimize_k1_stage_b),
            enable_cam_k2=bool(self.cfg.optimize_k2_stage_b),
            limit_rot_rad=float(np.deg2rad(20.0)),
            limit_trans_mm=50.0,
            limit_plane_d_mm=10.0,
            limit_plane_angle_rad=float(np.deg2rad(5.0)),
            max_nfev=int(self.cfg.max_nfev_final),
            ftol=float(self.cfg.ftol),
            xtol=float(self.cfg.xtol),
            gtol=float(self.cfg.gtol),
            loss=self.cfg.loss,
            plane_d_bounds=None,
        )
        self._log_two_plane_angle("Final Refined")

        aligned_points = self._align_final()
        self._sync_cpp()
        try:
            proj_stats, tri_stats, proj_detail, tri_detail = self._compute_error_stats()
        except Exception as exc:
            if self.cfg.verbosity >= 1:
                print(f"[PlateRefr] Warning: failed to compute final error stats: {exc}")
            proj_stats, tri_stats, proj_detail, tri_detail = {}, {}, {}, {}

        cams_out = {}
        for cid in self.cam_ids:
            p = self.cam_cur[cid]
            cams_out[cid] = {
                "rvec": p[0:3].copy(),
                "tvec": p[3:6].copy(),
                "f": float(p[6]),
                "cx": float(p[7]),
                "cy": float(p[8]),
                "k1": float(p[9]),
                "k2": float(p[10]),
                "width": int(self.cam_intrinsics[cid]["width"]),
                "height": int(self.cam_intrinsics[cid]["height"]),
                "window_id": int(self.cam_to_window[cid]),
            }

        wins_out = {}
        for wid in self.window_ids:
            media = self.window_media[wid]
            wins_out[wid] = {
                "plane_pt": self.win_cur[wid]["plane_pt"].copy(),
                "plane_n": self.win_cur[wid]["plane_n"].copy(),
                "n1": float(media["n1"]),
                "n2": float(media["n2"]),
                "n3": float(media["n3"]),
                "thickness": float(media["thickness"]),
                "proj_tol": float(media.get("proj_tol", 1e-6)),
                "proj_nmax": int(media.get("proj_nmax", 1000)),
                "lr": float(media.get("lr", 0.1)),
            }

        return {
            "success": bool(stage_a["success"] and stage_b["success"]),
            "stage_a": stage_a,
            "stage_b": stage_b,
            "loop_summaries": loop_summaries,
            "camera_params": cams_out,
            "window_params": wins_out,
            "cam_to_window": {int(k): int(v) for k, v in self.cam_to_window.items()},
            "per_camera_proj_err_stats": proj_stats,
            "per_camera_tri_err_stats": tri_stats,
            "per_camera_proj_err_detail": proj_detail,
            "per_camera_tri_err_detail": tri_detail,
            "aligned_points": aligned_points,
        }


class RefractionPlateWorker(QObject):
    finished = Signal(bool, object, str)
    progress = Signal(str, float, float)
    error = Signal(str)

    def __init__(
        self,
        observations: List[Dict[str, Any]],
        cam_intrinsics: Dict[int, Dict[str, float]],
        cam_to_window: Dict[int, int],
        window_media: Dict[int, Dict[str, float]],
        dist_mode: int,
        cfg: Optional[RefractionPlateConfig] = None,
    ):
        super().__init__()
        self.observations = observations
        self.cam_intrinsics = cam_intrinsics
        self.cam_to_window = cam_to_window
        self.window_media = window_media
        self.dist_mode = int(dist_mode)
        self.cfg = cfg if cfg is not None else RefractionPlateConfig()
        self._killed = False

    def kill(self):
        self._killed = True

    def run(self):
        try:
            if self._killed:
                raise RuntimeError("Worker killed")

            def cb(phase: str, proj_rmse: float, cost: float):
                if self._killed:
                    raise RuntimeError("Worker killed")
                self.progress.emit(str(phase), float(proj_rmse), float(cost))

            calibrator = RefractionPlateCalibrator(
                cfg=self.cfg,
                observations=self.observations,
                cam_intrinsics=self.cam_intrinsics,
                cam_to_window=self.cam_to_window,
                window_media=self.window_media,
                dist_mode=self.dist_mode,
                progress_callback=cb,
            )
            result = calibrator.run()
            ok = bool(result.get("success", False))
            if ok:
                msg = ""
            else:
                sa = result.get("stage_a", {})
                sb = result.get("stage_b", {})
                msg = (
                    f"Stage A: {sa.get('stop_reason', sa.get('message', 'N/A'))}; "
                    f"Stage B: {sb.get('stop_reason', sb.get('message', 'N/A'))}"
                )
            self.finished.emit(ok, result, msg)
        except Exception as e:
            self.error.emit(f"[RefractionPlate] Worker Error:\n{traceback.format_exc()}\n{e}")
