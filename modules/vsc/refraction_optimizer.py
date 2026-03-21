"""
Refraction-aware VSC optimizer for PINPLATE cameras.

Design goals:
- Optimize only camera extrinsics (rvec, tvec) for stability.
- Use ray-consistency residuals as the primary objective.
- Enforce side constraint (non-penetration) against refractive planes
  using the same barrier style as refraction_calibration_BA.py.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Callable, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares


class RefractionVSCOptimizer:
    def __init__(
        self,
        max_nfev: int = 50,
        ftol: float = 1e-6,
        xtol: float = 1e-6,
        f_scale: float = 1.5,
        sigma_ray_mm: float = 1.0,
        margin_side_mm: float = 0.05,
        alpha_side_gate: float = 10.0,
        beta_side_dir: float = 1e4,
        tau: float = 0.01,
    ):
        self.max_nfev = int(max_nfev)
        self.ftol = float(ftol)
        self.xtol = float(xtol)
        self.f_scale = float(f_scale)
        self.sigma_ray_mm = max(float(sigma_ray_mm), 1e-9)

        # Barrier parameters (aligned with BA defaults).
        self.margin_side_mm = float(margin_side_mm)
        self.alpha_side_gate = float(alpha_side_gate)
        self.beta_side_dir = float(beta_side_dir)
        self.tau = max(float(tau), 1e-6)

        self.log_callback: Optional[Callable[[str], None]] = None

        self._cam_indices: List[int] = []
        self._camera_models = {}
        self._camera_states = {}
        self._correspondences = []
        self._cam_to_window = {}
        self._window_planes = {}
        self._j_ref = 1.0
        self._layouts = []
        self._cam_layout_indices = {}
        self._cam_uv_inputs = {}
        self._nfev_count = 0
        self._window_refraction = {}
        self._window_to_cams = {}
        self._plane_window_ids: List[int] = []
        self._plane_base = {}

    def set_log_callback(self, callback: Callable[[str], None]):
        self.log_callback = callback

    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def optimize_all_cameras(
        self,
        camera_models: Dict[int, object],
        camera_states: Dict[int, dict],
        correspondences: List[dict],
        cam_to_window: Dict[int, int],
        window_planes: Dict[int, dict],
    ) -> Tuple[Dict[int, dict], dict]:
        self._setup_problem(
            camera_models,
            camera_states,
            correspondences,
            cam_to_window,
            window_planes,
        )

        if len(self._cam_indices) < 2 or len(self._layouts) < 10:
            return camera_states, {
                "ray_before": 0.0,
                "ray_after": 0.0,
                "proj_before": 0.0,
                "proj_after": 0.0,
                "n_points": len(self._layouts),
                "converged": False,
            }

        x0 = self._pack_params(camera_states)
        lb, ub = self._build_bounds(camera_states)
        self._nfev_count = 0

        stats_before = self._compute_metrics(x0)
        self._j_ref = max(stats_before["triang_rmse"] ** 2, 1e-6)

        self._log(
            f"  Initial (Refraction): RayErr={stats_before['triang_rmse']:.4f}mm, "
            f"ProjErr={stats_before['proj_rmse']:.4f}px"
        )

        result = least_squares(
            self._residuals,
            x0,
            method="trf",
            loss="huber",
            f_scale=self.f_scale,
            bounds=(lb, ub),
            max_nfev=self.max_nfev,
            ftol=self.ftol,
            xtol=self.xtol,
            x_scale="jac",
            verbose=0,
        )

        x_opt = result.x
        self._apply_params(x_opt)
        updated_states = self._states_from_x(x_opt, camera_states)
        stats_after = self._compute_metrics(x_opt)

        self._log(
            f"  Final (Refraction): RayErr={stats_after['triang_rmse']:.4f}mm, "
            f"ProjErr={stats_after['proj_rmse']:.4f}px"
        )
        self._log(f"  Converged: {result.success}, nfev={result.nfev}, message={result.message}")
        self._log(
            f"  Barrier: violations={stats_after['barrier_violations']}, "
            f"min_sX={stats_after['min_sX']:.6f}"
        )

        overlay_points_optim = self._compute_overlay_points(x_opt)

        return updated_states, {
            "ray_before": stats_before["triang_rmse"],
            "ray_after": stats_after["triang_rmse"],
            "proj_before": stats_before["proj_rmse"],
            "proj_after": stats_after["proj_rmse"],
            "n_points": len(self._layouts),
            "converged": bool(result.success),
            "n_cams": len(self._cam_indices),
            "nfev": int(result.nfev),
            "full_stats": stats_after,
            "overlay_points_optim": overlay_points_optim,
        }

    def optimize_window_planes(
        self,
        camera_models: Dict[int, object],
        camera_states: Dict[int, dict],
        correspondences: List[dict],
        cam_to_window: Dict[int, int],
        window_planes: Dict[int, dict],
        window_refraction: Dict[int, dict],
        max_nfev: int = 50,
        max_delta_d_mm: float = 5.0,
        max_delta_theta_deg: float = 2.0,
    ) -> Tuple[Dict[int, dict], dict]:
        self._setup_problem(
            camera_models,
            camera_states,
            correspondences,
            cam_to_window,
            window_planes,
        )
        self._window_refraction = window_refraction or {}

        if len(self._cam_indices) < 2 or len(self._layouts) < 10:
            return window_planes, {
                "ray_before": 0.0,
                "ray_after": 0.0,
                "proj_before": 0.0,
                "proj_after": 0.0,
                "n_points": len(self._layouts),
                "n_windows": 0,
                "converged": False,
            }

        self._window_to_cams = {}
        for cid in self._cam_indices:
            wid = self._cam_to_window.get(cid, None)
            if wid is None:
                continue
            if wid not in self._window_planes:
                continue
            self._window_to_cams.setdefault(int(wid), []).append(int(cid))

        wids_all = sorted({wid for _, _, wids in self._layouts for wid in wids})
        self._plane_window_ids = [
            int(wid)
            for wid in wids_all
            if wid in self._window_to_cams and wid in self._window_refraction and wid in self._window_planes
        ]

        if not self._plane_window_ids:
            return window_planes, {
                "ray_before": 0.0,
                "ray_after": 0.0,
                "proj_before": 0.0,
                "proj_after": 0.0,
                "n_points": len(self._layouts),
                "n_windows": 0,
                "converged": False,
            }

        self._plane_base = {}
        for wid in self._plane_window_ids:
            pl = self._window_planes[wid]
            pt0 = np.asarray(pl["plane_pt"], dtype=np.float64).reshape(3)
            n0 = np.asarray(pl["plane_n"], dtype=np.float64).reshape(3)
            n_norm = np.linalg.norm(n0)
            if n_norm <= 1e-12:
                n0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                n0 = n0 / n_norm
            u0, v0 = self._plane_tangent_basis(n0)
            self._plane_base[wid] = {
                "pt0": pt0,
                "n0": n0,
                "u0": u0,
                "v0": v0,
            }

        x_cam_fixed = self._pack_params(self._camera_states)
        self._apply_params(x_cam_fixed)

        x0 = np.zeros(3 * len(self._plane_window_ids), dtype=np.float64)
        max_d = float(abs(max_delta_d_mm))
        max_theta = np.deg2rad(float(abs(max_delta_theta_deg)))
        lb = np.tile(np.array([-max_d, -max_theta, -max_theta], dtype=np.float64), len(self._plane_window_ids))
        ub = np.tile(np.array([max_d, max_theta, max_theta], dtype=np.float64), len(self._plane_window_ids))

        self._nfev_count = 0
        stats_before = self._compute_metrics(x_cam_fixed)
        self._j_ref = max(stats_before["triang_rmse"] ** 2, 1e-6)

        self._log(
            f"  Initial (Refraction Plane): RayErr={stats_before['triang_rmse']:.4f}mm, "
            f"ProjErr={stats_before['proj_rmse']:.4f}px"
        )

        result = least_squares(
            self._residuals_plane,
            x0,
            method="trf",
            loss="huber",
            f_scale=self.f_scale,
            bounds=(lb, ub),
            max_nfev=int(max_nfev),
            ftol=self.ftol,
            xtol=self.xtol,
            x_scale="jac",
            verbose=0,
        )

        self._apply_plane_params(result.x)
        stats_after = self._compute_metrics(x_cam_fixed)
        overlay_points_optim = self._compute_overlay_points(x_cam_fixed)

        self._log(
            f"  Final (Refraction Plane): RayErr={stats_after['triang_rmse']:.4f}mm, "
            f"ProjErr={stats_after['proj_rmse']:.4f}px"
        )
        self._log(f"  Converged: {result.success}, nfev={result.nfev}, message={result.message}")
        self._log(
            f"  Barrier: violations={stats_after['barrier_violations']}, "
            f"min_sX={stats_after['min_sX']:.6f}"
        )

        window_planes_updated = {
            int(wid): {
                "plane_pt": np.asarray(pl["plane_pt"], dtype=np.float64).reshape(3),
                "plane_n": np.asarray(pl["plane_n"], dtype=np.float64).reshape(3),
            }
            for wid, pl in self._window_planes.items()
        }

        return window_planes_updated, {
            "ray_before": stats_before["triang_rmse"],
            "ray_after": stats_after["triang_rmse"],
            "proj_before": stats_before["proj_rmse"],
            "proj_after": stats_after["proj_rmse"],
            "n_points": len(self._layouts),
            "n_windows": len(self._plane_window_ids),
            "converged": bool(result.success),
            "nfev": int(result.nfev),
            "full_stats": stats_after,
            "overlay_points_optim": overlay_points_optim,
        }

    def _setup_problem(
        self,
        camera_models: Dict[int, object],
        camera_states: Dict[int, dict],
        correspondences: List[dict],
        cam_to_window: Dict[int, int],
        window_planes: Dict[int, dict],
    ):
        self._cam_indices = sorted(camera_models.keys())
        self._camera_models = camera_models
        self._camera_states = camera_states
        self._correspondences = correspondences
        self._cam_to_window = cam_to_window
        self._window_planes = window_planes

        self._layouts = []
        for corr in correspondences:
            obs = corr.get("2d_per_cam", {})
            cam_ids = [cid for cid in sorted(obs.keys()) if cid in camera_models]
            if len(cam_ids) < 2:
                continue
            wids = sorted({
                cam_to_window[cid]
                for cid in cam_ids
                if cid in cam_to_window and cam_to_window[cid] in window_planes
            })
            self._layouts.append((corr, cam_ids, wids))

        self._cam_layout_indices = {cid: [] for cid in self._cam_indices}
        self._cam_uv_inputs = {cid: [] for cid in self._cam_indices}
        for layout_idx, (corr, cam_ids, _) in enumerate(self._layouts):
            obs = corr.get("2d_per_cam", {})
            for cid in cam_ids:
                uv = obs.get(cid)
                if uv is None:
                    continue
                self._cam_layout_indices[cid].append(layout_idx)
                self._cam_uv_inputs[cid].append(np.asarray(uv, dtype=np.float64).reshape(2))

    def _compute_overlay_points(self, x: np.ndarray) -> Dict[int, list]:
        """Compute optimized 3D point for each correspondence layout."""
        import pyopenlpt as lpt

        self._apply_params(x)
        rays_by_layout = self._compute_rays_batch(lpt)

        out = {}
        for layout_idx, (corr, _, _) in enumerate(self._layouts):
            corr_id = int(corr.get("corr_id", layout_idx))
            rays = list(rays_by_layout[layout_idx].values())
            if len(rays) < 2:
                continue

            X = self._triangulate_from_lines(rays)
            if X is None:
                continue
            X = np.asarray(X, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(X)):
                continue
            out[corr_id] = [float(X[0]), float(X[1]), float(X[2])]

        return out

    def _pack_params(self, camera_states: Dict[int, dict]) -> np.ndarray:
        x = []
        for cam_idx in self._cam_indices:
            st = camera_states[cam_idx]
            rvec = np.asarray(st["rvec"], dtype=np.float64).reshape(3)
            tvec = np.asarray(st["tvec"], dtype=np.float64).reshape(3)
            x.extend([rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]])
        return np.asarray(x, dtype=np.float64)

    def _states_from_x(self, x: np.ndarray, base_states: Dict[int, dict]) -> Dict[int, dict]:
        out = {}
        for i, cam_idx in enumerate(self._cam_indices):
            b = 6 * i
            rvec = np.asarray(x[b:b + 3], dtype=np.float64)
            tvec = np.asarray(x[b + 3:b + 6], dtype=np.float64)
            R, _ = cv2.Rodrigues(rvec)

            st = dict(base_states[cam_idx])
            st["rvec"] = rvec.copy()
            st["tvec"] = tvec.copy()
            st["R"] = R
            st["R_inv"] = R.T
            st["tvec_inv"] = (-R.T @ tvec.reshape(3, 1)).reshape(3)
            out[cam_idx] = st
        return out

    def _build_bounds(self, camera_states: Dict[int, dict]) -> Tuple[np.ndarray, np.ndarray]:
        lb = []
        ub = []
        for cam_idx in self._cam_indices:
            st = camera_states[cam_idx]
            rvec = np.asarray(st["rvec"], dtype=np.float64).reshape(3)
            tvec = np.asarray(st["tvec"], dtype=np.float64).reshape(3)
            lb.extend([
                rvec[0] - 0.1,
                rvec[1] - 0.1,
                rvec[2] - 0.1,
                tvec[0] - 50.0,
                tvec[1] - 50.0,
                tvec[2] - 50.0,
            ])
            ub.extend([
                rvec[0] + 0.1,
                rvec[1] + 0.1,
                rvec[2] + 0.1,
                tvec[0] + 50.0,
                tvec[1] + 50.0,
                tvec[2] + 50.0,
            ])
        return np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)

    def _make_pt2d(self, lpt, uv):
        return lpt.Pt2D(float(uv[0]), float(uv[1]))

    def _make_pt3d(self, lpt, arr):
        return lpt.Pt3D(float(arr[0]), float(arr[1]), float(arr[2]))

    def _plane_tangent_basis(self, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = np.asarray(n, dtype=np.float64).reshape(3)
        n = n / (np.linalg.norm(n) + 1e-12)
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        ref = z if abs(float(np.dot(n, z))) < 0.9 else x
        u = np.cross(ref, n)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u)
        v = v / (np.linalg.norm(v) + 1e-12)
        return u, v

    def _apply_params(self, x: np.ndarray):
        import pyopenlpt as lpt

        for i, cam_idx in enumerate(self._cam_indices):
            b = 6 * i
            cam = self._camera_models[cam_idx]
            st = self._camera_states[cam_idx]
            rvec = self._make_pt3d(lpt, x[b:b + 3])
            tvec = self._make_pt3d(lpt, x[b + 3:b + 6])

            cam.setPinplateExtrinsics(rvec, tvec)
            cam.commitPinplateUpdate(bool(st.get("is_active", True)), float(st.get("max_intensity", 255.0)))

    def _apply_plane_params(self, x: np.ndarray):
        import pyopenlpt as lpt

        for i, wid in enumerate(self._plane_window_ids):
            b = 3 * i
            dd = float(x[b])
            du = float(x[b + 1])
            dv = float(x[b + 2])

            base = self._plane_base[wid]
            pt0 = base["pt0"]
            n0 = base["n0"]
            u0 = base["u0"]
            v0 = base["v0"]

            n_new = n0 + du * u0 + dv * v0
            n_norm = np.linalg.norm(n_new)
            if n_norm <= 1e-12:
                n_new = n0.copy()
            else:
                n_new = n_new / n_norm

            pt_new = pt0 + dd * n_new

            self._window_planes[wid]["plane_pt"] = np.asarray(pt_new, dtype=np.float64).reshape(3)
            self._window_planes[wid]["plane_n"] = np.asarray(n_new, dtype=np.float64).reshape(3)

            ref = self._window_refraction.get(wid, None)
            if not ref:
                continue

            refract_array = [float(v) for v in ref.get("refract_array", [])]
            w_array = [float(v) for v in ref.get("w_array", [])]
            proj_tol = float(ref.get("proj_tol", 1e-6))
            proj_nmax = int(ref.get("proj_nmax", 1000))
            lr = float(ref.get("lr", 0.1))
            if len(refract_array) < 3 or len(w_array) < 1:
                continue

            pt_obj = self._make_pt3d(lpt, pt_new)
            n_obj = self._make_pt3d(lpt, n_new)
            for cam_idx in self._window_to_cams.get(wid, []):
                cam = self._camera_models.get(cam_idx, None)
                st = self._camera_states.get(cam_idx, {})
                if cam is None:
                    continue
                cam.setPinplateRefraction(
                    pt_obj,
                    n_obj,
                    refract_array,
                    w_array,
                    proj_tol,
                    proj_nmax,
                    lr,
                )
                cam.commitPinplateUpdate(
                    bool(st.get("is_active", True)),
                    float(st.get("max_intensity", 255.0)),
                )

    def _residual_core(self, rays_by_layout):
        res_ray = []
        barrier = []
        penalty_ray = 1e3 / self.sigma_ray_mm
        ray_sse = 0.0
        ray_n = 0
        violations = 0

        C_gate = self.alpha_side_gate * self._j_ref
        r_fix_const = np.sqrt(2.0 * C_gate)
        r_grad_const = np.sqrt(2.0 * self.beta_side_dir)

        for layout_idx, (corr, cam_ids, wids) in enumerate(self._layouts):
            rays_by_cam = rays_by_layout[layout_idx]
            rays = list(rays_by_cam.values())

            X = None
            if len(rays) >= 2:
                X = self._triangulate_from_lines(rays)

            for cam_idx in cam_ids:
                ray = rays_by_cam.get(cam_idx)
                if X is None or ray is None:
                    res_ray.append(penalty_ray)
                    continue
                o, d = ray
                dn = np.linalg.norm(d)
                if dn <= 1e-12:
                    res_ray.append(penalty_ray)
                    continue
                d = d / dn
                v = X - o
                proj = np.dot(v, d)
                perp = v - proj * d
                err = float(np.linalg.norm(perp))
                res_ray.append(err / self.sigma_ray_mm)
                ray_sse += err * err
                ray_n += 1

            r3d_mm = float(corr.get("r3d_mm", 0.0))
            for wid in wids:
                if X is None:
                    barrier.append(0.0)
                    barrier.append(0.0)
                    continue
                pl = self._window_planes[wid]
                n = np.asarray(pl["plane_n"], dtype=np.float64).reshape(3)
                P = np.asarray(pl["plane_pt"], dtype=np.float64).reshape(3)
                sX = float(np.dot(n, X - P))

                gap = (self.margin_side_mm + r3d_mm) - sX
                if gap > 0.0:
                    barrier.append(r_fix_const * (1.0 - np.exp(-gap / self.tau)))
                    barrier.append(r_grad_const * gap)
                    violations += 1
                else:
                    barrier.append(0.0)
                    barrier.append(0.0)

        if not res_ray and not barrier:
            return np.array([1e6], dtype=np.float64), {"ray_rmse": 0.0, "violations": 0}

        ray_rmse = np.sqrt(ray_sse / max(ray_n, 1))
        return np.asarray(res_ray + barrier, dtype=np.float64), {
            "ray_rmse": float(ray_rmse),
            "violations": int(violations),
        }

    def _triangulate_from_lines(self, rays: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
        A = np.zeros((3, 3), dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        for o, d in rays:
            dn = np.linalg.norm(d)
            if dn <= 1e-12:
                continue
            d = d / dn
            M = np.eye(3) - np.outer(d, d)
            A += M
            b += M @ o
        try:
            X = np.linalg.solve(A, b)
            if not np.all(np.isfinite(X)):
                return None
            return X
        except Exception:
            return None

    def _compute_rays_batch(self, lpt):
        """Compute all LOS rays by camera in batch and map back to each layout."""
        rays_by_layout = [dict() for _ in self._layouts]

        for cam_idx in self._cam_indices:
            layout_ids = self._cam_layout_indices.get(cam_idx, [])
            uv_inputs = self._cam_uv_inputs.get(cam_idx, [])
            if not layout_ids or not uv_inputs:
                continue

            cam = self._camera_models[cam_idx]
            pts2d = [self._make_pt2d(lpt, uv) for uv in uv_inputs]
            results = cam.lineOfSightBatchStatus(pts2d)

            for layout_idx, status in zip(layout_ids, results):
                ok, line, _ = status
                if not ok:
                    continue
                o = np.array([line.pt[0], line.pt[1], line.pt[2]], dtype=np.float64)
                d = np.array([
                    line.unit_vector[0],
                    line.unit_vector[1],
                    line.unit_vector[2],
                ], dtype=np.float64)
                if np.all(np.isfinite(o)) and np.all(np.isfinite(d)):
                    rays_by_layout[layout_idx][cam_idx] = (o, d)

        return rays_by_layout

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        import pyopenlpt as lpt

        self._apply_params(x)
        self._nfev_count += 1

        rays_by_layout = self._compute_rays_batch(lpt)
        residuals, meta = self._residual_core(rays_by_layout)

        if self._nfev_count % 10 == 0:
            self._log(
                f"  [Refraction TRF] nfev={self._nfev_count} "
                f"ray_rmse={meta['ray_rmse']:.4f}mm violations={meta['violations']}"
            )

        return residuals

    def _residuals_plane(self, x: np.ndarray) -> np.ndarray:
        import pyopenlpt as lpt

        self._apply_plane_params(x)
        self._nfev_count += 1

        rays_by_layout = self._compute_rays_batch(lpt)
        residuals, meta = self._residual_core(rays_by_layout)

        if self._nfev_count % 10 == 0:
            self._log(
                f"  [Refraction Plane TRF] nfev={self._nfev_count} "
                f"ray_rmse={meta['ray_rmse']:.4f}mm violations={meta['violations']}"
            )

        return residuals

    def _compute_metrics(self, x: np.ndarray) -> dict:
        import pyopenlpt as lpt

        self._apply_params(x)

        ray_err = []
        proj_err = []
        min_sX = np.inf
        violations = 0

        rays_by_layout = self._compute_rays_batch(lpt)

        for layout_idx, (corr, cam_ids, wids) in enumerate(self._layouts):
            obs = corr.get("2d_per_cam", {})

            rays_by_cam = rays_by_layout[layout_idx]
            rays = list(rays_by_cam.values())

            if len(rays) < 2:
                continue

            X = self._triangulate_from_lines(rays)
            if X is None:
                continue

            for cam_idx in cam_ids:
                if cam_idx not in rays_by_cam:
                    continue
                o, d = rays_by_cam[cam_idx]
                dn = np.linalg.norm(d)
                if dn <= 1e-12:
                    continue
                d = d / dn
                v = X - o
                proj = np.dot(v, d)
                perp = v - proj * d
                ray_err.append(float(np.linalg.norm(perp)))

                cam = self._camera_models[cam_idx]
                okp, uvp, _ = cam.projectStatus(self._make_pt3d(lpt, X), False)
                if okp:
                    eu = float(uvp[0] - obs[cam_idx][0])
                    ev = float(uvp[1] - obs[cam_idx][1])
                    proj_err.append(float(np.hypot(eu, ev)))

            r3d_mm = float(corr.get("r3d_mm", 0.0))
            for wid in wids:
                pl = self._window_planes[wid]
                n = np.asarray(pl["plane_n"], dtype=np.float64).reshape(3)
                P = np.asarray(pl["plane_pt"], dtype=np.float64).reshape(3)
                sX = float(np.dot(n, X - P))
                min_sX = min(min_sX, sX)
                if (self.margin_side_mm + r3d_mm) - sX > 0.0:
                    violations += 1

        ray_rmse = float(np.sqrt(np.mean(np.square(ray_err)))) if ray_err else 0.0
        proj_rmse = float(np.sqrt(np.mean(np.square(proj_err)))) if proj_err else 0.0

        stats = {
            "barrier_violations": int(violations),
            "min_sX": float(min_sX if np.isfinite(min_sX) else 0.0),
        }

        t = np.asarray(ray_err, dtype=np.float64)
        p = np.asarray(proj_err, dtype=np.float64)

        if t.size > 0:
            stats["triang_rmse"] = float(np.sqrt(np.mean(t * t)))
            stats["triang_mean"] = float(np.mean(t))
            stats["triang_std"] = float(np.std(t))
            stats["triang_max"] = float(np.max(t))
            stats["triang_tol"] = float(stats["triang_mean"] + 3.0 * stats["triang_std"])
        else:
            stats.update({
                "triang_rmse": 0.0,
                "triang_mean": 0.0,
                "triang_std": 0.0,
                "triang_max": 0.0,
                "triang_tol": 0.0,
            })

        if p.size > 0:
            stats["proj_rmse"] = float(np.sqrt(np.mean(p * p)))
            stats["proj_mean"] = float(np.mean(p))
            stats["proj_std"] = float(np.std(p))
            stats["proj_max"] = float(np.max(p))
            stats["proj_tol"] = float(stats["proj_mean"] + 3.0 * stats["proj_std"])
        else:
            stats.update({
                "proj_rmse": 0.0,
                "proj_mean": 0.0,
                "proj_std": 0.0,
                "proj_max": 0.0,
                "proj_tol": 0.0,
            })

        return stats
