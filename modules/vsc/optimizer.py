"""
VSC Optimizer (Most Stable + Smooth-Gated Residual + Full Stats)

Key design (stability-first):
- Outer loop stages (e.g., 3 stages)
- At the start of each stage: update 3D point cache (DLT + 1 GN iteration)
- Inside scipy residual: DO NOT re-triangulate, DO NOT GN refine (use cached points only)
- Residual gating:
    * Hard gate for global validity: (views>=2) & finite 3D
    * Smooth gate for depth: sigmoid((Z - Z0)/Zsoft) to avoid discontinuities
- Cost (whitened):
    * point-to-ray distance / sigma_ray
    * reprojection x,y pixel residuals / sigma_pix
  Huber loss handled by scipy.

This version also computes correct:
- ProjErr mean/std/max/tol (px)
- TriangErr mean/std/max/tol (mm)

Notes:
- Units: assumes your camera tvec and 3D points are in mm (consistent with your code).
- Z0 should match your previous hard threshold (50 mm).
"""

import numpy as np
import cv2
from typing import List, Tuple, Callable, Optional, Dict
from scipy.optimize import least_squares


class VSCOptimizer:
    def __init__(self,
                 max_nfev: int = 30000,
                 ftol: float = 1e-7,
                 xtol: float = 1e-7,
                 f_scale: float = 1.5,
                 sigma_pix: float = 0.8,
                 z0_mm: float = 50.0,
                 zsoft_mm: float = 15.0):
        self.max_nfev = int(max_nfev)
        self.ftol = float(ftol)
        self.xtol = float(xtol)
        self.f_scale = float(f_scale)
        self.sigma_pix = float(sigma_pix)

        # Smooth depth gating parameters
        self.z0_mm = float(z0_mm)
        self.zsoft_mm = float(zsoft_mm)

        self.log_callback: Optional[Callable[[str], None]] = None

        # [rvec(3), tvec(3), f,cx,cy,k1,k2]
        self.n_cam_params = 11
        self.n_dist = 0

        # cache
        self._cached_x: Optional[np.ndarray] = None
        self._cached_pts: Optional[np.ndarray] = None
        self._valid_pt_mask: Optional[np.ndarray] = None

        # runtime
        self._nfev_count = 0

        # data holders (set in optimize_all_cameras)
        self._cam_id_map = None
        self._cam_indices = None
        self._n_cams = 0
        self._img_size = None
        self._observations = None
        self._cam_obs_map = None

    # ------------------- Logging -------------------

    def set_log_callback(self, callback: Callable[[str], None]):
        self.log_callback = callback

    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    # ------------------- Utilities -------------------

    def _count_distortion_params(self, cameras: Dict[int, dict]) -> int:
        max_dist = 0
        for cam_params in cameras.values():
            dist = cam_params.get('dist', np.zeros(5))
            if len(dist) > 0 and abs(dist[0]) > 1e-10:
                max_dist = max(max_dist, 1)
            if len(dist) > 1 and abs(dist[1]) > 1e-10:
                max_dist = max(max_dist, 2)
        return max_dist

    def _sigmoid_gate(self, z: np.ndarray, z0: float, z_soft: float) -> np.ndarray:
        """
        Smooth gate in (0,1): ~0 when z << z0, ~1 when z >> z0.
        z_soft controls transition width (mm). Larger => smoother.
        """
        z = np.asarray(z, dtype=np.float64)
        a = (z - z0) / max(z_soft, 1e-9)
        a = np.clip(a, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-a))

    # ------------------- Public API -------------------

    def optimize_all_cameras(self,
                             cameras: Dict[int, dict],
                             correspondences: List[dict],
                             img_size: Tuple[int, int]) -> Tuple[Dict[int, dict], dict]:

        n_cams = len(cameras)
        n_pts = len(correspondences)
        cam_indices = sorted(cameras.keys())
        cam_id_map = {ext: i for i, ext in enumerate(cam_indices)}

        if n_pts < 10:
            return cameras.copy(), {
                'triang_before': 0.0, 'triang_after': 0.0,
                'proj_before': 0.0, 'proj_after': 0.0,
                'n_points': n_pts, 'converged': False
            }

        # distortion count
        self.n_dist = self._count_distortion_params(cameras)
        self._log(f"  Using {self.n_dist} distortion parameter(s)")

        # store data
        self._cam_id_map = cam_id_map
        self._cam_indices = cam_indices
        self._n_cams = n_cams
        self._img_size = img_size

        # observations list: each entry is dict {cam_idx: (u,v)}
        self._observations = [corr['2d_per_cam'] for corr in correspondences]

        # cam-wise obs map: {internal_cam_i: (pt_indices, uv_array)}
        self._cam_obs_map = {}
        for i, cam_idx in enumerate(cam_indices):
            indices = []
            uvs = []
            for pt_idx, obs in enumerate(self._observations):
                if cam_idx in obs:
                    indices.append(pt_idx)
                    uvs.append(obs[cam_idx])
            if indices:
                self._cam_obs_map[i] = (np.array(indices, dtype=int),
                                        np.array(uvs, dtype=np.float64))

        # valid point mask: seen by >=2 cameras
        view_counts = np.array([len(c['2d_per_cam']) for c in correspondences], dtype=int)
        self._valid_pt_mask = view_counts >= 2
        self._log(f"  Valid Points (Views >= 2): {int(np.sum(self._valid_pt_mask))} / {n_pts}")

        # init params
        x0 = self._cameras_to_params(cameras)
        self._nfev_count = 0

        # initial cache update
        self._update_point_cache(x0)

        stats_before = self._compute_both_errors(x0)
        self._log(f"  Initial: TriangErr={stats_before['triang_rmse']:.4f}mm, ProjErr={stats_before['proj_rmse']:.4f}px")

        n_outer_iters = 3
        current_x = x0

        self._log(f"  Starting Iterative Optimization ({n_outer_iters} stages, cached points within each stage)...")
        self._log(f"  Smooth Z gate: Z0={self.z0_mm:.1f}mm, Zsoft={self.zsoft_mm:.1f}mm")

        final_result = None

        for k in range(n_outer_iters):
            # 0) update point cache ONCE per stage (MOST STABLE)
            self._log(f"  [Iter {k+1}/{n_outer_iters}] Updating point cache (DLT+GN)...")
            self._update_point_cache(current_x)

            # 1) bounds centered around current solution
            current_cameras = self._params_to_cameras(current_x, cameras)
            lb, ub = self._build_bounds(current_cameras)

            self._log(f"  [Iter {k+1}/{n_outer_iters}] Running TRF (points fixed in this stage)...")

            result = least_squares(
                self._residuals_cached_points,
                current_x,
                method='trf',
                loss='huber',
                f_scale=self.f_scale,
                bounds=(lb, ub),
                max_nfev=self.max_nfev,
                ftol=self.ftol,
                xtol=self.xtol,
                x_scale='jac',
                verbose=0
            )

            current_x = result.x
            final_result = result

            # update cache once more at stage end (recommended for consistent stage metrics)
            self._update_point_cache(current_x)

            stats_intermediate = self._compute_both_errors(current_x)
            self._log(f"  [Iter {k+1}] Result: TriangErr={stats_intermediate['triang_rmse']:.4f}mm, ProjErr={stats_intermediate['proj_rmse']:.4f}px")

        x_opt = current_x
        result = final_result

        stats_after = self._compute_both_errors(x_opt)
        self._log(f"  Final: TriangErr={stats_after['triang_rmse']:.4f}mm, ProjErr={stats_after['proj_rmse']:.4f}px")
        self._log(f"  Converged: {result.success}, nfev={result.nfev}, message={result.message}")

        # Log final tolerance-like stats (mean + 3*std) the way you used previously
        self._log(f"  Final ProjErr: {stats_after['proj_mean']:.4f} ± {stats_after['proj_std']:.4f} px (Tol: {stats_after['proj_tol']:.4f})")
        self._log(f"  Final TriangErr: {stats_after['triang_mean']:.4f} ± {stats_after['triang_std']:.4f} mm (Tol: {stats_after['triang_tol']:.4f})")

        optimized = self._params_to_cameras(x_opt, cameras)

        return optimized, {
            'triang_before': stats_before['triang_rmse'],
            'triang_after': stats_after['triang_rmse'],
            'proj_before': stats_before['proj_rmse'],
            'proj_after': stats_after['proj_rmse'],
            'n_points': n_pts,
            'converged': bool(result.success),
            'n_cams': n_cams,
            'nfev': int(result.nfev),
            'full_stats': stats_after
        }

    # ------------------- Parameter Packing -------------------

    def _cameras_to_params(self, cameras: Dict[int, dict]) -> np.ndarray:
        x = []
        for cam_idx in self._cam_indices:
            cam = cameras[cam_idx]
            K = cam['K']
            R = cam['R']
            tvec = np.asarray(cam['tvec'], dtype=np.float64).reshape(3)
            dist = cam.get('dist', np.zeros(5))

            rvec, _ = cv2.Rodrigues(R)
            rvec = rvec.flatten()

            f = float(K[0, 0])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            k1 = float(dist[0]) if len(dist) > 0 else 0.0
            k2 = float(dist[1]) if len(dist) > 1 else 0.0

            x.extend([rvec[0], rvec[1], rvec[2],
                      tvec[0], tvec[1], tvec[2],
                      f, cx, cy, k1, k2])
        return np.array(x, dtype=np.float64)

    def _params_to_cameras(self, x: np.ndarray, cameras: Dict[int, dict]) -> Dict[int, dict]:
        result = {}
        for i, cam_idx in enumerate(self._cam_indices):
            base = i * self.n_cam_params
            cp = x[base:base + self.n_cam_params]

            rvec = cp[0:3]
            tvec = cp[3:6]
            f, cx, cy = float(cp[6]), float(cp[7]), float(cp[8])
            k1, k2 = float(cp[9]), float(cp[10])

            R, _ = cv2.Rodrigues(rvec)
            K = np.array([[f, 0.0, cx],
                          [0.0, f,  cy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)

            out = cameras[cam_idx].copy()
            out['K'] = K
            out['R'] = R
            out['R_inv'] = R.T
            out['tvec'] = tvec
            out['tvec_inv'] = (-R.T @ tvec.reshape(3, 1)).flatten()
            out['rvec'] = rvec
            out['dist'] = dist
            result[cam_idx] = out
        return result

    def _build_bounds(self, cameras: Dict[int, dict]) -> Tuple[np.ndarray, np.ndarray]:
        lb = []
        ub = []
        for cam_idx in self._cam_indices:
            cam = cameras[cam_idx]
            K = cam['K']
            R = cam['R']
            tvec = np.asarray(cam['tvec'], dtype=np.float64).reshape(3)
            dist = cam.get('dist', np.zeros(5))

            rvec, _ = cv2.Rodrigues(R)
            rvec = rvec.flatten()

            f = float(K[0, 0])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            k1 = float(dist[0]) if len(dist) > 0 else 0.0
            k2 = float(dist[1]) if len(dist) > 1 else 0.0

            lb.extend([
                rvec[0] - 0.1, rvec[1] - 0.1, rvec[2] - 0.1,
                tvec[0] - 50.0, tvec[1] - 50.0, tvec[2] - 50.0,
                f * 0.95, cx - 50.0, cy - 50.0,
                k1 - (max(0.1, abs(k1) * 0.5) if self.n_dist >= 1 else 1e-10),
                k2 - (max(0.1, abs(k2) * 0.5) if self.n_dist >= 2 else 1e-10),
            ])
            ub.extend([
                rvec[0] + 0.1, rvec[1] + 0.1, rvec[2] + 0.1,
                tvec[0] + 50.0, tvec[1] + 50.0, tvec[2] + 50.0,
                f * 1.05, cx + 50.0, cy + 50.0,
                k1 + (max(0.1, abs(k1) * 0.5) if self.n_dist >= 1 else 1e-10),
                k2 + (max(0.1, abs(k2) * 0.5) if self.n_dist >= 2 else 1e-10),
            ])

        return np.array(lb, dtype=np.float64), np.array(ub, dtype=np.float64)

    # ------------------- Camera Parse -------------------

    def _parse_camera_params(self, x: np.ndarray) -> List[dict]:
        cam_params_list = []
        for i in range(self._n_cams):
            base = i * self.n_cam_params
            cp = x[base:base + self.n_cam_params]

            rvec = cp[0:3]
            tvec = cp[3:6]
            f, cx, cy = float(cp[6]), float(cp[7]), float(cp[8])

            k1 = float(cp[9]) if self.n_dist >= 1 else 0.0
            k2 = float(cp[10]) if self.n_dist >= 2 else 0.0

            K = np.array([[f, 0.0, cx],
                          [0.0, f,  cy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
            R, _ = cv2.Rodrigues(rvec)
            R_inv = R.T
            C = (-R_inv @ tvec.reshape(3, 1)).flatten()

            cam_params_list.append({
                'K': K, 'dist': dist,
                'R': R, 'R_inv': R_inv,
                'tvec': tvec, 'rvec': rvec,
                'C': C
            })
        return cam_params_list

    # ------------------- Point Cache (DLT + 1 GN) -------------------

    def _triangulate_dlt_normalized(self, cam_params_list: List[dict]) -> np.ndarray:
        """
        DLT triangulation in normalized undistorted plane.
        Only constraints from valid points (view>=2) are accumulated.
        """
        n_pts = len(self._observations)
        ATA = np.zeros((n_pts, 4, 4), dtype=np.float64)

        for i, cp in enumerate(cam_params_list):
            if i not in self._cam_obs_map:
                continue
            pt_indices, uvs = self._cam_obs_map[i]

            # filter to globally valid points
            m = self._valid_pt_mask[pt_indices]
            if not np.any(m):
                continue

            idx = pt_indices[m]
            uvs_m = uvs[m]

            pts_norm = cv2.undistortPoints(
                uvs_m.reshape(-1, 1, 2), cp['K'], cp['dist']
            ).reshape(-1, 2)

            P = np.hstack([cp['R'], cp['tvec'].reshape(3, 1)])  # normalized: [R|t]
            u = pts_norm[:, 0:1]
            v = pts_norm[:, 1:2]

            r1 = u * P[2:3, :] - P[0:1, :]
            r2 = v * P[2:3, :] - P[1:2, :]

            ATA[idx] += (r1[:, :, None] @ r1[:, None, :]) + (r2[:, :, None] @ r2[:, None, :])

        w, vec = np.linalg.eigh(ATA)
        Xh = vec[:, :, 0]  # (N,4)

        pts = np.zeros((n_pts, 3), dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            pts_tmp = Xh[:, :3] / Xh[:, 3:4]
            pts[:] = pts_tmp

        pts[~self._valid_pt_mask] = 0.0
        pts[~np.isfinite(pts).all(axis=1)] = 0.0
        return pts

    def _refine_points_gn_normalized(self, pts_3d: np.ndarray, cam_params_list: List[dict], iterations: int = 1) -> np.ndarray:
        """
        One small GN refinement in normalized plane (undistortPoints domain).
        Conservative: only update "good" points, clip step to avoid explosions.
        """
        n_pts = pts_3d.shape[0]
        current = pts_3d.copy()

        # precompute normalized observations per camera for valid points
        norm_obs_map = {}
        for i, cp in enumerate(cam_params_list):
            if i not in self._cam_obs_map:
                continue
            pt_indices, uvs = self._cam_obs_map[i]
            m = self._valid_pt_mask[pt_indices]
            if not np.any(m):
                continue
            idx = pt_indices[m]
            uvs_m = uvs[m]
            pts_norm = cv2.undistortPoints(
                uvs_m.reshape(-1, 1, 2), cp['K'], cp['dist']
            ).reshape(-1, 2)
            norm_obs_map[i] = (idx, pts_norm)

        for _ in range(iterations):
            H_all = np.zeros((n_pts, 3, 3), dtype=np.float64)
            b_all = np.zeros((n_pts, 3), dtype=np.float64)

            for i, cp in enumerate(cam_params_list):
                if i not in norm_obs_map:
                    continue
                idx, uvs_norm = norm_obs_map[i]
                Xw = current[idx]

                Xc = (cp['R'] @ Xw.T).T + cp['tvec']  # (M,3)
                x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]

                # consistent with depth threshold
                mask_z = z > self.z0_mm
                if not np.any(mask_z):
                    continue

                idx2 = idx[mask_z]
                x = x[mask_z]; y = y[mask_z]; z = z[mask_z]
                obs = uvs_norm[mask_z]

                invz = 1.0 / z
                invz2 = invz * invz

                proj_x = x * invz
                proj_y = y * invz
                rx = obs[:, 0] - proj_x
                ry = obs[:, 1] - proj_y

                # J_world = J_cam @ R
                R = cp['R']
                J_c00 = invz
                J_c02 = -x * invz2
                J_c11 = invz
                J_c12 = -y * invz2

                Jw0 = J_c00[:, None] * R[0] + J_c02[:, None] * R[2]
                Jw1 = J_c11[:, None] * R[1] + J_c12[:, None] * R[2]

                H_add = (Jw0[:, :, None] @ Jw0[:, None, :]) + (Jw1[:, :, None] @ Jw1[:, None, :])
                b_add = (Jw0 * rx[:, None]) + (Jw1 * ry[:, None])

                H_all[idx2] += H_add
                b_all[idx2] += b_add

            trace_H = H_all[:, 0, 0] + H_all[:, 1, 1] + H_all[:, 2, 2]
            good = (self._valid_pt_mask) & (trace_H > 1e-9) & np.isfinite(trace_H)
            if not np.any(good):
                break

            Hs = H_all[good].copy()
            bs = b_all[good].reshape(-1, 3, 1)

            # damping
            diag = np.diagonal(Hs, axis1=1, axis2=2)
            idxs = np.arange(3)
            Hs[:, idxs, idxs] += (1e-3 * (diag + 1e-9)) + 1e-6

            try:
                d = np.linalg.solve(Hs, bs).reshape(-1, 3)

                # clip step (mm)
                step = np.linalg.norm(d, axis=1)
                max_step = 0.5
                mclip = step > max_step
                if np.any(mclip):
                    d[mclip] *= (max_step / step[mclip])[:, None]

                current[good] += d
            except np.linalg.LinAlgError:
                break

        return current

    def _update_point_cache(self, x: np.ndarray):
        cam_params_list = self._parse_camera_params(x)

        pts = self._triangulate_dlt_normalized(cam_params_list)
        pts = self._refine_points_gn_normalized(pts, cam_params_list, iterations=1)

        pts[~self._valid_pt_mask] = 0.0
        pts[~np.isfinite(pts).all(axis=1)] = 0.0

        self._cached_pts = pts
        self._cached_x = x.copy()

    # ------------------- Residual (CACHED POINTS + SMOOTH GATE) -------------------

    def _residuals_cached_points(self, x: np.ndarray) -> np.ndarray:
        """
        Residual for least_squares:
        - Uses cached 3D points (fixed within current stage)
        - Smooth depth gate via sigmoid for Jacobian stability
        """
        self._nfev_count += 1

        if self._cached_pts is None:
            self._update_point_cache(x)

        pts_3d = self._cached_pts
        cam_params_list = self._parse_camera_params(x)

        w_proj = 1.0 / self.sigma_pix if self.sigma_pix > 1e-12 else 1.0

        residuals_list = []
        Z0 = self.z0_mm
        Zsoft = self.zsoft_mm

        for i, cp in enumerate(cam_params_list):
            if i not in self._cam_obs_map:
                continue

            pt_indices, pts_2d = self._cam_obs_map[i]
            Xw = pts_3d[pt_indices]

            # hard global gate: (views>=2) & finite
            gate_global = (self._valid_pt_mask[pt_indices] & np.isfinite(Xw).all(axis=1)).astype(np.float64)

            if gate_global.max() <= 0.0:
                residuals_list.append(np.zeros((len(pt_indices),), dtype=np.float64))
                residuals_list.append(np.zeros((2 * len(pt_indices),), dtype=np.float64))
                continue

            # depth and smooth gate
            Xc = (cp['R'] @ Xw.T).T + cp['tvec']
            Z = Xc[:, 2].astype(np.float64)

            gate_z = self._sigmoid_gate(Z, z0=Z0, z_soft=Zsoft)
            gate = gate_global * gate_z

            if gate.max() <= 0.0:
                residuals_list.append(np.zeros((len(pt_indices),), dtype=np.float64))
                residuals_list.append(np.zeros((2 * len(pt_indices),), dtype=np.float64))
                continue

            # adaptive sigma_ray ~ Z * (sigma_pix / f)
            Z_safe = np.maximum(Z, Z0)
            f_px = float(cp['K'][0, 0])
            sigma_ray = Z_safe * (self.sigma_pix / max(f_px, 1e-9))
            w_triang = 1.0 / np.maximum(sigma_ray, 1e-12)

            # point-to-ray distance in world
            pts_2d_undist = cv2.undistortPoints(
                pts_2d.reshape(-1, 1, 2), cp['K'], cp['dist']
            ).reshape(-1, 2)

            rays_cam = np.column_stack([pts_2d_undist, np.ones(len(pts_2d_undist))])
            rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
            rays_world = (cp['R_inv'] @ rays_cam.T).T

            v = Xw - cp['C']
            proj_len = np.sum(v * rays_world, axis=1, keepdims=True)
            perp = v - proj_len * rays_world
            dist = np.sqrt(np.sum(perp * perp, axis=1))

            res_triang = (dist * w_triang) * gate
            residuals_list.append(res_triang.astype(np.float64))

            # reprojection residual
            projected, _ = cv2.projectPoints(
                Xw.reshape(-1, 1, 3),
                cp['rvec'], cp['tvec'],
                cp['K'], cp['dist']
            )
            projected = projected.reshape(-1, 2)

            diffs = (projected - pts_2d).reshape(-1)  # (2N,)
            gate2 = np.repeat(gate, 2)
            res_proj = (diffs * w_proj) * gate2
            residuals_list.append(res_proj.astype(np.float64))

        residuals = np.concatenate(residuals_list) if residuals_list else np.array([], dtype=np.float64)

        if self._nfev_count % 50 == 0 and residuals.size > 0:
            stats = self._compute_both_errors(x)
            self._log(f"    nfev={self._nfev_count}: TriangErr={stats['triang_rmse']:.4f}mm, ProjErr={stats['proj_rmse']:.4f}px")

        return residuals

    # ------------------- Stats (Correct tol computation) -------------------

    def _compute_both_errors(self, x: np.ndarray) -> dict:
        """
        Compute unweighted error stats for:
        - Triangulation: point-to-ray distance (mm)
        - Reprojection: pixel distance (px)
        Returns rmse/mean/std/max/tol where tol = mean + 3*std.
        Uses cached points if cache corresponds exactly to x (common in our stage flow).
        """
        cam_params_list = self._parse_camera_params(x)

        use_cache = (self._cached_pts is not None) and (self._cached_x is not None) and np.allclose(self._cached_x, x)
        if use_cache:
            pts_3d = self._cached_pts
        else:
            pts_3d = self._triangulate_dlt_normalized(cam_params_list)
            pts_3d = self._refine_points_gn_normalized(pts_3d, cam_params_list, iterations=1)

        triang_dists = []
        reproj_dists = []

        for i, cp in enumerate(cam_params_list):
            if i not in self._cam_obs_map:
                continue

            pt_indices, pts_2d = self._cam_obs_map[i]
            Xw = pts_3d[pt_indices]

            # global validity
            mask_global = self._valid_pt_mask[pt_indices] & np.isfinite(Xw).all(axis=1)

            # depth validity (for reporting, keep it consistent with your previous rule)
            Xc = (cp['R'] @ Xw.T).T + cp['tvec']
            Z = Xc[:, 2]
            mask_valid = mask_global & (Z > self.z0_mm)

            if not np.any(mask_valid):
                continue

            # triang: point-to-ray
            pts_2d_undist = cv2.undistortPoints(
                pts_2d.reshape(-1, 1, 2), cp['K'], cp['dist']
            ).reshape(-1, 2)

            rays_cam = np.column_stack([pts_2d_undist, np.ones(len(pts_2d_undist))])
            rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
            rays_world = (cp['R_inv'] @ rays_cam.T).T

            v = Xw - cp['C']
            proj_len = np.sum(v * rays_world, axis=1, keepdims=True)
            perp = v - proj_len * rays_world
            dist = np.sqrt(np.sum(perp * perp, axis=1))
            triang_dists.append(dist[mask_valid])

            # reproj: pixel distance
            projected, _ = cv2.projectPoints(
                Xw.reshape(-1, 1, 3),
                cp['rvec'], cp['tvec'],
                cp['K'], cp['dist']
            )
            projected = projected.reshape(-1, 2)
            diffs = projected - pts_2d
            d = np.sqrt(np.sum(diffs * diffs, axis=1))
            reproj_dists.append(d[mask_valid])

        t = np.concatenate(triang_dists) if triang_dists else np.array([], dtype=np.float64)
        p = np.concatenate(reproj_dists) if reproj_dists else np.array([], dtype=np.float64)

        stats = {}

        if t.size > 0:
            stats['triang_rmse'] = float(np.sqrt(np.mean(t * t)))
            stats['triang_mean'] = float(np.mean(t))
            stats['triang_std'] = float(np.std(t))
            stats['triang_max'] = float(np.max(t))
            stats['triang_tol'] = float(stats['triang_mean'] + 3.0 * stats['triang_std'])
        else:
            stats.update({'triang_rmse': 0.0, 'triang_mean': 0.0, 'triang_std': 0.0, 'triang_max': 0.0, 'triang_tol': 0.0})

        if p.size > 0:
            stats['proj_rmse'] = float(np.sqrt(np.mean(p * p)))
            stats['proj_mean'] = float(np.mean(p))
            stats['proj_std'] = float(np.std(p))
            stats['proj_max'] = float(np.max(p))
            stats['proj_tol'] = float(stats['proj_mean'] + 3.0 * stats['proj_std'])
        else:
            stats.update({'proj_rmse': 0.0, 'proj_mean': 0.0, 'proj_std': 0.0, 'proj_max': 0.0, 'proj_tol': 0.0})

        return stats

    # ------------------- Legacy placeholder -------------------

    def optimize_camera(self, cam_params: dict, points_3d: np.ndarray, points_2d: np.ndarray) -> Tuple[dict, dict]:
        self._log("  Using legacy single-camera optimization (not implemented here)")
        return cam_params.copy(), {
            'rmse_before': 0.0, 'rmse_after': 0.0,
            'n_points': len(points_3d), 'converged': False
        }
