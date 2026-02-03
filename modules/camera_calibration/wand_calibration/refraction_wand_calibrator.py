import os
import builtins
import numpy as np
import cv2
import logging
import json
from datetime import datetime
from pathlib import Path
import random

try:
    import pyopenlpt as lpt
except ImportError:
    lpt = None

from .refractive_geometry import (
    build_pinplate_ray_cpp, closest_distance_rays, triangulate_point, 
    point_to_ray_dist, update_normal_tangent, camera_center,
    align_world_y_to_plane_intersection, update_cpp_camera_state
)

from .refractive_bootstrap import PinholeBootstrapP0, PinholeBootstrapP0Config, select_best_pair_via_precalib
from .refraction_calibration_BA import RefractiveBAOptimizer, RefractiveBAConfig
from scipy.optimize import least_squares


def _normalize_message(msg: str) -> str:
    leading = ""
    i = 0
    while i < len(msg) and msg[i] in " \n\t":
        leading += msg[i]
        i += 1
    core = msg[i:]

    tags = []
    while core.startswith("["):
        end = core.find("]")
        if end == -1:
            break
        tag = core[1:end].strip()
        if not tag:
            break
        if tag not in {"Refractive", "RefractiveCalib"}:
            tags.append(tag)
        core = core[end + 1:]
    core = core.lstrip()
    if tags:
        prefix = "/".join(tags)
        core = f"{prefix}: {core}" if core else prefix
    return leading + core


def _rc_print(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    file = kwargs.pop("file", None)
    msg = sep.join(str(a) for a in args)
    msg = _normalize_message(msg)
    builtins.print(msg, sep="", end=end, file=file, **kwargs)


print = _rc_print


class RefractiveCalibReporter:
    def section(self, title: str, width: int = 60):
        line = "=" * width
        builtins.print(line)
        builtins.print(_normalize_message(title))
        builtins.print(line)

    def header(self, title: str):
        builtins.print(_normalize_message(title))

    def info(self, message: str):
        builtins.print(_normalize_message(message))

    def warn(self, message: str):
        builtins.print(_normalize_message(message))

    def error(self, message: str):
        builtins.print(_normalize_message(message))

    def detail(self, message: str):
        builtins.print(_normalize_message(message))


class BootstrapCacheStore:
    def __init__(self, reporter: RefractiveCalibReporter):
        self.reporter = reporter

    def save(self, path, cam_params_by_id, err_px_by_id, active_cam_ids,
             chosen_pair, X_A_list, X_B_list, wand_len_mm, cam_ids, num_frames):
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

        self.reporter.info(f"[BOOT][CACHE] Saved to {path}")

    def load(self, path, wand_len_target, cam_ids_current, num_frames_current):
        path = Path(path)
        if not path.exists():
            self.reporter.info(f"[BOOT][CACHE] No cache found at {path}")
            return None

        try:
            with open(path, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            self.reporter.info(f"[BOOT][CACHE] Failed to load cache: {e}")
            return None

        if cache.get("version") != 1:
            self.reporter.info(f"[BOOT][CACHE] Cache version mismatch (expected 1, got {cache.get('version')})")
            return None

        cached_cam_ids = sorted(cache.get("cam_ids", []))
        current_cam_ids = sorted([int(c) for c in cam_ids_current])
        if cached_cam_ids != current_cam_ids:
            self.reporter.info("[BOOT][CACHE] Cache mismatch: cam_ids differ")
            self.reporter.detail(f"  Cached: {cached_cam_ids}")
            self.reporter.detail(f"  Current: {current_cam_ids}")
            return None

        cached_num_frames = cache.get("num_frames", 0)
        if cached_num_frames != num_frames_current:
            self.reporter.info(f"[BOOT][CACHE] Cache mismatch: num_frames ({cached_num_frames} != {num_frames_current})")
            return None

        cached_wand_len = cache.get("wand_len_mm", 0)
        if abs(cached_wand_len - wand_len_target) > 1e-6:
            self.reporter.info(f"[BOOT][CACHE] Cache mismatch: wand_len ({cached_wand_len} != {wand_len_target})")
            return None

        try:
            cam_params_by_id = {int(k): np.array(v, dtype=np.float64) for k, v in cache["cam_params_by_id"].items()}
            err_px_by_id = {int(k): float(v) for k, v in cache["err_px_by_id"].items()}
            active_cam_ids = [int(c) for c in cache["active_cam_ids"]]
            chosen_pair = (int(cache["chosen_pair"][0]), int(cache["chosen_pair"][1]))
            X_A_list = {int(k): np.array(v, dtype=np.float64) for k, v in cache["X_A_list"].items()}
            X_B_list = {int(k): np.array(v, dtype=np.float64) for k, v in cache["X_B_list"].items()}
        except Exception as e:
            self.reporter.info(f"[BOOT][CACHE] Failed to restore types: {e}")
            return None

        if chosen_pair[0] not in cam_params_by_id or chosen_pair[1] not in cam_params_by_id:
            self.reporter.info(f"[BOOT][CACHE] Cache mismatch: chosen_pair {chosen_pair} not in cam_params")
            return None

        if len(X_A_list) == 0 or len(X_B_list) == 0:
            self.reporter.info("[BOOT][CACHE] Cache mismatch: empty X_A_list or X_B_list")
            return None

        if len(X_A_list) != len(X_B_list):
            self.reporter.info(
                f"[BOOT][CACHE] Cache mismatch: X_A_list size ({len(X_A_list)}) != X_B_list size ({len(X_B_list)})"
            )
            return None

        sample_fids = list(X_A_list.keys())
        sample_lens = []
        for fid in sample_fids:
            if fid in X_B_list:
                dist = np.linalg.norm(X_B_list[fid] - X_A_list[fid])
                sample_lens.append(dist)

        if sample_lens:
            median_len = np.median(sample_lens)
            if median_len < wand_len_target / 2 or median_len > wand_len_target * 2:
                self.reporter.info(
                    f"[BOOT][CACHE] Cache sanity failed: median wand length {median_len:.2f}mm "
                    f"(expected ~{wand_len_target}mm)"
                )
                return None
            self.reporter.info(
                f"[BOOT][CACHE] Sanity check: median wand length = {median_len:.4f}mm "
                f"(target: {wand_len_target}mm)"
            )

        self.reporter.info(f"[BOOT][CACHE] Loaded OK from {path}")
        self.reporter.detail(f"  Cameras: {list(cam_params_by_id.keys())}")
        self.reporter.detail(f"  Frames: {len(X_A_list)}")
        self.reporter.detail(f"  Best pair: {chosen_pair}")

        return (cam_params_by_id, err_px_by_id, active_cam_ids, chosen_pair, X_A_list, X_B_list)


class ObservationBuilder:
    @staticmethod
    def collect(base, cam_to_window, reporter: RefractiveCalibReporter):
        wand_len = getattr(base, 'wand_length', 0)
        if wand_len <= 0:
            raise ValueError(f"CRITICAL: Invalid wand length: {wand_len}mm. Must be > 0.")

        dist_coeff_num = getattr(base, 'dist_coeff_num', 2)
        if dist_coeff_num not in [0, 1, 2]:
            raise ValueError(
                f"CRITICAL: Invalid distortion coefficient count: {dist_coeff_num}. Expected 0, 1, or 2."
            )

        if not base.wand_points:
            raise ValueError("No wand points detected or loaded.")

        source_points = base.wand_points_filtered if base.wand_points_filtered else base.wand_points
        frames_list = sorted(source_points.keys())
        reporter.info(
            f"[Refractive] Using {len(frames_list)} frames "
            f"(Filtered={base.wand_points_filtered is not None})"
        )

        active_cams = None
        for fid in frames_list:
            cams_in_frame = set(source_points[fid].keys())
            if active_cams is None:
                active_cams = cams_in_frame
            else:
                active_cams = active_cams.intersection(cams_in_frame)

        if not active_cams:
            raise ValueError("CRITICAL: No cameras found that consistently see the wand in all frames.")

        active_cams = sorted(list(active_cams))

        for cid in active_cams:
            if cid not in cam_to_window:
                raise ValueError(
                    f"CRITICAL: Camera {cid} is active in data but missing from Window Mapping table."
                )

        obs_data_A = {}
        obs_data_B = {}
        radii_small = {}
        radii_large = {}
        mask_A = {}
        mask_B = {}

        for fid in frames_list:
            obs_data_A[fid] = {}
            obs_data_B[fid] = {}
            radii_small[fid] = {}
            radii_large[fid] = {}
            mask_A[fid] = {}
            mask_B[fid] = {}
            frame_data = source_points[fid]

            for cid in active_cams:
                pts = frame_data[cid]

                uvA = None
                uvB = None
                rA = None
                rB = None

                for pt in pts:
                    if len(pt) < 5:
                        continue

                    label = pt[4]
                    pt_idx = pt[5] if len(pt) >= 6 else -1
                    radius_px = float(pt[2]) if len(pt) >= 3 else 0.0

                    if pt_idx != -1:
                        target_id = 0 if label == "Filtered_Small" else 1
                        if pt_idx != target_id:
                            reporter.warn(
                                f"  [Warning] Consistency Mismatch: Frame {fid} Cam {cid} "
                                f"{label} has PointIdx={pt_idx} (Expected {target_id}). Priority: Label."
                            )

                    if label == "Filtered_Small":
                        if uvA is not None:
                            raise ValueError(
                                f"CRITICAL: Duplicate 'Filtered_Small' in Frame {fid} Cam {cid}."
                            )
                        uvA = pt[:2]
                        rA = radius_px
                    elif label == "Filtered_Large":
                        if uvB is not None:
                            raise ValueError(
                                f"CRITICAL: Duplicate 'Filtered_Large' in Frame {fid} Cam {cid}."
                            )
                        uvB = pt[:2]
                        rB = radius_px

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

    @staticmethod
    def prepare_for_bootstrap(base, cam_to_window, reporter: RefractiveCalibReporter):
        dataset = ObservationBuilder.collect(base, cam_to_window, reporter)

        observations = {}
        for fid in dataset['frames']:
            observations[fid] = {}
            for cid in dataset['cam_ids']:
                uvA = dataset['obsA'][fid].get(cid)
                uvB = dataset['obsB'][fid].get(cid)
                if uvA is not None and uvB is not None:
                    observations[fid][cid] = (np.array(uvA), np.array(uvB))

        return observations


class CppSyncAdapter:
    @staticmethod
    def build_update_kwargs(cam_params, window_planes, window_media, cam_to_window, cam_id):
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
    def apply(cams_cpp, cam_id, update_kwargs):
        if update_kwargs:
            update_cpp_camera_state(cams_cpp[cam_id], **update_kwargs)


class PlaneInitializer:
    @staticmethod
    def init_window_planes_from_cameras(
        cam_params,
        cam_to_window,
        window_media,
        err_px,
        verbose=False,
        X_A_list=None,
        X_B_list=None,
        active_cam_ids=None,
    ):
        if active_cam_ids is None:
            active_cam_ids = list(cam_params.keys())
        active_cam_set = set(active_cam_ids)

        X_mids = []
        if X_A_list and X_B_list:
            sample_fids = list(X_A_list.keys())[:500]
            for fid in sample_fids:
                if fid in X_B_list:
                    X_mid = 0.5 * (np.array(X_A_list[fid]) + np.array(X_B_list[fid]))
                    X_mids.append(X_mid)

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

            centers = {}
            for cid in cams_used:
                p = cam_params[cid]
                rvec, tvec = p[0:3], p[3:6]
                R, _ = cv2.Rodrigues(rvec)
                centers[cid] = -R.T @ tvec

            C_mean = np.mean([centers[c] for c in cams_used], axis=0)
            if verbose:
                print(f"  C_mean (cams_used) = {C_mean.round(2)}")

            if X_mids:
                X_mean = np.mean(X_mids, axis=0)
                if verbose:
                    print(f"  X_mean (bootstrap midpoints) = {X_mean.round(2)}")

                optical_axes = []
                ref_dir_guide = X_mean - C_mean
                ref_dir_guide /= (np.linalg.norm(ref_dir_guide) + 1e-12)

                for cid in cams_used:
                    p = cam_params[cid]
                    R, _ = cv2.Rodrigues(p[0:3])
                    axis = (R.T @ np.array([0, 0, 1.0])).reshape(3,)
                    axis /= (np.linalg.norm(axis) + 1e-12)
                    optical_axes.append(axis)

                optical_axes = np.array(optical_axes)

                mean_internal = np.mean(optical_axes, axis=0)
                if np.linalg.norm(mean_internal) < 1e-6:
                    mean_internal = optical_axes[0]
                mean_internal /= (np.linalg.norm(mean_internal) + 1e-12)

                aligned_axes = []
                for axis in optical_axes:
                    if np.dot(axis, mean_internal) < 0:
                        axis *= -1.0
                    aligned_axes.append(axis)
                aligned_axes = np.array(aligned_axes)

                mean_aligned = np.mean(aligned_axes, axis=0)
                na = np.linalg.norm(mean_aligned)
                if na < 1e-6:
                    mean_aligned = mean_internal
                else:
                    mean_aligned /= (na + 1e-12)

                if np.dot(mean_aligned, ref_dir_guide) < 0:
                    aligned_axes *= -1.0

                mean_rough = np.mean(aligned_axes, axis=0)
                norm_mr = np.linalg.norm(mean_rough)
                if norm_mr < 1e-6:
                    mean_rough = aligned_axes[0]
                else:
                    mean_rough /= norm_mr

                weights = []
                for axis in aligned_axes:
                    w = max(0.0, np.dot(axis, mean_rough))**2
                    weights.append(w)

                weights = np.array(weights)
                w_sum = np.sum(weights)
                if w_sum < 1e-12:
                    n0 = np.mean(aligned_axes, axis=0)
                else:
                    n0 = np.average(aligned_axes, axis=0, weights=weights)

                n_win = n0 / (np.linalg.norm(n0) + 1e-12)

                dot_val = np.dot(n_win, ref_dir_guide)
                angle_deg = np.degrees(np.arccos(np.clip(dot_val, -1.0, 1.0)))

                if verbose:
                    print(f"  n_win (Robust Weighted) = {n_win.round(4)}")
                    print(f"  [DIAG] Angle vs (X-C): {angle_deg:.2f} deg")
                    print(f"  [CONVENTION] n_win points: camera -> object (away from camera)")

                depths = []
                for cid in cams_used:
                    C_cam = centers[cid]
                    for X_mid in X_mids[:200]:
                        depth = np.dot(n_win, X_mid - C_cam)
                        if depth > 0:
                            depths.append(depth)

                depth_med = np.median(depths) if depths else np.linalg.norm(X_mean - C_mean)

                s_X_vals = [np.dot(n_win, X - C_mean) for X in X_mids[:100]]
                obj_pct = np.mean(np.array(s_X_vals) > 0) * 100.0 if s_X_vals else 0.0
                if verbose:
                    print(f"  [DIR] Using C_mean anchor (NOT plane-side check): {obj_pct:.1f}% points have dot(n, X-C_mean)>0.")
                    print(f"  depth_med = {depth_med:.1f} mm (from {len(depths)} samples)")

                thick_mm = window_media.get(wid, {}).get('thickness', 10.0)
                d0_min = max(10.0 * thick_mm, 0.02 * depth_med, 20.0)
                d0_max = min(1.2 * depth_med, 2500.0)
                d0_init = 0.4 * depth_med
                d0_mm = np.clip(d0_init, d0_min, d0_max)

                if verbose:
                    print(f"  d0_init = 0.4 * {depth_med:.1f} = {d0_init:.1f} mm")
                    print(f"  d0 range = [{d0_min:.1f}, {d0_max:.1f}], d0_final = {d0_mm:.1f} mm")

                plane_pt = C_mean + n_win * d0_mm
                if verbose:
                    print(f"  plane_pt = C_mean + n*d0 = {plane_pt.round(2)}")

            else:
                print(f"  [WARNING] No bootstrap points - using fallback")
                n_win = np.array([0.0, 0.0, 1.0])
                d0_mm = 100.0
                depth_med = 200.0
                plane_pt = C_mean + n_win * d0_mm
                thick_mm = window_media.get(wid, {}).get('thickness', 10.0)

            def compute_score(n_test):
                cams_ok = 0
                for cid in cams_used:
                    s = np.dot(n_test, centers[cid] - plane_pt)
                    if s < 0:
                        cams_ok += 1

                obj_ok = 0
                if X_mids:
                    for X_mid in X_mids[:200]:
                        sX = np.dot(n_test, X_mid - plane_pt)
                        if sX > 0:
                            obj_ok += 1
                    obj_pct = 100.0 * obj_ok / min(len(X_mids), 200)
                else:
                    obj_pct = 100.0

                return cams_ok, obj_pct

            cams_ok_pos, obj_pct_pos = compute_score(n_win)
            cams_ok_neg, obj_pct_neg = compute_score(-n_win)

            print(f"\n[WIN_ORIENT] Win {wid}: Checking orientations (C++ convention: s(C)<0, s(X)>0)...")
            if verbose:
                print(f"  n_win:  cams_ok={cams_ok_pos}/{len(cams_used)}, obj_pct={obj_pct_pos:.1f}%")
                print(f"  -n_win: cams_ok={cams_ok_neg}/{len(cams_used)}, obj_pct={obj_pct_neg:.1f}%")

            if cams_ok_pos == len(cams_used) and obj_pct_pos >= 50.0:
                print(f"  [OK] Using n_win (all cams camera-side, {obj_pct_pos:.1f}% objects object-side)")
            elif cams_ok_neg == len(cams_used) and obj_pct_neg >= 50.0:
                n_win = -n_win
                print(f"  [OK] Using -n_win (all cams camera-side, {obj_pct_neg:.1f}% objects object-side)")
            elif cams_ok_pos == len(cams_used):
                print(f"  [WARN] Using n_win (all cams OK, but only {obj_pct_pos:.1f}% objects object-side)")
            elif cams_ok_neg == len(cams_used):
                n_win = -n_win
                if verbose:
                    print(f"  [WARN] Using -n_win (all cams OK, but only {obj_pct_neg:.1f}% objects object-side)")
            else:
                if cams_ok_pos >= cams_ok_neg:
                    n_choice, cams_ok_choice = n_win, cams_ok_pos
                else:
                    n_win = -n_win
                    cams_ok_choice = cams_ok_neg
                print(f"  [ERROR] Cannot satisfy all cams! Best: {cams_ok_choice}/{len(cams_used)} cams camera-side")
                print(f"  This may indicate: (1) cam_to_window mapping error, or (2) cameras on opposite sides of window")

            print(f"\n[WIN_SANITY] Win {wid}: s = dot(n_win, P - plane_pt)")

            if verbose:
                for cid in cams_used:
                    s = np.dot(n_win, centers[cid] - plane_pt)
                    if s >= 0:
                        print(f"  [WARN] Cam {cid} s={s:.2f} mm >= 0 (WRONG side)")

            s_cams = [np.dot(n_win, centers[c] - plane_pt) for c in cams_used]
            if s_cams:
                min_s_cam = min(s_cams)
                max_s_cam = max(s_cams)
                print(f"  [WIN_SANITY][STATS] cams_used s(C): min={min_s_cam:.2f} mm, max={max_s_cam:.2f} mm (expect <0)")

            obj_pct_plane = 0.0
            if X_mids:
                s_objs = [np.dot(n_win, X - plane_pt) for X in X_mids[:200]]
                obj_pct_plane = np.mean(np.array(s_objs) > 0) * 100.0
                print(f"  [WIN_SANITY][STATS] objects on object-side (s>0): {obj_pct_plane:.1f}%")

            dot_n_to_plane = np.dot(n_win, plane_pt - C_mean)
            print(f"\n  [KEY INVARIANT] dot(n, plane_pt - C_mean) = {dot_n_to_plane:.2f} (MUST be > 0)")
            if dot_n_to_plane <= 0:
                print(f"  [CRITICAL] KEY INVARIANT VIOLATED! Normal direction is WRONG!")

            window_planes[wid] = {
                'plane_pt': plane_pt,
                'plane_n': n_win,
                'thick_mm': thick_mm,
                'initialized': True
            }

            print(f"\n  n_win (final) = {n_win.round(4)}")
            print(f"  plane_pt (final) = {plane_pt.round(2)}")
            if verbose:
                print(f"  Plane at {d0_mm:.1f}mm from cameras, objects at ~{depth_med:.1f}mm")

        return window_planes


class CamFileExporter:
    @staticmethod
    def export_camfile_with_refraction(base, out_dir, cam_params, window_media, cam_to_window, window_planes=None):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        exported_files = []

        print(f"\n[Refractive][CAMFILE] Exporting PINPLATE files to {out_dir}")
        print(f"[Refractive][MAP] cam_to_window: {dict(cam_to_window)}")

        for cid in sorted(cam_params.keys()):
            p = cam_params[cid]
            rvec, tvec = p[0:3], p[3:6]
            f, cx, cy = p[6], p[7], p[8]
            k1, k2 = p[9], p[10]

            wid = cam_to_window[cid]
            media = window_media[wid]

            thick_mm = media.get('thickness', 0.0)

            if window_planes and wid in window_planes:
                wp = window_planes[wid]
                plane_pt = wp.get('plane_pt')
                plane_norm = wp.get('plane_n')

                data_valid = (
                    plane_pt is not None and plane_norm is not None and
                    np.shape(plane_pt) == (3,) and np.shape(plane_norm) == (3,) and
                    np.all(np.isfinite(plane_pt)) and np.all(np.isfinite(plane_norm)) and
                    0.99 <= np.linalg.norm(plane_norm) <= 1.01 and
                    thick_mm > 0.0
                )

                if data_valid:
                    plane_pt = np.array(plane_pt)
                    plane_norm = np.array(plane_norm)
                    if not wp.get('initialized', False):
                        wp['initialized'] = True
                else:
                    plane_pt = np.array([0.0, 0.0, 300.0])
                    plane_norm = np.array([0.0, 0.0, 1.0])
                    thick_mm = 10.0
                    print(
                        f"  [FALLBACK] Cam {cid} (Win {wid}): Corrupt plane data (NaN/thick<=0/non-unit n)! "
                        f"Using Z=300, t=10."
                    )
            else:
                plane_pt = np.array([0.0, 0.0, 300.0])
                plane_norm = np.array([0.0, 0.0, 1.0])
                if thick_mm <= 0.0:
                    thick_mm = 10.0
                print(
                    f"  [FALLBACK] Cam {cid} (Win {wid}): Missing plane data! Using default Z=300, t={thick_mm}."
                )

            n_air = media.get('n_air', media.get('n1', 1.0))
            n_win = media.get('n_window', media.get('n2', 1.49))
            n_obj = media.get('n_object', media.get('n3', 1.33))

            file_name = f"cam{cid}.txt"
            file_path = out_path / file_name

            R, _ = cv2.Rodrigues(rvec)
            R_inv = R.T
            t_vec = tvec.flatten()
            t_vec_inv = (-R_inv @ tvec).flatten()

            expected_t_inv = (-R_inv @ tvec).flatten()
            diff_t_inv = np.linalg.norm(t_vec_inv - expected_t_inv)
            assert diff_t_inv < 1e-6, f"Cam {cid}: t_vec_inv inconsistent! diff={diff_t_inv}"

            refract_array = [n_obj, n_win, n_air]
            w_array = [thick_mm]
            n_plate = len(w_array)

            P_closest = plane_pt.copy()
            P_farthest = P_closest + plane_norm * thick_mm
            plane_pt_export = P_farthest

            dot_check = np.dot(plane_norm, P_farthest - P_closest)
            print(f"  [CAMFILE][SANITY] Cam {cid}: Shifted plane point to farthest interface.")
            print(f"    P_closest: {P_closest}")
            print(f"    P_farthest (Export): {P_farthest}")
            print(f"    Shift along n: {dot_check:.4f} mm (Expected: {thick_mm:.4f} mm)")

            assert len(refract_array) == len(w_array) + 2, \
                f"Cam {cid}: len(refract_array)={len(refract_array)} != len(w_array)+2={len(w_array)+2}"
            assert len(refract_array) >= 3, f"Cam {cid}: refract_array too short"

            h, w = base.image_size if hasattr(base, 'image_size') else (800, 1280)
            img_size_str = f"{h},{w}"
            dist_coeff_str = f"{k1:.8g},{k2:.8g},0,0,0"
            rvec_placeholder = "0,0,0"
            refract_str = ",".join([f"{n:.4f}" for n in refract_array])
            w_str = ",".join([f"{w:.4f}" for w in w_array])

            with open(file_path, 'w') as f_out:
                f_out.write("PINPLATE\n")
                f_out.write("# (unused) error placeholders\n")
                f_out.write("0.0 0.0\n")
                f_out.write("\n")
                f_out.write("# Image Size: (n_row,n_col) as ONE token\n")
                f_out.write(f"{img_size_str}\n")
                f_out.write("\n")
                f_out.write("# Camera Matrix K (3x3)\n")
                f_out.write(f"{f:.8g} 0 {cx:.8g}\n")
                f_out.write(f"0 {f:.8g} {cy:.8g}\n")
                f_out.write("0 0 1\n")
                f_out.write("\n")
                f_out.write("# Distortion coefficients (ONE token, comma-separated)\n")
                f_out.write(f"{dist_coeff_str}\n")
                f_out.write("\n")
                f_out.write("# Rotation vector (unused placeholder, ONE token)\n")
                f_out.write(f"{rvec_placeholder}\n")
                f_out.write("\n")
                f_out.write("# Rotation Matrix R (world->camera)\n")
                for row in R:
                    f_out.write(f"{row[0]:.8g} {row[1]:.8g} {row[2]:.8g}\n")
                f_out.write("\n")
                f_out.write("# Inverse Rotation Matrix R_inv\n")
                for row in R_inv:
                    f_out.write(f"{row[0]:.8g} {row[1]:.8g} {row[2]:.8g}\n")
                f_out.write("\n")
                f_out.write("# Translation Vector T (world->camera)\n")
                f_out.write(f"{t_vec[0]:.8g} {t_vec[1]:.8g} {t_vec[2]:.8g}\n")
                f_out.write("\n")
                f_out.write("# Inverse Translation Vector (-R_inv*T) = camera center in world\n")
                f_out.write(f"{t_vec_inv[0]:.8g} {t_vec_inv[1]:.8g} {t_vec_inv[2]:.8g}\n")
                f_out.write("\n")
                f_out.write("# Refractive plane reference point plane.pt (Farthest Interface)\n")
                f_out.write(f"{plane_pt_export[0]:.8g} {plane_pt_export[1]:.8g} {plane_pt_export[2]:.8g}\n")
                f_out.write("\n")
                f_out.write("# Refractive plane normal plane.norm_vector (camera->object direction)\n")
                f_out.write(f"{plane_norm[0]:.8g} {plane_norm[1]:.8g} {plane_norm[2]:.8g}\n")
                f_out.write("\n")
                f_out.write("# refract_array (ONE token, comma-separated, farthest->nearest: obj->win->air)\n")
                f_out.write(f"# n_plate = {n_plate}\n")
                f_out.write(f"{refract_str}\n")
                f_out.write("\n")
                f_out.write("# w_array (ONE token, comma-separated, plate thicknesses in mm)\n")
                f_out.write(f"{w_str}\n")
                f_out.write("\n")
                f_out.write("# proj_tol\n")
                f_out.write("1e-6\n")
                f_out.write("# proj_nmax\n")
                f_out.write("50\n")
                f_out.write("# lr (learning rate)\n")
                f_out.write("0.1\n")
                f_out.write("\n")
                f_out.write("# --- BEGIN_REFRACTION_META ---\n")
                f_out.write(f"# VERSION=2\n")
                f_out.write(f"# CAM_ID={cid}\n")
                f_out.write(f"# WINDOW_ID={wid}\n")
                f_out.write(
                    f"# PLANE_PT_EXPORT=[{plane_pt_export[0]:.4f},{plane_pt_export[1]:.4f},{plane_pt_export[2]:.4f}]\n"
                )
                f_out.write(
                    f"# PLANE_N=[{plane_norm[0]:.6f},{plane_norm[1]:.6f},{plane_norm[2]:.6f}]\n"
                )
                f_out.write("# --- END_REFRACTION_META ---\n")

            print(f"  [CAMFILE] Cam {cid} -> Win {wid}")
            print(
                f"    plane_pt (Farthest): [{plane_pt_export[0]:.2f}, {plane_pt_export[1]:.2f}, {plane_pt_export[2]:.2f}]"
            )
            print(f"    plane_n:  [{plane_norm[0]:.6f}, {plane_norm[1]:.6f}, {plane_norm[2]:.6f}]")
            print(f"    refract:  [{n_obj:.4f}, {n_win:.4f}, {n_air:.4f}]")
            print(f"    thick:    {thick_mm:.4f} mm")
            print(f"    file:     {file_path}")

            exported_files.append(str(file_path))

        used_wids = sorted(set(cam_to_window[cid] for cid in cam_params.keys()))
        num_windows = len(window_media)
        print(f"\n[Refractive][MAP] used window ids: {used_wids}, num_windows={num_windows}")
        assert max(used_wids) < num_windows, f"Window ID {max(used_wids)} exceeds num_windows {num_windows}"
        assert len(used_wids) >= 1, "No windows used"

        return out_dir


class CppCameraFactory:
    @staticmethod
    def init_cams_cpp_in_memory(base, cam_params, window_media, cam_to_window, window_planes):
        cams_cpp = {}
        if lpt is None:
            print("[Warning] pyopenlpt not available. Skipping C++ initialization.")
            return cams_cpp

        print("\n[Refractive] Initializing C++ Camera objects in-memory...")

        for cid in sorted(cam_params.keys()):
            cam = lpt.Camera()
            try:
                cam._type = lpt.PINPLATE
            except AttributeError:
                cam._type = 2

            p = cam_params[cid]
            rvec, tvec = p[0:3], p[3:6]
            f, cx, cy = p[6], p[7], p[8]
            k1, k2 = p[9], p[10]

            wid = cam_to_window[cid]
            media = window_media[wid]
            thick_mm = media.get('thickness', 10.0)
            n_air = media.get('n_air', media.get('n1', 1.0))
            n_win = media.get('n_window', media.get('n2', 1.49))
            n_obj = media.get('n_object', media.get('n3', 1.33))

            wp = window_planes[wid]
            if 'plane_pt' not in wp or 'plane_n' not in wp:
                raise RuntimeError(
                    f"Cam {cid} (Win {wid}): Missing initialized plane data ('plane_pt' or 'plane_n') in window_planes."
                )

            pp = cam._pinplate_param
            pp.proj_tol = 1e-5
            pp.proj_nmax = 100
            pp.lr = 0.5
            pp.n_row = 0
            pp.n_col = 0

            pp.cam_mtx = lpt.MatrixDouble(3, 3, 0.0)
            pp.r_mtx = lpt.MatrixDouble(3, 3, 0.0)
            pp.r_mtx_inv = lpt.MatrixDouble(3, 3, 0.0)

            cam._pinplate_param = pp

            update_kwargs = CppSyncAdapter.build_update_kwargs(
                cam_params=cam_params,
                window_planes=window_planes,
                window_media=window_media,
                cam_to_window=cam_to_window,
                cam_id=cid,
            )
            CppSyncAdapter.apply({cid: cam}, cid, update_kwargs)

            image_size = base.image_size if hasattr(base, 'image_size') else (800, 1280)
            pp = cam._pinplate_param
            pp.n_row, pp.n_col = int(image_size[0]), int(image_size[1])
            pp.proj_tol = 1e-6
            pp.proj_nmax = 1000
            pp.lr = 0.1
            pp.refract_ratio_max = max(n_obj / n_win, n_obj / n_air, 1.0)

            cam._pinplate_param = pp
            cam.updatePinPlateParam()

            pp_check = cam._pinplate_param
            K_check = pp_check.cam_mtx
            if abs(K_check[0, 2] - cx) > 1e-4:
                print(f"[CRITICAL WARNING] Cam {cid} C++ Persistence Failed! CX_in={cx}, CX_out={K_check[0,2]}")

            cams_cpp[cid] = cam

        print(f"  Initialized {len(cams_cpp)} C++ Camera objects.")
        return cams_cpp


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
        self.reporter = RefractiveCalibReporter()
        self._bootstrap_cache = BootstrapCacheStore(self.reporter)
    
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
        self._bootstrap_cache.save(
            path=path,
            cam_params_by_id=cam_params_by_id,
            err_px_by_id=err_px_by_id,
            active_cam_ids=active_cam_ids,
            chosen_pair=chosen_pair,
            X_A_list=X_A_list,
            X_B_list=X_B_list,
            wand_len_mm=wand_len_mm,
            cam_ids=cam_ids,
            num_frames=num_frames,
        )
    
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
        return self._bootstrap_cache.load(
            path=path,
            wand_len_target=wand_len_target,
            cam_ids_current=cam_ids_current,
            num_frames_current=num_frames_current,
        )
    
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
    
        
        return window_planes, True

    def _collect_observations(self, cam_to_window):
        """
        Aggregate filtered wand points into a structured dict.
        Enforces strict 'fail-fast' consistency (Engineering Guardrail #5).
        """
        return ObservationBuilder.collect(self.base, cam_to_window, self.reporter)


    def _prepare_observations_for_bootstrap(self, cam_to_window: dict) -> dict:
        """
        Prepare observations in format for P0 bootstrap: {fid: {cid: (uvA, uvB)}}.
        
        Args:
            cam_to_window: Camera to window mapping (used for context)
            
        Returns:
            observations: {fid: {cid: (uvA, uvB)}}
        """
        return ObservationBuilder.prepare_for_bootstrap(self.base, cam_to_window, self.reporter)


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
        return PlaneInitializer.init_window_planes_from_cameras(
            cam_params=cam_params,
            cam_to_window=cam_to_window,
            window_media=window_media,
            err_px=err_px,
            verbose=self.verbose,
            X_A_list=X_A_list,
            X_B_list=X_B_list,
            active_cam_ids=active_cam_ids,
        )
        
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
        return CamFileExporter.export_camfile_with_refraction(
            base=self.base,
            out_dir=out_dir,
            cam_params=cam_params,
            window_media=window_media,
            cam_to_window=cam_to_window,
            window_planes=window_planes,
        )




    def _init_cams_cpp_in_memory(self, cam_params, window_media, cam_to_window, window_planes):
        """
        Initialize lpt.Camera objects directly in memory as PINPLATE models.
        Bypasses the 'export to file and reload' cycle.
        Ensures perfect geometric consistency with the C++ engine.
        """
        return CppCameraFactory.init_cams_cpp_in_memory(
            base=self.base,
            cam_params=cam_params,
            window_media=window_media,
            cam_to_window=cam_to_window,
            window_planes=window_planes,
        )

    def _align_and_export(self, cam_params, window_media, cam_to_window, ba_optimizer, cams_cpp, dataset, out_path, verbosity):
        if verbosity >= 0:
            self.reporter.section("Exporting Final parameters to camFiles")

        if len(self.window_planes) >= 2:
            try:
                v_fid = None
                v_X_raw = None
                if dataset['frames']:
                    v_fid = dataset['frames'][len(dataset['frames']) // 2]
                    v_rays = []
                    for cid in cams_cpp:
                        if cid not in cam_params:
                            continue
                        wid = cam_to_window.get(cid)
                        uv = dataset['obsA'][v_fid].get(cid)
                        if uv is not None:
                            r = build_pinplate_ray_cpp(
                                cams_cpp[cid], uv, cam_id=cid, window_id=wid, frame_id=v_fid, endpoint="A"
                            )
                            if r.valid:
                                v_rays.append(r)

                    if len(v_rays) >= 2:
                        v_X_raw, _, _, _ = triangulate_point(v_rays)
                        if verbosity >= 1:
                            print(f"[Verification] Frame {v_fid} Raw 3D (Unaligned): {v_X_raw}")

                points_3d_for_align = []
                obsA = dataset.get('obsA', {})
                obsB = dataset.get('obsB', {})

                def _tri_all(obs_dict):
                    pts = []
                    for fid, cam_obs in obs_dict.items():
                        rays = []
                        for cid, uv in cam_obs.items():
                            if cid in cams_cpp:
                                wid = cam_to_window.get(cid)
                                r = build_pinplate_ray_cpp(
                                    cams_cpp[cid], uv, cam_id=cid, window_id=wid, frame_id=fid, endpoint="?"
                                )
                                if r.valid:
                                    rays.append(r)
                        if len(rays) >= 2:
                            X, _, valid, _ = triangulate_point(rays)
                            if valid:
                                pts.append(X)
                    return pts

                points_3d_for_align.extend(_tri_all(obsA))
                points_3d_for_align.extend(_tri_all(obsB))

                if verbosity >= 1:
                    print(f"[Coordinate Alignment] Collected {len(points_3d_for_align)} points for centroid.")

                cam_params, self.window_planes, _, R_align, t_shift = align_world_y_to_plane_intersection(
                    self.window_planes, cam_params, points_3d=points_3d_for_align
                )
                if verbosity >= 1:
                    print("[Coordinate Alignment] Aligned Y-axis and Centered at Cloud Centroid.")

                if ba_optimizer:
                    ba_optimizer.cam_params = cam_params
                    ba_optimizer.window_planes = self.window_planes
                    ba_optimizer.window_media = window_media

                    for cid in cams_cpp:
                        if cid not in cam_params:
                            continue
                        update_kwargs = CppSyncAdapter.build_update_kwargs(
                            cam_params=cam_params,
                            window_planes=self.window_planes,
                            window_media=window_media,
                            cam_to_window=cam_to_window,
                            cam_id=cid,
                        )
                        CppSyncAdapter.apply(cams_cpp, cid, update_kwargs)

                    if verbosity >= 1:
                        print("[Coordinate Alignment] Synced C++ objects with aligned parameters.")

                    ba_optimizer.save_cache(out_path)
                    if verbosity >= 1:
                        print(f"[Coordinate Alignment] Saved ALIGNED parameters to cache: {out_path}")

                if v_X_raw is not None and R_align is not None:
                    if t_shift is None:
                        t_shift = np.zeros(3)
                    v_X_expected = R_align @ (v_X_raw + t_shift)

                    v_rays_new = []
                    for cid in cams_cpp:
                        if cid not in cam_params:
                            continue
                        wid = cam_to_window.get(cid)
                        uv = dataset['obsA'][v_fid].get(cid)
                        if uv is not None:
                            r = build_pinplate_ray_cpp(
                                cams_cpp[cid], uv, cam_id=cid, window_id=wid, frame_id=v_fid, endpoint="A"
                            )
                            if r.valid:
                                v_rays_new.append(r)

                    if len(v_rays_new) >= 2:
                        v_X_recalc, _, _, _ = triangulate_point(v_rays_new)

                        diff = np.linalg.norm(v_X_recalc - v_X_expected)
                        print(f"\n[Verification] End-to-End Consistency Check (Frame {v_fid}):")
                        print(f"  X_raw (Unaligned) : {v_X_raw}")
                        print(f"  X_expected (R*X)  : {v_X_expected}")
                        print(f"  X_recalc (C++ Tri): {v_X_recalc}")
                        print(f"  Difference        : {diff:.6f} mm")

                        if diff < 1e-4:
                            print("  [SUCCESS] C++ Engine is perfectly synced with Coordinate Alignment!")
                        else:
                            print("  [WARNING] Discrepancy detected! Check C++ parameter updates.")

            except Exception as e:
                print(f"[Coordinate Alignment] Warning: alignment failed: {e}")

        stored_dir = None
        try:
            stored_dir = self.export_camfile_with_refraction(
                out_path, cam_params, window_media, cam_to_window, self.window_planes
            )

            if verbosity >= 1:
                print(f"  Updated camFiles in: {stored_dir}")

            for cid in dataset['cam_ids']:
                cam_path = os.path.join(stored_dir, f"cam{cid}.txt")
                if lpt and os.path.exists(cam_path):
                    try:
                        cams_cpp[cid] = lpt.Camera(cam_path)
                    except Exception as e:
                        print(f"  [Warning] Cam {cid} reload failed: {e}")
        except Exception as e:
            print(f"  [Error] Export failed: {e}")

        return cam_params, cams_cpp, stored_dir

    def _build_labelled_rays(self, dataset, cam_params, cams_cpp, cam_to_window, active_cam_ids, inactive_cam_ids, verbosity):
        rays_db = {fid: {0: [], 1: []} for fid in dataset['frames']}
        invalid_reasons = {}
        total_obs = 0
        invalid_obs = 0

        self.reporter.section("Phase 2/3: Building Labelled Rays (C++ Kernel)")
        print(f"  Using active cameras only: {active_cam_ids}")

        for fid in dataset['frames']:
            for cid in active_cam_ids:
                wid = cam_to_window.get(cid)
                cam_obj = cams_cpp.get(cid)
                if not cam_obj:
                    continue

                uvA = dataset['obsA'][fid].get(cid)
                uvB = dataset['obsB'][fid].get(cid)

                if uvA is not None and uvB is not None:
                    dist_2d = np.linalg.norm(np.array(uvA) - np.array(uvB))
                    assert dist_2d > 1e-3, f"Frame {fid} Cam {cid}: Endpoints collapse in 2D (dist={dist_2d})"

                if uvA is not None:
                    total_obs += 1
                    if self.verbose and total_obs <= 10 and cid in cam_params:
                        u, v = uvA
                        cp_check = cam_params[cid]
                        f_check = cp_check[6]
                        cx_check = cp_check[7]
                        cy_check = cp_check[8]
                        u_norm_calc = (u - cx_check) / f_check
                        v_norm_calc = (v - cy_check) / f_check
                        print(f"[UV_NORM_CHECK] f={f_check:.1f}, cx={cx_check:.1f}, cy={cy_check:.1f}")
                        print(
                            f"  uv_px=({u:.1f},{v:.1f}) -> uv_norm_calc=({u_norm_calc:.4f}, {v_norm_calc:.4f})"
                        )

                    rayA = build_pinplate_ray_cpp(
                        cam_obj, uvA, cam_id=cid, window_id=wid, frame_id=fid, endpoint="A"
                    )
                    if rayA.valid:
                        rays_db[fid][0].append(rayA)
                    else:
                        invalid_obs += 1
                        invalid_reasons[rayA.reason or "unknown"] = invalid_reasons.get(rayA.reason or "unknown", 0) + 1

                if uvB is not None:
                    total_obs += 1
                    rayB = build_pinplate_ray_cpp(
                        cam_obj, uvB, cam_id=cid, window_id=wid, frame_id=fid, endpoint="B"
                    )
                    if rayB.valid:
                        rays_db[fid][1].append(rayB)
                    else:
                        invalid_obs += 1
                        invalid_reasons[rayB.reason or "unknown"] = invalid_reasons.get(rayB.reason or "unknown", 0) + 1

        print(f"  Total Observations: {total_obs}, Invalid Rays: {invalid_obs}")
        if inactive_cam_ids:
            print(f"  [Note] Skipped inactive cameras: {inactive_cam_ids}")

        if verbosity >= 0:
            print("\n[RAY_DIAG] Checking ray angle spread (Summary):")

        rays_grouped = {}
        for fid in list(rays_db.keys())[:200]:
            for k in [0, 1]:
                for r in rays_db[fid][k]:
                    key = (r.cam_id, k)
                    if key not in rays_grouped:
                        rays_grouped[key] = []
                    rays_grouped[key].append(r.d)

        total_suspicious = 0
        if verbosity >= 1:
            print(f"  {'Cam':<4} {'Pt':<2} {'N_rays':<8} {'AlignN':<8} {'P90_deg':<8} {'Status'}")
            print("  " + "-" * 50)

        for (cid, k), ds in sorted(rays_grouped.items()):
            if len(ds) < 10:
                continue

            ds = np.asarray(ds, dtype=float)
            mean_d = np.mean(ds, axis=0)
            norm_m = np.linalg.norm(mean_d)
            if norm_m < 1e-9:
                mean_d = ds[0]
            else:
                mean_d /= norm_m

            wid = cam_to_window.get(cid, 0)
            wp = self.window_planes.get(wid)
            n_win_r = np.array(wp['plane_n'], dtype=np.float64) if wp else np.array([0., 0., 1.], dtype=np.float64)
            n_win_r /= (np.linalg.norm(n_win_r) + 1e-12)
            align_to_n = abs(np.dot(mean_d, n_win_r))

            dots = np.clip(np.dot(ds, mean_d), -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(dots))

            ang_med = np.median(angles_deg)
            ang_p90 = np.percentile(angles_deg, 90)
            ang_p99 = np.percentile(angles_deg, 99)
            ang_max = np.max(angles_deg)

            label = "A" if k == 0 else "B"

            status = "OK"
            if ang_p90 < 0.05:
                status = "COLLAPSE"
                total_suspicious += 1
                if align_to_n > 0.999 and ang_p90 < 0.2:
                    status = "GLUED_N"

            if verbosity >= 1:
                print(f"  {cid:<4} {label:<2} {len(ds):<8} {align_to_n:<8.4f} {ang_p90:<8.4f} {status}")

            if verbosity >= 2:
                print(f"    Details: med={ang_med:.4f}, p99={ang_p99:.4f}, max={ang_max:.4f}")

        if total_suspicious > 0:
            print(f"  [SUMMARY] {total_suspicious} groups show strong signs of Ray Collapse. (Enable verbosity=1 for table)")
        elif verbosity >= 0:
            print("  [SUMMARY] All groups OK (p90 >= 0.05 deg)")

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
                        print(
                            f"  Frame {fid} Cam {r.cam_id} {label}: uv=({r.uv[0]:.1f}, {r.uv[1]:.1f}), "
                            f"o={r.o.round(1)}, d={r.d.round(4)}, dot(d,n)={dot_dn:.4f}"
                        )
                        probe_count += 1

        if 0 in rays_db and self.verbose:
            print("\n[Refractive][DEBUG] Frame 0 Ray Scale Diagnostics:")
            for k, label in [(0, "A"), (1, "B")]:
                print(f"  Endpoint {label}:")
                for r in rays_db[0][k]:
                    norm_o = np.linalg.norm(r.o)
                    wid_ray = cam_to_window.get(r.cam_id, 0)
                    n_win_ray = np.array(self.window_planes[wid_ray]['plane_n'])
                    dot_dn_final = np.dot(r.d, n_win_ray)
                    print(
                        f"    Cam {r.cam_id}: o=[{r.o[0]:.2f}, {r.o[1]:.2f}, {r.o[2]:.2f}], ||o||={norm_o:.1f}, "
                        f"d=[{r.d[0]:.4f}, {r.d[1]:.4f}, {r.d[2]:.4f}], dot(d,n)={dot_dn_final:.4f}"
                    )

        return rays_db, invalid_reasons, total_obs, invalid_obs

    def _run_triangulation_and_reports(
        self,
        dataset,
        rays_db,
        active_cam_ids,
        wand_len_target,
        cam_params,
        cam_to_window,
        window_media,
        stored_dir,
        out_path,
        invalid_reasons,
        total_obs,
        invalid_obs,
        loaded_cache,
        ba_optimizer,
        cams_cpp,
    ):
        self.reporter.section("Phase 4: Independent Multi-Camera Triangulation")

        tri_data = {}
        frames_valid_both = 0
        frames_valid_A = 0
        frames_valid_B = 0
        bad_frames = []
        px_min_2d_sep = 5.0

        for fid in dataset['frames']:
            res_A = triangulate_point(rays_db[fid][0])
            res_B = triangulate_point(rays_db[fid][1])

            XA, XB = res_A[0], res_B[0]
            validA, validB = res_A[2], res_B[2]

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
                        break

            is_collapse = False
            coord_check_fails = 0

            if validA and validB and not skip_2d_check:
                dist_3d = np.linalg.norm(XA - XB)
                if fid < 5 and self.verbose:
                    print(f"  Frame {fid:<3}: XA={XA.round(3)}, XB={XB.round(3)}, len_est={dist_3d:.4f}mm")

                    for k, (X_k, label) in enumerate([(XA, "A"), (XB, "B")]):
                        for ray in rays_db[fid][k]:
                            vec_to_X = X_k - ray.o
                            dot_fwd = np.dot(vec_to_X, ray.d)
                            if dot_fwd < 0:
                                coord_check_fails += 1
                                if self.verbose:
                                    print(
                                        f"    [COORD-CHECK FAIL] {label} Cam{ray.cam_id}: X.z={X_k[2]:.1f}, "
                                        f"o.z={ray.o[2]:.1f}, d.z={ray.d[2]:.4f}, dot={dot_fwd:.2f}"
                                    )

                if coord_check_fails > 0:
                    print(
                        f"  [WARNING] Frame {fid}: {coord_check_fails} rays failed coordinate check (dot < 0). "
                        "Enable verbose for details."
                    )

                if dist_3d < 0.2 * wand_len_target:
                    is_collapse = True
                    bad_frames.append({
                        'fid': fid, 'reason': 'collapse',
                        'dist_3d': dist_3d, 'XA': XA.tolist(), 'XB': XB.tolist(),
                        'num_rays_A': len(rays_db[fid][0]), 'num_rays_B': len(rays_db[fid][1])
                    })

            if is_collapse or skip_2d_check:
                validA = False
                validB = False

            tri_data[fid] = {
                "XA": XA, "condA": res_A[1], "validA": validA,
                "XB": XB, "condB": res_B[1], "validB": validB
            }
            if validA:
                frames_valid_A += 1
            if validB:
                frames_valid_B += 1
            if validA and validB:
                frames_valid_both += 1

        print(f"  Frames Valid (A/B/Both): {frames_valid_A} / {frames_valid_B} / {frames_valid_both}")

        total_frames = len(dataset['frames'])
        num_bad = len(bad_frames)
        bad_ratio = num_bad / total_frames if total_frames > 0 else 0

        if num_bad > 0:
            print(f"\n  [Outlier Report] Bad frames: {num_bad}/{total_frames} ({bad_ratio*100:.1f}%)")
            for bf in bad_frames[:5]:
                print(f"    Frame {bf['fid']}: {bf['reason']}")
            if num_bad > 5:
                print(f"    ... and {num_bad - 5} more")

            if bad_ratio > 0.10:
                print(f"\n  [WARNING] High bad frame ratio ({bad_ratio*100:.1f}% > 10%). Geometry may be degenerate.")

        self.reporter.section("Phase 5: Residual Analysis")

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
                X_k = f_tri["XA" if k == 0 else "XB"]
                if not f_tri["validA" if k == 0 else "validB"]:
                    continue
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
            if not arr:
                return {"median": 0.0, "mean": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
            arr = np.abs(np.array(arr))
            return {
                "median": float(np.median(arr)), "mean": float(np.mean(arr)),
                "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99)),
                "max": float(np.max(arr))
            }

        stats_ray = get_stats(r_ray_all)
        stats_len = get_stats(r_len_all)
        stats_len_est = get_stats(len_est_all)

        print(
            f"  Point-to-Ray Dist (mm): med={stats_ray['median']:.4f}, p90={stats_ray['p90']:.4f}, "
            f"max={stats_ray['max']:.4f}"
        )
        print(
            f"  Wand Length Error (mm): med={stats_len['median']:.4f}, p90={stats_len['p90']:.4f}, "
            f"max={stats_len['max']:.4f}"
        )

        frame_metrics.sort(key=lambda x: x['score'], reverse=True)
        print("\n  Worst 5 Frames (by Score):")
        for f in frame_metrics[:5]:
            print(
                f"    Frame {f['fid']:<5}: Score={f['score']:<8.4f} "
                f"RayMed={f['r_ray_med']:<8.4f} LenErr={f['r_len']:<8.4f}"
            )

        suggestions = [
            f['fid'] for f in frame_metrics
            if f['r_ray_med'] > stats_ray['p99'] or abs(f['r_len']) > stats_len['p99']
        ]

        sample_fids = random.sample(dataset['frames'], min(10, len(dataset['frames'])))
        samples = []
        for fid in sample_fids:
            f_tri = tri_data[fid]
            samples.append({
                "fid": fid, "XA": f_tri["XA"].tolist(), "XB": f_tri["XB"].tolist(),
                "raysA": [
                    {"cam": r.cam_id, "uv": r.uv, "o": r.o.tolist(), "d": r.d.tolist()}
                    for r in rays_db[fid][0]
                ],
                "raysB": [
                    {"cam": r.cam_id, "uv": r.uv, "o": r.o.tolist(), "d": r.d.tolist()}
                    for r in rays_db[fid][1]
                ]
            })

        report_path = Path(stored_dir).parent / "triangulation_report.json" if stored_dir else Path("triangulation_report.json")

        def safe_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            if isinstance(obj, (np.int64, np.int32, np.int16)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: safe_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [safe_json(x) for x in obj]
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
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

        self.reporter.info(f"Report exported to: {report_path}")
        self.reporter.info("Status: Round4 Intrinsic/Thickness Refinement Completed.")
        print("=" * 50 + "\n")

        report['window_planes'] = {
            wid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in pl.items()}
            for wid, pl in self.window_planes.items()
        }

        all_points_3d = []
        for fid, res in tri_data.items():
            if res['XA'] is not None:
                all_points_3d.extend(res['XA'].tolist())
            if res['XB'] is not None:
                all_points_3d.extend(res['XB'].tolist())
        dataset['points_3d'] = all_points_3d

        if tri_data and self.window_planes and cam_params:
            print("\n[ROUND4] Plane Side Verification (eps=0.05mm):")
            eps_mm = 0.05

            for wid, pl in self.window_planes.items():
                plane_pt = np.array(pl['plane_pt'])
                plane_n = np.array(pl['plane_n'])
                plane_n = plane_n / np.linalg.norm(plane_n)

                cams_for_win = [cid for cid, wid2 in cam_to_window.items() if wid2 == wid]

                for cid in cams_for_win:
                    if cid not in cam_params:
                        continue

                    cp = cam_params[cid]
                    rvec = cp[0:3]
                    tvec = cp[3:6]
                    R = cv2.Rodrigues(rvec)[0]
                    C = -R.T @ tvec
                    s_C = np.dot(C - plane_pt, plane_n)

                    bad_fids = []
                    total_pts = 0
                    pts_on_cam_side = 0

                    for fid, res in tri_data.items():
                        for label, X in [('A', res.get('XA')), ('B', res.get('XB'))]:
                            if X is not None:
                                total_pts += 1
                                sX = np.dot(X - plane_pt, plane_n)
                                if sX * s_C > 0 or sX < eps_mm:
                                    pts_on_cam_side += 1
                                    bad_fids.append(fid)

                    bad_fids = sorted(set(bad_fids))
                    pct_good = (total_pts - pts_on_cam_side) / max(total_pts, 1) * 100

                    print(f"  Win {wid} Cam {cid}: s(C)={s_C:.1f}mm | pct_object_side={pct_good:.1f}%")

                    if bad_fids:
                        print(
                            f"    Frames with points on camera side ({len(bad_fids)}): {bad_fids[:20]}" +
                            ("..." if len(bad_fids) > 20 else "")
                        )
            print("")

        if not loaded_cache:
            ba_optimizer.save_cache(out_path, points_3d=all_points_3d)

        v_cams_cpp = ba_optimizer.cams_cpp if ba_optimizer else None

        if v_cams_cpp:
            for cam in v_cams_cpp.values():
                pp = cam._pinplate_param
                pp.proj_nmax = 10000
                cam._pinplate_param = pp

        if tri_data and cam_params and v_cams_cpp:
            try:
                print("\n" + "=" * 50)
                print("[Verification] Starting Final Close-loop Validation...")

                valid_fids = [fid for fid, res in tri_data.items() if res['validA'] and res['validB']]
                if not valid_fids:
                    valid_fids = list(tri_data.keys())

                v_fid = random.choice(valid_fids)
                v_res = tri_data[v_fid]

                offset = 0
                for f in dataset['frames']:
                    if f == v_fid:
                        break
                    if tri_data[f]['XA'] is not None:
                        offset += 1
                    if tri_data[f]['XB'] is not None:
                        offset += 1

                v_idx_A = offset if v_res['XA'] is not None else -1
                v_idx_B = (offset + (1 if v_res['XA'] is not None else 0)) if v_res['XB'] is not None else -1

                print(f"[Verification] Selected Frame: {v_fid}")

                for label, k, v_idx in [('A', 0, v_idx_A), ('B', 1, v_idx_B)]:
                    if v_idx == -1:
                        continue

                    v_rays = []
                    obs_map = dataset['obsA' if k == 0 else 'obsB'][v_fid]
                    for cid, uv in obs_map.items():
                        if cid in v_cams_cpp:
                            r = build_pinplate_ray_cpp(
                                v_cams_cpp[cid], uv, cam_id=cid, window_id=cam_to_window[cid],
                                frame_id=v_fid, endpoint=label
                            )
                            if r.valid:
                                v_rays.append(r)

                    if len(v_rays) >= 2:
                        X_manual, _, _, _ = triangulate_point(v_rays)
                        resids = [point_to_ray_dist(X_manual, r.o, r.d) for r in v_rays]
                        ray_rmse = np.sqrt(np.mean(np.square(resids)))

                        X_aligned = np.array(dataset['points_3d'][v_idx * 3: v_idx * 3 + 3])

                        dist = np.linalg.norm(X_manual - X_aligned)

                        print(f"  Point {label}:")
                        print(f"    Manual 3D (Refract) : {X_manual.round(4)}")
                        print(f"    Aligned 3D (Dataset): {X_aligned.round(4)}")
                        print(f"    Ray RMSE            : {ray_rmse:.6f} mm ({len(v_rays)} cams)")
                        print(f"    L2 Distance Error   : {dist:.8e} mm")

                        if dist < 1e-6:
                            print(f"    [OK] Point {label} is perfectly consistent.")
                        else:
                            print(f"    [WARNING] Point {label} has discrepancy!")

                if v_idx_A != -1 and v_idx_B != -1:
                    pA = np.array(dataset['points_3d'][v_idx_A * 3: v_idx_A * 3 + 3])
                    pB = np.array(dataset['points_3d'][v_idx_B * 3: v_idx_B * 3 + 3])
                    L_dataset = np.linalg.norm(pB - pA)
                    print(
                        f"  Frame Wand Length: {L_dataset:.4f} mm (Target: {wand_len_target:.4f} mm, "
                        f"Error: {L_dataset - wand_len_target:.4f} mm)"
                    )

                print("=" * 50 + "\n")
            except Exception as e:
                print(f"[Verification] Close-loop check failed: {e}")

        self.calculate_per_frame_errors_refractive(
            dataset, tri_data, v_cams_cpp, wand_len_target
        )

        return True, cam_params, report, dataset

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
            self.reporter.section("Phase 1: Validating Inputs & Config")
        self.verbose = (verbosity >= 2) # Legacy compat
        self.reporter.detail(f"  Model: Pinhole+Refraction")
        self.reporter.detail(f"  Num Windows: {num_windows}")

        
        # 1. Validate and Structure Observations
        try:
            dataset = self._collect_observations(cam_to_window)
            wand_len_target = dataset.get('wand_length', 10.0)
            self.reporter.detail(
                f"  Frames: {dataset['num_frames']}, Cameras: {dataset['num_cams']}"
            )
            self.reporter.detail(
                f"  Total Valid Observations (2 dots per view): {dataset['total_observations']}"
            )
        except Exception as e:
            self.reporter.error(f"  [Error] Validation failed: {e}")
            raise
            
        # 2. Log Config Snapshot (Engineering Guardrail #2)
        self.reporter.header("Mapping Snapshot:")
        for cid in dataset['cam_ids']:
            wid = cam_to_window[cid]
            self.reporter.detail(f"  Cam {cid} -> Window {wid}")
            
        self.reporter.header("Media Parameters:")
        for wid, media in sorted(window_media.items()):
             self.reporter.detail(
                 f"  Win {wid}: n_air={media['n1']:.3f}, n_win={media['n2']:.3f}, "
                 f"n_obj={media['n3']:.3f}, thick={media['thickness']:.2f}mm"
             )

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
            
            
            # [CRITICAL] Ensure bootstrap points are always in dataset for P1 d_max
            dataset['X_A_bootstrap'] = X_A_scaled
            dataset['X_B_bootstrap'] = X_B_scaled
            
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
                 est_r_small, est_r_large = self._estimate_and_log_sphere_radii(dataset, cam_params, X_A_scaled, X_B_scaled, tag="Bootstrap")
            
            # Store in dataset for later stages (Step 2)
            dataset['est_radius_small_mm'] = est_r_small
            dataset['est_radius_large_mm'] = est_r_large
            print(f"[BOOT] Stored estimated radii in dataset: Small={est_r_small:.3f}mm, Large={est_r_large:.3f}mm")


            if self.window_planes is None:
                print("[BOOT] Plane initialization failed.")
                if progress_callback: progress_callback("Plane initialization failed.", 0, 0, 0)
                return None, False
                
            # [DEBUG] Inspect window_planes keys
            print(f"[BOOT] Initialized window_planes keys: {list(self.window_planes.keys())}")
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

        self.reporter.section("SANITY: Window Parameters (Initial)")
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
                 
        self.reporter.detail("-" * 30)

        # === Phase 2/3: Build Refracted Rays (C++ Authority) ===
        # REPLACED: Export/Reload loop with direct in-memory initialization
        cams_cpp = self._init_cams_cpp_in_memory(cam_params, window_media, cam_to_window, self.window_planes)
        
        self.reporter.section("Phase 2/3: Building Rays (C++ Kernel)")
        
        # Progress callback: report Phase 2/3
        if progress_callback:
            try:
                progress_callback("Phase 2/3: Building Rays", 0.0, 0.0, 0.0)
            except:
                pass


        

        # === Phase: Bundle Adjustment (Selective BA) ===
        # [USER REQUEST] Re-estimate Radii with latest params (P1 result)
        # [USER REQUEST] Re-estimate Radii with latest params (P1 result)
        # rs_pr4, rl_pr4 = 1.5, 2.0 # Defaults
        if X_A_scaled and X_B_scaled:
            rs, rl = self._estimate_and_log_sphere_radii(dataset, cam_params, X_A_scaled, X_B_scaled, tag="BA Pre-Calc")
            dataset['est_radius_small_mm'] = rs
            dataset['est_radius_large_mm'] = rl
            rs_pr4, rl_pr4 = rs, rl
            self.reporter.info(f"BA: Updated estimated radii: Small={rs:.3f}mm, Large={rl:.3f}mm")

        # Pass verbosity to config
        ba_config = RefractiveBAConfig(
            skip_optimization=False,
            stage=4,
            verbosity=verbosity,
            R_small_mm=rs_pr4,
            R_large_mm=rl_pr4,
            dist_coeff_num=self.base.dist_coeff_num
        )
        
        ba_optimizer = RefractiveBAOptimizer(
            dataset=dataset,
            cam_params=cam_params,
            cams_cpp=cams_cpp,
            cam_to_window=cam_to_window,
            window_media=window_media,
            window_planes=self.window_planes,
            wand_length=wand_len_target,
            config=ba_config,
            progress_callback=progress_callback
        )

        use_ba_cache = False
        loaded_cache = False
        if use_ba_cache:
            # Try to load cache
            loaded_cache = ba_optimizer.try_load_cache(out_path)
        
        if loaded_cache:
            # Cache loaded successfully, extract updated state
            self.window_planes = ba_optimizer.window_planes
            cam_params = ba_optimizer.cam_params
            window_media = ba_optimizer.window_media
            self.reporter.info("Cache: Using cached results, skipping optimization")
        else:
            # Run optimization
            self.window_planes, cam_params = ba_optimizer.optimize()
            window_media = ba_optimizer.window_media
            # Save cache
            ba_optimizer.save_cache(out_path)

        # === Phase Round4: Intrinsics + Thickness ===
        # [USER REQUEST] Re-estimate Radii with latest params (post-BA)
        if X_A_scaled and X_B_scaled:
            rs, rl = self._estimate_and_log_sphere_radii(dataset, cam_params, X_A_scaled, X_B_scaled, tag="Round4 Pre-Calc")
            dataset['est_radius_small_mm'] = rs
            dataset['est_radius_large_mm'] = rl
            print(f"[ROUND4] Updated estimated radii: Small={rs:.3f}mm, Large={rl:.3f}mm")

        cam_params, cams_cpp, stored_dir = self._align_and_export(
            cam_params=cam_params,
            window_media=window_media,
            cam_to_window=cam_to_window,
            ba_optimizer=ba_optimizer,
            cams_cpp=cams_cpp,
            dataset=dataset,
            out_path=out_path,
            verbosity=verbosity,
        )


        rays_db, invalid_reasons, total_obs, invalid_obs = self._build_labelled_rays(
            dataset=dataset,
            cam_params=cam_params,
            cams_cpp=cams_cpp,
            cam_to_window=cam_to_window,
            active_cam_ids=active_cam_ids,
            inactive_cam_ids=inactive_cam_ids,
            verbosity=verbosity,
        )
        
        return self._run_triangulation_and_reports(
            dataset=dataset,
            rays_db=rays_db,
            active_cam_ids=active_cam_ids,
            wand_len_target=wand_len_target,
            cam_params=cam_params,
            cam_to_window=cam_to_window,
            window_media=window_media,
            stored_dir=stored_dir,
            out_path=out_path,
            invalid_reasons=invalid_reasons,
            total_obs=total_obs,
            invalid_obs=invalid_obs,
            loaded_cache=loaded_cache,
            ba_optimizer=ba_optimizer,
            cams_cpp=cams_cpp,
        )

    def calculate_per_frame_errors_refractive(self, dataset, tri_data, cams_cpp, wand_len_target):
        """
        Calculate per-frame reprojection and wand length errors for UI Error Analysis table.
        
        Uses C++ cam.project() for accurate PINPLATE projection (handles refraction).
        Stores results in self.base.per_frame_errors (same format as pinhole calibration).
        
        Format: {frame_idx: {'cam_errors': {cam_id: max_err_px}, 'len_error': abs_mm}}
        
        Args:
            dataset: dict with 'frames', 'obsA', 'obsB', 'points_3d' keys
            tri_data: dict {fid: {'XA': np.ndarray, 'XB': np.ndarray, 'validA': bool, 'validB': bool}}
            cams_cpp: dict {cid: lpt.Camera} - loaded C++ camera objects
            wand_len_target: target wand length in mm
        """
        if lpt is None:
            print("[per_frame_errors] pyopenlpt not available, skipping.")
            return
        
        if not dataset or not tri_data or not cams_cpp:
            print("[per_frame_errors] Missing required data, skipping.")
            return
        
        print("\n[per_frame_errors] Calculating reprojection errors for all frames...")
        
        self.base.per_frame_errors = {}
        
        # Statistics accumulators per camera
        cam_error_sums = {cid: [] for cid in cams_cpp.keys()}     # For UI (Max per frame)
        cam_all_points_errs = {cid: [] for cid in cams_cpp.keys()} # For Logging (All points) [USER REQUEST]
        total_frames = 0
        
        for fid in dataset['frames']:
            if fid not in tri_data:
                continue
            
            res = tri_data[fid]
            XA = res.get('XA')
            XB = res.get('XB')
            
            # Skip frames without valid 3D points
            if XA is None and XB is None:
                continue
            
            # Calculate wand length error (if both points valid)
            len_err = 0.0
            if XA is not None and XB is not None:
                wand_len = np.linalg.norm(XB - XA)
                len_err = abs(wand_len - wand_len_target)
            
            # Calculate per-camera reprojection errors
            cam_errors = {}
            proj_pts = {}  # Store projected points for visualization
            
            obsA = dataset['obsA'].get(fid, {})
            obsB = dataset['obsB'].get(fid, {})
            
            for cid, cam in cams_cpp.items():
                err_max = 0.0
                proj_A = None
                proj_B = None
                
                try:
                    # [CPP_PROTOCOL] Project A point
                    if XA is not None and cid in obsA:
                        pt_world_A = lpt.Pt3D(float(XA[0]), float(XA[1]), float(XA[2]))
                        uv_proj_A = cam.project(pt_world_A)
                        proj_A = (float(uv_proj_A[0]), float(uv_proj_A[1]))
                        
                        uv_obs_A = obsA[cid]
                        # Verify finite projection
                        if abs(proj_A[0]) < 1e5 and abs(proj_A[1]) < 1e5:
                            err_A = np.sqrt((proj_A[0] - uv_obs_A[0])**2 + (proj_A[1] - uv_obs_A[1])**2)
                            err_max = max(err_max, err_A)
                            cam_all_points_errs[cid].append(err_A) # Log A
                    
                    # [CPP_PROTOCOL] Project B point
                    if XB is not None and cid in obsB:
                        pt_world_B = lpt.Pt3D(float(XB[0]), float(XB[1]), float(XB[2]))
                        uv_proj_B = cam.project(pt_world_B)
                        proj_B = (float(uv_proj_B[0]), float(uv_proj_B[1]))
                        
                        uv_obs_B = obsB[cid]
                        # Verify finite projection
                        if abs(proj_B[0]) < 1e5 and abs(proj_B[1]) < 1e5:
                            err_B = np.sqrt((proj_B[0] - uv_obs_B[0])**2 + (proj_B[1] - uv_obs_B[1])**2)
                            err_max = max(err_max, err_B)
                            cam_all_points_errs[cid].append(err_B) # Log B
                    
                    if err_max > 0:
                        cam_errors[cid] = err_max
                        cam_error_sums[cid].append(err_max)
                    
                    # Store projected points for visualization
                    proj_pts[cid] = (proj_A, proj_B)
                    
                except Exception as e:
                    # Silently skip projection failures
                    pass
            
            if cam_errors:
                self.base.per_frame_errors[fid] = {
                    'cam_errors': cam_errors,
                    'len_error': len_err,
                    'proj_pts': proj_pts  # For overlay visualization
                }
                total_frames += 1
        
        # Log summary per camera (Using All-Points stats per user request)
        print(f"[per_frame_errors] Computed errors for {total_frames} frames.")
        per_camera_mean = {}
        for cid in sorted(cam_all_points_errs.keys()):
            errs = cam_all_points_errs[cid]
            if errs:
                mean_err = np.mean(errs)
                std_err = np.std(errs)
                max_err = np.max(errs)
                per_camera_mean[cid] = float(mean_err)
                print(f"  Cam {cid}: Mean={mean_err:.3f} px, Std={std_err:.3f} px, Max={max_err:.3f} px ({len(errs)} samples)")

        if per_camera_mean:
            dataset['per_camera_mean_proj_err'] = per_camera_mean


