#!/usr/bin/env python
"""
Pretest whether initial plane positions change the final convergence basin.

This script reuses the same dataset, pinhole bootstrap, camera setup, and
window/media configuration as `test_refractive_staged_strategy.py`, but sweeps
the initial positions of both planes along their normals and runs the direct
BA alternating loop + joint (stage 3 only) with geometric-only plane d-bounds.

Run under the OpenLPT environment.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import faulthandler
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np

from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator
from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import RefractiveWandCalibrator
from modules.camera_calibration.wand_calibration.refractive_bootstrap import PinholeBootstrapP0, PinholeBootstrapP0Config
from modules.camera_calibration.wand_calibration.refraction_calibration_BA import RefractiveBAConfig, RefractiveBAOptimizer
from modules.camera_calibration.wand_calibration.refractive_geometry import validate_native_safety
from modules.camera_calibration.wand_calibration.test_refractive_staged_strategy import (
    _calc_p0_cam_error,
    _camera_settings_default,
    _cam_to_window_default,
    _compute_final_metrics,
    _deepcopy_dataset,
    _fallback_seed_pair_by_shared_frames,
    _select_seed_pair_same_window,
    _try_align_cam_ids,
    _window_media_default,
)


WORKER_SHARED: Dict[str, object] = {}


class PretestBAOptimizer(RefractiveBAOptimizer):
    def _perform_geometric_initialization(self, wid, cid):
        print(f"  [Pretest] Skip weak-window geometric plane override for window {wid}, cam {cid}")
        return None


def _build_step_a_plane_d_bounds_no_weak(opt: PretestBAOptimizer, loop_iter: int) -> Dict[int, Tuple[float, float]]:
    pts = opt._collect_points_for_alignment()
    if not pts:
        opt.reporter.detail(
            f"  [LOOP {loop_iter}] No triangulated points for geometric d-bounds; fallback to global bounds"
        )
        return {}

    pts_arr = np.asarray(pts, dtype=np.float64)
    eps = 0.05
    out: Dict[int, Tuple[float, float]] = {}

    opt.reporter.detail(
        f"  [LOOP {loop_iter}] Recomputing plane_d bounds from current cameras/points (geometric-only)"
    )
    for wid in opt.window_ids:
        pl0 = opt.initial_planes.get(wid, opt.window_planes.get(wid))
        if pl0 is None:
            continue
        pt0 = np.asarray(pl0["plane_pt"], dtype=np.float64)
        n0 = np.asarray(pl0["plane_n"], dtype=np.float64)
        nn = np.linalg.norm(n0)
        if nn < 1e-12:
            continue
        n0 = n0 / nn

        cams = [cid for cid in opt.window_to_cams.get(wid, []) if cid in opt.active_cam_ids and cid in opt.cam_params]
        if not cams:
            continue

        lo = -np.inf
        hi = np.inf
        cam_terms = []
        for cid in cams:
            C = _camera_center_from_param(opt.cam_params[cid])
            dists = np.linalg.norm(pts_arr - C.reshape(1, 3), axis=1)
            if dists.size == 0:
                continue
            idx = int(np.argmin(dists))
            X_min = pts_arr[idx]
            d_min = float(dists[idx])

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

        if hi <= lo:
            mid = 0.5 * (lo + hi)
            lo = mid - 1e-3
            hi = mid + 1e-3

        if not (lo <= 0.0 <= hi):
            raw_lo, raw_hi = lo, hi
            lo = min(lo, -1e-6)
            hi = max(hi, 1e-6)
            opt.reporter.detail(
                f"    [d-bound-fix] Win {wid}: raw [{raw_lo:.3f}, {raw_hi:.3f}] excluded 0; adjusted to [{lo:.3f}, {hi:.3f}] for delta x0=0"
            )

        out[int(wid)] = (float(lo), float(hi))
        cam_msg = ", ".join([f"cam{cid}:[{c_lo:.2f},{c_hi:.2f}] dmin={dmin:.1f}" for cid, c_lo, c_hi, dmin in cam_terms])
        opt.reporter.detail(f"    [d-bound] Win {wid} GEO-ONLY -> [{lo:.3f}, {hi:.3f}] mm; {cam_msg}")
    return out


def _run_pretest_stage3(opt: PretestBAOptimizer) -> Tuple[Dict[int, Dict], Dict[int, np.ndarray]]:
    opt._compute_physical_sigmas()
    opt._weak_window_refs = {}

    opt.reporter.section(f"Bundle Adjustment Start ({len(opt.active_cam_ids)} cameras, {len(opt.window_ids)} windows)")
    for wid, pl in sorted(opt.window_planes.items()):
        pt = pl["plane_pt"]
        n = pl["plane_n"]
        opt.reporter.detail(f"  [INIT] Win {wid}: pt=[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}], n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")

    max_loop_iters = 6
    loop_iter = 0
    hit_boundary = True
    pre_last_loop_align_done = False
    opt.reporter.section(f"Alternating Loop (Max {max_loop_iters} passes)")

    while loop_iter < max_loop_iters:
        loop_iter += 1
        loss_plane = opt.config.loss_plane or "linear"
        loss_cam = opt.config.loss_cam or "linear"

        opt._sync_initial_state()
        opt._compute_physical_sigmas()
        plane_d_bounds = _build_step_a_plane_d_bounds_no_weak(opt, loop_iter)

        opt.reporter.header(f"Loop {loop_iter} - Step A: Optimize Planes (Bounds: +/- 2.5 deg)")
        opt._print_plane_diagnostics(f"Pre-Loop {loop_iter} Planes")
        limit_angle_rad = np.radians(2.5)
        res_planes, layout_planes = opt._optimize_generic(
            mode=f"loop_{loop_iter}_planes",
            description="Adjusting plane parameters ...",
            enable_planes=True,
            enable_cam_t=False,
            enable_cam_r=False,
            limit_rot_rad=0.0,
            limit_trans_mm=0.0,
            limit_plane_d_mm=500.0,
            limit_plane_angle_rad=limit_angle_rad,
            plane_d_bounds=plane_d_bounds,
            ftol=5e-4,
            xtol=1e-5,
            gtol=1e-5,
            loss=loss_plane,
        )
        opt._print_plane_diagnostics(f"Loop {loop_iter} Planes")

        active_mask = res_planes.active_mask
        hit_boundary = False
        idx = 0
        for (ptype, _, _) in layout_planes:
            if ptype in ("plane_a", "plane_b") and active_mask[idx] != 0:
                hit_boundary = True
            idx += 1

        if hit_boundary:
            opt.reporter.detail(f"  [LOOP {loop_iter}] Plane constraints ACTIVE (hit 2.5 deg bound). Continuing loop.")
        else:
            opt.reporter.detail(f"  [LOOP {loop_iter}] Plane constraints INACTIVE (all within 2.5 deg). Loop condition satisfied.")

        is_last_loop_for_cam = (not hit_boundary) or (loop_iter == max_loop_iters)
        if (not pre_last_loop_align_done) and is_last_loop_for_cam:
            opt.reporter.section("Coordinate Alignment Before Last Loop Camera Step")
            pre_mode = opt._get_retry_alignment_mode()
            opt.reporter.detail(f"[Coordinate Alignment] pre-last-loop-cam mode={pre_mode}")
            opt._apply_coordinate_alignment(tag="pre-last-loop-cam", refresh_initial=True, align_mode=pre_mode)
            pre_last_loop_align_done = True

        opt.reporter.header(f"Loop {loop_iter} - Step B: Optimize Cameras (Free Bounds)")
        b_cam_free = (np.deg2rad(180.0), 2000.0)
        opt._optimize_generic(
            mode=f"loop_{loop_iter}_cams",
            description="Optimizing camera extrinsic parameters ...",
            enable_planes=False,
            enable_cam_t=True,
            enable_cam_r=True,
            limit_rot_rad=b_cam_free[0],
            limit_trans_mm=b_cam_free[1],
            limit_plane_d_mm=0.0,
            limit_plane_angle_rad=0.0,
            ftol=5e-4,
            xtol=1e-5,
            gtol=1e-5,
            loss=loss_cam,
        )
        opt._print_plane_diagnostics(f"Loop {loop_iter} Cams")

        if not hit_boundary:
            opt.reporter.info(f"Converged early at Loop {loop_iter} (Planes inside 2.5 deg). Stopping loop.")
            break

    if hit_boundary and loop_iter == max_loop_iters:
        opt.reporter.info(f"Loop reached max iterations ({max_loop_iters}). Proceeding to Joint.")

    opt.reporter.section("Joint Optimization (Round 3 Rules)")
    limit_rvec = np.radians(20.0)
    limit_plane_d = 50.0
    limit_plane_ang = np.radians(10.0)
    limit_tvec = 50.0
    print("  Bounds: rvec < 20deg, plane_d < 50mm, plane_ang < 10deg, tvec < 50mm")

    joint_kwargs = dict(
        mode="joint",
        description="Optimizing plane and camera extrinsic parameters ...",
        enable_planes=True,
        enable_cam_t=True,
        enable_cam_r=True,
        limit_rot_rad=limit_rvec,
        limit_trans_mm=limit_tvec,
        limit_plane_d_mm=limit_plane_d,
        limit_plane_angle_rad=limit_plane_ang,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
        loss=opt.config.loss_joint,
        max_nfev=50,
    )
    joint_chunks = opt._get_chunk_schedule_for_mode("joint")
    if joint_chunks:
        opt._run_round_chunked("joint", joint_kwargs, joint_chunks, freeze_bounds_reference=True)
    else:
        opt._optimize_generic(**joint_kwargs)
    opt._print_plane_diagnostics("Joint End")

    n_cams = max(1, len(opt.cam_params))
    lambda_fixed = 2.0 * n_cams
    opt._set_barrier_profile_for_mode("final", log=False)
    opt.evaluate_residuals(opt.window_planes, opt.cam_params, lambda_fixed, window_media=opt.window_media)
    return opt.window_planes, opt.cam_params


def _run_pretest_loop_only(
    opt: PretestBAOptimizer,
) -> Tuple[Dict[int, Dict[str, object]], Dict[int, object], Dict[str, object]]:
    opt._compute_physical_sigmas()
    opt._weak_window_refs = {}

    opt.reporter.section(f"Bundle Adjustment Start ({len(opt.active_cam_ids)} cameras, {len(opt.window_ids)} windows)")
    for wid, pl in sorted(opt.window_planes.items()):
        pt = pl["plane_pt"]
        n = pl["plane_n"]
        opt.reporter.detail(f"  [INIT] Win {wid}: pt=[{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}], n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")

    max_loop_iters = 6
    loop_iter = 0
    hit_boundary = True
    pre_last_loop_align_done = False
    opt.reporter.section(f"Alternating Loop (Max {max_loop_iters} passes)")

    while loop_iter < max_loop_iters:
        loop_iter += 1
        loss_plane = opt.config.loss_plane or "linear"
        loss_cam = opt.config.loss_cam or "linear"

        opt._sync_initial_state()
        opt._compute_physical_sigmas()
        plane_d_bounds = _build_step_a_plane_d_bounds_no_weak(opt, loop_iter)

        opt.reporter.header(f"Loop {loop_iter} - Step A: Optimize Planes (Bounds: +/- 2.5 deg)")
        opt._print_plane_diagnostics(f"Pre-Loop {loop_iter} Planes")
        limit_angle_rad = np.radians(2.5)
        res_planes, layout_planes = opt._optimize_generic(
            mode=f"loop_{loop_iter}_planes",
            description="Adjusting plane parameters ...",
            enable_planes=True,
            enable_cam_t=False,
            enable_cam_r=False,
            limit_rot_rad=0.0,
            limit_trans_mm=0.0,
            limit_plane_d_mm=500.0,
            limit_plane_angle_rad=limit_angle_rad,
            plane_d_bounds=plane_d_bounds,
            ftol=5e-4,
            xtol=1e-5,
            gtol=1e-5,
            loss=loss_plane,
        )
        opt._print_plane_diagnostics(f"Loop {loop_iter} Planes")

        active_mask = res_planes.active_mask
        hit_boundary = False
        idx = 0
        for (ptype, _, _) in layout_planes:
            if ptype in ("plane_a", "plane_b") and active_mask[idx] != 0:
                hit_boundary = True
            idx += 1

        if hit_boundary:
            opt.reporter.detail(f"  [LOOP {loop_iter}] Plane constraints ACTIVE (hit 2.5 deg bound). Continuing loop.")
        else:
            opt.reporter.detail(f"  [LOOP {loop_iter}] Plane constraints INACTIVE (all within 2.5 deg). Loop condition satisfied.")

        is_last_loop_for_cam = (not hit_boundary) or (loop_iter == max_loop_iters)
        if (not pre_last_loop_align_done) and is_last_loop_for_cam:
            opt.reporter.section("Coordinate Alignment Before Last Loop Camera Step")
            pre_mode = opt._get_retry_alignment_mode()
            opt.reporter.detail(f"[Coordinate Alignment] pre-last-loop-cam mode={pre_mode}")
            opt._apply_coordinate_alignment(tag="pre-last-loop-cam", refresh_initial=True, align_mode=pre_mode)
            pre_last_loop_align_done = True

        opt.reporter.header(f"Loop {loop_iter} - Step B: Optimize Cameras (Free Bounds)")
        b_cam_free = (np.deg2rad(180.0), 2000.0)
        opt._optimize_generic(
            mode=f"loop_{loop_iter}_cams",
            description="Optimizing camera extrinsic parameters ...",
            enable_planes=False,
            enable_cam_t=True,
            enable_cam_r=True,
            limit_rot_rad=b_cam_free[0],
            limit_trans_mm=b_cam_free[1],
            limit_plane_d_mm=0.0,
            limit_plane_angle_rad=0.0,
            ftol=5e-4,
            xtol=1e-5,
            gtol=1e-5,
            loss=loss_cam,
        )
        opt._print_plane_diagnostics(f"Loop {loop_iter} Cams")

        if not hit_boundary:
            opt.reporter.info(f"Converged early at Loop {loop_iter} (Planes inside 2.5 deg). Stopping loop.")
            break

    return opt.window_planes, opt.cam_params, {"loop_iters": loop_iter, "hit_boundary": hit_boundary}


def _create_base_and_refr(
    csv_path: str,
    wand_length: float,
    dist_num: int,
    cam_settings: Dict[int, Dict[str, float]],
) -> Tuple[WandCalibrator, RefractiveWandCalibrator]:
    base = WandCalibrator()
    ok, msg = base.load_wand_data_from_csv(csv_path)
    if not ok:
        raise RuntimeError(f"Failed to load wand CSV: {msg}")

    base.wand_length = float(wand_length)
    base.dist_coeff_num = int(dist_num)
    base.camera_settings = copy.deepcopy(cam_settings)
    base.cams = {cid: {} for cid in cam_settings.keys()}
    base.cameras = base.cams
    base.active_cam_ids = sorted(cam_settings.keys())
    first_cid = base.active_cam_ids[0]
    base.image_size = (cam_settings[first_cid]["height"], cam_settings[first_cid]["width"])
    refr = RefractiveWandCalibrator(base)
    return base, refr


def _camera_center_from_param(p: np.ndarray) -> np.ndarray:
    rvec = np.asarray(p[0:3], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(p[3:6], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ tvec).reshape(3)


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    return arr / (np.linalg.norm(arr) + 1e-12)


def _plane_angle_deg(n_ref: np.ndarray, n_cur: np.ndarray) -> float:
    a = _normalize(n_ref)
    b = _normalize(n_cur)
    cosang = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _copy_cam_params(cam_params: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    return {int(cid): np.asarray(p, dtype=np.float64).copy() for cid, p in cam_params.items()}


def _copy_window_planes(window_planes: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Dict[str, np.ndarray]]:
    return {
        int(wid): {
            "plane_pt": np.asarray(pl["plane_pt"], dtype=np.float64).copy(),
            "plane_n": np.asarray(pl["plane_n"], dtype=np.float64).copy(),
        }
        for wid, pl in window_planes.items()
    }


def _serialize_np(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _serialize_np(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_np(v) for v in value]
    return value


def _log_trial_header(shared: Dict[str, object], trial: Dict[str, object]) -> None:
    plane_ids = [int(w) for w in shared["plane_ids"]]
    print("=" * 72)
    print(f"[PRETEST][TRIAL {int(trial['trial_id']):03d}] {trial['trial_name']}")
    print(f"  pid={os.getpid()} start={time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  mode=direct_ba_stage3 weak_window_geo_init=disabled_in_pretest")
    for wid in plane_ids:
        meta = shared["plane_meta"][wid]
        idx = int(trial["plane_indices"][wid])
        depth = float(trial["initial_depths_mm"][wid])
        baseline_depth = float(meta["baseline_depth_mm"])
        print(f"  Window {wid}: init_index={idx} init_depth={depth:.6f} mm depth_shift={depth - baseline_depth:.6f} mm")
        print(f"    baseline_plane_pt={np.asarray(meta['baseline_plane_pt'], dtype=np.float64).round(6).tolist()}")
        print(f"    baseline_plane_n ={np.asarray(meta['plane_n'], dtype=np.float64).round(6).tolist()}")
        print(f"    trial_plane_pt   ={np.asarray(trial['initial_plane_points'][wid], dtype=np.float64).round(6).tolist()}")
    print("=" * 72)


def _log_sampling_context(shared: Dict[str, object], trial: Dict[str, object]) -> None:
    print("[PRETEST][SAMPLING]")
    for wid in [int(w) for w in shared["plane_ids"]]:
        meta = shared["plane_meta"][wid]
        sampled_depths = [float(rec["depth_mm"]) for rec in meta["positions"]]
        print(
            f"  Window {wid}: d_min={float(meta['depth_min_mm']):.6f} mm, "
            f"d_max={float(meta['depth_max_mm']):.6f} mm, baseline_depth={float(meta['baseline_depth_mm']):.6f} mm"
        )
        print(
            f"    depth_bound_strategy={meta.get('depth_bound_strategy', 'unknown')} "
            f"nearest_positive_depth_mm={meta.get('nearest_positive_depth_mm')} "
            f"positive_endpoint_depth_count={meta.get('positive_endpoint_depth_count', 0)}"
        )
        if bool(meta.get("depth_bound_fallback_used", False)):
            print(f"    depth_bound_fallback_reason={meta.get('depth_bound_fallback_reason', 'unknown')}")
        print(f"    sampled_depths_mm={sampled_depths}")
        print(f"    selected_index={int(trial['plane_indices'][wid])} selected_depth={float(trial['initial_depths_mm'][wid]):.6f} mm")
        print(f"    camera_ids={meta['camera_ids']}")
        print(f"    camera_center_mean={np.asarray(meta['camera_center_mean'], dtype=np.float64).round(6).tolist()}")
    mids_a = shared["dataset"].get("X_A_bootstrap", {})
    mids_b = shared["dataset"].get("X_B_bootstrap", {})
    print(f"  bootstrap_endpoint_counts: X_A={len(mids_a)}, X_B={len(mids_b)}")


def _log_trial_footer(result: Dict[str, object]) -> None:
    print("[PRETEST][RESULT]")
    print(f"  success={bool(result.get('success'))} elapsed_sec={float(result.get('elapsed_sec', float('nan'))):.3f}")
    print(f"  stage={result.get('stage')}")
    final_metrics = result.get("final_metrics") or {}
    if final_metrics:
        print(
            f"  final_metrics: ray={final_metrics.get('ray_rmse_mm')} mm, "
            f"len={final_metrics.get('len_rmse_mm')} mm, proj={final_metrics.get('proj_rmse_px')} px"
        )
    plane_summary = result.get("final_plane_summary") or {}
    for wid, ps in sorted(plane_summary.items(), key=lambda kv: int(kv[0])):
        print(
            f"  Window {wid}: final_depth={ps.get('final_depth_mm')} mm, "
            f"depth_delta={ps.get('depth_delta_mm')} mm, angle_delta={ps.get('angle_delta_deg')} deg"
        )
    if result.get("error"):
        print(f"  error={result.get('error')}")


def depths_to_plane_points(depths: List[float], plane_meta: Dict[int, Dict[str, object]]) -> Dict[int, np.ndarray]:
    depth_arr = np.asarray(depths, dtype=np.float64).reshape(-1)
    window_ids = sorted([int(wid) for wid in plane_meta.keys()])
    if depth_arr.size != len(window_ids):
        raise ValueError(
            f"Depth vector size mismatch: got {depth_arr.size}, expected {len(window_ids)} for windows {window_ids}."
        )

    out: Dict[int, np.ndarray] = {}
    for i, wid in enumerate(window_ids):
        meta = plane_meta[wid]
        C_mean = np.asarray(meta["camera_center_mean"], dtype=np.float64).reshape(3)
        plane_n = np.asarray(meta["plane_n"], dtype=np.float64).reshape(3)
        out[wid] = C_mean + float(depth_arr[i]) * plane_n
    return out


def _build_de_bounds(
    plane_meta: Dict[int, Dict[str, object]],
) -> Dict[int, Tuple[float, float]]:
    bounds: Dict[int, Tuple[float, float]] = {}
    bound_strategy: Dict[int, str] = {}

    for wid in sorted([int(k) for k in plane_meta.keys()]):
        meta = plane_meta[wid]
        depth_min_raw = meta.get("depth_min_mm")
        depth_max_raw = meta.get("depth_max_mm")
        depth_min = float(depth_min_raw) if depth_min_raw is not None else float("nan")
        depth_max = float(depth_max_raw) if depth_max_raw is not None else float("nan")
        if (not np.isfinite(depth_min)) or (not np.isfinite(depth_max)) or (depth_max <= depth_min):
            bound_meta = {
                "depth_min_mm": depth_min_raw,
                "depth_max_mm": depth_max_raw,
                "depth_bound_strategy": meta.get("depth_bound_strategy"),
                "positive_endpoint_depth_count": meta.get("positive_endpoint_depth_count"),
                "finite_endpoint_depth_count": meta.get("finite_endpoint_depth_count"),
                "nearest_positive_depth_mm": meta.get("nearest_positive_depth_mm"),
            }
            raise RuntimeError(
                f"Window {wid} missing valid DE bounds in plane_meta: "
                f"{bound_meta}"
            )

        lower, upper = depth_min, depth_max
        bound_strategy[wid] = "plane_meta_depth_interval"

        bounds[wid] = (float(lower), float(upper))

    _build_de_bounds.last_metadata = {
        "bound_strategy": bound_strategy,
    }
    return bounds


def _validate_de_candidate_geometry(
    depths: List[float],
    plane_meta: Dict[int, Dict[str, object]],
    plane_points: Dict[int, np.ndarray],
    min_depth_mm: float = 1.0,
) -> str:
    depth_arr = np.asarray(depths, dtype=np.float64).reshape(-1)
    window_ids = sorted([int(wid) for wid in plane_meta.keys()])
    if depth_arr.size != len(window_ids):
        return f"depth_size_mismatch:{depth_arr.size}!={len(window_ids)}"

    for i, wid in enumerate(window_ids):
        depth = float(depth_arr[i])
        meta = plane_meta[wid]
        C_mean = np.asarray(meta["camera_center_mean"], dtype=np.float64).reshape(3)
        camera_centers = np.asarray(meta.get("camera_centers", [C_mean]), dtype=np.float64).reshape(-1, 3)
        plane_n = np.asarray(meta["plane_n"], dtype=np.float64).reshape(3)
        plane_pt = np.asarray(plane_points[wid], dtype=np.float64).reshape(3)
        if not np.isfinite(depth):
            return f"non_finite_depth_w{wid}"
        if not np.all(np.isfinite(C_mean)):
            return f"non_finite_camera_center_w{wid}"
        if not np.all(np.isfinite(plane_n)):
            return f"non_finite_plane_normal_w{wid}"
        if np.linalg.norm(plane_n) <= 1e-9:
            return f"degenerate_plane_normal_w{wid}"
        if not np.all(np.isfinite(plane_pt)):
            return f"non_finite_plane_point_w{wid}"
        signed_depth = float(np.dot(plane_pt - C_mean, plane_n))
        if (not np.isfinite(signed_depth)) or signed_depth < float(min_depth_mm):
            return f"plane_too_close_to_camera_mean_w{wid}:{signed_depth}"
        euclid_dist = float(np.linalg.norm(plane_pt - C_mean))
        if (not np.isfinite(euclid_dist)) or euclid_dist < float(min_depth_mm):
            return f"plane_point_distance_too_small_w{wid}:{euclid_dist}"
        for cid, center in zip(meta.get("camera_ids", []), camera_centers):
            signed_cam_depth = float(np.dot(plane_pt - center, plane_n))
            if (not np.isfinite(signed_cam_depth)) or signed_cam_depth < float(min_depth_mm):
                return f"plane_invalid_for_cam{cid}_w{wid}:{signed_cam_depth}"
    return ""


def _build_plane_position_candidates(
    cam_params: Dict[int, np.ndarray],
    cam_to_window: Dict[int, int],
    window_planes: Dict[int, Dict[str, np.ndarray]],
    XA: Dict[int, np.ndarray],
    XB: Dict[int, np.ndarray],
    target_wid: int,
    num_positions: int,
) -> Dict[str, object]:
    endpoints = []
    for fid, pa in XA.items():
        pb = XB.get(fid)
        if pb is None:
            continue
        endpoints.append(np.asarray(pa, dtype=np.float64))
        endpoints.append(np.asarray(pb, dtype=np.float64))
    if not endpoints:
        raise RuntimeError(f"No bootstrap endpoint pairs available for plane sweep of window {target_wid}.")

    cam_ids = sorted([cid for cid, wid in cam_to_window.items() if int(wid) == int(target_wid) and cid in cam_params])
    if not cam_ids:
        raise RuntimeError(f"No cameras found for target window {target_wid}.")

    centers = np.vstack([_camera_center_from_param(cam_params[cid]) for cid in cam_ids])
    C_mean = np.mean(centers, axis=0)
    plane_pt0 = np.asarray(window_planes[target_wid]["plane_pt"], dtype=np.float64)
    plane_n0 = _normalize(window_planes[target_wid]["plane_n"])

    endpoints_arr = np.vstack(endpoints)
    depths = ((endpoints_arr - C_mean.reshape(1, 3)) @ plane_n0.reshape(3, 1)).reshape(-1)
    depths_finite = depths[np.isfinite(depths)]
    depths_pos = depths_finite[depths_finite > 1e-6]

    nearest_pos_depth = float("nan")
    if depths_pos.size > 0:
        nearest_pos_depth = float(np.min(depths_pos))
        d_min = 1.0
        d_max = nearest_pos_depth
        bound_strategy = "endpoint_nearest_positive_from_cmean"
    else:
        raise RuntimeError(
            f"Window {target_wid} has no valid positive endpoint depths along the pre-DE plane normal. "
            f"finite_endpoint_depth_count={int(depths_finite.size)} positive_endpoint_depth_count={int(depths_pos.size)} "
            f"plane_n={plane_n0.tolist()} C_mean={C_mean.tolist()}"
        )

    if (not np.isfinite(d_min)) or (not np.isfinite(d_max)) or (d_max <= d_min):
        raise RuntimeError(
            f"Window {target_wid} produced invalid DE depth interval from endpoint projections: "
            f"d_min={d_min} d_max={d_max} nearest_positive_depth_mm={nearest_pos_depth} "
            f"plane_n={plane_n0.tolist()} C_mean={C_mean.tolist()}"
        )

    depth_samples = np.linspace(d_min, d_max, num=max(2, int(num_positions)))
    point_samples = [C_mean + float(d) * plane_n0 for d in depth_samples]
    baseline_depth = float(np.dot(plane_pt0 - C_mean, plane_n0))
    baseline_dist = np.abs(depth_samples - baseline_depth)
    nearest_index = int(np.argmin(baseline_dist)) if baseline_dist.size > 0 else -1

    return {
        "window_id": int(target_wid),
        "camera_ids": [int(cid) for cid in cam_ids],
        "camera_centers": np.asarray(centers, dtype=np.float64),
        "camera_center_mean": np.asarray(C_mean, dtype=np.float64),
        "plane_n": np.asarray(plane_n0, dtype=np.float64),
        "baseline_plane_pt": np.asarray(plane_pt0, dtype=np.float64),
        "baseline_depth_mm": float(baseline_depth),
        "depth_min_mm": float(d_min),
        "depth_max_mm": float(d_max),
        "depth_bound_strategy": str(bound_strategy),
        "endpoint_depth_count": int(depths.size),
        "finite_endpoint_depth_count": int(depths_finite.size),
        "positive_endpoint_depth_count": int(depths_pos.size),
        "nearest_positive_depth_mm": float(nearest_pos_depth),
        "nearest_baseline_index": int(nearest_index),
        "positions": [
            {
                "index": int(i),
                "depth_mm": float(depth_samples[i]),
                "plane_pt": np.asarray(point_samples[i], dtype=np.float64),
            }
            for i in range(len(point_samples))
        ],
    }


def _rot_delta_deg(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    R_ref, _ = cv2.Rodrigues(np.asarray(p_ref[0:3], dtype=np.float64).reshape(3, 1))
    R_cur, _ = cv2.Rodrigues(np.asarray(p_cur[0:3], dtype=np.float64).reshape(3, 1))
    R_rel = R_cur @ R_ref.T
    rvec_rel, _ = cv2.Rodrigues(R_rel)
    return float(np.linalg.norm(rvec_rel.reshape(3)) * 180.0 / np.pi)


def _summarize_camera_deltas(
    cam_params_ref: Dict[int, np.ndarray],
    cam_params_cur: Dict[int, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cid in sorted(cam_params_cur.keys()):
        if cid not in cam_params_ref:
            continue
        p0 = np.asarray(cam_params_ref[cid], dtype=np.float64)
        p1 = np.asarray(cam_params_cur[cid], dtype=np.float64)
        out[str(cid)] = {
            "rot_deg": _rot_delta_deg(p0, p1),
            "trans_mm": float(np.linalg.norm(p1[3:6] - p0[3:6])),
            "df": float(p1[6] - p0[6]),
            "dcx": float(p1[7] - p0[7]),
            "dcy": float(p1[8] - p0[8]),
            "dk1": float(p1[9] - p0[9]),
            "dk2": float(p1[10] - p0[10]),
        }
    return out


def _summarize_plane_state(
    plane_meta: Dict[int, Dict[str, object]],
    window_planes_cur: Dict[int, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for wid, meta in plane_meta.items():
        pl = window_planes_cur[wid]
        C_mean = np.asarray(meta["camera_center_mean"], dtype=np.float64)
        n_ref = np.asarray(meta["plane_n"], dtype=np.float64)
        pt_cur = np.asarray(pl["plane_pt"], dtype=np.float64)
        n_cur = np.asarray(pl["plane_n"], dtype=np.float64)
        depth_cur = float(np.dot(pt_cur - C_mean, n_ref))
        out[str(wid)] = {
            "final_plane_pt": pt_cur.tolist(),
            "final_plane_n": n_cur.tolist(),
            "final_depth_mm": depth_cur,
            "depth_delta_mm": depth_cur - float(meta["baseline_depth_mm"]),
            "angle_delta_deg": _plane_angle_deg(n_ref, n_cur),
        }
    return out


def _init_worker(shared_payload: Dict[str, object]) -> None:
    global WORKER_SHARED
    WORKER_SHARED = shared_payload
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def _run_trial_worker(trial: Dict[str, object]) -> Dict[str, object]:
    shared = WORKER_SHARED
    plane_ids = [int(w) for w in shared["plane_ids"]]
    trial_id = int(trial["trial_id"])
    trial_name = str(trial["trial_name"])
    out_dir = Path(shared["out_dir"])
    trial_dir = out_dir / "trials" / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "trial.log"

    result: Dict[str, object] = {
        "trial_id": trial_id,
        "trial_name": trial_name,
        "plane_indices": {str(k): int(v) for k, v in trial["plane_indices"].items()},
        "initial_depths_mm": {str(k): float(v) for k, v in trial["initial_depths_mm"].items()},
        "initial_plane_points": {str(k): _serialize_np(v) for k, v in trial["initial_plane_points"].items()},
        "log_path": str(log_path),
        "stage": 3,
        "success": False,
    }

    start_t = time.time()
    current_stage = "init"
    try:
        with log_path.open("w", encoding="utf-8") as log_fp, contextlib.redirect_stdout(log_fp), contextlib.redirect_stderr(log_fp):
            _log_trial_header(shared, trial)
            _log_sampling_context(shared, trial)
            _, refr = _create_base_and_refr(
                csv_path=str(shared["csv"]),
                wand_length=float(shared["wand_length"]),
                dist_num=int(shared["dist_num"]),
                cam_settings=copy.deepcopy(shared["cam_settings"]),
            )

            dataset = _deepcopy_dataset(shared["dataset"])
            cam_params = _copy_cam_params(shared["baseline_cam_params"])
            cam_params_ref = _copy_cam_params(shared["baseline_cam_params"])
            window_planes = _copy_window_planes(shared["baseline_window_planes"])
            window_media = copy.deepcopy(shared["window_media"])
            cam_to_window = {int(cid): int(wid) for cid, wid in shared["cam_to_window"].items()}

            for wid in plane_ids:
                window_planes[wid]["plane_pt"] = np.asarray(trial["initial_plane_points"][wid], dtype=np.float64).copy()
            print(f"[PRETEST][TRIAL {trial_id:03d}] Direct BA stage=3 start")
            cams_cpp = refr._init_cams_cpp_in_memory(cam_params, window_media, cam_to_window, window_planes)
            cfg = RefractiveBAConfig(stage=3, dist_coeff_num=int(shared["dist_num"]), verbosity=1)
            current_stage = "ba_optimize_stage3"
            optimizer = PretestBAOptimizer(
                dataset=dataset,
                cam_params=cam_params,
                cams_cpp=cams_cpp,
                cam_to_window=cam_to_window,
                window_media=window_media,
                window_planes=window_planes,
                wand_length=float(shared["wand_length"]),
                config=cfg,
                progress_callback=None,
            )
            current_stage = "ba_optimize_stage3_loop_joint"
            wp_final, cp_final = _run_pretest_stage3(optimizer)
            cam_params = cp_final
            window_planes = wp_final
            print(f"[PRETEST][TRIAL {trial_id:03d}] Direct BA stage=3 done")

            current_stage = "metrics"
            final_metrics = _compute_final_metrics(
                dataset=dataset,
                cam_params=cam_params,
                cams_cpp=refr._init_cams_cpp_in_memory(cam_params, window_media, cam_to_window, window_planes),
                cam_to_window=cam_to_window,
                window_media=window_media,
                window_planes=window_planes,
                wand_length=float(shared["wand_length"]),
                cfg=cfg,
            )

            plane_summary = _summarize_plane_state(shared["plane_meta"], window_planes)
            cam_delta_summary = _summarize_camera_deltas(cam_params_ref, cam_params)
            print(f"[PRETEST][TRIAL {trial_id:03d}] Final metrics: {final_metrics}")
            for cid, cd in sorted(cam_delta_summary.items(), key=lambda kv: int(kv[0])):
                print(
                    f"  [CamDelta] Cam {cid}: drot={cd.get('rot_deg')} deg, dtrans={cd.get('trans_mm')} mm, "
                    f"df={cd.get('df')}, dk1={cd.get('dk1')}, dk2={cd.get('dk2')}"
                )

            result.update(
                {
                    "success": True,
                    "final_metrics": final_metrics,
                    "final_plane_summary": plane_summary,
                }
            )
            result["elapsed_sec"] = float(time.time() - start_t)
            _log_trial_footer(result)
    except Exception as e:
        result["error"] = repr(e)
        result["failed_stage"] = current_stage
        try:
            with log_path.open("a", encoding="utf-8") as log_fp:
                log_fp.write("\n[PRETEST][ERROR]\n")
                log_fp.write(f"failed_stage={current_stage}\n")
                log_fp.write(traceback.format_exc())
        except Exception:
            pass

    result["elapsed_sec"] = float(time.time() - start_t)
    return result


def _run_de_eval_worker(candidate: Dict[str, object]) -> Dict[str, object]:
    """Evaluate a single DE candidate (plane-depth vector) via loop-only BA.

    Mirrors ``_run_trial_worker`` but:
    * Accepts ``gen``, ``idx``, ``depths`` (one depth per window) instead of
      per-trial plane-point overrides.
    * Reuses the parent-computed P0 / bootstrap state from ``WORKER_SHARED``
      -- does **not** call ``PinholeBootstrapP0()`` or ``.run_all()``.
    * Runs ``_run_pretest_loop_only()`` (alternating loop only, NOT stage-3).
    * Returns a compact, serializable result dict suitable for DE scoring.
    """
    shared = WORKER_SHARED
    gen = int(candidate["gen"])
    idx = int(candidate["idx"])
    depths = list(candidate["depths"])  # [depth_w0, depth_w1, ...]
    plane_ids = [int(w) for w in shared["plane_ids"]]

    # Result structure -- kept small & serializable for multiprocessing
    result: Dict[str, object] = {
        "gen": gen,
        "idx": idx,
        "depths": depths,
        "success": False,
        "retry_status": str(candidate.get("retry_status", "not_requested")),
    }

    start_t = time.time()
    current_stage = "init"
    try:
        # Create log directory (flat layout -- no per-eval subdirs)
        out_dir = Path(shared["out_dir"])
        log_dir = out_dir / "de_eval_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"gen_{gen:03d}_eval_{idx:03d}.log"

        with log_path.open("w", encoding="utf-8") as log_fp, \
             contextlib.redirect_stdout(log_fp), contextlib.redirect_stderr(log_fp):
            try:
                try:
                    faulthandler.enable(file=log_fp, all_threads=True)
                except Exception:
                    pass

                print(f"[DE][GEN {gen:03d}][EVAL {idx:03d}] Start")
                print(f"  depths={depths}")

                # Reconstruct a RefractiveWandCalibrator for cams_cpp building
                # (lightweight -- no bootstrap, no run_all)
                _, refr = _create_base_and_refr(
                    csv_path=str(shared["csv"]),
                    wand_length=float(shared["wand_length"]),
                    dist_num=int(shared["dist_num"]),
                    cam_settings=copy.deepcopy(shared["cam_settings"]),
                )

                dataset = _deepcopy_dataset(shared["dataset"])
                cam_params = _copy_cam_params(shared["baseline_cam_params"])
                window_planes = _copy_window_planes(shared["baseline_window_planes"])
                window_media = copy.deepcopy(shared["window_media"])
                cam_to_window = {int(cid): int(wid) for cid, wid in shared["cam_to_window"].items()}

                # Inject DE plane depths via depths_to_plane_points
                plane_meta = shared["plane_meta"]
                new_pts = depths_to_plane_points(depths, plane_meta)
                current_stage = "validate_candidate"
                validation_error = _validate_de_candidate_geometry(depths, plane_meta, new_pts)
                if validation_error:
                    print(f"[DE][GEN {gen:03d}][EVAL {idx:03d}] Validation failed: {validation_error}")
                    result.update(
                        {
                            "error": validation_error,
                            "failed_stage": current_stage,
                            "ray_rmse_mm": float("inf"),
                            "log_path": str(log_path),
                            "elapsed_sec": float(time.time() - start_t),
                        }
                    )
                    return result
                for wid in plane_ids:
                    window_planes[wid]["plane_pt"] = new_pts[wid]

                print(f"[DE][GEN {gen:03d}][EVAL {idx:03d}] Injected plane depths")
                print(f"[DE][GEN {gen:03d}][EVAL {idx:03d}] Marker: before_cpp_camera_init")

                # Native-safety pre-flight check for each camera
                current_stage = "native_safety_check"
                for cid in sorted(cam_params.keys()):
                    wid = cam_to_window.get(cid)
                    p = cam_params[cid]
                    intr_kw = {'f': float(p[6]), 'cx': float(p[7]), 'cy': float(p[8]),
                               'dist': [float(p[9]), float(p[10])] + [0.0] * max(0, int(shared['dist_num']) - 2)}
                    ext_kw = {'rvec': p[0:3].tolist(), 'tvec': p[3:6].tolist()}
                    pl_kw = None
                    med_kw = None
                    if wid is not None and wid in window_planes:
                        wp = window_planes[wid]
                        pl_kw = {'pt': wp['plane_pt'].tolist() if hasattr(wp['plane_pt'], 'tolist') else list(wp['plane_pt']),
                                 'n': wp['plane_n'].tolist() if hasattr(wp['plane_n'], 'tolist') else list(wp['plane_n'])}
                    if wid is not None and wid in window_media:
                        med_kw = window_media[wid]
                    safety_errs = validate_native_safety(
                        intrinsics=intr_kw, extrinsics=ext_kw,
                        plane_geom=pl_kw, media_props=med_kw,
                    )
                    if safety_errs:
                        err_str = f"native_safety_cam{cid}:" + ";".join(safety_errs)
                        print(f"[DE][GEN {gen:03d}][EVAL {idx:03d}] {err_str}")
                        result.update({
                            "error": err_str,
                            "failed_stage": current_stage,
                            "ray_rmse_mm": float("inf"),
                            "log_path": str(log_path),
                            "elapsed_sec": float(time.time() - start_t),
                        })
                        return result

                # Build cams_cpp from injected state
                current_stage = "cpp_camera_init"
                cams_cpp = refr._init_cams_cpp_in_memory(cam_params, window_media, cam_to_window, window_planes)

                cfg = RefractiveBAConfig(stage=3, dist_coeff_num=int(shared["dist_num"]), verbosity=1)
                print(f"[DE][GEN {gen:03d}][EVAL {idx:03d}] Marker: before_optimizer_create")
                current_stage = "de_loop_only"
                optimizer = PretestBAOptimizer(
                    dataset=dataset,
                    cam_params=cam_params,
                    cams_cpp=cams_cpp,
                    cam_to_window=cam_to_window,
                    window_media=window_media,
                    window_planes=window_planes,
                    wand_length=float(shared["wand_length"]),
                    config=cfg,
                    progress_callback=None,
                )

                # Run alternating loop only (NOT _run_pretest_stage3)
                print(f"[DE][GEN {gen:03d}][EVAL {idx:03d}] Marker: before_loop_only")
                wp_final, cp_final, loop_info = _run_pretest_loop_only(optimizer)
                cam_params = cp_final
                window_planes = wp_final

                print(
                    f"[DE][GEN {gen:03d}][EVAL {idx:03d}] Loop done: "
                    f"iters={loop_info['loop_iters']}, hit_boundary={loop_info['hit_boundary']}"
                )

                # Compute compact final metrics
                current_stage = "metrics"
                print(f"[DE][GEN {gen:03d}][EVAL {idx:03d}] Marker: before_final_metrics")
                final_metrics = _compute_final_metrics(
                    dataset=dataset,
                    cam_params=cam_params,
                    cams_cpp=refr._init_cams_cpp_in_memory(cam_params, window_media, cam_to_window, window_planes),
                    cam_to_window=cam_to_window,
                    window_media=window_media,
                    window_planes=window_planes,
                    wand_length=float(shared["wand_length"]),
                    cfg=cfg,
                )

                ray_rmse = float(final_metrics.get("ray_rmse_mm", float("inf")))
                print(f"[DE][GEN {gen:03d}][EVAL {idx:03d}] Final ray_rmse_mm={ray_rmse:.6f}")

                result.update({
                    "success": True,
                    "ray_rmse_mm": ray_rmse,
                    "loop_iters": loop_info["loop_iters"],
                    "hit_boundary": loop_info["hit_boundary"],
                    "log_path": str(log_path),
                    "elapsed_sec": float(time.time() - start_t),
                })
            finally:
                try:
                    faulthandler.disable()
                except Exception:
                    pass

    except Exception as e:
        result["error"] = repr(e)
        result["failed_stage"] = current_stage
        result["ray_rmse_mm"] = float("inf")
        result["elapsed_sec"] = float(time.time() - start_t)
        result["log_path"] = str(log_path) if 'log_path' in locals() else ""

    return result


def _de_candidate_key(depths: List[float], decimals: int = 4) -> Tuple[float, ...]:
    return tuple(float(round(float(v), int(decimals))) for v in depths)


def _latent_to_depth(
    z: np.ndarray,
    d_mid: np.ndarray,
    d_half: np.ndarray,
) -> np.ndarray:
    """Map unconstrained latent *z* to physical depth via ``d = d_mid + d_half * tanh(z)``."""
    return d_mid + d_half * np.tanh(np.asarray(z, dtype=np.float64))


def _depth_to_latent(
    d: np.ndarray,
    d_mid: np.ndarray,
    d_half: np.ndarray,
    clamp_eps: float = 1e-7,
) -> np.ndarray:
    """Inverse of :func:`_latent_to_depth`: ``z = atanh((d - d_mid) / d_half)``.

    The argument to ``atanh`` is clamped to ``(-1+eps, 1-eps)`` to avoid
    infinities when *d* sits exactly on a bound.
    """
    d_arr = np.asarray(d, dtype=np.float64)
    d_half_safe = np.where(d_half > 0, d_half, 1.0)  # guard div-by-zero for degenerate intervals
    ratio = (d_arr - d_mid) / d_half_safe
    ratio = np.clip(ratio, -1.0 + clamp_eps, 1.0 - clamp_eps)
    return np.arctanh(ratio)


def _compute_latent_params(
    bounds_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """From bounds ``[[d_min, d_max], ...]`` compute ``(d_mid, d_half)`` arrays.

    Guards degenerate intervals where ``d_max <= d_min`` by falling back to
    ``d_mid = d_min, d_half = 1.0`` (a safe unit-scale mapping).
    """
    lo = bounds_arr[:, 0]
    hi = bounds_arr[:, 1]
    d_mid = 0.5 * (lo + hi)
    d_half = 0.5 * (hi - lo)
    # Guard degenerate intervals
    degenerate = d_half <= 0
    d_mid = np.where(degenerate, lo, d_mid)
    d_half = np.where(degenerate, 1.0, d_half)
    return d_mid, d_half

def _build_initial_de_population(
    baseline_depths: List[float],
    d_mid: np.ndarray,
    d_half: np.ndarray,
    popsize: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build initial DE population in **latent z space**.

    The first member is the baseline depths mapped through the inverse
    ``depth_to_latent`` transform; remaining members are sampled **uniformly
    in physical depth** within each ``[d_min, d_max]`` interval and then
    inverse-mapped to latent ``z`` via :func:`_depth_to_latent`.
    """
    dim = int(d_mid.shape[0])
    pop = np.zeros((int(popsize), dim), dtype=np.float64)
    # First member: baseline depths -> latent via atanh inverse
    baseline_arr = np.asarray(baseline_depths, dtype=np.float64)
    # Clamp baseline depths into the valid physical interval before converting
    lo_phys = d_mid - d_half
    hi_phys = d_mid + d_half
    baseline_clamped = np.clip(baseline_arr, lo_phys, hi_phys)
    pop[0, :] = _depth_to_latent(baseline_clamped, d_mid, d_half)
    if popsize > 1:
        # Sample uniformly in physical depth, then inverse-map to latent z
        phys_samples = rng.uniform(
            lo_phys, hi_phys, size=(int(popsize) - 1, dim)
        )
        pop[1:, :] = _depth_to_latent(phys_samples, d_mid, d_half)
    return pop

def _compact_de_result(res: Dict[str, object], gen: int, idx: int, depths: List[float]) -> Dict[str, object]:
    score = float(res.get("ray_rmse_mm", float("inf")))
    if (not bool(res.get("success", False))) or (not np.isfinite(score)):
        score = float("inf")
    return {
        "gen": int(res.get("gen", gen)),
        "idx": int(res.get("idx", idx)),
        "depths": [float(v) for v in res.get("depths", depths)],
        "score": float(score),
        "success": bool(res.get("success", False)),
        "error": str(res.get("error", "")),
        "failed_stage": str(res.get("failed_stage", "")),
        "loop_iters": int(res.get("loop_iters", -1)),
        "hit_boundary": bool(res.get("hit_boundary", False)),
        "elapsed_sec": float(res.get("elapsed_sec", float("nan"))),
        "log_path": str(res.get("log_path", "")),
        "retry_status": str(res.get("retry_status", "not_requested")),
    }


def _run_de_search(shared_setup: Dict[str, object], args) -> Dict[str, object]:
    plane_ids = [int(w) for w in sorted(shared_setup["plane_ids"])]
    plane_meta = shared_setup["plane_meta"]

    def _arg(name: str, default):
        value = getattr(args, name, default)
        return default if value is None else value

    popsize = int(_arg("de_popsize", 12))
    maxiter = int(_arg("de_maxiter", 20))
    patience = int(_arg("de_patience", 4))
    abs_tol = float(_arg("de_abs_tol", 1e-4))
    rel_tol = float(_arg("de_rel_tol", 1e-4))
    F = float(_arg("de_F", 0.7))
    CR = float(_arg("de_CR", 0.8))
    cache_decimals = int(_arg("de_cache_round", 4))
    seed = _arg("de_seed", None)
    rng = np.random.default_rng(None if seed is None else int(seed))

    bounds = _build_de_bounds(plane_meta)
    bounds_metadata = copy.deepcopy(getattr(_build_de_bounds, "last_metadata", {}))
    bounds_arr = np.asarray([bounds[wid] for wid in plane_ids], dtype=np.float64)
    dim = int(bounds_arr.shape[0])
    print("[PRETEST][DE] Depth ranges before search:")
    for wid in plane_ids:
        meta = plane_meta[wid]
        lower, upper = bounds[wid]
        print(
            f"  Window {wid}: depth_range_mm=[{float(lower):.6f}, {float(upper):.6f}] "
            f"baseline_depth_mm={float(meta['baseline_depth_mm']):.6f} "
            f"strategy={meta.get('depth_bound_strategy', 'unknown')} "
            f"positive_endpoint_depth_count={int(meta.get('positive_endpoint_depth_count', 0))} "
            f"nearest_positive_depth_mm={float(meta.get('nearest_positive_depth_mm', float('nan'))):.6f}"
        )

    if popsize < 4:
        popsize = 4

    baseline_depths = [float(plane_meta[wid]["baseline_depth_mm"]) for wid in plane_ids]

    # Compute latent-space parameterization: d = d_mid + d_half * tanh(z)
    d_mid, d_half = _compute_latent_params(bounds_arr)
    population = _build_initial_de_population(
        baseline_depths=baseline_depths, d_mid=d_mid, d_half=d_half,
        popsize=popsize, rng=rng,
    )
    cpu_count = os.cpu_count() or mp.cpu_count() or 1
    max_workers_arg = getattr(args, "max_workers", None)
    if max_workers_arg is not None:
        max_workers = max(1, int(max_workers_arg))
    else:
        worker_frac = float(getattr(args, "worker_frac", 1.0))
        max_workers = max(1, int(math.floor(float(cpu_count) * worker_frac)))
    max_workers = min(max_workers, popsize)

    eval_cache: Dict[Tuple[float, ...], Dict[str, object]] = {}
    eval_rows: List[Dict[str, object]] = []
    generation_rows: List[Dict[str, object]] = []
    cumulative_eval_count = 0

    def _make_failed_row(gen: int, idx: int, depths: List[float], error: str, failed_stage: str, retry_status: str = "not_requested") -> Dict[str, object]:
        return {
            "gen": int(gen),
            "idx": int(idx),
            "depths": [float(v) for v in depths],
            "score": float("inf"),
            "success": False,
            "error": str(error),
            "failed_stage": str(failed_stage),
            "loop_iters": -1,
            "hit_boundary": False,
            "elapsed_sec": float("nan"),
            "log_path": "",
            "retry_status": str(retry_status),
        }

    def _append_eval_row(compact: Dict[str, object], cached: bool, submitted: bool, resolved_by_cache: bool) -> None:
        eval_rows.append(
            {
                "gen": int(compact["gen"]),
                "idx": int(compact["idx"]),
                "depths": list(compact["depths"]),
                "score": float(compact["score"]),
                "success": bool(compact["success"]),
                "cached": bool(cached),
                "submitted": bool(submitted),
                "resolved_by_cache": bool(resolved_by_cache),
                "error": str(compact.get("error", "")),
                "failed_stage": str(compact.get("failed_stage", "")),
                "retry_status": str(compact.get("retry_status", "not_requested")),
                "log_path": str(compact.get("log_path", "")),
            }
        )

    def _serial_retry(gen: int, idx: int, depths: List[float]) -> Dict[str, object]:
        print(f"[PRETEST][DE] Serial retry gen={gen} idx={idx} depths={depths}")
        payload = {"gen": int(gen), "idx": int(idx), "depths": list(depths), "retry_status": "serial_retry"}
        try:
            mp_ctx_retry = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=mp_ctx_retry,
                initializer=_init_worker,
                initargs=(shared_setup,),
            ) as retry_executor:
                res = retry_executor.submit(_run_de_eval_worker, payload).result()
            compact = _compact_de_result(res, gen=gen, idx=idx, depths=depths)
            compact["retry_status"] = "serial_retry_success" if compact["success"] else "serial_retry_failed"
            return compact
        except Exception as e:
            return _make_failed_row(gen, idx, depths, repr(e), "serial_retry", retry_status="serial_retry_failed")

    def _evaluate_candidates(gen: int, candidate_arr: np.ndarray) -> Tuple[List[Dict[str, object]], bool, int, int]:
        nonlocal cumulative_eval_count
        rows: List[Dict[str, object]] = [None] * int(candidate_arr.shape[0])
        pending_by_idx: Dict[int, Tuple[Tuple[float, ...], List[float]]] = {}
        infra_failure = False
        retry_success_count = 0
        retry_failure_count = 0

        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_ctx,
            initializer=_init_worker,
            initargs=(shared_setup,),
        ) as executor:
            futures = {}
            for idx in range(int(candidate_arr.shape[0])):
                # candidate_arr is in latent z space; convert to physical depths
                z_vec = candidate_arr[idx, :]
                phys_depths = _latent_to_depth(z_vec, d_mid, d_half)
                depths = [float(v) for v in phys_depths.tolist()]
                key = _de_candidate_key(depths, decimals=cache_decimals)
                # Cache disabled for diagnostic search -- always submit
                cached = None

                print(f"[PRETEST][DE] gen={gen:03d} idx={idx:03d} submit key={key} depths={depths}")
                payload = {"gen": int(gen), "idx": int(idx), "depths": depths, "retry_status": "not_requested"}
                future = executor.submit(_run_de_eval_worker, payload)
                futures[future] = (idx, key, depths)
                pending_by_idx[idx] = (key, depths)

            resolved_idx = set()
            pool_error = None
            try:
                for future in as_completed(futures):
                    idx, key, depths = futures[future]
                    try:
                        res = future.result()
                        compact = _compact_de_result(res, gen=gen, idx=idx, depths=depths)
                    except BrokenProcessPool as e:
                        pool_error = e
                        infra_failure = True
                        break
                    except Exception as e:
                        compact = _make_failed_row(gen, idx, depths, repr(e), "de_eval_future")

                    rows[idx] = compact
                    resolved_idx.add(idx)
                    cumulative_eval_count += 1
                    # Cache writes disabled for diagnostic search
                    _append_eval_row(compact, cached=False, submitted=True, resolved_by_cache=False)
            except BrokenProcessPool as e:
                pool_error = e
                infra_failure = True

            if infra_failure:
                error_text = repr(pool_error) if pool_error is not None else "BrokenProcessPool()"
                print(f"[PRETEST][DE] gen={gen:03d} infrastructure failure: {error_text}")
                unresolved = [(idx, meta[0], meta[1]) for idx, meta in pending_by_idx.items() if idx not in resolved_idx]
                for idx, key, depths in unresolved:
                    compact = _make_failed_row(gen, idx, depths, error_text, "de_eval_future", retry_status="pending_serial_retry")
                    retry_compact = _serial_retry(gen, idx, depths)
                    if retry_compact.get("success"):
                        compact = retry_compact
                        retry_success_count += 1
                    else:
                        retry_compact["retry_status"] = str(retry_compact.get("retry_status", "serial_retry_failed"))
                        compact = retry_compact
                        retry_failure_count += 1
                    rows[idx] = compact
                    cumulative_eval_count += 1
                    # Cache writes disabled for diagnostic search
                    _append_eval_row(compact, cached=False, submitted=True, resolved_by_cache=False)

        final_rows = [dict(r) for r in rows if r is not None]
        if len(final_rows) != int(candidate_arr.shape[0]):
            raise RuntimeError(f"DE generation {gen} produced {len(final_rows)} rows for {candidate_arr.shape[0]} candidates")
        return final_rows, infra_failure, retry_success_count, retry_failure_count

    initial_eval_rows, initial_infra_failure, initial_retry_success, initial_retry_failure = _evaluate_candidates(gen=0, candidate_arr=population)
    scores = np.asarray([float(row["score"]) for row in initial_eval_rows], dtype=np.float64)

    best_idx = int(np.argmin(scores))
    best_score = float(scores[best_idx])
    best_depths = [float(v) for v in _latent_to_depth(population[best_idx, :], d_mid, d_half).tolist()]
    best_result = dict(initial_eval_rows[best_idx])

    initial_failures = int(np.sum([0 if row.get("success") else 1 for row in initial_eval_rows]))
    # Compute physical depths for population for depth diagnostics
    _pop_phys_0 = _latent_to_depth(population, d_mid, d_half)
    generation_rows.append(
        {
            "gen": 0,
            "best_score": float(np.min(scores)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "failure_count": int(initial_failures),
            "cumulative_eval_count": int(cumulative_eval_count),
            "best_candidate_depths": list(best_depths),
            "infra_failure": bool(initial_infra_failure),
            "retry_success_count": int(initial_retry_success),
            "retry_failure_count": int(initial_retry_failure),
            "submitted_count": int(popsize),
            "invalid_count": int(initial_failures),
            "accepted_count": 0,
            "acceptance_rate": 0.0,
            "cache_hit_count": 0,
            "pop_depth_min": float(np.min(_pop_phys_0)),
            "pop_depth_max": float(np.max(_pop_phys_0)),
        }
    )

    no_improve_gens = 0
    for gen in range(1, maxiter + 1):
        trial_population = np.zeros_like(population)
        for i in range(popsize):
            pool = [j for j in range(popsize) if j != i]
            a_idx, b_idx, c_idx = rng.choice(pool, size=3, replace=False)
            mutant = population[a_idx, :] + F * (population[b_idx, :] - population[c_idx, :])
            # No clipping: mutation/crossover stays in unconstrained latent z space

            cross_mask = rng.random(dim) < CR
            cross_mask[int(rng.integers(0, dim))] = True
            trial = np.where(cross_mask, mutant, population[i, :])
            trial_population[i, :] = trial

        trial_eval_rows, infra_failure, retry_success_count, retry_failure_count = _evaluate_candidates(gen=gen, candidate_arr=trial_population)
        trial_scores = np.asarray([float(row["score"]) for row in trial_eval_rows], dtype=np.float64)
        trial_failures = int(np.sum([0 if row.get("success") else 1 for row in trial_eval_rows]))

        accepted_count = 0
        for i in range(popsize):
            if np.isfinite(float(trial_scores[i])) and (float(trial_scores[i]) < float(scores[i])):
                population[i, :] = trial_population[i, :]
                scores[i] = float(trial_scores[i])
                accepted_count += 1

        gen_best_idx = int(np.argmin(scores))
        gen_best_score = float(scores[gen_best_idx])
        gen_best_depths = [float(v) for v in _latent_to_depth(population[gen_best_idx, :], d_mid, d_half).tolist()]
        # Compute physical depths for population for depth diagnostics
        _pop_phys_gen = _latent_to_depth(population, d_mid, d_half)
        generation_rows.append(
            {
                "gen": int(gen),
                "best_score": float(gen_best_score),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "failure_count": int(trial_failures),
                "cumulative_eval_count": int(cumulative_eval_count),
                "best_candidate_depths": list(gen_best_depths),
                "infra_failure": bool(infra_failure),
                "retry_success_count": int(retry_success_count),
                "retry_failure_count": int(retry_failure_count),
                "submitted_count": int(popsize),
                "invalid_count": int(trial_failures),
                "accepted_count": int(accepted_count),
                "acceptance_rate": float(accepted_count) / float(popsize),
                "cache_hit_count": 0,
                "pop_depth_min": float(np.min(_pop_phys_gen)),
                "pop_depth_max": float(np.max(_pop_phys_gen)),
            }
        )

        if gen_best_score < best_score:
            best_score = float(gen_best_score)
            best_depths = list(gen_best_depths)
            best_result = dict(trial_eval_rows[gen_best_idx])

        prev_best = float(generation_rows[-2]["best_score"])
        improvement = max(0.0, prev_best - gen_best_score)
        rel_improvement = improvement / max(abs(prev_best), 1e-12)
        if infra_failure:
            print(f"[PRETEST][DE] gen={gen:03d} skipped patience update due to infrastructure failure")
        elif (improvement < abs_tol) and (rel_improvement < rel_tol):
            no_improve_gens += 1
        else:
            no_improve_gens = 0

        if no_improve_gens >= patience:
            break

    return {
        "best_result": {
            "depths": list(best_depths),
            "score": float(best_score),
            "eval": dict(best_result),
        },
        "generation_rows": generation_rows,
        "eval_rows": eval_rows,
        "bounds": {str(wid): [float(bounds[wid][0]), float(bounds[wid][1])] for wid in plane_ids},
        "bounds_metadata": _serialize_np(bounds_metadata),
        "latent_reparameterization": {
            "method": "tanh",
            "d_mid": [float(v) for v in d_mid.tolist()],
            "d_half": [float(v) for v in d_half.tolist()],
            "description": "DE evolves unconstrained z; physical depth = d_mid + d_half * tanh(z)",
        },
        "config": {
            "plane_ids": [int(w) for w in plane_ids],
            "de_popsize": int(popsize),
            "de_maxiter": int(maxiter),
            "de_patience": int(patience),
            "de_abs_tol": float(abs_tol),
            "de_rel_tol": float(rel_tol),
            "de_F": float(F),
            "de_CR": float(CR),
            "de_cache_round": int(cache_decimals),
            "de_seed": None if seed is None else int(seed),
            "max_workers": int(max_workers),
        },
    }

def _flatten_result_for_csv(result: Dict[str, object], plane_ids: List[int]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "trial_id": int(result["trial_id"]),
        "trial_name": str(result["trial_name"]),
        "success": bool(result["success"]),
        "elapsed_sec": float(result.get("elapsed_sec", float("nan"))),
        "stage": int(result.get("stage", -1)),
        "log_path": str(result.get("log_path", "")),
        "error": str(result.get("error", "")),
    }
    final_metrics = result.get("final_metrics") or {}
    row["final_ray_rmse_mm"] = final_metrics.get("ray_rmse_mm")
    row["final_len_rmse_mm"] = final_metrics.get("len_rmse_mm")
    row["final_proj_rmse_px"] = final_metrics.get("proj_rmse_px")

    plane_indices = result.get("plane_indices") or {}
    init_depths = result.get("initial_depths_mm") or {}
    plane_summary = result.get("final_plane_summary") or {}
    for wid in plane_ids:
        key = str(wid)
        row[f"plane{wid}_init_index"] = plane_indices.get(key)
        row[f"plane{wid}_init_depth_mm"] = init_depths.get(key)
        ps = plane_summary.get(key) or {}
        row[f"plane{wid}_final_depth_mm"] = ps.get("final_depth_mm")
        row[f"plane{wid}_depth_delta_mm"] = ps.get("depth_delta_mm")
        row[f"plane{wid}_angle_delta_deg"] = ps.get("angle_delta_deg")
    return row


def _write_results_csv(csv_path: Path, results: List[Dict[str, object]], plane_ids: List[int]) -> None:
    rows = [_flatten_result_for_csv(r, plane_ids) for r in results]
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_de_results_csv(csv_path: Path, eval_rows: List[Dict[str, object]], plane_ids: List[int]) -> None:
    if not eval_rows:
        return
    fieldnames = ["gen", "idx"]
    for i, wid in enumerate(plane_ids):
        fieldnames.append(f"plane{wid}_depth_mm")
    fieldnames.extend(["score", "success", "cached", "submitted", "resolved_by_cache", "retry_status", "failed_stage", "log_path", "error"])

    rows = []
    for row in eval_rows:
        d = {
            "gen": row["gen"],
            "idx": row["idx"],
            "score": row["score"],
            "success": row["success"],
            "cached": row["cached"],
            "submitted": row.get("submitted", False),
            "resolved_by_cache": row.get("resolved_by_cache", False),
            "retry_status": row.get("retry_status", "not_requested"),
            "failed_stage": row.get("failed_stage", ""),
            "log_path": row.get("log_path", ""),
            "error": row["error"],
        }
        depths = row["depths"]
        for i, wid in enumerate(plane_ids):
            d[f"plane{wid}_depth_mm"] = depths[i]
        rows.append(d)

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_de_generation_csv(csv_path: Path, gen_rows: List[Dict[str, object]], plane_ids: List[int]) -> None:
    if not gen_rows:
        return
    fieldnames = [
        "gen",
        "best_score",
        "mean_score",
        "std_score",
        "failure_count",
        "cumulative_eval_count",
        "infra_failure",
        "retry_success_count",
        "retry_failure_count",
        "submitted_count",
        "invalid_count",
        "accepted_count",
        "acceptance_rate",
        "cache_hit_count",
        "pop_depth_min",
        "pop_depth_max",
    ]
    for i, wid in enumerate(plane_ids):
        fieldnames.append(f"best_plane{wid}_depth_mm")

    rows = []
    for row in gen_rows:
        d = {
            "gen": row["gen"],
            "best_score": row["best_score"],
            "mean_score": row["mean_score"],
            "std_score": row["std_score"],
            "failure_count": row["failure_count"],
            "cumulative_eval_count": row["cumulative_eval_count"],
            "infra_failure": row.get("infra_failure", False),
            "retry_success_count": row.get("retry_success_count", 0),
            "retry_failure_count": row.get("retry_failure_count", 0),
            "submitted_count": row.get("submitted_count", 0),
            "invalid_count": row.get("invalid_count", 0),
            "accepted_count": row.get("accepted_count", 0),
            "acceptance_rate": row.get("acceptance_rate", 0.0),
            "cache_hit_count": row.get("cache_hit_count", 0),
            "pop_depth_min": row.get("pop_depth_min", float("nan")),
            "pop_depth_max": row.get("pop_depth_max", float("nan")),
        }
        depths = row["best_candidate_depths"]
        for i, wid in enumerate(plane_ids):
            d[f"best_plane{wid}_depth_mm"] = depths[i]
        rows.append(d)

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def _build_shared_setup(args) -> Dict[str, object]:
    cam_settings = _camera_settings_default()
    cam_to_window = _cam_to_window_default()
    window_media = _window_media_default()

    base, refr = _create_base_and_refr(
        csv_path=str(args.csv),
        wand_length=float(args.wand_length),
        dist_num=int(args.dist_num),
        cam_settings=cam_settings,
    )

    observed_cam_ids = sorted(list({cid for _, frame in base.wand_points.items() for cid in frame.keys()}))
    cam_settings, cam_to_window = _try_align_cam_ids(cam_settings, cam_to_window, observed_cam_ids)

    base.camera_settings = cam_settings
    base.cams = {cid: {} for cid in cam_settings.keys()}
    base.cameras = base.cams
    base.active_cam_ids = sorted(cam_settings.keys())
    first_cid = base.active_cam_ids[0]
    base.image_size = (cam_settings[first_cid]["height"], cam_settings[first_cid]["width"])
    refr = RefractiveWandCalibrator(base)

    dataset = refr._collect_observations(cam_to_window)
    observations = refr._prepare_observations_for_bootstrap(cam_to_window)

    p0_cam_err: Dict[int, float] = {}
    try:
        base.run_precalibration_check(
            wand_length=float(args.wand_length),
            init_focal_length=cam_settings[first_cid]["focal"],
        )
        p0_cam_err = _calc_p0_cam_error(base)
    except Exception as e:
        print(f"[PRETEST][WARN] run_precalibration_check failed, fallback to shared-frame seed selection: {e}")

    bootstrap = PinholeBootstrapP0(PinholeBootstrapP0Config(wand_length_mm=float(args.wand_length)))
    cams_all = dataset["cam_ids"]
    if len(cams_all) < 2:
        raise RuntimeError("Need at least 2 cameras")
    p0_pair = (cams_all[0], cams_all[1])
    cam_params_p0, report = bootstrap.run_all(
        cam_i=p0_pair[0],
        cam_j=p0_pair[1],
        observations=observations,
        camera_settings=cam_settings,
        all_cam_ids=cams_all,
        progress_callback=None,
    )

    cam_params = {}
    for cid, p in cam_params_p0.items():
        f, cx, cy = refr._get_cam_intrinsics(cid)
        cam_params[cid] = np.concatenate([p, [f, cx, cy, 0.0, 0.0]])

    points_3d = report.get("points_3d", {})
    XA = {fid: ab[0] for fid, ab in points_3d.items()}
    XB = {fid: ab[1] for fid, ab in points_3d.items()}
    dataset["X_A_bootstrap"] = XA
    dataset["X_B_bootstrap"] = XB

    window_planes = refr._init_window_planes_from_cameras(
        cam_params=cam_params,
        cam_to_window=cam_to_window,
        window_media=window_media,
        err_px=p0_cam_err,
        X_A_list=XA,
        X_B_list=XB,
        active_cam_ids=cams_all,
    )

    if p0_cam_err:
        seed_wid, seed_c1, seed_c2 = _select_seed_pair_same_window(p0_cam_err, cam_to_window, cams_all)
    else:
        seed_wid, seed_c1, seed_c2 = _fallback_seed_pair_by_shared_frames(base.wand_points, cam_to_window, cams_all)

    plane_ids = sorted(int(wid) for wid in window_planes.keys())
    if len(plane_ids) != 2:
        raise RuntimeError(f"Expected exactly 2 planes/windows for this pretest, got {plane_ids}")

    plane_meta = {
        wid: _build_plane_position_candidates(
            cam_params=cam_params,
            cam_to_window=cam_to_window,
            window_planes=window_planes,
            XA=XA,
            XB=XB,
            target_wid=wid,
            num_positions=int(args.num_positions),
        )
        for wid in plane_ids
    }

    return {
        "csv": str(args.csv),
        "wand_length": float(args.wand_length),
        "dist_num": int(args.dist_num),
        "cam_settings": copy.deepcopy(cam_settings),
        "cam_to_window": {int(cid): int(wid) for cid, wid in cam_to_window.items()},
        "window_media": copy.deepcopy(window_media),
        "dataset": _deepcopy_dataset(dataset),
        "baseline_cam_params": _copy_cam_params(cam_params),
        "baseline_window_planes": _copy_window_planes(window_planes),
        "seed_wid": int(seed_wid),
        "seed_pair": [int(seed_c1), int(seed_c2)],
        "plane_ids": [int(wid) for wid in plane_ids],
        "plane_meta": plane_meta,
        "p0_cam_error_px": {str(k): float(v) for k, v in sorted(p0_cam_err.items())},
        "out_dir": str(args.out),
    }


def _build_trials(shared_setup: Dict[str, object]) -> List[Dict[str, object]]:
    plane_ids = [int(w) for w in shared_setup["plane_ids"]]
    plane0, plane1 = plane_ids[0], plane_ids[1]
    pos0 = shared_setup["plane_meta"][plane0]["positions"]
    pos1 = shared_setup["plane_meta"][plane1]["positions"]
    trials = []
    tid = 0
    for rec0 in pos0:
        for rec1 in pos1:
            tid += 1
            trials.append(
                {
                    "trial_id": tid,
                    "trial_name": f"trial_{tid:03d}_w{plane0}i{int(rec0['index'])}_w{plane1}i{int(rec1['index'])}",
                    "plane_indices": {plane0: int(rec0["index"]), plane1: int(rec1["index"])},
                    "initial_depths_mm": {plane0: float(rec0["depth_mm"]), plane1: float(rec1["depth_mm"])},
                    "initial_plane_points": {
                        plane0: np.asarray(rec0["plane_pt"], dtype=np.float64),
                        plane1: np.asarray(rec1["plane_pt"], dtype=np.float64),
                    },
                }
            )
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretest global search by sweeping initial plane positions for both planes")
    parser.add_argument("--mode", type=str, default="sweep", choices=["sweep", "de", "both"], help="Search mode")
    parser.add_argument("--csv", type=str, default=r"J:\Fish\T0\R2\wandpoints_filtered2.csv")
    parser.add_argument("--out", type=str, default=r"J:\Fish\T0\R2\GlobalSearchPretest")
    parser.add_argument("--wand-length", type=float, default=10.0)
    parser.add_argument("--dist-num", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--num-positions", type=int, default=5, help="Number of sampled initial positions per plane")
    parser.add_argument("--worker-frac", type=float, default=0.80, help="Fraction of CPU cores to use concurrently")
    parser.add_argument("--de-popsize", type=int, default=12)
    parser.add_argument("--de-maxiter", type=int, default=20)
    parser.add_argument("--de-patience", type=int, default=4)
    parser.add_argument("--de-abs-tol", type=float, default=1e-4)
    parser.add_argument("--de-rel-tol", type=float, default=1e-4)
    parser.add_argument("--de-F", type=float, default=0.7)
    parser.add_argument("--de-CR", type=float, default=0.8)
    parser.add_argument("--de-seed", type=int, default=None)
    parser.add_argument("--de-cache-round", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    shared_setup = _build_shared_setup(args)
    plane_ids = [int(w) for w in shared_setup["plane_ids"]]
    
    sweep_best_score = float("inf")
    sweep_best_name = None
    de_best_score = float("inf")
    de_best_depths = None

    # SWEEP PATH
    if args.mode in ["sweep", "both"]:
        trials = _build_trials(shared_setup)
        cpu_count = os.cpu_count() or mp.cpu_count() or 1
        max_workers = max(1, min(len(trials), int(math.floor(cpu_count * float(args.worker_frac)))))
        
        summary_setup = {
            "csv": str(args.csv),
            "out": str(args.out),
            "wand_length": float(args.wand_length),
            "dist_num": int(args.dist_num),
            "num_positions": int(args.num_positions),
            "cpu_count": int(cpu_count),
            "worker_frac": float(args.worker_frac),
            "max_workers": int(max_workers),
            "stage": 3,
            "weak_window_geo_init_disabled": True,
            "p0_cam_error_px": dict(shared_setup["p0_cam_error_px"]),
            "plane_sampling": {
                str(wid): {
                    "window_id": int(wid),
                    "camera_ids": [int(cid) for cid in shared_setup["plane_meta"][wid]["camera_ids"]],
                    "baseline_depth_mm": float(shared_setup["plane_meta"][wid]["baseline_depth_mm"]),
                    "depth_min_mm": float(shared_setup["plane_meta"][wid]["depth_min_mm"]),
                    "depth_max_mm": float(shared_setup["plane_meta"][wid]["depth_max_mm"]),
                    "depth_bound_strategy": str(shared_setup["plane_meta"][wid].get("depth_bound_strategy", "unknown")),
                    "depth_bound_fallback_used": bool(shared_setup["plane_meta"][wid].get("depth_bound_fallback_used", False)),
                    "depth_bound_fallback_reason": str(shared_setup["plane_meta"][wid].get("depth_bound_fallback_reason", "none")),
                    "nearest_positive_depth_mm": float(shared_setup["plane_meta"][wid].get("nearest_positive_depth_mm", float("nan"))),
                    "positive_endpoint_depth_count": int(shared_setup["plane_meta"][wid].get("positive_endpoint_depth_count", 0)),
                    "nearest_baseline_index": int(shared_setup["plane_meta"][wid]["nearest_baseline_index"]),
                    "positions": [
                        {
                            "index": int(rec["index"]),
                            "depth_mm": float(rec["depth_mm"]),
                            "plane_pt": _serialize_np(rec["plane_pt"]),
                        }
                        for rec in shared_setup["plane_meta"][wid]["positions"]
                    ],
                }
                for wid in plane_ids
            },
            "trial_count": int(len(trials)),
        }
        (out_dir / "pretest_setup.json").write_text(json.dumps(summary_setup, indent=2), encoding="utf-8")

        print(f"[PRETEST][SWEEP] Direct BA stage=3, weak-window geometric init disabled")
        print(f"[PRETEST][SWEEP] Plane windows={plane_ids}, trials={len(trials)}, max_workers={max_workers}")

        results: List[Dict[str, object]] = []
        start_t = time.time()
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_ctx,
            initializer=_init_worker,
            initargs=(shared_setup,),
        ) as executor:
            future_to_trial = {executor.submit(_run_trial_worker, trial): trial for trial in trials}
            for idx, future in enumerate(as_completed(future_to_trial), start=1):
                trial = future_to_trial[future]
                try:
                    res = future.result()
                except Exception as e:
                    res = {
                        "trial_id": int(trial["trial_id"]),
                        "trial_name": str(trial["trial_name"]),
                        "success": False,
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    }
                results.append(res)
                status = "OK" if res.get("success") else "FAIL"
                ray = (res.get("final_metrics") or {}).get("ray_rmse_mm")
                proj = (res.get("final_metrics") or {}).get("proj_rmse_px")
                print(
                    f"[PRETEST][SWEEP] Completed {idx:02d}/{len(trials)} {res.get('trial_name', trial['trial_name'])} "
                    f"status={status} ray={ray} proj={proj} elapsed={res.get('elapsed_sec')}"
                )

        results = sorted(results, key=lambda r: int(r.get("trial_id", 0)))
        success_results = [r for r in results if r.get("success")]
        best_by_ray = None
        if success_results:
            best_by_ray = min(success_results, key=lambda r: float((r.get("final_metrics") or {}).get("ray_rmse_mm", float("inf"))))
            if best_by_ray:
                sweep_best_score = float((best_by_ray.get("final_metrics") or {}).get("ray_rmse_mm", float("inf")))
                sweep_best_name = str(best_by_ray["trial_name"])

        summary = {
            "setup": summary_setup,
            "runtime_sec": float(time.time() - start_t),
            "successful_trials": int(len(success_results)),
            "failed_trials": int(len(results) - len(success_results)),
            "best_trial_by_ray": None if best_by_ray is None else {
                "trial_id": int(best_by_ray["trial_id"]),
                "trial_name": str(best_by_ray["trial_name"]),
                "final_metrics": _serialize_np(best_by_ray.get("final_metrics")),
            },
            "results": _serialize_np(results),
        }

        summary_path = out_dir / "global_search_pretest_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        csv_path = out_dir / "global_search_pretest_results.csv"
        _write_results_csv(csv_path, results, plane_ids)
        print(f"[PRETEST][SWEEP] Done. Summary: {summary_path}")

    # DE PATH
    if args.mode in ["de", "both"]:
        print(f"[PRETEST][DE] Starting Differential Evolution search ...")
        de_log_path = out_dir / "de_search.log"
        with de_log_path.open("w", encoding="utf-8") as log_fp, contextlib.redirect_stdout(log_fp), contextlib.redirect_stderr(log_fp):
            de_results = _run_de_search(shared_setup, args)
        
        # Write artifacts
        (out_dir / "de_search_summary.json").write_text(json.dumps(_serialize_np(de_results), indent=2), encoding="utf-8")
        _write_de_results_csv(out_dir / "de_search_results.csv", de_results["eval_rows"], plane_ids)
        _write_de_generation_csv(out_dir / "de_generations.csv", de_results["generation_rows"], plane_ids)
        
        de_best_score = float(de_results["best_result"]["score"])
        de_best_depths = de_results["best_result"]["depths"]
        print(f"[PRETEST][DE] Done. Best ray_rmse_mm={de_best_score:.6f}")

    # COMPARISON
    if args.mode == "both":
        compare_path = out_dir / "search_comparison_summary.json"
        comparison = {
            "sweep": {
                "best_trial_name": sweep_best_name,
                "best_ray_rmse_mm": sweep_best_score,
            },
            "de": {
                "best_ray_rmse_mm": de_best_score,
                "best_depths": de_best_depths,
            },
            "winner": "sweep" if sweep_best_score < de_best_score else "de",
            "score_diff": abs(sweep_best_score - de_best_score),
        }
        compare_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"[PRETEST][COMPARE] Winner: {comparison['winner']} (diff={comparison['score_diff']:.6f})")


if __name__ == "__main__":
    main()
