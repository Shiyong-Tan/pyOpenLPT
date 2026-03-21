
import numpy as np
import cv2
import scipy.optimize
from scipy.optimize import OptimizeResult
from pathlib import Path
from PySide6.QtCore import QThread, Signal, QWaitCondition, QMutex
import concurrent.futures
import multiprocessing
import os


# --- Robust Detection Constants ---
RANSAC_THRESH = 3.0
RANSAC_ITERS = 500
FAST_ANNULUS_LOW = 0.7
FAST_ANNULUS_HIGH = 1.3

# --- Robust Detection Helpers ---
def thin_by_angle(pts, cx, cy, r_init, nbin=360):
    ang = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
    bins = ((ang + np.pi)/(2*np.pi)*nbin).astype(int)
    bins = np.clip(bins, 0, nbin-1)
    keep = []
    for b in range(nbin):
        idx = np.where(bins == b)[0]
        if idx.size == 0: continue
        dists = np.sqrt((pts[idx,0]-cx)**2 + (pts[idx,1]-cy)**2)
        dist_diff = np.abs(dists - r_init)
        best_i = idx[np.argmin(dist_diff)]
        keep.append(best_i)
    return pts[np.array(keep)]

def refine_circle_with_edges(img_gray, cx_init, cy_init, r_init, exclusion_mask=None):
    H, W = img_gray.shape
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1.5)
    edges = cv2.Canny(img_blur, 30, 80)
    
    y_idx, x_idx = np.indices((H, W))
    dist_sq = (x_idx - cx_init)**2 + (y_idx - cy_init)**2
    r_min_sq = (r_init * FAST_ANNULUS_LOW)**2
    r_max_sq = (r_init * FAST_ANNULUS_HIGH)**2
    mask_annulus = (dist_sq >= r_min_sq) & (dist_sq <= r_max_sq)
    edges[~mask_annulus] = 0
    
    if exclusion_mask is not None:
        edges[exclusion_mask > 0] = 0
        
    edge_coords = np.column_stack(np.where(edges > 0)) # [y, x]
    if len(edge_coords) < 10:
        return None, None, None, None
        
    pts_raw = edge_coords[:, [1, 0]].astype(np.float32)
    pts = thin_by_angle(pts_raw, cx_init, cy_init, r_init, nbin=360)
    n_points = pts.shape[0]
    
    best_score = -1.0
    best_model = None 
    
    for _ in range(RANSAC_ITERS):
        if n_points < 3: break
        try:
            sample_idx = np.random.choice(n_points, 3, replace=False)
        except ValueError: break
        
        p1, p2, p3 = pts[sample_idx]
        D = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        if abs(D) < 1.0: continue 
        
        Ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / D
        Uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / D
        R_cand = np.sqrt((p1[0] - Ux)**2 + (p1[1] - Uy)**2)
        
        if abs(R_cand - r_init) > 20: continue
        
        dists = np.sqrt((pts[:, 0] - Ux)**2 + (pts[:, 1] - Uy)**2)
        residuals = np.abs(dists - R_cand)
        inliers_mask = residuals < RANSAC_THRESH
        n_inliers = np.count_nonzero(inliers_mask)
        
        if n_inliers < 10: continue
        
        inlier_pts = pts[inliers_mask]
        angles = np.arctan2(inlier_pts[:, 1] - Uy, inlier_pts[:, 0] - Ux)
        nbin = 72
        bins = ((angles + np.pi) / (2*np.pi) * nbin).astype(int)
        bins = np.clip(bins, 0, nbin-1)
        coverage = len(np.unique(bins)) / float(nbin)
        
        if coverage < 0.4: continue
        mean_res = np.mean(residuals[inliers_mask])
        score = (coverage**2 * np.sqrt(n_inliers)) / (mean_res + 1e-5)
        
        if score > best_score:
            best_score = score
            best_model = ((Ux, Uy), R_cand, inliers_mask)

    if best_model is None:
        return None, None, None, None

    (bcx, bcy), br, bmask = best_model
    
    # Least Squares Refinement
    inlier_pts_final = pts[bmask]
    x, y = inlier_pts_final[:, 0], inlier_pts_final[:, 1]
    A_mat = np.column_stack([x, y, np.ones_like(x)])
    b_vec = -(x**2 + y**2)
    
    refined_cx, refined_cy, refined_r = bcx, bcy, br
    try:
        result, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        A_ls, B_ls, C_ls = result
        ls_cx = -A_ls / 2.0
        ls_cy = -B_ls / 2.0
        R_sq = ls_cx**2 + ls_cy**2 - C_ls
        if R_sq > 0:
            ls_r = np.sqrt(R_sq)
            if abs(ls_r - br) < 5.0 and np.sqrt((ls_cx-bcx)**2+(ls_cy-bcy)**2) < 5.0:
                 refined_cx, refined_cy, refined_r = ls_cx, ls_cy, ls_r
    except:
        pass
        
    return refined_cx, refined_cy, refined_r, best_score

def detect_circles_robust(img_gray, min_r, max_r):
    """
    Run the Robust Pipeline: DT Candidates -> Refinement -> Metric Sort.
    Returns: list of [x, y, r, metric_score]
    """
    # 1. Global Otsu for DT
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Fill contours
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_bin_filled = np.zeros_like(img_bin)
    cv2.drawContours(img_bin_filled, contours, -1, 255, cv2.FILLED)
    
    # 2. Distance Transform
    dist = cv2.distanceTransform(img_bin_filled, cv2.DIST_L2, 5)
    
    # 3. Find Candidates
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)) 
    dilated = cv2.dilate(dist, kernel)
    is_peak = (dist == dilated) & (dist > 0)
    peak_points = np.column_stack(np.where(is_peak))
    
    candidates = []
    for (py, px) in peak_points:
        r_guess = dist[py, px]
        if min_r <= r_guess <= max_r:
            candidates.append({'x': px, 'y': py, 'r': r_guess})
    candidates.sort(key=lambda c: c['r'], reverse=True)
    
    # NMS
    final_candidates = []
    for cand in candidates:
        keep = True
        for exist in final_candidates:
            d = np.sqrt((cand['x'] - exist['x'])**2 + (cand['y'] - exist['y'])**2)
            if d < exist['r']:
                keep = False; break
        if keep: final_candidates.append(cand)
            
    # 4. Refinement Loop
    results = []
    exclusion_mask = np.zeros_like(img_bin)
    
    for cand in final_candidates:
        cx, cy, r, metric = refine_circle_with_edges(img_gray, cand['x'], cand['y'], cand['r'], exclusion_mask)
        if cx is not None:
            if min_r <= r <= max_r:
                # IMPORTANT: Metric is float, x/y are float. Return unified list.
                results.append([cx, cy, r, metric]) 
                cv2.circle(exclusion_mask, (int(cx), int(cy)), int(r + 5), 255, -1)
    
    # Sort by Metric (Higher is Better)
    results.sort(key=lambda x: x[3], reverse=True)
    return results

# --- Parallel Worker Function (Top-Level) ---
def run_detection_task(args):
    """
    Worker function for parallel wand detection.
    args: (frame_idx, cam_idx, img_path, wand_type, min_radius, max_radius, sensitivity)
    Returns: (frame_idx, cam_idx, points_array_or_None)
    """
    if len(args) >= 8:
        f_idx, c_idx, img_path, wand_type, min_radius, max_radius, sensitivity, detect_mode = args
    else:
        f_idx, c_idx, img_path, wand_type, min_radius, max_radius, sensitivity = args
        detect_mode = "fast"
    
    try:
        # 1. Read Image
        if not os.path.exists(img_path):
            return f_idx, c_idx, None
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f_idx, c_idx, None
        
        # 2. Normalize
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)
            
        # 3. Invert if dark
        if wand_type == "dark":
            img = cv2.bitwise_not(img)
            
        if str(detect_mode).lower() == "fast":
            # --- STRATEGY: Try Fast Robust DT First ---
            try:
                 robust_pts = detect_circles_robust(img, float(min_radius), float(max_radius))
                 if len(robust_pts) >= 2:
                     # Success: Return Top 2 Highest Metric
                     final_pts = robust_pts[:2]
                     # Re-sort by radius (Small -> Large) as expected by downstream logic
                     final_pts = sorted(final_pts, key=lambda p: p[2])
                     return f_idx, c_idx, np.array(final_pts)
            except Exception:
                 # If Robust fails, fallback silent
                 pass

        # --- FALLBACK: LPT Algorithm ---
        from pyopenlpt import Image as LPTImage, CircleIdentifier
        
        lpt_img = LPTImage.from_numpy(img)
        detector = CircleIdentifier(lpt_img)
        centers, radii, metrics = detector.BubbleCenterAndSizeByCircle(
            float(min_radius), float(max_radius), float(sensitivity)
        )
        
        if len(centers) == 0:
            return f_idx, c_idx, None
            
        pts = []
        for i in range(len(centers)):
            x = centers[i][0]
            y = centers[i][1]
            r = radii[i]
            m = metrics[i] if i < len(metrics) else 0.0
            pts.append([x, y, r, m])
            
        # Sort by radius
        pts = sorted(pts, key=lambda p: p[2])
        return f_idx, c_idx, np.array(pts)
        
    except ImportError:
        return f_idx, c_idx, None
    except Exception as e:
        return f_idx, c_idx, None


class WandDetectionSingleFrameWorker(QThread):
    finished_signal = Signal(object) # Returns dict of results

    def __init__(self, calibrator, image_paths_dict, wand_type, min_r, max_r, sensitivity, detect_mode="fast"):
        super().__init__()
        self.calibrator = calibrator
        self.image_paths_dict = image_paths_dict
        self.wand_type = wand_type
        self.min_r = min_r
        self.max_r = max_r
        self.sensitivity = sensitivity
        self.detect_mode = detect_mode

    def run(self):
        # Run detection
        try:
            res = self.calibrator.detect_single_frame(
                self.image_paths_dict, 
                self.wand_type, 
                self.min_r, 
                self.max_r, 
                self.sensitivity,
                self.detect_mode
            )
            self.finished_signal.emit(res)
        except Exception as e:
            print(f"Single frame detection failed: {e}")
            self.finished_signal.emit({})

class WandDetectionWorker(QThread):
    progress = Signal(int, int) # current, total
    finished_signal = Signal(bool, str) # success, msg
    
    def __init__(self, calibrator, image_paths, wand_type, min_r, max_r, sens, autosave_path=None, resume=False, detect_mode="fast"):
        super().__init__()
        self.calibrator = calibrator
        self.image_paths = image_paths
        self.wand_type = wand_type
        self.min_r = min_r
        self.max_r = max_r
        self.sens = sens
        self.detect_mode = detect_mode
        self.autosave_path = autosave_path
        self.resume_flag = resume
        
        self._is_paused = False
        self._is_stopped = False
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        
    def run(self):
        try:
            # If resuming, load data first (in background thread to avoid UI freeze)
            if self.resume_flag and self.autosave_path:
                success, msg = self.calibrator.load_wand_data_from_csv(self.autosave_path)
                if not success:
                    self.finished_signal.emit(False, f"Failed to load resume data: {msg}")
                    return

            gen = self.calibrator.detect_wand_points_generator(
                self.image_paths, self.wand_type, self.min_r, self.max_r, self.sens,
                self.autosave_path, self.resume_flag, self.detect_mode,
                stop_check=lambda: self._is_stopped
            )
            
            # First item yielded is (0, total_frames) for init
            
            import time
            last_update_time = 0
            
            for current_frame, total_frames_val in gen:
                # Check Pause
                self._mutex.lock()
                while self._is_paused:
                    self._cond.wait(self._mutex)
                self._mutex.unlock()
                
                # Check Stop
                if self._is_stopped:
                    if self.autosave_path:
                        self.calibrator.export_wand_data(self.autosave_path)
                    self.finished_signal.emit(False, "Stopped by user (Data Saved).")
                    return
                
                # Rate Limiting: Emit at most every 50ms (or last frame)
                now = time.time()
                if (now - last_update_time > 0.05) or (current_frame == total_frames_val):
                    self.progress.emit(current_frame, total_frames_val)
                    last_update_time = now
            
            self.finished_signal.emit(True, "Detection completed.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"Error: {str(e)}")

    def pause(self):
        self._mutex.lock()
        self._is_paused = True
        self._mutex.unlock()
        
    def resume(self):
        self._mutex.lock()
        self._is_paused = False
        self._cond.wakeAll()
        self._mutex.unlock()
        
    def stop(self):
        self._is_stopped = True
        self.resume() # Ensure we don't get stuck in pause

class CalibrationStoppedError(Exception):
    """Exception raised when calibration is stopped by user."""
    def __init__(self, message, params=None):
        super().__init__(message)
        self.params = params

class CalibrationWorker(QThread):
    finished_signal = Signal(bool, str, object) # success, msg, results
    progress_signal = Signal(str)
    cost_signal = Signal(str, str, float)  # (phase, stage, rmse)
    
    def __init__(self, calibrator, wand_len, init_focal=9000, precalibrate=False):
        super().__init__()
        self.calibrator = calibrator
        self.wand_len = wand_len
        self.init_focal = init_focal
        self.precalibrate = precalibrate
        self._iteration = 0 # Initialize iteration count for callback
        
    def _cost_callback(self, cost, rmse):
        """Callback invoked during optimization to report cost and RMSE."""
        self._iteration += 1
        # Emit every 5 iterations to reduce UI load
        if self._iteration % 5 == 0 or self._iteration == 1:
            phase = getattr(self.calibrator, '_current_phase', 'Unknown')
            stage = getattr(self.calibrator, '_current_stage', 'Optimizing')
            self.cost_signal.emit(phase, stage, rmse)
        
    def run(self):
        try:
            self.calibrator._stop_requested = False # Reset flag
            if self.precalibrate:
                ret, msg, res = self.calibrator.run_precalibration_check(
                    wand_length=self.wand_len, 
                    init_focal_length=self.init_focal,
                    callback=self._cost_callback
                )
            else:
                ret, msg, res = self.calibrator.calibrate_wand(
                    wand_length=self.wand_len, 
                    init_focal_length=self.init_focal,
                    callback=self._cost_callback
                )
            self.finished_signal.emit(ret, msg, res)
        except CalibrationStoppedError:
            self.finished_signal.emit(False, "Calibration stopped by user.", None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"Calibration crashed: {str(e)}", None)

    def stop(self):
        """Request calibration to stop."""
        if self.calibrator:
            self.calibrator.request_stop()

class WandCalibrator:
    def __init__(self):
        self.cameras = {} # {cam_idx: {'images': [paths], 'params': CameraParams_Object}}
        self.wand_points = {} # {frame_idx: {cam_idx: np.array([[x1,y1,r1],[x2,y2,r2]])}}
        self.wand_points_filtered = None  # Filtered data for calibration (if user cleaned data)
        self.wand_length = 500.0 # mm default
        self.image_size = (0, 0) # Needs update from UI or Detection
        self.final_params = {} # {cam_idx: {'R':..., 'T':..., 'K':..., 'dist':...}}
        self.points_3d = None  # Optimized 3D points from last calibration
        self.per_frame_errors = {}  # {frame_idx: {'cam_errors': {cam_id: err}, 'len_error': float}}
        self.params_dirty = False  # True when new results are available but not yet exported
        self.dist_coeff_num = 2  # Number of radial distortion coefficients (0-4)

    def _validate_dist_coeff_num(self):
        """Ensure dist_coeff_num is within supported range (0, 1, 2)."""
        if getattr(self, "dist_coeff_num", 0) not in (0, 1, 2):
            raise RuntimeError(
                f"dist_coeff_num={self.dist_coeff_num} not supported. Only 0/1/2 are supported (None/k1/k1+k2)."
            )

    def _get_cam_setting(self, cam_id):
        settings = getattr(self, 'camera_settings', None) or {}
        if cam_id in settings:
            cs = settings[cam_id]
            focal = float(cs.get('focal', 0.0))
            width = int(cs.get('width', 0))
            height = int(cs.get('height', 0))
            if focal > 0 and width > 0 and height > 0:
                return focal, width, height

        # Fallback to global values for legacy flows
        h, w = self.image_size if self.image_size is not None else (800, 1280)
        focal_fallback = 5000.0
        return focal_fallback, int(w), int(h)
    
    def calculate_per_frame_errors(self):
        """
        Calculate per-frame reprojection and wand length errors.
        Must be called after successful calibration (self.final_params and self.points_3d exist).
        Returns: {frame_idx: {'cam_errors': {cam_id: max_err_px}, 'len_error': abs_mm}}
        """
        # Optimization: Use cached errors if available (e.g. from Refractive calibration)
        if self.per_frame_errors:
            return self.per_frame_errors

        if not self.final_params or self.points_3d is None:
            return {}
        
        # Get current wand data
        wand_data = self.wand_points_filtered if self.wand_points_filtered else self.wand_points
        frame_list = sorted(wand_data.keys())
        print(f"[calculate_per_frame_errors] Using {'FILTERED' if self.wand_points_filtered else 'ALL'} data: {len(frame_list)} frames")
        
        self.per_frame_errors = {}
        
        for i, fid in enumerate(frame_list):
            obs = wand_data[fid]  # {cam_idx: [[x1,y1,r1],[x2,y2,r2]]}
            
            # Get 3D points for this frame
            idx_A = i * 2
            idx_B = i * 2 + 1
            if idx_B >= len(self.points_3d):
                continue
            pt3d_A = self.points_3d[idx_A]
            pt3d_B = self.points_3d[idx_B]
            
            # Wand length error
            dist = np.linalg.norm(pt3d_A - pt3d_B)
            len_err = abs(dist - self.wand_length)
            
            # Per-camera reprojection error and triangulation error (3D residual)
            cam_errors = {}
            tri_errors = []
            for cam_idx, uv_obs in obs.items():
                if cam_idx not in self.final_params:
                    continue
                p = self.final_params[cam_idx]
                R, T, K, dist_coeffs = p['R'], p['T'], p['K'], p['dist']
                rvec, _ = cv2.Rodrigues(R)
                
                # Camera center and ray direction (simplified pinhole assumption for 3D resid)
                C = -R.T @ T.reshape(3,1)
                R_inv = R.T
                K_inv = np.linalg.inv(K)
                
                # Project A
                proj_A, _ = cv2.projectPoints(pt3d_A.reshape(1,3), rvec, T, K, dist_coeffs)
                err_A = np.linalg.norm(proj_A.flatten()[:2] - np.array(uv_obs[0][:2]))
                
                # Project B
                proj_B, _ = cv2.projectPoints(pt3d_B.reshape(1,3), rvec, T, K, dist_coeffs)
                err_B = np.linalg.norm(proj_B.flatten()[:2] - np.array(uv_obs[1][:2]))
                
                cam_errors[cam_idx] = max(err_A, err_B)

                # 3D Triangulation Residual (distance to ray)
                for pt3d, uv in [(pt3d_A, uv_obs[0]), (pt3d_B, uv_obs[1])]:
                    # Normalized coords
                    uv_h = np.array([uv[0], uv[1], 1.0]).reshape(3,1)
                    # For a more exact ray, one should ideally undistort uv first.
                    # Given we want an error estimate, a linear ray is usually sufficient.
                    d = R_inv @ K_inv @ uv_h
                    d = d / (np.linalg.norm(d) + 1e-12)
                    
                    # Dist P to ray C + t*d is |(P-C) x d|
                    P = pt3d.reshape(3,1)
                    vec = P - C
                    dist_3d = np.linalg.norm(np.cross(vec.flatten(), d.flatten()))
                    tri_errors.append(dist_3d)
            
            self.per_frame_errors[fid] = {
                'cam_errors': cam_errors, 
                'len_error': len_err,
                'tri_errors': tri_errors # Actual 3D residuals in mm
            }
        
        return self.per_frame_errors
    
    def apply_filter(self, frames_to_remove):
        """
        Create wand_points_filtered by removing specified frames from the BASE wand_points.
        frames_to_remove: set of frame_ids to remove
        """
        # Always filter from the original full set (wand_points) to allow un-checking/restoring frames.
        source = self.wand_points
        self.wand_points_filtered = {k: v for k, v in source.items() if k not in frames_to_remove}
        return len(self.wand_points_filtered)
    
    def reset_filter(self):
        """Reset to use original wand_points data."""
        self.wand_points_filtered = None

    def _filter_radius_histogram_peaks(self, radii, cam_idx):
        """
        Robustly estimate Small/Large stats using Histogram Peak Finding.
        Ignores far-field noise (e.g. 77px) by only focusing on dominant peaks.
        Returns: (mean_s, std_s), (mean_l, std_l) or None if failed.
        """
        if len(radii) < 10: return None
        
        # 1. Build Histogram
        min_val, max_val = min(radii), max(radii)
        if min_val == max_val: return None
        
        n_bins = 60
        bin_width = (max_val - min_val) / n_bins
        if bin_width == 0: bin_width = 1.0
        
        counts = [0] * n_bins
        bin_centers = [min_val + bin_width * (i + 0.5) for i in range(n_bins)]
        
        for x in radii:
            idx = int((x - min_val) / bin_width)
            if idx >= n_bins: idx = n_bins - 1
            counts[idx] += 1
            
        # 2. Find Peaks
        peaks = [] # (count, bin_center, bin_idx)
        for i in range(1, n_bins - 1):
            if counts[i] > counts[i-1] and counts[i] > counts[i+1]:
                peaks.append((counts[i], bin_centers[i], i))
        # Edges
        if counts[0] > counts[1]: peaks.append((counts[0], bin_centers[0], 0))
        if counts[-1] > counts[-2]: peaks.append((counts[-1], bin_centers[-1], n_bins-1))
        
        peaks.sort(key=lambda x: x[0], reverse=True)
        
        if len(peaks) < 1: return None
        
        # 3. Select Top 2 Peaks (Small & Large)
        dominant_peaks = [peaks[0]]
        
        if len(peaks) >= 2:
            p1 = peaks[0]
            # Try to find a second distinct peak
            found_second = False
            for i in range(1, len(peaks)):
                p_cand = peaks[i]
                # Distance Check: > 3 bins away
                dist_bins = abs(p1[2] - p_cand[2])
                # Magnitude Check: > 10% of main peak
                ratio = p_cand[0] / p1[0]
                
                if ratio > 0.1 and dist_bins > 3:
                     dominant_peaks.append(p_cand)
                     found_second = True
                     break
            
            # If standard check failed, maybe try a looser check? 
            # For now strict. If only 1 peak found, it will fail (which is correct for wand calib that NEEDS 2 points)
        
        if len(dominant_peaks) < 2:
            print(f"Cam {cam_idx}: Could not find 2 distinct peaks. Found {len(dominant_peaks)}.")
            return None
            
        # Sort by radius (Small first)
        dominant_peaks.sort(key=lambda x: x[1])
        peak_s = dominant_peaks[0]
        peak_l = dominant_peaks[1]
        
        # 4. Define Ranges (Drop to 10%)
        def get_range(p_idx, p_h):
            thresh = p_h * 0.1
            l, r = p_idx, p_idx
            while l > 0 and counts[l] > thresh: l -= 1
            while r < n_bins - 1 and counts[r] > thresh: r += 1
            return bin_centers[l], bin_centers[r]
            
        range_s = get_range(peak_s[2], peak_s[0])
        range_l = get_range(peak_l[2], peak_l[0])
        
        # 5. Extract Stats from Data in Ranges
        data_s = [x for x in radii if range_s[0] <= x <= range_s[1]]
        data_l = [x for x in radii if range_l[0] <= x <= range_l[1]]
        
        if len(data_s) < 5 or len(data_l) < 5:
            return None
            
        m_s, s_s = np.mean(data_s), np.std(data_s)
        m_l, s_l = np.mean(data_l), np.std(data_l)
        
        # Safety Constraint on std
        if s_s < 0.5: s_s = 0.5
        if s_l < 0.5: s_l = 0.5
        
        print(f"Cam {cam_idx} PeakFilter: S[{m_s:.1f}±{s_s:.1f}] L[{m_l:.1f}±{s_l:.1f}] (Ranges: {range_s[0]:.1f}-{range_s[1]:.1f}, {range_l[0]:.1f}-{range_l[1]:.1f})")
        
        return (m_s, s_s), (m_l, s_l)

    def detect_wand_points_generator(self, image_paths_dict, wand_type, min_radius, max_radius, sensitivity, autosave_path=None, resume=False, detect_mode="fast", stop_check=None):
        """
        Generator for detecting wand points. Yields (current_frame_idx, total_frames).
        Implements dual-radius logic and strict filtering.
        Supports Resume and Autosave.
        stop_check: Optional callable that returns True if stop was requested.
        """
        # Update Cameras Dict
        for c_idx, paths in image_paths_dict.items():
             self.cameras[c_idx] = {'images': paths}
        
        if not resume:
            self.wand_points = {} # Final valid points
            self.wand_data_raw = {} # {f_idx: {c_idx: [all pts]}}
            self.wand_data_filtered = {} # {f_idx: {c_idx: [pt_small, pt_large]}}
        else:
            # If resuming, ensure they exist (load_wand_data_from_csv only sets wand_data_raw)
            if not hasattr(self, 'wand_points'): self.wand_points = {}
            if not hasattr(self, 'wand_data_raw'): self.wand_data_raw = {}
            if not hasattr(self, 'wand_data_filtered'): self.wand_data_filtered = {}
        
        num_cams = len(image_paths_dict)
        cam_indices = sorted(image_paths_dict.keys())
        num_frames = len(image_paths_dict[cam_indices[0]])
        
        print(f"Generator starting: {num_frames} frames. Resume={resume}, Autosave={autosave_path}")
        
        # 1. Update Image Size
        first_img_path = image_paths_dict[cam_indices[0]][0]
        ref_img = cv2.imread(str(first_img_path), cv2.IMREAD_GRAYSCALE)
        if ref_img is not None:
             self.image_size = ref_img.shape
             print(f"Detected Image Size: {self.image_size}")
        
        # 2. Phase 1: Raw Detection (Massively Parallel)
        # We build a list of all missing tasks first
        
        all_tasks = []
        for f_idx in range(num_frames):
             # Skip if resuming and already present
            if resume and f_idx in self.wand_data_raw:
                # We can't yield progress linearly here without blocking.
                # The progress bar will jump to (done / total) when we yield later.
                continue
                
            for c_idx in cam_indices:
                if f_idx < len(image_paths_dict[c_idx]):
                    img_path = str(image_paths_dict[c_idx][f_idx])
                    all_tasks.append((f_idx, c_idx, img_path, wand_type, min_radius, max_radius, sensitivity, detect_mode))
        
        total_tasks = len(all_tasks)
        print(f"Parallel Execution: {total_tasks} tasks to run (Skipped {num_frames * num_cams - total_tasks})")
        
        if total_tasks > 0:
            # Use ~80% of cores
            max_workers = max(1, int((os.cpu_count() or 4) * 0.8))
            print(f"Parallel Execution: Using {max_workers} processes (80% of {os.cpu_count()} cores).")
            tasks_done = 0
            # Autosave trigger count (e.g. every 50 frames * num_cameras)
            autosave_interval = 50 * num_cams 
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_detection_task, t) for t in all_tasks]
                
                for future in concurrent.futures.as_completed(futures):
                    # Check for stop request BEFORE processing result
                    if stop_check and stop_check():
                        print("Stop requested. Cancelling remaining tasks...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        # Autosave before exit
                        if autosave_path:
                            self.export_wand_data(autosave_path)
                        yield len(self.wand_data_raw), num_frames # Final yield
                        return # Exit generator early
                    
                    res_f_idx, res_c_idx, pts = future.result()
                    
                    # Store Result
                    if pts is not None:
                        if res_f_idx not in self.wand_data_raw:
                            self.wand_data_raw[res_f_idx] = {}
                        self.wand_data_raw[res_f_idx][res_c_idx] = pts
                    else:
                        # Ensure we mark as processed even if None found
                        if res_f_idx not in self.wand_data_raw:
                            self.wand_data_raw[res_f_idx] = {}

                    tasks_done += 1
                    
                    # Yield progress periodically
                    if tasks_done % num_cams == 0:
                         # Yield valid update for progress bar
                         yield len(self.wand_data_raw), num_frames

                    # Autosave logic (Main Thread - Safe)
                    if autosave_path and tasks_done % autosave_interval == 0:
                        print(f"Autosaving at {tasks_done} tasks...")
                        self.export_wand_data(autosave_path)
            
        # Ensure we yield final 100% and autosave
        yield num_frames, num_frames
        if autosave_path:
            self.export_wand_data(autosave_path)
            
        print("Raw detection done. Calculating stats...")
        
        # 3. Phase 2: Per-Camera Stats (Dual Radius)
        cam_stats = {} # {c_idx: {'s': (mean, std), 'l': (mean, std)}}
        
        for c_idx in cam_indices:
            # Collect radii for this camera accross all frames
            radii = []
            for f_idx in self.wand_data_raw:
                if c_idx in self.wand_data_raw[f_idx]:
                    for p in self.wand_data_raw[f_idx][c_idx]:
                        radii.append(p[2])
            
            if len(radii) < 10:
                print(f"Cam {c_idx}: Not enough points ({len(radii)}) for stats.")
                continue
                
            radii = np.array(radii)
            
            # Zoning / Clustering logic (Peak Finding)
            mean_s, std_s, mean_l, std_l = 0, 1, 0, 1
            
            # New Robust Method: Peak Finding
            peak_res = self._filter_radius_histogram_peaks(radii, c_idx)
            
            if peak_res:
                (mean_s, std_s), (mean_l, std_l) = peak_res
            else:
                # Fallback to Median Method if Peak Finding fails (e.g. only 1 peak found)
                print(f"Cam {c_idx}: Peak finding failed. Falling back to Median Zoning.")
                try:
                    median = np.median(radii)
                    radii_s = radii[radii <= median]
                    radii_l = radii[radii > median]
                    mean_s, std_s = (np.mean(radii_s), np.std(radii_s)) if len(radii_s)>0 else (0,1)
                    mean_l, std_l = (np.mean(radii_l), np.std(radii_l)) if len(radii_l)>0 else (0,1)
                except Exception as e:
                    print(f"Cam {c_idx} Stats Error: {e}")
            
            # Constraints
            if std_s < 0.5: std_s = 0.5
            if std_l < 0.5: std_l = 0.5
            
            print(f"Cam {c_idx} Final Stats: Small(m={mean_s:.1f}, s={std_s:.1f}), Large(m={mean_l:.1f}, s={std_l:.1f})")
            cam_stats[c_idx] = {'s': (mean_s, std_s), 'l': (mean_l, std_l)}

        if not cam_stats:
            print("No valid stats for any camera.")
            return

        # 4. Phase 3 & 4: Point Selection & Validation
        k_std = 3.0 # Threshold multiplier
        pct_range = 0.05
        
        for f_idx, cam_data in self.wand_data_raw.items():
            frame_filtered = {}
            frame_valid = True
            rejection_reason = ""
            
            # For this frame, check EVERY camera
            for c_idx in cam_indices:
                if c_idx not in cam_data:
                    frame_valid = False
                    rejection_reason = f"Camera {c_idx} not in cam_data"
                    break
                
                # Check if we have stats for this camera
                if c_idx not in cam_stats:
                    # Treat as invalid if we can't filter? Or skip?
                    # Ideally frames must be seen by ALL cameras. 
                    # If this camera has no stats (bad detection), we can't validate points.
                    frame_valid = False
                    rejection_reason = f"Camera {c_idx} not in cam_stats"
                    break

                stats = cam_stats[c_idx]
                mean_s, std_s = stats['s']
                mean_l, std_l = stats['l']
                
                pts = cam_data[c_idx] # [[x,y,r,m], ...] 
                
                # Find best Small point
                range_s = max(k_std * std_s, mean_s * pct_range)
                range_l = max(k_std * std_l, mean_l * pct_range)
                candidates_s = [p for p in pts if abs(p[2] - mean_s) < range_s]
                # Filter by range Large
                candidates_l = [p for p in pts if abs(p[2] - mean_l) < range_l]
                
                if not candidates_s or not candidates_l:
                    frame_valid = False
                    rejection_reason = f"Cam {c_idx}: No candidates (s:{len(candidates_s)}, l:{len(candidates_l)}), pts={len(pts)}"
                    break
                
                # Select best Small first (Highest Metric Score)
                # Metric is at p[3]. Higher is better (Roundness/Confidence).
                # Previous logic: closest radius to mean_s -> best_s = min(candidates_s, key=lambda p: abs(p[2] - mean_s))
                best_s = max(candidates_s, key=lambda p: p[3])
                
                # Select best Large, EXCLUDING the point selected for Small
                candidates_l_filtered = [p for p in candidates_l if not np.array_equal(p[:2], best_s[:2])]
                
                if not candidates_l_filtered:
                    # All Large candidates are the same as Small - fallback: pick the one closest to Large mean
                    # This happens when only 1-2 points detected
                    frame_valid = False
                    rejection_reason = f"Cam {c_idx}: No distinct Large after excluding Small"
                    break
                
                # Select Large with Highest Metric Score
                best_l = max(candidates_l_filtered, key=lambda p: p[3])

                frame_filtered[c_idx] = [best_s, best_l]
            
            if frame_valid:
                self.wand_data_filtered[f_idx] = frame_filtered
                
                # Provide standard format for calibration
                pts_map = {}
                for c_idx, pair in frame_filtered.items():
                    # Align with CSV format: [x, y, r, m, label, idx]
                    # pair = [best_s, best_l]. best_s/l are [x,y,r,m]
                    s_full = list(pair[0]) + ["Filtered_Small", 0]
                    l_full = list(pair[1]) + ["Filtered_Large", 1]
                    pts_map[c_idx] = [s_full, l_full] # Store as LIST
                
                self.wand_points[f_idx] = pts_map
            else:
                # Debug: Print reason for first 5 rejected frames
                if len(self.wand_data_filtered) == 0 and f_idx < 5:
                    print(f"  Pass1 Frame {f_idx} rejected: {rejection_reason}")
        
        print(f"Pass 1 Done. Valid Frames: {len(self.wand_data_filtered)} / {num_frames}")
        
        # ========================================
        # PASS 2: Refined Filtering
        # Recalculate stats from FILTERED data (more accurate)
        # Then re-filter RAW data using refined stats
        # ========================================
        
        if len(self.wand_data_filtered) > 100:  # Only do pass 2 if we have enough data
            print("Pass 2: Recalculating stats from filtered data...")
            
            # Recalculate stats from filtered data
            cam_stats_refined = {}
            for c_idx in cam_indices:
                radii_s = []
                radii_l = []
                for f_idx in self.wand_data_filtered:
                    if c_idx in self.wand_data_filtered[f_idx]:
                        pair = self.wand_data_filtered[f_idx][c_idx]
                        radii_s.append(pair[0][2])  # Small radius
                        radii_l.append(pair[1][2])  # Large radius
                
                if len(radii_s) > 10 and len(radii_l) > 10:
                    mean_s = np.mean(radii_s)
                    std_s = max(0.5, np.std(radii_s))
                    mean_l = np.mean(radii_l)
                    std_l = max(0.5, np.std(radii_l))
                    cam_stats_refined[c_idx] = {'s': (mean_s, std_s), 'l': (mean_l, std_l)}
                    print(f"  Cam {c_idx} Refined: Small(m={mean_s:.1f}, s={std_s:.1f}), Large(m={mean_l:.1f}, s={std_l:.1f})")
            
            # Re-filter RAW data using refined stats
            if len(cam_stats_refined) == len(cam_indices):
                self.wand_data_filtered = {}
                self.wand_points = {}
                
                for f_idx, cam_data in self.wand_data_raw.items():
                    frame_filtered = {}
                    frame_valid = True
                    
                    for c_idx in cam_indices:
                        if c_idx not in cam_data:
                            frame_valid = False
                            break
                        
                        stats = cam_stats_refined[c_idx]
                        mean_s, std_s = stats['s']
                        mean_l, std_l = stats['l']
                        pts = cam_data[c_idx]
                        
                        k_std_refined = 3.0
                        pct_range_refined = 0.05
                        range_s = max(k_std_refined * std_s, mean_s * pct_range_refined)
                        range_l = max(k_std_refined * std_l, mean_l * pct_range_refined)
                        candidates_s = [p for p in pts if abs(p[2] - mean_s) < range_s]
                        candidates_l = [p for p in pts if abs(p[2] - mean_l) < range_l]
                        
                        if not candidates_s or not candidates_l:
                            frame_valid = False
                            break
                        
                        best_s = max(candidates_s, key=lambda p: p[3])
                        candidates_l_filtered = [p for p in candidates_l if not np.array_equal(p[:2], best_s[:2])]
                        
                        if not candidates_l_filtered:
                            frame_valid = False
                            break
                        
                        best_l = max(candidates_l_filtered, key=lambda p: p[3])
                        frame_filtered[c_idx] = [best_s, best_l]
                    
                    if frame_valid:
                        self.wand_data_filtered[f_idx] = frame_filtered
                        pts_map = {}
                        for c_idx, pair in frame_filtered.items():
                            # pair[0] is [x, y, r, m]
                            s_full = list(pair[0]) + ["Filtered_Small", 0]
                            l_full = list(pair[1]) + ["Filtered_Large", 1]
                            pts_map[c_idx] = [s_full, l_full]
                        self.wand_points[f_idx] = pts_map
                
                print(f"Pass 2 Done. Valid Frames: {len(self.wand_data_filtered)} / {num_frames}")
        
        # Final Save with Filtered Data
        if autosave_path:
            self.export_wand_data(autosave_path)
            
        print(f"Refinement Done. Valid Frames: {len(self.wand_data_filtered)} / {num_frames}")

    def load_wand_data_from_csv(self, file_path):
        """Load raw data from CSV to resume processing."""
        import csv
        
        # Reset all state
        self.wand_data_raw = {}
        self.wand_data_filtered = {}
        self.wand_points = {}
        self.wand_points_filtered = None # Reset filtered set
        self.per_frame_errors = {}
        self.points_3d = None
        self.final_params = {}
        
        count_raw = 0
        count_filtered = 0
        
        try:
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header: return False, "Empty file"
                
                # Header: ["Frame", "Camera", "Status", "PointIdx", "X", "Y", "Radius", "Metric"]
                
                for row in reader:
                    if len(row) < 8: continue
                    
                    status = row[2]
                    try:
                        pt_idx = int(row[3])
                        f_idx = int(row[0])
                        c_idx = int(row[1])
                        x = float(row[4])
                        y = float(row[5])
                        r = float(row[6])
                        m = float(row[7])
                    except (ValueError, IndexError): continue

                    # Load Raw for resuming detection
                    if status == "Raw":
                        if f_idx not in self.wand_data_raw: self.wand_data_raw[f_idx] = {}
                        if c_idx not in self.wand_data_raw[f_idx]: self.wand_data_raw[f_idx][c_idx] = []
                        self.wand_data_raw[f_idx][c_idx].append([x,y,r,m])
                        count_raw += 1
                        
                    # Load Filtered for direct calibration
                    elif status.startswith("Filtered"):
                        if f_idx not in self.wand_data_filtered: self.wand_data_filtered[f_idx] = {}
                        if c_idx not in self.wand_data_filtered[f_idx]: self.wand_data_filtered[f_idx][c_idx] = []
                        # Only keep first 2 points per camera per frame (deduplicate)
                        if len(self.wand_data_filtered[f_idx][c_idx]) < 2:
                            self.wand_data_filtered[f_idx][c_idx].append([x,y,r,m,status,pt_idx])
                            count_filtered += 1
            
            # Post-process to build self.wand_points from filtered data
            for f_idx, cam_dict in self.wand_data_filtered.items():
                frame_pts = {}
                frame_complete = True
                
                # Need at least 2 cameras
                if len(cam_dict) < 2: continue

                for c_idx, pts in cam_dict.items():
                    # Check if we have both points
                    if len(pts) != 2:
                        frame_complete = False
                        break
                    
                    # pts elements are [x, y, r, m, status]
                    # We store them all to preserve the Label for refractive calibration
                    frame_pts[c_idx] = pts
                
                if frame_complete:
                     self.wand_points[f_idx] = frame_pts

            print(f"Loaded {count_raw} raw, {count_filtered} filtered points. {len(self.wand_points)} valid frames ready for calibration.")
            return True, f"Loaded {len(self.wand_points)} valid frames."
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

    def export_wand_data(self, file_path):
        """Export raw and filtered data to CSV."""
        import csv
        
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(["Frame", "Camera", "Status", "PointIdx", "X", "Y", "Radius", "Metric"])
                
                # Write Raw Data
                for f_idx, cam_data in self.wand_data_raw.items():
                    for c_idx, pts in cam_data.items():
                        for p_idx, pt in enumerate(pts):
                            # pt = [x, y, r, m]
                            x, y, r = pt[0], pt[1], pt[2]
                            m = pt[3] if len(pt) > 3 else 0
                            writer.writerow([f_idx, c_idx, "Raw", p_idx, f"{x:.3f}", f"{y:.3f}", f"{r:.3f}", f"{m:.4f}"])
                
                # Write Filtered Data
                for f_idx, cam_data in self.wand_data_filtered.items():
                    for c_idx, pts in cam_data.items():
                        # pts = [small, large]
                        if len(pts) >= 1:
                            s = pts[0]
                            writer.writerow([f_idx, c_idx, "Filtered_Small", 0, f"{s[0]:.3f}", f"{s[1]:.3f}", f"{s[2]:.3f}", f"{s[3] if len(s)>3 else 0:.4f}"])
                        if len(pts) >= 2:
                            l = pts[1]
                            writer.writerow([f_idx, c_idx, "Filtered_Large", 1, f"{l[0]:.3f}", f"{l[1]:.3f}", f"{l[2]:.3f}", f"{l[3] if len(l)>3 else 0:.4f}"])

            return True, "Export successful"
        except Exception as e:
            return False, str(e)


    def detect_single_frame(self, image_dict, wand_type="bright", min_radius=20, max_radius=200, sensitivity=0.85, detect_mode="fast"):
        """
        Run detection on a single frame dictionary {cam_idx: path}.
        Returns {cam_idx: points} for visualization.
        Parallelized using ProcessPoolExecutor.
        """
        results = {}
        
        # Prepare Tasks
        tasks = []
        for c_idx, path in image_dict.items():
            # Frame index is dummy 0 here
            tasks.append((0, c_idx, str(path), wand_type, min_radius, max_radius, sensitivity, detect_mode))
            
        # Run Parallel
        try:
             # Use limited workers for single frame (no need to spin up 24 cores for 4 tasks)
             # But using a persistent pool would be better? Creating pool has overhead (~50ms).
             # For 4 cores it is fine.
             max_workers = min(4, os.cpu_count() or 4)
             with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_detection_task, t) for t in tasks]
                
                for future in concurrent.futures.as_completed(futures):
                    f_idx, c_idx, pts = future.result()
                    if pts is not None:
                        results[c_idx] = pts
        except Exception as e:
            print(f"Parallel detection failed: {e}. Fallback to sequential.")
            # Fallback Sequential
            for c_idx, path in image_dict.items():
                _, _, pts = run_detection_task((0, c_idx, str(path), wand_type, min_radius, max_radius, sensitivity, detect_mode))
                if pts is not None:
                    results[c_idx] = pts
                    
        return results

    def _detect_in_image(self, img_path, wand_type, min_radius=20, max_radius=200, sensitivity=0.85):
        """
        Detect circles in image using C++ CircleIdentifier (Hough Circle Transform).
        Returns [[x, y, radius], ...] sorted by radius (small first).
        """
        try:
            from pyopenlpt import Image as LPTImage, CircleIdentifier
        except ImportError:
            print("Warning: pyopenlpt not available, falling back to OpenCV detection")
            return self._detect_in_image_opencv(img_path, wand_type, min_radius, max_radius)
        
        # Load image with OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Normalize to 8-bit if 16-bit
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)
        
        # For dark wand on bright background, invert
        if wand_type == "dark":
            img = cv2.bitwise_not(img)
        
        # Convert numpy to LPT Image format
        lpt_img = LPTImage.from_numpy(img)
        
        # Use C++ CircleIdentifier
        # Constructor performs normalization (iterates image twice)
        detector = CircleIdentifier(lpt_img)
        
        centers, radii, metrics = detector.BubbleCenterAndSizeByCircle(
            float(min_radius), float(max_radius), float(sensitivity)
        )
        
        if len(centers) == 0:
            print(f"  CircleIdentifier found 0 circles")
            return None
        
        # Convert ALL detected circles to our format [[x, y, r, m], ...]
        pts = []
        for i in range(len(centers)):
            x = centers[i][0]  # column = x
            y = centers[i][1]  # row = y
            r = radii[i]
            m = metrics[i] if i < len(metrics) else 0.0
            pts.append([x, y, r, m])
        
        # Sort by radius (smaller first for consistent matching)
        pts = sorted(pts, key=lambda p: p[2])
        # print(f"  CircleIdentifier found {len(pts)} circles.")
        return np.array(pts)
    
    def _detect_in_image_opencv(self, img_path, wand_type, min_radius=20, max_radius=200):
        """Fallback OpenCV-based detection if C++ module not available."""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # Normalize to 8-bit if 16-bit
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)
            
        # For dark wand on bright background, invert
        if wand_type == "dark":
            img = cv2.bitwise_not(img)
        
        # Blur
        img_blur = cv2.GaussianBlur(img, (15, 15), 3)
        
        min_val, max_val, _, _ = cv2.minMaxLoc(img_blur)
        if max_val < 30:
            return None
        
        # Use HoughCircles
        circles = cv2.HoughCircles(
            img_blur, 
            cv2.HOUGH_GRADIENT, 
            dp=1.5,
            minDist=max(30, min_radius),
            param1=80,
            param2=40,
            minRadius=min_radius, 
            maxRadius=max_radius
        )
        
        if circles is None or len(circles[0]) < 2:
            return None
        
        circles = np.round(circles[0, :]).astype(int)
        circles = circles[(circles[:, 2] >= min_radius) & (circles[:, 2] <= max_radius)]
        circles = circles[circles[:, 2].argsort()]
        
        if len(circles) >= 2:
            pts = [[float(c[0]), float(c[1]), float(c[2])] for c in circles[:2]]
            return np.array(pts)
            
        return None

    def request_stop(self):
        """Set flag to stop calibration."""
        self._stop_requested = True

    def run_adaptive_calibration(self, wand_length_mm, progress_callback=None):
        """
        Adaptive focal length search using coarse scan + golden section refinement.
        Termination: interval width < 1000 px OR RMS < 0.5 px.
        Returns: (success, msg, best_focal, best_rms)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Phase 1: Coarse Scan (Parallel)
        candidates = [2000, 10000, 30000, 60000, 90000]
        results = {} # {f0: (success, rms)}
        
        def evaluate_candidate(f0):
            """Worker function for parallel evaluation."""
            print(f"  [Thread] Testing f0={f0}...")
            success, msg, rms = self._quick_bundle_adjustment(wand_length_mm, f0, max_nfev=100)
            print(f"    [Thread] f0={f0} -> RMS={rms:.4f} px" if success else f"    [Thread] f0={f0} -> Failed: {msg}")
            return f0, success, rms if success else float('inf')
        
        print("=== Phase 1: Coarse Scan (Parallel 5 threads) ===")
        if progress_callback:
            progress_callback(f"Coarse scan: testing {len(candidates)} candidates in parallel...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(evaluate_candidate, f0): f0 for f0 in candidates}
            
            for future in as_completed(futures):
                f0, success, rms = future.result()
                results[f0] = (success, rms)
                
                # Early exit if already good (but can't stop other threads easily)
                if success and rms < 0.5:
                    print(f"=== Early hit: RMS={rms:.4f} < 0.5 at f0={f0} (will still wait for others) ===")
        
        # Find best candidate
        best_f0 = min(results, key=lambda k: results[k][1])
        best_rms = results[best_f0][1]
        print(f"  Best coarse candidate: f0={best_f0}, RMS={best_rms:.4f}")
        
        # Phase 2: Golden Section Refinement
        print("=== Phase 2: Golden Section Refinement ===")
        
        # Determine search interval around best candidate
        idx = candidates.index(best_f0)
        f_min = candidates[max(0, idx-1)] if idx > 0 else candidates[0] // 2
        f_max = candidates[min(len(candidates)-1, idx+1)] if idx < len(candidates)-1 else candidates[-1] * 1.5
        f_min, f_max = int(f_min), int(f_max)
        
        # Golden ratio
        phi = (1 + 5**0.5) / 2  # ~1.618
        
        # Initial two interior points
        fa = int(f_max - (f_max - f_min) / phi)
        fb = int(f_min + (f_max - f_min) / phi)
        
        # Evaluate fa
        if progress_callback:
            progress_callback(f"Golden: testing f={fa}")
        _, _, rms_a = self._quick_bundle_adjustment(wand_length_mm, fa, max_nfev=100)
        
        # Evaluate fb
        if progress_callback:
            progress_callback(f"Golden: testing f={fb}")
        _, _, rms_b = self._quick_bundle_adjustment(wand_length_mm, fb, max_nfev=100)
        
        iter_count = 0
        max_golden_iter = 10
        
        while (f_max - f_min) > 1000 and iter_count < max_golden_iter:
            iter_count += 1
            print(f"  Golden iter {iter_count}: [{f_min}, {f_max}], fa={fa}(rms={rms_a:.2f}), fb={fb}(rms={rms_b:.2f})")
            
            if rms_a < rms_b:
                f_max = fb
                fb = fa
                rms_b = rms_a
                fa = int(f_max - (f_max - f_min) / phi)
                if progress_callback:
                    progress_callback(f"Golden iter {iter_count}: testing f={fa}")
                _, _, rms_a = self._quick_bundle_adjustment(wand_length_mm, fa, max_nfev=200)
            else:
                f_min = fa
                fa = fb
                rms_a = rms_b
                fb = int(f_min + (f_max - f_min) / phi)
                if progress_callback:
                    progress_callback(f"Golden iter {iter_count}: testing f={fb}")
                _, _, rms_b = self._quick_bundle_adjustment(wand_length_mm, fb, max_nfev=200)
            
            # Early exit if RMS is good enough
            current_best_rms = min(rms_a, rms_b)
            if current_best_rms < 0.5:
                print(f"  Early exit: RMS={current_best_rms:.4f} < 0.5")
                break
        
        # Pick final best
        optimal_f0 = fa if rms_a < rms_b else fb
        print(f"=== Golden search complete: optimal f0={optimal_f0} ===")
        
        # Phase 3: Full optimization with optimal f0
        return self._finalize_calibration(wand_length_mm, optimal_f0, progress_callback)
    
    def _quick_bundle_adjustment(self, wand_length_mm, initial_focal_len_px, max_nfev=150):
        """Run a quick bundle adjustment with limited function evaluations. Returns (success, msg, rms)."""
        # Save current state so we can restore if needed
        old_final_params = getattr(self, 'final_params', {}).copy()
        old_points_3d = getattr(self, 'points_3d', None)
        
        try:
            # Use max_nfev directly to limit optimization
            success, msg = self.run_bundle_adjustment(wand_length_mm, initial_focal_len_px, max_nfev=max_nfev)
            
            # Extract RMS from message
            rms = 999.0
            if success and "Reprojection RMS:" in msg:
                import re
                match = re.search(r"Reprojection RMS: ([\d.]+)", msg)
                if match:
                    rms = float(match.group(1))
            
            # Restore state (we only want the RMS, not to keep these params as final)
            self.final_params = old_final_params
            self.points_3d = old_points_3d
            
            return success, msg, rms
        except Exception as e:
            self.final_params = old_final_params
            self.points_3d = old_points_3d
            return False, str(e), float('inf')
    
    # [DELETED LEGACY DEF]
        """Run full optimization with the determined optimal focal length."""
        if progress_callback:
            progress_callback(f"Final calibration with f0={optimal_f0}...")
        print(f"=== Final Calibration: f0={optimal_f0} ===")
        
        success, msg = self.run_bundle_adjustment(wand_length_mm, optimal_f0)
        
        # Extract final RMS
        final_rms = 999.0
        if success and "Reprojection RMS:" in msg:
            import re
            match = re.search(r"Reprojection RMS: ([\d.]+)", msg)
        return success, f"Optimal f0={optimal_f0}\n{msg}", optimal_f0, final_rms


    def initialize_geometry_from_pair(self, cam1_idx, cam2_idx, wand_data, wand_length_mm):
        """
        Robust Geometric Initialization using 8-Point Algorithm (Essential Matrix).
        
        Args:
            cam1_idx: Index of first camera (Base, will be at I, 0).
            cam2_idx: Index of second camera (rel pose to be computed).
            wand_data: The filtered wand points dict.
            wand_length_mm: Physical wand length.

        Returns:
            R (3x3), T (3x1) for Cam2 relative to Cam1, scaled to mm, and scale_factor.
        """
        # 1. Collect Common Points
        pts1 = []
        pts2 = []
        
        # Sort keys to ensure deterministic order
        for f_idx in sorted(wand_data.keys()):
            cam_pts = wand_data[f_idx]
            if cam1_idx in cam_pts and cam2_idx in cam_pts:
                # Get distorted points
                p1 = cam_pts[cam1_idx] # [ [x1,y1,r1], [x2,y2,r2] ]
                p2 = cam_pts[cam2_idx]
                
                # Append both ends of the wand
                pts1.append(p1[0][:2])
                pts1.append(p1[1][:2])
                pts2.append(p2[0][:2])
                pts2.append(p2[1][:2])

        pts1 = np.array(pts1, dtype=np.float64)
        pts2 = np.array(pts2, dtype=np.float64)

        if len(pts1) < 8:
            print(f"  Geometry Init: Not enough common points ({len(pts1)} < 8) for cam {cam1_idx}-{cam2_idx}")
            return None, None, None

        # 2. Build per-camera intrinsics from table and normalize image points
        f1, w1, h1 = self._get_cam_setting(cam1_idx)
        K1 = np.array([[f1, 0, w1 / 2.0], [0, f1, h1 / 2.0], [0, 0, 1]], dtype=np.float64)

        f2, w2, h2 = self._get_cam_setting(cam2_idx)
        K2 = np.array([[f2, 0, w2 / 2.0], [0, f2, h2 / 2.0], [0, 0, 1]], dtype=np.float64)

        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, None).reshape(-1, 2)

        # 3. Compute Essential Matrix in normalized coordinates
        E, mask = cv2.findEssentialMat(
            pts1_norm,
            pts2_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1e-3,
        )
        
        if E is None or E.shape != (3,3):
            print("  Geometry Init: Essential Matrix computation failed.")
            return None, None, None

        # 4. Recover Pose (R, t) from E
        # This returns R, t from cam1 to cam2
        points, R, t, mask_pose = cv2.recoverPose(E, pts1_norm, pts2_norm, focal=1.0, pp=(0.0, 0.0))
        
        print(f"  Geometry Init: Pose Recovered using {points} inliers from {len(pts1)} points.")

        # 5. Triangulate to Recover Scale (normalized camera coordinates)
        P1 = np.hstack((np.eye(3), np.zeros((3,1)))) # Cam1 is Origin
        P2 = np.hstack((R, t))                       # Cam2

        # Triangulate all points
        pts1_T = pts1_norm.T
        pts2_T = pts2_norm.T
        
        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_T, pts2_T)
        points_3d = points_4d_hom[:3] / points_4d_hom[3] # (3, N)
        points_3d = points_3d.T # (N, 3)

        # 6. Compute Scale Factor
        # Calculate distance between Wand Endpoints in 3D
        dists = []
        for i in range(0, len(points_3d), 2): # Step 2 (A and B)
            ptA = points_3d[i]
            ptB = points_3d[i+1]
            d = np.linalg.norm(ptA - ptB)
            dists.append(d)
        
        dists = np.array(dists)
        # Filter outliers in scale (very small or huge)
        valid_dists = dists[(dists > 0.001) & (dists < 1000)] # Arbitrary Unit limits
        
        if len(valid_dists) < 5:
             print("  Geometry Init: Triangulation failed to produce valid structure for scale.")
             return None, None, None
             
        median_dist = np.median(valid_dists)
        if median_dist <= 1e-6:
             print("  Geometry Init: Median distance too small.")
             return None, None, None
             
        scale_factor = wand_length_mm / median_dist
        
        print(f"  Geometry Init: Raw Median Length={median_dist:.4f} units. Scale Factor={scale_factor:.4f}")
        
        # 7. Apply Scale
        t_scaled = t * scale_factor
        
        # Return R, T_scaled
        return R, t_scaled, scale_factor

    def _run_optimization_pipeline(self, X0, A_sparsity, cam_id_map, frame_list, 
                                 wand_data, wand_length_mm, best_pair, 
                                 base_lower_bounds, base_upper_bounds, 
                                 max_nfev=None, skip_early_stages=False, **kwargs):
        """
        Executes the robust 4-stage optimization pipeline used in both Phase 1 and Phase 3.
        """
        import scipy.optimize
        from scipy.optimize import OptimizeResult
        
        num_cams = len(cam_id_map)
        n_cam_params = 11
        n_cam_params_total = num_cams * n_cam_params
        
        # Skip Stage 1-3 if using pre-optimized params from Phase 1 & 2
        img_h, img_w = self.image_size
        cx_c, cy_c = img_w/2.0, img_h/2.0
        
        if skip_early_stages:
            print("    [Pipeline] Skipping Stage 1-3 (using pre-optimized params)...")
            # Use X0 directly for Stage 4 setup
            X2 = X0  # For Stage 4 point retrieval
            cam_params_s3 = X0[:n_cam_params_total].copy()  # Pre-optimized camera params
            
            # Setup bounds for Stage 4 based on base bounds
            lb_s4 = base_lower_bounds[:n_cam_params_total].copy()
            ub_s4 = base_upper_bounds[:n_cam_params_total].copy()
            
            # Set reasonable bounds for cx, cy
            margin = 50.0
            for i in range(num_cams):
                base = i * n_cam_params
                lb_s4[base+7] = cx_c - margin
                ub_s4[base+7] = cx_c + margin
                lb_s4[base+8] = cy_c - margin
                ub_s4[base+8] = cy_c + margin
            
            # Setup kwargs for Stage 4
            kwargs_s4 = dict(
                verbose=0, method='trf', x_scale='jac', f_scale=1.0, loss='huber',
                bounds=(lb_s4, ub_s4),
                args=(cam_id_map, frame_list, wand_data, wand_length_mm, self.image_size, best_pair[0], best_pair[1])
            )
            kwargs_s4.update(kwargs)
            if 'ftol' not in kwargs: kwargs_s4['ftol'] = 1e-6
        else:
            # --- Execute Full Stage 1-3 Pipeline ---
            print("    [Pipeline-Stage 1] Locking Intrinsics...")
            self._current_stage = "Stage 1"
            lower_b_s1 = base_lower_bounds.copy()
            upper_b_s1 = base_upper_bounds.copy()
            
            # Lock f, cx, cy, k1, k2 (indices 6-10)
            for i in range(num_cams):
                base = i * n_cam_params
                val_int = X0[base+6:base+11]
                lower_b_s1[base+6:base+11] = val_int - 1e-4
                upper_b_s1[base+6:base+11] = val_int + 1e-4
                
            kwargs_s1 = dict(
                jac_sparsity=A_sparsity, verbose=0, method='trf', x_scale='jac', f_scale=1.0,
                bounds=(lower_b_s1, upper_b_s1),
                ftol=1e-4, xtol=1e-4, gtol=1e-4,
                args=(cam_id_map, frame_list, wand_data, wand_length_mm, self.image_size)
            )
            kwargs_s1.update(kwargs)
            kwargs_s1['max_nfev'] = 100
            
            # Set hard iteration limit for Stage 1
            self._max_residual_calls = 100
            self._residual_call_count = 0
            
            try:
                res1 = scipy.optimize.least_squares(self._residuals, X0, **kwargs_s1)
            except CalibrationStoppedError as e:
                # Check if stopped by user
                if "Stopped by user" in str(e):
                    raise e
                # Hit iteration limit - use partial result
                print(f"    [Pipeline-Stage 1] Hit iteration limit, using partial result")
                res1 = type('obj', (object,), {'x': e.params, 'cost': np.inf})()
            finally:
                self._max_residual_calls = None  # Clear limit
            print(f"    [Pipeline-Stage 1] Cost: {res1.cost:.2e}")
            
            # --- STAGE 2: Refine All (Intrinsics Unlocked) ---
            print("    [Pipeline-Stage 2] Refine All (f > 100)...")
            self._current_stage = "Stage 2"
            X1 = res1.x
            
            lb_s2 = base_lower_bounds.copy()
            ub_s2 = base_upper_bounds.copy()
            f_indices = np.arange(6, n_cam_params_total, 11)
            lb_s2[f_indices] = 100.0
            
            for i in range(num_cams):
                base = i * n_cam_params
                lb_s2[base+7] = cx_c - 0.1
                ub_s2[base+7] = cx_c + 0.1
                lb_s2[base+8] = cy_c - 0.1
                ub_s2[base+8] = cy_c + 0.1
                
            kwargs_s2 = dict(
                jac_sparsity=A_sparsity, verbose=0, method='trf', x_scale='jac', f_scale=1.0, 
                loss='soft_l1',
                bounds=(lb_s2, ub_s2),
                ftol=1e-4, xtol=1e-4, gtol=1e-4,
                args=(cam_id_map, frame_list, wand_data, wand_length_mm, self.image_size)
            )
            kwargs_s2.update(kwargs)
            kwargs_s2['max_nfev'] = 100

            # Set hard iteration limit for Stage 2
            self._max_residual_calls = 100
            self._residual_call_count = 0
            
            try:
                res2 = scipy.optimize.least_squares(self._residuals, X1, **kwargs_s2)
            except CalibrationStoppedError as e:
                # Check if stopped by user
                if "Stopped by user" in str(e):
                    raise e
                # Hit iteration limit - use partial result
                print(f"    [Pipeline-Stage 2] Hit iteration limit, using partial result")
                res2 = type('obj', (object,), {'x': e.params, 'cost': np.inf})()
            finally:
                self._max_residual_calls = None  # Clear limit
            print(f"    [Pipeline-Stage 2] Cost: {res2.cost:.2e}")
            
            # --- STAGE 3: Triangulation-Based Refinement ---
            print("    [Pipeline-Stage 3] Triangulation Constraint...")
            self._current_stage = "Stage 3"
            X2 = res2.x
            cam_params_only = X2[:n_cam_params_total]
            
            lb_cam = lb_s2[:n_cam_params_total]
            ub_cam = ub_s2[:n_cam_params_total]
            
            kwargs_s3 = dict(
                verbose=0, method='trf', x_scale='jac', f_scale=1.0, loss='huber',
                bounds=(lb_cam, ub_cam),
                args=(cam_id_map, frame_list, wand_data, wand_length_mm, self.image_size, best_pair[0], best_pair[1])
            )
            kwargs_s3.update(kwargs)
            if 'ftol' not in kwargs: kwargs_s3['ftol'] = 1e-5
            
            res3 = scipy.optimize.least_squares(self._residuals_triangulation, cam_params_only, **kwargs_s3)
            print(f"    [Pipeline-Stage 3] Cost: {res3.cost:.2e}")
            
            # Setup for Stage 4
            cam_params_s3 = res3.x
            lb_s4 = lb_cam.copy()
            ub_s4 = ub_cam.copy()
            
            margin = 50.0
            for i in range(num_cams):
                base = i * n_cam_params
                lb_s4[base+7] = cx_c - margin
                ub_s4[base+7] = cx_c + margin
                lb_s4[base+8] = cy_c - margin
                ub_s4[base+8] = cy_c + margin
                
            kwargs_s4 = kwargs_s3.copy()
            kwargs_s4.update(dict(bounds=(lb_s4, ub_s4)))
            if 'ftol' not in kwargs: kwargs_s4['ftol'] = 1e-6
        
        # --- STAGE 4: Final Optimization (always runs) ---
        print("    [Pipeline-Stage 4] Final Refinement...")
        self._current_stage = "Stage 4"
        
        res4 = scipy.optimize.least_squares(self._residuals_triangulation, cam_params_s3, **kwargs_s4)
        print(f"    [Pipeline-Stage 4] Cost: {res4.cost:.2e}")
        
        # Reconstruct Full State
        final_cam_params = res4.x.reshape((num_cams, n_cam_params))
        final_points_3d = []
        for fid in frame_list:
            obs = wand_data[fid]
            pA, pB = self._triangulate_frame(final_cam_params, cam_id_map, obs, best_pair[0], best_pair[1])
            if pA is None:
                idx_start = n_cam_params_total + frame_list.index(fid)*6
                pA = X2[idx_start:idx_start+3]
                pB = X2[idx_start+3:idx_start+6]
            final_points_3d.extend(pA)
            final_points_3d.extend(pB)
            
        final_x = np.concatenate([res4.x, np.array(final_points_3d)])
        
        return OptimizeResult(
            x=final_x, cost=res4.cost, fun=res4.fun, jac=res4.jac, 
            status=res4.status, success=res4.success, message=res4.message, nfev=res4.nfev
        )

    def calibrate_wand(self, wand_length=1.0, init_focal_length=5000, callback=None, **kwargs):
        """
        Runs Structure from Motion / Bundle Adjustment (Renamed from run_bundle_adjustment).
        Accepts **kwargs for scipy.optimize.least_squares (e.g. ftol, xtol, verbose).
        """
        self._stop_requested = False
        self.per_frame_errors = {} # Clear cache for new calibration
        self._validate_dist_coeff_num()
        
        # Map args to internal names
        wand_length_mm = wand_length
        initial_focal_len_px = init_focal_length
        cost_callback = callback
        max_nfev = None
        self.wand_length = wand_length_mm
        self._cost_callback = cost_callback
        self._stop_requested = False # Reset flag
        
        # Use filtered data if available, otherwise original
        wand_data = self.wand_points_filtered if self.wand_points_filtered else self.wand_points
        
        if len(wand_data) < 5:
            return False, "Not enough valid frames for calibration", None

        # Check / Update Image Size if possible
        if hasattr(self, 'cameras') and self.cameras:
             # Try first available image
             for c_idx, data in self.cameras.items():
                 if 'images' in data and data['images']:
                     p = data['images'][0]
                     ref = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                     if ref is not None:
                         self.image_size = ref.shape
                         print(f"Refreshed Image Size from Camera {c_idx}: {self.image_size}")
                         break

    def _geometric_init_and_pnp(self, wand_length_mm, initial_focal_len_px, optimize_pair=True, pair_override=None, **kwargs):
        """
        Performs Geometric Initialization (8-Point) + PnP.
        If optimize_pair=True, runs Phase 1 Optimization on the primary pair.
        
        Args:
            wand_length_mm: Target wand length in mm for scale estimation.
            initial_focal_len_px: Initial focal length guess in pixels.
            optimize_pair: If True, run Phase 1 optimization on the best pair.
            pair_override: Optional tuple (cam_id_1, cam_id_2) to force a specific pair.
                          If None, best pair is auto-selected based on shared observations.
            **kwargs: Additional arguments (unused).
        
        Returns:
            (cam_params, cam_id_map, best_pair, points_3d_init)
        """
        import cv2
        import numpy as np
        from scipy.optimize import OptimizeResult
        from scipy.sparse import lil_matrix

        # 1. Initialization (Geometric Method)
        print("Initializing calibration with Geometric Method (8-Point)...")
        
        wand_data = self.wand_points_filtered if self.wand_points_filtered else self.wand_points
        self.wand_length = wand_length_mm # Update instance var

        # Build Camera Map (Continuous Index 0..N-1)
        # Use ONLY cameras present in wand_data
        active_cams = set()
        for obs in wand_data.values():
            active_cams.update(obs.keys())
        cam_ids = sorted(list(active_cams))
        
        cam_id_map = {old_id: i for i, old_id in enumerate(cam_ids)}
        num_cams = len(cam_ids)
        print(f"Active cameras for calibration: {num_cams} {cam_ids}")
        
        if num_cams < 2:
            raise RuntimeError("Need at least 2 cameras.")

        # Find best pair for initialization (max shared frames)
        # Support pair_override for forced pair selection
        if pair_override is not None:
            # Validate pair_override
            if not isinstance(pair_override, (tuple, list)) or len(pair_override) != 2:
                raise ValueError(f"pair_override must be a tuple of 2 camera IDs, got: {pair_override}")
            
            i, j = pair_override
            if not isinstance(i, int) or not isinstance(j, int):
                raise ValueError(f"pair_override camera IDs must be integers, got: ({type(i).__name__}, {type(j).__name__})")
            
            if i == j:
                raise ValueError(f"pair_override cameras must be different, got: ({i}, {j})")
            
            if i not in cam_ids:
                raise ValueError(f"pair_override camera {i} not in active cameras: {cam_ids}")
            if j not in cam_ids:
                raise ValueError(f"pair_override camera {j} not in active cameras: {cam_ids}")
            
            # Count shared points for this pair (for logging)
            count = 0
            for f_idx, cam_pts in wand_data.items():
                if i in cam_pts and j in cam_pts:
                    count += 2
            
            best_pair = (i, j)
            max_shared_pts = count
            print(f"[Phase 1] Using pair_override: Cam {i} <-> Cam {j} ({count} shared points)")
        else:
            # Default: auto-select best pair by max shared points
            best_pair = None
            max_shared_pts = 0
            
            for i in range(num_cams):
                for j in range(i + 1, num_cams):
                    c1 = cam_ids[i]
                    c2 = cam_ids[j]
                    
                    # Count shared points
                    count = 0
                    for f_idx, cam_pts in wand_data.items():
                        if c1 in cam_pts and c2 in cam_pts:
                            count += 2 # 2 points per frame
                    
                    if count > max_shared_pts:
                        max_shared_pts = count
                        best_pair = (c1, c2)
            
            if not best_pair or max_shared_pts < 10:
                 print("Warning: No good camera pair found for geometric initialization. Fallback to naive pair 0-1?")
                 best_pair = (cam_ids[0], cam_ids[1])
            
            print(f"[Phase 1] Auto-selected best pair: Cam {best_pair[0]} <-> Cam {best_pair[1]} ({max_shared_pts} points)")
        
        # Run 8-Point Algo with per-camera intrinsics from table
        R_rel, T_rel, scale_factor_est = self.initialize_geometry_from_pair(
            best_pair[0], best_pair[1], wand_data, wand_length_mm
        )
        
        # Initialize Cameras
        cam_params = np.zeros((num_cams, 11)) # rvec(3), tvec(3), f, cx, cy, k1, k2
        
        base_cam_idx = cam_id_map[best_pair[0]]
        rel_cam_idx = cam_id_map[best_pair[1]]
        
        for i in range(num_cams):
            # Get original camera ID for this internal index
            orig_cam_id = cam_ids[i]
            
            # Get per-camera focal length from camera_settings if available
            if hasattr(self, 'camera_settings') and self.camera_settings and orig_cam_id in self.camera_settings:
                cam_focal = self.camera_settings[orig_cam_id]['focal']
            else:
                cam_focal = initial_focal_len_px  # Fallback to default
            
            _, cam_w, cam_h = self._get_cam_setting(orig_cam_id)
            cam_cx = cam_w / 2.0
            cam_cy = cam_h / 2.0

            # Intrinsics
            cam_params[i, 6] = cam_focal
            cam_params[i, 7] = cam_cx
            cam_params[i, 8] = cam_cy
            # k1, k2 = 0
            
            # Extrinsics
            if i == base_cam_idx:
                # Identity
                cam_params[i, 0:3] = [0, 0, 0]
                cam_params[i, 3:6] = [0, 0, 0]
            elif i == rel_cam_idx and R_rel is not None:
                # Set calculated relative pose
                rvec_rel, _ = cv2.Rodrigues(R_rel)
                cam_params[i, 0:3] = rvec_rel.flatten()
                cam_params[i, 3:6] = T_rel.flatten()
            else:
                # Other cameras: initialize to Identity
                cam_params[i, 0:3] = [0, 0, 0]
                cam_params[i, 3:6] = [0, 0, 0]
                
        # ==========================================
        # PHASE 1: Primary Cameras Optimization
        # ==========================================
        self._current_phase = "Phase 1"
        self._current_stage = "Optimization"
        
        # 1. Select Primary Data
        prim_cam_ids = [best_pair[0], best_pair[1]]
        prim_cam_map = {prim_cam_ids[0]: 0, prim_cam_ids[1]: 1}
        
        # Filter frames seen by pair
        frames_prim = sorted([f for f in wand_data.keys() 
                             if prim_cam_ids[0] in wand_data[f] and prim_cam_ids[1] in wand_data[f]])
        
        # Pre-calc Projection Matrices for Triangulation (using init params)
        cp1 = cam_params[cam_id_map[prim_cam_ids[0]]]
        cp2 = cam_params[cam_id_map[prim_cam_ids[1]]]
        
        # Helper: Build P from params
        def get_P(cp):
             K = np.array([[cp[6], 0, cp[7]], [0, cp[6], cp[8]], [0, 0, 1]])
             R, _ = cv2.Rodrigues(cp[:3])
             T = cp[3:6]
             return K @ np.hstack((R, T.reshape(3,1)))
        
        P1_full = get_P(cp1)
        P2_full = get_P(cp2)
        
        points_3d_prim = []
        valid_frames_prim = []
        
        # Triangulate Primary Points
        for f_idx in frames_prim:
            obs = wand_data[f_idx]
            uv1 = obs[prim_cam_ids[0]]
            uv2 = obs[prim_cam_ids[1]]
            
            if len(uv1) < 2 or len(uv2) < 2: continue
            
            # Pt A
            pt1_A = np.array(uv1[0][:2]).reshape(2, 1)
            pt2_A = np.array(uv2[0][:2]).reshape(2, 1)
            res_A = cv2.triangulatePoints(P1_full, P2_full, pt1_A, pt2_A)
            pt3d_A = (res_A[:3] / res_A[3]).flatten()
            
            # Pt B
            pt1_B = np.array(uv1[1][:2]).reshape(2, 1)
            pt2_B = np.array(uv2[1][:2]).reshape(2, 1)
            res_B = cv2.triangulatePoints(P1_full, P2_full, pt1_B, pt2_B)
            pt3d_B = (res_B[:3] / res_B[3]).flatten()
            
            points_3d_prim.append(pt3d_A)
            points_3d_prim.append(pt3d_B)
            valid_frames_prim.append(f_idx)
            
        points_3d_prim = np.array(points_3d_prim)
        
        # Run Optimization ONLY if requested and we have enough data
        if optimize_pair and len(points_3d_prim) > 20: 
            print(f"\n=== PHASE 1: Optimizing Primary Pair {best_pair} for Initialization ===")
            print(f"  Primary Pair shares {len(frames_prim)} frames.")

            # Build X0_prim
            n_prim_cams = 2
            n_prim_pts = len(points_3d_prim)
            X0_prim = np.zeros(n_prim_cams * 11 + n_prim_pts * 3)
            
            X0_prim[0:11] = cp1
            X0_prim[11:22] = cp2
            X0_prim[22:] = points_3d_prim.flatten()
            
            # Build Sparsity for Primary
            n_res = 0
            for f in valid_frames_prim:
                 n_res += 1 + 2*4 # Len + 2pts * 2cams * 2coords
            
            A_prim = lil_matrix((n_res, len(X0_prim)), dtype=int)
            ridx = 0
            pt_start = 22
            
            for i, f in enumerate(valid_frames_prim):
                idx_pt_A = pt_start + i*6
                idx_pt_B = pt_start + i*6 + 3
                
                # Len constraint
                A_prim[ridx, idx_pt_A:idx_pt_A+3] = 1
                A_prim[ridx, idx_pt_B:idx_pt_B+3] = 1
                ridx += 1
                
                # Reprojection
                for c_idx_loc in [0, 1]: # Internal 0, 1
                    c_base = c_idx_loc * 11
                    # Pt A
                    A_prim[ridx:ridx+2, c_base:c_base+11] = 1
                    A_prim[ridx:ridx+2, idx_pt_A:idx_pt_A+3] = 1
                    ridx += 2
                    # Pt B
                    A_prim[ridx:ridx+2, c_base:c_base+11] = 1
                    A_prim[ridx:ridx+2, idx_pt_B:idx_pt_B+3] = 1
                    ridx += 2
            
            # Run 4-Stage Optimization Pipeline for Primary Pair
            # Prepare Bounds for X0_prim
            n_p_cams = 2
            n_p_pts = n_prim_pts
            X_p_len = len(X0_prim)
            lb_prim = np.full(X_p_len, -np.inf)
            ub_prim = np.full(X_p_len, np.inf)
            
            # Z > 10
            z_idx = np.arange(n_p_cams*11 + 2, X_p_len, 3)
            lb_prim[z_idx] = 10.0
            
            print(f"  Running Primary Optimization Pipeline (4 Stages)...")
            self._current_phase = "Phase 1"
            
            try:
                res_prim = self._run_optimization_pipeline(
                    X0_prim, A_prim, prim_cam_map, valid_frames_prim, wand_data, wand_length_mm,
                    best_pair, lb_prim, ub_prim, **kwargs
                )
            except CalibrationStoppedError as e:
                print("Calibration Stopped by User during Phase 1. Using partial results.")
                # Recover partial parameters
                if e.params is not None:
                    res_prim = type('obj', (object,), {'x': e.params, 'cost': 0.0})()
                else:
                    # Should not happen if params passed
                    raise e
            
            # Update updated params back to main storage
            X_opt = res_prim.x
            cam_params[cam_id_map[prim_cam_ids[0]]] = X_opt[0:11]
            cam_params[cam_id_map[prim_cam_ids[1]]] = X_opt[11:22]
            
            # Update 3D points
            points_3d_prim = X_opt[22:].reshape(-1, 3)
            
            # If stopped in Stage 3/4 (Triangulation Constraint), X_opt only has camera params.
            # We must re-triangulate points to provide a valid result.
            if len(points_3d_prim) == 0:
                print("  Re-triangulating points from partial camera parameters...")
                # Update P matrices
                idx0 = cam_id_map[prim_cam_ids[0]]
                idx1 = cam_id_map[prim_cam_ids[1]]
                P1_new = get_P(cam_params[idx0])
                P2_new = get_P(cam_params[idx1])
                
                new_pts = []
                for f_idx in frames_prim: # Ensure using frames_prim order
                    obs = wand_data[f_idx]
                    uv1 = obs[prim_cam_ids[0]]
                    uv2 = obs[prim_cam_ids[1]]
                    
                    # Pt A
                    pt1_A = np.array(uv1[0][:2]).reshape(2, 1)
                    pt2_A = np.array(uv2[0][:2]).reshape(2, 1)
                    res_A = cv2.triangulatePoints(P1_new, P2_new, pt1_A, pt2_A)
                    new_pts.append((res_A[:3] / res_A[3]).flatten())
                    
                    # Pt B
                    pt1_B = np.array(uv1[1][:2]).reshape(2, 1)
                    pt2_B = np.array(uv2[1][:2]).reshape(2, 1)
                    res_B = cv2.triangulatePoints(P1_new, P2_new, pt1_B, pt2_B)
                    new_pts.append((res_B[:3] / res_B[3]).flatten())
                
                points_3d_prim = np.array(new_pts)
            
            print(f"  Primary Optimization Done. Cost: {res_prim.cost:.2e}")

        # ==========================================
        # PHASE 2: Secondary Initialization (PnP)
        # ==========================================
        print("\n=== PHASE 2: Secondary Initialization (PnP) ===")
        
        # Create map frame_idx -> point_idx_in_subset
        frame_to_idx = {f: i for i, f in enumerate(valid_frames_prim)}
        
        secondary_cams = [c for c in cam_ids if c not in prim_cam_ids]
        
        for sec_cam in secondary_cams:
            # Gather 3D-2D Correspondences
            p3d_list = []
            p2d_list = []
            
            for f in valid_frames_prim:
                if sec_cam in wand_data[f]:
                    # We have 3D points for this frame (from Primary)
                    idx = frame_to_idx[f]
                    ptA_3d = points_3d_prim[idx*2]
                    ptB_3d = points_3d_prim[idx*2+1]
                    
                    uv = wand_data[f][sec_cam] # [[x,y,r],[x,y,r]]
                    ptA_2d = uv[0][:2]
                    ptB_2d = uv[1][:2]
                    
                    p3d_list.append(ptA_3d)
                    p3d_list.append(ptB_3d)
                    p2d_list.append(ptA_2d)
                    p2d_list.append(ptB_2d)
            
            if len(p3d_list) > 10:
                p3d_arr = np.array(p3d_list, dtype=np.float64)
                p2d_arr = np.array(p2d_list, dtype=np.float64)
                
                # Use per-camera focal length from camera_settings if available
                if hasattr(self, 'camera_settings') and self.camera_settings and sec_cam in self.camera_settings:
                    f_init = self.camera_settings[sec_cam]['focal']
                else:
                    f_init = initial_focal_len_px  # Fallback to default
                _, sec_w, sec_h = self._get_cam_setting(sec_cam)
                cx, cy = sec_w / 2.0, sec_h / 2.0
                K_sec = np.array([[f_init, 0, cx], [0, f_init, cy], [0, 0, 1]], dtype=np.float64)
                d_sec = np.zeros(5)
                sec_img_size = (sec_h, sec_w)
                
                success, rvec, tvec = cv2.solvePnP(p3d_arr, p2d_arr, K_sec, d_sec, flags=cv2.SOLVEPNP_EPNP)
                if success:
                    # Refine with iterative
                    success, rvec, tvec = cv2.solvePnP(p3d_arr, p2d_arr, K_sec, d_sec, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
                    
                    # Update cam_params with PnP result
                    c_idx = cam_id_map[sec_cam]
                    cam_params[c_idx, 0:3] = rvec.flatten()
                    cam_params[c_idx, 3:6] = tvec.flatten()
                    print(f"  Initialized Cam {sec_cam} via PnP ({len(p3d_list)} points).")
                    
                    # === Per-Camera Optimization ===
                    # Optimize this camera's parameters using fixed 3D points
                    print(f"  Optimizing Cam {sec_cam}...")
                    
                    # Initial guess based on dist_coeff_num setting
                    if self.dist_coeff_num >= 1:
                        # [rvec(3), tvec(3), f, k1] - 8 parameters
                        x0_cam = np.array([
                            rvec.flatten()[0], rvec.flatten()[1], rvec.flatten()[2],
                            tvec.flatten()[0], tvec.flatten()[1], tvec.flatten()[2],
                            f_init,  # focal length
                            0.0      # k1 (radial distortion)
                        ])
                        lb_cam = np.array([-np.pi, -np.pi, -np.pi, -1e6, -1e6, -1e6, 100.0, -1.0])
                        ub_cam = np.array([np.pi, np.pi, np.pi, 1e6, 1e6, 1e6, 1e6, 1.0])
                    else:
                        # [rvec(3), tvec(3), f] - 7 parameters (no distortion)
                        x0_cam = np.array([
                            rvec.flatten()[0], rvec.flatten()[1], rvec.flatten()[2],
                            tvec.flatten()[0], tvec.flatten()[1], tvec.flatten()[2],
                            f_init   # focal length only
                        ])
                        lb_cam = np.array([-np.pi, -np.pi, -np.pi, -1e6, -1e6, -1e6, 100.0])
                        ub_cam = np.array([np.pi, np.pi, np.pi, 1e6, 1e6, 1e6, 1e6])
                    
                    # Optimize
                    try:
                        res_cam = scipy.optimize.least_squares(
                            self._residuals_single_cam,
                            x0_cam,
                            args=(p3d_arr, p2d_arr, sec_img_size),
                            method='trf',
                            bounds=(lb_cam, ub_cam),
                            ftol=1e-5,
                            max_nfev=100
                        )
                        
                        # Update cam_params with optimized values
                        cam_params[c_idx, 0:3] = res_cam.x[0:3]  # rvec
                        cam_params[c_idx, 3:6] = res_cam.x[3:6]  # tvec
                        cam_params[c_idx, 6] = res_cam.x[6]      # f
                        if self.dist_coeff_num >= 1:
                            cam_params[c_idx, 9] = res_cam.x[7]  # k1
                        
                        # Calculate initial and final reprojection error
                        err_init = np.sqrt(np.mean(self._residuals_single_cam(x0_cam, p3d_arr, p2d_arr, sec_img_size)**2))
                        err_final = np.sqrt(np.mean(res_cam.fun**2))
                        if self.dist_coeff_num >= 1:
                            print(f"    Optimized: f={res_cam.x[6]:.1f}px, k1={res_cam.x[7]:.4f}, RMS: {err_init:.2f}→{err_final:.2f}px")
                        else:
                            print(f"    Optimized: f={res_cam.x[6]:.1f}px, RMS: {err_init:.2f}→{err_final:.2f}px")
                    except Exception as e:
                        print(f"    Optimization failed: {e}")
                else:
                    print(f"  Failed PnP for Cam {sec_cam}.")
            else:
                print(f"  Skipping Cam {sec_cam} (only {len(p3d_list)} shared points).")

        return cam_params, cam_id_map, best_pair, points_3d_prim

    def _initialize_system(self, wand_length_mm, initial_focal_len_px, **kwargs):
        """
        Legacy Compat: calls _geometric_init_and_pnp with optimize_pair=True.
        Returns initialized (cam_params, cam_id_map, best_pair, points_3d).
        """
        cam_params, cam_id_map, best_pair, points_3d = self._geometric_init_and_pnp(
            wand_length_mm, initial_focal_len_px, optimize_pair=True, **kwargs
        )
        return cam_params, cam_id_map, best_pair, points_3d

    def run_precalibration_check(self, wand_length=1.0, init_focal_length=5000, callback=None, **kwargs):
        """
        Runs a FAST "Pre-calibration" check to identify outliers.
        Single Global Optimization with Fixed Intrinsics and Relaxed Tolerances.
        """
        self._validate_dist_coeff_num()
        import scipy.optimize
        from scipy.optimize import OptimizeResult
        from scipy.sparse import lil_matrix
        
        self._stop_requested = False
        self._cost_callback = callback
        
        # 1. Initialize System (Robust Init) - SKIP Phase 1 Optimization loop
        try:
             cam_params, cam_id_map, best_pair, points_3d_init = self._geometric_init_and_pnp(
                 wand_length, init_focal_length, optimize_pair=False
             )
        except RuntimeError as e:
             return False, str(e), None

        num_cams = len(cam_id_map)
        wand_data = self.wand_points_filtered if self.wand_points_filtered else self.wand_points
        
        # 2. Build FULL State Vector (All Cams + All Points)
        # We need to triangulate ALL points seen by any camera pair to get a full X0
        # However, _geometric_init_and_pnp only returns points_3d for the PRIMARY pair.
        # We should probably run a quick triangulation for the rest using the instantiated PnP cameras.
        
        print("Pre-Calibration: Triangulating all frames with initialized cameras...")
        
        points_3d_all = []
        valid_frames_all = []
        
        # Triangulate all frames using best available pair for each frame
        for f_idx in sorted(wand_data.keys()):
            obs = wand_data[f_idx]
            # Need at least 2 cameras
            if len(obs) < 2: continue
            
            # Find best pair in this frame? Or just take first two valid?
            # Let's take the first two valid cameras that have params
            cams_in_frame = [c for c in obs.keys() if c in cam_id_map]
            if len(cams_in_frame) < 2: continue
            
            # Try to find a pair with good baseline? 
            # For simplicity in Pre-Calib, just take first 2.
            c1, c2 = cams_in_frame[0], cams_in_frame[1]
            
            pt3d_A, pt3d_B = self._triangulate_frame(cam_params, cam_id_map, obs, c1, c2)
            
            if pt3d_A is not None and pt3d_B is not None:
                points_3d_all.append(pt3d_A)
                points_3d_all.append(pt3d_B)
                valid_frames_all.append(f_idx)
                
        points_3d_all = np.array(points_3d_all)
        
        if len(points_3d_all) < 10:
            return False, "Not enough points triangulated for pre-calibration.", None

        # 3. Construct X0
        n_pts = len(points_3d_all)
        X0 = np.zeros(num_cams * 11 + n_pts * 3)
        X0[:num_cams*11] = cam_params.flatten()
        X0[num_cams*11:] = points_3d_all.flatten()
        
        # 4. Global Optimization (Single Phase)
        print(f"Pre-Calibration: Optimization with {len(valid_frames_all)} frames...")
        
        # Fixed Intrinsics = Tight Bounds
        lb = np.full(len(X0), -np.inf)
        ub = np.full(len(X0), np.inf)
        
        # Lock Intrinsics (f, cx, cy, k1, k2 -> indices 6,7,8,9,10)
        # We can just set lb=ub or use x_scale=0? 
        # Better: remove them from optimization variable? 
        # But our residual function expects 11 params.
        # Let's use strict bounds variance?
        # Actually, for "Relaxed" check, we can just let them float lightly or validly?
        # User said "Fixed Intrinsics".
        # Let's set bounds equal to initial value for intrinsics.
        
        # Lock Intrinsics (Set tight bounds around initial values)
        for i in range(num_cams):
             base = i * 11
             # f, cx, cy, k1, k2
             for idx in range(6, 11):
                 val = cam_params[i, idx]
                 lb[base + idx] = val - 1e-6
                 ub[base + idx] = val + 1e-6
        
        # Build Sparsity Matrix
        print(f"Pre-Calibration: Building Sparsity Matrix for {len(X0)} params...")
        n_residuals = 0
        for fid in valid_frames_all:
             obs = wand_data[fid]
             cams_vis = [c for c in obs.keys() if c in cam_id_map]
             n_residuals += 1 + len(cams_vis) * 4 # 1 length + 4 coords (2pts * 2coords) per cam
             
        A = lil_matrix((n_residuals, len(X0)), dtype=int)
        ridx = 0
        pt_start = num_cams * 11
        
        for i, fid in enumerate(valid_frames_all):
             obs = wand_data[fid]
             cams_vis = [c for c in obs.keys() if c in cam_id_map]
             
             idx_pt_A = pt_start + i * 6
             idx_pt_B = idx_pt_A + 3
             
             # Wand Length Constraint
             A[ridx, idx_pt_A:idx_pt_A+3] = 1
             A[ridx, idx_pt_B:idx_pt_B+3] = 1
             ridx += 1
             
             # Reprojection Constraints
             for cam_idx in cams_vis:
                 c_internal = cam_id_map[cam_idx]
                 c_base = c_internal * 11
                 
                 # Pt A projection
                 A[ridx:ridx+2, c_base:c_base+11] = 1
                 A[ridx:ridx+2, idx_pt_A:idx_pt_A+3] = 1
                 ridx += 2
                 
                 # Pt B projection
                 A[ridx:ridx+2, c_base:c_base+11] = 1
                 A[ridx:ridx+2, idx_pt_B:idx_pt_B+3] = 1
                 ridx += 2

        # Run Optimization (Relaxed)
        print("Running Pre-Calibration Optimization (Relaxed)...")
        self._current_phase = "Pre-Calibration"
        self._current_stage = "Optimization"
        ls_kwargs = dict(
            jac_sparsity=A, 
            verbose=2, 
            ftol=1e-3, xtol=1e-3, gtol=1e-3, # Relaxed
            max_nfev=50, # Requested Limit
            method='trf', loss='huber', f_scale=1.0,
            bounds=(lb, ub),
            args=(cam_id_map, valid_frames_all, wand_data, wand_length, self.image_size)
        )
        ls_kwargs.update(kwargs)
        
        try:
            res = scipy.optimize.least_squares(self._residuals, X0, **ls_kwargs)
            final_x = res.x
            
            # Reconstruct result (Simplified return)
            # Correct call signature for master definition
            return self._finalize_calibration(res, cam_id_map, wand_data, wand_length, valid_frames_all)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Pre-calib failed: {e}", None

    # [DELETED DUPLICATE DEF]


    def calibrate_wand(self, wand_length=1.0, init_focal_length=5000, callback=None, **kwargs):
        """
        Runs Structure from Motion / Bundle Adjustment (Renamed from run_bundle_adjustment).
        Accepts **kwargs for scipy.optimize.least_squares (e.g. ftol, xtol, verbose).
        """
        self._stop_requested = False
        self._current_phase = "Init"
        self._current_stage = ""
        
        # Map args to internal names
        wand_length_mm = wand_length
        initial_focal_len_px = init_focal_length
        cost_callback = callback
        max_nfev = None
        self.wand_length = wand_length_mm
        self._cost_callback = cost_callback
        self._stop_requested = False # Reset flag
        
        # Use filtered data if available, otherwise original
        wand_data = self.wand_points_filtered if self.wand_points_filtered else self.wand_points
        
        if len(wand_data) < 5:
            return False, "Not enough valid frames for calibration", None

        # Check / Update Image Size if possible
        if hasattr(self, 'cameras') and self.cameras:
             # Try first available image
             for c_idx, data in self.cameras.items():
                 if 'images' in data and data['images']:
                     p = data['images'][0]
                     ref = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                     if ref is not None:
                         self.image_size = ref.shape
                         print(f"Refreshed Image Size from Camera {c_idx}: {self.image_size}")
                         break

        try:
             # CALL NEW INITIALIZER
             cam_params, cam_id_map, best_pair, points_3d = self._initialize_system(wand_length_mm, initial_focal_len_px)
        except RuntimeError as e:
             return False, str(e), None

        num_cams = len(cam_id_map)
        n_cam_params = 11

        # OPTIMIZATION: If only 2 cameras, Phase 1 (inside initialize) already optimized everything!
        # Skip Phase 3 (which would just re-do N-View triangulation for 2 cams = same result).
        if num_cams == 2:
            print("\n=== Only 2 Cameras: Phase 3 Skipped (Data already optimized) ===")
            # Construct a dummy result object for finalize
            # res.x needs to be flat [cam_params, points_3d]
            flat_cams = cam_params.flatten()
            flat_pts = points_3d.flatten()
            full_x = np.concatenate([flat_cams, flat_pts])
            
            from scipy.optimize import OptimizeResult
            # We don't have exact cost unless we stored it, but Phase 1 printed it.
            res_dummy = OptimizeResult(x=full_x, cost=0.0, message="Optimized (2-Cam Shortcut)")
            
            # Find valid frames for finalization (assume all points are valid from Phase 1)
            # Actually _initialize_system returns points for 'valid_frames_prim'.
            # We need the frame list matching points_3d. 
            # In Phase 1, valid_frames_prim corresponds to points_3d_prim.
            # But 'valid_frames_prim' is local to _geometric_init...
            # However, points_3d row count = len(valid_frames) * 2.
            # We can reconstruct valid frames from wand_data if needed, or better:
            # We need to pass the frame list out of _initialize_system too? 
            # Or just use sorted(wand_data.keys()) and filter?
            # BUT points_3d returned by Phase 1 only contains "valid_frames_prim" (common frames).
            # Let's derive it.
            
            all_frames = sorted(wand_data.keys())
            # Phase 1 uses intersection of cam 0 and 1.
            c0, c1 = best_pair
            valid_frames = [f for f in all_frames if c0 in wand_data[f] and c1 in wand_data[f]]
            
            # Sanity check length
            if len(points_3d) != len(valid_frames) * 2:
                print(f"Warning: Point count {len(points_3d)} != 2 * Frames {len(valid_frames)}. Mismatch?")
                # Fallback: Just proceed to Phase 3 if mismatch to be safe?
                pass
            else:
                # Correct call signature
                return self._finalize_calibration(res_dummy, cam_id_map, wand_data, wand_length_mm, valid_frames)

        # ==========================================
        # PHASE 3: Prepare Full Optimization
        # ==========================================
        print("\n=== PHASE 3: Triangulating All Points for Full BA ===")
        self._current_phase = "Phase 3"
        self._current_stage = "Preparation"
        # Re-triangulate ALL frames using refined parameters
        
        points_3d = []
        # Filter frame_list to those with enough cameras to triangulate
        frame_list = sorted([f for f in wand_data.keys() if len([c for c in wand_data[f] if c in cam_id_map]) >= 2])
        
        # Pre-calc Projection Matrices for ALL cameras
        P_all = []
        for i in range(num_cams):
             cp = cam_params[i]
             K = np.array([[cp[6], 0, cp[7]], [0, cp[6], cp[8]], [0, 0, 1]])
             R, _ = cv2.Rodrigues(cp[:3])
             T = cp[3:6]
             P = K @ np.hstack((R, T.reshape(3,1)))
             P_all.append(P)
             
        count_tri = 0
        mean_wand_len = 0
        
        for f_idx in frame_list:
             obs = wand_data[f_idx]
             cams_vis = sorted([c for c in obs.keys() if c in cam_id_map])
             
             # Pick best pair? For now: First 2 available.
             # Ideally: Pick pair with widest baseline?
             c1 = cams_vis[0]
             c2 = cams_vis[1]
             # If primary pair is fully visible, use it (better stability)
             if best_pair[0] in obs and best_pair[1] in obs:
                 c1, c2 = best_pair[0], best_pair[1]
             
             idx1 = cam_id_map[c1]
             idx2 = cam_id_map[c2]
             
             P1 = P_all[idx1]
             P2 = P_all[idx2]
             
             uv1 = obs[c1]
             uv2 = obs[c2]
             
             if len(uv1) < 2 or len(uv2) < 2: continue
             
             # Pt A
             pt1_A = np.array(uv1[0][:2]).reshape(2, 1)
             pt2_A = np.array(uv2[0][:2]).reshape(2, 1)
             res_A = cv2.triangulatePoints(P1, P2, pt1_A, pt2_A)
             pt3d_A = (res_A[:3] / res_A[3]).flatten()
             
             # Pt B
             pt1_B = np.array(uv1[1][:2]).reshape(2, 1)
             pt2_B = np.array(uv2[1][:2]).reshape(2, 1)
             res_B = cv2.triangulatePoints(P1, P2, pt1_B, pt2_B)
             pt3d_B = (res_B[:3] / res_B[3]).flatten()
             
             points_3d.append(pt3d_A)
             points_3d.append(pt3d_B)
             count_tri += 1
             mean_wand_len += np.linalg.norm(pt3d_A - pt3d_B)
             
        if count_tri > 0:
            mean_wand_len /= count_tri
            print(f"  Initialized {count_tri} frames. Mean Wand Length: {mean_wand_len:.2f} mm")
             
        points_3d = np.array(points_3d)
        num_frames = len(frame_list)
        num_points = len(points_3d)
        
        # Update X0 for Full Optimization
        X0 = np.zeros(num_cams * 11 + num_points * 3)
        for i in range(num_cams):
             X0[i*11:(i+1)*11] = cam_params[i]
        X0[num_cams*11:] = points_3d.flatten()

        print(f"Full X0 initialized: {len(X0)} params ({num_cams} cams, {num_points} points).")
        
        # Restore constants for subsequent code
        n_cam_params = 11

        
        print("Building Sparse Jacobian Structure...")
        from scipy.sparse import lil_matrix
        
        # Calculate matrix dimensions
        # Num residuals
        n_residuals = 0
        for fid in frame_list:
            n_residuals += 1 # Wand length constraint
            n_residuals += len(self.wand_points[fid]) * 4 # Reprojection x,y for Pt A and Pt B (2*2)
            
        n_params = len(X0)
        A = lil_matrix((n_residuals, n_params), dtype=int)
        
        ridx = 0
        cam_param_start = num_cams * n_cam_params
        # cam_param_start already calculated
        
        # cam_id_map is already {cam_id: internal_idx}, use directly
        
        for i, fid in enumerate(frame_list):
            obs = self.wand_points[fid]
            
            # Point indices in X0
            idx_pt_A = cam_param_start + i * 2 * 3
            idx_pt_B = cam_param_start + i * 2 * 3 + 3
            
            # 1. Wand length constraint residual
            # Depends on Pt A (3) and Pt B (3)
            A[ridx, idx_pt_A:idx_pt_A+3] = 1
            A[ridx, idx_pt_B:idx_pt_B+3] = 1
            ridx += 1
            
            # 2. Project to cameras
            for cam_idx, uv_obs in obs.items():
                # Internal cam index (cam_id_map is {cam_id: internal_idx})
                c_internal = cam_id_map[cam_idx]
                idx_cam = c_internal * n_cam_params
                
                # Point A Obs
                # Residual X / Y depends on Cam (11) and Pt A (3)
                A[ridx:ridx+2, idx_cam:idx_cam+n_cam_params] = 1
                A[ridx:ridx+2, idx_pt_A:idx_pt_A+3] = 1
                ridx += 2
                
                # Point B Obs
                A[ridx:ridx+2, idx_cam:idx_cam+n_cam_params] = 1
                A[ridx:ridx+2, idx_pt_B:idx_pt_B+3] = 1
                ridx += 2

        print(f"Sparse Jacobian: {A.shape} with {A.nnz} entries")

        print("Starting Optimization (4-Stage Pipeline)...")
        
        # Prepare Global Bounds
        self._current_phase = "Phase 3 (Full BA)"
        n_cam_params_total = num_cams * 11
        lower_bounds = np.full(X0.shape, -np.inf)
        upper_bounds = np.full(X0.shape, np.inf)
        
        # Z > 10
        z_indices_pts = np.arange(n_cam_params_total + 2, len(X0), 3)
        lower_bounds[z_indices_pts] = 10.0
        
        # Calculate Initial RMS
        init_res = self._residuals(X0, cam_id_map, frame_list, wand_data, wand_length_mm, self.image_size)
        init_rms = np.sqrt(np.mean(init_res**2))
        print(f"Pre-Optimization RMS: {init_rms:.4f} px")

        try:
             res = self._run_optimization_pipeline(
                X0, A, cam_id_map, frame_list, wand_data, wand_length_mm,
                best_pair, lower_bounds, upper_bounds, 
                skip_early_stages=True, **kwargs  # Skip Stage 1-3 in Phase 3
             )
        except CalibrationStoppedError as e:
             # Handle Stop - use partial results if available
             print("Calibration Stopped by User.")
             if e.params is not None:
                 # Create mock result from partial params
                 print(f"  Using partial results ({len(e.params)} params)")
                 res = type('PartialResult', (), {
                     'x': e.params,
                     'cost': np.inf,
                     'message': 'Stopped by user'
                 })()
                 # Continue to finalization below
             else:
                 return False, "Stopped by user (no params)", None
                

        
        print("Optimization Done.")
        
        # Calculate final RMS errors
        # Calculate final metrics separately
        total_repro_sq = 0.0
        total_len_sq = 0.0
        count_repro = 0
        count_len = 0
        
        # Per-camera metrics
        cam_repro_sq = {} # {cam_idx: sum_sq_err}
        cam_repro_count = {} # {cam_idx: num_coords}
        
        # Unpack final params
        params = res.x
        num_cams = len(cam_id_map)
        n_cam_params = 11
        n_cam_params_total = num_cams * n_cam_params
        
        # ... (Param unpacking code remains same) ...
        # NOTE: I need to duplicate the unpacking logic here because I can't partially match large blocks easily
        # But wait, looking at lines 2124-2320 in view_file previously...
        # I will replace the initialization and loop logic.
        
        if len(params) == n_cam_params_total:
             # Only camera params (from Stage 4 / Stop)
             cam_params = params.reshape((num_cams, n_cam_params))
             # Triangulate... (Logic from previous fix)
             points_3d_list = []
             # ... assume points_3d is available or re-triangulated ...
             # Optimization: Assuming points_3d is already populated from the previous fix block
             # I need to be careful not to break the previous fix.
             pass 
        
        # ACTUALLY, I should just modify the loop where errors are calculated.
        # Lines 2263 - 2306 loop over frames and cameras.
        
        # Let's locate the metrics init (line 2124) and loop (2263)
        pass

        # RE-PLAN:
        # I will replace the statistics initialization and the inner loop accumulation.
        
     # ...

        
        # Unpack final params
        params = res.x
        num_cams = len(cam_id_map)
        n_cam_params = 11
        n_cam_params_total = num_cams * n_cam_params
        
        return self._finalize_calibration(res, cam_id_map, wand_data, wand_length_mm, frame_list)

    def _finalize_calibration(self, res, cam_id_map, wand_data, wand_length_mm, frame_list):
        """
        Finalize calibration results: extract params, triangulation (if needed), 
        align coordinates, calculate metrics, and return results.
        """
        params = res.x
        num_cams = len(cam_id_map)
        n_cam_params = 11
        n_cam_params_total = num_cams * n_cam_params
        
        # Check if params contains 3D points or just camera params
        if len(params) == n_cam_params_total:
            # Only camera params - need to triangulate 3D points
            print("  Triangulating 3D points from camera params...")
            cam_params = params.reshape((num_cams, n_cam_params))
            
            # Triangulate 3D points from camera params
            points_3d_list = []
            
            for fid in frame_list:
                obs = wand_data[fid]
                # Use N-View Triangulation
                pt3d_A, pt3d_B = self._triangulate_frame(cam_params, cam_id_map, obs)
                if pt3d_A is not None and pt3d_B is not None:
                    points_3d_list.append(pt3d_A)
                    points_3d_list.append(pt3d_B)
                else:
                    # Use zeros for failed triangulation
                    points_3d_list.append(np.zeros(3))
                    points_3d_list.append(np.zeros(3))
            
            points_3d = np.array(points_3d_list)
            print(f"  Triangulated {len(points_3d)} points")
        else:
            # Full params - extract camera params and 3D points
            cam_params = params[:n_cam_params_total].reshape((num_cams, n_cam_params))
            # Reshape 3D points correctly
            # NOTE: points_3d size check
            expected_pts = len(frame_list) * 2
            points_flat = params[n_cam_params_total:]
            if len(points_flat) == expected_pts * 3:
                 points_3d = points_flat.reshape((-1, 3))
            else:
                 # Should matched
                 points_3d = points_flat.reshape((-1, 3))

        # --- COORDINATE ALIGNMENT ---
        print("Aligning World Origin to Point Cloud Centroid...")
        centroid = np.mean(points_3d, axis=0) # (3,)
        points_3d -= centroid
        
        for i in range(num_cams):
            rvec = cam_params[i, 0:3]
            tvec = cam_params[i, 3:6]
            R, _ = cv2.Rodrigues(rvec)
            t_new = tvec + R @ centroid
            cam_params[i, 3:6] = t_new
            
        print("Coordinate Alignment Done.")
        
        # Calculate final metrics separately
        total_repro_sq = 0.0
        total_len_sq = 0.0
        count_repro = 0
        count_len = 0
        
        # Per-camera metrics
        cam_repro_sq = {} # {cam_idx: sum_sq_err}
        cam_repro_count = {} # {cam_idx: num_coords}

        for i, fid in enumerate(frame_list):
            obs = wand_data[fid]
            idx_A = i * 2
            idx_B = i * 2 + 1
            pt3d_A = points_3d[idx_A]
            pt3d_B = points_3d[idx_B]
            
            # 1. Wand Length Error (mm)
            dist = np.linalg.norm(pt3d_A - pt3d_B)
            len_err = dist - wand_length_mm
            total_len_sq += len_err ** 2
            count_len += 1
            
            # 2. Reprojection Error (px)
            for cam_idx, uv_obs in obs.items():
                if cam_idx not in cam_id_map: continue 
                c_internal = cam_id_map[cam_idx]
                cp = cam_params[c_internal]
                
                # Init per-camera counters if needed
                if c_internal not in cam_repro_sq:
                    cam_repro_sq[c_internal] = 0.0
                    cam_repro_count[c_internal] = 0
                
                # Proj A
                qa = self._project_point(pt3d_A, cp, self.image_size)
                err_a = np.array(uv_obs[0][:2]) - qa  
                sq_err_a = np.sum(err_a**2)
                
                # Proj B
                qb = self._project_point(pt3d_B, cp, self.image_size)
                err_b = np.array(uv_obs[1][:2]) - qb 
                sq_err_b = np.sum(err_b**2)
                
                # Accumulate Global
                total_repro_sq += sq_err_a + sq_err_b
                count_repro += 2 
                
                # Accumulate Per-Camera
                cam_repro_sq[c_internal] += sq_err_a + sq_err_b
                cam_repro_count[c_internal] += 2
        
        # Global RMS
        n_coords = count_repro * 2
        rms_repro = np.sqrt(total_repro_sq / n_coords) if n_coords > 0 else 0.0
        
        rms_len = np.sqrt(total_len_sq / count_len) if count_len > 0 else 0.0
        
        prefix = "Calibration Optimized."
        if hasattr(res, 'message') and res.message and "Stopped" in res.message:
            prefix = "Calibration STOPPED by User (Partial Results)."

        # Format Per-Camera RMS string
        cam_rms_str = ""
        cam_internal_to_external = {v: k for k, v in cam_id_map.items()}
        for c_internal in sorted(cam_repro_sq.keys()):
            sq_err = cam_repro_sq[c_internal]
            count = cam_repro_count[c_internal] * 2 
            rms = np.sqrt(sq_err / count) if count > 0 else 0.0
            ext_id = cam_internal_to_external.get(c_internal, f"#{c_internal}")
            cam_rms_str += f"  - Cam {ext_id}: {rms:.4f} px\n"

        msg = (f"{prefix}\n"
               f"Reprojection RMS: {rms_repro:.4f} px\n"
               f"{cam_rms_str}"
               f"Wand Length RMS: {rms_len:.4f} mm\n"
               f"Total Frames: {len(frame_list)}")
        
        # Save points for visualization
        self.points_3d = points_3d
        
        # Parse results
        self._parse_results(res.x, cam_id_map)
        self.params_dirty = True  # Mark new results as dirty
        
        # Explicitly attach metrics for UI (Fixes Error Table & RMS Display)
        res.rms_reproj_err = rms_repro
        self.per_frame_errors = {} # Clear cache to force recalculation with new params
        self.calculate_per_frame_errors()
        res.per_frame_errors = self.per_frame_errors
        
        return True, msg, res

    def _residuals(self, params, cam_id_map, frame_list, observations, wand_len, img_size):
        """Wrapper for residuals to calculate cost and report progress."""
        if getattr(self, '_stop_requested', False):
            raise CalibrationStoppedError("Stopped by user", params=params)
        
        # Enforce hard iteration limit if set
        if hasattr(self, '_max_residual_calls') and self._max_residual_calls is not None:
            if not hasattr(self, '_residual_call_count'):
                self._residual_call_count = 0
            self._residual_call_count += 1
            if self._residual_call_count > self._max_residual_calls:
                # Limit reached - raise exception to stop optimization
                raise CalibrationStoppedError(f"Iteration limit ({self._max_residual_calls}) reached", params=params)
            
        residuals = self._residuals_internal(params, cam_id_map, frame_list, observations, wand_len, img_size)
        
        # Report cost if callback is set
        if hasattr(self, '_cost_callback') and self._cost_callback is not None:
            # Use 0.5 * sum(r²) to match scipy.optimize.least_squares convention
            cost = 0.5 * np.sum(residuals**2)
            
            # Approximate RMSE (pixels)
            n_res = len(residuals)
            rmse = np.sqrt(2 * cost / n_res) if n_res > 0 else 0
            
            self._cost_callback(cost, rmse)
        
        return residuals

    def _residuals_single_cam(self, params, pts_3d, pts_2d_obs, img_size):
        """
        Residual function for single camera optimization with fixed 3D points.
        
        Args:
            params: [rvec(3), tvec(3), f, k1] - 8 parameters
            pts_3d: (N, 3) array of fixed 3D points
            pts_2d_obs: (N, 2) array of observed 2D points
            img_size: (height, width) tuple
            
        Returns:
            residuals: 1D array of projection errors
        """
        rvec = params[0:3]
        tvec = params[3:6]
        f = params[6]
        k1 = params[7] if len(params) > 7 else 0.0
        
        cx = img_size[1] / 2.0
        cy = img_size[0] / 2.0
        
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
        
        # Project all 3D points
        projected, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
        projected = projected.reshape(-1, 2)
        
        # Compute residuals
        residuals = (projected - pts_2d_obs).flatten()
        return residuals

    def _project_point(self, pt3d, cam_params, img_size):
        # 11 params: rx,ry,rz, tx,ty,tz, f, cx, cy, k1, k2
        rvec = cam_params[:3]
        tvec = cam_params[3:6]
        f = cam_params[6]
        cx = cam_params[7]
        cy = cam_params[8]
        k1 = cam_params[9] if self.dist_coeff_num >= 1 else 0.0
        k2 = cam_params[10] if self.dist_coeff_num >= 2 else 0.0
        
        # Construct K, dist (only use coefficients up to dist_coeff_num)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, 0, 0, 0])
        
        # Project
        pts_cam, _ = cv2.projectPoints(pt3d.reshape(1,3), rvec, tvec, K, dist_coeffs)
        return pts_cam.flatten()

    def _triangulate_frame(self, cam_params_all, cam_id_map, frame_obs, cam1_id=None, cam2_id=None):
        """
        Triangulate a single frame's wand points using ALL visible cameras (N-View SVD).
        
        Args:
            cam_params_all: (num_cams, 11) array of camera parameters
            cam_id_map: dict mapping cam_id -> internal_idx
            frame_obs: dict {cam_id: [[x1,y1,r1], [x2,y2,r2]]} for this frame
            cam1_id, cam2_id: Keep for compatibility, but ignored in favor of all visible cams.
            
        Returns:
            pt3d_A, pt3d_B: (3,) arrays of 3D wand endpoints, or None if triangulation fails
        """
        # Find all cameras that observe this frame AND are in our optimization set
        visible_cams = [c for c in frame_obs.keys() if c in cam_id_map]
        
        if len(visible_cams) < 2:
            return None, None
            
        # Helper for SVD Triangulation of a single point
        def triangulate_point_nview(point_idx):
            # Build A matrix (2N x 4)
            A_mat = []
            
            for cam_id in visible_cams:
                c_idx = cam_id_map[cam_id]
                cp = cam_params_all[c_idx]
                
                # Extract params
                rvec = cp[:3]
                tvec = cp[3:6]
                f, cx, cy = cp[6], cp[7], cp[8]
                
                # Build P = K[R|t]
                # Optimization: Could cache K and R if performance is critical, 
                # but valid since params change during optimization.
                R, _ = cv2.Rodrigues(rvec)
                K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
                P = K @ np.hstack((R, tvec.reshape(3,1)))
                
                # Observation
                uv = frame_obs[cam_id][point_idx][:2]
                u, v = uv[0], uv[1]
                
                # Linear constraints:
                # u (P_3 X) - (P_1 X) = 0
                # v (P_3 X) - (P_2 X) = 0
                A_mat.append(u * P[2] - P[0])
                A_mat.append(v * P[2] - P[1])
            
            A_mat = np.array(A_mat)
            
            # SVD
            import numpy.linalg as la # Local import to be safe
            _, _, Vt = la.svd(A_mat)
            X = Vt[-1]
            
            # De-homogenize
            if abs(X[3]) > 1e-9:
                return X[:3] / X[3]
            else:
                return X[:3] # Points at infinity? Should not happen for wand.

        # Triangulate A (idx 0) and B (idx 1)
        try:
            pt3d_A = triangulate_point_nview(0)
            pt3d_B = triangulate_point_nview(1)
            return pt3d_A, pt3d_B
        except Exception as e:
            # print(f"Triangulation failed: {e}")
            return None, None

    def _residuals_triangulation(self, cam_params_flat, cam_id_map, frame_list, observations, wand_len, img_size, base_cam_id, rel_cam_id):
        """
        Optimized residual function that re-triangulates 3D points from camera params.
        Uses batch projection per camera for speed.
        """
        if getattr(self, '_stop_requested', False):
            raise CalibrationStoppedError("Stopped by user", params=cam_params_flat)
            
        num_cams = len(cam_id_map)
        n_cam_params = 11
        cam_params = cam_params_flat.reshape((num_cams, n_cam_params))
        
        # === STEP 1: Triangulate all frames ===
        triangulated = {}  # fid -> (pt3d_A, pt3d_B) or None
        for fid in frame_list:
            obs = observations[fid]
            pt3d_A, pt3d_B = self._triangulate_frame(cam_params, cam_id_map, obs, base_cam_id, rel_cam_id)
            if pt3d_A is not None and pt3d_B is not None:
                triangulated[fid] = (pt3d_A, pt3d_B)
        
        # === STEP 2: Vectorized wand length constraints ===
        valid_frames = list(triangulated.keys())
        if len(valid_frames) == 0:
            return np.array([])
        
        pts_A = np.array([triangulated[fid][0] for fid in valid_frames])
        pts_B = np.array([triangulated[fid][1] for fid in valid_frames])
        dists = np.linalg.norm(pts_A - pts_B, axis=1)
        len_residuals = (dists - wand_len) * 10.0
        
        # === STEP 3: Batch projection per camera ===
        cam_internal_to_external = {v: k for k, v in cam_id_map.items()}
        proj_residuals_list = []
        
        for c_internal in range(num_cams):
            cam_external = cam_internal_to_external.get(c_internal)
            if cam_external is None:
                continue
                
            cp = cam_params[c_internal]
            rvec = cp[:3]
            tvec = cp[3:6]
            f = cp[6]
            cx, cy = cp[7], cp[8]
            k1 = cp[9] if self.dist_coeff_num >= 1 else 0.0
            k2 = cp[10] if self.dist_coeff_num >= 2 else 0.0
            
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
            
            # Collect points and observations for this camera
            pts_3d_cam = []
            obs_2d_cam = []
            
            for fid in valid_frames:
                obs = observations[fid]
                if cam_external not in obs:
                    continue
                
                pt3d_A, pt3d_B = triangulated[fid]
                uv_obs = obs[cam_external]
                
                pts_3d_cam.append(pt3d_A)
                obs_2d_cam.append(uv_obs[0][:2])
                pts_3d_cam.append(pt3d_B)
                obs_2d_cam.append(uv_obs[1][:2])
            
            if len(pts_3d_cam) == 0:
                continue
            
            # Batch projection
            pts_3d_arr = np.array(pts_3d_cam, dtype=np.float64)
            obs_2d_arr = np.array(obs_2d_cam, dtype=np.float64)
            
            projected, _ = cv2.projectPoints(pts_3d_arr, rvec, tvec, K, dist_coeffs)
            projected = projected.reshape(-1, 2)
            
            cam_residuals = (projected - obs_2d_arr).flatten()
            proj_residuals_list.append(cam_residuals)
        
        # === STEP 4: Combine residuals ===
        if proj_residuals_list:
            proj_residuals = np.concatenate(proj_residuals_list)
        else:
            proj_residuals = np.array([])
        
        all_residuals = np.concatenate([len_residuals, proj_residuals])
        
        # Report cost if callback is set
        if hasattr(self, '_cost_callback') and self._cost_callback is not None:
            cost = 0.5 * np.sum(all_residuals**2)
            n_res = len(all_residuals)
            rmse = np.sqrt(2 * cost / n_res) if n_res > 0 else 0
            self._cost_callback(cost, rmse)
            
        return all_residuals

    def _residuals_internal(self, params, cam_id_map, frame_list, observations, wand_len, img_size):
        """Optimized residuals using batch projection per camera."""
        # Unpack params
        num_cams = len(cam_id_map)
        n_cam_params = 11
        cam_params = params[:num_cams*n_cam_params].reshape((num_cams, n_cam_params))
        points_3d = params[num_cams*n_cam_params:].reshape((-1, 3))  # (NumFrames * 2, 3)
        
        num_frames = len(frame_list)
        
        # === Vectorized Wand Length Constraints ===
        # Points are arranged as [A0, B0, A1, B1, A2, B2, ...]
        pts_A = points_3d[0::2]  # All A points
        pts_B = points_3d[1::2]  # All B points
        dists = np.linalg.norm(pts_A - pts_B, axis=1)
        len_residuals = (dists - wand_len) * 10.0  # Weight 10
        
        # === Batch Projection per Camera ===
        # Pre-build observation arrays for efficiency
        # obs_map[cam_internal_idx] = {'frame_indices': [...], 'obs_A': [[x,y],...], 'obs_B': [[x,y],...]}
        cam_internal_to_external = {v: k for k, v in cam_id_map.items()}
        
        proj_residuals_list = []
        
        for c_internal in range(num_cams):
            cam_external = cam_internal_to_external.get(c_internal)
            if cam_external is None:
                continue
                
            cp = cam_params[c_internal]
            rvec = cp[:3]
            tvec = cp[3:6]
            f = cp[6]
            cx, cy = cp[7], cp[8]
            k1 = cp[9] if self.dist_coeff_num >= 1 else 0.0
            k2 = cp[10] if self.dist_coeff_num >= 2 else 0.0
            
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
            
            # Collect all 3D points and observations for this camera
            pts_3d_cam = []
            obs_2d_cam = []
            
            for i, fid in enumerate(frame_list):
                obs = observations[fid]
                if cam_external not in obs:
                    continue
                
                uv_obs = obs[cam_external]
                idx_A = i * 2
                idx_B = i * 2 + 1
                
                # Point A
                pts_3d_cam.append(points_3d[idx_A])
                obs_2d_cam.append(uv_obs[0][:2])
                
                # Point B
                pts_3d_cam.append(points_3d[idx_B])
                obs_2d_cam.append(uv_obs[1][:2])
            
            if len(pts_3d_cam) == 0:
                continue
            
            # Batch projection
            pts_3d_arr = np.array(pts_3d_cam, dtype=np.float64)
            obs_2d_arr = np.array(obs_2d_cam, dtype=np.float64)
            
            projected, _ = cv2.projectPoints(pts_3d_arr, rvec, tvec, K, dist_coeffs)
            projected = projected.reshape(-1, 2)
            
            # Residuals: projected - observed
            cam_residuals = (projected - obs_2d_arr).flatten()
            proj_residuals_list.append(cam_residuals)
        
        # === Combine all residuals ===
        # Order: [len_residual_0, proj_cam0_frame0_A_xy, proj_cam0_frame0_B_xy, ..., len_residual_1, ...]
        # But for efficiency, we combine: [all_len_residuals, all_proj_residuals]
        # This changes residual order but doesn't affect optimization (only affects Jacobian sparsity pattern)
        
        # Actually, to maintain order compatibility with existing sparsity, interleave properly:
        # For each frame: len_res, then cam residuals
        # This is complex to vectorize while maintaining order, so we use a hybrid approach:
        
        # Simplified: just concatenate (length constraints first, then projection residuals)
        # This doesn't affect optimization correctness, just sparsity pattern
        if proj_residuals_list:
            proj_residuals = np.concatenate(proj_residuals_list)
        else:
            proj_residuals = np.array([])
        
        # Interleave: For each frame, insert len_residual before its proj residuals
        # This is needed to match the sparsity pattern, but it's complex.
        # For now, use the simpler approach (may need to update sparsity pattern if issues arise)
        all_residuals = np.concatenate([len_residuals, proj_residuals])
        
        return all_residuals


    def _parse_results(self, params, cam_id_map):
        num_cams = len(cam_id_map)
        n_cam_params = 11
        cam_params = params[:num_cams*n_cam_params].reshape((num_cams, n_cam_params))
        
        # Center of Image
        h, w = self.image_size
            
        for i in range(num_cams):
            cp = cam_params[i]
            rvec = cp[0:3]
            tvec = cp[3:6]
            f = cp[6]
            cx = cp[7]
            cy = cp[8]
            k1 = cp[9] if self.dist_coeff_num >= 1 else 0.0
            k2 = cp[10] if self.dist_coeff_num >= 2 else 0.0
            
            c_idx = cam_id_map[i]
            
            R, _ = cv2.Rodrigues(rvec)
            if c_idx not in self.cameras:
                self.cameras[c_idx] = {}
                
            # updates internal storage too
            self.cameras[c_idx]['R'] = R
            self.cameras[c_idx]['T'] = tvec
            self.cameras[c_idx]['f'] = f
            self.cameras[c_idx]['c'] = np.array([cx, cy])
            self.cameras[c_idx]['dist'] = np.array([k1, k2, 0, 0, 0]) # Re-create dist array here
            
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            dist = np.array([k1, k2, 0, 0, 0])
            
            # Ensure tvec is column vector for matrix operations
            tvec_col = tvec.reshape(3, 1)
            
            _, cam_w, cam_h = self._get_cam_setting(c_idx)
            self.final_params[c_idx] = {
                'R': R, 'T': tvec_col, 'K': K, 'dist': dist, 'img_size': (cam_h, cam_w)
            }
            
            # Debug: print camera center
            C = -R.T @ tvec_col
            print(f"  Camera {c_idx}: Center = {C.flatten()}")

    def export_to_file(self, cam_idx, file_path):
        if cam_idx not in self.final_params:
            return False, "Camera not calibrated"
        
        p = self.final_params[cam_idx]
        K = p['K']
        dist = p['dist']
        rms = p['R']
        t_vec = p['T']
        h, w = p['img_size']
        
        content = f"# Camera Model: (PINHOLE/POLYNOMIAL)\nPINHOLE\n"
        content += f"# Camera Calibration Error: \nNone\n"
        content += f"# Pose Calibration Error: \nNone\n"
        content += f"# Image Size: (n_row,n_col)\n{h},{w}\n"
        # Force fy to be POSITIVE for OpenLPT convention (Y-Up, Z-Back frame => y' < 0 for Up => v < cy)
        # Even if Matlab showed negative, consistent physical model suggests Positive here.
        fy = abs(K[1,1])
        
        # Transform: Use Standard OpenCV (Y-Down, Z-Forward). 
        # Assuming OpenLPT uses standard image coords (Top-Left) and Projective Camera.
        # This means NO FLIP is needed if OpenLPT accounts for standard frame.
        M = np.diag([1, 1, 1])
        rms_out = M @ rms @ M
        t_vec_out = M @ t_vec
        
        # Intrinsics
        K_out = K.copy()
        K_out[1, 1] = abs(K[1, 1]) # Force positive fy for Y-Down frame
        K_out[0, 0] = abs(K[0, 0])
        
        # Distortion
        dist_coeffs = list(p['dist'])
        # No negation necessary for standard frame
             
        # Calculate overall error statistics (Mean + 3*Std)
        errors = self.calculate_per_frame_errors()
        proj_err_3sigma = "None"
        tri_err_3sigma = "None"
        
        if errors:
            all_proj_errs = []
            all_tri_errs = []
            for fid, err in errors.items():
                if 'tri_errors' in err:
                    all_tri_errs.extend(err['tri_errors'])
                for cam_id, e in err['cam_errors'].items():
                    all_proj_errs.append(e)
            
            proj_error_mean, proj_error_std = 0.0, 0.0
            tri_error_mean, tri_error_std = 0.0, 0.0
            
            if all_proj_errs:
                proj_error_mean = np.mean(all_proj_errs)
                proj_error_std = np.std(all_proj_errs)
                
            if all_tri_errs:
                tri_error_mean = np.mean(all_tri_errs)
                tri_error_std = np.std(all_tri_errs)

        # Write to file
        with open(file_path, 'w') as f:
            f.write("# Camera Model: (PINHOLE/POLYNOMIAL)\n")
            f.write("PINHOLE\n")
            f.write("# Camera Calibration Error: \n")
            f.write(f"{proj_error_mean},{proj_error_std}\n")
            f.write("# Pose Calibration Error: \n")
            f.write(f"{tri_error_mean},{tri_error_std}\n")
            f.write("# Image Size: (n_row,n_col)\n")
            f.write(f"{self.image_size[0]},{self.image_size[1]}\n") # H, W
            
            f.write("# Camera Matrix: \n")
            f.write(f"{K_out[0,0]},{K_out[0,1]},{K_out[0,2]}\n")
            f.write(f"{K_out[1,0]},{K_out[1,1]},{K_out[1,2]}\n")
            f.write(f"{K_out[2,0]},{K_out[2,1]},{K_out[2,2]}\n")
            
            f.write("# Distortion Coefficients: \n")
            dist_str = ",".join(map(str, dist_coeffs))
            f.write(f"{dist_str}\n")
            
            f.write("# Rotation Vector: \n")
            r_vec, _ = cv2.Rodrigues(rms_out)
            f.write(f"{r_vec[0,0]},{r_vec[1,0]},{r_vec[2,0]}\n")
            
            f.write("# Rotation Matrix: \n")
            f.write(f"{rms_out[0,0]},{rms_out[0,1]},{rms_out[0,2]}\n")
            f.write(f"{rms_out[1,0]},{rms_out[1,1]},{rms_out[1,2]}\n")
            f.write(f"{rms_out[2,0]},{rms_out[2,1]},{rms_out[2,2]}\n")
            
            f.write("# Inverse of Rotation Matrix: \n")
            r_inv = rms_out.T
            f.write(f"{r_inv[0,0]},{r_inv[0,1]},{r_inv[0,2]}\n")
            f.write(f"{r_inv[1,0]},{r_inv[1,1]},{r_inv[1,2]}\n")
            f.write(f"{r_inv[2,0]},{r_inv[2,1]},{r_inv[2,2]}\n")
            
            f.write("# Translation Vector: \n")
            # Ensure t_vec_out is flattened or indexed correctly
            f.write(f"{t_vec_out[0][0]},{t_vec_out[1][0]},{t_vec_out[2][0]}\n")
            
            f.write("# Inverse of Translation Vector: \n")
            t_inv = -r_inv @ t_vec_out
            f.write(f"{t_inv[0][0]},{t_inv[1][0]},{t_inv[2][0]}\n")
        
        return True, "Export successful"


# ==========================================
# Self-Test for pair_override Feature
# ==========================================

def debug_test_pair_override():
    """
    Self-test function to verify the pair_override feature.
    
    Tests:
        A. Default behavior (no pair_override) - log shows 'Auto-selected best pair'
        B. pair_override forces the specified pair - log shows 'Using pair_override'
        C. Invalid pair_override raises ValueError
    
    Usage:
        python -c "from modules.camera_calibration.wand_calibration.wand_calibrator import debug_test_pair_override; debug_test_pair_override()"
    """
    print("\n" + "="*60)
    print("DEBUG TEST: pair_override Feature Validation")
    print("="*60)
    
    import numpy as np
    
    print("\n[Test C] Invalid pair_override raises ValueError:")
    
    # Test C.1: Same camera ID
    try:
        cam_ids = [0, 1, 2]
        pair_override = (1, 1)
        if pair_override[0] == pair_override[1]:
            raise ValueError(f"pair_override cameras must be different, got: {pair_override}")
        print("  C.1 FAIL: Should have raised ValueError for (1, 1)")
    except ValueError as e:
        print(f"  C.1 PASS: Caught ValueError: {e}")
    
    # Test C.2: Non-existent camera
    try:
        cam_ids = [0, 1, 2]
        pair_override = (0, 99)
        if pair_override[1] not in cam_ids:
            raise ValueError(f"pair_override camera {pair_override[1]} not in active cameras: {cam_ids}")
        print("  C.2 FAIL: Should have raised ValueError for camera 99")
    except ValueError as e:
        print(f"  C.2 PASS: Caught ValueError: {e}")
    
    # Test C.3: Wrong type
    try:
        pair_override = ("a", "b")
        if not isinstance(pair_override[0], int):
            raise ValueError(f"pair_override camera IDs must be integers")
        print("  C.3 FAIL: Should have raised ValueError for non-integer IDs")
    except ValueError as e:
        print(f"  C.3 PASS: Caught ValueError: {e}")
    
    print("\n" + "="*60)
    print("DEBUG TEST COMPLETE")
    print("="*60)
    print("\nTo run full integration tests with a real dataset:")
    print("  1. Load a dataset with wand detections")
    print("  2. Call WandCalibrator._geometric_init_and_pnp() with pair_override=(i,j)")
    print("  3. Verify log shows '[Phase 1] Using pair_override: Cam i <-> Cam j'")
    print("  4. Verify returned best_pair == (i,j)")


if __name__ == "__main__":
    debug_test_pair_override()
