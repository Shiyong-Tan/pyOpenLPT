"""
Image Preprocessing View
Main view for the Image Preprocessing module with Camera Calibration style layout.
"""

import numpy as np
import cv2
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, as_completed

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QGroupBox, QGridLayout, QSlider, QSpinBox,
    QCheckBox, QComboBox, QFileDialog, QScrollArea, QApplication,
    QLineEdit, QTableWidgetItem, QMessageBox, QDialog, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QObject, QThread, Slot
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor

from .widgets import RangeSlider, ProcessingDialog

try:
    from pycine.raw import read_frames as pycine_read_frames
    pycine = True
except Exception as e:
    print(f"Warning: Failed to import pycine: {e}")
    pycine = None


def imadjust_opencv(img, low_in, high_in, low_out=0, high_out=255, gamma=1.0):
    """
    img: uint8 or float image
    low_in, high_in, low_out, high_out: same scale as img
    gamma: gamma correction
    """
    # Ensure float for calculation
    img = img.astype(np.float32)

    # normalize to [0,1]
    # Handle division by zero
    diff = high_in - low_in
    if diff < 1e-5:
        diff = 1e-5
        
    img = (img - low_in) / diff
    img = np.clip(img, 0, 1)

    # gamma
    if gamma != 1.0:
        img = img ** gamma

    # scale to output range
    img = img * (high_out - low_out) + low_out
    img = np.clip(img, low_out, high_out)

    return img.astype(np.uint8)


def _apply_processing_pipeline_with_settings(img_data, bg_data, cam_idx, settings):
    """Pure processing pipeline for worker thread use."""
    # 0. Ensure grayscale and float32
    if len(img_data.shape) == 3:
        gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img_data.astype(np.float32)

    # 1. Background Subtraction (float32)
    if settings["bg_enabled"] and bg_data is not None:
        if settings["invert"]:
            result = bg_data - gray
        else:
            result = gray - bg_data
        result = np.clip(result, 0, None)
    else:
        result = gray

    # 2. Bit shift to 8-bit
    shift = settings["cine_shifts"].get(cam_idx, 0)
    if shift > 0:
        result = (result / (2 ** shift))
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 3. Invert (only if not already handled by BG subtraction)
    if settings["invert"] and not (settings["bg_enabled"] and bg_data is not None):
        result = 255 - result

    # 4. Range adjustment
    result = imadjust_opencv(result, settings["low_in"], settings["high_in"])

    # 5. Denoise
    if settings["denoise"]:
        a = result.astype(np.float32)
        kernel = np.ones((3, 3), np.uint8)
        b = cv2.erode(a, kernel, iterations=1)
        c = a - b
        b = cv2.erode(a, kernel, iterations=1)
        c = c - b

        d = cv2.GaussianBlur(c, (0, 0), 0.5)
        e = cv2.blur(d, (100, 100))
        f = a - e

        blurred_f = cv2.GaussianBlur(f, (0, 0), 1.0)
        sharp = f + 0.8 * (f - blurred_f)
        result = np.clip(sharp, 0, 255).astype(np.uint8)

    return result.astype(np.uint8)


class PreprocessWorker(QObject):
    """Background worker for batch preprocessing."""

    progress = Signal(int, int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, tasks, settings, process_workers, write_workers, max_pending_writes=64):
        super().__init__()
        self.tasks = tasks
        self.settings = settings
        self.process_workers = max(1, int(process_workers))
        self.write_workers = max(1, int(write_workers))
        self.max_pending_writes = max(8, int(max_pending_writes))

        self._stop = False
        self._paused = False
        self._pause_cv = threading.Condition()

    @Slot()
    def run(self):
        start_t = time.time()
        total = len(self.tasks)
        done = 0
        failed = 0
        write_failed = 0

        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        # Split task types once
        cine_tasks = [t for t in self.tasks if t.get("is_cine", False)]
        file_tasks = [t for t in self.tasks if not t.get("is_cine", False)]

        def check_pause_stop():
            if self._stop:
                return True
            with self._pause_cv:
                while self._paused and not self._stop:
                    self._pause_cv.wait(timeout=0.2)
            return self._stop

        def process_image_data(img, cam_idx):
            if img is None:
                return None
            bg = self.settings["camera_backgrounds"].get(cam_idx)
            return _apply_processing_pipeline_with_settings(img, bg, cam_idx, self.settings)

        def process_file_task(task):
            src = task["src"]
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is None:
                return task, None, "read_failed"
            out = process_image_data(img, task["cam_idx"])
            if out is None:
                return task, None, "process_failed"
            return task, out, "ok"

        def write_task(dst, img):
            ok = cv2.imwrite(dst, img)
            return bool(ok)

        def flush_one_write(pending_writes):
            nonlocal done, write_failed
            if not pending_writes:
                return
            completed, _ = wait(pending_writes, return_when=FIRST_COMPLETED)
            for wf in completed:
                pending_writes.remove(wf)
                try:
                    okw = bool(wf.result())
                    if not okw:
                        write_failed += 1
                except Exception:
                    write_failed += 1
                done += 1
                if done % 10 == 0 or done == total:
                    self.progress.emit(done, total, f"Processed {done}/{total}")

        with ThreadPoolExecutor(max_workers=self.process_workers) as proc_pool, \
             ThreadPoolExecutor(max_workers=self.write_workers) as write_pool:
            pending_writes = set()

            # 1) Standard image files: bounded process futures
            pending_proc = set()
            max_pending_proc = max(self.process_workers * 2, 8)

            def submit_processed_image(task, img_out):
                nonlocal failed
                while len(pending_writes) >= self.max_pending_writes and not self._stop:
                    flush_one_write(pending_writes)
                if self._stop:
                    return
                pending_writes.add(write_pool.submit(write_task, task["dst"], img_out))

            for task in file_tasks:
                if check_pause_stop():
                    break
                pending_proc.add(proc_pool.submit(process_file_task, task))
                if len(pending_proc) >= max_pending_proc:
                    completed, _ = wait(pending_proc, return_when=FIRST_COMPLETED)
                    for pf in completed:
                        pending_proc.remove(pf)
                        try:
                            task_ret, img_out, status = pf.result()
                        except Exception:
                            failed += 1
                            done += 1
                            continue
                        if status != "ok" or img_out is None:
                            failed += 1
                            done += 1
                            if done % 10 == 0 or done == total:
                                self.progress.emit(done, total, f"Processed {done}/{total}")
                            continue
                        submit_processed_image(task_ret, img_out)

            while pending_proc and not self._stop:
                completed, _ = wait(pending_proc, return_when=FIRST_COMPLETED)
                for pf in completed:
                    pending_proc.remove(pf)
                    try:
                        task_ret, img_out, status = pf.result()
                    except Exception:
                        failed += 1
                        done += 1
                        continue
                    if status != "ok" or img_out is None:
                        failed += 1
                        done += 1
                        if done % 10 == 0 or done == total:
                            self.progress.emit(done, total, f"Processed {done}/{total}")
                        continue
                    submit_processed_image(task_ret, img_out)

            # 2) Cine files: chunked read then parallel processing
            if not self._stop and cine_tasks:
                from collections import defaultdict
                by_file = defaultdict(list)
                for t in cine_tasks:
                    by_file[t["cine_file"]].append(t)

                chunk_size = 256
                for cine_file, lst in by_file.items():
                    if check_pause_stop():
                        break
                    lst.sort(key=lambda x: x["cine_frame"])
                    frames = [t["cine_frame"] for t in lst]
                    want = {t["cine_frame"]: t for t in lst}

                    p = 0
                    while p < len(frames):
                        if check_pause_stop():
                            break
                        start_fr = frames[p]
                        q = p
                        while q + 1 < len(frames) and (frames[q + 1] - start_fr) < chunk_size:
                            q += 1
                        end_fr = frames[q]
                        cnt = end_fr - start_fr + 1

                        try:
                            if not pycine:
                                raise RuntimeError("pycine not available")
                            raw_images, _, _ = pycine_read_frames(cine_file, start_frame=start_fr, count=cnt)
                            imgs = list(raw_images)
                        except Exception:
                            # fallback frame-by-frame for this chunk
                            imgs = None

                        if imgs is not None:
                            future_to_task = {}
                            for fr in frames[p:q + 1]:
                                if fr < start_fr or fr > end_fr:
                                    continue
                                task = want[fr]
                                raw = np.array(imgs[fr - start_fr])
                                fut = proc_pool.submit(process_image_data, raw, task["cam_idx"])
                                future_to_task[fut] = task

                            while future_to_task and not self._stop:
                                completed, _ = wait(set(future_to_task.keys()), return_when=FIRST_COMPLETED)
                                for pf in completed:
                                    task = future_to_task.pop(pf)
                                    try:
                                        out = pf.result()
                                    except Exception:
                                        out = None
                                    if out is None:
                                        failed += 1
                                        done += 1
                                        if done % 10 == 0 or done == total:
                                            self.progress.emit(done, total, f"Processed {done}/{total}")
                                        continue
                                    while len(pending_writes) >= self.max_pending_writes and not self._stop:
                                        flush_one_write(pending_writes)
                                    if self._stop:
                                        break
                                    pending_writes.add(write_pool.submit(write_task, task["dst"], out))
                        else:
                            for fr in frames[p:q + 1]:
                                task = want[fr]
                                try:
                                    raw_images, _, _ = pycine_read_frames(cine_file, start_frame=fr, count=1)
                                    one = list(raw_images)
                                    if not one:
                                        failed += 1
                                        done += 1
                                        continue
                                    raw = np.array(one[0])
                                    out = process_image_data(raw, task["cam_idx"])
                                    if out is None:
                                        failed += 1
                                        done += 1
                                        continue
                                    while len(pending_writes) >= self.max_pending_writes and not self._stop:
                                        flush_one_write(pending_writes)
                                    if self._stop:
                                        break
                                    pending_writes.add(write_pool.submit(write_task, task["dst"], out))
                                except Exception:
                                    failed += 1
                                    done += 1

                        p = q + 1

            while pending_writes:
                flush_one_write(pending_writes)

        elapsed = max(time.time() - start_t, 1e-9)
        fps = done / elapsed
        self.finished.emit({
            "total": total,
            "processed": done,
            "failed": failed + write_failed,
            "stopped": bool(self._stop),
            "elapsed_sec": elapsed,
            "fps": fps,
        })

    @Slot(bool)
    def set_paused(self, paused):
        with self._pause_cv:
            self._paused = bool(paused)
            if not self._paused:
                self._pause_cv.notify_all()

    @Slot()
    def request_stop(self):
        self._stop = True
        with self._pause_cv:
            self._paused = False
            self._pause_cv.notify_all()


class BackgroundProgressDialog(QDialog):
    """Compact progress dialog for background calculation."""

    def __init__(self, parent=None, total=100):
        super().__init__(parent)
        self.setWindowTitle("Background Calculation")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setFixedSize(430, 108)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 8)
        layout.setSpacing(8)

        self.status_label = QLabel("Calculating backgrounds...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: bold; margin: 0px; padding: 0px;")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max(1, int(total)))
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(22)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #444; border-radius: 4px; text-align: center; color: white; }"
            "QProgressBar::chunk { background-color: #00bcd4; border-radius: 4px; }"
        )
        layout.addWidget(self.progress_bar)

    def update_progress(self, current, total, text):
        self.progress_bar.setRange(0, max(1, int(total)))
        self.progress_bar.setValue(max(0, min(int(current), int(total))))
        self.status_label.setText(str(text))


class BackgroundWorker(QObject):
    """Background computation worker (runs in QThread)."""

    progress = Signal(int, int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, cam_jobs, cam_budget, invert_enabled, max_workers=4):
        super().__init__()
        self.cam_jobs = cam_jobs
        self.cam_budget = cam_budget
        self.invert_enabled = bool(invert_enabled)
        self.max_workers = max(1, int(max_workers))

    @staticmethod
    def _to_gray_float32(img):
        if img is None:
            return None
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32)

    def _compute_one_camera(self, cam_idx, images, selected_indices, progress_step_cb=None):
        from collections import defaultdict

        accumulator = None
        count = 0

        num_samples = min(10, len(selected_indices))
        sample_step = max(1, len(selected_indices) // max(1, num_samples))
        sample_frame_indices = set(range(0, len(selected_indices), sample_step)[:num_samples])
        cached_frames = []

        cine_items = []
        non_cine_items = []
        for i, idx in enumerate(selected_indices):
            path = images[idx]
            if "#" in path:
                file_path, frame_idx = path.split("#")
                cine_items.append((file_path, int(frame_idx), i, idx))
            else:
                non_cine_items.append((i, idx, path))

        groups = defaultdict(list)
        for file_path, frame_idx, i, idx in cine_items:
            groups[file_path].append((frame_idx, i, idx))

        chunk_size = 100
        for file_path, lst in groups.items():
            try:
                from pycine.file import read_header
                header = read_header(file_path)
                cfh = header['cinefileheader']
                first_no = cfh.FirstImageNo
                img_count = cfh.ImageCount
                last_no = first_no + img_count - 1
            except Exception as e:
                print(f"DEBUG: Could not read metadata for {file_path}: {e}")
                first_no, last_no = 0, 999999

            lst.sort(key=lambda x: x[0])
            frame_list = [x[0] for x in lst]
            want = {frame_idx: (i, idx) for frame_idx, i, idx in lst}

            out_of_range = [f for f in frame_list if f < first_no or f > last_no]
            if out_of_range:
                print(f"WARNING: Cam {cam_idx} has {len(out_of_range)} out-of-range frames. First 5: {out_of_range[:5]}")

            p = 0
            while p < len(frame_list):
                start = frame_list[p]
                q = p
                while q + 1 < len(frame_list) and (frame_list[q + 1] - start) < chunk_size:
                    q += 1
                end = frame_list[q]

                try:
                    if not pycine:
                        break
                    chunk_count = end - start + 1
                    raw_images, _, _ = pycine_read_frames(file_path, start_frame=start, count=chunk_count)
                    imgs = list(raw_images)

                    if imgs:
                        for k, fr in enumerate(range(start, end + 1)):
                            if fr not in want:
                                continue
                            raw = np.array(imgs[k])
                            img = self._to_gray_float32(raw)
                            if img is not None:
                                if accumulator is None:
                                    accumulator = img.astype(np.float64)
                                else:
                                    accumulator += img.astype(np.float64)
                                count += 1

                                i_orig, _ = want[fr]
                                if i_orig in sample_frame_indices:
                                    cached_frames.append(img.copy())
                            if progress_step_cb is not None:
                                progress_step_cb(cam_idx)

                except Exception as e:
                    print(f"Error reading chunk {start}-{end} for cam {cam_idx}: {e}")
                    for fr, i_orig, _ in lst[p:q + 1]:
                        try:
                            raw_images, _, _ = pycine_read_frames(file_path, start_frame=fr, count=1)
                            one = list(raw_images)
                            if one:
                                raw = np.array(one[0])
                                img = self._to_gray_float32(raw)
                                if img is not None:
                                    if accumulator is None:
                                        accumulator = img.astype(np.float64)
                                    else:
                                        accumulator += img.astype(np.float64)
                                    count += 1
                                    if i_orig in sample_frame_indices:
                                        cached_frames.append(img.copy())
                        except Exception as e2:
                            print(f"Error reading cine frame {fr} for cam {cam_idx}: {e2}")
                        finally:
                            if progress_step_cb is not None:
                                progress_step_cb(cam_idx)

                p = q + 1

        for i, _, path in non_cine_items:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                img_gray = self._to_gray_float32(img)
                if img_gray is not None:
                    if accumulator is None:
                        accumulator = img_gray.astype(np.float64)
                    else:
                        accumulator += img_gray.astype(np.float64)
                    count += 1

                    if i in sample_frame_indices:
                        cached_frames.append(img_gray.copy())
            if progress_step_cb is not None:
                progress_step_cb(cam_idx)

        if count <= 0 or accumulator is None:
            return {'cam_idx': cam_idx, 'bg': None, 'shift': 0, 'count': 0}

        bg = (accumulator / count).astype(np.float32)

        subtracted_stats = []
        for raw_img in cached_frames:
            if self.invert_enabled:
                subtracted = np.clip(bg - raw_img, 0, None)
            else:
                subtracted = np.clip(raw_img - bg, 0, None)
            subset = subtracted[::8, ::8].ravel()
            p_val = np.percentile(subset, 99.5)
            subtracted_stats.append(p_val)

        if subtracted_stats:
            final_p_val = np.mean(subtracted_stats)
            if final_p_val <= 1:
                n = 8
            else:
                n = np.ceil(np.log2(final_p_val))
            shift = max(0, int(n - 8))
        else:
            shift = 0

        if progress_step_cb is not None:
            progress_step_cb(cam_idx)
        return {'cam_idx': cam_idx, 'bg': bg, 'shift': shift, 'count': count}

    @Slot()
    def run(self):
        try:
            total_budget = sum(self.cam_budget.get(cam_idx, 0) for cam_idx, _, _ in self.cam_jobs)
            backgrounds = {}
            shifts = {}
            progress_lock = threading.Lock()
            done_budget = 0
            next_emit_at = 20

            def report_step(_cam_idx):
                nonlocal done_budget, next_emit_at
                emit_payload = None
                with progress_lock:
                    done_budget += 1
                    if done_budget >= total_budget:
                        emit_payload = (total_budget, max(1, total_budget), f"Reading {total_budget}/{max(1, total_budget)}")
                    elif done_budget >= next_emit_at:
                        while next_emit_at <= done_budget:
                            next_emit_at += 20
                        emit_payload = (done_budget, max(1, total_budget), f"Reading {done_budget}/{max(1, total_budget)}")
                if emit_payload is not None:
                    self.progress.emit(*emit_payload)

            self.progress.emit(0, max(1, total_budget), f"Reading 0/{max(1, total_budget)}")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {
                    executor.submit(self._compute_one_camera, cam_idx, images, selected_indices, report_step): cam_idx
                    for cam_idx, images, selected_indices in self.cam_jobs
                }

                for future in as_completed(future_map):
                    cam_idx = future_map[future]
                    msg = f"Reading {min(done_budget, total_budget)}/{max(1, total_budget)}"
                    try:
                        result = future.result()
                        if result and result.get('bg') is not None:
                            backgrounds[cam_idx] = result['bg']
                            shifts[cam_idx] = int(result.get('shift', 0))
                        else:
                            pass
                    except Exception as e:
                        print(f"Background calculation failed for cam {cam_idx}: {e}")
                    with progress_lock:
                        cur = min(done_budget, total_budget)
                    self.progress.emit(cur, max(1, total_budget), f"Reading {cur}/{max(1, total_budget)}")

            self.progress.emit(max(1, total_budget), max(1, total_budget), f"Reading {max(1, total_budget)}/{max(1, total_budget)}")

            self.finished.emit({
                "backgrounds": backgrounds,
                "shifts": shifts,
                "total_cams": len(self.cam_jobs),
            })
        except Exception as e:
            self.error.emit(str(e))


class ZoomableImageLabel(QLabel):
    """
    Label with zoom and pan functionality for image preview.
    Simplified version for preprocessing.
    """
    
    pixelClicked = Signal(int, int, int) # x, y, intensity

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        
        # Image Data
        self._pixmap = None
        self._cv_image = None  # Store original cv2 image for processing
        
        # View State
        self._user_zoom = 1.0
        self._user_pan_x = 0.0
        self._user_pan_y = 0.0
        self.last_mouse_pos = None
        self.is_panning = False
        
    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.resetView()
        self.update()
        
    def setCvImage(self, cv_image):
        """Set image from cv2/numpy array."""
        self._cv_image = cv_image
        if cv_image is not None:
            # Convert to QPixmap
            if len(cv_image.shape) == 2:
                # Grayscale
                h, w = cv_image.shape
                bytes_per_line = w
                # Ensure data is contiguous
                if not cv_image.flags['C_CONTIGUOUS']:
                    cv_image = np.ascontiguousarray(cv_image)
                qimg = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            else:
                # Color (BGR to RGB)
                h, w, ch = cv_image.shape
                rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                bytes_per_line = ch * w
                if not rgb.flags['C_CONTIGUOUS']:
                    rgb = np.ascontiguousarray(rgb)
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimg)
        else:
            self._pixmap = None
        self.update()
        
    def getCvImage(self):
        """Get the current cv2 image."""
        return self._cv_image
        
    def resetView(self):
        """Reset zoom and pan."""
        self._user_zoom = 1.0
        self._user_pan_x = 0.0
        self._user_pan_y = 0.0
        self.update()
        
    def _calc_transform_params(self):
        """Calculate display transform parameters."""
        if not self._pixmap or self._pixmap.isNull():
            return 1.0, 0, 0
        
        p_w = self._pixmap.width()
        p_h = self._pixmap.height()
        w_w = self.width()
        w_h = self.height()
        
        if p_w <= 0 or p_h <= 0 or w_w <= 0 or w_h <= 0:
            return 1.0, 0, 0
        
        base_scale = min(w_w / p_w, w_h / p_h)
        scale = base_scale * self._user_zoom
        
        t_w = int(p_w * scale)
        t_h = int(p_h * scale)
        
        base_x = (w_w - t_w) / 2
        base_y = (w_h - t_h) / 2
        
        t_x = int(base_x + self._user_pan_x)
        t_y = int(base_y + self._user_pan_y)
        
        return scale, t_x, t_y

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if not self._pixmap or self._pixmap.isNull():
            return
            
        mouse_pos = event.position().toPoint()
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else (1.0 / 1.15)
        
        new_zoom = self._user_zoom * zoom_factor
        new_zoom = max(0.1, min(20.0, new_zoom))
        
        # Zoom towards cursor
        old_scale, old_tx, old_ty = self._calc_transform_params()
        img_x = (mouse_pos.x() - old_tx) / old_scale if old_scale > 0 else 0
        img_y = (mouse_pos.y() - old_ty) / old_scale if old_scale > 0 else 0
        
        self._user_zoom = new_zoom
        
        new_scale, new_tx, new_ty = self._calc_transform_params()
        new_widget_x = img_x * new_scale + new_tx
        new_widget_y = img_y * new_scale + new_ty
        
        self._user_pan_x += mouse_pos.x() - new_widget_x
        self._user_pan_y += mouse_pos.y() - new_widget_y
        
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = True
            self.last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            # Handle pixel click for intensity inspection
            if self._cv_image is not None:
                scale, tx, ty = self._calc_transform_params()
                if scale > 0:
                    pos = event.position().toPoint()
                    # Map to image coordinates
                    img_x = int((pos.x() - tx) / scale)
                    img_y = int((pos.y() - ty) / scale)
                    
                    h, w = self._cv_image.shape if len(self._cv_image.shape) == 2 else self._cv_image.shape[:2]
                    
                    if 0 <= img_x < w and 0 <= img_y < h:
                        # Get intensity
                        if len(self._cv_image.shape) == 2:
                            val = self._cv_image[img_y, img_x]
                        else:
                            # Convert to simplified intensity (grayscale equivalent) if color
                            val = int(np.mean(self._cv_image[img_y, img_x]))
                            
                        self.pixelClicked.emit(img_x, img_y, int(val))

    def mouseMoveEvent(self, event):
        if self.is_panning:
            current_pos = event.position().toPoint()
            delta = current_pos - self.last_mouse_pos
            self._user_pan_x += delta.x()
            self._user_pan_y += delta.y()
            self.last_mouse_pos = current_pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(13, 17, 23))  # Dark background
        
        if self._pixmap and not self._pixmap.isNull():
            scale, t_x, t_y = self._calc_transform_params()
            
            p_w = self._pixmap.width()
            p_h = self._pixmap.height()
            t_w = int(p_w * scale)
            t_h = int(p_h * scale)
            
            from PySide6.QtCore import QRect
            # Use nearest-neighbor interpolation for pixel-sharp display when zoomed
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawPixmap(QRect(t_x, t_y, t_w, t_h), self._pixmap)
        else:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load an image to preview")


class ImagePreprocessingView(QWidget):
    """View for image preprocessing functionality with Camera Calibration style layout."""
    
    def __init__(self):
        super().__init__()
        
        # Data
        self.root_path = ""  # Main directory path
        self.camera_folders = []  # List of camera folder paths
        self.camera_images = {}  # {cam_idx: [image_paths]}
        self.camera_backgrounds = {}  # {cam_idx: background_image}
        self.cine_shifts = {}   # {abs_path: shift_bits}
        self.current_cam = 0
        self.current_frame = 0
        self.original_image = None
        self.processed_image = None
        self.current_view_mode = "original"  # original, processed, background
        self._stop_requested = False
        self._is_processing = False
        self._is_paused = False
        self.processing_dialog = None
        self.preprocess_thread = None
        self.preprocess_worker = None
        self.bg_calc_dialog = None
        self.bg_calc_thread = None
        self.bg_calc_worker = None
        self._bg_cache_path = ""
        self._bg_cache_stride = 0
        self._bg_cache_avg_count = 0
        self._bg_cache_invert_enabled = False
        self._batch_cam_output_paths = {}
        self._batch_img_file_dir = ""
        self._busy_tokens = {}
        
        self._setup_ui()

    def _busy_begin(self, key, task_name):
        if key in self._busy_tokens:
            return
        wnd = self.window()
        if wnd is not None and hasattr(wnd, 'begin_busy'):
            self._busy_tokens[key] = wnd.begin_busy(task_name)

    def _busy_end(self, key):
        token = self._busy_tokens.pop(key, None)
        if token is None:
            return
        wnd = self.window()
        if wnd is not None and hasattr(wnd, 'end_busy'):
            wnd.end_busy(token)
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)
        
        # === Title ===
        title = QLabel("Image Preprocessing")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4ff; margin-bottom: 10px;")
        main_layout.addWidget(title)
        
        # === Main Content (Left: View, Right: Settings) ===
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        
        # === Left: Image Preview ===
        preview_frame = QFrame()
        preview_frame.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        
        # Camera tabs (left-aligned) - FIRST
        self.cam_tabs_layout = QHBoxLayout()
        self.cam_tabs_layout.setSpacing(0)
        self.cam_tabs_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.cam_buttons = []
        # Create initial 4 camera tabs (will be updated when images loaded)
        self._create_camera_tabs(4)
        preview_layout.addLayout(self.cam_tabs_layout)
        
        # Original/Processed/Background toggle - SECOND (below camera tabs)
        toggle_layout = QHBoxLayout()
        self.original_btn = QPushButton("Original")
        self.original_btn.setCheckable(True)
        self.original_btn.setChecked(True)
        self.original_btn.setStyleSheet("""
            QPushButton { background-color: #00d4ff; color: black; border-radius: 4px; padding: 6px 16px; font-weight: bold; }
            QPushButton:checked { background-color: #00d4ff; }
            QPushButton:!checked { background-color: #333; color: #888; }
        """)
        self.original_btn.clicked.connect(lambda: self._toggle_view("original"))
        
        self.processed_btn = QPushButton("Processed")
        self.processed_btn.setCheckable(True)
        self.processed_btn.setStyleSheet("""
            QPushButton { background-color: #333; color: #888; border-radius: 4px; padding: 6px 16px; font-weight: bold; }
            QPushButton:checked { background-color: #00d4ff; color: black; }
            QPushButton:!checked { background-color: #333; color: #888; }
        """)
        self.processed_btn.clicked.connect(lambda: self._toggle_view("processed"))
        
        self.background_btn = QPushButton("Background")
        self.background_btn.setCheckable(True)
        self.background_btn.setStyleSheet("""
            QPushButton { background-color: #333; color: #888; border-radius: 4px; padding: 6px 16px; font-weight: bold; }
            QPushButton:checked { background-color: #00d4ff; color: black; }
            QPushButton:!checked { background-color: #333; color: #888; }
        """)
        self.background_btn.clicked.connect(lambda: self._toggle_view("background"))
        
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.original_btn)
        toggle_layout.addWidget(self.processed_btn)
        toggle_layout.addWidget(self.background_btn)
        toggle_layout.addStretch()
        preview_layout.addLayout(toggle_layout)
        
        # Image display area (Zoomable)
        self.image_label = ZoomableImageLabel("Load an image to preview")
        self.image_label.setMinimumHeight(500)
        self.image_label.pixelClicked.connect(self._on_pixel_clicked)
        preview_layout.addWidget(self.image_label, stretch=1)
        
        content_layout.addWidget(preview_frame, stretch=2)
        
        # === Right: Settings Panel ===
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_scroll.setMinimumWidth(280)
        settings_scroll.setMaximumWidth(400)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        settings_scroll.setStyleSheet("""
            QScrollArea { background-color: transparent; }
            QScrollBar:vertical {
                background: #1a1a2e;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #444;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #555;
            }
        """)
        
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setSpacing(12)
        settings_layout.setContentsMargins(0, 0, 10, 0)
        
        # Group box style
        group_style = """
            QGroupBox { 
                background-color: #000; 
                border: 1px solid #444; 
                font-weight: bold; 
                color: #00d4ff; 
                border-radius: 6px; 
                margin-top: 15px;
                padding-top: 15px;
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px; 
            }
        """
        
        # === Image Source ===
        source_group = QGroupBox("Image Source")
        source_group.setStyleSheet(group_style)
        source_layout = QVBoxLayout(source_group)
        
        # Num Cameras row
        from PySide6.QtWidgets import QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView
        cam_row = QHBoxLayout()
        cam_label = QLabel("Num Cameras:")
        cam_label.setStyleSheet("color: white;")
        cam_row.addWidget(cam_label)
        self.num_cameras_spin = QSpinBox()
        self.num_cameras_spin.setRange(1, 16)
        self.num_cameras_spin.setValue(4)
        self.num_cameras_spin.setStyleSheet("""
            QSpinBox { 
                background-color: #1a1a2e; 
                color: white; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 5px;
            }
        """)
        cam_row.addWidget(self.num_cameras_spin)
        source_layout.addLayout(cam_row)
        
        import qtawesome as qta
        
        # === Project Path (For Export) ===
        project_label = QLabel("Project Path (for Output):")
        project_label.setStyleSheet("color: white;")
        source_layout.addWidget(project_label)
        
        proj_row = QHBoxLayout()
        self.project_path_input = QLineEdit()
        self.project_path_input.setPlaceholderText("Select project output folder...")
        self.project_path_input.setStyleSheet("background-color: #1a1a2e; color: white; border: 1px solid #444; padding: 5px;")
        proj_row.addWidget(self.project_path_input)
        
        proj_browse_btn = QPushButton("")
        proj_browse_btn.setFixedWidth(40)
        proj_browse_btn.setIcon(qta.icon("fa5s.folder-open", color="white"))
        proj_browse_btn.setStyleSheet("background-color: #333; color: white; border: 1px solid #444;")
        proj_browse_btn.clicked.connect(self._browse_project_path)
        proj_row.addWidget(proj_browse_btn)
        source_layout.addLayout(proj_row)
        
        # Load Images Button
        browse_btn = QPushButton(" Load Images from Folder")
        browse_btn.setIcon(qta.icon("fa5s.images", color="black"))
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff; 
                color: black; 
                border: 1px solid #00a0cc; 
                border-radius: 4px; 
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #66e5ff; }
        """)
        browse_btn.clicked.connect(self._browse_images)
        source_layout.addWidget(browse_btn)
        
        self.image_count_label = QLabel("0 images loaded")
        self.image_count_label.setStyleSheet("color: #a0a0a0;")
        source_layout.addWidget(self.image_count_label)
        
        # Invert checkbox (applied to all image operations)
        self.invert_check = QCheckBox("Invert")
        self.invert_check.setStyleSheet("color: white;")
        self.invert_check.stateChanged.connect(self._on_settings_changed)
        source_layout.addWidget(self.invert_check)
        
        # Frame List
        frame_list_label = QLabel("Frame List (Click to Preview):")
        frame_list_label.setStyleSheet("color: white;")
        source_layout.addWidget(frame_list_label)
        
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(2)
        self.frame_table.setHorizontalHeaderLabels(["Index", "Filename"])
        self.frame_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.frame_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.frame_table.setStyleSheet("background-color: #0d1117; border: 1px solid #333; color: white;")
        self.frame_table.setFixedHeight(120)
        self.frame_table.currentCellChanged.connect(lambda r, c, pr, pc: self._on_frame_clicked(r, c))
        source_layout.addWidget(self.frame_table)
        
        settings_layout.addWidget(source_group)
        
        # === Background Subtraction ===
        bg_group = QGroupBox("Background Subtraction")
        bg_group.setStyleSheet(group_style)
        bg_layout = QGridLayout(bg_group)
        bg_layout.setVerticalSpacing(10)
        
        self.bg_enabled = QCheckBox("Enable")
        self.bg_enabled.setStyleSheet("color: white;")
        self.bg_enabled.stateChanged.connect(self._on_settings_changed)
        bg_layout.addWidget(self.bg_enabled, 0, 0)
        
        # Calculate Background button
        self.calc_bg_btn = QPushButton("Calculate")
        self.calc_bg_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff; 
                color: black; 
                border-radius: 4px; 
                padding: 5px 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #66e5ff; }
        """)
        self.calc_bg_btn.clicked.connect(self._calculate_all_backgrounds)
        bg_layout.addWidget(self.calc_bg_btn, 0, 1)
        
        # Skip Frames
        skip_label = QLabel("Skip Frames:")
        skip_label.setStyleSheet("color: white;")
        bg_layout.addWidget(skip_label, 1, 0)
        self.skip_frames_spin = QSpinBox()
        self.skip_frames_spin.setRange(0, 100)
        self.skip_frames_spin.setValue(5)
        self.skip_frames_spin.setStyleSheet("""
            QSpinBox { 
                background-color: #1a1a2e; 
                color: white; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 5px;
            }
        """)
        bg_layout.addWidget(self.skip_frames_spin, 1, 1)
        
        # Avg Count
        avg_label = QLabel("Avg Count:")
        avg_label.setStyleSheet("color: white;")
        bg_layout.addWidget(avg_label, 2, 0)
        self.avg_count_spin = QSpinBox()
        self.avg_count_spin.setRange(1, 999999)
        self.avg_count_spin.setValue(1000)
        self.avg_count_spin.setStyleSheet("""
            QSpinBox { 
                background-color: #1a1a2e; 
                color: white; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 5px;
            }
        """)
        bg_layout.addWidget(self.avg_count_spin, 2, 1)
        
        settings_layout.addWidget(bg_group)
        
        # === Image Source Info & Pixel Inspector ===
        # Place pixel info here
        self.pixel_info_label = QLabel("Click image to inspect pixel")
        self.pixel_info_label.setStyleSheet("color: #00d4ff; font-size: 11px;")
        self.pixel_info_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Add to top layout or somewhere visible. 
        # Putting it in main layout top bar or overlay might be complex.
        # Let's put it at the bottom of the settings scroll for now, or inside a group.
        # Actually, let's put it in the "Image Source" group for high visibility
        pass # Just initialization logic, placement happens in layout construction
        
        # ... Re-arrange layouts slightly to fit it ...
        # Let's add it to main settings layout for now
        settings_layout.addWidget(self.pixel_info_label)
        
        # === Image Adjustment ===
        adjust_group = QGroupBox("Intensity Adjustment")
        adjust_group.setStyleSheet(group_style)
        adjust_layout = QGridLayout(adjust_group)
        adjust_layout.setVerticalSpacing(12)  # 10% more spacing
        adjust_layout.setHorizontalSpacing(10)
        
        # Intensity Range Slider (Dual Handle + SpinBoxes)
        range_label = QLabel("Input Range:")
        range_label.setStyleSheet("color: white;")
        adjust_layout.addWidget(range_label, 0, 0, 1, 3)
        
        self.range_slider = RangeSlider(initial_min=0, initial_max=255)
        self.range_slider.rangeChanged.connect(self._on_settings_changed)
        adjust_layout.addWidget(self.range_slider, 1, 0, 1, 3)
        
        # Denoise (LaVision Processing)
        self.denoise_check = QCheckBox("Enhanced Denoise")
        self.denoise_check.setStyleSheet("color: white; font-weight: bold;")
        self.denoise_check.stateChanged.connect(self._on_settings_changed)
        adjust_layout.addWidget(self.denoise_check, 2, 0, 1, 3)
        
        settings_layout.addWidget(adjust_group)
        
        # === Buttons ===
        settings_layout.addStretch()
        
        preview_btn = QPushButton("Preview")
        preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff; 
                color: black; 
                border-radius: 4px; 
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #66e5ff; }
        """)
        settings_layout.addWidget(preview_btn)
        
        # Batch Process Range
        range_group = QGroupBox("Batch Process Range")
        range_group.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 10px; padding-top: 10px; color: #ddd; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        range_layout = QHBoxLayout(range_group)
        
        range_layout.addWidget(QLabel("Start:"))
        self.batch_start_spin = QSpinBox()
        self.batch_start_spin.setRange(0, 999999)
        self.batch_start_spin.setValue(0)
        range_layout.addWidget(self.batch_start_spin)
        
        range_layout.addWidget(QLabel("End:"))
        self.batch_end_spin = QSpinBox()
        self.batch_end_spin.setRange(0, 999999)
        self.batch_end_spin.setValue(1000)
        range_layout.addWidget(self.batch_end_spin)
        
        settings_layout.addWidget(range_group)
        
        apply_btn = QPushButton("Process Image (Batch Export)")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a3f5f; 
                color: white; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #3b5278; }
        """)
        apply_btn.clicked.connect(self._on_process_clicked)
        settings_layout.addWidget(apply_btn)
        
        settings_scroll.setWidget(settings_widget)
        content_layout.addWidget(settings_scroll)
        
        main_layout.addLayout(content_layout)
    
    def _create_camera_tabs(self, num_cams):
        """Create or update camera tab buttons."""
        # Clear existing buttons and stretch
        for btn in self.cam_buttons:
            self.cam_tabs_layout.removeWidget(btn)
            btn.deleteLater()
        self.cam_buttons.clear()
        
        # Remove all items from layout (including stretch)
        while self.cam_tabs_layout.count():
            item = self.cam_tabs_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Create new buttons
        for i in range(num_cams):
            btn = QPushButton(f"Cam {i}")
            btn.setCheckable(True)
            btn.setChecked(i == self.current_cam)
            btn.setStyleSheet("""
                QPushButton { 
                    background-color: #333; 
                    color: #888; 
                    border: 1px solid #444; 
                    border-radius: 4px; 
                    padding: 6px 16px; 
                    font-weight: bold; 
                    margin-right: 2px;
                }
                QPushButton:checked { 
                    background-color: #444; 
                    color: white; 
                    border-bottom: 2px solid #00d4ff;
                }
                QPushButton:hover { background-color: #3a3a3a; }
            """)
            btn.clicked.connect(lambda checked, idx=i: self._on_cam_tab_clicked(idx))
            self.cam_tabs_layout.addWidget(btn)
            self.cam_buttons.append(btn)
        
        # Add stretch at end for left alignment
        self.cam_tabs_layout.addStretch(1)
    
    def _on_cam_tab_clicked(self, cam_idx):
        """Handle click on camera tab."""
        self.current_cam = cam_idx
        # Update button states
        for i, btn in enumerate(self.cam_buttons):
            btn.setChecked(i == cam_idx)
        self._load_current_image()
    
    def _browse_images(self):
        """Open directory dialog to select main image folder."""
        import os
        
        root_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Main Image Directory",
            ""
        )
        if not root_dir:
            return
        
        self.root_path = root_dir
            
        self._scan_images(root_dir)

    def _browse_project_path(self):
        """Open directory dialog to select project output folder."""
        path = QFileDialog.getExistingDirectory(self, "Select Project Output Directory", "")
        if path:
            self.project_path_input.setText(path)

    def _scan_images(self, root_dir):
        """Scan directory for camera folders and images or cine files."""
        print(f"\n--- Scanning Directory: {root_dir} ---")
        import os
        num_cams = self.num_cameras_spin.value()
        print(f"Looking for {num_cams} cameras (settings)...")
        self.camera_images = {}
        total_images = 9999999 # Large number to find min
        self.camera_folders = []
        print(f"pycine library status: {'Available' if pycine is not None else 'NOT FOUND'}")
        
        # 1. Try Rule 2 (Flat Cine) - cine files directly in root
        root_cine_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith('.cine')
        ])
        
        if len(root_cine_files) > 0:
            print(f"Detected {len(root_cine_files)} Cine files in root: {root_cine_files}")
            # Use as many as possible up to num_cams
            actual_cams = min(len(root_cine_files), num_cams)
            for i in range(actual_cams):
                cine_path = os.path.join(root_dir, root_cine_files[i])
                if pycine:
                    try:
                        from pycine.file import read_header
                        header = read_header(cine_path)
                        cfh = header['cinefileheader']
                        n_frames = cfh.ImageCount
                        first_idx = cfh.FirstImageNo
                        # Store as virtual paths
                        self.camera_images[i] = [f"{cine_path}#{j}" for j in range(first_idx, first_idx + n_frames)]
                        total_images = min(total_images, n_frames)
                        self.camera_folders.append(cine_path)
                        print(f"Loaded Cine: {cine_path} with {n_frames} frames (Start: {first_idx})")
                    except Exception as e:
                        print(f"Error reading cine {cine_path}: {e}")
                else:
                    print("Error: pycine library not found but .cine files detected.")
            
            if len(root_cine_files) < num_cams and len(root_cine_files) > 0:
                print(f"Warning: Expected {num_cams} cameras, but found only {len(root_cine_files)} Cine files.")
            
        else:
            # 2. Try Rule 1 (Subfolders with Cine) or Standard Images
            try:
                subdirs = sorted([
                    d for d in os.listdir(root_dir) 
                    if os.path.isdir(os.path.join(root_dir, d))
                ])
            except Exception as e:
                print(f"Error scanning directory: {e}")
                return
            
            num_cams = min(num_cams, len(subdirs))
            self.camera_folders = [os.path.join(root_dir, d) for d in subdirs[:num_cams]]
            print(f"Found {len(subdirs)} subfolders. Using first {num_cams}: {subdirs[:num_cams]}")
            
            for i, folder in enumerate(self.camera_folders):
                # Check for images first
                img_files = sorted([
                    f for f in os.listdir(folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
                ])
                
                if img_files:
                    # Standard Image Folder
                    self.camera_images[i] = [os.path.join(folder, f) for f in img_files]
                    total_images = min(total_images, len(img_files))
                else:
                    # Look for Cine (Rule 1)
                    cine_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.cine')])
                    if cine_files and pycine:
                        cine_path = os.path.join(folder, cine_files[0])
                        try:
                            from pycine.file import read_header
                            header = read_header(cine_path)
                            cfh = header['cinefileheader']
                            n_frames = cfh.ImageCount
                            first_idx = cfh.FirstImageNo
                            self.camera_images[i] = [f"{cine_path}#{j}" for j in range(first_idx, first_idx + n_frames)]
                            total_images = min(total_images, n_frames)
                            print(f"Loaded Cine from subfolder: {cine_path} with {n_frames} frames")
                        except Exception as e:
                            print(f"Error reading cine {cine_path}: {e}")
                    else:
                        self.camera_images[i] = []
                        total_images = 0

        if total_images == 9999999:
            total_images = 0
            
        # Update UI
        self.current_cam = 0
        self.current_frame = 0
        self.image_count_label.setText(f"{total_images} frames per camera")
        
        # Update batch range end to max frames
        if hasattr(self, 'batch_end_spin'):
            self.batch_end_spin.setValue(max(0, total_images - 1))
        
        # Update Camera Tabs
        self._create_camera_tabs(len(self.camera_folders))
        
        # Update Frame List
        self.frame_table.setRowCount(total_images)
        self.image_paths = [] # Keep track of frame 0 paths for table
        
        if 0 in self.camera_images:
            self.image_paths = self.camera_images[0] # Use cam 1 for list
            count = min(len(self.image_paths), total_images)
            for row in range(count):
                path = self.image_paths[row]
                if "#" in path:
                    fname = os.path.basename(path.split("#")[0]) + f" [Frame {path.split('#')[1]}]"
                else:
                    fname = os.path.basename(path)
                
                item_idx = QTableWidgetItem(str(row + 1))
                item_idx.setData(Qt.ItemDataRole.ForegroundRole, QColor("white"))
                self.frame_table.setItem(row, 0, item_idx)
                
                item_name = QTableWidgetItem(fname)
                item_name.setData(Qt.ItemDataRole.ForegroundRole, QColor("white"))
                self.frame_table.setItem(row, 1, item_name)
        
        if total_images > 0:
            self._load_current_image()
        

    
    def _populate_frame_table(self):
        """Populate the frame list table with images from first camera."""
        import os
        from PySide6.QtWidgets import QTableWidgetItem
        
        # Use first camera's images as reference
        if 0 not in self.camera_images:
            return
        
        images = self.camera_images[0]
        self.frame_table.setRowCount(len(images))
        for i, path in enumerate(images):
            idx_item = QTableWidgetItem(str(i))
            filename_item = QTableWidgetItem(os.path.basename(path))
            self.frame_table.setItem(i, 0, idx_item)
            self.frame_table.setItem(i, 1, filename_item)
        
        # Select first row
        if images:
            self.frame_table.selectRow(0)
    
    def _on_frame_clicked(self, row, col):
        """Handle click on frame table row."""
        if self.current_cam in self.camera_images:
            images = self.camera_images[self.current_cam]
            if 0 <= row < len(images):
                self.current_frame = row
                self._load_current_image()
    
    def _load_current_image(self):
        """Load the current image for preview."""
        if self.current_cam not in self.camera_images:
            return
        
        images = self.camera_images[self.current_cam]
        if not images or self.current_frame >= len(images):
            return
        
        # If we are simply viewing the background, don't read the raw image
        if self.current_view_mode == "background":
            self._toggle_view("background")
            return

        path = images[self.current_frame]
        raw_image = self._read_image(path)
        
        if raw_image is not None:
            self.original_image = raw_image  # Keep RAW (may be uint16)
            self.processed_image = None
            
            if self.current_view_mode == "processed":
                self._preview_processing()
            else:
                # For "original" view: normalize high bit-depth for display
                display_img = self._normalize_for_display(raw_image)
                if self.invert_check.isChecked():
                    display_img = 255 - display_img
                self.image_label.setCvImage(display_img)
                self._update_toggle_buttons()
    
    def _normalize_for_display(self, img):
        """Normalize high bit-depth image to 8-bit for display using percentile stretch."""
        if img.dtype == np.uint8:
            return img
        
        # Use percentile-based normalization for stable display
        p_low = np.percentile(img, 1)
        p_high = np.percentile(img, 99.5)
        
        if p_high - p_low < 1:
            p_high = p_low + 1
        
        normalized = (img.astype(np.float32) - p_low) / (p_high - p_low) * 255
        return np.clip(normalized, 0, 255).astype(np.uint8)
    
    def _toggle_view(self, view_mode):
        """Toggle between original, processed, and background view."""
        self.current_view_mode = view_mode
        self._update_toggle_buttons()
        
        if view_mode == "original":
            if self.original_image is not None:
                img = self._normalize_for_display(self.original_image)
                if self.invert_check.isChecked():
                    img = 255 - img
                self.image_label.setCvImage(img)
        elif view_mode == "processed":
            if self.processed_image is not None:
                self.image_label.setCvImage(self.processed_image)
            else:
                self._preview_processing()
        elif view_mode == "background":
            if self.current_cam in self.camera_backgrounds:
                bg = self.camera_backgrounds[self.current_cam]
                # Normalize float32 background for display
                bg_display = self._normalize_for_display(bg)
                if self.invert_check.isChecked():
                    bg_display = 255 - bg_display
                self.image_label.setCvImage(bg_display)

    def _get_project_output_dir(self):
        project_path = self.project_path_input.text().strip()
        if project_path:
            return os.path.abspath(project_path)
        return ""

    def _background_cache_paths(self):
        project_path = self._get_project_output_dir()
        if not project_path:
            return "", ""
        img_file_dir = os.path.join(project_path, "imgFile")
        cache_path = os.path.join(img_file_dir, "background_cache.mat")
        return img_file_dir, cache_path

    def _ask_background_mode(self, existing_count, total_count):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Background Cache Found")
        msg.setText(
            f"Found cached background for {existing_count}/{total_count} cameras.\n"
            "Choose how to proceed:"
        )
        load_btn = msg.addButton("Load Existing", QMessageBox.ButtonRole.AcceptRole)
        recalc_btn = msg.addButton("Recalculate", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(load_btn)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == load_btn:
            return "load"
        if clicked == recalc_btn:
            return "recalculate"
        return "cancel"

    def _load_background_cache(self, cache_path):
        backgrounds = {}
        shifts = {}
        if not cache_path or not os.path.exists(cache_path):
            return backgrounds, shifts

        try:
            from scipy.io import loadmat

            data = loadmat(cache_path)
            cam_ids = data.get("cam_ids", np.array([], dtype=np.int32)).flatten()
            shift_arr = data.get("cine_shifts", np.array([], dtype=np.int32)).flatten()

            for i, cam_id_val in enumerate(cam_ids):
                cam_idx = int(cam_id_val)
                bg_key = f"bg_cam{cam_idx}"
                if bg_key not in data:
                    continue
                bg = np.array(data[bg_key], dtype=np.float32, copy=True)
                backgrounds[cam_idx] = bg
                if i < len(shift_arr):
                    shifts[cam_idx] = int(shift_arr[i])
                else:
                    shifts[cam_idx] = 0
        except Exception as e:
            print(f"Failed to load background cache: {e}")

        return backgrounds, shifts

    def _save_background_cache(self, cache_path, backgrounds, shifts, stride, avg_count, invert_enabled):
        if not cache_path:
            return False
        try:
            from scipy.io import savemat

            cam_ids = sorted(int(k) for k in backgrounds.keys())
            mdict = {
                "cam_ids": np.array(cam_ids, dtype=np.int32),
                "cine_shifts": np.array([int(shifts.get(cid, 0)) for cid in cam_ids], dtype=np.int32),
                "skip_frames": np.array([[int(stride)]], dtype=np.int32),
                "avg_count": np.array([[int(avg_count)]], dtype=np.int32),
                "invert_enabled": np.array([[1 if invert_enabled else 0]], dtype=np.int32),
                "version": np.array([[1]], dtype=np.int32),
                "timestamp": np.array([time.strftime("%Y-%m-%d %H:%M:%S")], dtype=object),
            }
            for cam_idx in cam_ids:
                mdict[f"bg_cam{cam_idx}"] = np.array(backgrounds[cam_idx], dtype=np.float32, copy=False)
            savemat(cache_path, mdict, do_compression=True)
            return True
        except Exception as e:
            print(f"Failed to save background cache: {e}")
            return False
    
    
    def _calculate_all_backgrounds(self):
        """Calculate background for all cameras."""
        if not self.camera_images:
            return
        if self.bg_calc_thread is not None:
            return
        self._busy_begin('calc_background', 'Calculating backgrounds')
        try:
            stride = max(1, self.skip_frames_spin.value())
            avg_count = self.avg_count_spin.value()
            invert_enabled = self.invert_check.isChecked()

            # Build worklist and progress budgets.
            cam_jobs = []
            cam_budget = {}
            total_frames = 0
            for cam_idx, images in self.camera_images.items():
                if not images:
                    continue
                available_indices = list(range(0, len(images), stride))
                selected_indices = available_indices[:avg_count]
                if not selected_indices:
                    continue
                cam_jobs.append((cam_idx, images, selected_indices))
                budget = len(selected_indices) + 1  # +1 for bit-shift estimation
                cam_budget[cam_idx] = budget
                total_frames += budget

            if not cam_jobs:
                self._busy_end('calc_background')
                return

            requested_cam_ids = [cam_idx for cam_idx, _, _ in cam_jobs]
            _, cache_path = self._background_cache_paths()
            cached_backgrounds = {}
            cached_shifts = {}
            if cache_path:
                cached_backgrounds, cached_shifts = self._load_background_cache(cache_path)

            existing_for_request = [cid for cid in requested_cam_ids if cid in cached_backgrounds]
            mode = "recalculate"
            if existing_for_request:
                mode = self._ask_background_mode(len(existing_for_request), len(requested_cam_ids))
                if mode == "cancel":
                    self._busy_end('calc_background')
                    return

            # Load existing cache first if requested. Missing cameras will be computed.
            if mode == "load":
                for cid in existing_for_request:
                    self.camera_backgrounds[cid] = np.array(cached_backgrounds[cid], dtype=np.float32, copy=True)
                    self.cine_shifts[cid] = int(cached_shifts.get(cid, 0))

            if mode == "load":
                cam_jobs = [job for job in cam_jobs if job[0] not in existing_for_request]
                if not cam_jobs:
                    print("Loaded background cache for all cameras. Skipped computation.")
                    if self.current_view_mode == "background":
                        self._toggle_view("background")
                    self._busy_end('calc_background')
                    return
            total_frames = sum(cam_budget[job[0]] for job in cam_jobs)
            max_workers = min(len(cam_jobs), max(1, (os.cpu_count() or 2) - 1), 4)

            self._bg_cache_path = cache_path
            self._bg_cache_stride = stride
            self._bg_cache_avg_count = avg_count
            self._bg_cache_invert_enabled = invert_enabled

            self.bg_calc_dialog = BackgroundProgressDialog(self, total=total_frames)
            self.bg_calc_dialog.update_progress(0, total_frames, f"Calculating backgrounds... ({max_workers} workers)")
            self.bg_calc_dialog.show()

            self.bg_calc_thread = QThread(self)
            self.bg_calc_worker = BackgroundWorker(
                cam_jobs=cam_jobs,
                cam_budget=cam_budget,
                invert_enabled=invert_enabled,
                max_workers=max_workers,
            )
            self.bg_calc_worker.moveToThread(self.bg_calc_thread)

            self.bg_calc_thread.started.connect(self.bg_calc_worker.run)
            self.bg_calc_worker.progress.connect(self._on_bg_progress)
            self.bg_calc_worker.finished.connect(self._on_bg_finished)
            self.bg_calc_worker.error.connect(self._on_bg_error)

            self.bg_calc_worker.finished.connect(self.bg_calc_thread.quit)
            self.bg_calc_worker.error.connect(self.bg_calc_thread.quit)
            self.bg_calc_worker.finished.connect(self.bg_calc_worker.deleteLater)
            self.bg_calc_worker.error.connect(self.bg_calc_worker.deleteLater)
            self.bg_calc_thread.finished.connect(self.bg_calc_thread.deleteLater)

            self.bg_calc_thread.start()
        except Exception as e:
            if self.bg_calc_dialog is not None:
                self.bg_calc_dialog.close()
                self.bg_calc_dialog = None
            self.bg_calc_worker = None
            self.bg_calc_thread = None
            QMessageBox.critical(self, "Background Calculation Failed", str(e))
            self._busy_end('calc_background')

    @Slot(int, int, str)
    def _on_bg_progress(self, current, total, message):
        if self.bg_calc_dialog is not None:
            self.bg_calc_dialog.update_progress(current, total, message)

    @Slot(dict)
    def _on_bg_finished(self, summary):
        try:
            for cam_idx, bg in summary.get("backgrounds", {}).items():
                self.camera_backgrounds[int(cam_idx)] = np.array(bg, dtype=np.float32, copy=True)
            for cam_idx, shift in summary.get("shifts", {}).items():
                self.cine_shifts[int(cam_idx)] = int(shift)

            if self._bg_cache_path:
                os.makedirs(os.path.dirname(self._bg_cache_path), exist_ok=True)
                saved = self._save_background_cache(
                    cache_path=self._bg_cache_path,
                    backgrounds=self.camera_backgrounds,
                    shifts=self.cine_shifts,
                    stride=self._bg_cache_stride,
                    avg_count=self._bg_cache_avg_count,
                    invert_enabled=self._bg_cache_invert_enabled,
                )
                if saved:
                    print(f"Background cache saved: {self._bg_cache_path}")

            if self.current_view_mode == "background":
                self._toggle_view("background")

            print(f"Calculated backgrounds for {len(self.camera_backgrounds)} cameras")
        finally:
            if self.bg_calc_dialog is not None:
                self.bg_calc_dialog.close()
                self.bg_calc_dialog = None
            self.bg_calc_worker = None
            self.bg_calc_thread = None
            self._busy_end('calc_background')

    @Slot(str)
    def _on_bg_error(self, message):
        if self.bg_calc_dialog is not None:
            self.bg_calc_dialog.close()
            self.bg_calc_dialog = None
        self.bg_calc_worker = None
        self.bg_calc_thread = None
        QMessageBox.critical(self, "Background Calculation Failed", str(message))
        self._busy_end('calc_background')
    
    def _update_toggle_buttons(self):
        """Update toggle button states."""
        mode = getattr(self, 'current_view_mode', 'original')
        self.original_btn.setChecked(mode == "original")
        self.processed_btn.setChecked(mode == "processed")
        self.background_btn.setChecked(mode == "background")
    
    def _on_settings_changed(self):
        """Called when any adjustment setting changes."""
        if self.current_view_mode == "processed":
            self._preview_processing()
        else:
            # If in original or background mode, just refresh current display (handles Invert toggle)
            self._toggle_view(self.current_view_mode)
    
    
    def _apply_processing_pipeline(self, img_data, bg_data=None, cam_idx=None):
        """
        Apply the full processing pipeline to a single image.
        Pipeline: Background Subtraction (float32) -> Bit Shift (8-bit) -> Input Range -> Denoise
        """
        if cam_idx is None:
            cam_idx = self.current_cam
            
        # 0. Ensure grayscale and float32
        if len(img_data.shape) == 3:
            gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = img_data.astype(np.float32)
        
        # 1. Background Subtraction (in float32, before bit shift)
        if self.bg_enabled.isChecked() and bg_data is not None:
            # bg_data is already float32
            # Handle Invert logic here:
            # If Invert is checked (Shadowgraphy), Signal = Background - Image
            # If Norm/Fluorescence, Signal = Image - Background
            if self.invert_check.isChecked():
                result = bg_data - gray
            else:
                result = gray - bg_data
            
            result = np.clip(result, 0, None)  # Allow values > 255 for now
        else:
            result = gray
        
        # 2. Bit Shift to 8-bit (using pre-calculated N from subtracted frame statistics)
        shift = self.cine_shifts.get(cam_idx, 0)
        
        # DEBUG Pipeline
        # p_val = np.percentile(result, 99.5)
        # Only print occasionally or for single frame preview (not batch)
        # But for now, we print always (it's fast enough for preview)
        # print(f"DEBUG Pipeline Cam {self.current_cam}: Shift={shift}, Pre-Shift Max={result.max()}, P99.5={p_val}")
        
        if shift > 0:
            result = (result / (2**shift))
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 3. Invert (Apply ONLY if BG subtraction was NOT done)
        # If we did (BG - Image), we already have "Bright Signal". We don't want to invert back.
        # If BG disabled, and Invert checked, we still need to invert.
        if self.invert_check.isChecked() and not (self.bg_enabled.isChecked() and bg_data is not None):
            result = 255 - result
        
        # 4. Input Range Adjustment (imadjust on 8-bit)
        low_in = self.range_slider.minValue()
        high_in = self.range_slider.maxValue()
        result = imadjust_opencv(result, low_in, high_in)
        
        # 5. LaVision Processing (Enhanced Denoise, on 8-bit)
        if self.denoise_check.isChecked():
            a = result.astype(np.float32)
            kernel = np.ones((3, 3), np.uint8)
            b = cv2.erode(a, kernel, iterations=1)
            c = a - b
            b = cv2.erode(a, kernel, iterations=1)
            c = c - b
            
            d = cv2.GaussianBlur(c, (0, 0), 0.5)
            
            k_size = 100
            e = cv2.blur(d, (k_size, k_size))
            f = a - e
            
            blurred_f = cv2.GaussianBlur(f, (0, 0), 1.0)
            sharp = f + 0.8 * (f - blurred_f)
            
            result = np.clip(sharp, 0, 255).astype(np.uint8)
        
        return result.astype(np.uint8)
    def _preview_processing(self):
        """Apply current settings and show preview."""
        if self.original_image is None:
            return
        
        # Prepare background
        bg = None
        if self.current_cam in self.camera_backgrounds:
            bg = self.camera_backgrounds[self.current_cam]
            
        # Use unified pipeline
        processed = self._apply_processing_pipeline(self.original_image, bg)
        
        self.processed_image = processed
        self._toggle_view("processed")
    def _apply_to_all(self):
        """Apply current settings to all loaded images."""
        # TODO: Implement batch processing
        print(f"Apply to all images")

    def _on_process_clicked(self):
        """Start batch processing in worker thread."""
        if self._is_processing:
            return
        self._busy_begin('batch_process', 'Batch image preprocessing')

        project_path = self.project_path_input.text().strip()
        if not project_path:
            QMessageBox.warning(self, "Project Path Required", "Please set Project Path (for Output) before batch processing.")
            self._busy_end('batch_process')
            return
        project_path = os.path.abspath(project_path)
        self._batch_project_path = project_path

        os.makedirs(project_path, exist_ok=True)
        img_file_dir = os.path.join(project_path, "imgFile")
        os.makedirs(img_file_dir, exist_ok=True)

        start_idx = self.batch_start_spin.value()
        end_idx = self.batch_end_spin.value()

        all_entries = []
        cam_output_paths = {}

        for cam_idx, file_list in self.camera_images.items():
            cam_out_dir = os.path.join(img_file_dir, f"cam{cam_idx}")
            os.makedirs(cam_out_dir, exist_ok=True)
            cam_output_paths[cam_idx] = []

            for i, src_path in enumerate(file_list):
                if not (start_idx <= i <= end_idx):
                    continue
                dst_path = os.path.join(cam_out_dir, f"img{i:06d}.tif")
                abs_dst = os.path.abspath(dst_path)
                cam_output_paths[cam_idx].append((i, abs_dst))
                entry = {
                    "src": src_path,
                    "dst": abs_dst,
                    "cam_idx": cam_idx,
                    "frame_idx": i,
                }
                if "#" in src_path:
                    cine_file, frame_str = src_path.split("#")
                    entry["is_cine"] = True
                    entry["cine_file"] = cine_file
                    entry["cine_frame"] = int(frame_str)
                else:
                    entry["is_cine"] = False
                all_entries.append(entry)

        if not all_entries:
            self._busy_end('batch_process')
            return

        existing = [e for e in all_entries if os.path.exists(e["dst"]) and os.path.getsize(e["dst"]) > 0]
        mode = "reprocess"
        if existing:
            mode = self._ask_processing_mode(len(existing), len(all_entries))
            if mode == "cancel":
                self._busy_end('batch_process')
                return

        if mode == "continue":
            tasks = [e for e in all_entries if not (os.path.exists(e["dst"]) and os.path.getsize(e["dst"]) > 0)]
        else:
            tasks = all_entries

        if not tasks:
            self._write_image_name_files(img_file_dir, cam_output_paths)
            self._busy_end('batch_process')
            return

        settings = {
            "bg_enabled": bool(self.bg_enabled.isChecked()),
            "invert": bool(self.invert_check.isChecked()),
            "low_in": int(self.range_slider.minValue()),
            "high_in": int(self.range_slider.maxValue()),
            "denoise": bool(self.denoise_check.isChecked()),
            "cine_shifts": dict(self.cine_shifts),
            "camera_backgrounds": {k: np.array(v, copy=True) for k, v in self.camera_backgrounds.items()},
        }

        cpu_cores = max(1, int(os.cpu_count() or 1))
        total_workers = min(cpu_cores, 20)
        write_workers = int(round(total_workers * 0.2))
        write_workers = max(2, min(6, write_workers))
        process_workers = min(max(cpu_cores - write_workers, 1), 16)

        self._batch_cam_output_paths = cam_output_paths
        self._batch_img_file_dir = img_file_dir
        self._is_processing = True
        self._is_paused = False

        self.processing_dialog = ProcessingDialog(self, title="Batch Processing Images")
        self.processing_dialog.show()

        self.preprocess_thread = QThread(self)
        self.preprocess_worker = PreprocessWorker(
            tasks=tasks,
            settings=settings,
            process_workers=process_workers,
            write_workers=write_workers,
            max_pending_writes=64,
        )
        self.preprocess_worker.moveToThread(self.preprocess_thread)

        self.preprocess_thread.started.connect(self.preprocess_worker.run)
        self.preprocess_worker.progress.connect(self._on_worker_progress)
        self.preprocess_worker.finished.connect(self._on_worker_finished)
        self.preprocess_worker.error.connect(lambda msg: print(f"Preprocess error: {msg}"))

        self.processing_dialog.pause_signal.connect(self.preprocess_worker.set_paused)
        self.processing_dialog.stop_signal.connect(self.preprocess_worker.request_stop)

        self.preprocess_worker.finished.connect(self.preprocess_thread.quit)
        self.preprocess_worker.finished.connect(self.preprocess_worker.deleteLater)
        self.preprocess_thread.finished.connect(self.preprocess_thread.deleteLater)

        self.preprocess_thread.start()
        
    def _stop_batch_processing(self):
        if self.preprocess_worker is not None:
            self.preprocess_worker.request_stop()
        
    def _read_image(self, path):
        """Helper to read image from either file or cine frame. Returns raw data (no bit shift)."""
        import os
        if "#" in path:
            file_path, frame_idx = path.split("#")
            frame_idx = int(frame_idx)
            
            if not pycine:
                return None
            
            try:
                raw_images, setup, bpp = pycine_read_frames(file_path, start_frame=frame_idx, count=1)
                images = list(raw_images)
                if images and len(images) > 0:
                    # Return raw data (uint16 or uint8) - no bit shifting here
                    return np.array(images[0])
                else:
                    return None
            except Exception as e:
                print(f"Error reading frame {frame_idx} from {file_path}: {e}")
                return None
        else:
            return cv2.imread(path, cv2.IMREAD_UNCHANGED)

        
    def _pause_batch_processing(self, paused):
        self._is_paused = bool(paused)
        if self.preprocess_worker is not None:
            self.preprocess_worker.set_paused(paused)

    def _ask_processing_mode(self, existing_count, total_count):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Existing Output Detected")
        msg.setText(
            f"Detected {existing_count}/{total_count} already processed images.\n"
            "Choose how to proceed:"
        )
        continue_btn = msg.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
        reprocess_btn = msg.addButton("Reprocess", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(continue_btn)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == continue_btn:
            return "continue"
        if clicked == reprocess_btn:
            return "reprocess"
        return "cancel"

    @Slot(int, int, str)
    def _on_worker_progress(self, current, total, message):
        if self.processing_dialog is not None:
            self.processing_dialog.update_progress(current, total)
            self.processing_dialog.status_label.setText(message)

    @Slot(dict)
    def _on_worker_finished(self, summary):
        try:
            if not summary.get("stopped", False):
                self._write_image_name_files(self._batch_img_file_dir, self._batch_cam_output_paths)
        finally:
            if self.processing_dialog is not None:
                self.processing_dialog.close()
                self.processing_dialog = None
            self._is_processing = False
            self._is_paused = False
            self.preprocess_worker = None
            self.preprocess_thread = None
            self._busy_end('batch_process')

        print(
            "Batch preprocessing done: "
            f"processed={summary.get('processed', 0)}/{summary.get('total', 0)}, "
            f"failed={summary.get('failed', 0)}, "
            f"elapsed={summary.get('elapsed_sec', 0.0):.2f}s, "
            f"fps={summary.get('fps', 0.0):.2f}"
        )

    def _write_image_name_files(self, img_file_dir, cam_output_paths):
        project_root = getattr(self, "_batch_project_path", None) or self.project_path_input.text().strip()
        project_root = os.path.abspath(project_root) if project_root else ""

        for cam_idx, indexed_paths in cam_output_paths.items():
            indexed_paths = sorted(indexed_paths, key=lambda x: x[0])
            txt_path = os.path.join(img_file_dir, f"cam{cam_idx}ImageNames.txt")
            with open(txt_path, "w") as f:
                for _, p in indexed_paths:
                    out_path = p
                    if project_root:
                        try:
                            out_path = os.path.relpath(p, start=project_root)
                        except ValueError:
                            out_path = p
                    f.write(out_path.replace('\\', '/') + "\n")

    def _on_pixel_clicked(self, x, y, intensity):
        """Handle pixel click signal."""
        self.pixel_info_label.setText(f"Pixel ({x}, {y}): {intensity}")
