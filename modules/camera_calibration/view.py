import numpy as np
import cv2
import os
from datetime import datetime

import matplotlib
matplotlib.use('qtagg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QTabWidget, QComboBox, QSpinBox, 
                              QDoubleSpinBox, QListWidget, QGroupBox, QFormLayout,
                              QCheckBox, QFileDialog, QScrollArea, QFrame,
                              QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                              QSizePolicy, QGridLayout, QRadioButton, QButtonGroup)
from PySide6.QtCore import Qt
from .widgets import RangeSlider
from .wand_calibration.wand_calibrator import WandCalibrator
from .wand_calibration.refraction_wand_calibrator import RefractiveWandCalibrator
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap, QPainterPath
from PySide6.QtCore import QRect, Signal, QPoint, QThread, QTimer, Slot, QObject

class ZoomableImageLabel(QLabel):
    """
    Label with zoom, pan, and multiple interaction modes.
    """
    template_selected = Signal(QRect)
    roi_points_changed = Signal(list)
    remove_region_selected = Signal(QRect)  # Signal for remove mode
    add_region_selected = Signal(QRect)     # Signal for add mode
    origin_selected = Signal(QPoint)        # Signal for origin selection
    axis_point_selected = Signal(QPoint, int)  # Signal for axis direction (point, axis_index)
    point_clicked = Signal(QPoint)          # Signal for manual point verification click
    
    # Modes
    MODE_NAV = 0
    MODE_TEMPLATE = 1
    MODE_ROI = 2
    MODE_REMOVE = 3
    MODE_ADD = 4
    MODE_ORIGIN = 5
    MODE_AXES = 6
    MODE_CHECK_POS = 7
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        
        # Image Data
        self._pixmap = None
        
        # View State (user-controlled zoom and pan)
        self._user_zoom = 1.0      # User zoom factor (1.0 = fit to window)
        self._user_pan_x = 0.0     # User pan offset in widget coords
        self._user_pan_y = 0.0
        self.last_mouse_pos = None
        
        # Interaction State
        self.mode = self.MODE_NAV
        self.is_panning = False
        self.is_selecting_rect = False
        
        # Template Selection
        self.rect_start = None
        self.rect_end = None # Current drag end
        self.selection_rect = None # Finished rect (Image Coords)
        
        # ROI Selection
        self.roi_points = [] # List of QPoint (Image Coords)
        
        # Interaction Feedback (Hints)
        self._hint_text = ""
        self._mouse_pos_widget = QPoint(0, 0)
        
    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        # Reset view on new image load
        self.resetView()
        self.update()
        
    def fit_to_window(self):
        """Reset to fit-to-window view."""
        self._user_zoom = 1.0
        self._user_pan_x = 0.0
        self._user_pan_y = 0.0
        self.update()
            
    def set_mode(self, mode):
        self.mode = mode
        if mode == self.MODE_NAV:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._hint_text = ""  # Clear hint on reset
        elif mode == self.MODE_TEMPLATE:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == self.MODE_ROI:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        elif mode == self.MODE_REMOVE:
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
        elif mode in (self.MODE_ADD, self.MODE_ORIGIN, self.MODE_AXES, self.MODE_CHECK_POS):
            self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()
        
    def clear_overlays(self):
        self.selection_rect = None
        self.roi_points = []
        self.update()

    def _calc_transform_params(self):
        """Calculate current display transform parameters including user zoom/pan."""
        if not self._pixmap or self._pixmap.isNull():
            return 1.0, 0, 0
        
        p_w = self._pixmap.width()
        p_h = self._pixmap.height()
        w_w = self.width()
        w_h = self.height()
        
        if p_w <= 0 or p_h <= 0 or w_w <= 0 or w_h <= 0:
            return 1.0, 0, 0
        
        # Base fit-to-window scale
        base_scale = min(w_w / p_w, w_h / p_h)
        
        # Apply user zoom
        scale = base_scale * self._user_zoom
        
        t_w = int(p_w * scale)
        t_h = int(p_h * scale)
        
        # Base centered position
        base_x = (w_w - t_w) / 2
        base_y = (w_h - t_h) / 2
        
        # Apply user pan
        t_x = int(base_x + self._user_pan_x)
        t_y = int(base_y + self._user_pan_y)
        
        return scale, t_x, t_y
        
    def _to_image_coords(self, widget_pt):
        # Dynamically calculate transform params
        scale, off_x, off_y = self._calc_transform_params()
        
        ix = (widget_pt.x() - off_x) / scale if scale > 0 else 0
        iy = (widget_pt.y() - off_y) / scale if scale > 0 else 0
        return QPoint(int(ix), int(iy))
        
    def _to_widget_coords(self, img_pt):
        # Dynamically calculate transform params
        scale, off_x, off_y = self._calc_transform_params()
        
        wx = img_pt.x() * scale + off_x
        wy = img_pt.y() * scale + off_y
        return QPoint(int(wx), int(wy))

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if not self._pixmap or self._pixmap.isNull():
            return
            
        # Get mouse position for zoom-towards-cursor
        mouse_pos = event.position().toPoint()
        
        # Zoom factor
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else (1.0 / 1.15)
        
        # Limit zoom range
        new_zoom = self._user_zoom * zoom_factor
        if new_zoom < 0.1:
            new_zoom = 0.1
        elif new_zoom > 20.0:
            new_zoom = 20.0
            
        # To zoom towards cursor, we need to adjust pan so the point under cursor stays fixed
        # 1. Get image coord under cursor before zoom
        old_scale, old_tx, old_ty = self._calc_transform_params()
        img_x = (mouse_pos.x() - old_tx) / old_scale if old_scale > 0 else 0
        img_y = (mouse_pos.y() - old_ty) / old_scale if old_scale > 0 else 0
        
        # 2. Apply new zoom
        self._user_zoom = new_zoom
        
        # 3. Calculate where that image point would be with new zoom (at center, no pan adjustment yet)
        new_scale, new_tx, new_ty = self._calc_transform_params()
        new_widget_x = img_x * new_scale + new_tx
        new_widget_y = img_y * new_scale + new_ty
        
        # 4. Adjust pan so the image point stays at mouse position
        self._user_pan_x += mouse_pos.x() - new_widget_x
        self._user_pan_y += mouse_pos.y() - new_widget_y
        
        self.update()
        event.accept()

    def resetView(self):
        """Reset zoom and pan to default fit-to-window."""
        self._user_zoom = 1.0
        self._user_pan_x = 0.0
        self._user_pan_y = 0.0
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = True
            self.last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
        elif event.button() == Qt.MouseButton.LeftButton:
            img_pt = self._to_image_coords(event.position().toPoint())
            
            if self.mode == self.MODE_TEMPLATE:
                self.is_selecting_rect = True
                self.rect_start = img_pt
                self.rect_end = img_pt
                
            elif self.mode == self.MODE_ROI:
                # Add point (max 4)
                if len(self.roi_points) < 4:
                    self.roi_points.append(img_pt)
                    self.roi_points_changed.emit(self.roi_points)
                    self.update()
                    
            elif self.mode in (self.MODE_REMOVE, self.MODE_ADD):
                # Start rect selection for remove/add
                self.is_selecting_rect = True
                self.rect_start = img_pt
                self.rect_end = img_pt
                
            elif self.mode == self.MODE_ORIGIN:
                # Single click to select origin
                self.origin_selected.emit(img_pt)
                
            elif self.mode == self.MODE_AXES:
                # Click to select axis direction point
                axis_idx = getattr(self, '_current_axis_index', 0)
                self.axis_point_selected.emit(img_pt, axis_idx)
                
            elif self.mode == self.MODE_CHECK_POS:
                # Click to check 3D position
                self.point_clicked.emit(img_pt)

    def mouseMoveEvent(self, event):
        self._mouse_pos_widget = event.position().toPoint()
        
        if self.is_panning:
            delta = self._mouse_pos_widget - self.last_mouse_pos
            self._user_pan_x += delta.x()
            self._user_pan_y += delta.y()
            self.last_mouse_pos = self._mouse_pos_widget
            self.update()
            
        elif self.is_selecting_rect:
            self.rect_end = self._to_image_coords(self._mouse_pos_widget)
            self.update()
            
        elif self.mode in (self.MODE_ORIGIN, self.MODE_AXES):
            # Just trigger repaint for hint following
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = False
            # If in Remove or Add mode, exit to NAV mode on right-click
            if self.mode in (self.MODE_REMOVE, self.MODE_ADD):
                self.set_mode(self.MODE_NAV)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor if self.mode == self.MODE_NAV else Qt.CursorShape.CrossCursor)
            
        elif event.button() == Qt.MouseButton.LeftButton and self.is_selecting_rect:
            self.is_selecting_rect = False
            # Normalize rect
            if self.rect_start and self.rect_end:
                x1, y1 = self.rect_start.x(), self.rect_start.y()
                x2, y2 = self.rect_end.x(), self.rect_end.y()
                r = QRect(QPoint(min(x1,x2), min(y1,y2)), QPoint(max(x1,x2), max(y1,y2)))
                if r.width() > 2 and r.height() > 2:
                    if self.mode == self.MODE_TEMPLATE:
                        self.selection_rect = r
                        self.template_selected.emit(r)
                    elif self.mode == self.MODE_REMOVE:
                        self.remove_region_selected.emit(r)
                    elif self.mode == self.MODE_ADD:
                        self.add_region_selected.emit(r)
            # Clear temp rect
            self.rect_start = None
            self.rect_end = None
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20)) # Dark bg
        
        if self._pixmap and not self._pixmap.isNull():
            # Get transform params (includes user zoom and pan)
            scale, t_x, t_y = self._calc_transform_params()
            
            p_w = self._pixmap.width()
            p_h = self._pixmap.height()
            t_w = int(p_w * scale)
            t_h = int(p_h * scale)
            
            # Store for coordinate transforms (used by _to_image_coords)
            self._draw_scale = scale
            self._draw_offset_x = t_x
            self._draw_offset_y = t_y
            
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawPixmap(QRect(t_x, t_y, t_w, t_h), self._pixmap)
            
            # Draw Overlays in Widget Coords
            # 1. Template Selection Rect (finished)
            if self.selection_rect:
                pen = QPen(QColor(0, 255, 0), 2)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                # Convert image coords to widget coords
                r = self.selection_rect
                wx = int(r.x() * scale + t_x)
                wy = int(r.y() * scale + t_y)
                ww = int(r.width() * scale)
                wh = int(r.height() * scale)
                painter.drawRect(wx, wy, ww, wh)
                
            # 2. Dragging rect (in progress)
            if self.is_selecting_rect and self.rect_start and self.rect_end:
                pen = QPen(QColor(0, 255, 255), 2, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                x1, y1 = self.rect_start.x(), self.rect_start.y()
                x2, y2 = self.rect_end.x(), self.rect_end.y()
                wx1 = int(min(x1, x2) * scale + t_x)
                wy1 = int(min(y1, y2) * scale + t_y)
                ww = int(abs(x2 - x1) * scale)
                wh = int(abs(y2 - y1) * scale)
                painter.drawRect(wx1, wy1, ww, wh)
                
            # 3. ROI Polygon
            if self.roi_points:
                pen = QPen(QColor(255, 0, 0), 2)
                painter.setPen(pen)
                # Convert to widget coords
                widget_pts = []
                for p in self.roi_points:
                    wx = int(p.x() * scale + t_x)
                    wy = int(p.y() * scale + t_y)
                    widget_pts.append(QPoint(wx, wy))
                # Draw lines
                for i in range(len(widget_pts) - 1):
                    painter.drawLine(widget_pts[i], widget_pts[i+1])
                # Close loop if 4 points
                if len(widget_pts) == 4:
                    painter.drawLine(widget_pts[-1], widget_pts[0])
                    # Fill semi-transparent
                    path = QPainterPath()
                    path.moveTo(widget_pts[0])
                    for p in widget_pts[1:]:
                        path.lineTo(p)
                    path.closeSubpath()
                    painter.fillPath(path, QColor(255, 0, 0, 50))
                # Draw vertices
                painter.setBrush(QColor(255, 255, 0))
                for p in widget_pts:
                    painter.drawEllipse(p, 5, 5)

            # --- Draw Hint Text ---
            if self._hint_text:
                painter.setPen(QPen(QColor(255, 255, 0), 2)) # Yellow hint
                font = painter.font()
                font.setPointSize(12)
                font.setBold(True)
                painter.setFont(font)
                # Offset from cursor
                painter.drawText(self._mouse_pos_widget.x() + 15, self._mouse_pos_widget.y() + 25, self._hint_text)

        else:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image")

class RefractiveCalibWorker(QObject):
    """
    Worker thread for refractive calibration to prevent GUI freeze.
    """
    progress = Signal(str, float, float, float, float)
    finished = Signal(bool, object, object, object)
    error = Signal(str)
    
    def __init__(self, wand_points, wand_length, initial_focal, dist_coeff_num, 
                 active_cam_ids, all_cam_ids, cams_intrinsics, image_size,
                 num_windows, cam_to_window, window_media, out_path, 
                 wand_points_filtered=None, camera_settings=None,
                 precalib_provider=None, use_proj_residuals=False):
        super().__init__()
        self.wand_points = wand_points
        self.wand_points_filtered = wand_points_filtered
        self.wand_length = wand_length
        self.initial_focal = initial_focal
        self.dist_coeff_num = dist_coeff_num
        self.active_cam_ids = active_cam_ids
        self.all_cam_ids = all_cam_ids
        self.cams_intrinsics = cams_intrinsics 
        self.image_size = image_size
        self.camera_settings = camera_settings or {}
        self.precalib_provider = precalib_provider
        self.use_proj_residuals = bool(use_proj_residuals)
        
        # Runtime args
        self.num_windows = num_windows
        self.cam_to_window = cam_to_window
        self.window_media = window_media
        self.out_path = out_path
        
        self._killed = False
        
    def run(self):
        import traceback
        try:
            # Mock base calibrator interface
            class MockWandCalibrator:
                pass
            
            mock_base = MockWandCalibrator()
            mock_base.wand_points = self.wand_points
            mock_base.wand_points_filtered = self.wand_points_filtered
            mock_base.active_cam_ids = self.active_cam_ids
            mock_base.cams = self.cams_intrinsics
            mock_base.cameras = self.cams_intrinsics
            mock_base.dist_coeff_num = self.dist_coeff_num
            mock_base.wand_length = self.wand_length
            mock_base.initial_focal = self.initial_focal
            mock_base.image_size = self.image_size
            mock_base.camera_settings = self.camera_settings

            # Expose real precalibration capability when available.
            if self.precalib_provider is not None and hasattr(self.precalib_provider, 'run_precalibration_check'):
                mock_base.run_precalibration_check = self.precalib_provider.run_precalibration_check
            
            # Instantiate
            from .wand_calibration.refraction_wand_calibrator import RefractiveWandCalibrator
            calibrator = RefractiveWandCalibrator(mock_base)
            
            # Define callback
            def cb(phase, r=None, l=None, p=None, c=None):
                if self._killed: raise RuntimeError("Worker Killed")
                # Backward compatibility:
                # old callbacks pass (phase, ray, len, cost)
                # new callbacks pass (phase, ray, len, proj, cost)
                if c is None:
                    c = p if p is not None else 0.0
                    p = 0.0
                self.progress.emit(str(phase), float(r) if r is not None else -1.0, 
                                          float(l) if l is not None else -1.0,
                                          float(p) if p is not None else -1.0,
                                          float(c))
            
            # Run calibration
            success, cam_params, report, dataset = calibrator.calibrate(
                num_windows=self.num_windows,
                cam_to_window=self.cam_to_window,
                window_media=self.window_media,
                out_path=self.out_path,
                verbosity=1,
                progress_callback=cb,
                use_proj_residuals=self.use_proj_residuals,
            )
            
            # [FIX] Propagate per-frame errors back to main thread
            dataset['per_frame_errors'] = getattr(calibrator.base, 'per_frame_errors', {})
            
            self.finished.emit(success, cam_params, report, dataset)
            
        except Exception as e:
            self.error.emit(traceback.format_exc())

class NumericTableWidgetItem(QTableWidgetItem):
    """TableWidgetItem that sorts numerically instead of alphabetically."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


class TrimmedDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that trims trailing zeros in display text."""
    def textFromValue(self, value):
        text = f"{float(value):.{self.decimals()}f}"
        text = text.rstrip('0').rstrip('.')
        if text in ("", "-0"):
            return "0"
        return text

class Calibration3DViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            self.figure = Figure(figsize=(5, 5), dpi=100)
            self.figure.patch.set_facecolor('black') # Dark theme
            self.canvas = FigureCanvas(self.figure)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0,0,0,0)
            layout.addWidget(self.canvas)
            
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.reset_plot()
            self._initialized = True
            
            # Scroll wheel zoom
            self._zoom_factor = 1.0
            self.canvas.mpl_connect('scroll_event', self._on_scroll)
            
            print("[Calibration3DViewer] Initialized successfully")
        except Exception as e:
            print(f"[Calibration3DViewer] ERROR during init: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            # Fallback: show error label
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(f"3D Viewer Error: {e}"))
    
    def _on_scroll(self, event):
        """Handle mouse scroll wheel for zooming."""
        if event.inaxes != self.ax:
            return
        
        # Zoom factor adjustment
        if event.button == 'up':
            scale = 0.9  # Zoom in (reduce limits)
        elif event.button == 'down':
            scale = 1.1  # Zoom out (expand limits)
        else:
            return
        
        self._zoom_factor *= scale
        
        # Get current limits
        xlim = self.ax.get_xlim3d()
        ylim = self.ax.get_ylim3d()
        zlim = self.ax.get_zlim3d()
        
        # Compute centers
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        # Compute new half-widths
        x_half = (xlim[1] - xlim[0]) / 2 * scale
        y_half = (ylim[1] - ylim[0]) / 2 * scale
        z_half = (zlim[1] - zlim[0]) / 2 * scale
        
        # Apply new limits
        self.ax.set_xlim3d([x_center - x_half, x_center + x_half])
        self.ax.set_ylim3d([y_center - y_half, y_center + y_half])
        self.ax.set_zlim3d([z_center - z_half, z_center + z_half])
        
        self.canvas.draw_idle()
        

    def reset_plot(self):
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # Keep pane borders but remove fill
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('gray')
        self.ax.yaxis.pane.set_edgecolor('gray')
        self.ax.zaxis.pane.set_edgecolor('gray')
        
        # Equal aspect
        self.ax.set_box_aspect([1, 1, 1])
        
        self.canvas.draw()
        
    def plot_calibration(self, cameras, points_3d=None, axis_map: str = None):
        """Plot cameras and points in 3D (MATLAB style)."""
        self.ax.clear()
        
        # Re-apply dark theme styles
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        if axis_map == 'y_up':
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Z (m)')
            self.ax.set_zlabel('Y (m)')
        else:
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
        
        # Keep pane borders but remove fill
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('gray')
        self.ax.yaxis.pane.set_edgecolor('gray')
        self.ax.zaxis.pane.set_edgecolor('gray')
        
        # Convert mm to m (divide by 1000)
        scale = 1000.0
        def map_point(p):
            if axis_map == 'y_up':
                return np.array([p[0], p[2], p[1]])
            return np.array(p)
        
        all_x, all_y, all_z = [], [], []
        
        # 1. Plot Wand Points (Blue)
        if points_3d is not None and len(points_3d) > 0:
            pts = points_3d / scale
            if axis_map == 'y_up':
                pts = pts[:, [0, 2, 1]]
            xs = pts[:, 0]
            ys = pts[:, 1]
            zs = pts[:, 2]
            self.ax.scatter(xs, ys, zs, c='#0077FF', s=5, marker='.', alpha=0.7, label='Wand Points')
            all_x.extend(xs)
            all_y.extend(ys)
            all_z.extend(zs)
        
        # Collect camera positions first
        camera_data = []
        if cameras:
            for c_id, params in cameras.items():
                if 'R' in params and 'T' in params:
                    R = params['R']
                    t = params['T']
                    C = -R.T @ t
                    C = C.flatten() / scale
                    C = map_point(C)
                    camera_data.append((c_id, C, R))
                    all_x.append(C[0])
                    all_y.append(C[1])
                    all_z.append(C[2])
        
        # True axis equal (like MATLAB's axis equal) - same physical scale
        max_range = 0.5  # default
        if all_x and all_y and all_z:
            max_range = max(max(all_x)-min(all_x), max(all_y)-min(all_y), max(all_z)-min(all_z)) / 2.0
            mid_x = (max(all_x) + min(all_x)) / 2.0
            mid_y = (max(all_y) + min(all_y)) / 2.0
            mid_z = (max(all_z) + min(all_z)) / 2.0
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Now draw cameras with direction lines (length = half of max_range)
        axis_len_m = max_range / 2.0
        for c_id, C, R in camera_data:
            # Plot Camera Center
            self.ax.scatter(C[0], C[1], C[2], c='black', s=100, marker='s', edgecolors='white', linewidths=2)
            
            # Text Label
            self.ax.text(C[0] + 0.02, C[1] + 0.02, C[2] + 0.02, f'Camera {c_id}', 
                        color='white', fontsize=10, fontweight='bold')
            
            # Direction line (length = half of range)
            z_dir = R.T @ np.array([0, 0, 1]) * axis_len_m
            z_dir = map_point(z_dir)
            end_pt = C + z_dir
            self.ax.plot3D([C[0], end_pt[0]], [C[1], end_pt[1]], [C[2], end_pt[2]], 
                           color='#FFFF00', linewidth=2)
        
        self.ax.set_box_aspect([1, 1, 1])
        
        self.canvas.draw()
        
    def plot_refractive(self, cameras, window_planes, points_3d=None):
        """Plot refractive calibration results (Cameras + Points + Window Planes)."""
        # Ensure cameras have 'R' and 'T' keys (convert from rvec/tvec if needed)
        import cv2
        converted_cams = {}
        for cid, params in cameras.items():
            # Handle both dict format and numpy array format (refractive calibrator)
            if isinstance(params, np.ndarray):
                # Array format: [rvec(3), tvec(3), f, cx, cy, k1, k2]
                arr = params.flatten()
                rvec = arr[0:3].reshape(3, 1)
                tvec = arr[3:6].reshape(3, 1)
                R, _ = cv2.Rodrigues(rvec)
                cam_data = {'R': R, 'T': tvec, 'rvec': rvec, 'tvec': tvec}
            elif isinstance(params, dict):
                cam_data = dict(params)  # Copy
                if 'R' not in cam_data and 'rvec' in cam_data:
                    rvec = np.array(cam_data['rvec']).reshape(3, 1)
                    R, _ = cv2.Rodrigues(rvec)
                    cam_data['R'] = R
                if 'T' not in cam_data and 'tvec' in cam_data:
                    cam_data['T'] = np.array(cam_data['tvec']).reshape(3, 1)
            else:
                # Unknown format, skip
                continue
            converted_cams[cid] = cam_data

        
        # First plot standard elements (cameras + points)
        self.plot_calibration(converted_cams, points_3d, axis_map='y_up')
        
        # Now add window planes (positioned between cameras and 3D points)
        scale = 1000.0  # mm -> m
        
        try:
            # Calculate bounding box of 3D points and cameras (in meters)
            all_pts = []
            cam_positions = {}
            
            # Collect camera positions
            for cid, cam_data in converted_cams.items():
                if 'R' in cam_data and 'T' in cam_data:
                    R = cam_data['R']
                    t = cam_data['T']
                    C = -R.T @ t  # Camera center in world coords
                    C = C.flatten() / scale
                    cam_positions[cid] = C
                    all_pts.append(C)
            
            # Collect 3D point positions
            if points_3d is not None and len(points_3d) > 0:
                pts_m = points_3d / scale
                pts_center = np.mean(pts_m, axis=0)
                pts_min = np.min(pts_m, axis=0)
                pts_max = np.max(pts_m, axis=0)
                all_pts.append(pts_min)
                all_pts.append(pts_max)
            else:
                pts_center = np.zeros(3)
            
            # Reference point (mean of points if available, otherwise origin)
            if points_3d is not None and len(points_3d) > 0:
                pts_m = points_3d / scale
                x_ref = np.mean(pts_m, axis=0)
            else:
                x_ref = np.zeros(3)

            # Overall bounding box from cameras + points (or origin fallback)
            bbox_sources = []
            if cam_positions:
                bbox_sources.extend(cam_positions.values())
            if points_3d is not None and len(points_3d) > 0:
                bbox_sources.extend((points_3d / scale).tolist())
            else:
                bbox_sources.append(x_ref)

            if bbox_sources:
                bbox_sources = np.array(bbox_sources)
                overall_min = np.min(bbox_sources, axis=0)
                overall_max = np.max(bbox_sources, axis=0)
                diag = np.linalg.norm(overall_max - overall_min)
                pad = 0.1 * diag if diag > 1e-9 else 0.1
                overall_min = overall_min - pad
                overall_max = overall_max + pad
            else:
                overall_min = np.array([-0.5, -0.5, -0.5])
                overall_max = np.array([0.5, 0.5, 0.5])

            # AABB vertices (0..7) and edges
            x0, y0, z0 = overall_min
            x1, y1, z1 = overall_max
            bbox_pts = np.array([
                [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1]
            ])
            bbox_edges = [
                (0, 1), (1, 3), (3, 2), (2, 0),
                (4, 5), (5, 7), (7, 6), (6, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]

            def _unique_points(points, eps=1e-6):
                uniq = []
                for p in points:
                    if not any(np.linalg.norm(p - q) < eps for q in uniq):
                        uniq.append(p)
                return uniq

            def _plane_aabb_intersections(n, p_ref):
                pts = []
                for i, j in bbox_edges:
                    p0 = bbox_pts[i]
                    p1 = bbox_pts[j]
                    s0 = np.dot(n, p0 - p_ref)
                    s1 = np.dot(n, p1 - p_ref)

                    if abs(s0) < 1e-9:
                        pts.append(p0)
                    if abs(s1) < 1e-9:
                        pts.append(p1)

                    if s0 * s1 < 0:
                        denom = s1 - s0
                        if abs(denom) < 1e-12:
                            continue
                        t = -s0 / denom
                        if 0.0 <= t <= 1.0:
                            pts.append(p0 + t * (p1 - p0))
                return _unique_points(pts)

            def _sort_points_on_plane(points, n):
                if abs(n[2]) > 0.9:
                    temp = np.array([1.0, 0.0, 0.0])
                else:
                    temp = np.array([0.0, 0.0, 1.0])
                u = np.cross(n, temp)
                u = u / (np.linalg.norm(u) + 1e-12)
                v = np.cross(n, u)
                v = v / (np.linalg.norm(v) + 1e-12)

                pts = np.array(points)
                proj = np.stack([pts @ u, pts @ v], axis=1)
                center = np.mean(proj, axis=0)
                angles = np.arctan2(proj[:, 1] - center[1], proj[:, 0] - center[0])
                order = np.argsort(angles)
                return pts[order]

            def _clip_polygon_by_plane(poly, n, p_ref, eps=1e-9):
                if poly is None or len(poly) == 0:
                    return []
                clipped = []
                num = len(poly)
                for i in range(num):
                    curr = poly[i]
                    prev = poly[i - 1]
                    d_curr = np.dot(n, curr - p_ref)
                    d_prev = np.dot(n, prev - p_ref)
                    curr_in = d_curr <= eps
                    prev_in = d_prev <= eps

                    if curr_in and prev_in:
                        clipped.append(curr)
                    elif prev_in and not curr_in:
                        t = d_prev / (d_prev - d_curr + eps)
                        clipped.append(prev + t * (curr - prev))
                    elif not prev_in and curr_in:
                        t = d_prev / (d_prev - d_curr + eps)
                        clipped.append(prev + t * (curr - prev))
                        clipped.append(curr)
                return _unique_points(clipped, eps=1e-6)

            plane_defs = []
            for wid, plane in window_planes.items():
                if 'plane_n' not in plane or 'plane_pt' not in plane:
                    continue

                n = np.array(plane['plane_n'], dtype=float)
                p_ref = np.array(plane['plane_pt'], dtype=float) / scale

                norm_n = np.linalg.norm(n)
                if norm_n < 1e-6:
                    continue
                n = n / norm_n
                if np.dot(n, x_ref - p_ref) > 0:
                    n = -n

                plane_defs.append((wid, n, p_ref))

            for wid, n, p_ref in plane_defs:
                inter_pts = _plane_aabb_intersections(n, p_ref)
                if len(inter_pts) < 3:
                    continue

                poly_pts = _sort_points_on_plane(inter_pts, n)

                for owid, on, op_ref in plane_defs:
                    if owid == wid:
                        continue
                    poly_pts = _clip_polygon_by_plane(poly_pts, on, op_ref)
                    if len(poly_pts) < 3:
                        break

                if len(poly_pts) < 3:
                    continue

                poly_pts_viz = np.array(poly_pts)[:, [0, 2, 1]]
                poly = Poly3DCollection([poly_pts_viz], alpha=0.3, facecolors='#00d4ff',
                                        edgecolors='#00d4ff', linewidths=0.5)
                self.ax.add_collection3d(poly)

                label_pt = np.mean(poly_pts, axis=0)
                label_pt = np.array([label_pt[0], label_pt[2], label_pt[1]])
                self.ax.text(label_pt[0], label_pt[1], label_pt[2],
                             f"Win {wid}", color='#00d4ff', fontsize=8)
            
            self.canvas.draw()
            
        except Exception as e:
            import traceback
            print(f"[Viz] Error plotting windows: {e}")
            print(traceback.format_exc())





class CameraCalibrationView(QWidget):
# ... (skip to create_wand_tab_v2)

    def create_wand_tab_v2(self):
        """Create the Wand Calibration tab (Multi-Camera) - Tabbed Interface."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 1. Visualization (LEFT)
        vis_frame = QFrame()
        vis_frame.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.setContentsMargins(0,0,0,0)

        self.vis_tabs = QTabWidget()
        self.vis_tabs.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QTabBar::tab { background: #222; color: #888; padding: 5px; }
            QTabBar::tab:selected { background: #444; color: #fff; }
        """)
        
        # Tab 1: 2D Detection Images
        vis_2d_widget = QWidget()
        vis_2d_layout = QVBoxLayout(vis_2d_widget)
        vis_2d_layout.setContentsMargins(0,0,0,0)
        
        # Scroll Area for images
        vis_scroll = QScrollArea()
        vis_scroll.setWidgetResizable(True)
        vis_scroll_content = QWidget()
        self.vis_grid_layout = QFormLayout(vis_scroll_content) # Or Grid? Form is Vertical.
        vis_scroll.setWidget(vis_scroll_content)
        vis_2d_layout.addWidget(vis_scroll)
        
        self.vis_tabs.addTab(vis_2d_widget, "2D Detection")
        
        # Tab 2: 3D Visualization
        self.calib_3d_view = Calibration3DViewer()
        self.vis_tabs.addTab(self.calib_3d_view, "3D View")
        
        vis_layout.addWidget(self.vis_tabs)
        
        self.cam_vis_labels = {} # Will be populated dynamically in _update_wand_table? 
        # Or I should add initial labels here?
        # Let's keep it empty and rely on dynamic update if exists, or just leave as is.
        # But wait, `_detect_single_frame` uses `self.cam_vis_labels`.
        # I need to ensure they are created.
        # I'll create 4 default labels.
        for i in range(4):
            lbl = QLabel(f"Cam {i} (No Image)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background: #111; border: 1px dashed #333; color: #555;")
            lbl.setMinimumHeight(200)
            self.vis_grid_layout.addRow(f"Cam {i}:", lbl)
            self.cam_vis_labels[i] = lbl
        
        # 2. Controls Panel (RIGHT)
        right_panel = QWidget()
        right_panel.setFixedWidth(370)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Style for Tabs - Black Background
        controls_tabs = QTabWidget()
        controls_tabs.setStyleSheet("""
             QTabWidget::pane { border: 1px solid #444; background: #000000; }
             QTabBar::tab { background: #333; color: #aaa; padding: 8px; min-width: 100px; }
             QTabBar::tab:selected { background: #444; color: #fff; border-bottom: 2px solid #00d4ff; }
        """)

        # Common Button Style
        btn_style = """
            QPushButton {
                background-color: #2a3f5f; 
                color: white; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 8px;
                font-size: 13px;
                min-height: 25px;
            }
            QPushButton:hover { background-color: #3b5278; }
            QPushButton:pressed { background-color: #1e2d42; }
        """
        
        btn_style_primary = """
            QPushButton {
                background-color: #00d4ff; 
                color: black; 
                border: 1px solid #00a0cc; 
                border-radius: 4px; 
                padding: 8px;
                font-weight: bold;
                font-size: 13px;
                min-height: 25px;
            }
            QPushButton:hover { background-color: #66e5ff; }
            QPushButton:pressed { background-color: #008fb3; }
        """

        # --- Tab 1: Detection (with Scroll Area) ---
        det_scroll = QScrollArea()
        det_scroll.setWidgetResizable(True)
        det_scroll.setFrameShape(QFrame.Shape.NoFrame)
        det_scroll.setStyleSheet("background-color: transparent;")
        
        det_content = QWidget()
        det_layout = QVBoxLayout(det_content)
        det_layout.setSpacing(10)
        det_layout.setContentsMargins(10, 10, 10, 10)
        
        # Conf Group
        conf_group = QGroupBox("Detection Settings")
        conf_layout = QFormLayout(conf_group)
        
        self.wand_num_cams = QSpinBox()
        self._apply_input_style(self.wand_num_cams)
        self.wand_num_cams.setValue(4)
        self.wand_num_cams.setRange(1, 16)
        self.wand_num_cams.valueChanged.connect(self._update_wand_table)
        
        self.wand_type_combo = QComboBox()
        self._apply_input_style(self.wand_type_combo)
        self.wand_type_combo.addItems(["Dark on Bright", "Bright on Dark"])
        
        self.radius_range = RangeSlider(min_val=1, max_val=200, initial_min=20, initial_max=200, suffix=" px")
        
        from .widgets import SimpleSlider
        self.sensitivity_slider = SimpleSlider(min_val=0.5, max_val=1.0, initial=0.85, decimals=2)
        
        conf_layout.addRow("Num Cameras:", self.wand_num_cams)
        conf_layout.addRow("Wand Type:", self.wand_type_combo)
        conf_layout.addRow("Radius Range:", self.radius_range)
        conf_layout.addRow("Sensitivity:", self.sensitivity_slider)
        det_layout.addWidget(conf_group)
        
        # Table (with per-camera focal length and image size)
        det_layout.addWidget(QLabel("Camera Images:"))
        self.wand_cam_table = QTableWidget()
        self.wand_cam_table.setColumnCount(5)
        self.wand_cam_table.setHorizontalHeaderLabels(["", "Cam ID", "Focal (px)", "Width", "Height"])
        
        # Add Tooltip for Focal Length
        # Need to access the QTableWidgetItem for the header
        focal_header_item = self.wand_cam_table.horizontalHeaderItem(2)
        if focal_header_item:
            focal_header_item.setToolTip("Focal Length (px) = Focal Length (mm) / Sensor Pixel Size (mm)")

        header = self.wand_cam_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.wand_cam_table.verticalHeader().setVisible(False)
        self.wand_cam_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.wand_cam_table.setShowGrid(True)
        self.wand_cam_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.wand_cam_table.setMinimumHeight(130) 
        self._update_wand_table(4)
        det_layout.addWidget(self.wand_cam_table)
        
        # Frame List
        det_layout.addWidget(QLabel("Frame List (Click to Preview):"))
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(2)
        self.frame_table.setHorizontalHeaderLabels(["Index", "Filename"])
        self.frame_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.frame_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.frame_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.frame_table.cellClicked.connect(self._on_frame_table_clicked)
        self.frame_table.setFixedHeight(120) 
        det_layout.addWidget(self.frame_table)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(14)
        mode_row.addWidget(QLabel("Detection Mode:"))

        self.detect_mode_group = QButtonGroup(self)
        self.detect_mode_fast = QRadioButton("fast")
        self.detect_mode_reliable = QRadioButton("reliable")
        self.detect_mode_fast.setChecked(True)

        detect_mode_radio_style = """
            QRadioButton {
                color: #eaeaea;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 8px;
                height: 8px;
                border: 1px solid #ffffff;
                background: transparent;
                border-radius: 5px;
                image: none;
            }
            QRadioButton::indicator:unchecked {
                background: transparent;
            }
            QRadioButton::indicator:checked {
                border: 1px solid #ffffff;
                background: #00d26a;
            }
        """
        self.detect_mode_fast.setStyleSheet(detect_mode_radio_style)
        self.detect_mode_reliable.setStyleSheet(detect_mode_radio_style)

        self.detect_mode_fast.setToolTip(
            "fast but less accurate, recommended for good quality image without distortion."
        )
        self.detect_mode_reliable.setToolTip(
            "reliable but slower, recommended for camera system with refraction."
        )

        self.detect_mode_group.addButton(self.detect_mode_fast)
        self.detect_mode_group.addButton(self.detect_mode_reliable)
        mode_row.addWidget(self.detect_mode_fast)
        mode_row.addWidget(self.detect_mode_reliable)
        mode_row.addStretch()
        det_layout.addLayout(mode_row)

        # Actions
        self.btn_detect_single = QPushButton("Test Detect (Current Frame)")
        self.btn_detect_single.setStyleSheet(btn_style)
        self.btn_detect_single.clicked.connect(self._detect_single_frame)
        det_layout.addWidget(self.btn_detect_single)

        self.btn_process_wand = QPushButton("Process All Frames / Resume")
        self.btn_process_wand.setStyleSheet(btn_style)
        self.btn_process_wand.clicked.connect(self._process_wand_frames)
        det_layout.addWidget(self.btn_process_wand)
        
        det_layout.addStretch()
        
        det_scroll.setWidget(det_content)
        det_tab = QWidget()
        det_tab_layout = QVBoxLayout(det_tab)
        det_tab_layout.setContentsMargins(0, 0, 0, 0)
        det_tab_layout.addWidget(det_scroll)

        # --- Tab 2: Calibration (with scroll area) ---
        cal_tab = QWidget()
        cal_scroll = QScrollArea()
        cal_scroll.setWidgetResizable(True)
        cal_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        cal_content = QWidget()
        cal_layout = QVBoxLayout(cal_content)
        cal_layout.setSpacing(15)
        cal_layout.setContentsMargins(10, 10, 10, 10)
        
        cal_group = QGroupBox("Calibration Settings")
        cal_form = QFormLayout(cal_group)
        
        self.wand_model_combo = QComboBox()
        self._apply_input_style(self.wand_model_combo)
        self.wand_model_combo.addItems(["Pinhole", "Pinhole+Refraction"])
        self.wand_model_combo.currentIndexChanged.connect(self._on_model_changed)
        
        self.wand_len_spin = QDoubleSpinBox()
        self._apply_input_style(self.wand_len_spin)
        self.wand_len_spin.setValue(10.0)
        self.wand_len_spin.setRange(1, 5000)
        self.wand_len_spin.setSuffix(" mm")
        
        # Distortion Model Selection (None, k1, k1+k2)
        self.dist_model_combo = QComboBox()
        self._apply_input_style(self.dist_model_combo)
        self.dist_model_combo.addItems(["None (0)", "Radial k1 (1)", "Radial k1+k2 (2)"])
        self.dist_model_combo.setCurrentIndex(0)

        cal_form.addRow("Camera Model:", self.wand_model_combo)
        cal_form.addRow("Wand Length:", self.wand_len_spin)
        cal_form.addRow("Distortion params:", self.dist_model_combo)
        cal_layout.addWidget(cal_group)

        # --- Refraction Settings (shown only if Pinhole+Refraction) ---
        self.refraction_group = QGroupBox("Refraction Settings")
        self.refraction_group.setStyleSheet(self.GROUP_BOX_STYLE)
        ref_layout = QVBoxLayout(self.refraction_group)
        ref_layout.setContentsMargins(5, 15, 5, 5)
        
        # Window Count and Camera-Window Mapping
        ref_top_layout = QFormLayout()
        self.window_count_spin = QSpinBox()
        self.window_count_spin.setRange(1, 10)
        self.window_count_spin.setValue(1)
        self._apply_input_style(self.window_count_spin)
        self.window_count_spin.valueChanged.connect(self._on_window_count_changed)
        ref_top_layout.addRow("Number of Windows:", self.window_count_spin)
        ref_layout.addLayout(ref_top_layout)
        
        ref_layout.addWidget(QLabel("Camera-Window Mapping:"))
        self.cam_window_table = QTableWidget()
        self.cam_window_table.setColumnCount(2)
        self.cam_window_table.setHorizontalHeaderLabels(["Camera ID", "Window"])
        self.cam_window_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cam_window_table.verticalHeader().setVisible(False)
        self.cam_window_table.setFixedHeight(120)
        self.cam_window_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        ref_layout.addWidget(self.cam_window_table)
        
        ref_layout.addSpacing(10)
        ref_layout.addWidget(QLabel("Window Configuration (for selected type):"))

        # Diagrammatic Layout (QHBoxLayout)
        diagram_layout = QHBoxLayout()
        diagram_layout.setSpacing(2)

        # Segment 1: Camera Side (Air)
        seg1 = QWidget()
        seg1_layout = QVBoxLayout(seg1)
        seg1_layout.setContentsMargins(2, 2, 2, 2)
        seg1_layout.addWidget(QLabel("Camera side"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.media1_combo = QComboBox()
        self.media1_combo.addItems(["Air", "Other"])
        self.media1_index = QDoubleSpinBox()
        self.media1_index.setRange(1.0, 2.0)
        self.media1_index.setValue(1.0)
        self._apply_input_style(self.media1_combo)
        self._apply_input_style(self.media1_index)
        seg1_layout.addWidget(self.media1_combo)
        seg1_layout.addWidget(self.media1_index)
        diagram_layout.addWidget(seg1)

        # Segment 2: Window (Acrylic/Glass)
        seg2 = QFrame()
        # Blue vertical bar style: bluer background, left and right borders only
        seg2.setStyleSheet("""
            background-color: rgba(0, 170, 255, 60); 
            border-left: 2px solid #00aaff; 
            border-right: 2px solid #00aaff;
            border-top: none;
            border-bottom: none;
        """)
        seg2_layout = QVBoxLayout(seg2)
        seg2_layout.setContentsMargins(2, 2, 2, 2)
        
        lbl_window = QLabel("Window")
        lbl_window.setStyleSheet("border: none; background: transparent; font-weight: bold;")
        seg2_layout.addWidget(lbl_window, alignment=Qt.AlignmentFlag.AlignCenter)
        self.media2_combo = QComboBox()
        self.media2_combo.addItems(["Acrylic", "Glass", "Other"])
        self.media2_index = QDoubleSpinBox()
        self.media2_index.setRange(1.0, 2.5)
        self.media2_index.setValue(1.49)
        self.media2_thick = QDoubleSpinBox()
        self.media2_thick.setRange(0.1, 100.0)
        self.media2_thick.setValue(10.0)
        self.media2_thick.setSuffix(" mm")
        self._apply_input_style(self.media2_combo)
        self._apply_input_style(self.media2_index)
        self._apply_input_style(self.media2_thick)
        seg2_layout.addWidget(self.media2_combo)
        seg2_layout.addWidget(self.media2_index)
        
        lbl_thick = QLabel("Thickness:")
        lbl_thick.setStyleSheet("border: none; background: transparent;")
        seg2_layout.addWidget(lbl_thick, alignment=Qt.AlignmentFlag.AlignCenter)
        seg2_layout.addWidget(self.media2_thick)
        diagram_layout.addWidget(seg2)

        # Segment 3: Object Side (Water)
        seg3 = QWidget()
        seg3_layout = QVBoxLayout(seg3)
        seg3_layout.setContentsMargins(2, 2, 2, 2)
        seg3_layout.addWidget(QLabel("Object side"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.media3_combo = QComboBox()
        self.media3_combo.addItems(["Water", "Other"])
        self.media3_index = QDoubleSpinBox()
        self.media3_index.setRange(1.0, 2.0)
        self.media3_index.setValue(1.33)
        self._apply_input_style(self.media3_combo)
        self._apply_input_style(self.media3_index)
        seg3_layout.addWidget(self.media3_combo)
        seg3_layout.addWidget(self.media3_index)
        diagram_layout.addWidget(seg3)

        ref_layout.addLayout(diagram_layout)
        cal_layout.addWidget(self.refraction_group)
        self.refraction_group.setVisible(False) # Hidden by default

        # Connect Refraction Logic
        self.media1_combo.currentIndexChanged.connect(self._on_refraction_media_changed)
        self.media2_combo.currentIndexChanged.connect(self._on_refraction_media_changed)
        self.media3_combo.currentIndexChanged.connect(self._on_refraction_media_changed)
        self._on_refraction_media_changed() # Initialize read-only states
        
        # Initial population of refraction camera table
        self._update_refraction_cam_table(4)
        
        cal_layout.addStretch()
        
        # Load Points Button
        self.btn_load_points = QPushButton("Load Wand Points (from CSV)")
        self.btn_load_points.setStyleSheet(btn_style)
        self.btn_load_points.clicked.connect(self._load_wand_points_for_calibration)
        cal_layout.addWidget(self.btn_load_points)
        
        # Precalibrate Check
        self.btn_precalibrate = QPushButton("Precalibrate to Check")
        self.btn_precalibrate.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 10px;")
        self.btn_precalibrate.clicked.connect(lambda: self._run_wand_calibration(precalibrate=True))
        cal_layout.addWidget(self.btn_precalibrate)
        
        # Run Calibration
        self.btn_calibrate_wand = QPushButton("Run Calibration")
        self.btn_calibrate_wand.setStyleSheet(btn_style_primary)
        self.btn_calibrate_wand.clicked.connect(self._run_wand_calibration)
        cal_layout.addWidget(self.btn_calibrate_wand)
        
        # --- Error Analysis Section (shown after calibration) ---
        error_header_row = QHBoxLayout()
        error_header_row.addWidget(QLabel("Error Analysis:"))
        
        # Warning label for missing image paths (inline, hidden by default)
        self.error_warning_label = QLabel("")
        self.error_warning_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        self.error_warning_label.setVisible(False)
        error_header_row.addWidget(self.error_warning_label)

        
        error_header_row.addStretch()
        cal_layout.addLayout(error_header_row)
        
        # Error Table with horizontal scroll and sorting
        # Error Table with Frozen "Remove" Column
        error_table_container = QWidget()
        error_table_layout = QHBoxLayout(error_table_container)
        error_table_layout.setContentsMargins(0, 0, 0, 0)
        error_table_layout.setSpacing(0)
        
        # 1. Fixed "Frozen" Table (Left, Col 0 only)
        self.frozen_table = QTableWidget()
        self.frozen_table.setMinimumHeight(200)
        self.frozen_table.setStyleSheet("""
            QTableWidget { background-color: #1a1a2e; color: white; border: none; border-right: 1px solid #444; }
            QHeaderView::section { background-color: #2a2a4e; color: white; border: none; }
        """)
        self.frozen_table.setColumnCount(1)
        self.frozen_table.setHorizontalHeaderLabels(["Del"])
        self.frozen_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # Scroll controlled by right table
        self.frozen_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frozen_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.frozen_table.setFixedWidth(42) # Compact width for checkbox
        self.frozen_table.setColumnWidth(0, 42)
        self.frozen_table.verticalHeader().setVisible(False)
        self.frozen_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        
        # 2. Main Scrollable Table (Right, Cols 1..N)
        self.error_table = QTableWidget()
        self.error_table.setMinimumHeight(200)
        self.error_table.setStyleSheet("""
            QTableWidget { background-color: #1a1a2e; color: white; border: none; }
            QHeaderView::section { background-color: #2a2a4e; color: white; border: none; }
        """)
        self.error_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.error_table.verticalHeader().setVisible(False)
        self.error_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.error_table.cellClicked.connect(self._on_error_table_clicked)
        
        # Sync Vertical Scrollbars
        self.error_table.verticalScrollBar().valueChanged.connect(self.frozen_table.verticalScrollBar().setValue)
        self.frozen_table.verticalScrollBar().valueChanged.connect(self.error_table.verticalScrollBar().setValue)
        
        # Sync Sorting: when error_table is sorted, reorder frozen_table to match
        self.error_table.horizontalHeader().sortIndicatorChanged.connect(self._sync_frozen_table_sort)
        
        error_table_layout.addWidget(self.frozen_table)
        error_table_layout.addWidget(self.error_table)
        cal_layout.addWidget(error_table_container)
        
        # Batch Filter Controls (auto-update when changed)
        filter_row1 = QHBoxLayout()
        self.filter_proj_check = QCheckBox("Delete when proj error >")
        self.filter_proj_check.toggled.connect(self._auto_update_filter_marks)
        self.filter_proj_spin = QDoubleSpinBox()
        self.filter_proj_spin.setRange(0.1, 100)
        self.filter_proj_spin.setValue(5.0)
        self.filter_proj_spin.setSuffix(" px")
        self.filter_proj_spin.valueChanged.connect(self._auto_update_filter_marks)
        self._apply_input_style(self.filter_proj_spin)
        filter_row1.addWidget(self.filter_proj_check)
        filter_row1.addWidget(self.filter_proj_spin)
        filter_row1.addStretch()
        cal_layout.addLayout(filter_row1)
        
        filter_row2 = QHBoxLayout()
        self.filter_len_check = QCheckBox("Delete when wand len error >")
        self.filter_len_check.toggled.connect(self._auto_update_filter_marks)
        self.filter_len_spin = QDoubleSpinBox()
        self.filter_len_spin.setRange(0.01, 100)
        self.filter_len_spin.setValue(1.0)
        self.filter_len_spin.setSuffix(" mm")
        self.filter_len_spin.valueChanged.connect(self._auto_update_filter_marks)
        self._apply_input_style(self.filter_len_spin)
        filter_row2.addWidget(self.filter_len_check)
        filter_row2.addWidget(self.filter_len_spin)
        filter_row2.addStretch()
        cal_layout.addLayout(filter_row2)

        # Save Button (Relocated to bottom)
        save_row = QHBoxLayout()
        save_row.addStretch()
        self.btn_save_points = QPushButton("Save Filtered Points")
        # Make it slightly more prominent
        self.btn_save_points.setStyleSheet("background-color: #2a3f5f; color: white; font-size: 12px; padding: 6px 20px; border-radius: 4px;")
        self.btn_save_points.clicked.connect(self._save_filtered_points)
        save_row.addWidget(self.btn_save_points)
        cal_layout.addLayout(save_row)

        
        cal_layout.addStretch()
        
        # Setup scroll area and cal_tab layout
        cal_scroll.setWidget(cal_content)
        cal_tab_layout = QVBoxLayout(cal_tab)
        cal_tab_layout.setContentsMargins(0, 0, 0, 0)
        cal_tab_layout.addWidget(cal_scroll)

        # Add tabs
        controls_tabs.addTab(det_tab, "Point Detection")
        controls_tabs.addTab(cal_tab, "Calibration")
        
        # --- Tab 3: Tutorial ---
        tut_tab = QWidget()
        tut_layout = QVBoxLayout(tut_tab)
        tut_layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_info = QLabel("Need help with Wand Calibration?")
        lbl_info.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff;")
        lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tut_layout.addWidget(lbl_info)
        
        lbl_desc = QLabel("Click the button below to open the comprehensive step-by-step user guide in your browser.")
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tut_layout.addWidget(lbl_desc)
        
        tut_layout.addSpacing(20)
        
        self.btn_open_guide = QPushButton("Open User Guide")
        self.btn_open_guide.setStyleSheet(btn_style_primary)
        self.btn_open_guide.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_guide.clicked.connect(self._open_user_guide)
        tut_layout.addWidget(self.btn_open_guide)
        
        tut_layout.addStretch()
        controls_tabs.addTab(tut_tab, "Tutorial")
        
        right_layout.addWidget(controls_tabs)
        
        # Progress Label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 11px; margin-top: 5px;")
        right_layout.addWidget(self.status_label)

        layout.addWidget(vis_frame, stretch=2)
        layout.addWidget(right_panel)
        
        return tab

    def _open_user_guide(self):
        """Open the HTML user guide in the default browser."""
        import os
        import webbrowser
        from pathlib import Path
        
        # Assume the HTML file is in the same directory as this script
        current_dir = Path(__file__).parent
        guide_path = current_dir / "wand_calibration" / "WAND_CALIBRATION_USER_GUIDE.html"
        
        if guide_path.exists():
            webbrowser.open(guide_path.as_uri())
        else:
            QMessageBox.warning(self, "Guide Not Found", f"Could not find user guide at:\n{guide_path}")

    def _load_wand_points_for_calibration(self):
        """Prompt to load a CSV file, populate wand points, and ready for calibration."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Wand Points", "", "CSV Files (*.csv)")
        if not file_path:
            return
            
        success, msg = self.wand_calibrator.load_wand_data_from_csv(file_path)
        if success:
            self.error_table.setRowCount(0)
            self._update_3d_viz()
            # Ensure calibrator has access to cameras/size
            if self.wand_images:
                 self.wand_calibrator.cameras = {}
                 for c, imgs in self.wand_images.items():
                     self.wand_calibrator.cameras[c] = {'images': imgs}
                     
            QMessageBox.information(self, "Success", msg + "\nand make sure image size is inputed.")
            count_frames = len(self.wand_calibrator.wand_points)
            self.status_label.setText(f"Loaded {count_frames} frames. Ready to calibrate.")
        else:
            QMessageBox.critical(self, "Error", f"Failed to load points:\n{msg}")
    def __init__(self):
        super().__init__()
        self._busy_tokens = {}
        
        # Style Constants
        self.GROUP_BOX_STYLE = """
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
        self.INDEXING_BTN_STYLE = "background-color: #2a2a2a; border: 1px solid #555; color: #fff; padding: 6px; font-weight: bold;"
        
        self.setup_ui()

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

    def setup_ui(self):
        # Initialize data structures
        self.plate_images = [] # List of absolute paths
        self.wand_images = {}  # Dict {cam_idx: [paths]}
        self.wand_calibrator = WandCalibrator() # Init calibrator
        self._refr_has_result = False
        self._refr_params_dirty = False
        self._refr_final_cam_params = None
        self._refr_window_planes = None
        self._refr_cam_to_window = None
        self._refr_window_media = None
        self._refr_proj_err_stats = {}
        self._refr_tri_err_stats = {}
        self._refr_use_proj_residuals = False
        self.plate_cam_labels = {} # Map {cam_idx: ZoomableImageLabel}
        self.plate_3d_viewer = Calibration3DViewer() # Init 3D viewer for plate
        self.all_camera_params = {} # Accumulate calibrated camera params {cam_idx: {...}}
        self.plate_refraction_settings = {}  # Per-camera UI-only refraction settings
        self.plate_intrinsics_settings = {}  # Per-camera intrinsic UI settings for refraction mode
        self.plate_image_size_hints = {}  # {cam_idx: (width, height)} from loaded plate images
        self._plate_intrinsics_autofilled_once = set()  # cam_idx set; do not auto-overwrite after first fill
        
        # New: Tracking for indexed points across images
        self.saved_calibration_data = {} # {(cam_idx, path): {'keypoints': [], 'indices': [], 'plane': int}}
        self.plane_num_offset = 0
        self._is_manually_changing_plane = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Title
        title = QLabel("Camera Calibration")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4ff; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Force input field styles for this view (Style fix)
        self.setStyleSheet("""
             QComboBox::drop-down { border: none; }
             QToolTip {
                 background-color: #101722;
                 color: #dbe9ff;
                 border: none;
                 padding: 6px 8px;
                 border-radius: 4px;
             }
        """)

        # Main Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_plate_tab(), "Plate Calibration")
        self.tabs.addTab(self.create_wand_tab_v2(), "Wand Calibration")
        layout.addWidget(self.tabs)

    def _apply_input_style(self, widget):
        """Force style on input widgets to ensure background color."""
        # Note: We apply this to specific widgets to override any parent transparency issues
        widget.setStyleSheet("""
            background-color: #2d3a4a;
            border: 1px solid #3d4a5a;
            border-radius: 6px;
            padding: 6px 10px;
            color: #eaeaea;
            selection-background-color: #0f3460;
            max-width: 140px;
        """)
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setStyleSheet(widget.styleSheet() + """
                QSpinBox::up-button, QSpinBox::down-button,
                QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                    width: 0px; height: 0px;
                }
            """)
        if isinstance(widget, QComboBox):
             widget.setStyleSheet(widget.styleSheet() + """
                QComboBox::drop-down { border: none; }
             """)

    def create_plate_tab(self):
        """Create the Plate Calibration tab (Single/Multi Camera) - Refactored Layout."""
        container = QWidget()
        layout = QHBoxLayout(container)
        
        # 1. Visualization (LEFT) - Wrapped in Frame for Border
        vis_frame = QFrame()
        vis_frame.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.setContentsMargins(0,0,0,0)

        self.plate_vis_tabs = QTabWidget()
        self.plate_vis_tabs.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QTabBar::tab { background: #222; color: #888; padding: 5px; }
            QTabBar::tab:selected { background: #444; color: #fff; }
        """)
        vis_layout.addWidget(self.plate_vis_tabs)
        self.plate_vis_tabs.addTab(self.plate_3d_viewer, "3D View")
        layout.addWidget(vis_frame, stretch=2)
        
        # 2. Controls (RIGHT) - Tabbed
        self.plate_ctrl_tabs = QTabWidget()
        self.plate_ctrl_tabs.setFixedWidth(370) # Match Wand Tab Width
        self.plate_ctrl_tabs.setStyleSheet("""
             QTabWidget::pane { border: 1px solid #444; background: #000000; }
             QTabBar::tab { background: #222; color: #aaa; padding: 8px; min-width: 100px; }
             QTabBar::tab:selected { background: #000000; color: #fff; border-top: 2px solid #00d4ff; border-bottom: 0px; font-weight: bold; }
        """) 
        # Note: Wand screenshot seems to have Top Blue Border for active tab and Black background

        
        # Styles
        btn_style = """
            QPushButton {
                background-color: #2a3f5f; 
                color: white; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 6px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #3b5278; }
            QPushButton:pressed { background-color: #1e2d42; }
        """
        
        btn_style_primary = """
            QPushButton {
                background-color: #00d4ff; 
                color: black; 
                border: 1px solid #00a0cc; 
                border-radius: 4px; 
                padding: 8px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #33e0ff; }
            QPushButton:pressed { background-color: #008cb3; }
        """
        
        # --- CONTROL TAB 1: Detection ---
        det_tab = QWidget()
        det_layout_wrap = QVBoxLayout(det_tab)
        det_layout_wrap.setContentsMargins(0,0,0,0)
        
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)  # Remove frame like Wand
        controls_scroll.setStyleSheet("background-color: transparent;")  # Match Wand
        
        controls = QWidget()
        controls.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        controls.setStyleSheet("background-color: #000000;") # Pure Black
        
        det_layout = QVBoxLayout(controls)
        det_layout.setContentsMargins(10, 10, 10, 10)  # Match Wand Calibration margins
        det_layout.setSpacing(15)
        
        # 1. Camera Images Group (Merged Settings + Images)
        img_group = QGroupBox("Camera Images")
        img_group.setStyleSheet(self.GROUP_BOX_STYLE)
        img_group.setMaximumWidth(345)  # Fit within 370px panel minus margins
        img_layout = QVBoxLayout(img_group)
        img_layout.setSpacing(10)
        
        # Upper Section: Camera Controls
        cam_ctrl_layout = QFormLayout()
        cam_ctrl_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Num Cameras
        self.plate_num_cams = QSpinBox()
        self._apply_input_style(self.plate_num_cams)
        self.plate_num_cams.setRange(1, 16)
        self.plate_num_cams.setValue(4)
        self.plate_num_cams.setMaximumWidth(200)  # Constrain width
        self.plate_num_cams.valueChanged.connect(lambda v: self._update_cam_list(v))
        cam_ctrl_layout.addRow("Num Cameras:", self.plate_num_cams)
        
        # Target Camera
        self.plate_cam_combo = QComboBox()
        self._apply_input_style(self.plate_cam_combo)
        self.plate_cam_combo.setMaximumWidth(200)  # Constrain width
        self.plate_cam_combo.currentIndexChanged.connect(self._sync_vis_tab)
        cam_ctrl_layout.addRow("Target Camera:", self.plate_cam_combo)
        
        # Detection Settings Group - Moved outside Camera Images
        self.adv_group = QGroupBox("Detection Settings")
        self.adv_group.setStyleSheet(self.GROUP_BOX_STYLE)
        
        adv_det_layout = QFormLayout()
        
        # 1. Point Color
        self.plate_color_combo = QComboBox()
        self._apply_input_style(self.plate_color_combo)
        self.plate_color_combo.setMaximumWidth(200)  # Constrain width
        self.plate_color_combo.addItems(["Bright (Standard)", "Dark"])
        adv_det_layout.addRow("Point Color:", self.plate_color_combo)
        
        # 2. Select Template Button
        btn_layout = QHBoxLayout()
        self.btn_select_template = QPushButton("Select Template")
        self.btn_select_template.clicked.connect(self._start_template_selection)
        self.btn_select_template.setStyleSheet("background-color: #555; color: #fff; padding: 5px;")
        self.btn_select_template.setMaximumWidth(130)  # Constrain width
        btn_layout.addWidget(self.btn_select_template)
        
        # 3. Select ROI Button
        self.btn_select_roi = QPushButton("Select Search ROI")
        self.btn_select_roi.clicked.connect(self._start_roi_selection)
        self.btn_select_roi.setStyleSheet("background-color: #555; color: #fff; padding: 5px;")
        self.btn_select_roi.setMaximumWidth(130)  # Constrain width
        btn_layout.addWidget(self.btn_select_roi)
        
        adv_det_layout.addRow(btn_layout)
        
        # 4. Status & Preview
        self.lbl_template_status = QLabel("Ready")
        self.lbl_template_status.setStyleSheet("color: #ff5555;")
        self.lbl_template_status.setMaximumWidth(150)  # Prevent layout expansion
        self.lbl_template_status.setWordWrap(True)
        adv_det_layout.addRow("Template:", self.lbl_template_status)
        
        self.lbl_template_preview = QLabel()
        self.lbl_template_preview.setFixedSize(80, 80)
        self.lbl_template_preview.setStyleSheet("border: 1px solid #555; background-color: #000;")
        adv_det_layout.addRow("Preview:", self.lbl_template_preview)
        
        self.lbl_roi_status = QLabel("Region: Full Image")
        self.lbl_roi_status.setStyleSheet("color: #aaa;")
        adv_det_layout.addRow("ROI:", self.lbl_roi_status)
        
        # 5. Threshold Slider
        from .widgets import SimpleSlider
        self.slider_match_thresh = SimpleSlider(min_val=0.1, max_val=1.0, initial=0.7, decimals=2)
        adv_det_layout.addRow("Match Threshold:", self.slider_match_thresh)
        
        # 6. Action Buttons Row
        action_btn_layout = QHBoxLayout()
        
        self.btn_detect = QPushButton("Detect")
        self.btn_detect.setStyleSheet("background-color: #2d6a4f; color: #fff; padding: 5px; font-weight: bold;")
        self.btn_detect.clicked.connect(self._run_detection_and_show)
        action_btn_layout.addWidget(self.btn_detect)
        
        self.btn_remove_points = QPushButton("Remove")
        self.btn_remove_points.setStyleSheet("background-color: #9d0208; color: #fff; padding: 5px;")
        self.btn_remove_points.clicked.connect(self._start_remove_mode)
        action_btn_layout.addWidget(self.btn_remove_points)
        
        self.btn_add_points = QPushButton("Add")
        self.btn_add_points.setStyleSheet("background-color: #3a86ff; color: #fff; padding: 5px;")
        self.btn_add_points.clicked.connect(self._start_add_mode)
        action_btn_layout.addWidget(self.btn_add_points)
        
        adv_det_layout.addRow(action_btn_layout)
        
        # 7. Points count status
        self.lbl_points_count = QLabel("Points: 0")
        self.lbl_points_count.setStyleSheet("color: #00ff00;")
        adv_det_layout.addRow("Detected:", self.lbl_points_count)
        
        self.adv_group.setLayout(adv_det_layout)
        # cam_ctrl_layout.addWidget(self.adv_group) # Removed - will add to main det_layout instead
        
        img_layout.addLayout(cam_ctrl_layout)
        
        # Init State
        self.current_template = None
        self.template_offset = (0.0, 0.0)
        self.search_roi_points = []
        self.detected_keypoints = []  # Store detected keypoints for modification
        
        # Middle Section: Buttons
        btn_row = QHBoxLayout()
        self.btn_load_plate = QPushButton("Open Files")
        self.btn_load_plate.setStyleSheet(btn_style)
        self.btn_load_plate.clicked.connect(self._load_plate_images)
        
        self.btn_clear_plate = QPushButton("Clear")
        self.btn_clear_plate.setStyleSheet(btn_style)
        self.btn_clear_plate.clicked.connect(self._clear_plate_images)
        
        btn_row.addWidget(self.btn_load_plate)
        btn_row.addWidget(self.btn_clear_plate)
        img_layout.addLayout(btn_row)
        
        # Lower Section: List
        img_layout.addWidget(QLabel("Frame List (Click to Preview):"))
        self.plate_img_list = QListWidget()
        self.plate_img_list.setStyleSheet("background-color: #111; color: #aaa; border: 1px solid #333;")
        self.plate_img_list.setFixedHeight(80) # Reduced height (approx 3 rows)
        self.plate_img_list.currentRowChanged.connect(self._display_plate_image)
        img_layout.addWidget(self.plate_img_list)
        
        det_layout.addWidget(img_group)
        # Detection Settings Group (moved here from inside Camera Images)
        det_layout.addWidget(self.adv_group)
        
        # 4. Indexing Group
        indexing_group = QGroupBox("Indexing")
        indexing_group.setStyleSheet(self.GROUP_BOX_STYLE)
        indexing_group.setMaximumWidth(345)  # Fit within 370px panel
        indexing_layout = QFormLayout()
        indexing_layout.setContentsMargins(8, 20, 8, 10)  # Reduced horizontal margins
        
        # Set Origin button
        self.btn_set_origin = QPushButton("Set Origin")
        self.btn_set_origin.setStyleSheet(self.INDEXING_BTN_STYLE)
        self.btn_set_origin.clicked.connect(self._start_origin_selection)
        indexing_layout.addRow(self.btn_set_origin)
        
        # Fixed Axis Checkboxes (mutually exclusive) with single Plane Number input
        axis_row = QHBoxLayout()
        
        self.chk_fix_x = QCheckBox("X")
        self.chk_fix_y = QCheckBox("Y") 
        self.chk_fix_z = QCheckBox("Z")
        self.chk_fix_z.setChecked(True)  # Default: Z is fixed (2D plate)
        
        for chk in [self.chk_fix_x, self.chk_fix_y, self.chk_fix_z]:
            chk.setStyleSheet("color: #fff;")
        
        # Make checkboxes mutually exclusive
        self.chk_fix_x.toggled.connect(lambda checked: self._on_axis_check_toggled('x', checked))
        self.chk_fix_y.toggled.connect(lambda checked: self._on_axis_check_toggled('y', checked))
        self.chk_fix_z.toggled.connect(lambda checked: self._on_axis_check_toggled('z', checked))
        
        axis_row.addWidget(self.chk_fix_x)
        axis_row.addWidget(self.chk_fix_y)
        axis_row.addWidget(self.chk_fix_z)
        
        # Single plane number spinbox
        plane_label = QLabel("Plane:")
        plane_label.setStyleSheet("color: #aaa;")
        axis_row.addWidget(plane_label)
        self.spin_plane_num = QSpinBox()
        self._apply_input_style(self.spin_plane_num)
        self.spin_plane_num.setRange(-1000, 10000)
        self.spin_plane_num.setValue(0)
        self.spin_plane_num.setFixedWidth(60)
        self.spin_plane_num.valueChanged.connect(self._on_plane_num_manually_changed)
        axis_row.addWidget(self.spin_plane_num)
        axis_row.addStretch()
        
        indexing_layout.addRow("Fixed Axis:", axis_row)
        
        # Set Axes Direction button
        self.btn_set_axes = QPushButton("Set Axis Directions")
        self.btn_set_axes.setStyleSheet(self.INDEXING_BTN_STYLE)
        self.btn_set_axes.clicked.connect(self._start_axes_selection)
        indexing_layout.addRow(self.btn_set_axes)
        
        # Run Indexing button
        self.btn_index_points = QPushButton("Index Points")
        self.btn_index_points.setStyleSheet("background-color: #2d6a4f; color: #fff; padding: 6px; font-weight: bold; margin-top: 5px;")
        self.btn_index_points.clicked.connect(self._run_indexing)
        indexing_layout.addRow(self.btn_index_points)

        # Verification/Sparse settings
        viz_row = QHBoxLayout()
        viz_row.addWidget(QLabel("Step:"))
        self.spin_index_step = QSpinBox()
        self._apply_input_style(self.spin_index_step)
        self.spin_index_step.setRange(1, 100)
        self.spin_index_step.setValue(1)
        self.spin_index_step.setFixedWidth(50)
        viz_row.addWidget(self.spin_index_step)
        
        self.btn_check_index = QPushButton("Check Index")
        self.btn_check_index.setStyleSheet(self.INDEXING_BTN_STYLE)
        self.btn_check_index.clicked.connect(self._visualize_keypoints_with_origin)
        viz_row.addWidget(self.btn_check_index)
        viz_row.addStretch()
        
        indexing_layout.addRow("Verify View:", viz_row)
        
        # New: Physical Size Settings (dx, dy, dz) - Vertical Layout
        size_header = QLabel("Physical Spacing (mm):")
        size_header.setStyleSheet("color: #00d4ff; font-weight: bold; margin-top: 5px;")
        indexing_layout.addRow(size_header)
        
        self.spin_dx = QDoubleSpinBox()
        self.spin_dy = QDoubleSpinBox()
        self.spin_dz = QDoubleSpinBox()
        for i, (spin, lbl) in enumerate(zip([self.spin_dx, self.spin_dy, self.spin_dz], ["  dx:", "  dy:", "  dz:"])):
            spin.setRange(0, 1000)
            spin.setValue(10.0 if i < 2 else 0.0) # Default 10x10mm, 0 for Z
            spin.setSuffix(" mm")
            spin.setFixedWidth(100)
            self._apply_input_style(spin)
            indexing_layout.addRow(lbl, spin)
        
        # New: Check Position button
        self.btn_check_pos = QPushButton("Check Position")
        self.btn_check_pos.setStyleSheet(self.INDEXING_BTN_STYLE)
        self.btn_check_pos.clicked.connect(self._start_check_position_mode)
        indexing_layout.addRow(self.btn_check_pos)
        
        # New: Add to Calibration Data button
        self.btn_add_to_data = QPushButton("Add to Calibration Data")
        self.btn_add_to_data.setStyleSheet("background-color: #00d4ff; color: #000; padding: 6px; font-weight: bold; margin-top: 5px;")
        self.btn_add_to_data.clicked.connect(self._add_to_calibration_data)
        indexing_layout.addRow(self.btn_add_to_data)
        
        indexing_group.setLayout(indexing_layout)
        det_layout.addWidget(indexing_group)
        
        # New: Export CSV button (outside indexing group)
        self.btn_export_csv = QPushButton("Export All to Calibration CSV")
        self.btn_export_csv.setStyleSheet("background-color: #34495e; color: #fff; padding: 8px 15px; font-weight: bold; margin-top: 10px;")
        self.btn_export_csv.clicked.connect(self._export_calibration_to_csv)
        det_layout.addWidget(self.btn_export_csv, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Initialize indexing state
        self.origin_point = None        # (x, y) in image coords
        self.axis1_point = None         # Point defining first axis direction
        self.axis2_point = None         # Point defining second axis direction
        self._axes_selection_step = 0   # 0 = not selecting, 1 = selecting first axis, 2 = selecting second
        self.point_indices = []         # List of [i, j, k] tuples corresponding to detected_keypoints
        self.check_pos_point = None    # User-clicked point for 3D verification
        self.check_pos_label = ""      # Label text for 3D verification
        
        det_layout.addStretch()
        
        controls_scroll.setWidget(controls)
        det_layout_wrap.addWidget(controls_scroll)
        
        self.plate_ctrl_tabs.addTab(det_tab, "Point Detection")
        
        # --- CONTROL TAB 2: Calibration ---
        cal_tab = QWidget()
        cal_tab_wrap_layout = QVBoxLayout(cal_tab)
        cal_tab_wrap_layout.setContentsMargins(0, 0, 0, 0)

        cal_scroll = QScrollArea()
        cal_scroll.setWidgetResizable(True)
        cal_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cal_scroll.setFrameShape(QFrame.Shape.NoFrame)
        cal_scroll.setStyleSheet("background-color: transparent;")

        cal_content = QWidget()
        cal_content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        cal_content.setStyleSheet("background-color: #000000;")

        cal_layout = QVBoxLayout(cal_content)
        cal_layout.setSpacing(15)
        cal_layout.setContentsMargins(15, 15, 15, 15)
        
        # Camera Settings
        cal_group = QGroupBox("Camera Settings")
        cal_group.setStyleSheet("QGroupBox { border: 1px solid #444; font-weight: bold; color: #00d4ff; border-radius: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        cal_form = QFormLayout(cal_group)
        cal_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        # 1. Target Camera
        self.cal_target_cam_combo = QComboBox()
        self._apply_input_style(self.cal_target_cam_combo)
        # Sync with the main cam combo
        self.cal_target_cam_combo.addItems([f"Camera {i}" for i in range(4)]) # Initial default
        cal_form.addRow("Target Camera:", self.cal_target_cam_combo)
        
        # 2. Model
        self.plate_model_combo = QComboBox()
        self._apply_input_style(self.plate_model_combo)
        self.plate_model_combo.addItems(["Pinhole", "Pinhole+Refraction"])
        self.plate_model_combo.currentIndexChanged.connect(self._on_plate_model_changed)
        cal_form.addRow("Model:", self.plate_model_combo)
        
        # 3. Resolution
        self.cal_img_width = QSpinBox()
        self.cal_img_height = QSpinBox()
        for sb in [self.cal_img_width, self.cal_img_height]:
            self._apply_input_style(sb)
            sb.setRange(1, 10000)
        self.cal_img_width.setValue(1280)
        self.cal_img_height.setValue(800)
        self.cal_img_width_label = QLabel("Image Width (px):")
        self.cal_img_height_label = QLabel("Image Height (px):")
        cal_form.addRow(self.cal_img_width_label, self.cal_img_width)
        cal_form.addRow(self.cal_img_height_label, self.cal_img_height)
        
        # 4. Sensor Width
        self.cal_sensor_width = QDoubleSpinBox()
        self._apply_input_style(self.cal_sensor_width)
        self.cal_sensor_width.setRange(0.0001, 100.0)
        self.cal_sensor_width.setDecimals(4)
        self.cal_sensor_width.setValue(0.0200) 
        self.cal_sensor_width_label = QLabel("Sensor Width (mm/px):")
        cal_form.addRow(self.cal_sensor_width_label, self.cal_sensor_width)
        
        # 5. Focal Length
        self.init_focal_spin = QDoubleSpinBox()
        self._apply_input_style(self.init_focal_spin)
        self.init_focal_spin.setRange(0.1, 1000.0)
        self.init_focal_spin.setValue(180.0)
        self.init_focal_spin.setSuffix(" mm")
        self.cal_focal_length_label = QLabel("Focal Length (mm):")
        cal_form.addRow(self.cal_focal_length_label, self.init_focal_spin)
        
        # 6. Distortion params (match Wand Calibration)
        self.cal_dist_model_combo = QComboBox()
        self._apply_input_style(self.cal_dist_model_combo)
        self.cal_dist_model_combo.addItems(["None (0)", "Radial k1 (1)", "Radial k1+k2 (2)"])
        self.cal_dist_model_combo.setCurrentIndex(0)
        cal_form.addRow("Distortion params:", self.cal_dist_model_combo)

        # Per-camera intrinsics (refraction mode, all-camera calibration)
        self.plate_intrinsics_label = QLabel("Camera Intrinsics:")
        self.plate_intrinsics_table = QTableWidget()
        self.plate_intrinsics_table.setColumnCount(4)
        self.plate_intrinsics_table.setHorizontalHeaderLabels(["Cam ID", "Focal (px)", "Width", "Height"])
        self.plate_intrinsics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.plate_intrinsics_table.verticalHeader().setVisible(False)
        self.plate_intrinsics_table.setFixedHeight(140)
        self.plate_intrinsics_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.plate_intrinsics_label.setVisible(False)
        self.plate_intrinsics_table.setVisible(False)
        cal_form.addRow(self.plate_intrinsics_label)
        cal_form.addRow(self.plate_intrinsics_table)

        # Target camera changed: persist/restore per-camera plate refraction settings
        self.cal_target_cam_combo.currentIndexChanged.connect(self._on_plate_target_cam_changed)

        cal_layout.addWidget(cal_group)

        # --- Plate Refraction Settings (UI only; per-camera) ---
        self.plate_refraction_group = QGroupBox("Refraction Settings")
        self.plate_refraction_group.setStyleSheet(self.GROUP_BOX_STYLE)
        plate_ref_layout = QVBoxLayout(self.plate_refraction_group)
        plate_ref_layout.setContentsMargins(5, 15, 5, 5)

        top_row = QFormLayout()
        self.plate_window_count_spin = QSpinBox()
        self._apply_input_style(self.plate_window_count_spin)
        self.plate_window_count_spin.setRange(1, 10)
        self.plate_window_count_spin.setValue(1)
        self.plate_window_count_spin.valueChanged.connect(self._on_plate_window_count_changed)
        top_row.addRow("Number of Windows:", self.plate_window_count_spin)
        plate_ref_layout.addLayout(top_row)

        plate_ref_layout.addWidget(QLabel("Camera-Window Mapping:"))
        self.plate_cam_window_table = QTableWidget()
        self.plate_cam_window_table.setColumnCount(2)
        self.plate_cam_window_table.setHorizontalHeaderLabels(["Camera ID", "Window"])
        self.plate_cam_window_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.plate_cam_window_table.verticalHeader().setVisible(False)
        self.plate_cam_window_table.setFixedHeight(120)
        self.plate_cam_window_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        plate_ref_layout.addWidget(self.plate_cam_window_table)

        plate_ref_layout.addSpacing(10)

        plate_ref_layout.addWidget(QLabel("Window Configuration (for selected type):"))

        diagram_layout = QHBoxLayout()
        diagram_layout.setSpacing(2)

        # Camera side
        seg1 = QWidget()
        seg1_layout = QVBoxLayout(seg1)
        seg1_layout.setContentsMargins(2, 2, 2, 2)
        seg1_layout.addWidget(QLabel("Camera side"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.plate_media1_combo = QComboBox()
        self.plate_media1_combo.addItems(["Air", "Other"])
        self.plate_media1_index = QDoubleSpinBox()
        self.plate_media1_index.setRange(1.0, 2.5)
        self.plate_media1_index.setValue(1.0)
        self._apply_input_style(self.plate_media1_combo)
        self._apply_input_style(self.plate_media1_index)
        self.plate_media1_combo.currentIndexChanged.connect(self._on_plate_refraction_media_changed)
        self.plate_media1_index.valueChanged.connect(self._on_plate_refraction_setting_changed)
        seg1_layout.addWidget(self.plate_media1_combo)
        seg1_layout.addWidget(self.plate_media1_index)
        diagram_layout.addWidget(seg1)

        # Window
        seg2 = QFrame()
        seg2.setStyleSheet("""
            background-color: rgba(0, 170, 255, 60);
            border-left: 2px solid #00aaff;
            border-right: 2px solid #00aaff;
            border-top: none;
            border-bottom: none;
        """)
        seg2_layout = QVBoxLayout(seg2)
        seg2_layout.setContentsMargins(2, 2, 2, 2)
        lbl_window = QLabel("Window")
        lbl_window.setStyleSheet("border: none; background: transparent; font-weight: bold;")
        seg2_layout.addWidget(lbl_window, alignment=Qt.AlignmentFlag.AlignCenter)
        self.plate_media2_combo = QComboBox()
        self.plate_media2_combo.addItems(["Acrylic", "Glass", "Other"])
        self.plate_media2_index = QDoubleSpinBox()
        self.plate_media2_index.setRange(1.0, 2.5)
        self.plate_media2_index.setValue(1.49)
        self.plate_media2_thick = QDoubleSpinBox()
        self.plate_media2_thick.setRange(0.1, 100.0)
        self.plate_media2_thick.setValue(10.0)
        self.plate_media2_thick.setSuffix(" mm")
        self._apply_input_style(self.plate_media2_combo)
        self._apply_input_style(self.plate_media2_index)
        self._apply_input_style(self.plate_media2_thick)
        self.plate_media2_combo.currentIndexChanged.connect(self._on_plate_refraction_media_changed)
        self.plate_media2_index.valueChanged.connect(self._on_plate_refraction_setting_changed)
        self.plate_media2_thick.valueChanged.connect(self._on_plate_refraction_setting_changed)
        seg2_layout.addWidget(self.plate_media2_combo)
        seg2_layout.addWidget(self.plate_media2_index)
        lbl_thick = QLabel("Thickness:")
        lbl_thick.setStyleSheet("border: none; background: transparent;")
        seg2_layout.addWidget(lbl_thick, alignment=Qt.AlignmentFlag.AlignCenter)
        seg2_layout.addWidget(self.plate_media2_thick)
        diagram_layout.addWidget(seg2)

        # Object side
        seg3 = QWidget()
        seg3_layout = QVBoxLayout(seg3)
        seg3_layout.setContentsMargins(2, 2, 2, 2)
        seg3_layout.addWidget(QLabel("Object side"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.plate_media3_combo = QComboBox()
        self.plate_media3_combo.addItems(["Water", "Other"])
        self.plate_media3_index = QDoubleSpinBox()
        self.plate_media3_index.setRange(1.0, 2.5)
        self.plate_media3_index.setValue(1.33)
        self._apply_input_style(self.plate_media3_combo)
        self._apply_input_style(self.plate_media3_index)
        self.plate_media3_combo.currentIndexChanged.connect(self._on_plate_refraction_media_changed)
        self.plate_media3_index.valueChanged.connect(self._on_plate_refraction_setting_changed)
        seg3_layout.addWidget(self.plate_media3_combo)
        seg3_layout.addWidget(self.plate_media3_index)
        diagram_layout.addWidget(seg3)

        plate_ref_layout.addLayout(diagram_layout)
        self.plate_refraction_group.setVisible(False)
        cal_layout.addWidget(self.plate_refraction_group)
        
        # Data Import Button
        self.btn_read_csv = QPushButton("Read Plate Points (from CSV)")
        self.btn_read_csv.setStyleSheet("background-color: #27ae60; color: #fff; padding: 8px; font-weight: bold; margin-top: 10px;")
        self.btn_read_csv.clicked.connect(self._read_calibration_from_csv)
        cal_layout.addWidget(self.btn_read_csv)
        
        self.btn_calibrate_plate = QPushButton("Run Calibration")
        self.btn_calibrate_plate.setStyleSheet(btn_style_primary)
        self.btn_calibrate_plate.clicked.connect(self._run_plate_calibration)
        cal_layout.addWidget(self.btn_calibrate_plate)
        
        # Results Group
        res_group = QGroupBox("Calibration Results")
        res_group.setStyleSheet("QGroupBox { border: 1px solid #444; font-weight: bold; color: #a0a0a0; border-radius: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        res_layout = QFormLayout(res_group)
        self.lbl_cal_rms = QLabel("N/A")
        self.lbl_cal_rms.setStyleSheet("color: #00ff88; font-weight: bold;")
        res_layout.addRow("Mean RMS Error:", self.lbl_cal_rms)
        cal_layout.addWidget(res_group)
        
        # Save All Button
        self.btn_save_all_cams = QPushButton("Save All Camera Parameters")
        self.btn_save_all_cams.setStyleSheet("background-color: #00d4ff; color: #000; font-weight: bold; padding: 10px; margin-top: 10px;")
        self.btn_save_all_cams.clicked.connect(self._save_all_camera_params)
        cal_layout.addWidget(self.btn_save_all_cams)
        
        cal_layout.addStretch()

        cal_scroll.setWidget(cal_content)
        cal_tab_wrap_layout.addWidget(cal_scroll)

        self.plate_ctrl_tabs.addTab(cal_tab, "Calibration")
        
        # --- CONTROL TAB 3: Tutorial ---
        tutorial_tab = QWidget()
        tutorial_layout = QVBoxLayout(tutorial_tab)
        tutorial_layout.setContentsMargins(15, 20, 15, 20)
        
        tutorial_label = QLabel("Learn how to use Plate Calibration with our step-by-step guide.")
        tutorial_label.setWordWrap(True)
        tutorial_label.setStyleSheet("color: #aaa; font-size: 13px;")
        tutorial_layout.addWidget(tutorial_label)
        
        btn_open_guide = QPushButton("Open User Guide")
        btn_open_guide.setStyleSheet("""
            QPushButton {
                background-color: #00bcd4;
                color: #000;
                font-weight: bold;
                padding: 12px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #4dd0e1; }
        """)
        btn_open_guide.clicked.connect(self._open_plate_calibration_guide)
        tutorial_layout.addWidget(btn_open_guide)
        
        tutorial_layout.addStretch()
        self.plate_ctrl_tabs.addTab(tutorial_tab, "Tutorial")

        layout.addWidget(self.plate_ctrl_tabs)
        
        # Initialize labels
        self._update_cam_list(4)
        self._on_plate_model_changed()
        self._load_plate_refraction_for_cam(self.cal_target_cam_combo.currentIndex())
              
        return container
    

    def create_wand_tab(self):
        """Create the Wand Calibration tab (Multi-Camera)."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 1. Visualization (LEFT)
        vis_frame = QFrame()
        vis_frame.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.setContentsMargins(0,0,0,0)

        # TabWidget for Cameras or Grid? 
        # User requested: "give each camera load first image"
        # Let's use a TabWidget because a Grid of 4 images at full Res inside a small frame is hard to see.
        # But user also said "visualization there give each camera load first image" which might imply simultaneous view.
        # Let's simple TabWidget for now, with "Camera 1", "Camera 2" tabs.
        
        self.vis_tabs = QTabWidget()
        self.vis_tabs.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QTabBar::tab { background: #222; color: #888; padding: 5px; }
            QTabBar::tab:selected { background: #444; color: #fff; }
        """)
        vis_layout.addWidget(self.vis_tabs)
        
        # We will populate tabs dynamically or pre-create for max cams?
        # Let's pre-create labels and store them
        self.cam_vis_labels = {} # {cam_idx: QLabel}
        
        # 2. Controls (RIGHT)
        controls = QWidget()
        controls.setFixedWidth(350)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 0, 10, 0)
        controls_layout.setSpacing(15)
        
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setFixedWidth(370)

        # -- Configuration --
        conf_group = QGroupBox("Configuration")
        conf_layout = QFormLayout(conf_group)
        
        self.wand_num_cams = QSpinBox()
        self._apply_input_style(self.wand_num_cams)
        self.wand_num_cams.setValue(4)
        self.wand_num_cams.setRange(1, 16)
        self.wand_num_cams.valueChanged.connect(self._update_wand_table)
        
        self.wand_model_combo = QComboBox()
        self._apply_input_style(self.wand_model_combo)
        self.wand_model_combo.addItems(["Pinhole", "Pinhole+Refraction"])
        self.wand_model_combo.currentIndexChanged.connect(self._on_model_changed)
        
        self.wand_len_spin = QDoubleSpinBox()
        self._apply_input_style(self.wand_len_spin)
        self.wand_len_spin.setValue(10.0)
        self.wand_len_spin.setSuffix(" mm")
        
        # New: Wand Type (Bright/Dark)
        self.wand_type_combo = QComboBox()
        self._apply_input_style(self.wand_type_combo)
        self.wand_type_combo.addItems(["Dark on Bright", "Bright on Dark"])
        
        # New: Circle Radius Range for detection (Range Slider)
        self.radius_range = RangeSlider(min_val=1, max_val=200, initial_min=20, initial_max=200, suffix=" px")
        
        # New: Sensitivity slider for detection
        from .widgets import SimpleSlider
        self.sensitivity_slider = SimpleSlider(min_val=0.5, max_val=1.0, initial=0.85, decimals=2)
        
        conf_layout.addRow("Num Cameras:", self.wand_num_cams)
        conf_layout.addRow("Camera Model:", self.wand_model_combo)
        conf_layout.addRow("Wand Length:", self.wand_len_spin)
        conf_layout.addRow("Wand Type:", self.wand_type_combo)
        conf_layout.addRow("Radius Range:", self.radius_range)
        conf_layout.addRow("Sensitivity:", self.sensitivity_slider)

        controls_layout.addWidget(conf_group)

        # -- Camera Images Table --
        controls_layout.addWidget(QLabel("Camera Images:"))
        
        self.wand_cam_table = QTableWidget()
        self.wand_cam_table.setColumnCount(3)
        self.wand_cam_table.setHorizontalHeaderLabels(["", "Camera", "Source"])
        
        header = self.wand_cam_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        self.wand_cam_table.verticalHeader().setVisible(False)
        self.wand_cam_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.wand_cam_table.setShowGrid(False)
        self.wand_cam_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.wand_cam_table.setFixedHeight(120) 
        
        self._update_wand_table(4)
        controls_layout.addWidget(self.wand_cam_table)

        # -- Frame Navigation (Directly in layout) --
        controls_layout.addWidget(QLabel("Frame List:"))
        
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(2)
        self.frame_table.setHorizontalHeaderLabels(["Index", "Filename"])
        self.frame_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.frame_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.frame_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.frame_table.cellClicked.connect(self._on_frame_table_clicked)
        self.frame_table.setFixedHeight(150) 
        
        controls_layout.addWidget(self.frame_table)

        # -- Actions --
        action_layout = QVBoxLayout()
        self.btn_detect_single = QPushButton("Detect Points (Current Frame)")
        self.btn_detect_single.setStyleSheet("background-color: #2a3f5f;")
        self.btn_detect_single.clicked.connect(self._detect_single_frame)

        self.btn_process_wand = QPushButton("1. Process All Frames")
        self.btn_process_wand.setStyleSheet("background-color: #2a3f5f;")
        self.btn_process_wand.clicked.connect(self._process_wand_frames)
        
        self.btn_calibrate_wand = QPushButton("2. Run Calibration")
        self.btn_calibrate_wand.setStyleSheet("background-color: #00d4ff; color: #000000; font-weight: bold;")
        self.btn_calibrate_wand.clicked.connect(self._run_wand_calibration)

        action_layout.addWidget(self.btn_detect_single)
        action_layout.addWidget(self.btn_process_wand)
        action_layout.addWidget(self.btn_calibrate_wand)
        
        # Progress Label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_layout.addWidget(self.status_label)

        controls_layout.addLayout(action_layout)

        controls_layout.addStretch()
        controls_scroll.setWidget(controls)
        layout.addWidget(vis_frame, stretch=2)
        layout.addWidget(controls_scroll)
        
        return tab

    def _update_cam_list(self, count):
        """Update camera dropdown and visualization tabs based on count."""
        # 1. Update Combo
        self.plate_cam_combo.blockSignals(True) # Avoid triggering update while changing
        current_idx = self.plate_cam_combo.currentIndex()
        self.plate_cam_combo.clear()
        items = [f"Camera {i}" for i in range(count)]
        self.plate_cam_combo.addItems(items)
        if current_idx >= 0 and current_idx < count:
            self.plate_cam_combo.setCurrentIndex(current_idx)
        else:
            self.plate_cam_combo.setCurrentIndex(0)
        self.plate_cam_combo.blockSignals(False)
        
        # New: Update Calibration Cam Combo
        if hasattr(self, 'cal_target_cam_combo'):
            self._refresh_plate_target_camera_combo()

        # Update plate refraction camera-window mapping table
        if hasattr(self, 'plate_cam_window_table'):
            self._update_plate_refraction_cam_table(count)
        if hasattr(self, 'plate_intrinsics_table'):
            self._update_plate_intrinsics_table(count)
            
        # 2. Update Visualization Tabs
        self.plate_vis_tabs.clear()
        self.plate_cam_labels = {}
        for i in range(count):
            lbl = ZoomableImageLabel(f"Camera {i+1} View")
            lbl.setStyleSheet("background-color: #202020; color: #666; font-size: 16px;")
            
            # Connect signals
            lbl.template_selected.connect(lambda rect, idx=i: self._on_template_roi_selected(rect, idx))
            lbl.roi_points_changed.connect(lambda pts, idx=i: self._on_roi_points_changed(pts, idx))
            lbl.remove_region_selected.connect(lambda rect, idx=i: self._on_remove_region(rect, idx))
            lbl.add_region_selected.connect(lambda rect, idx=i: self._on_add_region(rect, idx))
            lbl.origin_selected.connect(lambda pt, idx=i: self._on_origin_selected(pt, idx))
            lbl.axis_point_selected.connect(lambda pt, axis_idx, idx=i: self._on_axis_point_selected(pt, axis_idx, idx))
            lbl.point_clicked.connect(self._on_check_pos_clicked)
            
            self.plate_cam_labels[i] = lbl
            self.plate_vis_tabs.addTab(lbl, f"Cam {i+1}")
            
        # Re-add 3D View tab after clearing
        self.plate_vis_tabs.addTab(self.plate_3d_viewer, "3D View")

    def _refresh_plate_target_camera_combo(self):
        """Refresh target camera list depending on plate model.

        - Pinhole: per-camera selection.
        - Pinhole+Refraction: all-camera joint mode only.
        """
        if not hasattr(self, 'cal_target_cam_combo') or self.cal_target_cam_combo is None:
            return

        is_refraction = hasattr(self, 'plate_model_combo') and self.plate_model_combo.currentText() == "Pinhole+Refraction"
        cam_count = int(self.plate_num_cams.value()) if hasattr(self, 'plate_num_cams') else 0

        self.cal_target_cam_combo.blockSignals(True)
        self.cal_target_cam_combo.clear()
        if is_refraction:
            self.cal_target_cam_combo.addItem("All Cameras")
            self.cal_target_cam_combo.setCurrentIndex(0)
            self.cal_target_cam_combo.setEnabled(False)
        else:
            self.cal_target_cam_combo.addItems([f"Camera {i}" for i in range(cam_count)])
            cur = self.plate_cam_combo.currentIndex() if hasattr(self, 'plate_cam_combo') else 0
            if cur < 0:
                cur = 0
            self.cal_target_cam_combo.setCurrentIndex(min(cur, max(0, cam_count - 1)))
            self.cal_target_cam_combo.setEnabled(True)
        self.cal_target_cam_combo.blockSignals(False)

    def _update_plate_refraction_cam_table(self, cam_count):
        """Sync plate refraction camera-window mapping table."""
        if not hasattr(self, 'plate_cam_window_table') or self.plate_cam_window_table is None:
            return

        self.plate_cam_window_table.setRowCount(int(cam_count))
        win_count = int(self.plate_window_count_spin.value()) if hasattr(self, 'plate_window_count_spin') else 1
        window_options = [f"Window {i}" for i in range(max(1, win_count))]

        for i in range(int(cam_count)):
            cam_id_item = QTableWidgetItem(f"{i}")
            cam_id_item.setFlags(Qt.ItemFlag.NoItemFlags)
            cam_id_item.setForeground(Qt.GlobalColor.white)
            cam_id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.plate_cam_window_table.setItem(i, 0, cam_id_item)

            prev_combo = self.plate_cam_window_table.cellWidget(i, 1)
            prev_text = prev_combo.currentText() if isinstance(prev_combo, QComboBox) else ""

            win_combo = QComboBox()
            self._apply_input_style(win_combo)
            win_combo.addItems(window_options)
            if prev_text in window_options:
                win_combo.setCurrentText(prev_text)
            self.plate_cam_window_table.setCellWidget(i, 1, win_combo)

    def _on_plate_window_count_changed(self):
        if hasattr(self, 'plate_num_cams'):
            self._update_plate_refraction_cam_table(int(self.plate_num_cams.value()))

    def _default_plate_intrinsic_setting(self):
        f_mm = float(self.init_focal_spin.value()) if hasattr(self, 'init_focal_spin') else 180.0
        pw_mm = float(self.cal_sensor_width.value()) if hasattr(self, 'cal_sensor_width') else 0.02
        f_px = f_mm / max(pw_mm, 1e-12)
        w = int(self.cal_img_width.value()) if hasattr(self, 'cal_img_width') else 1280
        h = int(self.cal_img_height.value()) if hasattr(self, 'cal_img_height') else 800
        return {
            'focal_px': float(f_px),
            'width': int(w),
            'height': int(h),
        }

    def _save_plate_intrinsics_value(self, cam_idx: int, key: str, value):
        if not hasattr(self, 'plate_intrinsics_settings'):
            self.plate_intrinsics_settings = {}
        st = self.plate_intrinsics_settings.get(int(cam_idx), self._default_plate_intrinsic_setting())
        st = dict(st)
        st[key] = value
        self.plate_intrinsics_settings[int(cam_idx)] = st

    def _capture_plate_image_size_hint(self, cam_idx: int, img_path) -> bool:
        """Capture image size hint for a camera from a plate image path."""
        try:
            import cv2
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None or len(img.shape) < 2:
                return False
            h, w = int(img.shape[0]), int(img.shape[1])
            if w <= 0 or h <= 0:
                return False
            self.plate_image_size_hints[int(cam_idx)] = (w, h)
            return True
        except Exception:
            return False

    def _build_plate_image_size_hints_from_saved_data(self):
        """Backfill size hints using known image paths in saved calibration data."""
        if not hasattr(self, 'saved_calibration_data') or not self.saved_calibration_data:
            return
        for (cam_idx, img_path), _data in self.saved_calibration_data.items():
            cid = int(cam_idx)
            if cid in self.plate_image_size_hints:
                continue
            self._capture_plate_image_size_hint(cid, img_path)

    def _autofill_plate_intrinsics_once_from_hints(self):
        """Auto-fill width/height only once per camera; user edits remain authoritative."""
        if not hasattr(self, 'plate_num_cams'):
            return
        cam_count = int(self.plate_num_cams.value())
        for cid in range(cam_count):
            if cid in self._plate_intrinsics_autofilled_once:
                continue
            hint = self.plate_image_size_hints.get(cid)
            if not hint:
                continue
            w, h = int(hint[0]), int(hint[1])
            st = dict(self.plate_intrinsics_settings.get(cid, self._default_plate_intrinsic_setting()))
            st['width'] = w
            st['height'] = h
            self.plate_intrinsics_settings[cid] = st
            self._plate_intrinsics_autofilled_once.add(cid)

    def _update_plate_intrinsics_table(self, cam_count):
        if not hasattr(self, 'plate_intrinsics_table') or self.plate_intrinsics_table is None:
            return
        self.plate_intrinsics_table.setRowCount(int(cam_count))

        for i in range(int(cam_count)):
            st = self.plate_intrinsics_settings.get(i, self._default_plate_intrinsic_setting())
            self.plate_intrinsics_settings[i] = dict(st)

            cam_id_item = QTableWidgetItem(f"{i}")
            cam_id_item.setFlags(Qt.ItemFlag.NoItemFlags)
            cam_id_item.setForeground(Qt.GlobalColor.white)
            cam_id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.plate_intrinsics_table.setItem(i, 0, cam_id_item)

            focal_spin = TrimmedDoubleSpinBox()
            self._apply_input_style(focal_spin)
            focal_spin.setRange(1.0, 100000.0)
            focal_spin.setDecimals(6)
            focal_spin.setValue(float(st.get('focal_px', 9000.0)))
            focal_spin.valueChanged.connect(lambda v, cam=i: self._save_plate_intrinsics_value(cam, 'focal_px', float(v)))
            self.plate_intrinsics_table.setCellWidget(i, 1, focal_spin)

            w_spin = QSpinBox()
            self._apply_input_style(w_spin)
            w_spin.setRange(1, 10000)
            w_spin.setValue(int(st.get('width', 1280)))
            w_spin.valueChanged.connect(lambda v, cam=i: self._save_plate_intrinsics_value(cam, 'width', int(v)))
            self.plate_intrinsics_table.setCellWidget(i, 2, w_spin)

            h_spin = QSpinBox()
            self._apply_input_style(h_spin)
            h_spin.setRange(1, 10000)
            h_spin.setValue(int(st.get('height', 800)))
            h_spin.valueChanged.connect(lambda v, cam=i: self._save_plate_intrinsics_value(cam, 'height', int(v)))
            self.plate_intrinsics_table.setCellWidget(i, 3, h_spin)

    def _set_plate_global_intrinsic_controls_visible(self, visible: bool):
        for label_attr, field_attr in [
            ('cal_img_width_label', 'cal_img_width'),
            ('cal_img_height_label', 'cal_img_height'),
            ('cal_sensor_width_label', 'cal_sensor_width'),
            ('cal_focal_length_label', 'init_focal_spin'),
        ]:
            if hasattr(self, label_attr):
                getattr(self, label_attr).setVisible(bool(visible))
            if hasattr(self, field_attr):
                getattr(self, field_attr).setVisible(bool(visible))

    def _plate_refraction_setting_key(self) -> int:
        """Storage key for plate refraction settings.

        Refraction mode uses one shared config for all cameras.
        """
        is_refraction = hasattr(self, 'plate_model_combo') and self.plate_model_combo.currentText() == "Pinhole+Refraction"
        if is_refraction:
            return -1
        if hasattr(self, 'cal_target_cam_combo'):
            return int(self.cal_target_cam_combo.currentIndex())
        return 0
            
    def _on_template_roi_selected(self, rect, cam_idx):
        """Handle ROI selection for template matching."""
        # rect is already in IMAGE coordinates (converted by ZoomableImageLabel._to_image_coords)
            
        # Get raw image
        row = self.plate_img_list.currentRow()
        if row < 0 or row >= len(self.plate_images):
            return
            
        img_path = self.plate_images[row]
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return
            
        orig_h, orig_w = img.shape[:2]
        
        # Use rect directly as image coordinates
        ix = rect.x()
        iy = rect.y()
        iw = rect.width()
        ih = rect.height()
        
        # Clip to image bounds
        ix = max(0, min(ix, orig_w - 1))
        iy = max(0, min(iy, orig_h - 1))
        iw = min(iw, orig_w - ix)
        ih = min(ih, orig_h - iy)
        
        if iw > 0 and ih > 0:
            # 1. Extract Initial Patch
            patch = img[iy:iy+ih, ix:ix+iw]
            
            # 2. Auto-Center Logic
            try:
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                
                # Threshold
                # Use Otsu's thresholding
                # Note: We don't strictly rely on is_dark for inv/norm here anymore, 
                # we'll use a border check to ensure the 'dot' is the foreground.
                gray_u8 = gray.astype(np.uint8)
                _, binary = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Check borders to see if background is white (255)
                # If background is white, findContours will treat it as object. We want dot as object.
                h_b, w_b = binary.shape
                border_mean = (np.mean(binary[0, :]) + np.mean(binary[-1, :]) + 
                               np.mean(binary[:, 0]) + np.mean(binary[:, -1])) / 4.0
                               
                if border_mean > 127:
                    # Background is white -> Invert so Background is black, Dot is white
                    binary = 255 - binary
                
                # Find Contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Assume largest contour is the dot
                    largest_cnt = max(contours, key=cv2.contourArea)
                    
                    # Use Bounding Rect for geometric center (more robust than intensity centroid)
                    bx, by, bw, bh = cv2.boundingRect(largest_cnt)
                    
                    # Center of the contour (relative to patch)
                    cnt_cx = bx + bw / 2.0
                    cnt_cy = by + bh / 2.0
                    
                    # Absolute Center in Original Image
                    abs_cx = ix + cnt_cx
                    abs_cy = iy + cnt_cy
                    
                    # Define New Crop Window (preserving original width/height)
                    # We want abs_center to be at (iw/2, ih/2) of new patch
                    new_ix = int(abs_cx - iw / 2.0)
                    new_iy = int(abs_cy - ih / 2.0)
                    
                    # Check bounds
                    if 0 <= new_ix and 0 <= new_iy and new_ix + iw <= orig_w and new_iy + ih <= orig_h:
                        # Apply shift
                        ix = new_ix
                        iy = new_iy
                        patch = img[iy:iy+ih, ix:ix+iw] # Re-crop
                        self.lbl_template_status.setText(f"{iw}x{ih} OK")
                    else:
                        # Fallback for edge cases: clip to bounds
                        ix = max(0, min(new_ix, orig_w - iw))
                        iy = max(0, min(new_iy, orig_h - ih))
                        patch = img[iy:iy+ih, ix:ix+iw]
                        self.lbl_template_status.setText(f"{iw}x{ih} OK*")
                
                    # 3. Calculate Sub-Pixel Offset
                    # Re-run contour on the FINAL patch to find exact centroid offset
                    # This handles the fact that integer shifting might still be 0.5px off
                    try:
                        p_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                        # Check border for auto-polarity
                        if (np.mean(p_gray[0,:]) + np.mean(p_gray[-1,:]) + np.mean(p_gray[:,0]) + np.mean(p_gray[:,-1]))/4 > 127:
                            p_bin = 255 - p_gray # Invert
                        else:
                            p_bin = p_gray
                            
                        # Threshold
                        _, p_bin = cv2.threshold(p_bin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        
                        p_contours, _ = cv2.findContours(p_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if p_contours:
                            largest_p = max(p_contours, key=cv2.contourArea)
                            M = cv2.moments(largest_p)
                            if M["m00"] > 0:
                                fcx = M["m10"] / M["m00"]
                                fcy = M["m01"] / M["m00"]
                                
                                # Geometric center of patch
                                gcx = patch.shape[1] / 2.0
                                gcy = patch.shape[0] / 2.0
                                
                                # Offset
                                off_x = fcx - gcx
                                off_y = fcy - gcy
                                self.template_offset = (off_x, off_y)
                                # Sub-pixel offset stored but not displayed to save space
                    except Exception as ex:
                        print(f"Sub-pixel calc failed: {ex}")
                        self.template_offset = (0.0, 0.0)

                else:
                    # No contour found
                    self.lbl_template_status.setText(f"{iw}x{ih} (raw)")
            except Exception as e:
                print(f"Auto-center failed: {e}")
                self.lbl_template_status.setText(f"{iw}x{ih}")

            self.current_template = patch
            self.lbl_template_status.setStyleSheet("color: #00ff00;")
            
            # Show Preview
            # Convert BGR to RGB
            preview_img = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            h, w, ch = preview_img.shape
            
            # Draw Crosshair on preview to show center
            cx, cy = w // 2, h // 2
            cv2.line(preview_img, (cx - 5, cy), (cx + 5, cy), (0, 255, 0), 1)
            cv2.line(preview_img, (cx, cy - 5), (cx, cy + 5), (0, 255, 0), 1)
            
            from PySide6.QtGui import QImage, QPixmap
            q_img = QImage(preview_img.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(q_img)
            self.lbl_template_preview.setPixmap(pix.scaled(
                self.lbl_template_preview.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.FastTransformation
            ))
            
        else:
             self.lbl_template_status.setText("Invalid Selection")
             self.lbl_template_preview.clear()
        
    def _sync_vis_tab(self, idx):
        """Sync visualization tab with combo box selection."""
        if idx >= 0 and idx < self.plate_vis_tabs.count():
            self.plate_vis_tabs.setCurrentIndex(idx)

    def _update_wand_table(self, count):
        self.wand_cam_table.setRowCount(count)
        self.wand_images = {i: [] for i in range(count)}
        
        # Update Vis Tabs (Wand)
        try:
            self.vis_tabs.clear()
            self.cam_vis_labels = {}
            for i in range(count):
                lbl = QLabel("No Image")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet("background: #1a1a1a; color: #666; font-size: 18px;")
                self.cam_vis_labels[i] = lbl
                self.vis_tabs.addTab(lbl, f"Cam {i+1}")
            
            # Add 3D View tab at the end
            if not hasattr(self, 'calib_3d_view') or self.calib_3d_view is None:
                self.calib_3d_view = Calibration3DViewer()
            self.vis_tabs.addTab(self.calib_3d_view, "3D View")
            # self.vis_tabs.addTab(QLabel("3D View Disabled (Debug)"), "3D View")
            
        except RuntimeError:
            return 

        for i in range(count):
            # Col 0: Load Folder button (compact)
            btn = QPushButton("Load")
            btn.setStyleSheet("background-color: #2a3f5f; padding: 2px 6px; font-size: 10px;")
            btn.clicked.connect(lambda checked=False, idx=i: self._load_wand_folder_for_cam(idx))
            self.wand_cam_table.setCellWidget(i, 0, btn)
            
            # Col 1: Cam ID (read-only)
            name_item = QTableWidgetItem(f"{i}")
            name_item.setFlags(Qt.ItemFlag.NoItemFlags)
            name_item.setForeground(Qt.GlobalColor.white)
            self.wand_cam_table.setItem(i, 1, name_item)
            
            # Col 2: Focal Length (editable spinbox)
            focal_spin = QSpinBox()
            focal_spin.setRange(100, 1000000)
            focal_spin.setValue(9000)
            focal_spin.setStyleSheet("background: #222; color: white; border: none;")
            self.wand_cam_table.setCellWidget(i, 2, focal_spin)
            
            # Col 3: Width (editable spinbox)
            width_spin = QSpinBox()
            width_spin.setRange(1, 10000)
            width_spin.setValue(1280)
            width_spin.setStyleSheet("background: #222; color: white; border: none;")
            width_spin.valueChanged.connect(lambda _v: self._refresh_wand_radius_range_limit())
            self.wand_cam_table.setCellWidget(i, 3, width_spin)
            
            # Col 4: Height (editable spinbox)
            height_spin = QSpinBox()
            height_spin.setRange(1, 10000)
            height_spin.setValue(800)
            height_spin.setStyleSheet("background: #222; color: white; border: none;")
            height_spin.valueChanged.connect(lambda _v: self._refresh_wand_radius_range_limit())
            self.wand_cam_table.setCellWidget(i, 4, height_spin)
            
        # Update refraction camera table if it exists
        if hasattr(self, 'cam_window_table'):
            self._update_refraction_cam_table(count)

        self._refresh_wand_radius_range_limit()

    # --- Logic Implementation ---

    def _load_plate_images(self, checked=False):
        """Load selected image files for the currently selected camera."""
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Image Files", 
            "", 
            "Image Files (*.png *.jpg *.bmp *.tif *.jpeg);;All Files (*)"
        )
        if not files:
            return
        
        from pathlib import Path
        files = sorted(files)
        
        if files:
            # Clear previous images as requested
            self.plate_images.clear()
            self.plate_img_list.clear() # Clear list widget
            
            # If list was empty, we want to show the FIRST new image.
            # (It is now always "empty" before adding since we cleared it)
            was_empty = True
            
            self.plate_images.extend(files)

            # Capture size hint from first image for current camera.
            cam_idx = int(self.plate_cam_combo.currentIndex()) if hasattr(self, 'plate_cam_combo') else 0
            self._capture_plate_image_size_hint(cam_idx, files[0])
            
            for f in files:
                self.plate_img_list.addItem(Path(f).name)
            
            # Auto-display logic
            if was_empty:
                self.plate_img_list.setCurrentRow(0) # Selects first
                self._display_plate_image(0) # Explicitly call display

            # In refraction mode, auto-fill one time then keep user editable values.
            is_refraction = hasattr(self, 'plate_model_combo') and self.plate_model_combo.currentText() == "Pinhole+Refraction"
            if is_refraction and hasattr(self, 'plate_intrinsics_table') and hasattr(self, 'plate_num_cams'):
                self._autofill_plate_intrinsics_once_from_hints()
                self._update_plate_intrinsics_table(int(self.plate_num_cams.value()))

        else:
            QMessageBox.warning(self, "No Images", "No images found in selected folder.")

    def _clear_plate_images(self, checked=False):
        self.plate_images.clear()
        self.plate_img_list.clear()
        # Clear current label
        idx = self.plate_cam_combo.currentIndex()
        if idx in self.plate_cam_labels:
            self.plate_cam_labels[idx].clear()
            self.plate_cam_labels[idx].setText("No Images")

    def _display_plate_image(self, row):
        if row < 0 or row >= len(self.plate_images):
            return
        img_path = self.plate_images[row]
        
        # --- Clear Previous State ---
        # 1. Clear overlays on current label
        cam_idx = self.plate_cam_combo.currentIndex()
        target_label = self.plate_cam_labels.get(cam_idx, None)
        if target_label:
            target_label.clear_overlays()
        
        # 2. Reset origin and axis info to prevent stale data usage
        self.origin_point = None
        self.axis1_pt = None
        self.axis2_pt = None
        self._axes_selection_step = 0
        # Clear axis direction labels if they exist
        if hasattr(self, 'lbl_axis1_status'):
            self.lbl_axis1_status.setText("Not Set")
        if hasattr(self, 'lbl_axis2_status'):
            self.lbl_axis2_status.setText("Not Set")
        if hasattr(self, 'lbl_origin_status'):
            self.lbl_origin_status.setText("Not Set")
        # --- End Clear ---
        
        # New: Update plane number based on current row and offset
        self._is_manually_changing_plane = False # Block offset update
        self.spin_plane_num.blockSignals(True)
        self.spin_plane_num.setValue(row + self.plane_num_offset)
        self.spin_plane_num.blockSignals(False)

        # New: Load saved data if exists (using Cam ID + Path)
        data_key = (cam_idx, img_path)
        if data_key in self.saved_calibration_data:
            data = self.saved_calibration_data[data_key]
            self.detected_keypoints = data['keypoints']
            self.point_indices = data['indices']
            self.lbl_points_count.setText(f"Points: {len(self.detected_keypoints)} (Loaded from Cam {cam_idx})")
        else:
            # Optionally clear or keep? Let's clear to avoid confusion across images
            self.detected_keypoints = []
            self.point_indices = []
            self.lbl_points_count.setText("Points: 0")

        # Robust loading with OpenCV
        import cv2
        import numpy as np
        from PySide6.QtGui import QPixmap, QImage
        
        pixmap = QPixmap() # Default empty
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        
        if img is not None:
             # Save size hint before any conversion.
             self.plate_image_size_hints[int(cam_idx)] = (int(img.shape[1]), int(img.shape[0]))

              # Convert to RGB for Qt
             if len(img.shape) == 2:  # Grayscale
                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
             elif img.shape[2] == 4:  # RGBA
                 img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
             else:  # BGR
                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
             
             if img.dtype != np.uint8:
                 img = (img / img.max() * 255).astype(np.uint8)
                  
             h, w, ch = img.shape
             bytes_per_line = ch * w
             q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
             pixmap = QPixmap.fromImage(q_img)
        else:
             pixmap = QPixmap(str(img_path))
        
        cam_idx = self.plate_cam_combo.currentIndex()
        target_label = self.plate_cam_labels.get(cam_idx, None)
        
        if target_label:
            if not pixmap.isNull():
                target_label.setPixmap(pixmap)
                # Ensure we also visualize origin/points if loaded
                self._visualize_keypoints_with_origin()
            else:
                target_label.setText(f"Failed to load image:\n{Path(img_path).name}")

    def _load_wand_folder_for_cam(self, cam_idx):
        folder = QFileDialog.getExistingDirectory(self, f"Select Image Folder for Camera {cam_idx+1}")
        if folder:
            from pathlib import Path
            p = Path(folder)
            # Find images
            files = sorted([str(f) for f in p.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.bmp', '.tif', '.jpeg']])
            
            if files:
                self.wand_images[cam_idx] = files
                btn = self.wand_cam_table.cellWidget(cam_idx, 0)  # Column 0 now
                if btn:
                    btn.setText(f"{len(files)}")

                self._update_wand_cam_size_from_first_image(cam_idx, files)
                self._refresh_wand_radius_range_limit()
                 
                # Check consistency and update Frames Table
                self.populate_wand_table()

    def _update_wand_cam_size_from_first_image(self, cam_idx, files):
        """Read first image size and update Width/Height columns for this camera."""
        if not files:
            return

        try:
            import cv2
            img = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
            if img is None or len(img.shape) < 2:
                return
            h, w = int(img.shape[0]), int(img.shape[1])
        except Exception:
            return

        width_spin = self.wand_cam_table.cellWidget(cam_idx, 3)
        height_spin = self.wand_cam_table.cellWidget(cam_idx, 4)
        if width_spin is not None:
            width_spin.setValue(w)
        if height_spin is not None:
            height_spin.setValue(h)

    def _refresh_wand_radius_range_limit(self):
        """Set radius max to min(image side)/4 across loaded cameras; default to 200."""
        if not hasattr(self, 'radius_range') or self.radius_range is None:
            return

        min_sides = []
        row_count = self.wand_cam_table.rowCount() if hasattr(self, 'wand_cam_table') else 0
        for row in range(row_count):
            imgs = self.wand_images.get(row, []) if hasattr(self, 'wand_images') else []
            if not imgs:
                continue
            width_spin = self.wand_cam_table.cellWidget(row, 3)
            height_spin = self.wand_cam_table.cellWidget(row, 4)
            if width_spin is None or height_spin is None:
                continue
            w = int(width_spin.value())
            h = int(height_spin.value())
            if w > 0 and h > 0:
                min_sides.append(min(w, h))

        if min_sides:
            max_r = max(1, int(min(min_sides) // 4))
        else:
            max_r = 200

        self.radius_range.setRange(1, max_r)

    def _get_wand_detect_mode(self):
        if hasattr(self, 'detect_mode_reliable') and self.detect_mode_reliable.isChecked():
            return "reliable"
        return "fast"
    
    def populate_wand_table(self):
        """Populates the frame table with filenames and status."""
        # Find max length and reference cam
        max_len = 0
        max_idx = -1
        for k, v in self.wand_images.items():
            if len(v) > max_len:
                max_len = len(v)
                max_idx = k
        
        self.frame_table.setRowCount(max_len)
        reference_files = self.wand_images.get(max_idx, [])
        
        from PySide6.QtGui import QFont
        
        for i in range(max_len):
            # Index
            idx_item = QTableWidgetItem(str(i+1))
            idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.frame_table.setItem(i, 0, idx_item)
            
            # Filename
            fname = "Frame " + str(i+1)
            if i < len(reference_files):
                from pathlib import Path
                fname = Path(reference_files[i]).name
            
            name_item = QTableWidgetItem(fname)
            self.frame_table.setItem(i, 1, name_item)
            
            # Check Status (Valid/Invalid)
            if hasattr(self, 'wand_calibrator') and \
               hasattr(self.wand_calibrator, 'wand_data_raw') and \
               i in self.wand_calibrator.wand_data_raw:
                
                # If frame processed but NOT in filtered, it's invalid
                if hasattr(self.wand_calibrator, 'wand_data_filtered') and \
                   i not in self.wand_calibrator.wand_data_filtered:
                    
                    # Mark as Invalid (Strikethrough + Red)
                    font = name_item.font()
                    font.setStrikeOut(True)
                    name_item.setFont(font)
                    name_item.setForeground(Qt.GlobalColor.red)
                    idx_item.setForeground(Qt.GlobalColor.red)
                else:
                    # Valid
                    name_item.setForeground(Qt.GlobalColor.white)
                    idx_item.setForeground(Qt.GlobalColor.white)
        
        # Select first row if valid
        if max_len > 0 and self.frame_table.currentRow() < 0:
             self.frame_table.selectRow(0)
             self._update_vis_frame(0)
                     


    def _on_frame_table_clicked(self, row, col):
        self._update_vis_frame(row)

    def _update_vis_frame(self, frame_idx):
        from PySide6.QtGui import QPixmap, QImage
        import cv2
        import numpy as np
        
        # Update labels for all cams
        for cam_idx, lbl in self.cam_vis_labels.items():
            if cam_idx in self.wand_images and frame_idx < len(self.wand_images[cam_idx]):
                 path = self.wand_images[cam_idx][frame_idx]
                 
                 # Use OpenCV to load (supports .tif and more formats)
                 img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                 if img is None:
                     lbl.setText("Image Load Error")
                     continue
                 
                 # Convert to RGB for Qt
                 if len(img.shape) == 2:  # Grayscale
                     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                 elif img.shape[2] == 4:  # RGBA
                     img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                 else:  # BGR
                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                 
                 # Normalize to 8-bit if needed (for 16-bit TIF)
                 # Normalize to 8-bit if needed (for 16-bit TIF)
                 if img.dtype != np.uint8:
                     img = (img / img.max() * 255).astype(np.uint8)
                 
                 # Draw Points if available
                 # 1. Raw Data (Red Circles)
                 if hasattr(self, 'wand_calibrator') and \
                    hasattr(self.wand_calibrator, 'wand_data_raw') and \
                    frame_idx in self.wand_calibrator.wand_data_raw:
                    
                     ct_dict = self.wand_calibrator.wand_data_raw[frame_idx]
                     if cam_idx in ct_dict:
                         raw_pts = ct_dict[cam_idx]
                         # Draw Red
                         for p in raw_pts:
                             x, y, r = int(p[0]), int(p[1]), int(p[2])
                             cv2.circle(img, (x, y), r, (0, 0, 255), 2) # Red
                 
                 # 2. Filtered Data (Green/Cyan Circles)
                 if hasattr(self, 'wand_calibrator') and \
                    hasattr(self.wand_calibrator, 'wand_data_filtered') and \
                    frame_idx in self.wand_calibrator.wand_data_filtered:
                     
                     filt_dict = self.wand_calibrator.wand_data_filtered[frame_idx]
                     if cam_idx in filt_dict:
                         filt_pts = filt_dict[cam_idx] # [pt_small, pt_large]
                         
                         # Small point (Light Green)
                         p_s = filt_pts[0]
                         cv2.circle(img, (int(p_s[0]), int(p_s[1])), int(p_s[2]), (0, 255, 100), 2)
                         # Large point (Dark Green)
                         p_l = filt_pts[1]
                         cv2.circle(img, (int(p_l[0]), int(p_l[1])), int(p_l[2]), (0, 150, 50), 2)

                 h, w, ch = img.shape
                 bytes_per_line = ch * w
                 q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                 pix = QPixmap.fromImage(q_img)
                 
                 # Get label size, use default if too small
                 lbl_size = lbl.size()
                 if lbl_size.width() < 50 or lbl_size.height() < 50:
                     lbl_size = self.vis_tabs.size()
                 
                 scaled = pix.scaled(lbl_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                 lbl.setPixmap(scaled)
            else:
                lbl.setText("No Image")

    def _test_grid_detection(self):
        self._run_detection(smart_fill=False)
        
    def _test_smart_fill(self):
        self._run_detection(smart_fill=True)

    def _toggle_detection_method(self, checked):
        # 1. Toggle Template Group
        self.template_group.setVisible(checked)
        
        # 2. Toggle Blob Settings Group
        # Iterate over layout items in Detection Settings group to hide/show them
        # Except the Method Toggle itself
        # This is a bit hacky, easier if we had a sub-widget or layout.
        # But we added "adv_group" (Detection Settings) in create_plate_tab.
        # Let's just assume we want to hide sliders.
        return

    def _run_detection(self, smart_fill=False):
        """Test the grid detection algorithm on the currently selected image."""
        row = self.plate_img_list.currentRow()
        if row < 0 or row >= len(self.plate_images):
            QMessageBox.warning(self, "No Image", "Please select an image from the list first.")
            return

        img_path = self.plate_images[row]
        
        try:
            from .plate_calibration.grid_detector import GridDetector
            
            # --- Template Matching Branch ---
            if self.check_use_template.isChecked():
                if self.current_template is None:
                    QMessageBox.warning(self, "No Template", "Please select a template ROI on the image first.")
                    return
                
                threshold = self.slider_match_thresh.value()
                
                keypoints, vis_img = GridDetector.detect_template(
                    img_path,
                    self.current_template,
                    threshold=threshold,
                    smart_fill=smart_fill,
                    center_offset=self.template_offset
                )
            
            # --- Blob Detection Branch ---
            else:
                import math
                min_r, max_r = self.grid_radius_range.value()
                min_area = int(math.pi * (min_r**2))
                max_area = int(math.pi * (max_r**2))
                
                min_circ = self.grid_min_circ.value()
                
                # Point Color
                color_text = self.plate_color_combo.currentText()
                blob_color = 0 if "Dark" in color_text else 255
                
                # Run Detection
                keypoints, vis_img = GridDetector.detect(
                    img_path,
                    min_area=min_area,
                    max_area=max_area,
                    min_circ=min_circ,
                    min_conv=0.1, 
                    min_inertia=0.1,
                    blob_color=blob_color,
                    smart_fill=smart_fill
                )
            
            # Convert to QPixmap for display
            import cv2
            h, w, ch = vis_img.shape
            bytes_per_line = ch * w
            # OpenCV is BGR, Qt needs RGB
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            
            from PySide6.QtGui import QImage, QPixmap
            q_img = QImage(vis_img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(q_img)
            
            # Display on the CORRECT label (active camera)
            cam_idx = self.plate_cam_combo.currentIndex()
            target_label = self.plate_cam_labels.get(cam_idx, None)

            if target_label:
                # Set ORIGINAL pixmap - paintEvent will handle scaling
                target_label.setPixmap(pix)
            
            # Show stats
            QMessageBox.information(self, "Detection Result", f"Found {len(keypoints)} points.")
            
        except ImportError:
             QMessageBox.critical(self, "Import Error", "Could not import GridDetector.")
        except TypeError as te:
             # Likely GridDetector.detect signature mismatch if I added max_area
             QMessageBox.critical(self, "Detection Error", f"Argument Error: {te}")
        except Exception as e:
            QMessageBox.critical(self, "Detection Error", f"An error occurred during detection:\n{e}")

    def _detect_plate_points(self, checked=False):
        print("Detecting points...")

    def _run_plate_calibration(self, checked=False):
        """
        Run camera calibration for the selected target camera using 3D-2D correspondences.
        Optimization:
        1. Fix intrinsics, solve for extrinsics (Pose) via PnP.
        2. Refine both intrinsics (f, cx, cy) and extrinsics via calibrateCamera.
        """
        self._busy_begin('plate_calibration', 'Running plate calibration')

        if hasattr(self, 'plate_model_combo') and self.plate_model_combo.currentText() == "Pinhole+Refraction":
            return self._run_plate_refraction_calibration()

        target_cam_idx = self.cal_target_cam_combo.currentIndex()
        print(f"Running Plate Calibration for Camera {target_cam_idx+1}...")
        
        # 1. Gather all points for this camera
        img_points = []
        obj_points = []
        
        # Unique paths that have data for this camera
        relevant_keys = [k for k in self.saved_calibration_data.keys() if k[0] == target_cam_idx]
        
        if not relevant_keys:
            QMessageBox.warning(self, "No Data", f"No calibration data found for Camera {target_cam_idx+1}.\nPlease add data point in detection tab or read from CSV first.")
            self._busy_end('plate_calibration')
            return
            
        for key in relevant_keys:
            data = self.saved_calibration_data[key]
            # Filter out invalid points (None in world_coords from unvisited keypoints)
            valid_pairs = [(kp, wc) for kp, wc in zip(data['keypoints'], data['world_coords']) if wc is not None]
            if not valid_pairs:
                continue
            
            # Convert KeyPoints to numpy Nx2
            kpts = np.array([[kp.pt[0], kp.pt[1]] for kp, _ in valid_pairs], dtype=np.float32)
            world = np.array([wc for _, wc in valid_pairs], dtype=np.float32)
            
            if len(kpts) > 0:
                img_points.append(kpts)
                obj_points.append(world)
                
        if not img_points:
            QMessageBox.warning(self, "No Data", "No valid point sets found.")
            self._busy_end('plate_calibration')
            return

        # Image size
        w = self.cal_img_width.value()
        h = self.cal_img_height.value()
        img_size = (w, h)
        
        # 2. Initialization of Intrinsics
        f_mm = self.init_focal_spin.value()
        pw_mm = self.cal_sensor_width.value() 
        f_px = f_mm / pw_mm
        
        cx_init = w / 2.0
        cy_init = h / 2.0
        
        K = np.array([[f_px, 0, cx_init],
                      [0, f_px, cy_init],
                      [0, 0, 1]], dtype=np.float32)
        
        # Distortion initialization based on UI setting
        dist_num = self.cal_dist_model_combo.currentIndex() if hasattr(self, 'cal_dist_model_combo') else 0
        dist_coeffs = np.zeros(5, dtype=np.float32) # Always use 5 for standard pinhole logic
        
        # 3. Stage 1: Solve for Pose (Extrinsics) using solvePnP for initial guess
        # We pick the first set of points for a rough pose initialization.
        success_pnp, rvec, tvec = cv2.solvePnP(obj_points[0], img_points[0], K, dist_coeffs)
        if not success_pnp:
            QMessageBox.critical(self, "Error", "Initial pose estimation (PnP) failed.")
            self._busy_end('plate_calibration')
            return
            
        # 4. Stage 2: Refine All (Intrinsics + Extrinsics)
        # calibrateCamera expects lists of rvecs/tvecs for initial guess.
        rvecs = [rvec.copy() for _ in range(len(img_points))]
        tvecs = [tvec.copy() for _ in range(len(img_points))]
        
        # Calibration Flags: Always use intrinsic guess
        flags = cv2.CALIB_USE_INTRINSIC_GUESS 
        
        # Standard OpenCV dist order: (k1, k2, p1, p2, k3)
        # We apply FIX flags based on dist_num
        if dist_num == 0:
            flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST
        elif dist_num == 1:
            flags |= cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST
        elif dist_num == 2:
            flags |= cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST
        # dist_num only supports 0/1/2 to match Wand Calibration UI
        
        try:
            rms, K_opt, dist_opt, rvecs_opt, tvecs_opt = cv2.calibrateCamera(
                obj_points, img_points, img_size, K, dist_coeffs, 
                flags=flags
            )
        except Exception as e:
            QMessageBox.critical(self, "Calibration Failed", f"Optimization error: {str(e)}")
            self._busy_end('plate_calibration')
            return
            
        # 5. Display Results
        self.lbl_cal_rms.setText(f"{rms:.4f} px")
        
        # 6. Store Calibration Results
        R_first, _ = cv2.Rodrigues(rvecs_opt[0])
        T_first = tvecs_opt[0]
        
        # Store in accumulation dict (keyed by cam_idx for 0-based, +1 for display)
        self.all_camera_params[target_cam_idx] = {
            'R': R_first,
            'T': T_first,
            'K': K_opt,
            'dist': dist_opt,
            'img_size': (h, w),
            'rms': rms
        }
        
        # 7. Update 3D View with ALL calibrated cameras
        # Reformat for plot_calibration (expects 1-based keys)
        cam_viz_data = {idx + 1: params for idx, params in self.all_camera_params.items()}
        
        # Combine all 3D points from all images into one cloud
        all_3d = np.vstack(obj_points)
        
        self.plate_3d_viewer.plot_calibration(cam_viz_data, all_3d)
        self.plate_vis_tabs.setCurrentWidget(self.plate_3d_viewer)
        
        # 8. Notify user (no per-camera save prompt - use Save All button instead)
        QMessageBox.information(self, "Calibration Complete", 
                               f"Camera {target_cam_idx+1} calibrated successfully.\nRMS: {rms:.4f} px\n\nUse 'Save All Camera Parameters' to export.")
        self._busy_end('plate_calibration')

    def _collect_plate_refraction_inputs(self):
        if not self.saved_calibration_data:
            raise RuntimeError("No calibration data loaded for plate refraction calibration.")

        cam_count = int(self.plate_num_cams.value()) if hasattr(self, 'plate_num_cams') else 0
        cam_ids = list(range(cam_count))

        # Intrinsics from per-camera table
        cam_intrinsics = {}
        for cid in cam_ids:
            st = self.plate_intrinsics_settings.get(cid, self._default_plate_intrinsic_setting())
            cam_intrinsics[cid] = {
                'focal_px': float(st.get('focal_px', 9000.0)),
                'width': int(st.get('width', 1280)),
                'height': int(st.get('height', 800)),
            }

        # Camera-window mapping from table
        cam_to_window = {}
        if not hasattr(self, 'plate_cam_window_table'):
            raise RuntimeError("Camera-window mapping table is missing.")
        for r in range(self.plate_cam_window_table.rowCount()):
            w_combo = self.plate_cam_window_table.cellWidget(r, 1)
            if isinstance(w_combo, QComboBox):
                txt = w_combo.currentText()
                wid = int(txt.split()[-1]) if txt.startswith("Window") else 0
            else:
                wid = 0
            cam_to_window[int(r)] = int(wid)

        used_windows = sorted(set(cam_to_window.values()))

        # Shared media settings for each window (UI currently one set of media controls)
        n1 = float(self.plate_media1_index.value())
        n2 = float(self.plate_media2_index.value())
        n3 = float(self.plate_media3_index.value())
        thickness = float(self.plate_media2_thick.value())
        window_media = {
            int(wid): {
                'n1': n1,
                'n2': n2,
                'n3': n3,
                'thickness': thickness,
                'proj_tol': 1e-6,
                'proj_nmax': 1000,
                'lr': 0.1,
            }
            for wid in used_windows
        }

        # Build merged observations: same (image, world point) merges all cameras
        obs_map = {}
        for (cid, img_path), data in self.saved_calibration_data.items():
            cid = int(cid)
            if cid not in cam_ids:
                continue
            kpts = data.get('keypoints', [])
            worlds = data.get('world_coords', [])
            for kp, w in zip(kpts, worlds):
                if w is None:
                    continue
                wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
                key = (str(img_path), round(wx, 6), round(wy, 6), round(wz, 6))
                if key not in obs_map:
                    obs_map[key] = {
                        'X_world': np.array([wx, wy, wz], dtype=np.float64),
                        'uv_by_cam': {},
                    }
                obs_map[key]['uv_by_cam'][cid] = np.array([float(kp.pt[0]), float(kp.pt[1])], dtype=np.float64)

        observations = list(obs_map.values())
        if len(observations) < 20:
            raise RuntimeError(f"Insufficient observations for refraction calibration: {len(observations)}")

        return observations, cam_intrinsics, cam_to_window, window_media

    def _run_plate_refraction_calibration(self):
        try:
            from .plate_calibration.refraction_plate_calibration import RefractionPlateWorker, RefractionPlateConfig

            observations, cam_intrinsics, cam_to_window, window_media = self._collect_plate_refraction_inputs()

            self._plate_refr_thread = QThread()
            self._plate_refr_worker = RefractionPlateWorker(
                observations=observations,
                cam_intrinsics=cam_intrinsics,
                cam_to_window=cam_to_window,
                window_media=window_media,
                dist_mode=self.cal_dist_model_combo.currentIndex(),
                cfg=RefractionPlateConfig(),
            )
            self._plate_refr_worker.moveToThread(self._plate_refr_thread)

            self._create_plate_refractive_calibration_dialog()

            self._plate_refr_thread.started.connect(self._plate_refr_worker.run)
            self._plate_refr_worker.progress.connect(self._on_plate_refr_progress, Qt.QueuedConnection)
            self._plate_refr_worker.finished.connect(self._on_plate_refr_finished, Qt.QueuedConnection)
            self._plate_refr_worker.error.connect(self._on_plate_refr_error, Qt.QueuedConnection)

            self._plate_refr_worker.finished.connect(self._plate_refr_thread.quit)
            self._plate_refr_worker.finished.connect(self._plate_refr_worker.deleteLater)
            self._plate_refr_worker.error.connect(self._plate_refr_thread.quit)
            self._plate_refr_worker.error.connect(self._plate_refr_worker.deleteLater)
            self._plate_refr_thread.finished.connect(self._plate_refr_thread.deleteLater)

            self._plate_refr_iter_count = 0
            self._plate_refr_last_phase = None
            self._plate_refr_last_metrics = None
            self._plate_refr_timer = QTimer()
            self._plate_refr_timer.setInterval(200)
            self._plate_refr_timer.timeout.connect(self._flush_plate_refr_progress)
            self._plate_refr_timer.start()

            self._plate_refr_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Refraction Calibration Error", str(e))
            self._busy_end('plate_calibration')

    def _create_plate_refractive_calibration_dialog(self):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QApplication

        self._plate_refr_dialog = QDialog(self)
        self._plate_refr_dialog.setWindowTitle("Plate Refraction Calibration")
        self._plate_refr_dialog.setModal(False)
        self._plate_refr_dialog.setMinimumSize(440, 340)
        self._plate_refr_dialog.setStyleSheet("background-color: #000000;")

        layout = QVBoxLayout(self._plate_refr_dialog)
        layout.setContentsMargins(10, 10, 10, 10)

        self._plate_refr_phase_label = QLabel("Initializing...")
        self._plate_refr_phase_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff; background: transparent;")
        self._plate_refr_phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._plate_refr_phase_label.setWordWrap(True)
        layout.addWidget(self._plate_refr_phase_label)

        self._plate_refr_fig = Figure(figsize=(4.2, 2.4), facecolor='#000000')
        self._plate_refr_canvas = FigureCanvas(self._plate_refr_fig)
        self._plate_refr_ax = self._plate_refr_fig.add_subplot(111)
        self._plate_refr_ax.set_facecolor('#000000')
        self._plate_refr_ax.set_xlabel('Iteration', color='white', fontsize=9)
        self._plate_refr_ax.set_ylabel('Proj RMSE (px)', color='#ff5bd2', fontsize=9)
        self._plate_refr_ax.tick_params(colors='white', labelsize=8)
        for spine in self._plate_refr_ax.spines.values():
            spine.set_color('#444')
        self._plate_refr_ax.set_yscale('log')
        self._plate_refr_ax.grid(True, which="both", ls="--", color='#333', alpha=0.5)
        self._plate_refr_proj_line, = self._plate_refr_ax.plot([], [], color='#ff5bd2', marker='o', markersize=3, linewidth=1.5, alpha=0.9, label='Proj')
        self._plate_refr_ax.legend(loc='upper right', fontsize=8, facecolor='#222', edgecolor='#444', labelcolor='white')
        self._plate_refr_fig.subplots_adjust(left=0.16, right=0.95, bottom=0.18, top=0.88)
        layout.addWidget(self._plate_refr_canvas)

        self._plate_refr_proj_label = QLabel("Proj: --")
        self._plate_refr_proj_label.setStyleSheet("font-size: 14px; color: #ff5bd2; background: transparent;")
        self._plate_refr_proj_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._plate_refr_proj_label)

        self._plate_refr_iterations = []
        self._plate_refr_proj_values = []

        self._plate_refr_dialog.show()
        QApplication.processEvents()

    @Slot(str, float, float)
    def _on_plate_refr_progress(self, phase, proj_rmse, cost):
        self._plate_refr_last_phase = str(phase)
        if proj_rmse is not None and proj_rmse > 0:
            self._plate_refr_last_metrics = (float(proj_rmse), float(cost))

    @Slot()
    def _flush_plate_refr_progress(self):
        if hasattr(self, '_plate_refr_last_phase') and self._plate_refr_last_phase is not None:
            if hasattr(self, '_plate_refr_phase_label'):
                self._plate_refr_phase_label.setText(f"Phase: {self._plate_refr_last_phase}")
            self._plate_refr_last_phase = None

        if hasattr(self, '_plate_refr_last_metrics') and self._plate_refr_last_metrics is not None:
            proj_rmse, _cost = self._plate_refr_last_metrics
            self._plate_refr_last_metrics = None

            self._plate_refr_iter_count = getattr(self, '_plate_refr_iter_count', 0) + 1
            it = self._plate_refr_iter_count

            if hasattr(self, '_plate_refr_proj_label'):
                self._plate_refr_proj_label.setText(f"Proj: {proj_rmse:.4f} px")

            self._plate_refr_iterations.append(it)
            self._plate_refr_proj_values.append(max(float(proj_rmse), 1e-12))
            self._plate_refr_proj_line.set_data(self._plate_refr_iterations, self._plate_refr_proj_values)

            x_vals = np.asarray(self._plate_refr_iterations, dtype=float)
            y_vals = np.asarray(self._plate_refr_proj_values, dtype=float)
            if x_vals.size > 0:
                self._plate_refr_ax.set_xlim(max(0.0, float(np.min(x_vals)) - 0.5), float(np.max(x_vals)) + 0.5)
            if y_vals.size > 0:
                y_min = 10.0 ** np.floor(np.log10(max(float(np.min(y_vals)), 1e-12)))
                y_max = 10.0 ** np.ceil(np.log10(max(float(np.max(y_vals)), 1e-12)))
                if y_max <= y_min:
                    y_min *= 0.1
                    y_max *= 10.0
                self._plate_refr_ax.set_ylim(y_min, y_max)
            self._plate_refr_canvas.draw_idle()

    @Slot(bool, object, str)
    def _on_plate_refr_finished(self, success, result, message):
        if hasattr(self, '_plate_refr_timer'):
            self._plate_refr_timer.stop()
        self._flush_plate_refr_progress()

        if hasattr(self, '_plate_refr_dialog') and self._plate_refr_dialog:
            self._plate_refr_dialog.close()
            self._plate_refr_dialog = None

        if not success:
            QMessageBox.critical(self, "Refraction Calibration Failed", message or "Unknown error")
            self._busy_end('plate_calibration')
            return

        self._plate_refraction_result = result

        # Fill legacy all_camera_params for consistency
        self.all_camera_params = {}
        cams = result.get('camera_params', {})
        for cid, cp in cams.items():
            rv = np.asarray(cp['rvec'], dtype=np.float64).reshape(3, 1)
            tv = np.asarray(cp['tvec'], dtype=np.float64).reshape(3, 1)
            R, _ = cv2.Rodrigues(rv)
            K = np.array([[cp['f'], 0.0, cp['cx']], [0.0, cp['f'], cp['cy']], [0.0, 0.0, 1.0]], dtype=np.float64)
            dist = np.array([cp['k1'], cp['k2'], 0.0, 0.0, 0.0], dtype=np.float64).reshape(1, 5)
            self.all_camera_params[int(cid)] = {
                'R': R,
                'T': tv,
                'K': K,
                'dist': dist,
                'img_size': (int(cp['height']), int(cp['width'])),
                'rms': float(result.get('stage_b', {}).get('proj_rmse', 0.0)),
            }

        # Update 3D view (include refractive window planes, same style as wand)
        window_params = result.get('window_params', {})
        window_planes = {
            int(wid): {
                'plane_pt': np.asarray(wp.get('plane_pt', [0.0, 0.0, 0.0]), dtype=np.float64),
                'plane_n': np.asarray(wp.get('plane_n', [0.0, 0.0, 1.0]), dtype=np.float64),
            }
            for wid, wp in window_params.items()
            if isinstance(wp, dict)
        }

        pts = np.asarray(result.get('aligned_points', []), dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3) if pts.size else np.zeros((0, 3), dtype=np.float64)
        self.plate_3d_viewer.plot_refractive(self.all_camera_params, window_planes, pts)
        self.plate_vis_tabs.setCurrentWidget(self.plate_3d_viewer)

        s1 = result.get('stage_a', {})
        s2 = result.get('stage_b', {})
        print(
            "[PlateRefr] Stage A proj RMSE: "
            f"[{float(s1.get('proj_rmse_before', 0.0)):.6f}] -> [{float(s1.get('proj_rmse_after', 0.0)):.6f}] px"
        )
        print(
            "[PlateRefr] Stage B proj RMSE: "
            f"[{float(s2.get('proj_rmse_before', 0.0)):.6f}] -> [{float(s2.get('proj_rmse_after', 0.0)):.6f}] px"
        )
        print(
            "[PlateRefr] Stop condition A: "
            f"{s1.get('stop_reason', s1.get('message', 'N/A'))}"
        )
        print(
            "[PlateRefr] Stop condition B: "
            f"{s2.get('stop_reason', s2.get('message', 'N/A'))}"
        )

        QMessageBox.information(
            self,
            "Refraction Calibration Complete",
            f"Stage A proj RMSE: [{float(s1.get('proj_rmse_before', 0.0)):.4f}] -> [{float(s1.get('proj_rmse_after', 0.0)):.4f}] px\n"
            f"Stage B proj RMSE: [{float(s2.get('proj_rmse_before', 0.0)):.4f}] -> [{float(s2.get('proj_rmse_after', 0.0)):.4f}] px\n\n"
            f"Stop A: {s1.get('stop_reason', s1.get('message', 'N/A'))}\n"
            f"Stop B: {s2.get('stop_reason', s2.get('message', 'N/A'))}\n\n"
            f"Use 'Save All Camera Parameters' to export PINPLATE files.",
        )
        self._busy_end('plate_calibration')

    @Slot(str)
    def _on_plate_refr_error(self, msg):
        if hasattr(self, '_plate_refr_timer'):
            self._plate_refr_timer.stop()
        if hasattr(self, '_plate_refr_dialog') and self._plate_refr_dialog:
            self._plate_refr_dialog.close()
            self._plate_refr_dialog = None
        QMessageBox.critical(self, "Refraction Calibration Error", msg)
        self._busy_end('plate_calibration')

    def _save_plate_calibration_params(self, cam_idx, K, dist, R, T, img_size):
        """Save calibrated parameters in OpenLPT format (consistent with Wand calibration)."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Camera Parameters", 
                                                 f"camera_{cam_idx+1}_params.txt", 
                                                 "Text Files (*.txt)")
        if not file_path:
            return
            
        try:
            with open(file_path, 'w') as f:
                f.write("# Camera Model: (PINHOLE/POLYNOMIAL)\n")
                f.write("PINHOLE\n")
                f.write("# Camera Calibration Error: \n")
                f.write("None\n")
                f.write("# Pose Calibration Error: \n")
                f.write("None\n")
                f.write("# Image Size: (n_row,n_col)\n")
                f.write(f"{img_size[1]},{img_size[0]}\n") # H, W
                
                f.write("# Camera Matrix: \n")
                f.write(f"{K[0,0]},{K[0,1]},{K[0,2]}\n")
                f.write(f"{K[1,0]},{K[1,1]},{K[1,2]}\n")
                f.write(f"{K[2,0]},{K[2,1]},{K[2,2]}\n")
                
                f.write("# Distortion Coefficients: \n")
                dist_str = ",".join(map(str, dist.flatten()))
                f.write(f"{dist_str}\n")
                
                f.write("# Rotation Vector: \n")
                r_vec, _ = cv2.Rodrigues(R)
                f.write(f"{r_vec[0,0]},{r_vec[1,0]},{r_vec[2,0]}\n")
                
                f.write("# Rotation Matrix: \n")
                f.write(f"{R[0,0]},{R[0,1]},{R[0,2]}\n")
                f.write(f"{R[1,0]},{R[1,1]},{R[1,2]}\n")
                f.write(f"{R[2,0]},{R[2,1]},{R[2,2]}\n")
                
                f.write("# Inverse of Rotation Matrix: \n")
                r_inv = R.T
                f.write(f"{r_inv[0,0]},{r_inv[0,1]},{r_inv[0,2]}\n")
                f.write(f"{r_inv[1,0]},{r_inv[1,1]},{r_inv[1,2]}\n")
                f.write(f"{r_inv[2,0]},{r_inv[2,1]},{r_inv[2,2]}\n")
                
                f.write("# Translation Vector: \n")
                # Ensure T is formatted correctly as a single vector
                f.write(f"{T[0][0]},{T[1][0]},{T[2][0]}\n")
                
                f.write("# Inverse of Translation Vector: \n")
                t_inv = -r_inv @ T
                f.write(f"{t_inv[0][0]},{t_inv[1][0]},{t_inv[2][0]}\n")
                
            QMessageBox.information(self, "Success", f"Parameters saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save parameters: {str(e)}")

    def _save_all_camera_params(self):
        """Save all calibrated camera parameters to a camFile folder."""
        if hasattr(self, 'plate_model_combo') and self.plate_model_combo.currentText() == "Pinhole+Refraction":
            result = getattr(self, '_plate_refraction_result', None)
            if not result or not result.get('success', False):
                QMessageBox.warning(self, "No Data", "No refraction plate calibration result available.")
                return

            folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if not folder:
                return

            from pathlib import Path
            cam_folder = Path(folder) / "camFile"
            cam_folder.mkdir(parents=True, exist_ok=True)

            try:
                from .wand_calibration.refraction_wand_calibrator import CamFileExporter

                cam_params = {}
                camera_settings = {}
                for cid, cp in result.get('camera_params', {}).items():
                    cid_i = int(cid)
                    cam_params[cid_i] = np.array([
                        float(cp['rvec'][0]), float(cp['rvec'][1]), float(cp['rvec'][2]),
                        float(cp['tvec'][0]), float(cp['tvec'][1]), float(cp['tvec'][2]),
                        float(cp['f']), float(cp['cx']), float(cp['cy']),
                        float(cp['k1']), float(cp['k2'])
                    ], dtype=np.float64)
                    camera_settings[cid_i] = {
                        'height': int(cp['height']),
                        'width': int(cp['width']),
                    }

                window_media = {}
                for wid, wp in result.get('window_params', {}).items():
                    window_media[int(wid)] = {
                        'n1': float(wp['n1']),
                        'n2': float(wp['n2']),
                        'n3': float(wp['n3']),
                        'thickness': float(wp['thickness']),
                        'n_air': float(wp['n1']),
                        'n_window': float(wp['n2']),
                        'n_object': float(wp['n3']),
                    }

                window_planes = {
                    int(wid): {
                        'plane_pt': np.asarray(wp['plane_pt'], dtype=np.float64),
                        'plane_n': np.asarray(wp['plane_n'], dtype=np.float64),
                        'initialized': True,
                    }
                    for wid, wp in result.get('window_params', {}).items()
                }

                cam_to_window = {int(k): int(v) for k, v in result.get('cam_to_window', {}).items()}
                proj_stats = {int(k): tuple(v) for k, v in result.get('per_camera_proj_err_stats', {}).items()}
                tri_stats = {int(k): tuple(v) for k, v in result.get('per_camera_tri_err_stats', {}).items()}

                class _BaseStub:
                    pass

                base_stub = _BaseStub()
                base_stub.camera_settings = camera_settings
                base_stub.image_size = (
                    int(next(iter(camera_settings.values()))['height']) if camera_settings else 800,
                    int(next(iter(camera_settings.values()))['width']) if camera_settings else 1280,
                )

                CamFileExporter.export_camfile_with_refraction(
                    base=base_stub,
                    out_dir=str(cam_folder),
                    cam_params=cam_params,
                    window_media=window_media,
                    cam_to_window=cam_to_window,
                    window_planes=window_planes,
                    proj_err_stats=proj_stats,
                    tri_err_stats=tri_stats,
                )

                QMessageBox.information(self, "Success", f"Saved refraction plate camera files to:\n{cam_folder}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save refraction plate parameters: {str(e)}")
            return

        if not self.all_camera_params:
            QMessageBox.warning(self, "No Data", "No cameras have been calibrated yet.")
            return
            
        # Prompt for directory
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not folder:
            return
            
        from pathlib import Path
        cam_folder = Path(folder) / "camFile"
        cam_folder.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        try:
            for cam_idx, params in sorted(self.all_camera_params.items()):
                file_path = cam_folder / f"cam{cam_idx}.txt"
                
                K = params['K']
                dist = params['dist']
                R = params['R']
                T = params['T']
                img_size = params['img_size']
                
                with open(file_path, 'w') as f:
                    f.write("# Camera Model: (PINHOLE/POLYNOMIAL)\n")
                    f.write("PINHOLE\n")
                    f.write("# Camera Calibration Error: \n")
                    f.write("None\n")
                    f.write("# Pose Calibration Error: \n")
                    f.write("None\n")
                    f.write("# Image Size: (n_row,n_col)\n")
                    f.write(f"{img_size[0]},{img_size[1]}\n")  # H, W
                    
                    f.write("# Camera Matrix: \n")
                    f.write(f"{K[0,0]},{K[0,1]},{K[0,2]}\n")
                    f.write(f"{K[1,0]},{K[1,1]},{K[1,2]}\n")
                    f.write(f"{K[2,0]},{K[2,1]},{K[2,2]}\n")
                    
                    f.write("# Distortion Coefficients: \n")
                    dist_str = ",".join(map(str, dist.flatten()))
                    f.write(f"{dist_str}\n")
                    
                    f.write("# Rotation Vector: \n")
                    r_vec, _ = cv2.Rodrigues(R)
                    f.write(f"{r_vec[0,0]},{r_vec[1,0]},{r_vec[2,0]}\n")
                    
                    f.write("# Rotation Matrix: \n")
                    f.write(f"{R[0,0]},{R[0,1]},{R[0,2]}\n")
                    f.write(f"{R[1,0]},{R[1,1]},{R[1,2]}\n")
                    f.write(f"{R[2,0]},{R[2,1]},{R[2,2]}\n")
                    
                    f.write("# Inverse of Rotation Matrix: \n")
                    r_inv = R.T
                    f.write(f"{r_inv[0,0]},{r_inv[0,1]},{r_inv[0,2]}\n")
                    f.write(f"{r_inv[1,0]},{r_inv[1,1]},{r_inv[1,2]}\n")
                    f.write(f"{r_inv[2,0]},{r_inv[2,1]},{r_inv[2,2]}\n")
                    
                    f.write("# Translation Vector: \n")
                    f.write(f"{T[0][0]},{T[1][0]},{T[2][0]}\n")
                    
                    f.write("# Inverse of Translation Vector: \n")
                    t_inv = -r_inv @ T
                    f.write(f"{t_inv[0][0]},{t_inv[1][0]},{t_inv[2][0]}\n")
                    
                saved_count += 1
                
            QMessageBox.information(self, "Success", 
                                   f"Saved {saved_count} camera(s) to:\n{cam_folder}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save parameters: {str(e)}")
    
    def _detect_single_frame(self, checked=False):
        # Run detection on current frame only and visualize (Async)
        from .wand_calibration.wand_calibrator import WandCalibrator, WandDetectionSingleFrameWorker
        import cv2
        import numpy as np
        from PySide6.QtGui import QPixmap, QImage
        from PySide6.QtWidgets import QProgressDialog
        
        if not hasattr(self, 'wand_calibrator'):
             self.wand_calibrator = WandCalibrator()
             
        idx = self.frame_table.currentRow()
        if idx < 0:
            self.status_label.setText("Please select a frame first.")
            return

        wand_type = "dark" if "Dark" in self.wand_type_combo.currentText() else "bright"
        min_r, max_r = self.radius_range.value()
        sensitivity = self.sensitivity_slider.value()
        detect_mode = self._get_wand_detect_mode()
        
        # Create dict for single frame
        single_frame_dict = {}
        for c, files in self.wand_images.items():
            if idx < len(files):
                 single_frame_dict[c] = files[idx] 
        
        if not single_frame_dict:
            return

        self._busy_begin('wand_detect_single', 'Detecting single frame')

        print(
            f"Detecting frame {idx}, mode='{wand_type}', radius=[{min_r},{max_r}], "
            f"sensitivity={sensitivity}, detect_mode='{detect_mode}'"
        )
        
        # Worker Setup
        self._single_detect_worker = WandDetectionSingleFrameWorker(
            self.wand_calibrator, single_frame_dict, wand_type, min_r, max_r, sensitivity, detect_mode
        )
        self._single_detect_worker.finished_signal.connect(self._on_single_detection_finished)
        
        # Dialog Setup
        self._detect_dialog = QProgressDialog("Detecting points...", None, 0, 0, self)
        self._detect_dialog.setWindowTitle("Please Wait")
        self._detect_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._detect_dialog.setMinimumDuration(0) # Show immediately
        self._detect_dialog.setStyleSheet("""
            QProgressDialog { background-color: #2b2b2b; color: #ffffff; padding: 15px; border: 1px solid #444; }
            QLabel { color: #ffffff; font-size: 13px; font-weight: bold; background-color: transparent; }
            QProgressBar { 
                min-height: 12px; max-height: 12px; margin: 10px 15px; 
                background-color: #444; border-radius: 4px; text-align: center; color: white;
            }
            QProgressBar::chunk { background-color: #00bcd4; border-radius: 4px; }
        """)
        self._detect_dialog.show()
        
        # Connect finish to close
        self._single_detect_worker.finished_signal.connect(self._detect_dialog.close)
        self._single_detect_worker.finished_signal.connect(self._single_detect_worker.deleteLater)
        
        self._single_detect_worker.start()

    def _on_single_detection_finished(self, res):
        """Handle async detection result."""
        # print(f"Detection results: {res}")
        
        import cv2
        import numpy as np
        from PySide6.QtGui import QPixmap, QImage
        
        idx = self.frame_table.currentRow()
        
        # Visualize result for each camera
        total_points = 0
        for cam_idx, pts in res.items():
            if cam_idx in self.cam_vis_labels:
                lbl = self.cam_vis_labels[cam_idx]
                if cam_idx in self.wand_images and idx < len(self.wand_images[cam_idx]):
                    path = self.wand_images[cam_idx][idx]
                    
                    # Load image with OpenCV
                    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    
                    # Convert to RGB
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Normalize to 8-bit if needed
                    if img.dtype != np.uint8:
                        img = (img / img.max() * 255).astype(np.uint8)
                    
                    # Draw ALL detected circles with radius labels
                    if pts is not None and len(pts) > 0:
                        for pt in pts:
                            x, y = int(pt[0]), int(pt[1])
                            r = int(pt[2]) if len(pt) > 2 else 15
                            # Draw circle
                            cv2.circle(img, (x, y), r, (0, 100, 255), 3)
                            # Draw radius label next to circle
                            label = f"r={r}"
                            cv2.putText(img, label, (x + r + 5, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        total_points += len(pts)
                    
                    # Convert to QPixmap
                    h, w, ch = img.shape
                    bytes_per_line = ch * w
                    q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pix = QPixmap.fromImage(q_img)
                    
                    # Scale and Set
                    lbl_size = lbl.size()
                    if lbl_size.width() < 50 or lbl_size.height() < 50:
                        lbl_size = self.vis_tabs.size()
                    
                    scaled = pix.scaled(lbl_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    lbl.setPixmap(scaled)
        
        self.status_label.setText(f"Frame {idx}: Found {total_points} points in {len(res)} cameras.")
        self._busy_end('wand_detect_single')

    def _process_wand_frames(self, checked=False):
        from .wand_calibration.wand_calibrator import WandCalibrator, WandDetectionWorker
        from .widgets import ProcessingDialog
        from PySide6.QtWidgets import QMessageBox, QFileDialog, QDialog
        from PySide6.QtCore import QFileInfo
        
        if not hasattr(self, 'wand_calibrator'):
             self.wand_calibrator = WandCalibrator()
        
        # Check if we have images
        count = sum(len(imgs) for imgs in self.wand_images.values())
        if count == 0:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return

        # 0. Info Popup (UX Improvement)
        QMessageBox.information(self, "Process Frames", 
                                "Please select the path to save the detection results.\n\n"
                                "You can also select a previously-saved results file to RESUME processing.",
                                QMessageBox.StandardButton.Ok)

        # 1. Prompt for Save Path (Autosave)
        autosave_path, _ = QFileDialog.getSaveFileName(self, "Select Save File (Autosave)", "wand_points.csv", "CSV Files (*.csv)")
        if not autosave_path:
            return # User cancelled
            
        resume = False
        # 2. Check if file exists -> Resume?
        if QFileInfo(autosave_path).exists():
            # Ask user
            reply = QMessageBox.question(self, "Resume?", 
                                         f"File '{QFileInfo(autosave_path).fileName()}' exists.\nDo you want to RESUME processing from where it left off?\n\nYes: Resume (keep existing data)\nNo: Overwrite (restart)",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            
            if reply == QMessageBox.StandardButton.Cancel:
                return
            
            if reply == QMessageBox.StandardButton.Yes:
                resume = True
                # Data loading moved to Worker thread to prevent UI freeze
        
        # Get UI parameters
        wand_type = "dark" if "Dark" in self.wand_type_combo.currentText() else "bright"
        min_r, max_r = self.radius_range.value()
        sensitivity = self.sensitivity_slider.value()
        detect_mode = self._get_wand_detect_mode()

        # Start Worker
        self._proc_thread = WandDetectionWorker(
            self.wand_calibrator, 
            self.wand_images, 
            wand_type, min_r, max_r, sensitivity,
            autosave_path=autosave_path,
            resume=resume,
            detect_mode=detect_mode,
        )
        
        # Setup dialog
        self._proc_dialog = ProcessingDialog(self)
        self._proc_dialog.stop_signal.connect(self._proc_thread.stop)
        
        # Correctly handle Pause/Resume toggle
        self._proc_dialog.pause_signal.connect(lambda p: self._proc_thread.pause() if p else self._proc_thread.resume())
        # self._proc_dialog.resume_signal.connect(self._proc_thread.resume) # Removed invalid signal
        
        self._proc_thread.progress.connect(self._proc_dialog.update_progress)
        self._proc_thread.finished_signal.connect(self._on_process_finished)
        self._busy_begin('wand_process_all', 'Processing wand frames')
        
        self._proc_thread.start()
        self._proc_dialog.exec() # Modal blocking
        
    def _on_process_finished(self, success, msg):
        self._proc_dialog.close()
        self._busy_end('wand_process_all')
        
        if success:
             self.status_label.setText(f"Done: {msg}")
             self.populate_wand_table()
             self.populate_wand_table()
             
             # Update Image Size UI if available (Note: these spinboxes are in the camera table now)
             # if hasattr(self.wand_calibrator, 'image_size') and self.wand_calibrator.image_size != (0,0):
             #     h, w = self.wand_calibrator.image_size
             #     # Image size is now read from camera table, not separate spinboxes
             
             # Data is auto-saved, no need to prompt export here
        else:
             self.status_label.setText(f"Status: {msg}")
             
             # If stopped by user, don't show error popup (Data already saved by worker)
             if "Stopped by user" in msg:
                 self.populate_wand_table()
                 return # Done
             
             # Check if we have partial data
             has_data = False
             if hasattr(self, 'wand_calibrator'):
                 if hasattr(self.wand_calibrator, 'wand_data_raw') and len(self.wand_calibrator.wand_data_raw) > 0:
                     has_data = True
                 elif hasattr(self.wand_calibrator, 'wand_points') and len(self.wand_calibrator.wand_points) > 0:
                     has_data = True
             
             if has_data:
                 from PySide6.QtWidgets import QMessageBox, QFileDialog
                 reply = QMessageBox.question(self, "Detection Failed", 
                                              f"Error: {msg}\n\nHowever, partial data was captured ({len(self.wand_calibrator.wand_data_raw)} frames). Do you want to export it?",
                                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                 if reply == QMessageBox.StandardButton.Yes:
                     path, _ = QFileDialog.getSaveFileName(self, "Save Rescue Data", "wand_rescue.csv", "CSV Files (*.csv)")
                     if path:
                         self.wand_calibrator.export_wand_data(path)
             else:
                 from PySide6.QtWidgets import QMessageBox
                 QMessageBox.warning(self, "Detection Failed", msg)
        from PySide6.QtWidgets import QMessageBox, QFileDialog
        
        reply = QMessageBox.question(self, "Export Data", "Detection complete. Do you want to export the point data?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
             path, _ = QFileDialog.getSaveFileName(self, "Save Wand Data", "wand_points.csv", "CSV Files (*.csv)")
             if path:
                 success, msg = self.wand_calibrator.export_wand_data(path)
                 if success:
                     QMessageBox.information(self, "Export", "Data exported successfully.")
                 else:
                     QMessageBox.warning(self, "Export Failed", msg)
        
    def _apply_current_ui_filter(self):
        """Collect checked frames from table and apply to calibrator."""
        from PySide6.QtWidgets import QCheckBox
        
        # Initialize persistent set if not exists
        if not hasattr(self, '_removed_frames'):
            self._removed_frames = set()
        
        if not hasattr(self, 'frozen_table'):
             return

        # Read currently checked frames from UI
        currently_checked = set()
        for row in range(self.frozen_table.rowCount()):
            widget = self.frozen_table.cellWidget(row, 0)
            if widget:
                chk = widget.findChild(QCheckBox)
                if chk and chk.isChecked():
                    fid = chk.property('frame_id')
                    if fid is not None:
                         currently_checked.add(int(fid))
        
        # Accumulate: add newly checked frames to persistent set
        # (Frames already removed stay removed)
        self._removed_frames.update(currently_checked)
        
        # Apply accumulated removal
        self.wand_calibrator.reset_filter()
        if self._removed_frames:
             print(f"[Filter] UI: removing {len(self._removed_frames)} frames total: {sorted(list(self._removed_frames))}")
             remaining = self.wand_calibrator.apply_filter(self._removed_frames)
             print(f"[Filter] After apply_filter: {remaining} frames remaining in wand_points_filtered")
        else:
             print(f"[Filter] UI: no frames marked for removal, using all {len(self.wand_calibrator.wand_points)} frames")
            
    def _run_wand_calibration(self, checked=False, precalibrate=False):
        if not hasattr(self, 'wand_calibrator'):
             self.status_label.setText("Please detect points first.")
             return

        # Read per-camera settings from table
        camera_settings = self._collect_camera_settings_from_table()
        
        # Use first camera's image size for the calibrator (or largest resolution)
        if camera_settings:
            first_cam = list(camera_settings.values())[0]
            h, w = first_cam['height'], first_cam['width']
            self.wand_calibrator.image_size = (h, w)
            init_focal = first_cam['focal']
            # Store all camera settings for per-camera focal length initialization
            self.wand_calibrator.camera_settings = camera_settings
            print(f"Using Camera Settings: image_size=({h},{w}), init_focal={init_focal}")
            print(f"  Per-camera focal lengths: {[(k, v['focal']) for k, v in camera_settings.items()]}")
        else:
            QMessageBox.warning(self, "No Cameras", "No camera settings found.")
            return

        wand_len = self.wand_len_spin.value()
        
        # Auto-filter: collect marked frames from error table and apply filter
        self._apply_current_ui_filter()
        
        # NOTE: Do NOT reset filter if frames_to_remove is empty. 
        # This allows cumulative filtering (i.e., previously removed frames stay removed).
        
        dist_coeff_num = self.dist_model_combo.currentIndex()
        print(f"[Wand] distortion_params={dist_coeff_num}")
        print(f"Running Wand Calibration with length {wand_len}mm, f0={init_focal}px, dist_coeffs={dist_coeff_num}...")
        self.status_label.setText(f"Calibrating (L={wand_len}mm, f0={init_focal}px, k={dist_coeff_num})...")
        self._busy_begin('wand_calibration', 'Running wand calibration')
    
        # Store precalibration state
        self._is_precalibrating = precalibrate

        # --- Dispatch based on Camera Model ---
        model_name = self.wand_model_combo.currentText()
        
        if model_name == "Pinhole+Refraction" and not precalibrate:
            self._run_refractive_wand_calibration(wand_len, dist_coeff_num)
            return
        
        # Calculate num_cams based on actual data in wand_points, NOT the UI table
        # User might have 4 cameras in UI but only imported data for 2.
        if self.wand_calibrator.wand_points:
            unique_cams = set()
            # Optimization: Check first 100 frames or all? 
            # All is safer and fast enough for typical calibration sizes.
            for obs in self.wand_calibrator.wand_points.values():
                unique_cams.update(obs.keys())
            self._calib_num_cams = len(unique_cams)
            print(f"[UI] Active cameras in wand data: {self._calib_num_cams} {sorted(list(unique_cams))}")
        else:
            self._calib_num_cams = len(camera_settings) if camera_settings else 0
    
        # Disable button to prevent double click
        sender = self.sender()
        if sender: sender.setEnabled(False)
        self._calib_btn_sender = sender
        
        self.wand_calibrator.dist_coeff_num = dist_coeff_num
        
        # Use Worker
        from .wand_calibration.wand_calibrator import CalibrationWorker
        self._calib_worker = CalibrationWorker(self.wand_calibrator, wand_len, init_focal, precalibrate=precalibrate)
        self._calib_worker.finished_signal.connect(self._on_calibration_finished)
        self._calib_worker.cost_signal.connect(self._on_cost_update)
        
        self._calib_worker.start()
        
        # Create custom progress dialog with cost plot
        self._create_calibration_dialog()

    def _collect_camera_settings_from_table(self):
        """Collect per-camera settings from detection table.

        Returns:
            dict[int, dict]: {cam_id: {'focal': float, 'width': int, 'height': int}}
        """
        camera_settings = {}
        if not hasattr(self, 'wand_cam_table'):
            return camera_settings

        num_cams = self.wand_cam_table.rowCount()
        for row in range(num_cams):
            id_item = self.wand_cam_table.item(row, 1)
            if id_item is None:
                continue
            try:
                cam_id = int(id_item.text())
            except Exception:
                cam_id = row

            focal_spin = self.wand_cam_table.cellWidget(row, 2)
            width_spin = self.wand_cam_table.cellWidget(row, 3)
            height_spin = self.wand_cam_table.cellWidget(row, 4)

            focal = float(focal_spin.value()) if focal_spin else 9000.0
            width = int(width_spin.value()) if width_spin else 1280
            height = int(height_spin.value()) if height_spin else 800

            camera_settings[cam_id] = {
                'focal': focal,
                'width': width,
                'height': height,
            }
        return camera_settings
    
    def _save_filtered_points(self):
        """Save wand points to CSV. Includes all 'Raw' points and 'Filtered_Small/Large' for active frames."""
        if not hasattr(self, 'wand_calibrator'):
             QMessageBox.warning(self, "No Data", "No calibration data available.")
             return

        # Apply current UI filter marks before saving
        self._apply_current_ui_filter()
        
        raw_points = self.wand_calibrator.wand_points
        if not raw_points:
            QMessageBox.information(self, "Info", "No data points to save.")
            return

        # Determine which frames are kept (Raw) vs filtered
        if self.wand_calibrator.wand_points_filtered is not None:
            kept_frames = set(self.wand_calibrator.wand_points_filtered.keys())
        else:
            kept_frames = set(raw_points.keys()) # All kept

        # Prompt for save file (remembering last directory persistently)
        import os
        from PySide6.QtCore import QSettings
        settings = QSettings("OpenLPT", "GUI")
        init_dir = settings.value("last_wand_save_dir", "")
        print(f"[DEBUG] QSettings loaded last_wand_save_dir: '{init_dir}'")
        
        default_filename = "wandpoints_filtered.csv"
        if init_dir and isinstance(init_dir, str) and os.path.exists(init_dir):
             init_path = os.path.join(init_dir, default_filename)
             print(f"[DEBUG] Using init_path: '{init_path}'")
        else:
             init_path = default_filename
        
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Points Data", init_path, "CSV Files (*.csv)")
        if not filepath:
            return
        
        # Remember directory persistently
        save_dir = os.path.dirname(filepath)
        save_dir = os.path.normpath(save_dir)
        settings.setValue("last_wand_save_dir", save_dir)
        settings.sync() # Force write
        print(f"[DEBUG] Saved last_wand_save_dir: '{save_dir}'")
        
        self.last_save_dir = save_dir
            
        # Collect data
        all_rows = []
        
        # Columns: Frame, Camera, Status, PointIdx, X, Y, Radius, Metric
        # Iterate over RAW points (superset)
        for fid in sorted(raw_points.keys()):
            is_kept = (fid in kept_frames)
            
            cams = raw_points[fid]
            for cam_idx in sorted(cams.keys()):
                pts_list = cams[cam_idx]
                
                # 1. Write RAW entries (ALL points)
                for p_idx, pt in enumerate(pts_list):
                    x, y = pt[0], pt[1]
                    r = pt[2] if len(pt) > 2 else 0
                    row = [fid, cam_idx, "Raw", p_idx, f"{x:.4f}", f"{y:.4f}", f"{r:.4f}", f"{0.0:.4f}"]
                    all_rows.append(row)
                
                # 2. Write FILTERED entries (Only kept frames)
                if is_kept and len(pts_list) >= 2:
                    # Sort by radius to identify Small/Large
                    # (Note: This assumes we are dealing with a standard 2-point wand)
                    pts_sorted = sorted(pts_list, key=lambda p: p[2] if len(p) > 2 else 0)
                    
                    # We expect exactly 2 points for a valid wand frame.
                    # If more, we might take the smallest and largest? 
                    # For now, let's take the first two (Smallest) and label them? 
                    # Or assume exactly 2.
                    # Loader logic checks len(pts) != 2 => invalid. 
                    # So we should probably output exactly 2 if we want it to work.
                    
                    # Let's take the smallest and the largest? Or just the first two?
                    # Screenshot implies Small and Large.
                    pt_small = pts_sorted[0]
                    pt_large = pts_sorted[-1] # Largest
                    
                    # Small
                    x, y, r = pt_small[0], pt_small[1], (pt_small[2] if len(pt_small)>2 else 0)
                    row_s = [fid, cam_idx, "Filtered_Small", 0, f"{x:.4f}", f"{y:.4f}", f"{r:.4f}", f"{0.0:.4f}"]
                    all_rows.append(row_s)
                    
                    # Large
                    x, y, r = pt_large[0], pt_large[1], (pt_large[2] if len(pt_large)>2 else 0)
                    row_l = [fid, cam_idx, "Filtered_Large", 1, f"{x:.4f}", f"{y:.4f}", f"{r:.4f}", f"{0.0:.4f}"]
                    all_rows.append(row_l)
                    
        # Write
        import csv
        header = ["Frame", "Camera", "Status", "PointIdx", "X", "Y", "Radius", "Metric"]
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(all_rows)
            
            QMessageBox.information(self, "Success", f"Saved {len(all_rows)} points to:\n{filepath}")
        except Exception as e:
             QMessageBox.critical(self, "Error", f"Failed to save file:\n{e}")

    def _create_calibration_dialog(self):
        """Create a dialog with matplotlib cost plot."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        
        self._calib_dialog = QDialog(self)
        self._calib_dialog.setWindowTitle("Calibrating...")
        self._calib_dialog.setModal(True)
        self._calib_dialog.setMinimumSize(400, 300)
        self._calib_dialog.setStyleSheet("background-color: #000000;")
        
        layout = QVBoxLayout(self._calib_dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Matplotlib figure for cost curve
        self._cost_fig = Figure(figsize=(4, 2.5), facecolor='#000000')
        self._cost_canvas = FigureCanvasQTAgg(self._cost_fig)
        self._cost_ax = self._cost_fig.add_subplot(111)
        self._cost_ax.set_facecolor('#000000')
        self._cost_ax.set_xlabel('Iteration', color='white', fontsize=9)
        self._cost_ax.set_ylabel('Cost', color='white', fontsize=9)
        self._cost_ax.tick_params(colors='white', labelsize=8)
        for spine in self._cost_ax.spines.values():
            spine.set_color('#444')
        self._cost_ax.set_title('Cost vs Iteration', color='white', fontsize=10)
        
        # Initialize data and line
        self._cost_iterations = []
        self._cost_values = []
        self._cost_line, = self._cost_ax.plot([], [], 'c-', linewidth=1.5)
        
        self._cost_fig.tight_layout()
        layout.addWidget(self._cost_canvas)
        
        # Status label (below plot, centered)
        self._calib_status_label = QLabel("Running Bundle Adjustment... Please wait.")
        self._calib_status_label.setStyleSheet("font-size: 12px; color: #00d4ff; background: transparent;")
        self._calib_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._calib_status_label)
    
        # Stop Button
        self._stop_calib_btn = QPushButton("Stop")
        self._stop_calib_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 5px;")
        self._stop_calib_btn.clicked.connect(self._stop_calibration)
        self._stop_calib_btn.setEnabled(False)  # Disabled until Phase 3
        layout.addWidget(self._stop_calib_btn)
        
        self._calib_dialog.show()
    
    def _stop_calibration(self):
        """Request stop of running calibration."""
        if hasattr(self, '_calib_worker') and self._calib_worker.isRunning():
            self._calib_status_label.setText("Stopping calibration...")
            self._calib_worker.stop()
            # Disable button to indicate request sent
            self._stop_calib_btn.setEnabled(False)
    
    def _on_cost_update(self, phase, stage, rmse):
        """Called when optimizer reports new cost value."""
        # Ensure main event loop processes events (keeps UI responsive)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Determine iteration (hacky counter since we removed iter from signal)
        if hasattr(self, '_cost_iter_count'):
             self._cost_iter_count += 1
        else:
             self._cost_iter_count = 1
             
        iteration = self._cost_iter_count
        
        if not hasattr(self, '_cost_iterations'):
            return
        
        self._cost_iterations.append(iteration)
        self._cost_values.append(rmse) # Plot RMSE instead of cost (more intuitive?) or keep cost? 
        # Actually user didn't ask to change plot, just text. 
        # But previous signal was (iteration, cost, rmse). Now (phase, stage, rmse).
        # We lost 'cost'. Let's plot RMSE then.
        
        # Update line data (efficient - no recreating)
        self._cost_line.set_data(self._cost_iterations, self._cost_values)
        self._cost_ax.relim()
        self._cost_ax.autoscale_view()
        
        # Update status label
        if hasattr(self, '_calib_status_label'):
             if getattr(self, '_is_precalibrating', False) or phase == "Pre-Calibration":
                 self._calib_status_label.setText(f"Proj error: {rmse:.4f} px.")
             else:
                 self._calib_status_label.setText(f"{phase}, {stage}, Proj error: {rmse:.4f} px.")
        
        if hasattr(self, '_stop_calib_btn'):
            should_enable = False
            
            # Condition 1: Phase 3 or Refinement (always allow stop)
            if "Phase 3" in phase or "Refinement" in phase:
                should_enable = True
            # Condition 2: 2 Cameras (allow stop in Phase 1 since it's the only phase)
            elif hasattr(self, '_calib_num_cams') and self._calib_num_cams == 2:
                should_enable = True
                
            if should_enable:
                self._stop_calib_btn.setEnabled(True)
                self._stop_calib_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 5px;")
            else:
                self._stop_calib_btn.setEnabled(False)
                self._stop_calib_btn.setStyleSheet("background-color: #555; color: #888; font-weight: bold; padding: 5px;")
        
        # Redraw canvas
        self._cost_canvas.draw_idle()

    def _on_calibration_finished(self, success, msg, res=None):
        from PySide6.QtWidgets import QMessageBox, QFileDialog
        
        self._calib_dialog.close()
        if hasattr(self, '_calib_btn_sender') and self._calib_btn_sender:
            self._calib_btn_sender.setEnabled(True)
            
        print(f"Calibration Result: {success}, {msg}")
        
        if success:
            self.status_label.setText("Calibration Successful!")
            
            # Update 3D Visualization and Error Table FIRST so user can see results in background
            self._update_3d_viz()
            self._populate_error_table()
            
            # If Pre-calibration, SKIP save prompt
            if getattr(self, '_is_precalibrating', False):
                QMessageBox.information(self, "Check Complete", "Pre-calibration check complete.\nReview errors in the table.")
                self._busy_end('wand_calibration')
                return

            # Prompt to save with custom buttons
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Calibration Result")
            msg_box.setText(f"Calibration Optimized Successfully!\n\n{msg}\n\nDo you want to save the camera parameters?")
            btn_save = msg_box.addButton("Save", QMessageBox.AcceptRole)
            btn_not_save = msg_box.addButton("Not Save", QMessageBox.RejectRole)
            msg_box.exec()
            
            if msg_box.clickedButton() == btn_save:
                # Prompt to save
                output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Camera Parameters")
                if output_dir:
                    from pathlib import Path
                    # Create camFile subfolder
                    cam_file_dir = Path(output_dir) / "camFile"
                    cam_file_dir.mkdir(parents=True, exist_ok=True)
                    
                    for cam_idx in self.wand_calibrator.final_params.keys():
                        path = cam_file_dir / f"cam{cam_idx}.txt"
                        self.wand_calibrator.export_to_file(cam_idx, str(path))
                    print(f"Parameters saved to {cam_file_dir}")
                    QMessageBox.information(self, "Saved", f"Camera parameters saved to:\n{cam_file_dir}")
        else:
            self.status_label.setText(f"Calibration Failed: {msg}")
            QMessageBox.critical(self, "Calibration Failed", msg)
        self._busy_end('wand_calibration')

    def _update_3d_viz(self):
        """Update the 3D visualization with calibrated camera positions and wand points."""
        import numpy as np
        
        if not hasattr(self, 'calib_3d_view'):
            return
            
        # Get camera params
        cameras = {}
        if hasattr(self.wand_calibrator, 'final_params'):
            cameras = self.wand_calibrator.final_params
        
        if not cameras: 
            return # Minimal silent return

        # Get points
        points_3d = None
        if hasattr(self.wand_calibrator, 'points_3d') and self.wand_calibrator.points_3d is not None:
            points_3d = self.wand_calibrator.points_3d
            pass
        else:
            pass
            
        # Plot
        self.calib_3d_view.plot_calibration(cameras, points_3d)
        
        # Switch to 3D Tab
        if hasattr(self, 'vis_tabs'):
            self.vis_tabs.setCurrentWidget(self.calib_3d_view)
    
    def _populate_error_table(self):
        """Populate error table with per-frame errors after calibration."""
        errors = self.wand_calibrator.calculate_per_frame_errors()
        if not errors:
            self.frozen_table.setRowCount(0)
            return
        
        # Identify filtered frames to persist checkbox state
        filtered_out_frames = set()
        if getattr(self.wand_calibrator, 'wand_points_filtered', None) is not None:
             all_frames = set(self.wand_calibrator.wand_points.keys())
             active_frames = set(self.wand_calibrator.wand_points_filtered.keys())
             filtered_out_frames = all_frames - active_frames
        
        # Determine camera list from all frames
        cam_ids_set = set()
        for frame_err in errors.values():
            cam_ids_set.update(frame_err['cam_errors'].keys())
        cam_ids = sorted(cam_ids_set)
        
        if not cam_ids:
            self.error_table.setRowCount(0)
            self.frozen_table.setRowCount(0)
            return
        
        # Setup table columns: Frame, Cam1, Cam2, ..., Len Err, Remove
        # IMPORTANT: Clear old contents first to avoid stale visual data
        self.error_table.clearContents()
        self.frozen_table.clearContents()
        self.error_table.setSortingEnabled(False)  # Always disable sorting while populating
        self.frozen_table.setSortingEnabled(False)
        
        # Right Table: Frame + cams + len_err
        col_count = 1 + len(cam_ids) + 1  
        self.error_table.setColumnCount(col_count)
        
        headers = ["Frame"] + [f"Cam {c+1}" for c in cam_ids] + ["Len Err (mm)"]
        self.error_table.setHorizontalHeaderLabels(headers)
        
        # Frozen Table: Just "Remove"
        self.frozen_table.setColumnCount(1)
        self.frozen_table.setHorizontalHeaderLabels(["Del"])
        
        # Populate rows - Only show frames with calculated errors (filtered frames)
        # Removed frames will not appear in the table at all
        frame_ids = sorted(errors.keys())
        self.error_table.setRowCount(len(frame_ids))
        self.frozen_table.setRowCount(len(frame_ids))
        self._error_frame_map = {}
        
        # Set Row Heights to match
        for row in range(len(frame_ids)):
            self.error_table.setRowHeight(row, 24)
            self.frozen_table.setRowHeight(row, 24)
        
        for row, fid in enumerate(frame_ids):
            self._error_frame_map[row] = fid
            err = errors[fid]
            
            # --- Frozen Table: Remove checkbox (Col 0) ---
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk = QCheckBox()
            chk.setProperty('frame_id', fid)
            chk_layout.addWidget(chk)
            self.frozen_table.setCellWidget(row, 0, chk_widget)
            
            # --- Right Table: Frame ID (Col 0) ---
            item = NumericTableWidgetItem(str(fid))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.error_table.setItem(row, 0, item)
            
            # Per-camera errors (Col 1..N)
            for i, cam_id in enumerate(cam_ids):
                val = err['cam_errors'].get(cam_id, float('nan'))
                item = NumericTableWidgetItem(f"{val:.2f}")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if val > self.filter_proj_spin.value():
                    item.setBackground(Qt.GlobalColor.darkRed)
                self.error_table.setItem(row, i + 1, item)
            
            # Length error (Col N+1)
            len_err = err['len_error']
            item = NumericTableWidgetItem(f"{len_err:.3f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if len_err > self.filter_len_spin.value():
                item.setBackground(Qt.GlobalColor.darkRed)
            self.error_table.setItem(row, len(cam_ids) + 1, item)
        
        # Sizing
        self.error_table.resizeColumnsToContents()
        self.error_table.horizontalHeader().setMinimumSectionSize(60)
        for c in range(col_count):
             if self.error_table.columnWidth(c) < 60:
                 self.error_table.setColumnWidth(c, 60)
                 
        # Enable sorting - frozen_table will sync via _sync_frozen_table_sort
        self.error_table.setSortingEnabled(True)
        # Frozen table has no direct sorting (synced from error_table)
        
        # IMPORTANT: Trigger sync now in case a sort indicator is already active
        # (repopulating data doesn't trigger sortIndicatorChanged signal)
        header = self.error_table.horizontalHeader()
        self._sync_frozen_table_sort(header.sortIndicatorSection(), header.sortIndicatorOrder())
    
    def _sync_frozen_table_sort(self, logical_index, order):
        """Handle sort request by manually sorting and repopulating BOTH tables."""
        from PySide6.QtWidgets import QCheckBox, QWidget, QHBoxLayout
        from PySide6.QtCore import Qt
        
        if not hasattr(self, 'wand_calibrator') or not hasattr(self.wand_calibrator, 'per_frame_errors'):
            return
        
        errors = getattr(self.wand_calibrator, 'per_frame_errors', {})
        if not errors:
            return
        
        # Collect current checkbox states by frame_id BEFORE repopulating
        checkbox_states = {}
        for row in range(self.frozen_table.rowCount()):
            widget = self.frozen_table.cellWidget(row, 0)
            if widget:
                chk = widget.findChild(QCheckBox)
                if chk:
                    fid = chk.property('frame_id')
                    if fid is not None:
                        checkbox_states[int(fid)] = chk.isChecked()
        
        # Build sortable list based on sort column
        # Columns: 0=Frame, 1..N=Cam errors, N+1=Len Err
        cam_ids_set = set()
        for frame_err in errors.values():
            cam_ids_set.update(frame_err['cam_errors'].keys())
        cam_ids = sorted(cam_ids_set)
        
        sortable_data = []
        for fid, err in errors.items():
            if logical_index == 0:
                # Sort by Frame ID
                sort_key = fid
            elif logical_index <= len(cam_ids):
                # Sort by camera error
                cam_id = cam_ids[logical_index - 1]
                sort_key = err['cam_errors'].get(cam_id, float('nan'))
            else:
                # Sort by length error
                sort_key = err['len_error']
            sortable_data.append((fid, sort_key, err))
        
        # Sort with NaN handling
        from PySide6.QtCore import Qt as QtCore
        reverse = (order == QtCore.SortOrder.DescendingOrder)
        
        def safe_sort_key(item):
            val = item[1]
            if isinstance(val, float) and (val != val):  # NaN check
                return (1, 0)  # Push NaN to end
            return (0, val)
        
        sortable_data.sort(key=safe_sort_key, reverse=reverse)
        
        # Clear and repopulate BOTH tables
        self.error_table.setSortingEnabled(False)  # Disable to prevent recursive trigger
        self.error_table.clearContents()
        self.frozen_table.clearContents()
        
        num_rows = len(sortable_data)
        self.error_table.setRowCount(num_rows)
        self.frozen_table.setRowCount(num_rows)
        
        for row, (fid, _, err) in enumerate(sortable_data):
            self.error_table.setRowHeight(row, 24)
            self.frozen_table.setRowHeight(row, 24)
            
            # --- Frozen Table: Checkbox ---
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk = QCheckBox()
            chk.setProperty('frame_id', fid)
            if checkbox_states.get(fid, False):
                chk.setChecked(True)
            chk_layout.addWidget(chk)
            self.frozen_table.setCellWidget(row, 0, chk_widget)
            
            # --- Error Table: Frame ID (Col 0) ---
            item = NumericTableWidgetItem(str(fid))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.error_table.setItem(row, 0, item)
            
            # --- Error Table: Camera errors (Col 1..N) ---
            for i, cam_id in enumerate(cam_ids):
                val = err['cam_errors'].get(cam_id, float('nan'))
                item = NumericTableWidgetItem(f"{val:.2f}")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if val > self.filter_proj_spin.value():
                    item.setBackground(Qt.GlobalColor.darkRed)
                self.error_table.setItem(row, i + 1, item)
            
            # --- Error Table: Length error (Col N+1) ---
            len_err = err['len_error']
            item = NumericTableWidgetItem(f"{len_err:.3f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if len_err > self.filter_len_spin.value():
                item.setBackground(Qt.GlobalColor.darkRed)
            self.error_table.setItem(row, len(cam_ids) + 1, item)
        
        # Re-enable sorting (but since we repopulated, the built-in sorting is now a no-op)
        self.error_table.setSortingEnabled(True)
    
    def _on_error_table_clicked(self, row, col):
        """Handle click on error table row - show frame images with overlay."""
        # Frame ID is now in column 0 of error_table
        item = self.error_table.item(row, 0)
        if item is None:
            return
        try:
            fid = int(item.text())
        except:
            return
        
        # Check if image paths are available
        has_images = bool(self.wand_images) and any(self.wand_images.values())
        if not has_images:
            # Show inline warning instead of popup
            if hasattr(self, 'error_warning_label'):
                self.error_warning_label.setText("⚠ Load image paths in Point Detection to preview.")
                self.error_warning_label.setVisible(True)
            return
        else:
            # Hide warning if images are available
            if hasattr(self, 'error_warning_label'):
                self.error_warning_label.setVisible(False)
        
        # Determine strict mapping for tab switching
        num_cams = len(self.wand_images)
        target_cam = None
        
        # Camera columns start at index 1 (0 is Frame ID)
        if 1 <= col < 1 + num_cams:
            target_cam = col - 1  # Column 1 = Camera 0
        
        # Display frame images with detection circles and reprojection overlay
        self._display_frame_with_overlay(fid, target_cam=target_cam)
    
    def _display_frame_with_overlay(self, frame_id, target_cam=None):
        """Display images for given frame with detection and reprojection overlay."""
        import cv2
        from PySide6.QtGui import QImage, QPixmap
        
        wand_data = self.wand_calibrator.wand_points_filtered or self.wand_calibrator.wand_points
        if frame_id not in wand_data:
            print(f"[Overlay] Frame {frame_id} not in wand_data (keys range: {min(wand_data.keys()) if wand_data else 0}-{max(wand_data.keys()) if wand_data else 0})")
            return
        
        obs = wand_data[frame_id]  # {cam_idx: [[x,y,r], [x,y,r]]}
        
        # Get 3D points for this frame
        frame_list = sorted(wand_data.keys())
        try:
            frame_i = frame_list.index(frame_id)
        except ValueError:
            print(f"[Overlay] Frame {frame_id} not found in sorted list")
            return
        
        pt3d_A = pt3d_B = None
        if hasattr(self.wand_calibrator, 'points_3d') and self.wand_calibrator.points_3d is not None:
            idx_A = frame_i * 2
            idx_B = frame_i * 2 + 1
            if idx_B < len(self.wand_calibrator.points_3d):
                pt3d_A = self.wand_calibrator.points_3d[idx_A]
                pt3d_B = self.wand_calibrator.points_3d[idx_B]
        
        for cam_idx, pts in obs.items():
            if cam_idx not in self.wand_images or not self.wand_images[cam_idx]:
                continue
            if frame_id >= len(self.wand_images[cam_idx]):
                continue
            
            img_path = self.wand_images[cam_idx][frame_id]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Draw detection circles (green)
            for pt in pts:
                x, y, r = int(pt[0]), int(pt[1]), int(pt[2]) if len(pt) > 2 else 20
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            
            # Draw reprojections (red cross) if calibration done
            # Check for refractive per_frame_errors with pre-computed proj_pts first
            if hasattr(self.wand_calibrator, 'per_frame_errors') and frame_id in self.wand_calibrator.per_frame_errors:
                err_data = self.wand_calibrator.per_frame_errors[frame_id]
                proj_pts = err_data.get('proj_pts', {})
                
                if cam_idx in proj_pts:
                    proj_A, proj_B = proj_pts[cam_idx]
                    for proj in [proj_A, proj_B]:
                        if proj is not None:
                            px, py = int(proj[0]), int(proj[1])
                            cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            # Fallback to pinhole cv2.projectPoints (for pinhole calibration mode)
            elif pt3d_A is not None and cam_idx in self.wand_calibrator.final_params:
                p = self.wand_calibrator.final_params[cam_idx]
                R, T, K, dist = p['R'], p['T'], p['K'], p['dist']
                rvec, _ = cv2.Rodrigues(R)
                
                for pt3d in [pt3d_A, pt3d_B]:
                    proj, _ = cv2.projectPoints(pt3d.reshape(1,3), rvec, T, K, dist)
                    px, py = int(proj[0,0,0]), int(proj[0,0,1])
                    cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            
            # Convert to QPixmap and display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Scale to fit label
            if cam_idx in self.cam_vis_labels:
                lbl = self.cam_vis_labels[cam_idx]
                scaled = pixmap.scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio)
                lbl.setPixmap(scaled)
        
        # Switch to target camera tab (if specified) or first camera
        if hasattr(self, 'vis_tabs') and self.cam_vis_labels:
            switch_cam = target_cam if target_cam is not None and target_cam in self.cam_vis_labels else 0
            # Tab index matches camera index (0 = Cam 1, 1 = Cam 2, etc.)
            if switch_cam in self.cam_vis_labels:
                self.vis_tabs.setCurrentIndex(switch_cam)
    
    def _auto_update_filter_marks(self, *args):
        """Auto-update Remove checkboxes when filter criteria change."""
        if not hasattr(self.wand_calibrator, 'per_frame_errors') or not self.wand_calibrator.per_frame_errors:
            return  # No error data yet, skip silently
        
        proj_thresh = self.filter_proj_spin.value() if self.filter_proj_check.isChecked() else float('inf')
        len_thresh = self.filter_len_spin.value() if self.filter_len_check.isChecked() else float('inf')
        
        marked_count = 0
        # Update checkboxes in table
        for row in range(self.error_table.rowCount()):
            # Frame ID is in column 0 (not column 1!)
            item = self.error_table.item(row, 0)
            if item is None:
                continue
            try:
                fid = int(item.text())
            except:
                continue
            
            err = self.wand_calibrator.per_frame_errors.get(fid, {})
            max_proj = max(err.get('cam_errors', {}).values(), default=0)
            len_err = err.get('len_error', 0)
            
            should_remove = bool((max_proj > proj_thresh) or (len_err > len_thresh))
            if should_remove:
                marked_count += 1
            
            # Checkbox is in frozen_table column 0 (not error_table!)
            widget = self.frozen_table.cellWidget(row, 0)
            if widget:
                chk = widget.findChild(QCheckBox)
                if chk:
                    chk.setChecked(should_remove)
        
        total = self.error_table.rowCount()
        self.status_label.setText(f"Marked {marked_count}/{total} frames for removal.")
    
    def _save_filtered_data(self):
        """Save filtered data by removing marked frames."""
        frames_to_remove = set()
        for row in range(self.error_table.rowCount()):
            # Checkbox is in column 0
            widget = self.error_table.cellWidget(row, 0)
            if widget:
                chk = widget.findChild(QCheckBox)
                if chk and chk.isChecked():
                    # Get fid from table cell (column 1) - works after sorting
                    item = self.error_table.item(row, 1)
                    if item:
                        try:
                            fid = int(item.text())
                            frames_to_remove.add(fid)
                        except:
                            pass
        
        if not frames_to_remove:
            QMessageBox.information(self, "No Changes", "No frames marked for removal.")
            return
        
        remaining = self.wand_calibrator.apply_filter(frames_to_remove)
        QMessageBox.information(self, "Filtered", 
            f"Removed {len(frames_to_remove)} frames.\n{remaining} frames remaining.\n\nRun Calibration again to use filtered data.")
        self.status_label.setText(f"{remaining} frames ready. Re-run calibration.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'plate_vis_label') and self.tabs.currentIndex() == 0:
            if self.plate_img_list.currentRow() >= 0:
                self._display_plate_image(self.plate_img_list.currentRow())

    def _start_template_selection(self):
        """Enable template selection mode on the active camera view."""
        cam_idx = self.plate_cam_combo.currentIndex()
        lbl = self.plate_cam_labels.get(cam_idx, None)
        if lbl:
            lbl.set_mode(ZoomableImageLabel.MODE_TEMPLATE)
    
    def _start_roi_selection(self):
        """Enable ROI selection mode."""
        self.search_roi_points = []
        self.lbl_roi_status.setText("Region: Click 4 points")
        
        cam_idx = self.plate_cam_combo.currentIndex()
        lbl = self.plate_cam_labels.get(cam_idx, None)
        if lbl:
            lbl.clear_overlays() # Clear visual ROI
            lbl.set_mode(ZoomableImageLabel.MODE_ROI)

    def _on_roi_points_changed(self, points, idx):
        """Update ROI status when points added."""
        # IMPORTANT: Make a copy of points, not a reference!
        # Otherwise if ZoomableImageLabel clears its roi_points, our copy will also be emptied.
        self.search_roi_points = [QPoint(p.x(), p.y()) for p in points]
        count = len(points)
        if count == 4:
            self.lbl_roi_status.setText("Region: Defined (4 pts)")
            # Auto switch back to nav mode?
            lbl = self.plate_cam_labels.get(idx, None)
            if lbl:
                lbl.set_mode(ZoomableImageLabel.MODE_NAV)
        else:
            self.lbl_roi_status.setText(f"Region: {count}/4 points")

    def _run_detection_and_show(self):
        """Run detection and display results."""
        self._run_detection(smart_fill=False)
        
    def _test_grid_detection(self):
        self._run_detection(smart_fill=False)
        
    def _test_smart_fill(self):
        self._run_detection(smart_fill=True)

    def _start_remove_mode(self):
        """Enable remove mode - drag to remove points."""
        if not self.detected_keypoints:
            QMessageBox.warning(self, "No Points", "Please run detection first.")
            return
        cam_idx = self.plate_cam_combo.currentIndex()
        lbl = self.plate_cam_labels.get(cam_idx, None)
        if lbl:
            lbl.set_mode(ZoomableImageLabel.MODE_REMOVE)
            
    def _start_add_mode(self):
        """Enable add mode - drag to add points via blob detection."""
        cam_idx = self.plate_cam_combo.currentIndex()
        lbl = self.plate_cam_labels.get(cam_idx, None)
        if lbl:
            lbl.set_mode(ZoomableImageLabel.MODE_ADD)

    def _on_remove_region(self, rect, cam_idx):
        """Remove keypoints within the selected region."""
        if not self.detected_keypoints:
            return
            
        # Filter out points inside the rect
        new_keypoints = []
        for kp in self.detected_keypoints:
            x, y = kp.pt
            if not rect.contains(QPoint(int(x), int(y))):
                new_keypoints.append(kp)
                
        removed_count = len(self.detected_keypoints) - len(new_keypoints)
        self.detected_keypoints = new_keypoints
        self.lbl_points_count.setText(f"Points: {len(self.detected_keypoints)}")
        
        # Re-visualize (stay in Remove mode for continuous removal)
        self._visualize_keypoints()

    def _on_add_region(self, rect, cam_idx):
        """Add keypoints via blob detection within the selected region."""
        import cv2
        
        row = self.plate_img_list.currentRow()
        if row < 0 or row >= len(self.plate_images):
            return
            
        img_path = self.plate_images[row]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
            
        # Extract ROI
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        # Clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        x2 = min(img.shape[1], x + w)
        y2 = min(img.shape[0], y + h)
        
        if x2 <= x or y2 <= y:
            return
            
        roi = img[y:y2, x:x2]
        
        # Simple blob detection using adaptive threshold + contours
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # Determine polarity from Point Color setting
        is_dark = self.plate_color_combo.currentIndex() == 1
        
        if is_dark:
            # Dark on bright - invert
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        added_count = 0
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"] + x  # Convert back to image coords
                cy = M["m01"] / M["m00"] + y
                
                # Check if point already exists nearby
                is_duplicate = False
                for kp in self.detected_keypoints:
                    dist = ((kp.pt[0] - cx)**2 + (kp.pt[1] - cy)**2)**0.5
                    if dist < 5:  # 5 pixel threshold
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    kp = cv2.KeyPoint(x=float(cx), y=float(cy), size=10.0)
                    self.detected_keypoints.append(kp)
                    added_count += 1
                    
        self.lbl_points_count.setText(f"Points: {len(self.detected_keypoints)}")
        
        # Re-visualize (stay in Add mode for continuous addition)
        self._visualize_keypoints()

    def _visualize_keypoints(self):
        """Redraw the image with current keypoints."""
        import cv2
        from PySide6.QtGui import QImage, QPixmap
        
        row = self.plate_img_list.currentRow()
        if row < 0 or row >= len(self.plate_images):
            return
            
        img_path = self.plate_images[row]
        img = cv2.imread(str(img_path))
        if img is None:
            return
            
        # Draw keypoints
        marker_size = 5
        for kp in self.detected_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.line(img, (x - marker_size, y), (x + marker_size, y), (0, 255, 0), 2)
            cv2.line(img, (x, y - marker_size), (x, y + marker_size), (0, 255, 0), 2)
            
        # Convert to QPixmap
        h, w, ch = img.shape
        bytes_per_line = ch * w
        vis_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_img = QImage(vis_img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        
        cam_idx = self.plate_cam_combo.currentIndex()
        target_label = self.plate_cam_labels.get(cam_idx, None)
        if target_label:
            target_label.setPixmap(pix)

    # ========== INDEXING METHODS ==========
    
    def _start_origin_selection(self):
        """Enable origin selection mode."""
        cam_idx = self.plate_cam_combo.currentIndex()
        lbl = self.plate_cam_labels.get(cam_idx, None)
        if lbl:
            lbl.set_mode(ZoomableImageLabel.MODE_ORIGIN)
            
    def _start_axes_selection(self):
        """Enable axis direction selection mode."""
        if self.origin_point is None:
            QMessageBox.warning(self, "No Origin", "Please set the origin first.")
            return
            
        # Determine which axes are free (not fixed)
        free_axes = self._get_free_axis_names()
        if len(free_axes) < 2:
            QMessageBox.warning(self, "Error", "Need exactly 2 free axes to set directions.")
            return
            
        self._axes_selection_step = 1
        self._free_axes = free_axes
        
        cam_idx = self.plate_cam_combo.currentIndex()
        # Set MODE_AXES on current label
        lbl = self.plate_cam_labels.get(cam_idx, None)
        if lbl:
            lbl.set_mode(ZoomableImageLabel.MODE_AXES)
            lbl._current_axis_index = 0
            # Show mouse-following hint text
            first_axis = self._free_axes[0] if self._free_axes else "axis"
            lbl._hint_text = f"Click +{first_axis}"
            lbl.update()
        
    def _on_origin_selected(self, pt, cam_idx):
        """Handle origin point selection."""
        # Find nearest keypoint
        if self.detected_keypoints:
            min_dist = float('inf')
            nearest_kp = None
            for kp in self.detected_keypoints:
                dist = ((kp.pt[0] - pt.x())**2 + (kp.pt[1] - pt.y())**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_kp = kp
            
            if nearest_kp and min_dist < 30:  # 30 pixel threshold
                self.origin_point = (nearest_kp.pt[0], nearest_kp.pt[1])
            else:
                # Use exact click position
                self.origin_point = (pt.x(), pt.y())
        else:
            self.origin_point = (pt.x(), pt.y())
        
        # Visualize origin
        self._visualize_keypoints_with_origin()
        
        # Switch back to nav mode
        lbl = self.plate_cam_labels.get(cam_idx, None)
        if lbl:
            lbl.set_mode(ZoomableImageLabel.MODE_NAV)
            
    def _on_axis_point_selected(self, pt, axis_idx, cam_idx):
        """Handle axis direction point selection."""
        print(f"Indexing Debug: Axis point selected at ({pt.x()}, {pt.y()}) for Cam {cam_idx}, Step {self._axes_selection_step}")
        # Find nearest keypoint
        if self.detected_keypoints:
            min_dist = float('inf')
            nearest_kp = None
            for kp in self.detected_keypoints:
                dist = ((kp.pt[0] - pt.x())**2 + (kp.pt[1] - pt.y())**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_kp = kp
            
            if nearest_kp and min_dist < 30:
                selected_pt = (nearest_kp.pt[0], nearest_kp.pt[1])
            else:
                selected_pt = (pt.x(), pt.y())
        else:
            selected_pt = (pt.x(), pt.y())
        
        if self._axes_selection_step == 1:
            self.axis1_point = selected_pt
            self._axes_selection_step = 2
            
            # Prompt for second axis
            lbl = self.plate_cam_labels.get(cam_idx, None)
            if lbl:
                lbl._current_axis_index = 1
                # Update hint text to show second axis
                if len(self._free_axes) >= 2:
                    second_axis = self._free_axes[1]
                    lbl._hint_text = f"Click +{second_axis}"
                    lbl.update()
            
        elif self._axes_selection_step == 2:
            self.axis2_point = selected_pt
            self._axes_selection_step = 0
            
            # Visualize axes
            self._visualize_keypoints_with_origin()
            
            # Switch back to nav mode
            lbl = self.plate_cam_labels.get(cam_idx, None)
            if lbl:
                lbl.set_mode(ZoomableImageLabel.MODE_NAV)
                lbl._hint_text = ""
                
    def _run_indexing(self):
        """Calculate integer indices [i, j, k] for all detected points."""
        if not self.detected_keypoints:
            QMessageBox.warning(self, "No Points", "No keypoints detected. Run detection first.")
            return
        if self.origin_point is None or self.axis1_point is None or self.axis2_point is None:
            QMessageBox.warning(self, "Missing Indexing Info", "Please set origin and both axis directions first.")
            return

        # 1. Setup Axes
        origin = np.array(self.origin_point)
        p1 = np.array(self.axis1_point)
        p2 = np.array(self.axis2_point)
        
        v1 = p1 - origin
        v2 = p2 - origin
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            QMessageBox.warning(self, "Error", "Axis points are too close to origin.")
            return
            
        u1 = v1 / norm1
        u2 = v2 / norm2
        
        # 2. Project all points into the custom basis defined by u1, u2
        # We solve the system: [u1, u2] * [d1; d2] = (pt - origin)
        pts = np.array([kp.pt for kp in self.detected_keypoints])
        diffs = pts - origin
        
        # Matrix A = [u1.x u2.x; u1.y u2.y]
        A = np.stack((u1, u2), axis=1)
        try:
            # Solving for all points at once: D = A * X => X = inv(A) * D
            # (X will be [d1, d2] for each point)
            coords = np.linalg.solve(A, diffs.T).T
            proj1 = coords[:, 0]
            proj2 = coords[:, 1]
        except np.linalg.LinAlgError:
            # Fallback to dot if axes are nearly parallel
            proj1 = np.dot(diffs, u1)
            proj2 = np.dot(diffs, u2)
        
    def _run_indexing(self):
        """Calculate integer indices [i, j, k] using a topological 'Greedy Walk' strategy."""
        if not self.detected_keypoints:
            QMessageBox.warning(self, "No Points", "No keypoints detected. Run detection first.")
            return
        if self.origin_point is None or self.axis1_point is None or self.axis2_point is None:
            QMessageBox.warning(self, "Missing Indexing Info", "Please set origin and both axis directions first.")
            return

        # 0. Setup and Delta Estimation
        all_pts = np.array([kp.pt for kp in self.detected_keypoints])
        n_pts = len(all_pts)
        
        # Estimate delta (median distance to nearest 5 neighbors)
        all_deltas = []
        for i in range(min(n_pts, 100)): # Check first 100 points for speed
            dists = np.linalg.norm(all_pts - all_pts[i], axis=1)
            neighbor_dists = np.sort(dists)[1:6]
            all_deltas.extend(neighbor_dists)
        delta = np.median(all_deltas) if all_deltas else 50.0
        print(f"Indexing Debug: Estimated delta = {delta:.2f}px")

        # 1. Directions
        origin_pt = np.array(self.origin_point)
        p1 = np.array(self.axis1_point)
        p2 = np.array(self.axis2_point)
        
        u1 = (p1 - origin_pt) / np.linalg.norm(p1 - origin_pt)
        u2 = (p2 - origin_pt) / np.linalg.norm(p2 - origin_pt)
        
        # 2. Helper to find nearest neighbor in a specific direction
        def find_next_point(current_pt, direction, pool_indices, d_max=1.5*delta, lat_tol=0.3*delta):
            best_idx = -1
            min_dist = d_max
            
            for idx in pool_indices:
                vec = all_pts[idx] - current_pt
                dist_along = np.dot(vec, direction)
                if dist_along < 5: continue # Must move forward significantly
                
                dist_lat = np.linalg.norm(vec - dist_along * direction)
                if dist_lat < lat_tol and dist_along < d_max:
                    if dist_along < min_dist:
                        min_dist = dist_along
                        best_idx = idx
            return best_idx

        # 3. Backbone Walk (Primary Axis)
        # Determine which direction has more points to pick as primary
        def count_points_on_axis(u):
            count = 0
            curr = origin_pt
            temp_pool = list(range(n_pts))
            while True:
                nxt = find_next_point(curr, u, temp_pool)
                if nxt == -1: break
                count += 1
                curr = all_pts[nxt]
                temp_pool.remove(nxt)
            return count

        c1 = count_points_on_axis(u1) + count_points_on_axis(-u1)
        c2 = count_points_on_axis(u2) + count_points_on_axis(-u2)
        
        if c1 >= c2:
            primary_u, secondary_u = u1, u2
            primary_is_v1 = True
        else:
            primary_u, secondary_u = u2, u1
            primary_is_v1 = False

        print(f"Indexing Debug: Primary Axis: {'V1' if primary_is_v1 else 'V2'} (Count: {max(c1, c2)})")

        # 4. Perform the Walks
        # Store results as a map: {point_index: (walk1_idx, walk2_idx)}
        visit_map = {}
        
        # Origin is (0, 0)
        origin_idx = -1
        for i, pt in enumerate(all_pts):
            if np.linalg.norm(pt - origin_pt) < 5:
                origin_idx = i
                break
        
        if origin_idx == -1:
            # Add origin if not found in detected keypoints? No, use the closest one.
            origin_idx = np.argmin(np.linalg.norm(all_pts - origin_pt, axis=1))
            
        visit_map[origin_idx] = (0, 0)
        backbone = [(origin_idx, 0)] # (idx, w1_pos)

        # Walk primary axis (pos & neg)
        for direction, step in [(primary_u, 1), (-primary_u, -1)]:
            curr_idx = origin_idx
            curr_w = 0
            while True:
                pool = [i for i in range(n_pts) if i not in visit_map]
                nxt = find_next_point(all_pts[curr_idx], direction, pool)
                if nxt == -1: break
                curr_w += step
                visit_map[nxt] = (curr_w, 0)
                backbone.append((nxt, curr_w))
                curr_idx = nxt

        # Walk secondary axis from each backbone point
        for b_idx, w1 in backbone:
            for direction, step in [(secondary_u, 1), (-secondary_u, -1)]:
                curr_idx = b_idx
                curr_w2 = 0
                while True:
                    pool = [i for i in range(n_pts) if i not in visit_map]
                    nxt = find_next_point(all_pts[curr_idx], direction, pool)
                    if nxt == -1: break
                    curr_w2 += step
                    visit_map[nxt] = (w1, curr_w2)
                    curr_idx = nxt

        # 5. Final Mapping to (i, j, k)
        fixed_val = self.spin_plane_num.value()
        self.point_indices = [None] * n_pts
        
        # Figure out which walk corresponds to which logical index i, j, k
        # UI selection: Fixed X -> Y, Z free. Fixed Y -> X, Z free. Fixed Z -> X, Y free.
        # secondary walk follows the order of user selection or standard (X then Y).
        
        for idx, (w1, w2) in visit_map.items():
            # n1, n2 are the relative integer steps from primary/secondary walk
            n1, n2 = w1, w2
            if not primary_is_v1: n1, n2 = n2, n1 # Swap if V2 was primary
            
            if self.chk_fix_x.isChecked():
                self.point_indices[idx] = (fixed_val, n1, n2)
            elif self.chk_fix_y.isChecked():
                self.point_indices[idx] = (n1, fixed_val, n2)
            else: # Fix Z
                self.point_indices[idx] = (n1, n2, fixed_val)

        # Remove unvisited points (those with None indices) from both lists
        visited_count = sum(1 for idx in self.point_indices if idx is not None)
        if visited_count < n_pts:
            # Filter out unvisited points
            new_keypoints = []
            new_indices = []
            for kp, idx in zip(self.detected_keypoints, self.point_indices):
                if idx is not None:
                    new_keypoints.append(kp)
                    new_indices.append(idx)
            self.detected_keypoints = new_keypoints
            self.point_indices = new_indices
            removed_count = n_pts - visited_count
            print(f"Removed {removed_count} unvisited points")

        # 6. Visualize
        self._visualize_keypoints_with_origin()
        self.lbl_points_count.setText(f"Points: {len(self.detected_keypoints)}")
        QMessageBox.information(self, "Indexing Complete", f"Assigned indices to {len(self.detected_keypoints)} points ({n_pts - len(self.detected_keypoints)} removed).")

    def _on_axis_check_toggled(self, axis, checked):
        """Handle axis checkbox toggle - enforce mutual exclusion and sync plane number."""
        if checked:
            # Block signals to prevent recursion
            self.chk_fix_x.blockSignals(True)
            self.chk_fix_y.blockSignals(True)
            self.chk_fix_z.blockSignals(True)
            
            # Uncheck others (mutual exclusion)
            if axis == 'x':
                self.chk_fix_y.setChecked(False)
                self.chk_fix_z.setChecked(False)
            elif axis == 'y':
                self.chk_fix_x.setChecked(False)
                self.chk_fix_z.setChecked(False)
            elif axis == 'z':
                self.chk_fix_x.setChecked(False)
                self.chk_fix_y.setChecked(False)
            
            self.chk_fix_x.blockSignals(False)
            self.chk_fix_y.blockSignals(False)
            self.chk_fix_z.blockSignals(False)
            
            # Set plane number to current frame index + offset
            frame_idx = self.plate_img_list.currentRow()
            if frame_idx < 0:
                frame_idx = 0
            self.spin_plane_num.setValue(frame_idx + self.plane_num_offset)

    def _on_plane_num_manually_changed(self, new_val):
        """Handle manual change of plane number - updates the global offset."""
        if not self.spin_plane_num.signalsBlocked():
            curr_row = self.plate_img_list.currentRow()
            if curr_row >= 0:
                self.plane_num_offset = new_val - curr_row
                print(f"Indexing Debug: New Plane Offset = {self.plane_num_offset}")

    def _start_check_position_mode(self):
        """Enable click-to-check 3D position mode."""
        idx = self.plate_cam_combo.currentIndex()
        if idx in self.plate_cam_labels:
            self.plate_cam_labels[idx].set_mode(ZoomableImageLabel.MODE_CHECK_POS)

    def _on_check_pos_clicked(self, pt):
        """Handle point click to show physical position."""
        if not self.detected_keypoints or not self.point_indices:
            return
            
        # Find nearest point
        best_dist = float('inf')
        best_idx = -1
        
        for i, kp in enumerate(self.detected_keypoints):
            dx = kp.pt[0] - pt.x()
            dy = kp.pt[1] - pt.y()
            dist = dx*dx + dy*dy
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                
        if best_idx >= 0 and self.point_indices[best_idx] != (999,999,999):
            idx = self.point_indices[best_idx]
            dx, dy, dz = self.spin_dx.value(), self.spin_dy.value(), self.spin_dz.value()
            phys_x = idx[0] * dx
            phys_y = idx[1] * dy
            phys_z = idx[2] * dz
            
            # Store for visualization
            self.check_pos_point = self.detected_keypoints[best_idx].pt
            self.check_pos_label = f"X:{phys_x:.2f} Y:{phys_y:.2f} Z:{phys_z:.2f} mm"
            # Clear previous markings by refreshing
            self._visualize_keypoints_with_origin()
            
    def _add_to_calibration_data(self):
        """Save currently detected keypoints and their 3D world coordinates."""
        row = self.plate_img_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No Image", "Select an image first.")
            return
            
        if not self.point_indices or len(self.point_indices) != len(self.detected_keypoints):
            QMessageBox.warning(self, "No Indices", "Please run indexing before adding points.")
            return

        cam_idx = self.plate_cam_combo.currentIndex()
        img_path = self.plate_images[row]
        
        # Calculate world coordinates
        dx, dy, dz = self.spin_dx.value(), self.spin_dy.value(), self.spin_dz.value()
        world_coords = []
        for idx in self.point_indices:
            if idx == (999,999,999):
                world_coords.append(None)
            else:
                world_coords.append((idx[0]*dx, idx[1]*dy, idx[2]*dz))

        # Consistent key: (camera_index, image_path)
        data_key = (cam_idx, img_path)
        self.saved_calibration_data[data_key] = {
            'keypoints': list(self.detected_keypoints), # Copy list
            'indices': list(self.point_indices),
            'world_coords': world_coords,
            'plane': self.spin_plane_num.value()
        }
        
        self.lbl_points_count.setText(f"Points: {len(self.detected_keypoints)} (Added to Cam {cam_idx})")
        QMessageBox.information(self, "Added", f"Added {len(self.detected_keypoints)} points with 3D positions for Camera {cam_idx}, Image {self.plate_img_list.currentItem().text()}")

    def _export_calibration_to_csv(self):
        """Export all saved calibration data across all cameras and images to a CSV file."""
        import csv
        if not self.saved_calibration_data:
            QMessageBox.warning(self, "No Data", "There is no calibration data to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export Calibration Data", "", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(["CameraID", "ImagePath", "Plane", "PixelX", "PixelY", "WorldX", "WorldY", "WorldZ"])
                
                # Data
                for (cam_idx, img_path), data in self.saved_calibration_data.items():
                    kpts = data['keypoints']
                    world = data['world_coords']
                    plane = data['plane']
                    
                    for kp, w_pos in zip(kpts, world):
                        if w_pos is None: continue # Skip un-indexed points
                        
                        writer.writerow([
                            cam_idx,
                            img_path,
                            plane,
                            f"{kp.pt[0]:.3f}",
                            f"{kp.pt[1]:.3f}",
                            f"{w_pos[0]:.3f}",
                            f"{w_pos[1]:.3f}",
                            f"{w_pos[2]:.3f}"
                        ])
            
            QMessageBox.information(self, "Success", f"Successfully exported data to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {str(e)}")

    def _read_calibration_from_csv(self):
        """Read calibration data from a CSV file and populate saved_calibration_data."""
        import csv
        import cv2
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Calibration CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            new_data = {} # (cam_idx, path) -> {kpts, world, plane}
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse header: CameraID, ImagePath, Plane, PixelX, PixelY, WorldX, WorldY, WorldZ
                    c_id = int(row['CameraID'])
                    i_path = row['ImagePath']
                    plane = int(row['Plane'])
                    px_x = float(row['PixelX'])
                    px_y = float(row['PixelY'])
                    wx = float(row['WorldX'])
                    wy = float(row['WorldY'])
                    wz = float(row['WorldZ'])
                    
                    key = (c_id, i_path)
                    if key not in new_data:
                        new_data[key] = {
                            'keypoints': [],
                            'world_coords': [],
                            'indices': [], # We don't strictly need indices for calibration if we have world_coords
                            'plane': plane
                        }
                    
                    kp = cv2.KeyPoint(px_x, px_y, 1.0) # Dummy size 1.0
                    new_data[key]['keypoints'].append(kp)
                    new_data[key]['world_coords'].append([wx, wy, wz])
                    new_data[key]['indices'].append([0, 0, 0]) # Placeholder indices
            
            if not new_data:
                QMessageBox.warning(self, "No Data", "No valid data found in CSV.")
                return

            # Merge or Overwrite? Let's Merge
            self.saved_calibration_data.update(new_data)
            
            # Refresh current view
            row = self.plate_img_list.currentRow()
            if row >= 0:
                self._display_plate_image(row)
                
            QMessageBox.information(self, "Success", f"Successfully loaded data for {len(new_data)} image-camera pairs.")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to parse CSV: {str(e)}")
    def _get_fixed_axis_name(self):
        """Get the name of the fixed axis."""
        if self.chk_fix_x.isChecked():
            return 'X'
        elif self.chk_fix_y.isChecked():
            return 'Y'
        elif self.chk_fix_z.isChecked():
            return 'Z'
        return None
        
    def _get_free_axis_names(self):
        """Get the names of the free (unfixed) axes."""
        axes = []
        if not self.chk_fix_x.isChecked():
            axes.append('X')
        if not self.chk_fix_y.isChecked():
            axes.append('Y')
        if not self.chk_fix_z.isChecked():
            axes.append('Z')
        return axes
        
    def _visualize_keypoints_with_origin(self):
        """Visualize keypoints with origin and axis markers."""
        import cv2
        from PySide6.QtGui import QImage, QPixmap
        
        row = self.plate_img_list.currentRow()
        if row < 0 or row >= len(self.plate_images):
            return
            
        img_path = self.plate_images[row]
        img = cv2.imread(str(img_path))
        if img is None:
            return
            
        # Draw keypoints
        marker_size = 5
        for kp in self.detected_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.line(img, (x - marker_size, y), (x + marker_size, y), (0, 255, 0), 2)
            cv2.line(img, (x, y - marker_size), (x, y + marker_size), (0, 255, 0), 2)
        
        # Draw origin
        if self.origin_point:
            ox, oy = int(self.origin_point[0]), int(self.origin_point[1])
            cv2.circle(img, (ox, oy), 10, (255, 0, 255), 3)  # Magenta circle
            cv2.drawMarker(img, (ox, oy), (255, 0, 255), cv2.MARKER_CROSS, 20, 3)
            
        # Draw axes
        if self.origin_point and self.axis1_point:
            ox, oy = int(self.origin_point[0]), int(self.origin_point[1])
            ax1, ay1 = int(self.axis1_point[0]), int(self.axis1_point[1])
            cv2.arrowedLine(img, (ox, oy), (ax1, ay1), (255, 0, 0), 3, tipLength=0.2)  # Blue = axis1
            
            if hasattr(self, '_free_axes') and len(self._free_axes) > 0:
                cv2.putText(img, f"+{self._free_axes[0]}", (ax1 + 5, ay1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
        if self.origin_point and self.axis2_point:
            ox, oy = int(self.origin_point[0]), int(self.origin_point[1])
            ax2, ay2 = int(self.axis2_point[0]), int(self.axis2_point[1])
            cv2.arrowedLine(img, (ox, oy), (ax2, ay2), (0, 165, 255), 3, tipLength=0.2)  # Orange = axis2
            
            if hasattr(self, '_free_axes') and len(self._free_axes) > 1:
                cv2.putText(img, f"+{self._free_axes[1]}", (ax2 + 5, ay2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
        # Draw Indices
        if hasattr(self, 'point_indices') and self.point_indices and len(self.point_indices) == len(self.detected_keypoints):
            # Check for sparse visualization step
            viz_step = 1
            if hasattr(self, 'spin_index_step'):
                viz_step = self.spin_index_step.value()
            
            for kp, idx in zip(self.detected_keypoints, self.point_indices):
                if idx is None: continue
                # Skip filtering for origin [0,0,0]
                is_origin = (idx[0] == 0 and idx[1] == 0 and idx[2] == 0)
                
                # Filter based on step
                show = is_origin
                if not show:
                    # Only show if free indices are multiples of viz_step
                    if self.chk_fix_x.isChecked():
                        if idx[1] % viz_step == 0 and idx[2] % viz_step == 0: show = True
                    elif self.chk_fix_y.isChecked():
                        if idx[0] % viz_step == 0 and idx[2] % viz_step == 0: show = True
                    else: # Z fixed
                        if idx[0] % viz_step == 0 and idx[1] % viz_step == 0: show = True
                
                if not show: continue

                x, y = int(kp.pt[0]), int(kp.pt[1])
                # Draw a highlighted marker for the indexed point
                marker_size = 6
                highlight_color = (255, 0, 0) # Blue in BGR
                cv2.line(img, (x - marker_size, y), (x + marker_size, y), highlight_color, 2)
                cv2.line(img, (x, y - marker_size), (x, y + marker_size), highlight_color, 2)

                # Label: [i,j,k]
                label = f"[{idx[0]},{idx[1]},{idx[2]}]"
                cv2.putText(img, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, highlight_color, 1)

        # Draw check position highlight
        if hasattr(self, 'check_pos_point') and self.check_pos_point is not None:
            cx, cy = int(self.check_pos_point[0]), int(self.check_pos_point[1])
            cv2.circle(img, (cx, cy), 8, (0, 255, 255), 2)  # Yellow circle
            cv2.putText(img, self.check_pos_label, (cx + 12, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
        # Convert to QPixmap
        h, w, ch = img.shape
        bytes_per_line = ch * w
        vis_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_img = QImage(vis_img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        
        cam_idx = self.plate_cam_combo.currentIndex()
        target_label = self.plate_cam_labels.get(cam_idx, None)
        if target_label:
            target_label.setPixmap(pix)

    def _run_detection(self, smart_fill=False):
        """Test the grid detection algorithm (Template Matching Only)."""
        row = self.plate_img_list.currentRow()
        if row < 0 or row >= len(self.plate_images):
            QMessageBox.warning(self, "No Image", "Please select an image from the list first.")
            return

        img_path = self.plate_images[row]
        
        try:
            if self.current_template is None:
                QMessageBox.warning(self, "No Template", "Please select a template ROI on the image first.")
                return
            
            from .plate_calibration.grid_detector import GridDetector

            threshold = self.slider_match_thresh.value()
            
            # Prepare ROI Mask
            roi_mask = None
            if len(self.search_roi_points) == 4:
                # Create mask same size as image
                import cv2
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    roi_mask = np.zeros((h, w), dtype=np.uint8)
                    pts = np.array([[p.x(), p.y()] for p in self.search_roi_points], dtype=np.int32)
                    cv2.fillPoly(roi_mask, [pts], 255)
            
            keypoints, vis_img = GridDetector.detect_template(
                img_path,
                self.current_template,
                threshold=threshold,
                smart_fill=smart_fill,
                center_offset=self.template_offset,
                search_mask=roi_mask
            )
            
            # Convert to QPixmap for display
            import cv2
            h, w, ch = vis_img.shape
            bytes_per_line = ch * w
            # OpenCV is BGR, Qt needs RGB
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            
            from PySide6.QtGui import QImage, QPixmap
            q_img = QImage(vis_img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(q_img)
            
            # Display on the CORRECT label (active camera)
            cam_idx = self.plate_cam_combo.currentIndex()
            target_label = self.plate_cam_labels.get(cam_idx, None)
            
            if target_label:
                # Set Pixmap on Zoomable Label
                target_label.setPixmap(pix)
            
            # Store keypoints for later modification (Remove/Add)
            self.detected_keypoints = keypoints
            self.lbl_points_count.setText(f"Points: {len(keypoints)}")
            
            # Show stats
            msg = f"Found {len(keypoints)} points."
            if roi_mask is not None:
                msg += " (Restricted to ROI)"
            QMessageBox.information(self, "Detection Result", msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Detection Error", f"An error occurred during detection:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def create_wand_tab_v2_OLD_DUPLICATE_IGNORE(self):
        """Create the Wand Calibration tab (Multi-Camera) - Tabbed Interface."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 1. Visualization (LEFT)
        vis_frame = QFrame()
        vis_frame.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.setContentsMargins(0,0,0,0)

        self.vis_tabs = QTabWidget()
        self.vis_tabs.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QTabBar::tab { background: #222; color: #888; padding: 5px; }
            QTabBar::tab:selected { background: #444; color: #fff; }
        """)
        vis_layout.addWidget(self.vis_tabs)
        
        self.cam_vis_labels = {} # {cam_idx: QLabel}
        
        # 2. Controls Panel (RIGHT)
        right_panel = QWidget()
        right_panel.setFixedWidth(370)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        controls_tabs = QTabWidget()
        controls_tabs.setStyleSheet("""
             QTabWidget::pane { border: 1px solid #444; background: #222; }
             QTabBar::tab { background: #333; color: #aaa; padding: 8px; min-width: 100px; }
             QTabBar::tab:selected { background: #444; color: #fff; border-bottom: 2px solid #00d4ff; }
        """)

        # --- Tab 1: Detection ---
        det_tab = QWidget()
        det_layout = QVBoxLayout(det_tab)
        det_layout.setSpacing(10)
        det_layout.setContentsMargins(10, 10, 10, 10)
        
        # Conf Group
        conf_group = QGroupBox("Detection Settings")
        conf_layout = QFormLayout(conf_group)
        
        self.wand_num_cams = QSpinBox()
        self._apply_input_style(self.wand_num_cams)
        self.wand_num_cams.setValue(4)
        self.wand_num_cams.setRange(1, 16)
        self.wand_num_cams.valueChanged.connect(self._update_wand_table)
        
        # Wand Type (Bright/Dark)
        self.wand_type_combo = QComboBox()
        self._apply_input_style(self.wand_type_combo)
        self.wand_type_combo.addItems(["Dark on Bright", "Bright on Dark"])
        
        # Circle Radius Range (Range Slider)
        self.radius_range = RangeSlider(min_val=1, max_val=200, initial_min=20, initial_max=200, suffix=" px")
        
        # Sensitivity slider
        from .widgets import SimpleSlider
        self.sensitivity_slider = SimpleSlider(min_val=0.5, max_val=1.0, initial=0.85, decimals=2)
        
        conf_layout.addRow("Num Cameras:", self.wand_num_cams)
        conf_layout.addRow("Wand Type:", self.wand_type_combo)
        conf_layout.addRow("Radius Range:", self.radius_range)
        conf_layout.addRow("Sensitivity:", self.sensitivity_slider)
        det_layout.addWidget(conf_group)
        
        # Table
        det_layout.addWidget(QLabel("Camera Images:"))
        self.wand_cam_table = QTableWidget()
        self.wand_cam_table.setColumnCount(3)
        self.wand_cam_table.setHorizontalHeaderLabels(["", "Camera", "Source"])
        header = self.wand_cam_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.wand_cam_table.verticalHeader().setVisible(False)
        self.wand_cam_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.wand_cam_table.setShowGrid(False)
        self.wand_cam_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.wand_cam_table.setFixedHeight(100) 
        self._update_wand_table(4)
        det_layout.addWidget(self.wand_cam_table)
        
        # Frame List
        det_layout.addWidget(QLabel("Frame List (Click to Preview):"))
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(2)
        self.frame_table.setHorizontalHeaderLabels(["Index", "Filename"])
        self.frame_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.frame_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.frame_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.frame_table.cellClicked.connect(self._on_frame_table_clicked)
        self.frame_table.setFixedHeight(120) 
        det_layout.addWidget(self.frame_table)

        # Actions
        self.btn_detect_single = QPushButton("Test Detect (Current Frame)")
        self.btn_detect_single.setStyleSheet("background-color: #2a3f5f;")
        self.btn_detect_single.clicked.connect(self._detect_single_frame)
        det_layout.addWidget(self.btn_detect_single)

        self.btn_process_wand = QPushButton("1. Process All Frames / Resume")
        self.btn_process_wand.setStyleSheet("background-color: #2a3f5f; font-weight: bold;")
        self.btn_process_wand.clicked.connect(self._process_wand_frames)
        det_layout.addWidget(self.btn_process_wand)
        
        det_layout.addStretch()

        # --- Tab 2: Calibration ---
        cal_tab = QWidget()
        cal_layout = QVBoxLayout(cal_tab)
        cal_layout.setSpacing(15)
        cal_layout.setContentsMargins(10, 10, 10, 10)
        
        cal_group = QGroupBox("Calibration Settings")
        cal_form = QFormLayout(cal_group)
        
        self.wand_model_combo = QComboBox()
        self._apply_input_style(self.wand_model_combo)
        self.wand_model_combo.addItems(["Pinhole", "Pinhole+Refraction"])
        self.wand_model_combo.currentIndexChanged.connect(self._on_model_changed)
        
        self.wand_len_spin = QDoubleSpinBox()
        self._apply_input_style(self.wand_len_spin)
        self.wand_len_spin.setValue(500.0)
        self.wand_len_spin.setRange(10, 5000)
        self.wand_len_spin.setSuffix(" mm")
        
        # Image Resolution Manual Override
        self.img_width_spin = QSpinBox()
        self._apply_input_style(self.img_width_spin)
        self.img_width_spin.setRange(0, 10000)
        self.img_width_spin.setValue(0)
        self.img_width_spin.setSuffix(" px")
        self.img_width_spin.setToolTip("Set to 0 to auto-detect from loaded images.")
        
        self.img_height_spin = QSpinBox()
        self._apply_input_style(self.img_height_spin)
        self.img_height_spin.setRange(0, 10000)
        self.img_height_spin.setValue(0)
        self.img_height_spin.setSuffix(" px")
        self.img_height_spin.setToolTip("Set to 0 to auto-detect from loaded images.")

        cal_form.addRow("Camera Model:", self.wand_model_combo)
        cal_form.addRow("Wand Length:", self.wand_len_spin)
        cal_form.addRow("Image Width:", self.img_width_spin)
        cal_form.addRow("Image Height:", self.img_height_spin)
        cal_layout.addWidget(cal_group)
        
        cal_layout.addStretch()
        
        # Load Points Button
        self.btn_load_points = QPushButton("Load Wand Points (from CSV)")
        self.btn_load_points.setStyleSheet("background-color: #4a4a4a; border: 1px solid #666;")
        self.btn_load_points.clicked.connect(self._load_wand_points_for_calibration)
        cal_layout.addWidget(self.btn_load_points)
        
        # Precalibrate Check
        self.btn_precalibrate = QPushButton("Precalibrate to Check")
        self.btn_precalibrate.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 10px;")
        self.btn_precalibrate.clicked.connect(lambda: self._run_wand_calibration(precalibrate=True))
        cal_layout.addWidget(self.btn_precalibrate)

        # Run Calibration
        self.btn_calibrate_wand = QPushButton("2. Run Calibration")
        self.btn_calibrate_wand.setStyleSheet("background-color: #00d4ff; color: #000000; font-weight: bold; height: 50px; font-size: 14px;")
        self.btn_calibrate_wand.clicked.connect(lambda: self._run_wand_calibration(precalibrate=False))
        cal_layout.addWidget(self.btn_calibrate_wand)
        
        cal_layout.addStretch()

        # Add tabs
        controls_tabs.addTab(det_tab, "Point Detection")
        controls_tabs.addTab(cal_tab, "Calibration")
        
        right_layout.addWidget(controls_tabs)
        
        # Progress Label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 11px; margin-top: 5px;")
        right_layout.addWidget(self.status_label)

        layout.addWidget(vis_frame, stretch=2)
        layout.addWidget(right_panel)
        
        return tab

    def _load_wand_points_for_calibration_OLD_DUPLICATE(self):
        """Prompt to load a CSV file, populate wand points, and ready for calibration."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Wand Points", "", "CSV Files (*.csv)")
        if not file_path:
            return
            
        success, msg = self.wand_calibrator.load_wand_data_from_csv(file_path)
        if success:
            # Ensure calibrator has access to cameras/size
            if self.wand_images:
                 self.wand_calibrator.cameras = {}
                 for c, imgs in self.wand_images.items():
                     self.wand_calibrator.cameras[c] = {'images': imgs}
                     
            QMessageBox.information(self, "Success", msg + "\nYou can now run calibration.")
            count_frames = len(self.wand_calibrator.wand_points)
            self.status_label.setText(f"Loaded {count_frames} frames. Ready to calibrate.")
            # Switch to Vis tab 0 if needed or show loaded points on current frame?
            # Ideally we'd visualize, but just loading is fine for now.
        else:
            QMessageBox.critical(self, "Error", f"Failed to load points:\n{msg}")
    
    def _open_plate_calibration_guide(self):
        """Open the Plate Calibration user guide in the default browser."""
        import webbrowser
        from pathlib import Path
        
        guide_path = Path(__file__).parent / "plate_calibration" / "PLATE_CALIBRATION_USER_GUIDE.html"
        if guide_path.exists():
            webbrowser.open(guide_path.as_uri())
        else:
            QMessageBox.warning(self, "Guide Not Found", 
                               f"User guide not found at:\n{guide_path}")

    def _on_model_changed(self):
        """Show/hide refraction group based on selected camera model."""
        if not hasattr(self, 'refraction_group') or self.refraction_group is None:
            return
        model_name = self.wand_model_combo.currentText()
        is_refraction = (model_name == "Pinhole+Refraction")
        self.refraction_group.setVisible(is_refraction)

    def _default_plate_refraction_setting(self):
        return {
            'media1_type': 'Air',
            'media2_type': 'Acrylic',
            'media3_type': 'Water',
            'n1': 1.0,
            'n2': 1.49,
            'n3': 1.33,
            'thickness': 10.0,
        }

    def _save_plate_refraction_for_cam(self, cam_idx: int):
        if not hasattr(self, 'plate_refraction_settings'):
            self.plate_refraction_settings = {}
        if not hasattr(self, 'plate_media1_combo'):
            return
        key = int(self._plate_refraction_setting_key())
        self.plate_refraction_settings[key] = {
            'media1_type': self.plate_media1_combo.currentText(),
            'media2_type': self.plate_media2_combo.currentText(),
            'media3_type': self.plate_media3_combo.currentText(),
            'n1': float(self.plate_media1_index.value()),
            'n2': float(self.plate_media2_index.value()),
            'n3': float(self.plate_media3_index.value()),
            'thickness': float(self.plate_media2_thick.value()),
        }

    def _load_plate_refraction_for_cam(self, cam_idx: int):
        if not hasattr(self, 'plate_media1_combo'):
            return
        key = int(self._plate_refraction_setting_key())
        st = self.plate_refraction_settings.get(key, self._default_plate_refraction_setting())
        self.plate_refraction_settings[key] = dict(st)

        widgets = [
            self.plate_media1_combo,
            self.plate_media2_combo,
            self.plate_media3_combo,
            self.plate_media1_index,
            self.plate_media2_index,
            self.plate_media3_index,
            self.plate_media2_thick,
        ]
        for w in widgets:
            w.blockSignals(True)

        try:
            self.plate_media1_combo.setCurrentText(str(st.get('media1_type', 'Air')))
            self.plate_media2_combo.setCurrentText(str(st.get('media2_type', 'Acrylic')))
            self.plate_media3_combo.setCurrentText(str(st.get('media3_type', 'Water')))
            self.plate_media1_index.setValue(float(st.get('n1', 1.0)))
            self.plate_media2_index.setValue(float(st.get('n2', 1.49)))
            self.plate_media3_index.setValue(float(st.get('n3', 1.33)))
            self.plate_media2_thick.setValue(float(st.get('thickness', 10.0)))
        finally:
            for w in widgets:
                w.blockSignals(False)

        self._on_plate_refraction_media_changed()

    def _on_plate_target_cam_changed(self, idx):
        if idx < 0:
            return
        self._load_plate_refraction_for_cam(int(idx))

    def _on_plate_model_changed(self):
        if not hasattr(self, 'plate_refraction_group'):
            return
        is_refraction = (self.plate_model_combo.currentText() == "Pinhole+Refraction")
        self.plate_refraction_group.setVisible(is_refraction)
        if hasattr(self, 'plate_intrinsics_label'):
            self.plate_intrinsics_label.setVisible(is_refraction)
        if hasattr(self, 'plate_intrinsics_table'):
            self.plate_intrinsics_table.setVisible(is_refraction)
        self._set_plate_global_intrinsic_controls_visible(not is_refraction)
        if hasattr(self, 'cal_target_cam_combo'):
            self._refresh_plate_target_camera_combo()
        if hasattr(self, 'plate_num_cams') and hasattr(self, 'plate_cam_window_table'):
            self._update_plate_refraction_cam_table(int(self.plate_num_cams.value()))
        if hasattr(self, 'plate_num_cams') and hasattr(self, 'plate_intrinsics_table'):
            if is_refraction:
                self._build_plate_image_size_hints_from_saved_data()
                self._autofill_plate_intrinsics_once_from_hints()
            self._update_plate_intrinsics_table(int(self.plate_num_cams.value()))
        if hasattr(self, 'cal_target_cam_combo'):
            self._load_plate_refraction_for_cam(self.cal_target_cam_combo.currentIndex())

    def _on_plate_refraction_setting_changed(self, _value=None):
        if not hasattr(self, 'cal_target_cam_combo'):
            return
        cam_idx = self.cal_target_cam_combo.currentIndex()
        if cam_idx >= 0:
            self._save_plate_refraction_for_cam(cam_idx)

    def _on_plate_refraction_media_changed(self):
        if not hasattr(self, 'plate_media1_combo'):
            return
        if self.plate_media1_combo.currentText() == "Air":
            self.plate_media1_index.setValue(1.0)
            self.plate_media1_index.setReadOnly(True)
        else:
            self.plate_media1_index.setReadOnly(False)

        if self.plate_media2_combo.currentText() == "Acrylic":
            self.plate_media2_index.setValue(1.49)
            self.plate_media2_index.setReadOnly(True)
        elif self.plate_media2_combo.currentText() == "Glass":
            self.plate_media2_index.setValue(1.52)
            self.plate_media2_index.setReadOnly(True)
        else:
            self.plate_media2_index.setReadOnly(False)

        if self.plate_media3_combo.currentText() == "Water":
            self.plate_media3_index.setValue(1.33)
            self.plate_media3_index.setReadOnly(True)
        else:
            self.plate_media3_index.setReadOnly(False)

        self._on_plate_refraction_setting_changed()

    def _on_refraction_media_changed(self):
        """Handle material selection and update refractive index inputs."""
        # Media 1: Camera Side
        if self.media1_combo.currentText() == "Air":
            self.media1_index.setValue(1.0)
            self.media1_index.setReadOnly(True)
        else:
            self.media1_index.setReadOnly(False)

        # Media 2: Window
        if self.media2_combo.currentText() == "Acrylic":
            self.media2_index.setValue(1.49)
            self.media2_index.setReadOnly(True)
        elif self.media2_combo.currentText() == "Glass":
            self.media2_index.setValue(1.52)
            self.media2_index.setReadOnly(True)
        else:
            self.media2_index.setReadOnly(False)

        # Media 3: Measure Side
        if self.media3_combo.currentText() == "Water":
            self.media3_index.setValue(1.33)
            self.media3_index.setReadOnly(True)
        else:
            self.media3_index.setReadOnly(False)

    def _run_refractive_wand_calibration(self, wand_len, dist_coeff_num):
        """Extract refraction config and run RefractiveWandCalibrator (PR1)."""
        try:
            # 1. Extract Refraction Config from UI
            num_windows = self.window_count_spin.value()
            
            # cam_to_window mapping from table
            cam_to_window = {}
            for row in range(self.cam_window_table.rowCount()):
                id_item = self.cam_window_table.item(row, 0)
                if id_item:
                    cid = int(id_item.text())
                    win_combo = self.cam_window_table.cellWidget(row, 1)
                    if win_combo:
                        # Extract "X" from "Window X"
                        win_id = int(win_combo.currentText().split()[-1])
                        cam_to_window[cid] = win_id

            # media parameters (currently shared/per-window)
            window_media = {}
            for wid in range(num_windows):
                window_media[wid] = {
                    'n1': self.media1_index.value(),
                    'n2': self.media2_index.value(),
                    'n3': self.media3_index.value(),
                    'thickness': self.media2_thick.value()
                }

            self._refr_cam_to_window = dict(cam_to_window)
            self._refr_window_media = {wid: dict(media) for wid, media in window_media.items()}

            # 2. Output Path (camFile directory)
            parent_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Refraction camFile")
            if not parent_dir:
                print("[Refractive] Calibration cancelled: No directory selected.")
                self._busy_end('wand_calibration')
                return
            
            # Create a timestamped subfolder
            # [MODIFIED] Use 'camFile' per user request (overwrites if exists)
            out_folder_name = "camFile"
            out_path = os.path.join(parent_dir, out_folder_name)

            # 3. Initialize worker inputs from per-camera table values
            camera_settings = self._collect_camera_settings_from_table()
            active_cam_ids = sorted(self.wand_calibrator.cameras.keys())

            missing_cam_settings = [cid for cid in active_cam_ids if cid not in camera_settings]
            if missing_cam_settings:
                raise ValueError(
                    f"[Refractive] Missing table settings for active cameras: {missing_cam_settings}"
                )

            invalid_cam_settings = []
            for cid in active_cam_ids:
                cs = camera_settings[cid]
                if cs['focal'] <= 0 or cs['width'] <= 0 or cs['height'] <= 0:
                    invalid_cam_settings.append((cid, cs['focal'], cs['width'], cs['height']))
            if invalid_cam_settings:
                raise ValueError(
                    "[Refractive] Invalid per-camera table values (cam_id, focal, width, height): "
                    f"{invalid_cam_settings}"
                )

            # Keep this only as compatibility hint for legacy code paths.
            first_active = active_cam_ids[0]
            initial_focal = float(camera_settings[first_active]['focal'])
            print("[Refractive] Per-camera settings from table:")
            for cid in active_cam_ids:
                cs = camera_settings[cid]
                print(
                    f"  Cam {cid}: focal={cs['focal']:.1f}px, "
                    f"size=({cs['height']},{cs['width']})"
                )
            
            # === CHECK FOR EXISTING CACHE ===
            cache_path = os.path.join(parent_dir, "bundle_cache.json")
            if os.path.exists(cache_path):
                # Ask user if they want to use cache
                from PySide6.QtWidgets import QMessageBox as QMB
                reply = QMB.question(
                    self, "Cache Found",
                    f"Found existing calibration cache:\n{cache_path}\n\nLoad cached results and skip calibration?",
                    QMB.Yes | QMB.No,
                    QMB.Yes
                )
                
                if reply == QMB.Yes:
                    try:
                        import json
                        import traceback
                        with open(cache_path, 'r') as f:
                            cache_data = json.load(f)
                        
                        # Extract cached results (using correct keys from cache format)
                        cam_params = {}
                        for cid_str, params in cache_data.get('cam_params', {}).items():
                            cid = int(cid_str)
                            # cam_params in cache is an array (11 params), convert to dict with rvec/tvec
                            arr = np.array(params)
                            cam_params[cid] = {
                                'rvec': arr[0:3].reshape(3,1),
                                'tvec': arr[3:6].reshape(3,1),
                                'f': arr[6] if len(arr) > 6 else 5000.0,
                                'cx': arr[7] if len(arr) > 7 else 640.0,
                                'cy': arr[8] if len(arr) > 8 else 400.0,
                                'k1': arr[9] if len(arr) > 9 else 0.0,
                                'k2': arr[10] if len(arr) > 10 else 0.0
                            }
                        
                        # Window planes (cache uses 'planes' key, not 'window_planes')
                        win_planes = {}
                        planes_data = cache_data.get('planes', cache_data.get('window_planes', {}))
                        for wid_str, pl in planes_data.items():
                            wid = int(wid_str)
                            win_planes[wid] = {k: np.array(v) if isinstance(v, list) else v 
                                                for k, v in pl.items()}
                        
                        # Points (cache may not have these)
                        pts_3d_list = cache_data.get('points_3d', [])
                        pts_3d = np.array(pts_3d_list).reshape(-1, 3) if pts_3d_list else None
                        
                        # Subsample if too many
                        if pts_3d is not None and len(pts_3d) > 3000:
                            import random
                            indices = sorted(random.sample(range(len(pts_3d)), 3000))
                            pts_3d = pts_3d[indices]
                        
                        # Display 3D visualization
                        if hasattr(self, 'calib_3d_view'):
                            self.calib_3d_view.plot_refractive(cam_params, win_planes, pts_3d)
                            self.vis_tabs.setCurrentIndex(1)  # Switch to 3D View tab
                        
                        QMB.information(self, "Loaded", "Calibration loaded from cache.")
                        print(f"[Refractive] Loaded cached results from: {cache_path}")
                        self._busy_end('wand_calibration')
                        return  # Exit early, skip calibration
                        
                    except Exception as e:
                        print(f"[Refractive] Cache load error: {e}")
                        print(traceback.format_exc())
                        # Continue to run calibration if cache fails

            
            # BA projection residual switch controls both objective and UI display.
            use_proj_residuals = bool(getattr(self, '_refr_use_proj_residuals', False))
            self._refr_show_proj = use_proj_residuals
            self._refr_force_show_proj_for_p0 = False

            # Create progress dialog for refractive calibration
            self._create_refractive_calibration_dialog()
            
            # Disable start button
            if hasattr(self, 'btn_calibrate_wand'):
                self.btn_calibrate_wand.setEnabled(False)
            
            # Legacy fallback image size (should not be used when camera_settings is present)
            c_img_size = getattr(self.wand_calibrator, 'image_size', (800, 1280))

            # --- Setup Worker Thread ---
            self._refr_thread = QThread()
            self._refr_worker = RefractiveCalibWorker(
                wand_points=self.wand_calibrator.wand_points,
                wand_points_filtered=getattr(self.wand_calibrator, 'wand_points_filtered', None),
                wand_length=wand_len,
                initial_focal=initial_focal,
                dist_coeff_num=dist_coeff_num,
                active_cam_ids=active_cam_ids,
                all_cam_ids=active_cam_ids,
                cams_intrinsics=self.wand_calibrator.cameras,
                image_size=c_img_size,
                camera_settings=camera_settings,
                precalib_provider=self.wand_calibrator,
                use_proj_residuals=use_proj_residuals,
                num_windows=num_windows,
                cam_to_window=cam_to_window,
                window_media=window_media,
                out_path=out_path
            )
            
            self._refr_worker.moveToThread(self._refr_thread)
            
            # Connections
            self._refr_thread.started.connect(self._refr_worker.run)
            self._refr_worker.progress.connect(self._on_refractive_progress, Qt.QueuedConnection)
            self._refr_worker.finished.connect(self._on_refractive_finished, Qt.QueuedConnection)
            self._refr_worker.error.connect(self._on_refractive_error, Qt.QueuedConnection)
            
            # Cleanup
            self._refr_worker.finished.connect(self._refr_thread.quit)
            self._refr_worker.finished.connect(self._refr_worker.deleteLater)
            self._refr_thread.finished.connect(self._refr_thread.deleteLater)
            self._refr_worker.error.connect(self._refr_thread.quit)
            self._refr_worker.error.connect(self._refr_worker.deleteLater)
            
            # UI Throttling Buffer
            self._refr_iter_count = 0
            self._refr_last_phase = None
            self._refr_last_metrics = None
            
            self._refr_timer = QTimer()
            self._refr_timer.setInterval(200)
            self._refr_timer.timeout.connect(self._flush_refr_progress)
            self._refr_timer.start()
            
            # Start
            self._refr_thread.start()
            print("[Refractive] Calibration Thread Started.")
            
        except Exception as e:
            import traceback
            print(f"[Refractive] Setup Error:\n{traceback.format_exc()}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Setup Error", f"Failed to start calibration:\n{str(e)}")
            if hasattr(self, 'btn_calibrate_wand'):
                self.btn_calibrate_wand.setEnabled(True)
            self._busy_end('wand_calibration')


    def _on_window_count_changed(self):
        """Update window options in the camera-window table."""
        count = self.wand_cam_table.rowCount()
        self._update_refraction_cam_table(count)
    
    def _create_refractive_calibration_dialog(self):
        """Create a dialog with matplotlib cost plot for refractive calibration."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout, QApplication

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import LogLocator, LogFormatterMathtext
        show_proj = self._get_refr_show_proj()
        
        self._refr_calib_dialog = QDialog(self)
        self._refr_calib_dialog.setWindowTitle("Refractive Calibration Progress")
        self._refr_calib_dialog.setModal(False)  # Non-modal so processEvents() can work
        self._refr_calib_dialog.setMinimumSize(450, 380)
        self._refr_calib_dialog.setStyleSheet("background-color: #000000;")
        
        layout = QVBoxLayout(self._refr_calib_dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Phase label (top)
        self._refr_phase_label = QLabel("Initializing...")
        self._refr_phase_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff; background: transparent;")
        self._refr_phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._refr_phase_label.setWordWrap(True)
        layout.addWidget(self._refr_phase_label)
        
        # Matplotlib figure for cost curve
        self._refr_cost_fig = Figure(figsize=(4.5, 2.5), facecolor='#000000')
        self._refr_cost_canvas = FigureCanvasQTAgg(self._refr_cost_fig)
        self._refr_cost_ax = self._refr_cost_fig.add_subplot(111)
        self._refr_cost_ax.set_facecolor('#000000')
        self._refr_cost_ax.set_xlabel('Iteration', color='white', fontsize=9)
        self._refr_cost_ax.set_ylabel('Cost', color='white', fontsize=9)
        self._refr_cost_ax.tick_params(colors='white', labelsize=8)
        for spine in self._refr_cost_ax.spines.values():
            spine.set_color('#444')
        self._refr_cost_ax.set_title('Optimization Error', color='white', fontsize=10)
        self._refr_cost_ax.set_ylabel('Ray/Len (mm)', color='white')
        self._refr_cost_ax.set_yscale('log')
        self._refr_cost_ax.yaxis.set_label_position('left')
        self._refr_cost_ax.yaxis.tick_left()
        self._refr_cost_ax.yaxis.set_major_locator(LogLocator(base=10))
        self._refr_cost_ax.yaxis.set_major_formatter(LogFormatterMathtext())
        self._refr_cost_ax.grid(True, which="both", ls="--", color='#333', alpha=0.5)

        self._refr_proj_ax = self._refr_cost_ax.twinx()
        self._refr_proj_ax.set_facecolor('none')
        self._refr_proj_ax.set_ylabel('Proj (px)', color='#ff5bd2', fontsize=9)
        self._refr_proj_ax.tick_params(colors='#ff5bd2', labelsize=8)
        self._refr_proj_ax.spines['right'].set_color('#ff5bd2')
        self._refr_proj_ax.set_yscale('log')
        self._refr_proj_ax.yaxis.set_label_position('right')
        self._refr_proj_ax.yaxis.tick_right()
        self._refr_proj_ax.yaxis.set_major_locator(LogLocator(base=10))
        self._refr_proj_ax.yaxis.set_major_formatter(LogFormatterMathtext())
        self._refr_proj_ax.set_visible(show_proj)

        
        # Initialize data and line
        self._refr_cost_iterations = []
        self._refr_cost_values = []
        self._refr_len_values = []
        self._refr_proj_values = []
        self._refr_proj_iterations = []
        self._refr_cost_line, = self._refr_cost_ax.plot([], [], color='#2196F3', marker='o', markersize=3, linewidth=1.5, alpha=0.85, label='Ray')
        self._refr_len_line, = self._refr_cost_ax.plot([], [], 'orange', marker='s', markersize=3, linewidth=1.5, alpha=0.8, linestyle='--', label='Len')
        self._refr_proj_line, = self._refr_proj_ax.plot([], [], color='#ff5bd2', marker='^', markersize=3, linewidth=1.5, alpha=0.85, linestyle='-.', label='Proj')
        self._refr_proj_line.set_visible(show_proj)
        handles = [self._refr_cost_line, self._refr_len_line] + ([self._refr_proj_line] if show_proj else [])
        labels = [h.get_label() for h in handles]
        self._refr_cost_ax.legend(
            handles,
            labels,
            loc='upper left',
            bbox_to_anchor=(0.01, 0.99),
            fontsize=8,
            facecolor='#222',
            edgecolor='#444',
            labelcolor='white'
        )
        
        self._refr_cost_fig.subplots_adjust(left=0.16, right=0.86, bottom=0.18, top=0.88)
        layout.addWidget(self._refr_cost_canvas)
        
        # Metrics row (Ray | Len | Proj)
        metrics_layout = QHBoxLayout()
        
        self._refr_ray_label = QLabel("Ray: --")
        self._refr_ray_label.setStyleSheet("font-size: 14px; color: #2196F3; background: transparent;")
        self._refr_ray_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metrics_layout.addWidget(self._refr_ray_label)
        
        self._refr_len_label = QLabel("Len: --")
        self._refr_len_label.setStyleSheet("font-size: 14px; color: #FF9800; background: transparent;")
        self._refr_len_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metrics_layout.addWidget(self._refr_len_label)

        self._refr_proj_label = QLabel("Proj: --")
        self._refr_proj_label.setStyleSheet("font-size: 14px; color: #ff5bd2; background: transparent;")
        self._refr_proj_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._refr_proj_label.setVisible(show_proj)
        metrics_layout.addWidget(self._refr_proj_label)
        
        layout.addLayout(metrics_layout)
        
        # Reset iteration counter
        self._refr_iter_count = 0

        self._apply_refr_axes_layout_defaults()
        
        self._refr_calib_dialog.show()
        QApplication.processEvents()  # Ensure dialog is visible

    def _apply_refr_axes_layout_defaults(self):
        """Apply fixed axis label positions and fixed margins for all stages."""
        if not hasattr(self, '_refr_cost_ax') or not hasattr(self, '_refr_proj_ax'):
            return
        show_proj = self._get_refr_show_proj()

        self._refr_cost_ax.yaxis.set_label_position('left')
        self._refr_cost_ax.yaxis.tick_left()
        self._refr_cost_ax.yaxis.label.set_visible(True)
        self._refr_cost_ax.yaxis.label.set_clip_on(False)
        self._refr_cost_ax.yaxis.set_label_coords(-0.11, 0.5)

        self._refr_proj_ax.yaxis.set_label_position('right')
        self._refr_proj_ax.yaxis.tick_right()
        self._refr_proj_ax.yaxis.label.set_visible(show_proj)
        self._refr_proj_ax.yaxis.label.set_clip_on(False)
        self._refr_proj_ax.yaxis.set_label_coords(1.1, 0.5)
        self._refr_proj_ax.tick_params(axis='y', which='both', right=show_proj, labelright=show_proj)
        self._refr_proj_ax.spines['right'].set_visible(show_proj)
        self._refr_proj_ax.set_visible(show_proj)

    def _get_refr_show_proj(self) -> bool:
        return bool(getattr(self, '_refr_show_proj', True) or getattr(self, '_refr_force_show_proj_for_p0', False))
    
    @Slot(str, float, float, float, float)
    def _on_refractive_progress(self, phase, ray_rmse, len_rmse, proj_rmse, cost):
        """Receive progress from worker thread."""
        # Update phase always
        self._refr_last_phase = phase

        phase_l = str(phase).lower() if phase is not None else ""
        is_p0 = ("p0" in phase_l) or ("pinhole model" in phase_l)
        if is_p0:
            self._refr_force_show_proj_for_p0 = True
        
        # Only update metrics buffer if it's a real cost update (not evaluating placeholder)
        if cost > 0:
            self._refr_last_metrics = (ray_rmse, len_rmse, proj_rmse, cost)
    
    @Slot()
    def _flush_refr_progress(self):
        """Update UI from buffered progress data (Timer driven)."""
        # 1. Update Phase (status text)
        if hasattr(self, '_refr_last_phase') and self._refr_last_phase is not None:
            if hasattr(self, '_refr_phase_label'):
                self._refr_phase_label.setText(f"Phase: {self._refr_last_phase}")
            self._refr_last_phase = None # Clear to avoid redundant updates

        # 2. Update Metrics & Plot (if a real optimization step occurred)
        if hasattr(self, '_refr_last_metrics') and self._refr_last_metrics is not None:
            ray_rmse, len_rmse, proj_rmse, cost = self._refr_last_metrics
            self._refr_last_metrics = None # Clear after processing
            show_proj = self._get_refr_show_proj()

            if hasattr(self, '_refr_proj_ax'):
                self._refr_proj_ax.set_visible(show_proj)
            if hasattr(self, '_refr_proj_line'):
                self._refr_proj_line.set_visible(show_proj)
            if hasattr(self, '_refr_proj_label'):
                self._refr_proj_label.setVisible(show_proj)
            
            # Increment iteration
            if hasattr(self, '_refr_iter_count'):
                self._refr_iter_count += 1
            else:
                self._refr_iter_count = 1
            iteration = self._refr_iter_count
            
            # Update labels
            if hasattr(self, '_refr_ray_label'):
                if ray_rmse is None or ray_rmse <= 0:
                    self._refr_ray_label.setText("Ray: N.A.")
                else:
                    self._refr_ray_label.setText(f"Ray: {ray_rmse:.4f} mm")
            
            if hasattr(self, '_refr_len_label'):
                self._refr_len_label.setText(f"Len: {len_rmse:.4f} mm")

            if show_proj and hasattr(self, '_refr_proj_label'):
                if proj_rmse is None or proj_rmse <= 0:
                    self._refr_proj_label.setText("Proj: N.A.")
                else:
                    self._refr_proj_label.setText(f"Proj: {proj_rmse:.4f} px")

            # Update plot
            if hasattr(self, '_refr_cost_iterations'):
                self._refr_cost_iterations.append(iteration)
                ray_valid = not (ray_rmse is None or ray_rmse <= 0)
                if not ray_valid:
                    self._refr_cost_values.append(np.nan)
                else:
                    self._refr_cost_values.append(ray_rmse)
                self._refr_len_values.append(len_rmse)
                if show_proj:
                    proj_valid = not (proj_rmse is None or proj_rmse <= 0)
                    if proj_valid:
                        self._refr_proj_iterations.append(iteration)
                        self._refr_proj_values.append(float(proj_rmse))

                if hasattr(self, '_refr_cost_line'):
                    self._refr_cost_line.set_visible(bool(ray_valid))

                self._refr_cost_line.set_data(self._refr_cost_iterations, self._refr_cost_values)
                self._refr_len_line.set_data(self._refr_cost_iterations, self._refr_len_values)
                if show_proj:
                    self._refr_proj_line.set_data(self._refr_proj_iterations, self._refr_proj_values)

                # Keep axis styling stable across redraws.
                self._refr_cost_ax.set_xlabel('Iteration', color='white', fontsize=9)
                self._refr_cost_ax.set_ylabel('Ray/Len (mm)', color='white')
                self._refr_cost_ax.tick_params(colors='white', labelsize=8)
                self._refr_cost_ax.set_yscale('log')
                self._refr_cost_ax.yaxis.set_label_position('left')
                self._refr_cost_ax.yaxis.tick_left()
                self._refr_cost_ax.yaxis.set_visible(True)
                self._refr_cost_ax.spines['left'].set_visible(True)
                if show_proj:
                    self._refr_proj_ax.set_ylabel('Proj (px)', color='#ff5bd2', fontsize=9)
                    self._refr_proj_ax.tick_params(colors='#ff5bd2', labelsize=8)
                    self._refr_proj_ax.set_yscale('log')
                    self._refr_proj_ax.yaxis.set_label_position('right')
                    self._refr_proj_ax.yaxis.tick_right()
                    self._refr_proj_ax.yaxis.set_visible(True)
                    self._refr_proj_ax.spines['right'].set_visible(True)

                # Robust autoscale: only use positive finite values for log axes.
                x_vals = np.asarray(self._refr_cost_iterations, dtype=float)
                left_vals = []
                left_vals.extend([v for v in self._refr_len_values if np.isfinite(v) and v > 0])
                left_vals.extend([v for v in self._refr_cost_values if np.isfinite(v) and v > 0])
                right_vals = [v for v in self._refr_proj_values if np.isfinite(v) and v > 0] if show_proj else []

                if x_vals.size > 0:
                    x_min = max(0.0, float(np.min(x_vals)) - 0.5)
                    x_max = float(np.max(x_vals)) + 0.5
                    self._refr_cost_ax.set_xlim(x_min, x_max)
                    if show_proj:
                        self._refr_proj_ax.set_xlim(x_min, x_max)

                if left_vals:
                    left_vals_np = np.asarray(left_vals, dtype=float)
                    lv_min = float(np.min(left_vals_np[left_vals_np > 0]))
                    lv_max = float(np.max(left_vals_np))
                    y_min = 10.0 ** np.floor(np.log10(max(lv_min, 1e-12)))
                    y_max = 10.0 ** np.ceil(np.log10(max(lv_max, 1e-12)))
                    if y_max <= y_min:
                        y_min = y_min * 0.1
                        y_max = y_max * 10.0
                    self._refr_cost_ax.set_ylim(y_min, y_max)

                if show_proj and right_vals:
                    right_vals_np = np.asarray(right_vals, dtype=float)
                    rv_min = float(np.min(right_vals_np[right_vals_np > 0]))
                    rv_max = float(np.max(right_vals_np))
                    y2_min = 10.0 ** np.floor(np.log10(max(rv_min, 1e-12)))
                    y2_max = 10.0 ** np.ceil(np.log10(max(rv_max, 1e-12)))
                    if y2_max <= y2_min:
                        y2_min = y2_min * 0.1
                        y2_max = y2_max * 10.0
                    self._refr_proj_ax.set_ylim(y2_min, y2_max)

                self._apply_refr_axes_layout_defaults()
                self._refr_cost_canvas.draw_idle()

    @Slot(bool, object, object, object)
    def _on_refractive_finished(self, success, cam_params, report, dataset):
        """Handle calibration completion."""
        # Stop timer
        if hasattr(self, '_refr_timer'):
            self._refr_timer.stop()
        
        # Final flush
        self._flush_refr_progress()
        
        # Close dialog
        if hasattr(self, '_refr_calib_dialog') and self._refr_calib_dialog:
             self._refr_calib_dialog.close()
             
        # Re-enable button
        if hasattr(self, 'btn_calibrate_wand'):
            self.btn_calibrate_wand.setEnabled(True)
            
        if success and cam_params:
            self._refr_has_result = True
            self._refr_params_dirty = True
            self._refr_final_cam_params = cam_params
            self._refr_window_planes = report.get('window_planes', {}) if isinstance(report, dict) else {}
            self._refr_proj_err_stats = dataset.get('per_camera_proj_err_stats', {}) if isinstance(dataset, dict) else {}
            self._refr_tri_err_stats = dataset.get('per_camera_tri_err_stats', {}) if isinstance(dataset, dict) else {}

            msg_lines = [
                "Refractive Calibration Completed.",
                "Report saved to output folder."
            ]
            if hasattr(self, 'status_label'):
                self.status_label.setText("Refractive Calibration Successful")
            
            # [FIX] Inject computed errors into main calibrator for table display
            errors = dataset.get('per_frame_errors', {})
            if errors:
                self.wand_calibrator.per_frame_errors = errors
                print(f"[Refractive] Injected {len(errors)} frames of error data into UI calibrator.")

                cam_errs = {}
                len_errs = []
                per_camera_mean = dataset.get('per_camera_mean_proj_err', {})
                for err in errors.values():
                    if 'len_error' in err:
                        len_errs.append(float(err['len_error']))
                    for cid, e in err.get('cam_errors', {}).items():
                        cam_errs.setdefault(int(cid), []).append(float(e))

                if len_errs:
                    wand_rmse = float(np.sqrt(np.mean(np.square(len_errs))))
                    msg_lines.append(f"Wand Length RMSE: {wand_rmse:.4f} mm")

                if per_camera_mean:
                    msg_lines.append("Per-Camera Mean Projection Error:")
                    for cid in sorted(per_camera_mean.keys()):
                        mean_err = float(per_camera_mean[cid])
                        msg_lines.append(f"  Cam {cid}: {mean_err:.3f} px")
                elif cam_errs:
                    msg_lines.append("Per-Camera Mean Projection Error:")
                    for cid in sorted(cam_errs.keys()):
                        vals = cam_errs[cid]
                        if vals:
                            mean_err = float(np.mean(vals))
                            msg_lines.append(f"  Cam {cid}: {mean_err:.3f} px")

            QMessageBox.information(self, "Success", "\n".join(msg_lines))
            
            # 3D Visualization
            try:
                # Add window planes and 3D points
                # 1. Extract Window Planes from report
                win_planes = report.get('window_planes', {})
                
                # 2. Extract 3D Points from dataset
                pts_3d_list = dataset.get('points_3d', [])
                pts_3d = None
                if pts_3d_list:
                    pts_3d = np.array(pts_3d_list).reshape(-1, 3)
                    # Subsample if too many points for Matplotlib (e.g., > 5000)
                    if len(pts_3d) > 3000:
                        import random
                        indices = sorted(random.sample(range(len(pts_3d)), 3000))
                        pts_3d = pts_3d[indices]
                
                if hasattr(self, 'calib_3d_view'):
                    self.calib_3d_view.plot_refractive(cam_params, win_planes, pts_3d)
                    if hasattr(self, 'vis_tabs'):
                        self.vis_tabs.setCurrentIndex(1)
                    
            except Exception as e:
                print(f"[Refractive] Visualization Error: {e}")
            
            # Populate Error Analysis table (per_frame_errors was computed in calibrator)
            try:
                self._populate_error_table()
            except Exception as e:
                print(f"[Refractive] Error table population failed: {e}")
        else:
            QMessageBox.warning(self, "Result", "Calibration finished but no params returned.")
        self._busy_end('wand_calibration')

    def export_refractive_camfiles_to_dir(self, out_dir):
        """Export latest refractive calibration result into target camFile directory."""
        if not out_dir:
            return False
        if not self._refr_has_result or not self._refr_final_cam_params:
            return False
        if not self._refr_cam_to_window or not self._refr_window_media:
            return False

        try:
            exporter = RefractiveWandCalibrator(self.wand_calibrator)
            exporter.export_camfile_with_refraction(
                out_dir=out_dir,
                cam_params=self._refr_final_cam_params,
                window_media=self._refr_window_media,
                cam_to_window=self._refr_cam_to_window,
                window_planes=self._refr_window_planes,
                proj_err_stats=self._refr_proj_err_stats,
                tri_err_stats=self._refr_tri_err_stats,
            )
            self._refr_params_dirty = False
            return True
        except Exception as e:
            print(f"[Refractive] Export to project camFile failed: {e}")
            return False
            
    @Slot(str)
    def _on_refractive_error(self, traceback_str):
        """Handle worker error."""
        if hasattr(self, '_refr_timer'):
            self._refr_timer.stop()
            
        if hasattr(self, '_refr_calib_dialog') and self._refr_calib_dialog:
             self._refr_calib_dialog.close()
             
        if hasattr(self, 'btn_calibrate_wand'):
            self.btn_calibrate_wand.setEnabled(True)
            
        print(f"[Refractive] Worker Error:\n{traceback_str}")
        QMessageBox.critical(self, "Calibration Error", f"Worker thread failed:\n{traceback_str}")
        if hasattr(self, 'status_label'):
            self.status_label.setText("Refractive Calibration Failed")
        self._busy_end('wand_calibration')

    def _update_refraction_cam_table(self, cam_count):
        """Sync refraction camera table with current cameras and window options."""
        if not hasattr(self, 'cam_window_table') or self.cam_window_table is None:
            return
            
        self.cam_window_table.setRowCount(cam_count)
        win_count = self.window_count_spin.value()
        window_options = [f"Window {i}" for i in range(win_count)]
        
        for i in range(cam_count):
            # Col 0: Camera ID
            cam_id_item = QTableWidgetItem(f"{i}")
            cam_id_item.setFlags(Qt.ItemFlag.NoItemFlags)
            cam_id_item.setForeground(Qt.GlobalColor.white)
            cam_id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.cam_window_table.setItem(i, 0, cam_id_item)
            
            # Col 1: Window Selection
            win_combo = QComboBox()
            self._apply_input_style(win_combo)
            win_combo.addItems(window_options)
            self.cam_window_table.setCellWidget(i, 1, win_combo)
