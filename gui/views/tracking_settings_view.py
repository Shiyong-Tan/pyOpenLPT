"""
Tracking Settings View
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea, QFileDialog, QLineEdit, QTabWidget, QMessageBox,
    QComboBox
)
from PySide6.QtCore import Qt
import os
import multiprocessing
import qtawesome as qta
import numpy as np
import re
import cv2


class TrackingSettingsView(QWidget):
    """View for configuring tracking parameters."""
    
    def __init__(self, calibration_view=None, preprocessing_view=None):
        super().__init__()
        self.calibration_view = calibration_view
        self.preprocessing_view = preprocessing_view
        self.tri_err_3sigma_mm = None # Store for dynamic voxel conversion
        self.detected_cam_files = [] # Store detected camera filenames (e.g. cam0.txt or vsc_cam1.txt)
        self.last_project_path = None # Track changes to prevent overwriting manual paths
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
    
    def showEvent(self, event):
        """Called when this view is shown. Refresh paths/data from other modules."""
        super().showEvent(event)
        self._on_cam_path_changed()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # === Main Settings Area (Scrollable) ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(16)
        
        # Title
        title = QLabel("Tracking Settings")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4ff; margin-bottom: 10px;")
        scroll_layout.addWidget(title)
        
        # === Project Directory ===
        proj_group = QGroupBox("Project Directory")
        proj_group.setStyleSheet("""
            QGroupBox { 
                background-color: #0b0f19; 
                border: 1px solid #333; 
                border-radius: 6px; 
                margin-top: 20px; 
                padding-top: 15px;
                color: #00d4ff;
                font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)
        proj_layout = QGridLayout(proj_group)
        proj_layout.setVerticalSpacing(10)
        proj_layout.setColumnStretch(1, 1) # Spacer
        proj_layout.setColumnStretch(2, 1) # Input
        
        # Row 1: Project Path
        proj_layout.addWidget(QLabel("Project Path:"), 0, 0)
        self.project_path = QLineEdit()
        self.project_path.setPlaceholderText("Select project directory...")
        self.project_path.setStyleSheet("background-color: #1a1a2e; color: white; border: 1px solid #444; padding: 5px;")
        proj_layout.addWidget(self.project_path, 0, 2)
        
        browse_btn = QPushButton("")
        browse_btn.setFixedWidth(40)
        browse_btn.setIcon(qta.icon("fa5s.folder-open", color="white"))
        browse_btn.setStyleSheet("background-color: #333; color: white; border: 1px solid #444; padding: 5px;")
        browse_btn.clicked.connect(self._browse_project)
        proj_layout.addWidget(browse_btn, 0, 3)
        
        # Row 2: Image Path
        proj_layout.addWidget(QLabel("Image Path:"), 1, 0)
        self.image_path_display = QLineEdit()
        self.image_path_display.setPlaceholderText("Select image directory (default: Project/imgFile)...")
        self.image_path_display.setStyleSheet("background-color: #151520; color: white; border: 1px solid #333; padding: 5px;")
        proj_layout.addWidget(self.image_path_display, 1, 2)
        
        img_browse_btn = QPushButton("")
        img_browse_btn.setFixedWidth(40)
        img_browse_btn.setIcon(qta.icon("fa5s.folder-open", color="white"))
        img_browse_btn.setStyleSheet("background-color: #333; color: white; border: 1px solid #444; padding: 5px;")
        img_browse_btn.clicked.connect(self._browse_image_path)
        proj_layout.addWidget(img_browse_btn, 1, 3)
        
        # Row 3: Camera Path
        proj_layout.addWidget(QLabel("Camera Path:"), 2, 0)
        self.camera_path_display = QLineEdit()
        self.camera_path_display.setPlaceholderText("Select camera directory (default: Project/camFile)...")
        self.camera_path_display.setStyleSheet("background-color: #151520; color: white; border: 1px solid #333; padding: 5px;")
        self.camera_path_display.textChanged.connect(self._on_cam_path_changed)
        proj_layout.addWidget(self.camera_path_display, 2, 2)
        
        cam_browse_btn = QPushButton("")
        cam_browse_btn.setFixedWidth(40)
        cam_browse_btn.setIcon(qta.icon("fa5s.folder-open", color="white"))
        cam_browse_btn.setStyleSheet("background-color: #333; color: white; border: 1px solid #444; padding: 5px;")
        cam_browse_btn.clicked.connect(self._browse_camera_path)
        proj_layout.addWidget(cam_browse_btn, 2, 3)
        
        scroll_layout.addWidget(proj_group)
        
        # Create tabs for different parameter groups
        tabs = QTabWidget()
        
        # === Basic Tab ===
        basic_widget = QWidget()
        basic_layout = QVBoxLayout(basic_widget)
        
        basic_group = QGroupBox("Basic Settings")
        basic_grid = QGridLayout(basic_group)
        basic_grid.setColumnStretch(1, 1) # Spacer
        basic_grid.setColumnStretch(2, 1) # Input
        
        basic_grid.addWidget(QLabel("Number of Cameras:"), 0, 0)
        self.n_cam_spin = QSpinBox()
        self.n_cam_spin.setRange(2, 16)
        self.n_cam_spin.setValue(4)
        basic_grid.addWidget(self.n_cam_spin, 0, 2)
        
        basic_grid.addWidget(QLabel("Frame Start:"), 1, 0)
        self.frame_start_spin = QSpinBox()
        self.frame_start_spin.setRange(0, 100000)
        basic_grid.addWidget(self.frame_start_spin, 1, 2)
        
        basic_grid.addWidget(QLabel("Frame End:"), 2, 0)
        self.frame_end_spin = QSpinBox()
        self.frame_end_spin.setRange(1, 100000)
        self.frame_end_spin.setValue(1000)
        basic_grid.addWidget(self.frame_end_spin, 2, 2)
        
        basic_grid.addWidget(QLabel("FPS:"), 3, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 10000)
        self.fps_spin.setValue(1000)
        basic_grid.addWidget(self.fps_spin, 3, 2)
        
        # Threads
        basic_grid.addWidget(QLabel("Number of Threads:"), 4, 0)
        self.n_threads_spin = QSpinBox()
        self.n_threads_spin.setRange(1, 128)
        self.n_threads_spin.setValue(multiprocessing.cpu_count())
        basic_grid.addWidget(self.n_threads_spin, 4, 2)
        
        basic_grid.addWidget(QLabel("Voxel to mm:"), 5, 0)
        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setDecimals(6)
        self.voxel_spin.setRange(0.000001, 100)
        self.voxel_spin.setValue(0.001)
        self.voxel_spin.valueChanged.connect(self._on_voxel_scale_changed)
        basic_grid.addWidget(self.voxel_spin, 5, 2)

        # Output Path (Moved from Actions Panel)
        basic_grid.addWidget(QLabel("Output Path:"), 6, 0)
        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Select output directory (default: Project/Results)...")
        output_layout.addWidget(self.output_path)
        
        output_browse = QPushButton("")
        output_browse.setFixedWidth(40)
        output_browse.setIcon(qta.icon("fa5s.folder-open", color="white"))
        output_browse.clicked.connect(self._browse_output)
        output_layout.addWidget(output_browse)
        
        basic_grid.addLayout(output_layout, 6, 2)
        
        # Resume Settings
        resume_group = QGroupBox("Resume Settings")
        resume_layout = QGridLayout(resume_group)
        resume_layout.setColumnStretch(1, 1)
        resume_layout.setColumnStretch(2, 1)
        
        resume_layout.addWidget(QLabel("Resume from Previous:"), 0, 0)
        self.resume_check = QCheckBox()
        resume_layout.addWidget(self.resume_check, 0, 2)
        
        resume_layout.addWidget(QLabel("Resume Frame ID:"), 1, 0)
        self.resume_frame_spin = QSpinBox()
        self.resume_frame_spin.setRange(0, 1000000)
        resume_layout.addWidget(self.resume_frame_spin, 1, 2)
        
        basic_layout.addWidget(basic_group)
        basic_layout.addWidget(resume_group)



        
        # View Volume (X, Y, Z)
        vol_group = QGroupBox("View Volume")
        vol_grid = QGridLayout(vol_group)
        vol_grid.setContentsMargins(5, 5, 5, 5)
        vol_grid.setColumnStretch(1, 1)
        vol_grid.setColumnStretch(2, 1)
        
        # X
        vol_grid.addWidget(QLabel("X Min/Max:"), 0, 0)
        self.vol_x_min = QDoubleSpinBox()
        self.vol_x_min.setDecimals(6)
        self.vol_x_min.setRange(-10000, 10000)
        self.vol_x_min.setValue(-200)
        vol_grid.addWidget(self.vol_x_min, 0, 1)
        self.vol_x_max = QDoubleSpinBox()
        self.vol_x_max.setDecimals(6)
        self.vol_x_max.setRange(-10000, 10000)
        self.vol_x_max.setValue(200)
        vol_grid.addWidget(self.vol_x_max, 0, 2)
        
        # Y
        vol_grid.addWidget(QLabel("Y Min/Max:"), 1, 0)
        self.vol_y_min = QDoubleSpinBox()
        self.vol_y_min.setDecimals(6)
        self.vol_y_min.setRange(-10000, 10000)
        self.vol_y_min.setValue(-200)
        vol_grid.addWidget(self.vol_y_min, 1, 1)
        self.vol_y_max = QDoubleSpinBox()
        self.vol_y_max.setDecimals(6)
        self.vol_y_max.setRange(-10000, 10000)
        self.vol_y_max.setValue(200)
        vol_grid.addWidget(self.vol_y_max, 1, 2)
        
        # Z
        vol_grid.addWidget(QLabel("Z Min/Max:"), 2, 0)
        self.vol_z_min = QDoubleSpinBox()
        self.vol_z_min.setDecimals(6)
        self.vol_z_min.setRange(-10000, 10000)
        self.vol_z_min.setValue(-200)
        vol_grid.addWidget(self.vol_z_min, 2, 1)
        self.vol_z_max = QDoubleSpinBox()
        self.vol_z_max.setDecimals(6)
        self.vol_z_max.setRange(-10000, 10000)
        self.vol_z_max.setValue(200)
        vol_grid.addWidget(self.vol_z_max, 2, 2)
        
        # Object Type
        obj_group = QGroupBox("Object Settings")
        obj_layout = QGridLayout(obj_group)
        obj_layout.setColumnStretch(1, 1)
        obj_layout.setColumnStretch(2, 1)
        obj_layout.addWidget(QLabel("Object Type:"), 0, 0)
        self.obj_type_combo = QComboBox()
        self.obj_type_combo.addItems(["Tracer", "Bubble"])
        obj_layout.addWidget(self.obj_type_combo, 0, 2)

        basic_layout.addWidget(obj_group)
        basic_layout.addWidget(vol_group)
        basic_layout.addStretch()
        tabs.addTab(basic_widget, "Basic")
        
        # === IPR Tab ===
        ipr_widget = QWidget()
        ipr_layout = QVBoxLayout(ipr_widget)
        
        ipr_group = QGroupBox("IPR Parameters")
        ipr_grid = QGridLayout(ipr_group)
        ipr_grid.setColumnStretch(1, 1)
        ipr_grid.setColumnStretch(2, 1)
        
        ipr_grid.addWidget(QLabel("Cameras to Reduce:"), 0, 0)
        self.ipr_reduce_spin = QSpinBox()
        self.ipr_reduce_spin.setRange(0, 4)
        self.ipr_reduce_spin.setValue(1)
        ipr_grid.addWidget(self.ipr_reduce_spin, 0, 2)
        
        ipr_grid.addWidget(QLabel("IPR Loops:"), 1, 0)
        self.ipr_loop_spin = QSpinBox()
        self.ipr_loop_spin.setRange(1, 20)
        self.ipr_loop_spin.setValue(4)
        ipr_grid.addWidget(self.ipr_loop_spin, 1, 2)
        
        ipr_grid.addWidget(QLabel("Reduced Loops:"), 2, 0)
        self.ipr_reduced_spin = QSpinBox()
        self.ipr_reduced_spin.setRange(1, 20)
        self.ipr_reduced_spin.setValue(2)
        ipr_grid.addWidget(self.ipr_reduced_spin, 2, 2)
        
        ipr_grid.addWidget(QLabel("2D Tolerance (px):"), 3, 0)
        self.ipr_2d_tol = QDoubleSpinBox()
        self.ipr_2d_tol.setDecimals(4)
        self.ipr_2d_tol.setRange(0.0001, 100)
        self.ipr_2d_tol.setValue(2.0) 
        ipr_grid.addWidget(self.ipr_2d_tol, 3, 2)
        
        ipr_grid.addWidget(QLabel("3D Tolerance (voxel):"), 4, 0)
        self.ipr_3d_tol = QDoubleSpinBox()
        self.ipr_3d_tol.setDecimals(4)
        self.ipr_3d_tol.setRange(0.0001, 100)
        self.ipr_3d_tol.setValue(1.0)
        ipr_grid.addWidget(self.ipr_3d_tol, 4, 2)

        ipr_layout.addWidget(ipr_group)
        ipr_layout.addStretch()
        tabs.addTab(ipr_widget, "IPR")
        
        # === STB Tab ===
        stb_widget = QWidget()
        stb_layout = QVBoxLayout(stb_widget)
        
        stb_group = QGroupBox("STB Parameters")
        stb_grid = QGridLayout(stb_group)
        stb_grid.setColumnStretch(1, 1)
        stb_grid.setColumnStretch(2, 1)
        
        stb_grid.addWidget(QLabel("Initial Phase Search Radius (vox):"), 0, 0)
        self.stb_initial_radius = QDoubleSpinBox()
        self.stb_initial_radius.setDecimals(2)
        self.stb_initial_radius.setRange(0.01, 1000)
        self.stb_initial_radius.setValue(10.0)
        stb_grid.addWidget(self.stb_initial_radius, 0, 2)
        
        stb_grid.addWidget(QLabel("Initial Phase Frames:"), 1, 0)
        self.stb_initial_frames = QSpinBox()
        self.stb_initial_frames.setRange(1, 100)
        self.stb_initial_frames.setValue(4)
        stb_grid.addWidget(self.stb_initial_frames, 1, 2)
        
        stb_grid.addWidget(QLabel("Convergence Avg Spacing (vox):"), 2, 0)
        self.stb_avg_spacing = QDoubleSpinBox()
        self.stb_avg_spacing.setDecimals(2)
        self.stb_avg_spacing.setRange(0.01, 1000)
        self.stb_avg_spacing.setValue(30.0)
        stb_grid.addWidget(self.stb_avg_spacing, 2, 2)
        
        stb_layout.addWidget(stb_group)

        # Predict Field Group
        pred_group = QGroupBox("Predict Field")
        pred_grid = QGridLayout(pred_group)
        pred_grid.setColumnStretch(1, 1)
        pred_grid.setColumnStretch(2, 1)
        
        pred_grid.addWidget(QLabel("Grid Number (X/Y/Z):"), 0, 0)
        
        grid_xyz_layout = QHBoxLayout()
        self.pred_grid_x = QSpinBox()
        self.pred_grid_x.setRange(1, 1000)
        self.pred_grid_x.setValue(51)
        self.pred_grid_y = QSpinBox()
        self.pred_grid_y.setRange(1, 1000)
        self.pred_grid_y.setValue(51)
        self.pred_grid_z = QSpinBox()
        self.pred_grid_z.setRange(1, 1000)
        self.pred_grid_z.setValue(51)
        
        grid_xyz_layout.addWidget(self.pred_grid_x)
        grid_xyz_layout.addWidget(self.pred_grid_y)
        grid_xyz_layout.addWidget(self.pred_grid_z)
        pred_grid.addLayout(grid_xyz_layout, 0, 2)
        
        pred_grid.addWidget(QLabel("Search Radius (voxel):"), 1, 0)
        self.pred_search_radius = QDoubleSpinBox()
        self.pred_search_radius.setRange(0.0001, 1000)
        self.pred_search_radius.setValue(25.0)
        pred_grid.addWidget(self.pred_search_radius, 1, 2)
        
        stb_layout.addWidget(pred_group)
        
        # Shake Group (Moved from Shake Tab)
        shake_group = QGroupBox("Shake")
        shake_grid = QGridLayout(shake_group)
        shake_grid.setColumnStretch(1, 1)
        shake_grid.setColumnStretch(2, 1)
        
        shake_grid.addWidget(QLabel("Shake Width (voxel):"), 0, 0)
        self.shake_width = QDoubleSpinBox()
        self.shake_width.setDecimals(4)
        self.shake_width.setRange(0.0001, 100)
        self.shake_width.setValue(0.25) # User requested default
        shake_grid.addWidget(self.shake_width, 0, 2)
        
        # Added Shake Loops just in case, default 4 as before or maybe user doesn't want it?
        # User didn't request it but it might be necessary for backend.
        shake_grid.addWidget(QLabel("Shake Loops:"), 1, 0)
        self.shake_loops = QSpinBox()
        self.shake_loops.setRange(1, 20)
        self.shake_loops.setValue(4)
        shake_grid.addWidget(self.shake_loops, 1, 2)

        shake_grid.addWidget(QLabel("Ghost Threshold:"), 2, 0)
        self.shake_ghost = QDoubleSpinBox()
        self.shake_ghost.setDecimals(3)
        self.shake_ghost.setRange(0.001, 1.0)
        self.shake_ghost.setValue(0.01)  # Default changed to 0.01
        shake_grid.addWidget(self.shake_ghost, 2, 2)
        
        stb_layout.addWidget(shake_group)
        stb_layout.addStretch()
        
        tabs.addTab(stb_widget, "STB")
        
        # === Object Tab (New) ===
        obj_tab_widget = QWidget()
        obj_tab_layout = QVBoxLayout(obj_tab_widget)
        
        # Stacked widget to switch between Tracer and Bubble settings
        from PySide6.QtWidgets import QStackedWidget
        self.obj_stack = QStackedWidget()
        
        # 1. Tracer Settings
        tracer_page = QWidget()
        tracer_layout = QVBoxLayout(tracer_page)
        tracer_group = QGroupBox("Tracer Parameters")
        tracer_grid = QGridLayout(tracer_group)
        tracer_grid.setColumnStretch(1, 1)
        tracer_grid.setColumnStretch(2, 1)
        
        tracer_grid.addWidget(QLabel("Tracer Intensity Threshold:"), 0, 0)
        self.tracer_int_thresh = QSpinBox()
        self.tracer_int_thresh.setRange(0, 255)
        self.tracer_int_thresh.setValue(30)
        tracer_grid.addWidget(self.tracer_int_thresh, 0, 2)
        
        tracer_grid.addWidget(QLabel("Tracer Radius (px):"), 1, 0)
        self.tracer_radius = QDoubleSpinBox()
        self.tracer_radius.setRange(0.1, 100)
        self.tracer_radius.setValue(2.0)
        tracer_grid.addWidget(self.tracer_radius, 1, 2)
        
        tracer_layout.addWidget(tracer_group)
        tracer_layout.addStretch()
        self.obj_stack.addWidget(tracer_page)
        
        # 2. Bubble Settings
        bubble_page = QWidget()
        bubble_layout = QVBoxLayout(bubble_page)
        bubble_group = QGroupBox("Bubble Parameters")
        bubble_grid = QGridLayout(bubble_group)
        bubble_grid.setColumnStretch(1, 1)
        bubble_grid.setColumnStretch(2, 1)
        
        bubble_grid.addWidget(QLabel("Min Bubble Radius:"), 0, 0)
        self.bubble_min_rad = QDoubleSpinBox()
        self.bubble_min_rad.setRange(0.1, 1000)
        self.bubble_min_rad.setValue(5.0)
        bubble_grid.addWidget(self.bubble_min_rad, 0, 2)
        
        bubble_grid.addWidget(QLabel("Max Bubble Radius:"), 1, 0)
        self.bubble_max_rad = QDoubleSpinBox()
        self.bubble_max_rad.setRange(0.1, 1000)
        self.bubble_max_rad.setValue(50.0)
        bubble_grid.addWidget(self.bubble_max_rad, 1, 2)
        
        bubble_grid.addWidget(QLabel("Sensitivity:"), 2, 0)
        self.bubble_sens = QDoubleSpinBox()
        self.bubble_sens.setRange(0.01, 1.0)
        self.bubble_sens.setValue(0.8)
        self.bubble_sens.setSingleStep(0.1)
        bubble_grid.addWidget(self.bubble_sens, 2, 2)
        
        bubble_layout.addWidget(bubble_group)
        bubble_layout.addStretch()
        self.obj_stack.addWidget(bubble_page)
        
        obj_tab_layout.addWidget(self.obj_stack)
        tabs.addTab(obj_tab_widget, "Object")
        
        # Connect Object Type combo to stack switch
        self.obj_type_combo.currentIndexChanged.connect(self._update_object_tab)
        # Initialize state
        self._update_object_tab(self.obj_type_combo.currentIndex())
        
        scroll_layout.addWidget(tabs)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, stretch=2)
        
        # === Right: Actions Panel ===
        actions_frame = QFrame()
        actions_frame.setObjectName("paramPanel")
        actions_frame.setFixedWidth(280)
        actions_layout = QVBoxLayout(actions_frame)
        actions_layout.setSpacing(12)
        
        actions_title = QLabel("Actions")
        actions_title.setObjectName("sectionTitle")
        actions_layout.addWidget(actions_title)
        
        actions_layout.addWidget(actions_title)
        
        validate_btn = QPushButton(" Validate Settings")
        validate_btn.setIcon(qta.icon("fa5s.check", color="white"))
        validate_btn.clicked.connect(self._validate_settings)
        actions_layout.addWidget(validate_btn)

        save_btn = QPushButton(" Save Configuration")
        save_btn.setIcon(qta.icon("fa5s.save", color="white"))
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._save_configuration)
        actions_layout.addWidget(save_btn)

        actions_layout.addStretch()
        
        layout.addWidget(actions_frame)
    
    def _browse_project(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if dir_path:
            self.project_path.setText(dir_path.replace('\\', '/'))
            self._update_paths()
            
    def _browse_image_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_path:
            self.image_path_display.setText(dir_path.replace('\\', '/'))
            
    def _browse_camera_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Camera Directory")
        if dir_path:
            self.camera_path_display.setText(dir_path.replace('\\', '/'))
            # Re-trigger save if path changes manually?
            self._save_camera_params()
            
    def _browse_config(self):
        # Legacy stub
        pass
    
    def _load_config(self):
        # TODO: Implement config loading
        pass
    
    def _browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_path.setText(dir_path.replace('\\', '/'))
            
    def showEvent(self, event):
        """Called when widget is shown. Sync paths and data."""
        super().showEvent(event)
        self._sync_from_preprocessing()
        
        current_proj = self.project_path.text().strip()
        if current_proj != self.last_project_path:
            self._update_paths()
            self.last_project_path = current_proj
        # self._save_camera_params() # Removed: handled by _on_cam_path_changed or manual triggers
        

    def _sync_from_preprocessing(self):
        """Try to fetch project path from Preprocessing View."""
        if self.preprocessing_view and hasattr(self.preprocessing_view, 'project_path_input'):
             pre_path = self.preprocessing_view.project_path_input.text().strip()
             if pre_path:
                 pre_path = pre_path.replace('\\', '/')
                 current = self.project_path.text().strip()
                 if not current or current == pre_path:
                     self.project_path.setText(pre_path)
                     
    def _update_paths(self):
        """Update derived Image and Camera paths based on Project Path."""
        project_dir = self.project_path.text().strip()
        if not project_dir:
            self.image_path_display.setText("")
            self.camera_path_display.setText("")
            return
            
        # Image Path
        if project_dir.endswith("/imgFile") or project_dir.endswith("/imgFile/"):
            img_path = project_dir.rstrip('/')
        else:
            img_path = os.path.join(project_dir, "imgFile").replace('\\', '/')
            
        if os.path.exists(img_path):
            self.image_path_display.setText(img_path)
        else:
            self.image_path_display.setText(f"{img_path} (Not Found)")
            
        # Camera Path
        if project_dir.endswith("/camFile") or project_dir.endswith("/camFile/"):
            cam_path = project_dir.rstrip('/')
        else:
            cam_path = os.path.join(project_dir, "camFile").replace('\\', '/')
            
        self.camera_path_display.setText(cam_path)
        self._on_cam_path_changed() # Manually trigger after setting text

        # Output Path (Default)
        if not self.output_path.text():
             res_path = os.path.join(project_dir, "Results").replace('\\', '/')
             self.output_path.setText(res_path)
        
        # Dynamic Defaults
        # 1. Number of Cameras (Count subdirs in imgFile)
        if os.path.exists(img_path):
            try:
                subdirs = [d for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))]
                # Filter camX folders just in case? Or just count all?
                # User req: "Count folders"
                count_cams = len(subdirs)
                if count_cams >= 2:
                    self.n_cam_spin.setValue(count_cams)
            except OSError:
                pass
                
        # 2. Frame End (Count images in imgFile/cam0)
        # Using first found camera folder if cam0 doesn't exist? usually cam1 in OpenLPT?
        # Let's try to find a valid camera folder
        first_cam_dir = None
        if os.path.exists(img_path):
            # Check likely names
            for name in ["cam1", "cam0", "cam_1", "cam_0"]:
                p = os.path.join(img_path, name)
                if os.path.isdir(p):
                    first_cam_dir = p
                    break
            # Fallback to first subdir
            if not first_cam_dir:
                try:
                    subdirs = [d for d in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, d))]
                    if subdirs:
                         first_cam_dir = os.path.join(img_path, subdirs[0])
                except OSError:
                    pass
        
        if first_cam_dir:
            try:
                # Count files
                files = [f for f in os.listdir(first_cam_dir) if f.lower().endswith(('.tif', '.png', '.jpg', '.bmp'))]
                count_frames = len(files)
                if count_frames > 0:
                    self.frame_end_spin.setValue(count_frames - 1)
            except OSError:
                pass
                
        # 3. Frame Start (Reset to 0)
        self.frame_start_spin.setValue(0)
        
    def _save_camera_params(self):
        """Auto-save camera parameters if available from Calibration module."""
        cam_dir = self.camera_path_display.text().strip()
        if not cam_dir:
            return
            
        # Strip potential " (Not Found)" or other status labels from UI copy
        if " (" in cam_dir:
            cam_dir = cam_dir.split(" (")[0]
            
        if not self.calibration_view:
            return

        calib_view = self.calibration_view

        # Create directory if it doesn't exist
        if not os.path.exists(cam_dir):
            try:
                os.makedirs(cam_dir, exist_ok=True)
            except OSError as e:
                print(f"[TrackingSettings] Error creating {cam_dir}: {e}")
                return

        is_dir_empty = not os.listdir(cam_dir)

        # Refractive export path (PINPLATE)
        has_refr_result = bool(getattr(calib_view, '_refr_has_result', False))
        refr_dirty = bool(getattr(calib_view, '_refr_params_dirty', False))
        if has_refr_result:
            if refr_dirty or is_dir_empty:
                if hasattr(calib_view, 'export_refractive_camfiles_to_dir'):
                    ok = calib_view.export_refractive_camfiles_to_dir(cam_dir)
                    if ok:
                        n_cam = len(getattr(calib_view, '_refr_final_cam_params', {}) or {})
                        print(f"[TrackingSettings] Auto-saved {n_cam} refractive camera params to {cam_dir}")
                        return
            # Refractive result exists and output is already synced.
            # Do not fall back to pinhole export, which would overwrite PINPLATE files.
            return

        if not hasattr(calib_view, 'wand_calibrator'):
            return
             
        calibrator = calib_view.wand_calibrator
        if not calibrator.final_params or calibrator.points_3d is None:
            return # No calibration data or 3D points
        
        # Save each camera directly into cam_dir
        saved_count = 0
        for cam_idx in calibrator.final_params:
            # cam_idx depends on mapping, but export uses internal logic
            # Convention: cam1.txt, cam2.txt... based on 1-based index usually?
            # User said "camX.txt". Standard OpenLPT uses 1-based naming usually? 
            # Or 0-based? "cam0.txt"?
            # File naming requirement: "camX.txt".
            # If cam_idx is 0, is it cam0.txt or cam1.txt?
            # Preprocessing used "cam{cam_idx + 1}" for folders.
            # I will check preprocess logic: `f"cam{cam_idx + 1}"`
            # So I should use 1-based index for consistency?
            # Let's use cam_idx as is if it matches context, or map it.
            # Calibrator stores by index.
            
            # Using 0-based index to match user examples (cam0.txt, cam1.txt...)
            fname = f"cam{cam_idx}.txt" 
            fpath = os.path.join(cam_dir, fname)
            
            # Use calibrator's export
            calibrator.export_to_file(cam_idx, fpath)
            saved_count += 1
            
        if saved_count > 0:
            print(f"[TrackingSettings] Auto-saved {saved_count} camera params to {cam_dir}")
            calibrator.params_dirty = False # Clear dirty flag after successful save
            
    def _render_config_files(self, project_dir):
        """Build the config.txt and [type]Config.txt CONTENTS from the
        current UI state, without writing anything to disk.

        Returns (master_config_text, stb_config_text, stb_config_name).
        """
        project_dir = os.path.abspath(project_dir)

        def _clean_ui_path(path_text):
            p = (path_text or "").strip().replace('\\', '/')
            if " (" in p:
                p = p.split(" (")[0]
            if " (Not Found)" in p:
                p = p.replace(" (Not Found)", "")
            return p

        def _to_rel(path_text):
            p = _clean_ui_path(path_text)
            if not p:
                return ""
            if not os.path.isabs(p):
                return p.replace('\\', '/')
            try:
                rel = os.path.relpath(os.path.abspath(p), start=project_dir)
            except ValueError:
                rel = p
            return rel.replace('\\', '/')

        def _join_rel(base, tail):
            base = (base or "").replace('\\', '/').rstrip('/')
            tail = (tail or "").replace('\\', '/').lstrip('/')
            if not base:
                return tail
            if not tail:
                return base
            return f"{base}/{tail}"

        obj_type = self.obj_type_combo.currentText()
        stb_config_name = "tracerConfig.txt" if obj_type == "Tracer" else "bubbleConfig.txt"
        stb_config_rel = stb_config_name

        master_lines = []
        master_lines.append("# Frame Range: [startID,endID]")
        master_lines.append(f"{self.frame_start_spin.value()},{self.frame_end_spin.value()}")

        master_lines.append("# Frame Rate: [Hz]")
        master_lines.append(f"{self.fps_spin.value()}")

        master_lines.append("# Number of Threads: (0: use as many as possible)")
        master_lines.append(f"{self.n_threads_spin.value()}")

        master_lines.append("# Number of Cameras: ")
        n_cams = self.n_cam_spin.value()
        master_lines.append(f"{n_cams}")

        master_lines.append("# Camera File Path, Max Intensity")
        cam_dir_rel = _to_rel(self.camera_path_display.text())
        for i in range(n_cams):
            if i < len(self.detected_cam_files):
                fname = self.detected_cam_files[i]
                master_lines.append(f"{_join_rel(cam_dir_rel, fname)},255")
            else:
                master_lines.append(f"{_join_rel(cam_dir_rel, f'cam{i}.txt')},255")

        master_lines.append("# Image File Path")
        img_dir_rel = _to_rel(self.image_path_display.text())
        for i in range(n_cams):
            master_lines.append(f"{_join_rel(img_dir_rel, f'cam{i}ImageNames.txt')}")

        master_lines.append("# View Volume: (xmin,xmax,ymin,ymax,zmin,zmax)")
        vol_str = f"{self.vol_x_min.value()},{self.vol_x_max.value()}," \
                  f"{self.vol_y_min.value()},{self.vol_y_max.value()}," \
                  f"{self.vol_z_min.value()},{self.vol_z_max.value()}"
        master_lines.append(f"{vol_str}")

        master_lines.append("# Voxel to MM: e.g. use 1000^3 voxel, (xmax-xmin)/1000")
        master_lines.append(f"{self.voxel_spin.value()}")

        master_lines.append("# Output Folder Path: ")
        master_lines.append(f"{_to_rel(self.output_path.text())}")

        master_lines.append("# Object Types: ")
        master_lines.append(f"{obj_type}")

        master_lines.append("# STB Config Files:")
        master_lines.append(f"{stb_config_rel}")

        master_lines.append("# Flag to load previous track files, previous frameID")
        resume_flag = 1 if self.resume_check.isChecked() else 0
        master_lines.append(f"{resume_flag},{self.resume_frame_spin.value()}")

        results_path_rel = _to_rel(self.output_path.text())
        master_lines.append("# Path to active long track files")
        master_lines.append(f"{_join_rel(results_path_rel, 'ConvergeTrack')}/")
        master_lines.append("# Path to active short track files")
        master_lines.append(f"{_join_rel(results_path_rel, 'ConvergeTrack')}/")

        master_config_text = "\n".join(master_lines) + "\n"

        stb_lines = []
        stb_lines.append("############################")
        stb_lines.append("######### Tracking #########")
        stb_lines.append("############################")
        stb_lines.append("######### Initial Phase ############## ")
        stb_lines.append(f"{self.stb_initial_radius.value()} # Search radius for connecting tracks to objects")
        stb_lines.append(f"{self.stb_initial_frames.value()} # Number of frames for initial phase")
        stb_lines.append("######### Convergence Phase ############# ")
        stb_lines.append(f"{self.stb_avg_spacing.value()} # Avg Interparticle spacing. (vox) to identify neighbour tracks ")
        stb_lines.append("")

        stb_lines.append("#########################")
        stb_lines.append("######### Shake #########")
        stb_lines.append("#########################")
        stb_lines.append(f"{self.shake_width.value()} # shake width 0.25")
        stb_lines.append("")

        stb_lines.append("#################################")
        stb_lines.append("######### Predict Field #########")
        stb_lines.append("#################################")
        stb_lines.append(f"{self.pred_grid_x.value()} # xgrid ")
        stb_lines.append(f"{self.pred_grid_y.value()} # ygrid")
        stb_lines.append(f"{self.pred_grid_z.value()} # zgrid")
        stb_lines.append(f"{self.pred_search_radius.value()} # searchRadius [voxel]")
        stb_lines.append("")

        stb_lines.append("#######################")
        stb_lines.append("######### IPR #########")
        stb_lines.append("#######################")
        stb_lines.append(f"{self.ipr_loop_spin.value()}   # No. of IPR loop")
        stb_lines.append(f"{self.shake_loops.value()}   # No. of Shake loop")
        stb_lines.append(f"{self.shake_ghost.value()} # ghost threshold")
        stb_lines.append(f"{self.ipr_2d_tol.value()}   # 2D tolerance [px]")
        stb_lines.append(f"{self.ipr_3d_tol.value()}  # 3D tolerance [voxel]")
        stb_lines.append("")

        stb_lines.append(f"{self.ipr_reduce_spin.value()} # number of reduced camera")
        stb_lines.append(f"{self.ipr_reduced_spin.value()} # no. of ipr loops for each reduced camera combination")
        stb_lines.append("")
        stb_lines.append("")

        stb_lines.append("###############################")
        stb_lines.append("######### Object Info #########")
        stb_lines.append("###############################")
        if obj_type == "Tracer":
            stb_lines.append(f"{self.tracer_int_thresh.value()} # 2D particle finder threshold")
            stb_lines.append(f"{self.tracer_radius.value()} # Particle radius [px], for calculating residue image and shaking")
        else:
            stb_lines.append(f"{self.bubble_min_rad.value()}   # minimum bubble size to track")
            stb_lines.append(f"{self.bubble_max_rad.value()}  # maximum bubble size to track")
            stb_lines.append(f"{self.bubble_sens.value()} # sensitivity of identify circles")

        stb_config_text = "\n".join(stb_lines) + "\n"

        return master_config_text, stb_config_text, stb_config_name

    def _save_configuration(self):
        """Save config.txt and [type]Config.txt to project directory."""
        project_dir = self.project_path.text().strip()
        if not project_dir or not os.path.isdir(project_dir):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid Project Directory first.")
            return

        project_dir = os.path.abspath(project_dir)

        try:
            master_config_text, stb_config_text, stb_config_name = self._render_config_files(project_dir)

            master_config_path_abs = os.path.join(project_dir, "config.txt").replace('\\', '/')
            stb_config_path_abs = os.path.join(project_dir, stb_config_name).replace('\\', '/')

            with open(master_config_path_abs, 'w') as f:
                f.write(master_config_text)

            with open(stb_config_path_abs, 'w') as f:
                f.write(stb_config_text)

            QMessageBox.information(self, "Success", f"Configuration saved to:\n{master_config_path_abs}\n{stb_config_path_abs}")
            print(f"[TrackingSettings] Saved config files to {project_dir}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save configuration:\n{str(e)}")

    def _update_object_tab(self, index):
        """Update Object tab content based on selected Object Type."""
        # Index 0: Tracer, 1: Bubble
        # Stack widgets ordered: 0->Tracer, 1->Bubble
        if index in [0, 1]:
            self.obj_stack.setCurrentIndex(index)
        else:
            self.obj_stack.setCurrentIndex(0) # Default to Tracer

    def _on_cam_path_changed(self):
        """Called when camera path is updated, manually or via project sync."""
        cam_dir = self.camera_path_display.text().strip()
        if not cam_dir:
            return
            
        # 1. Check if we have live calibration data to sync
        has_live_data = False
        if self.calibration_view:
            calibrator = self.calibration_view.wand_calibrator if hasattr(self.calibration_view, 'wand_calibrator') else None
            has_pin_live = bool(calibrator and calibrator.final_params)
            has_refr_live = bool(getattr(self.calibration_view, '_refr_has_result', False))
            if has_pin_live or has_refr_live:
                # If we have live data, we should probably update the files to ensure they are in sync
                
                # Check if sync is actually needed:
                # 1. New results available (dirty flag)
                # 2. OR Destination folder is missing/empty
                is_dir_empty = not os.path.exists(cam_dir) or not os.listdir(cam_dir)
                pin_dirty = bool(getattr(calibrator, 'params_dirty', False)) if has_pin_live else False
                refr_dirty = bool(getattr(self.calibration_view, '_refr_params_dirty', False)) if has_refr_live else False
                needs_sync = pin_dirty or refr_dirty or is_dir_empty
                
                if needs_sync:
                    # Check if heavy calculation is needed (not in cache)
                    needs_calc = bool(has_pin_live and pin_dirty and not (hasattr(calibrator, 'per_frame_errors') and calibrator.per_frame_errors))
                    
                    if needs_calc:
                        from PySide6.QtWidgets import QProgressDialog, QApplication
                        from PySide6.QtCore import Qt
                        
                        progress = QProgressDialog("Calculating IPR parameters...", None, 0, 0, self)
                        progress.setWindowTitle("Synchronizing Calibration")
                        progress.setWindowModality(Qt.WindowModality.WindowModal)
                        progress.setMinimumDuration(0)
                        progress.show()
                        QApplication.processEvents()
                        
                        try:
                            self._save_camera_params()
                        finally:
                            progress.close()
                    else:
                        # Cache exists, saving is nearly instant, skip dialog
                        self._save_camera_params()
                
                has_live_data = True
                
        # 2. Find all *cam*.txt files (Relaxed check)
        cam_files = []
        if os.path.isdir(cam_dir):
            # Look for any .txt file with "cam" in the name (e.g. vsc_cam1.txt, cam0.txt)
            cam_files = [f for f in os.listdir(cam_dir) if "cam" in f.lower() and f.endswith(".txt")]
        
        # 3. If no files found and no live data to export, show warning
        if not cam_files and not has_live_data:
            self._show_cam_params_warning()
            return
            
        def natural_key(string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
        
        # Sort and store detected files for config generation
        self.detected_cam_files = sorted(cam_files, key=natural_key)
            
        cams_data = []
        for cf in self.detected_cam_files:
            cam_file_path = os.path.join(cam_dir, cf)
            data = self._parse_cam_file(cam_file_path)
            if data:
                data['cam_file_path'] = cam_file_path
                cams_data.append(data)
        
        if cams_data:
            self._estimate_ipr_tolerance_from_cam_errors(cams_data)
        if self.detected_cam_files:
            self._estimate_volume_from_cameras(cams_data)

    def _estimate_ipr_tolerance_from_cam_errors(self, cams_data):
        """Update IPR tolerances from camera reprojection/triangulation statistics."""
        proj_stats = [c['proj_err'] for c in cams_data if 'proj_err' in c]
        tri_stats = [c['tri_err'] for c in cams_data if 'tri_err' in c]

        if proj_stats:
            means_2d = [s[0] for s in proj_stats]
            stds_2d = [s[1] for s in proj_stats]
            tol_2d = np.mean(means_2d) + 3 * np.mean(stds_2d)
            self.ipr_2d_tol.setValue(round(tol_2d, 4))

        if tri_stats:
            means_3d = [s[0] for s in tri_stats]
            stds_3d = [s[1] for s in tri_stats]
            self.tri_err_3sigma_mm = np.mean(means_3d) + 3 * np.mean(stds_3d)
            self._update_3d_tolerance_voxel()

    def _show_cam_params_warning(self):
        """Show warning if camera parameters are missing."""
        QMessageBox.warning(
            self,
            "Camera Parameters Missing",
            "No camera parameter files (cam*.txt) were found in the specified directory, "
            "and no calibrated parameters are available in the Calibration module.\n\n"
            "Please provide camera parameters in the 'camFile' directory or use the "
            "Camera Calibration tab to calibrate your cameras first.",
            QMessageBox.Ok
        )

    def _parse_cam_file(self, file_path):
        """Parse camera parameter file (Internal Python Implementation)."""
        data = {}
        try:
            def _split_num_tokens(s):
                return [tok for tok in s.replace(',', ' ').split() if tok]

            def _parse_num_list(s):
                return [float(x) for x in _split_num_tokens(s)]

            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            current_section = None
            section_lines = []
            
            for line in lines:
                line = line.strip()
                if not line: continue
                
                if line.startswith("#"):
                    if current_section and section_lines:
                        data[current_section] = section_lines
                    current_section = line.replace("#", "").split(":")[0].strip()
                    section_lines = []
                else:
                    section_lines.append(line)
            
            # Last section
            if current_section and section_lines:
                data[current_section] = section_lines
                
            # Process specific fields
            params = {}
            if "Camera Model" in data:
                params['model'] = data['Camera Model'][0]
            
            if "Image Size" in data:
                # row, col
                parts = _split_num_tokens(data['Image Size'][0])
                if len(parts) >= 2:
                    params['h'] = int(float(parts[0]))
                    params['w'] = int(float(parts[1]))
                
            if "Inverse of Rotation Matrix" in data:
                R_inv = []
                for row in data["Inverse of Rotation Matrix"]:
                    vals = _parse_num_list(row)
                    if len(vals) >= 3:
                        R_inv.append(vals[:3])
                if len(R_inv) == 3:
                    params['R_inv'] = np.array(R_inv, dtype=np.float64)
                
            if "Inverse of Translation Vector" in data:
                t_inv = _parse_num_list(data["Inverse of Translation Vector"][0])
                params['t_inv'] = np.array(t_inv) # Camera center in world space
                
            if "Camera Matrix" in data:
                K = []
                for row in data["Camera Matrix"]:
                    vals = _parse_num_list(row)
                    if len(vals) >= 3:
                        K.append(vals[:3])
                if len(K) == 3:
                    params['K'] = np.array(K, dtype=np.float64)

            if "Rotation Matrix" in data:
                R = []
                for row in data["Rotation Matrix"]:
                    vals = _parse_num_list(row)
                    if len(vals) >= 3:
                        R.append(vals[:3])
                if len(R) == 3:
                    params['R'] = np.array(R, dtype=np.float64)

            rvec_file = None
            if "Rotation Vector" in data:
                rvec_vals = _parse_num_list(data["Rotation Vector"][0])
                if len(rvec_vals) >= 3:
                    rvec_file = np.array(rvec_vals[:3], dtype=np.float64)
                    params['rvec_file'] = rvec_file
                
            if "Translation Vector" in data:
                tvec = _parse_num_list(data["Translation Vector"][0])
                if len(tvec) >= 3:
                    params['tvec'] = np.array(tvec[:3], dtype=np.float64)
                
            if "Distortion Coefficients" in data:
                # Handle comma separated list
                params['dist'] = np.array(_parse_num_list(data["Distortion Coefficients"][0]), dtype=np.float64)

            # Resolve rotation robustly from both Rotation Matrix and Rotation Vector.
            # Priority: valid Rotation Matrix -> Rodrigues(rvec).
            rvec_from_R = None
            if 'R' in params:
                R = params['R']
                # Basic orthonormality check
                try:
                    RtR = R.T @ R
                    detR = np.linalg.det(R)
                    if np.all(np.isfinite(R)) and np.linalg.norm(RtR - np.eye(3)) < 1e-2 and abs(detR - 1.0) < 1e-2:
                        rv, _ = cv2.Rodrigues(R)
                        rvec_from_R = rv.ravel().astype(np.float64)
                except Exception:
                    rvec_from_R = None

            if rvec_from_R is not None:
                params['rvec'] = rvec_from_R

                if rvec_file is not None:
                    # Cross-check consistency (angle between two rotations)
                    try:
                        R_file, _ = cv2.Rodrigues(rvec_file.reshape(3, 1))
                        dR = R_file @ params['R'].T
                        tr = float(np.trace(dR))
                        ang = np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))
                        if ang > 0.1:
                            print(f"[TrackingSettings] Warning: Rotation Vector/Matrix mismatch in {os.path.basename(file_path)} (angle diff={ang:.4f}deg). Using Rotation Matrix.")
                    except Exception:
                        pass

                    # Auto-fix placeholder rvec=0,0,0 from valid rotation matrix.
                    if np.linalg.norm(rvec_file) < 1e-12:
                        self._patch_rotation_vector_in_cam_file(file_path, rvec_from_R)
            elif rvec_file is not None:
                # Fallback: only Rotation Vector is available
                params['rvec'] = rvec_file
                try:
                    Rv, _ = cv2.Rodrigues(rvec_file.reshape(3, 1))
                    params['R'] = Rv.astype(np.float64)
                except Exception:
                    pass

            if "Camera Calibration Error" in data:
                 val = data["Camera Calibration Error"][0]
                 if val != "None":
                     try:
                         parts = _parse_num_list(val)
                         if len(parts) == 2:
                             params['proj_err'] = (parts[0], parts[1]) # (mean, std)
                         elif len(parts) == 1:
                             params['proj_err'] = (parts[0], 0.0)
                     except: pass
                     
            if "Pose Calibration Error" in data:
                 val = data["Pose Calibration Error"][0]
                 if val != "None":
                     try:
                         parts = _parse_num_list(val)
                         if len(parts) == 2:
                             params['tri_err'] = (parts[0], parts[1]) # (mean, std)
                         elif len(parts) == 1:
                             params['tri_err'] = (parts[0], 0.0)
                     except: pass

            return params
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def _patch_rotation_vector_in_cam_file(self, file_path, rvec):
        """Patch '# Rotation Vector' value in-place when file contains placeholder zeros."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            line_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('# Rotation Vector'):
                    if i + 1 < len(lines):
                        line_idx = i + 1
                    break

            if line_idx is None:
                return

            new_line = f"{float(rvec[0]):.8g},{float(rvec[1]):.8g},{float(rvec[2]):.8g}\n"
            old_line = lines[line_idx].strip()
            if old_line == new_line.strip():
                return

            lines[line_idx] = new_line
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print(f"[TrackingSettings] Patched Rotation Vector from Rotation Matrix: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[TrackingSettings] Warning: failed to patch Rotation Vector in {os.path.basename(file_path)}: {e}")

    def _estimate_volume_from_cameras(self, cams_data):
        """Apply only a resolved, converged common-FOV estimate."""
        from gui.utils.view_volume import CameraVisibility, estimate_volume

        def warn(reason):
            print(f"[TrackingSettings] View volume unchanged: {reason}")
            QMessageBox.warning(
                self, "View Volume Estimate: Low Confidence",
                f"{reason}\n\nThe current view volume and voxel scale were kept. "
                "Check the camera calibration or adjust the volume manually.")

        if len(cams_data) != len(self.detected_cam_files):
            warn("Some camera files could not be parsed.")
            return
        try:
            classify = CameraVisibility(cams_data)
            centers = []
            for cam in cams_data:
                if 't_inv' in cam:
                    center = np.asarray(cam['t_inv'], dtype=float).reshape(3)
                else:
                    R, _ = cv2.Rodrigues(np.asarray(cam['rvec'], dtype=float))
                    center = -R.T @ np.asarray(cam['tvec'], dtype=float).reshape(3)
                if not np.isfinite(center).all():
                    raise ValueError("A camera center is invalid.")
                centers.append(center)
            centers = np.asarray(centers)
            candidates = [np.zeros(3)]
            robust = self._robust_estimate_working_center(cams_data)
            if robust is not None and np.isfinite(robust).all():
                candidates.append(np.asarray(robust, dtype=float))

            result = None
            for center in candidates:
                distance = float(np.median(np.linalg.norm(centers - center, axis=1)))
                if distance <= 0 or not np.isfinite(distance):
                    continue
                # Intrinsics set dimensionless FOV factors. All seed lengths
                # scale with the camera geometry; there are no millimeter floors.
                factors = []
                for cam in cams_data:
                    K = np.asarray(cam.get('K'), dtype=float)
                    if K.shape == (3, 3) and np.isfinite(K).all() and min(abs(K[0, 0]), abs(K[1, 1])) > 0:
                        factors.append([cam['w'] / (2 * abs(K[0, 0])), cam['h'] / (2 * abs(K[1, 1]))])
                xy = np.min(factors, axis=0) if factors else np.ones(2)
                half = distance * np.array([xy[0], xy[1], 0.8])
                result = estimate_volume(classify, center - half, center + half)
                print(f"[TrackingSettings] View volume: {result.reason} samples={result.samples}")
                if result.confident:
                    break
            if result is None or not result.confident:
                warn(result.reason if result else "Camera geometry cannot define an initial search box.")
                return
        except (ValueError, KeyError, TypeError, cv2.error) as exc:
            warn(str(exc))
            return

        lower_widgets = [self.vol_x_min, self.vol_y_min, self.vol_z_min]
        upper_widgets = [self.vol_x_max, self.vol_y_max, self.vol_z_max]
        # Round only to the UI's precision, not to a fixed 5 mm lattice.
        lower = np.array([np.floor(v * 10 ** w.decimals()) / 10 ** w.decimals()
                          for v, w in zip(result.minimum, lower_widgets)])
        upper = np.array([np.ceil(v * 10 ** w.decimals()) / 10 ** w.decimals()
                          for v, w in zip(result.maximum, upper_widgets)])
        voxel = (upper[0] - lower[0]) / 1000.0
        values = list(zip(lower, lower_widgets)) + list(zip(upper, upper_widgets))
        if (np.any(upper <= lower)
                or any(not w.minimum() <= v <= w.maximum() for v, w in values)
                or not self.voxel_spin.minimum() <= voxel <= self.voxel_spin.maximum()):
            warn("The estimate is outside the supported settings range.")
            return
        for value, widget in values:
            widget.setValue(float(value))
        self.voxel_spin.setValue(float(voxel))

    def _on_voxel_scale_changed(self):
        """Update 3D tolerance in voxels if scale changes."""
        self._update_3d_tolerance_voxel()

    def _update_3d_tolerance_voxel(self):
        """Helper to convert stored 3D error (mm) to voxel units."""
        if self.tri_err_3sigma_mm is not None:
            voxel_to_mm = self.voxel_spin.value()
            if voxel_to_mm > 0:
                tol_3d_voxel = self.tri_err_3sigma_mm / voxel_to_mm
                self.ipr_3d_tol.setValue(round(tol_3d_voxel, 4))

    def _robust_estimate_working_center(self, cams_data):
        """Calculate center via pairwise ray midpoints + median."""
        def get_axis(c):
            if 'rvec' in c and 'tvec' in c:
                R, _ = cv2.Rodrigues(c['rvec'])
                C = -R.T @ c['tvec'].reshape(3,1)
                a = R.T @ np.array([[0.0],[0.0],[1.0]])
                return C.ravel(), a.ravel() / (np.linalg.norm(a) + 1e-12)
            elif 't_inv' in c and 'R_inv' in c:
                return c['t_inv'], c['R_inv'][:, 2]
            return None, None

        centers = []
        axes = []
        for cam in cams_data:
            C, a = get_axis(cam)
            if C is not None:
                centers.append(C)
                axes.append(a)
        
        if len(centers) < 2:
            return None
            
        centers = np.array(centers)
        axes = np.array(axes)
        
        mids = []
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                mids.append(self._closest_midpoint(centers[i], axes[i], centers[j], axes[j]))
        
        mids = np.array(mids)
        P0 = np.median(mids, axis=0)
        
        # Refine (reject outer 30%)
        dists = np.linalg.norm(mids - P0, axis=1)
        if len(dists) > 3:
            keep = dists < np.percentile(dists, 70)
            if np.any(keep):
                P0 = np.median(mids[keep], axis=0)
        
        return P0

    def _closest_midpoint(self, C1, a1, C2, a2):
        """Find midpoint of shortest segment between two lines."""
        w0 = C1 - C2
        a = np.dot(a1, a1)
        b = np.dot(a1, a2)
        c = np.dot(a2, a2)
        d = np.dot(a1, w0)
        e = np.dot(a2, w0)
        denom = a*c - b*b
        if abs(denom) < 1e-9:
            s = -d / (a + 1e-12)
            return 0.5 * ((C1 + s*a1) + C2)
        s = (b*e - c*d) / denom
        t = (a*e - b*d) / denom
        return 0.5 * ((C1 + s*a1) + (C2 + t*a2))

    def _validate_settings(self):
        """Validate current settings by running 2D detection and 3D matching on the configured start frame."""
        from PySide6.QtWidgets import QProgressDialog, QApplication
        from PySide6.QtCore import Qt
        self._busy_begin('validate_settings', 'Validating tracking settings')
        
        # 1. Save configuration first to ensure files are up to date
        self._save_configuration()
        
        project_dir = self.project_path.text().strip()
        if not project_dir or " (" in project_dir:
            project_dir = project_dir.split(" (")[0]
            
        config_file = os.path.join(project_dir, "config.txt").replace('\\', '/')
        if not os.path.exists(config_file):
            QMessageBox.warning(self, "Error", "Config file not found. Please save configuration first.")
            return

        # Setup Progress Dialog
        progress = QProgressDialog("Verifying Settings...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setStyleSheet("""
            QProgressDialog { background-color: #2b2b2b; color: #ffffff; padding: 15px; border: 1px solid #444; }
            QLabel { color: #ffffff; font-size: 13px; font-weight: bold; background-color: transparent; }
            QProgressBar { 
                min-height: 12px; max-height: 12px; margin: 10px 15px; 
                background-color: #444; border-radius: 4px; text-align: center; color: white;
            }
            QProgressBar::chunk { background-color: #00bcd4; border-radius: 4px; }
        """)
        progress.show()
        QApplication.processEvents()

        try:
            import pyopenlpt as lpt
            
            progress.setLabelText("Loading Configuration...")
            QApplication.processEvents()
            
            # Load basic settings
            basic_settings = lpt.BasicSetting()
            basic_settings.readConfig(config_file)
            camera_models = basic_settings._cam_list
            
            # Select correct config object
            obj_type = self.obj_type_combo.currentText()
            if obj_type == "Tracer":
                obj_cfg = lpt.TracerConfig()
            else:
                obj_cfg = lpt.BubbleConfig()
            
            # Read object specific config
            if not basic_settings._object_config_paths:
                progress.close()
                QMessageBox.warning(self, "Error", "Object configuration path missing in basic settings.")
                return
            
            obj_cfg.readConfig(basic_settings._object_config_paths[0], basic_settings)
            
            progress.setLabelText("Loading Images...")
            QApplication.processEvents()
            
            # Load images for the configured start frame.
            imgio_list = []
            folder_base = os.path.abspath(project_dir).replace('\\', '/') + '/'
            for path in basic_settings._image_file_paths:
                io = lpt.ImageIO()
                io.loadImgPath(folder_base, path)
                imgio_list.append(io)
                
            num_cams = len(imgio_list)
            frame_id = int(getattr(basic_settings, '_frame_start', 0))
            image_list = []
            progress.setLabelText(f"Loading Images (frame {frame_id})...")
            QApplication.processEvents()
            for i in range(num_cams):
                image_list.append(imgio_list[i].loadImg(frame_id))
                
            progress.setLabelText(f"Detecting 2D Objects (frame {frame_id})...")
            QApplication.processEvents()
            
            # Detect 2D objects
            obj_finder = lpt.ObjectFinder2D()
            obj2d_list = []
            total_2d_count = 0
            per_camera_2d_counts = []
            for cam_id in range(num_cams):
                obj2ds = obj_finder.findObject2D(image_list[cam_id], obj_cfg)
                obj2d_list.append(obj2ds)
                count = len(obj2ds)
                per_camera_2d_counts.append(count)
                total_2d_count += count
                print(f"[Validation] Frame {frame_id}, Camera {cam_id}: found {count} 2D objects.")
                 
            avg_2d_count = total_2d_count / num_cams if num_cams > 0 else 0
            camera_count_summary = ", ".join(
                f"Cam {cam_id}: {count}" for cam_id, count in enumerate(per_camera_2d_counts)
            ) or "No cameras"
            zero_camera_ids = [cam_id for cam_id, count in enumerate(per_camera_2d_counts) if count == 0]
            zero_camera_warning = ""
            if zero_camera_ids:
                zero_camera_warning = (
                    "\n\nWarning: no 2D objects were detected in camera(s): "
                    + ", ".join(str(cam_id) for cam_id in zero_camera_ids)
                    + "."
                )

            if total_2d_count == 0:
                progress.close()
                error_msg = f"Validation failed for frame {frame_id}.\n\n" \
                            "No 2D objects were detected in any camera, so 3D matching cannot proceed.\n\n" \
                            f"Per-camera 2D counts: {camera_count_summary}\n\n" \
                            "Check image paths, frame range, object detection thresholds, and image preprocessing settings."
                print(f"[Validation] Frame {frame_id}: no 2D objects detected in any camera.")
                QMessageBox.warning(self, "Validation Failed", error_msg)
                return
             
            progress.setLabelText(f"Matching 3D Objects (frame {frame_id}, 2D Avg: {avg_2d_count:.1f})...")
            QApplication.processEvents()
            
            # Initial 3D match
            stereomath = lpt.StereoMatch(camera_models, obj2d_list, obj_cfg)
            obj3d_list = stereomath.match()
            count_3d = len(obj3d_list)
            print(f"[Validation] Frame {frame_id} Initial Match: found {count_3d} 3D objects.")
            
            # Step 1: Iterative 2D tolerance increase if 3D count is too low (< 25% of avg 2D)
            orig_tol_2d = obj_cfg._sm_param.tol_2d_px
            current_tol_2d = orig_tol_2d
            max_tol_2d_increase = 5.0
            tol_2d_step = 0.5
            modified_2d = False
            
            while count_3d < (avg_2d_count / 4.0) and (current_tol_2d - orig_tol_2d) < max_tol_2d_increase:
                current_tol_2d += tol_2d_step
                obj_cfg._sm_param.tol_2d_px = current_tol_2d
                
                progress.setLabelText(f"Stage 1 (2D Tol): Matching 3D frame {frame_id} (tol={current_tol_2d:.2f})...")
                if progress.wasCanceled(): break
                QApplication.processEvents()
                
                # Retry matching
                stereomath = lpt.StereoMatch(camera_models, obj2d_list, obj_cfg)
                obj3d_list = stereomath.match()
                count_3d = len(obj3d_list)
                modified_2d = True
                print(f"[Validation] Frame {frame_id} Retry Match (2D tol={current_tol_2d:.2f}): found {count_3d} 3D objects.")

            # Step 2: Iterative 3D tolerance increase if still insufficient
            orig_tol_3d_mm = obj_cfg._sm_param.tol_3d_mm
            current_tol_3d_mm = orig_tol_3d_mm
            max_tol_3d_increase_mm = 1.0
            tol_3d_step_mm = 0.2
            modified_3d = False

            while count_3d < (avg_2d_count / 4.0) and (current_tol_3d_mm - orig_tol_3d_mm) < max_tol_3d_increase_mm:
                current_tol_3d_mm += tol_3d_step_mm
                obj_cfg._sm_param.tol_3d_mm = current_tol_3d_mm
                
                progress.setLabelText(f"Stage 2 (3D Tol): Matching 3D frame {frame_id} (tol={current_tol_3d_mm:.2f}mm)...")
                if progress.wasCanceled(): break
                QApplication.processEvents()
                
                # Retry matching
                stereomath = lpt.StereoMatch(camera_models, obj2d_list, obj_cfg)
                obj3d_list = stereomath.match()
                count_3d = len(obj3d_list)
                modified_3d = True
                print(f"[Validation] Frame {frame_id} Retry Match (3D tol={current_tol_3d_mm:.2f}mm): found {count_3d} 3D objects.")
            
            progress.close()
            
            # Check final result
            if count_3d < (avg_2d_count / 4.0):
                error_msg = f"Validation failed for frame {frame_id}.\n\n" \
                            f"Even with 2D tolerance increased by {current_tol_2d - orig_tol_2d:.1f}px " \
                            f"and 3D tolerance increased by {current_tol_3d_mm - orig_tol_3d_mm:.1f}mm, " \
                            f"only {count_3d} 3D objects were reconstructed from ~{avg_2d_count:.1f} 2D objects.\n\n" \
                            f"Per-camera 2D counts: {camera_count_summary}" \
                            f"{zero_camera_warning}\n\n" \
                            "The current camera parameters may be inaccurate or invalid for tracking."
                QMessageBox.warning(self, "Validation Failed", error_msg)
            else:
                if modified_2d or modified_3d:
                    # Update UI
                    if modified_2d:
                        self.ipr_2d_tol.setValue(current_tol_2d)
                    if modified_3d:
                        # Convert adjusted 3D mm back to voxel units for the UI
                        v_scale = self.voxel_spin.value()
                        new_3d_vox = current_tol_3d_mm / v_scale if v_scale > 0 else current_tol_3d_mm
                        self.ipr_3d_tol.setValue(new_3d_vox)
                        
                    # Regenerate config files with new settings
                    self._save_configuration()
                    
                    adjust_info = []
                    if modified_2d: adjust_info.append(f"2D tolerance -> {current_tol_2d:.2f}px")
                    if modified_3d: adjust_info.append(f"3D tolerance -> {current_tol_3d_mm:.2f}mm")
                    
                    msg = f"Validation successful with adjustment!\n\n" \
                          f"Validated Frame: {frame_id}\n" \
                          f"Adjustments: {', '.join(adjust_info)}\n" \
                          f"3D Objects: {count_3d}\n" \
                          f"Average 2D Objects: {avg_2d_count:.1f}\n" \
                          f"Per-camera 2D counts: {camera_count_summary}" \
                          f"{zero_camera_warning}"
                else:
                    msg = f"Validation Successful!\n\n" \
                          f"Validated Frame: {frame_id}\n" \
                          f"3D Objects: {count_3d}\n" \
                          f"Average 2D Objects: {avg_2d_count:.1f}\n" \
                          f"Per-camera 2D counts: {camera_count_summary}" \
                          f"{zero_camera_warning}"
                
                QMessageBox.information(self, "Validation Result", msg)

        except ImportError:
            progress.close()
            QMessageBox.critical(self, "Error", "pyopenlpt module not found. Please ensure it is correctly installed.")
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Validation Error", f"An error occurred during validation:\n{str(e)}")
        finally:
            self._busy_end('validate_settings')
