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
        self._setup_ui()
    
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
        self.vol_x_min.setRange(-10000, 10000)
        self.vol_x_min.setValue(-200)
        vol_grid.addWidget(self.vol_x_min, 0, 1)
        self.vol_x_max = QDoubleSpinBox()
        self.vol_x_max.setRange(-10000, 10000)
        self.vol_x_max.setValue(200)
        vol_grid.addWidget(self.vol_x_max, 0, 2)
        
        # Y
        vol_grid.addWidget(QLabel("Y Min/Max:"), 1, 0)
        self.vol_y_min = QDoubleSpinBox()
        self.vol_y_min.setRange(-10000, 10000)
        self.vol_y_min.setValue(-200)
        vol_grid.addWidget(self.vol_y_min, 1, 1)
        self.vol_y_max = QDoubleSpinBox()
        self.vol_y_max.setRange(-10000, 10000)
        self.vol_y_max.setValue(200)
        vol_grid.addWidget(self.vol_y_max, 1, 2)
        
        # Z
        vol_grid.addWidget(QLabel("Z Min/Max:"), 2, 0)
        self.vol_z_min = QDoubleSpinBox()
        self.vol_z_min.setRange(-10000, 10000)
        self.vol_z_min.setValue(-200)
        vol_grid.addWidget(self.vol_z_min, 2, 1)
        self.vol_z_max = QDoubleSpinBox()
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
        
        save_btn = QPushButton(" Save Configuration")
        save_btn.setIcon(qta.icon("fa5s.save", color="white"))
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._save_configuration)
        actions_layout.addWidget(save_btn)

        validate_btn = QPushButton(" Validate Settings")
        validate_btn.setIcon(qta.icon("fa5s.check", color="white"))
        validate_btn.clicked.connect(self._validate_settings)
        actions_layout.addWidget(validate_btn)
        
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
        if has_refr_result and (refr_dirty or is_dir_empty):
            if hasattr(calib_view, 'export_refractive_camfiles_to_dir'):
                ok = calib_view.export_refractive_camfiles_to_dir(cam_dir)
                if ok:
                    n_cam = len(getattr(calib_view, '_refr_final_cam_params', {}) or {})
                    print(f"[TrackingSettings] Auto-saved {n_cam} refractive camera params to {cam_dir}")
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
            
    def _save_configuration(self):
        """Save config.txt and [type]Config.txt to project directory."""
        project_dir = self.project_path.text().strip()
        if not project_dir or not os.path.isdir(project_dir):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid Project Directory first.")
            return

        obj_type = self.obj_type_combo.currentText()
        stb_config_name = "tracerConfig.txt" if obj_type == "Tracer" else "bubbleConfig.txt"
        stb_config_path = os.path.join(project_dir, stb_config_name).replace('\\', '/')
        master_config_path = os.path.join(project_dir, "config.txt").replace('\\', '/')

        try:
            # 1. Save Master config.txt
            with open(master_config_path, 'w') as f:
                f.write("# Frame Range: [startID,endID]\n")
                f.write(f"{self.frame_start_spin.value()},{self.frame_end_spin.value()}\n")
                
                f.write("# Frame Rate: [Hz]\n")
                f.write(f"{self.fps_spin.value()}\n")
                
                f.write("# Number of Threads: (0: use as many as possible)\n")
                f.write(f"{self.n_threads_spin.value()}\n")
                
                f.write("# Number of Cameras: \n")
                n_cams = self.n_cam_spin.value()
                f.write(f"{n_cams}\n")
                
                fwrite_cam_info = "# Camera File Path, Max Intensity\n"
                f.write(fwrite_cam_info)
                cam_dir = self.camera_path_display.text().strip().replace('\\', '/')
                if " (" in cam_dir:
                    cam_dir = cam_dir.split(" (")[0]
                for i in range(n_cams):
                    if i < len(self.detected_cam_files):
                        # Use actual detected filename (e.g. vsc_cam1.txt)
                        fname = self.detected_cam_files[i]
                        f.write(f"{cam_dir}/{fname},255\n")
                    else:
                        # Fallback if request n_cams > detected files
                        f.write(f"{cam_dir}/cam{i}.txt,255\n")
                
                f.write("# Image File Path\n")
                img_dir = self.image_path_display.text().strip().replace('\\', '/')
                # Image path display might contain "(Not Found)", strip it
                if " (Not Found)" in img_dir:
                    img_dir = img_dir.replace(" (Not Found)", "")
                for i in range(n_cams):
                    f.write(f"{img_dir}/cam{i}ImageNames.txt\n")
                
                f.write("# View Volume: (xmin,xmax,ymin,ymax,zmin,zmax)\n")
                vol_str = f"{self.vol_x_min.value()},{self.vol_x_max.value()}," \
                          f"{self.vol_y_min.value()},{self.vol_y_max.value()}," \
                          f"{self.vol_z_min.value()},{self.vol_z_max.value()}"
                f.write(f"{vol_str}\n")
                
                f.write("# Voxel to MM: e.g. use 1000^3 voxel, (xmax-xmin)/1000\n")
                f.write(f"{self.voxel_spin.value()}\n")
                
                f.write("# Output Folder Path: \n")
                f.write(f"{self.output_path.text().strip()}\n")
                
                f.write("# Object Types: \n")
                f.write(f"{obj_type}\n")
                
                f.write("# STB Config Files:\n")
                f.write(f"{stb_config_path}\n")
                
                f.write("# Flag to load previous track files, previous frameID\n")
                resume_flag = 1 if self.resume_check.isChecked() else 0
                f.write(f"{resume_flag},{self.resume_frame_spin.value()}\n")
                
                results_path = self.output_path.text().strip().replace('\\', '/')
                f.write("# Path to active long track files\n")
                f.write(f"{results_path}/ConvergeTrack/\n")
                f.write("# Path to active short track files\n")
                f.write(f"{results_path}/ConvergeTrack/\n")

            # 2. Save [type]Config.txt
            with open(stb_config_path, 'w') as f:
                f.write("############################\n")
                f.write("######### Tracking #########\n")
                f.write("############################\n")
                f.write("######### Initial Phase ############## \n")
                f.write(f"{self.stb_initial_radius.value()} # Search radius for connecting tracks to objects\n")
                f.write(f"{self.stb_initial_frames.value()} # Number of frames for initial phase\n")
                f.write("######### Convergence Phase ############# \n")
                f.write(f"{self.stb_avg_spacing.value()} # Avg Interparticle spacing. (vox) to identify neighbour tracks \n\n")

                f.write("#########################\n")
                f.write("######### Shake #########\n")
                f.write("#########################\n")
                f.write(f"{self.shake_width.value()} # shake width 0.25\n\n")

                f.write("#################################\n")
                f.write("######### Predict Field #########\n")
                f.write("#################################\n")
                f.write(f"{self.pred_grid_x.value()} # xgrid \n")
                f.write(f"{self.pred_grid_y.value()} # ygrid\n")
                f.write(f"{self.pred_grid_z.value()} # zgrid\n")
                f.write(f"{self.pred_search_radius.value()} # searchRadius [voxel]\n\n")

                f.write("#######################\n")
                f.write("######### IPR #########\n")
                f.write("#######################\n")
                f.write(f"{self.ipr_loop_spin.value()}   # No. of IPR loop\n")
                f.write(f"{self.shake_loops.value()}   # No. of Shake loop\n")
                f.write(f"{self.shake_ghost.value()} # ghost threshold\n")
                f.write(f"{self.ipr_2d_tol.value()}   # 2D tolerance [px]\n")
                f.write(f"{self.ipr_3d_tol.value()}  # 3D tolerance [voxel]\n\n")

                f.write(f"{self.ipr_reduce_spin.value()} # number of reduced camera\n")
                f.write(f"{self.ipr_reduced_spin.value()} # no. of ipr loops for each reduced camera combination\n\n\n")

                f.write("###############################\n")
                f.write("######### Object Info #########\n")
                f.write("###############################\n")
                if obj_type == "Tracer":
                    f.write(f"{self.tracer_int_thresh.value()} # 2D particle finder threshold\n")
                    f.write(f"{self.tracer_radius.value()} # Particle radius [px], for calculating residue image and shaking\n")
                else:
                    f.write(f"{self.bubble_min_rad.value()}   # minimum bubble size to track\n")
                    f.write(f"{self.bubble_max_rad.value()}  # maximum bubble size to track\n")
                    f.write(f"{self.bubble_sens.value()} # sensitivity of identify circles\n")

            QMessageBox.information(self, "Success", f"Configuration saved to:\n{master_config_path}\n{stb_config_path}")
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
                params['R_inv'] = np.array(R_inv)
                
            if "Inverse of Translation Vector" in data:
                t_inv = _parse_num_list(data["Inverse of Translation Vector"][0])
                params['t_inv'] = np.array(t_inv) # Camera center in world space
                
            if "Camera Matrix" in data:
                K = []
                for row in data["Camera Matrix"]:
                    vals = _parse_num_list(row)
                    if len(vals) >= 3:
                        K.append(vals[:3])
                params['K'] = np.array(K)
                
            if "Rotation Vector" in data:
                rvec = _parse_num_list(data["Rotation Vector"][0])
                params['rvec'] = np.array(rvec)
                
            if "Translation Vector" in data:
                tvec = _parse_num_list(data["Translation Vector"][0])
                params['tvec'] = np.array(tvec)
                
            if "Distortion Coefficients" in data:
                # Handle comma separated list
                params['dist'] = np.array(_parse_num_list(data["Distortion Coefficients"][0]))

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

    def _estimate_volume_from_cameras(self, cams_data):
        """Estimate common view volume using voxel grid visibility check."""
        # 1. Estimate robust working center
        P0 = self._robust_estimate_working_center(cams_data)
        if P0 is None:
            return

        # 2. Choose automatic coarse box size
        centers = []
        for c in cams_data:
            if 't_inv' in c:
                centers.append(c['t_inv'])
            elif 'rvec' in c and 'tvec' in c:
                R, _ = cv2.Rodrigues(c['rvec'])
                centers.append((-R.T @ c['tvec'].reshape(3,1)).ravel())
        
        if not centers:
            return
            
        centers = np.array(centers)
        d = np.median(np.linalg.norm(centers - P0, axis=1))
        # Initial search box is large enough to cover expected volume
        half = np.array([0.8*d, 0.8*d, 0.8*d])
        bbmin0, bbmax0 = P0 - half, P0 + half

        # 3. Coarse scan (20mm steps)
        bbmin1, bbmax1 = self._common_fov_bbox_voxel(cams_data, bbmin0, bbmax0, step=20.0)
        if bbmin1 is None:
            print("[TrackingSettings] No coarse common FOV found.")
            return

        # 4. Fine scan (2mm steps for precision, pad 20mm)
        bbminF, bbmaxF = self._common_fov_bbox_voxel(cams_data, bbmin1 - 20, bbmax1 + 20, step=2.0)
        
        if bbminF is not None:
            # Round outward to nearest multiple of 5
            x_min = np.floor(bbminF[0] / 5.0) * 5.0
            x_max = np.ceil(bbmaxF[0] / 5.0) * 5.0
            y_min = np.floor(bbminF[1] / 5.0) * 5.0
            y_max = np.ceil(bbmaxF[1] / 5.0) * 5.0
            z_min = np.floor(bbminF[2] / 5.0) * 5.0
            z_max = np.ceil(bbmaxF[2] / 5.0) * 5.0

            self.vol_x_min.setValue(x_min)
            self.vol_x_max.setValue(x_max)
            self.vol_y_min.setValue(y_min)
            self.vol_y_max.setValue(y_max)
            self.vol_z_min.setValue(z_min)
            self.vol_z_max.setValue(z_max)
            
            # Update Voxel Size: (maxX - minX) / 1000
            voxel_size = (x_max - x_min) / 1000.0
            self.voxel_spin.setValue(voxel_size)

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

    def _common_fov_bbox_voxel(self, cams, bbox_min, bbox_max, step):
        """Check visibility on grid and return new bbox."""
        # Limiting grid size to prevent memory issues
        dim_limit = 60 # Smaller limit for performance
        x_steps = int((bbox_max[0] - bbox_min[0]) / step)
        y_steps = int((bbox_max[1] - bbox_min[1]) / step)
        z_steps = int((bbox_max[2] - bbox_min[2]) / step)
        
        if x_steps > dim_limit or y_steps > dim_limit or z_steps > dim_limit:
            step = max(step, (bbox_max - bbox_min).max() / dim_limit)
            x_steps = int((bbox_max[0] - bbox_min[0]) / step)
            y_steps = int((bbox_max[1] - bbox_min[1]) / step)
            z_steps = int((bbox_max[2] - bbox_min[2]) / step)

        xs = np.linspace(bbox_min[0], bbox_max[0], x_steps + 1)
        ys = np.linspace(bbox_min[1], bbox_max[1], y_steps + 1)
        zs = np.linspace(bbox_min[2], bbox_max[2], z_steps + 1)
        
        try:
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")
            pts_w = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
        except MemoryError:
            return None, None

        visible_all = np.ones(len(pts_w), dtype=bool)
        pinplate_cam_cache = {}
        lpt_mod = None
        lpt_import_failed = False

        for cam in cams:
            model = str(cam.get('model', '')).strip().upper()
            H = cam.get('h')
            W = cam.get('w')

            ok = None

            if model == 'PINPLATE' and H is not None and W is not None and 'cam_file_path' in cam:
                if lpt_mod is None and not lpt_import_failed:
                    try:
                        import pyopenlpt as lpt_mod
                    except Exception:
                        lpt_import_failed = True

                if lpt_mod is not None:
                    cam_path = cam['cam_file_path']
                    cam_obj = pinplate_cam_cache.get(cam_path)
                    if cam_obj is None:
                        try:
                            cam_obj = lpt_mod.Camera(cam_path)
                            pinplate_cam_cache[cam_path] = cam_obj
                        except Exception:
                            cam_obj = None

                    if cam_obj is not None:
                        try:
                            pts_world = [
                                lpt_mod.Pt3D(float(p[0]), float(p[1]), float(p[2]))
                                for p in pts_w
                            ]
                            try:
                                proj_status = cam_obj.projectBatchStatus(pts_world, False)
                            except TypeError:
                                proj_status = cam_obj.projectBatchStatus(pts_world)

                            ok_local = np.zeros(len(pts_w), dtype=bool)
                            for idx, st in enumerate(proj_status):
                                if not st or not st[0]:
                                    continue
                                uv = st[1]
                                u = float(uv[0])
                                v = float(uv[1])
                                if 0 <= u < W and 0 <= v < H:
                                    ok_local[idx] = True
                            ok = ok_local
                        except Exception:
                            ok = None

            if ok is None:
                if not all(k in cam for k in ['K', 'rvec', 'tvec', 'h', 'w']):
                    continue

                K = cam['K']
                dist = cam.get('dist', np.zeros(5))
                rvec = cam['rvec']
                tvec = cam['tvec']
                H, W = cam['h'], cam['w']

                img_pts, _ = cv2.projectPoints(pts_w, rvec, tvec, K, dist)
                uv = img_pts.reshape(-1, 2)

                R, _ = cv2.Rodrigues(rvec)
                pts_c = (R @ pts_w.T + tvec.reshape(3,1)).T
                Zc = pts_c[:, 2]

                ok = (Zc > 0) & (uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H)

            visible_all &= ok
            if not np.any(visible_all):
                return None, None

        pts_common = pts_w[visible_all]
        if len(pts_common) == 0:
            return None, None

        return pts_common.min(axis=0), pts_common.max(axis=0)

    def _validate_settings(self):
        """Validate current settings by running 2D detection and 3D matching on the first frame."""
        from PySide6.QtWidgets import QProgressDialog, QApplication
        from PySide6.QtCore import Qt
        
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
            
            # Load images for the first frame (frame_id = 0)
            imgio_list = []
            for path in basic_settings._image_file_paths:
                io = lpt.ImageIO()
                io.loadImgPath("", path)
                imgio_list.append(io)
                
            num_cams = len(imgio_list)
            frame_id = 0
            image_list = []
            for i in range(num_cams):
                image_list.append(imgio_list[i].loadImg(frame_id))
                
            progress.setLabelText("Detecting 2D Objects...")
            QApplication.processEvents()
            
            # Detect 2D objects
            obj_finder = lpt.ObjectFinder2D()
            obj2d_list = []
            total_2d_count = 0
            for cam_id in range(num_cams):
                obj2ds = obj_finder.findObject2D(image_list[cam_id], obj_cfg)
                obj2d_list.append(obj2ds)
                count = len(obj2ds)
                total_2d_count += count
                print(f"[Validation] Camera {cam_id}: found {count} 2D objects.")
                
            avg_2d_count = total_2d_count / num_cams if num_cams > 0 else 0
            
            progress.setLabelText(f"Matching 3D Objects (2D Avg: {avg_2d_count:.1f})...")
            QApplication.processEvents()
            
            # Initial 3D match
            stereomath = lpt.StereoMatch(camera_models, obj2d_list, obj_cfg)
            obj3d_list = stereomath.match()
            count_3d = len(obj3d_list)
            print(f"[Validation] Initial Match: found {count_3d} 3D objects.")
            
            # Step 1: Iterative 2D tolerance increase if 3D count is too low (< 25% of avg 2D)
            orig_tol_2d = obj_cfg._sm_param.tol_2d_px
            current_tol_2d = orig_tol_2d
            max_tol_2d_increase = 5.0
            tol_2d_step = 0.5
            modified_2d = False
            
            while count_3d < (avg_2d_count / 4.0) and (current_tol_2d - orig_tol_2d) < max_tol_2d_increase:
                current_tol_2d += tol_2d_step
                obj_cfg._sm_param.tol_2d_px = current_tol_2d
                
                progress.setLabelText(f"Stage 1 (2D Tol): Matching 3D (tol={current_tol_2d:.2f})...")
                if progress.wasCanceled(): break
                QApplication.processEvents()
                
                # Retry matching
                stereomath = lpt.StereoMatch(camera_models, obj2d_list, obj_cfg)
                obj3d_list = stereomath.match()
                count_3d = len(obj3d_list)
                modified_2d = True
                print(f"[Validation] Retry Match (2D tol={current_tol_2d:.2f}): found {count_3d} 3D objects.")

            # Step 2: Iterative 3D tolerance increase if still insufficient
            orig_tol_3d_mm = obj_cfg._sm_param.tol_3d_mm
            current_tol_3d_mm = orig_tol_3d_mm
            max_tol_3d_increase_mm = 1.0
            tol_3d_step_mm = 0.2
            modified_3d = False

            while count_3d < (avg_2d_count / 4.0) and (current_tol_3d_mm - orig_tol_3d_mm) < max_tol_3d_increase_mm:
                current_tol_3d_mm += tol_3d_step_mm
                obj_cfg._sm_param.tol_3d_mm = current_tol_3d_mm
                
                progress.setLabelText(f"Stage 2 (3D Tol): Matching 3D (tol={current_tol_3d_mm:.2f}mm)...")
                if progress.wasCanceled(): break
                QApplication.processEvents()
                
                # Retry matching
                stereomath = lpt.StereoMatch(camera_models, obj2d_list, obj_cfg)
                obj3d_list = stereomath.match()
                count_3d = len(obj3d_list)
                modified_3d = True
                print(f"[Validation] Retry Match (3D tol={current_tol_3d_mm:.2f}mm): found {count_3d} 3D objects.")
            
            progress.close()
            
            # Check final result
            if count_3d < (avg_2d_count / 4.0):
                error_msg = f"Validation failed.\n\n" \
                            f"Even with 2D tolerance increased by {current_tol_2d - orig_tol_2d:.1f}px " \
                            f"and 3D tolerance increased by {current_tol_3d_mm - orig_tol_3d_mm:.1f}mm, " \
                            f"only {count_3d} 3D objects were reconstructed from ~{avg_2d_count:.1f} 2D objects.\n\n" \
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
                          f"Adjustments: {', '.join(adjust_info)}\n" \
                          f"3D Objects: {count_3d}\n" \
                          f"Average 2D Objects: {avg_2d_count:.1f}"
                else:
                    msg = f"Validation Successful!\n\n" \
                          f"3D Objects: {count_3d}\n" \
                          f"Average 2D Objects: {avg_2d_count:.1f}"
                
                QMessageBox.information(self, "Validation Result", msg)

        except ImportError:
            progress.close()
            QMessageBox.critical(self, "Error", "pyopenlpt module not found. Please ensure it is correctly installed.")
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Validation Error", f"An error occurred during validation:\n{str(e)}")
