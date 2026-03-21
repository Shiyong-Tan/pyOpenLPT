
"""
Results View - Post-processing and Visualization
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg') # MUST be first
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QGroupBox, QComboBox, QSlider, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QProgressDialog, QScrollArea, QSizePolicy, QLineEdit,
    QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QProgressDialog, QScrollArea, QSizePolicy, QLineEdit,
    QTabWidget, QApplication, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt, QSize, QThread
import qtawesome as qta

# Import Processor
from modules.post_processing.processor import ResultsProcessor

class ResultsView(QWidget):
    """View for displaying results, filtering tracks, and kinematics analysis."""
    
    def __init__(self, settings_view=None):
        super().__init__()
        self.settings_view = settings_view
        self.processor_thread = QThread()
        self.processor = ResultsProcessor()
        self.processor.moveToThread(self.processor_thread)
        self.processor_thread.start()
        
        self.processor.data_loaded.connect(self._on_data_loaded)
        self.processor.processing_finished.connect(self._on_processing_finished)
        self.processor.export_finished.connect(self._on_export_finished)
        self.processor.error.connect(self._on_error)
        
        self.loaded_tracks = {} # Processed data
        self.metadata = {}
        self.current_fig = None
        self.current_ax = None
        self.current_canvas = None
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
        """Auto-fill project path from settings if available."""
        super().showEvent(event)
        # Always try to sync if local path is empty
        if self.settings_view and hasattr(self, 'proj_path_edit') and not self.proj_path_edit.text():
            try:
                # Access the QLineEdit in TrackingSettingsView directly
                if hasattr(self.settings_view, 'project_path'):
                    proj_path = self.settings_view.project_path.text().strip()
                    if proj_path:
                        self.proj_path_edit.setText(proj_path)
            except Exception as e:
                print(f"Sync path error: {e}")
    
    def _setup_ui(self):
        # Main Vertical Layout (Title + Content)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)
        
        # Title
        title = QLabel("Results")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4ff; margin-bottom: 10px;")
        main_layout.addWidget(title)
        
        # Content Layout (Horizontal: Vis + Controls)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        
        # === Left: Visualization Area ===
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        
        # 3D Plot Frame
        self.viz_frame = QFrame()
        self.viz_frame.setObjectName("viewFrame")
        self.viz_frame.setStyleSheet("background-color: #000; border-radius: 8px;")
        self.viz_inner_layout = QVBoxLayout(self.viz_frame)
        self.viz_inner_layout.setContentsMargins(0, 0, 0, 0)
        
        # Matplotlib Canvas Initial Setup
        self.current_fig = Figure(figsize=(8, 8), facecolor='#000000')
        self.current_canvas = FigureCanvas(self.current_fig)
        self.current_ax = self.current_fig.add_subplot(111, projection='3d')
        self.current_ax.set_facecolor('#000000')
        
        # Canvas Focus Policy for events
        self.current_canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.current_canvas.setFocus()
        
        self._setup_ax_style()
        self._setup_plot_interactions()
        
        # Toolbar Container (Top)
        toolbar_container = QWidget()
        toolbar_container.setStyleSheet("background-color: #000; border: none;")
        tb_layout = QVBoxLayout(toolbar_container)
        tb_layout.setContentsMargins(0, 5, 0, 5)
        tb_layout.setSpacing(5)
        
        # Row 1: Toolbar Buttons (Centered)
        tb_row1 = QHBoxLayout()
        tb_row1.addStretch()
        
        self.toolbar = NavigationToolbar(self.current_canvas, self.viz_frame)
        self.toolbar.setIconSize(QSize(18, 18))
        self.toolbar.setStyleSheet("""
            QToolBar { background-color: transparent; border: none; spacing: 8px; }
            QToolButton { 
                color: #fff; 
                background-color: #1a1a1a; 
                border: 1px solid #333; 
                border-radius: 12px; 
                padding: 4px 12px;
            }
            QToolButton:hover { background-color: #333; border-color: #00d4ff; }
            QToolButton:checked { background-color: #00d4ff; color: #000; }
        """)
        self.toolbar.setFixedHeight(40)
        
        # Hide default coords in toolbar
        self.toolbar.set_message = lambda s: None
        for child in self.toolbar.findChildren(QLabel):
            child.hide()
            
        tb_row1.addWidget(self.toolbar)
        tb_row1.addStretch()
        tb_layout.addLayout(tb_row1)
        
        # Row 2: Coordinate Label (Green)
        self.coord_label = QLabel("X: 0.000, Y: 0.000, Z: 0.000")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.coord_label.setStyleSheet("color: #00ff00; font-family: 'Consolas'; font-size: 13px; margin-bottom: 2px;")
        tb_layout.addWidget(self.coord_label)
        
        self.viz_inner_layout.addWidget(toolbar_container)
        self.viz_inner_layout.addWidget(self.current_canvas)
        
        viz_layout.addWidget(self.viz_frame)
        content_layout.addWidget(viz_container, stretch=3) # Visualization takes more space
        
        # === Right: Controls ===
        controls_scroll = QScrollArea()
        controls_scroll.setFixedWidth(370) # MATCHING TRACKING VIEW WIDTH
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setStyleSheet("""
            QScrollArea { background-color: transparent; border: none; }
            QWidget#scrollContents { background-color: transparent; }
        """)
        
        controls_content = QWidget()
        controls_content.setObjectName("scrollContents")
        controls_layout = QVBoxLayout(controls_content)
        controls_layout.setSpacing(16)
        controls_layout.setContentsMargins(0, 0, 5, 0) # Slightly adjust for scrollbar
        
        # 1. Project Directory & Data Loading
        load_group = QGroupBox("Project Directory")
        load_layout = QVBoxLayout(load_group)
        
        # Path Input and Browse
        path_layout = QHBoxLayout()
        self.proj_path_edit = QLineEdit()
        self.proj_path_edit.setPlaceholderText("Select project directory...")
        self.proj_path_edit.setStyleSheet("background-color: #1a1a2e; color: white; border: 1px solid #444; padding: 5px;")
        path_layout.addWidget(self.proj_path_edit)
        
        browse_btn = QPushButton()
        browse_btn.setIcon(qta.icon("fa5s.folder-open", color="white"))
        browse_btn.setFixedSize(30, 30)
        browse_btn.clicked.connect(self._browse_project)
        path_layout.addWidget(browse_btn)
        load_layout.addLayout(path_layout)
        
        # Load Button
        self.load_btn = QPushButton(" Load Tracks")
        self.load_btn.setIcon(qta.icon("fa5s.upload", color="white"))
        self.load_btn.setStyleSheet("""
            QPushButton { 
                background-color: #222; border: 1px solid #444; padding: 8px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #333; border-color: #00d4ff; }
        """)
        self.load_btn.clicked.connect(self._load_data)
        load_layout.addWidget(self.load_btn)
        
        self.status_label = QLabel("No Data Loaded")
        self.status_label.setStyleSheet("color: #888; font-style: italic;")
        load_layout.addWidget(self.status_label)
        
        controls_layout.addWidget(load_group)
        
        # 2. Filter & Kinematics Settings
        proc_group = QGroupBox("Processing && Filters")
        proc_layout = QVBoxLayout(proc_group)
        
        # Frame Rate (FPS) - Moved to top
        proc_layout.addWidget(QLabel("Frame Rate:"))
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 100000.0)
        self.fps_spin.setValue(3000.0) # Default 3000
        proc_layout.addWidget(self.fps_spin)

        # Filter Width
        proc_layout.addWidget(QLabel("Filter Width (Sigma):"))
        self.filter_width_spin = QDoubleSpinBox()
        self.filter_width_spin.setRange(0.0, 10.0)
        self.filter_width_spin.setSingleStep(0.1)
        self.filter_width_spin.setValue(1.0) # Default
        proc_layout.addWidget(self.filter_width_spin)
        
        # Acceleration Filter Width
        proc_layout.addWidget(QLabel("Acc Filter Width (Sigma):"))
        self.acc_filter_width_spin = QDoubleSpinBox()
        self.acc_filter_width_spin.setRange(0.0, 10.0)
        self.acc_filter_width_spin.setSingleStep(0.1)
        self.acc_filter_width_spin.setValue(1.0) # Default
        proc_layout.addWidget(self.acc_filter_width_spin)
        
        # --- Analysis Tabs ---
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setFixedHeight(250)
        
        # Velocity Tab
        self.vel_tab = QWidget()
        self.vel_layout = QVBoxLayout(self.vel_tab)
        self.vel_fig = Figure(figsize=(3, 2), dpi=80, facecolor='black') # BLACK Background
        self.vel_canvas = FigureCanvas(self.vel_fig)
        self.vel_layout.addWidget(self.vel_canvas)
        self.vel_layout.setContentsMargins(0,0,0,0)
        self.analysis_tabs.addTab(self.vel_tab, "Velocity Noise")
        
        # Acceleration Tab
        self.acc_tab = QWidget()
        self.acc_layout = QVBoxLayout(self.acc_tab)
        self.acc_fig = Figure(figsize=(3, 2), dpi=80, facecolor='black') # BLACK Background
        self.acc_canvas = FigureCanvas(self.acc_fig)
        self.acc_layout.addWidget(self.acc_canvas)
        self.acc_layout.setContentsMargins(0,0,0,0)
        self.analysis_tabs.addTab(self.acc_tab, "Acc Noise")
        
        proc_layout.addWidget(self.analysis_tabs)
        
        # --- Signal Check Tabs (Separate Group) ---
        self.check_tabs = QTabWidget()
        self.check_tabs.setFixedHeight(300)
        
        # 1. Position Check Tab
        self.pos_check_tab = QWidget()
        self.pos_check_layout = QVBoxLayout(self.pos_check_tab)
        self.pos_check_fig = Figure(figsize=(3, 2), dpi=80, facecolor='black')
        self.pos_check_canvas = FigureCanvas(self.pos_check_fig)
        self.pos_check_layout.addWidget(self.pos_check_canvas)
        self.pos_check_layout.setContentsMargins(0,0,0,0)
        self.check_tabs.addTab(self.pos_check_tab, "Pos Check")
        
        # 2. Velocity Check Tab
        self.vel_check_tab = QWidget()
        self.vel_check_layout = QVBoxLayout(self.vel_check_tab)
        self.vel_check_fig = Figure(figsize=(3, 2), dpi=80, facecolor='black')
        self.vel_check_canvas = FigureCanvas(self.vel_check_fig)
        self.vel_check_layout.addWidget(self.vel_check_canvas)
        self.vel_check_layout.setContentsMargins(0,0,0,0)
        self.check_tabs.addTab(self.vel_check_tab, "Vel Check")
        
        # 3. Acc Check Tab
        self.check_tab = QWidget()
        self.check_layout = QVBoxLayout(self.check_tab)
        self.check_fig = Figure(figsize=(3, 2), dpi=80, facecolor='black')
        self.check_canvas = FigureCanvas(self.check_fig)
        self.check_layout.addWidget(self.check_canvas)
        self.check_layout.setContentsMargins(0,0,0,0)
        self.check_tabs.addTab(self.check_tab, "Acc Check")

        proc_layout.addWidget(self.check_tabs)
        
        # Analysis Button
        self.analyze_btn = QPushButton(" Analyze Noise Curve")
        self.analyze_btn.setIcon(qta.icon("fa5s.microscope", color="white"))
        self.analyze_btn.setStyleSheet("background-color: #512da8;")
        self.analyze_btn.clicked.connect(self._run_optimization_analysis)
        self.analyze_btn.setEnabled(False) # Enabled when data loaded
        proc_layout.addWidget(self.analyze_btn)
        
        # Replot Check Button
        self.replot_check_btn = QPushButton(" Replot Check")
        self.replot_check_btn.setIcon(qta.icon("fa5s.sync", color="white"))
        self.replot_check_btn.setStyleSheet("background-color: #4527a0;") 
        self.replot_check_btn.clicked.connect(self._replot_check)
        self.replot_check_btn.setEnabled(False)
        proc_layout.addWidget(self.replot_check_btn)
        
        # Process Button
        self.process_btn = QPushButton(" Calculate Kinematics")
        self.process_btn.setIcon(qta.icon("fa5s.calculator", color="white"))
        self.process_btn.setStyleSheet("background-color: #006064;")
        self.process_btn.clicked.connect(self._run_processing)
        self.process_btn.setEnabled(False)
        proc_layout.addWidget(self.process_btn)
        
        controls_layout.addWidget(proc_group)
        
        # 3. Visualization Settings
        vis_group = QGroupBox("Visualization")
        vis_layout = QVBoxLayout(vis_group)
        
        vis_layout.addWidget(QLabel("Color By:"))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["Uniform", "Velocity Magnitude", "Acceleration Magnitude", "Track ID"])
        # self.color_mode_combo.currentIndexChanged.connect(self._update_plot) # Manual update now
        vis_layout.addWidget(self.color_mode_combo)
        
        # Track Count Slider
        self.count_label = QLabel("Show Top N Tracks: All")
        vis_layout.addWidget(self.count_label)
        
        slider_layout = QHBoxLayout()
        self.track_count_slider = QSlider(Qt.Orientation.Horizontal)
        self.track_count_slider.setRange(1, 100) # Default placeholder
        self.track_count_slider.setValue(100)
        self.track_count_slider.valueChanged.connect(self._on_slider_change)
        slider_layout.addWidget(self.track_count_slider)
        vis_layout.addLayout(slider_layout)
        
        # Visualize Button
        self.visualize_btn = QPushButton(" Visualize")
        self.visualize_btn.setIcon(qta.icon("fa5s.chart-line", color="white"))
        self.visualize_btn.setStyleSheet("""
            QPushButton { 
                background-color: #2e7d32; border: 1px solid #444; padding: 8px; border-radius: 4px; color: white;
            }
            QPushButton:hover { background-color: #388e3c; border-color: #00d4ff; }
        """)
        self.visualize_btn.clicked.connect(self._update_plot)
        vis_layout.addWidget(self.visualize_btn)
        
        controls_layout.addWidget(vis_group)
        
        # 4. Export
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        self.export_mat_btn = QPushButton(" Export to .mat")
        self.export_mat_btn.setIcon(qta.icon("fa5s.file-export", color="white"))
        self.export_mat_btn.clicked.connect(self._export_mat)
        self.export_mat_btn.setEnabled(False)
        export_layout.addWidget(self.export_mat_btn)
        
        controls_layout.addWidget(export_group)
        
        controls_layout.addStretch()
        controls_scroll.setWidget(controls_content)
        content_layout.addWidget(controls_scroll)
        
        main_layout.addLayout(content_layout)
    def _setup_ax_style(self):
        """Configure 3D axis style."""
        ax = self.current_ax
        ax.set_xlabel('X (mm)', color='white')
        ax.set_ylabel('Y (mm)', color='white')
        ax.set_zlabel('Z (mm)', color='white')
        ax.tick_params(colors='white', labelsize=8)
        
        pane_color = (0.0, 0.0, 0.0, 1.0)
        ax.xaxis.pane.set_facecolor(pane_color)
        ax.yaxis.pane.set_facecolor(pane_color)
        ax.zaxis.pane.set_facecolor(pane_color)
        # Grid lines color
        ax.grid(True, color='#222')
        # Edge lines color
        ax.xaxis.pane.set_edgecolor('#333')
        ax.yaxis.pane.set_edgecolor('#333')
        ax.zaxis.pane.set_edgecolor('#333')

    def _setup_plot_interactions(self):
        """Setup custom scroll zoom and mouse coordinate tracking matching TrackingView."""
        fig = self.current_fig
        ax = self.current_ax
        
        def on_scroll(event):
            if event.inaxes != ax: return
            base_scale = 1.25
            try:
                cur_xlim = ax.get_xlim3d()
                cur_ylim = ax.get_ylim3d()
                cur_zlim = ax.get_zlim3d()
                
                x_mid = (cur_xlim[0] + cur_xlim[1]) / 2.0
                y_mid = (cur_ylim[0] + cur_ylim[1]) / 2.0
                z_mid = (cur_zlim[0] + cur_zlim[1]) / 2.0
                
                x_half = (cur_xlim[1] - cur_xlim[0]) / 2.0
                y_half = (cur_ylim[1] - cur_ylim[0]) / 2.0
                z_half = (cur_zlim[1] - cur_zlim[0]) / 2.0
                
                if event.button == 'up':
                    scale_factor = 1.0 / base_scale
                elif event.button == 'down':
                    scale_factor = base_scale
                else:
                    return
                    
                ax.set_xlim3d([x_mid - x_half * scale_factor, x_mid + x_half * scale_factor])
                ax.set_ylim3d([y_mid - y_half * scale_factor, y_mid + y_half * scale_factor])
                ax.set_zlim3d([z_mid - z_half * scale_factor, z_mid + z_half * scale_factor])
                
                fig.canvas.draw_idle()
            except Exception: pass
            
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        
        def on_mouse_move(event):
            if event.inaxes != ax:
                return
            # Optimization: Skip coordinate update while dragging/rotating (button pressed)
            if event.button is not None:
                return
            try:
                coord_str = ax.format_coord(event.xdata, event.ydata)
                parts = re.findall(r"[-+]?\d*\.\d+|\d+", coord_str)
                if len(parts) >= 3:
                     formatted = f"X: {float(parts[0]):.2f}, Y: {float(parts[1]):.2f}, Z: {float(parts[2]):.2f}"
                     self.coord_label.setText(formatted)
            except Exception: pass
            
        fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    def _browse_project(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if dir_path:
            self.proj_path_edit.setText(dir_path.replace('\\', '/'))

    def _load_data(self):
        proj_dir = self.proj_path_edit.text().strip()
        if not proj_dir or not os.path.exists(proj_dir):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid Project Directory.")
            return
            
        self.load_btn.setEnabled(False)
        self.status_label.setText("Loading data...")
        self._busy_begin('load_data', 'Loading results data')
        
        # Show Progress Dialog
        self.progress_dialog = QProgressDialog("Loading tracks... Please wait.", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.show()
        
        self.processor.load_data(proj_dir)

    def _on_data_loaded(self, raw_data, metadata):
        """Data loaded callback."""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
            
        count = len(raw_data)
        self.load_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.replot_check_btn.setEnabled(True)
        self.status_label.setText(f"Loaded {count} tracks ({metadata.get('obj_type', 'Tracer')})")
        self.metadata = metadata
        
        # Update Slider Range
        self.track_count_slider.setRange(1, count)
        self.track_count_slider.setValue(min(count, 500)) # Default to showing top 500
        self._on_slider_change() # Update label
        
        # Initial Plot (Raw)
        self.loaded_tracks = raw_data # Treat raw as current until processed
        # self._update_plot() # Do NOT auto-plot raw data if it's huge, or plot top N?
        # User requested manual "Visualize" button generally, but initial load usually shows something.
        # Let's show top N automatically on load.
        self._update_plot()
        self._busy_end('load_data')
        
    def _on_error(self, msg):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
             self.progress_dialog.close()
             self.progress_dialog = None
             
        self.load_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.export_mat_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)
        self._busy_end('load_data')
        self._busy_end('process_data')
        self._busy_end('export_mat')

    def _run_optimization_analysis(self):
        """Run noise analysis curve."""
        self.analyze_btn.setEnabled(False)
        self.status_label.setText("Analyzing noise...")
        self._busy_begin('analyze_noise', 'Analyzing noise')
        QApplication.processEvents()
        
        fps = self.fps_spin.value()
        # Define range: sigma 1 to 20 (Integers, matching MATLAB)
        widths = np.arange(1, 21, 1)
        
        # Run in processor (should be quick enough on main thread if efficient, 
        # but let's assume it returns directly for simpler logic or use existing pattern)
        # For simplicity in this edit, I'll call a blocking function in processor.
        
        try:
            results = self.processor.calculate_optimization_curve(widths, fps)
            # results: {'vel': (w, stds, opt_w), 'acc': (w, stds, opt_w)}
            
            # 1. Plot Velocity Curve
            self.vel_fig.clear()
            ax_v = self.vel_fig.add_subplot(111)
            w_vel, std_vel, opt_v = results['vel']
            ax_v.plot(w_vel, std_vel, 'c.-', label='StdDev')
            # Mark optimal
            if opt_v is not None:
                idx = np.where(w_vel == opt_v)[0]
                if len(idx) > 0:
                    ax_v.plot(opt_v, std_vel[idx[0]], 'ro', label=f'Rec: {opt_v}')
                    self.filter_width_spin.setValue(float(opt_v)) # Auto-populate
            
            ax_v.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=6)
            ax_v.set_title("Velocity Noise (Norm. Avg)", color='white', fontsize=9)
            ax_v.set_xlabel("Filter Width (Sigma)", color='#aaa', fontsize=7)
            ax_v.set_ylabel("Norm. Std Dev", color='#aaa', fontsize=7)
            ax_v.tick_params(colors='white', labelsize=6)
            ax_v.grid(True, color='#333')
            ax_v.set_facecolor('black') # Black Axes
            self.vel_canvas.draw()
            
            # 2. Plot Acceleration Curve
            self.acc_fig.clear()
            ax_a = self.acc_fig.add_subplot(111)
            w_acc, std_acc, opt_a = results['acc']
            ax_a.plot(w_acc, std_acc, 'm.-', label='Avg Curve')
             # Mark optimal
            if opt_a is not None:
                idx = np.where(w_acc == opt_a)[0]
                if len(idx) > 0:
                    ax_a.plot(opt_a, std_acc[idx[0]], 'ro', label=f'Rec: {opt_a}')
                    self.acc_filter_width_spin.setValue(float(opt_a)) # Auto-populate

            ax_a.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=6)
            ax_a.set_title("Accel Noise (Norm. Avg)", color='white', fontsize=9)
            ax_a.set_xlabel("Filter Width (Sigma)", color='#aaa', fontsize=7)
            ax_a.set_ylabel("Norm. Std Dev", color='#aaa', fontsize=7)
            ax_a.tick_params(colors='white', labelsize=6)
            ax_a.grid(True, color='#333')
            ax_a.set_facecolor('black') # Black Axes
            self.acc_canvas.draw()
            
            # 3. Get Comparison Data
            rec_w_v = opt_v if opt_v else 1.0
            rec_w_a = opt_a if opt_a else 1.0
            data = self.processor.get_comparison_data(rec_w_v, rec_w_a, fps)
            
            if data:
                frames = data['frames']
                
                # --- Plot Acc Check ---
                self.check_fig.clear()
                ax_c = self.check_fig.add_subplot(111)
                ax_c.plot(frames, data['acc_raw'], color='#555', linewidth=0.5, alpha=0.7, label='Raw')
                ax_c.plot(frames, data['acc_filt'], color='red', linewidth=1.5, label=f'Filt (w={rec_w_a})')
                ax_c.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=6)
                ax_c.set_title("Acc Signal Check", color='white', fontsize=9)
                ax_c.set_xlabel("Frame", color='#aaa', fontsize=7)
                ax_c.set_ylabel("Acc Magnitude", color='#aaa', fontsize=7)
                ax_c.grid(True, color='#333')
                ax_c.set_facecolor('black')
                ax_c.tick_params(colors='white', labelsize=6)
                self.check_canvas.draw()
                
                # --- Plot Vel Check ---
                self.vel_check_fig.clear()
                ax_vc = self.vel_check_fig.add_subplot(111)
                ax_vc.plot(frames, data['vel_raw'], color='#555', linewidth=0.5, alpha=0.7, label='Raw')
                ax_vc.plot(frames, data['vel_filt'], color='cyan', linewidth=1.5, label=f'Filt (w={rec_w_v})')
                ax_vc.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=6)
                ax_vc.set_title("Vel Signal Check", color='white', fontsize=9)
                ax_vc.set_xlabel("Frame", color='#aaa', fontsize=7)
                ax_vc.set_ylabel("Vel Magnitude", color='#aaa', fontsize=7)
                ax_vc.grid(True, color='#333')
                ax_vc.set_facecolor('black')
                ax_vc.tick_params(colors='white', labelsize=6)
                self.vel_check_canvas.draw()
                
                # --- Plot Pos Check ---
                self.pos_check_fig.clear()
                ax_p = self.pos_check_fig.add_subplot(111)
                
                # Plot Raw Path (Lighter Grey, Thicker)
                raw_x = data['pos_raw'][:, 0]
                raw_y = data['pos_raw'][:, 1]
                ax_p.plot(raw_x, raw_y, color='#888', linewidth=1.0, alpha=0.8, label='Raw Path')
                
                # Plot Filtered Path (Scatter colored by Speed)
                filt_x = data['pos_filt'][:, 0]
                filt_y = data['pos_filt'][:, 1]
                speed = data['vel_filt']
                
                sc = ax_p.scatter(filt_x, filt_y, c=speed, cmap='plasma', s=3, zorder=10, label='Filt')
                
                ax_p.set_title("Pos Check (X-Y)", color='white', fontsize=9)
                ax_p.set_xlabel("X (mm)", color='#aaa', fontsize=7)
                ax_p.set_ylabel("Y (mm)", color='#aaa', fontsize=7)
                ax_p.grid(True, color='#333')
                ax_p.set_facecolor('black')
                ax_p.tick_params(colors='white', labelsize=6)
                ax_p.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=6)
                ax_p.axis('equal')
                
                self.pos_check_canvas.draw()
                
                # Enable Zoom/Pan for all check plots
                self._enable_mouse_interaction(self.pos_check_canvas, ax_p)
                self._enable_mouse_interaction(self.vel_check_canvas, ax_vc)
                self._enable_mouse_interaction(self.check_canvas, ax_c)

            self.status_label.setText("Analysis complete. Recommended widths set.")
            
        except Exception as e:
            QMessageBox.warning(self, "Analysis Failed", str(e))
        finally:
            self.analyze_btn.setEnabled(True)
            self._busy_end('analyze_noise')

    def _enable_mouse_interaction(self, canvas, ax):
        """Enable Zoom on Scroll and Pan on Right-Click Drag."""
        def zoom(event):
            base_scale = 1.2
            if event.inaxes != ax: return
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            xdata = event.xdata
            ydata = event.ydata
            if xdata is None or ydata is None: return 

            if event.button == 'up': scale_factor = 1/base_scale
            elif event.button == 'down': scale_factor = base_scale
            else: scale_factor = 1.0

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            
            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * relx])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * rely])
            canvas.draw()

        def on_press(event):
            if event.button == 3: # Right click
                canvas._pan_start = (event.x, event.y)
                canvas._pan_xlim = ax.get_xlim()
                canvas._pan_ylim = ax.get_ylim()

        def on_drag(event):
            if event.button == 3 and hasattr(canvas, '_pan_start'): # Right click drag
                dx = event.x - canvas._pan_start[0]
                dy = event.y - canvas._pan_start[1]
                
                # Convert pixel delta to data delta is tricky without transform inverted
                # Simpler: Get per-pixel scale
                bbox = ax.get_window_extent()
                xscale = (canvas._pan_xlim[1] - canvas._pan_xlim[0]) / bbox.width
                yscale = (canvas._pan_ylim[1] - canvas._pan_ylim[0]) / bbox.height
                
                ax.set_xlim(canvas._pan_xlim[0] - dx*xscale, canvas._pan_xlim[1] - dx*xscale)
                ax.set_ylim(canvas._pan_ylim[0] - dy*yscale, canvas._pan_ylim[1] - dy*yscale)
                canvas.draw()
                
        # Disconnect old IDs if re-running to avoid stacking callbacks?
        # A simple way is to clear the figure first (which we do) or store CIDs.
        # Since we fig.clear() every time, the canvas handlers *might* persist if attached to canvas?
        # Actually canvas persists. We should disconnect.
        if hasattr(canvas, '_cids'):
            for cid in canvas._cids: canvas.mpl_disconnect(cid)
        
        canvas._cids = [
            canvas.mpl_connect('scroll_event', zoom),
            canvas.mpl_connect('button_press_event', on_press),
            canvas.mpl_connect('motion_notify_event', on_drag)
        ]

    def _run_processing(self):
        """Run kinematics computation."""
        self.process_btn.setEnabled(False)
        self.status_label.setText("Processing...")
        self._busy_begin('process_data', 'Processing results')
        
        filter_width = self.filter_width_spin.value()
        acc_filter_width = self.acc_filter_width_spin.value()
        fps = self.fps_spin.value()
        
        self.processor.compute_kinematics_and_filter(filter_width, acc_filter_width, fps)

    def _on_processing_finished(self, processed_data):
        self.process_btn.setEnabled(True)
        self.export_mat_btn.setEnabled(True)
        self.loaded_tracks = processed_data
        
        count = len(processed_data)
        self.status_label.setText(f"Processed {count} tracks")
        
        # Update Slider
        self.track_count_slider.setRange(1, max(1, count))
        # Keep current relative position or reset? Reset to show all valid?
        # Let's keep value if it's within range, or clamp.
        old_val = self.track_count_slider.value()
        self.track_count_slider.setValue(min(old_val, count))
        self._on_slider_change()
        
        self._update_plot()
        QMessageBox.information(self, "Success", "Kinematics calculation complete.")
        self._busy_end('process_data')


    def _on_slider_change(self):
        """Update label only. Plotting triggered by button."""
        count = self.track_count_slider.value()
        self.count_label.setText(f"Show Top N Tracks: {count}")
    
    
    def _update_plot(self):
        """Refreshes 3D plot with Top N filtering."""
        if not self.loaded_tracks:
            return
        self._busy_begin('plot_tracks', 'Plotting tracks')
            
        # UI Feedback for Plotting
        progress = QProgressDialog("Plotting tracks... Please wait.", None, 0, 0, self)
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setCancelButton(None) 
        progress.show()
        
        # Force UI update to show dialog immediately
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
            
        ax = self.current_ax
        
        # Initialize scatter objects if they don't exist
        if not hasattr(self, 'sc_main') or self.sc_main is None:
            ax.clear()
            self._setup_ax_style()
            # Initialize empty scatter
            self.sc_main = ax.scatter([], [], [], s=1, alpha=0.6, linewidth=0)
            # Remove heads scatter (user request)
        
        # Gather data for plotting
        try:
            # 1. Sort by Length (Descending) and Filter Top N
            # Create list of (tid, track, length)
            track_items = []
            for tid, t in self.loaded_tracks.items():
                track_items.append((tid, t, len(t)))
            
            # Sort
            track_items.sort(key=lambda x: x[2], reverse=True)
            
            # Limit
            top_n = self.track_count_slider.value()
            if top_n < len(track_items):
                track_items = track_items[:top_n]
            
            # 2. Extract Data
            
            all_x, all_y, all_z = [], [], []
            all_c = [] # Color values
            
            mode = self.color_mode_combo.currentText()
            cmap_name = 'jet' # Default to jet
            
            # Check kinematics availability (from first track in filtered list)
            if not track_items:
                self.sc_main._offsets3d = ([], [], [])
                self.current_canvas.draw_idle()
                progress.close()
                return

            first_track = track_items[0][1]
            d = first_track.shape[1]
            has_kinematics = (d >= 12)
            
            # OPTIMIZATION: Use ax.plot for Uniform mode (Fastest)
            # Use ax.scatter for Colored modes (Flexible but Slower)
            is_uniform = (mode == "Uniform")
            
            if is_uniform:
                # Use ax.plot logic
                for tid, track, length in track_items:
                    all_x.extend(track[:, 1])
                    all_y.extend(track[:, 2])
                    all_z.extend(track[:, 3])
            else:
                 # Use ax.scatter logic
                 for tid, track, length in track_items:
                    sub_track = track
                    all_x.extend(sub_track[:, 1])
                    all_y.extend(sub_track[:, 2])
                    all_z.extend(sub_track[:, 3])
                    
                    n_pts = len(sub_track)
                    if mode == "Velocity Magnitude" and has_kinematics:
                        all_c.extend(sub_track[:, 7])
                    elif mode == "Acceleration Magnitude" and has_kinematics:
                        all_c.extend(sub_track[:, 11])
                    elif mode == "Track ID":
                         all_c.extend([float(tid) for _ in range(n_pts)])
                    else:
                        all_c.extend([0.5] * n_pts)

            if not all_x:
                progress.close()
                return

            # --- RENDER ---
            
            # 1. Check if we need to switch renderers (Scenario: Uniform <-> Colored switch)
            # If we are in Uniform mode, we need 'line_main' and hide 'sc_main'
            # If we are in Colored mode, we need 'sc_main' and hide 'line_main'
            
            if is_uniform:
                # Ensure line_main exists
                if not hasattr(self, 'line_main') or self.line_main is None:
                    # Create it: blue dots, small size
                    self.line_main, = ax.plot([], [], [], 'b.', markersize=0.5, alpha=0.5)
                
                # Update Line Data
                self.line_main.set_data(all_x, all_y)
                self.line_main.set_3d_properties(all_z)
                self.line_main.set_visible(True)
                
                # Hide Scatter
                if hasattr(self, 'sc_main') and self.sc_main:
                    self.sc_main.set_visible(False)
                    
            else:
                # Ensure scatter exists
                if not hasattr(self, 'sc_main') or self.sc_main is None:
                    # OPTIMIZATION: depthshade=False improves performance by skipping depth-based alpha calc
                    self.sc_main = ax.scatter([], [], [], s=1, alpha=0.6, linewidth=0, depthshade=False)
                
                # Update Scatter Data
                self.sc_main._offsets3d = (all_x, all_y, all_z)
                c_array = np.array(all_c)
                self.sc_main.set_array(c_array)
                self.sc_main.set_cmap(cmap_name)
                if len(all_c) > 0:
                     # Robust Scaling: Use percentiles to ignore outliers
                     vmin = np.nanpercentile(c_array, 1)
                     vmax = np.nanpercentile(c_array, 99)
                     self.sc_main.set_clim(vmin=vmin, vmax=vmax)
                
                self.sc_main.set_visible(True)
                
                # Hide Line
                if hasattr(self, 'line_main') and self.line_main:
                     self.line_main.set_visible(False)
            
            # --- Colorbar Management ---
            # Remove existing colorbar if any
            if hasattr(self, 'current_cbar') and self.current_cbar:
                self.current_cbar.remove()
                self.current_cbar = None
            
            # Add new colorbar if colored mode
            if not is_uniform and self.sc_main.get_visible():
                self.current_cbar = self.current_fig.colorbar(self.sc_main, ax=ax, fraction=0.03, pad=0.1)
                self.current_cbar.set_label(mode, color='white')
                self.current_cbar.ax.yaxis.set_tick_params(color='white')
                self.current_cbar.outline.set_edgecolor('white')
                plt.setp(plt.getp(self.current_cbar.ax.axes, 'yticklabels'), color='white')

            
            # Update Limits Manually (Critical for correct view)
            if all_x:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                min_z, max_z = min(all_z), max(all_z)
                
                # Add margin
                margin = 0.5
                self.current_xlim = [min_x - margin, max_x + margin]
                self.current_ylim = [min_y - margin, max_y + margin]
                self.current_zlim = [min_z - margin, max_z + margin]
                
                ax.set_xlim3d(self.current_xlim)
                ax.set_ylim3d(self.current_ylim)
                ax.set_zlim3d(self.current_zlim)
            
            # Ensure custom scroll handler is invoked via draw_idle
            self.current_canvas.draw_idle()
            
        except Exception as e:
            print(f"Plot error: {e}")
            if hasattr(self, 'sc_main'): self.sc_main = None
            if hasattr(self, 'line_main'): self.line_main = None
        finally:
            progress.close()
            self._busy_end('plot_tracks')

    def _export_mat(self):
        if not self.loaded_tracks:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save .mat File", "", "MATLAB Files (*.mat)")
        if not path:
            return

        # Selection Dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Data to Export")
        dialog.resize(300, 250)
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("Select columns to include:"))
        
        # Checkboxes
        chk_id = QCheckBox("Track ID")
        chk_frame = QCheckBox("Frame ID") 
        chk_pos = QCheckBox("Position (X, Y, Z, +R3D)")
        chk_vel = QCheckBox("Velocity (Vx, Vy, Vz)")
        chk_acc = QCheckBox("Acceleration (Ax, Ay, Az)")
        chk_2d = QCheckBox("Image 2D Coordinates (+r2D)")
        
        # Defaults: All Checked
        chk_id.setChecked(True)
        chk_frame.setChecked(True)
        chk_pos.setChecked(True)
        chk_vel.setChecked(True)
        chk_acc.setChecked(True)
        chk_2d.setChecked(True)
        
        layout.addWidget(chk_id)
        layout.addWidget(chk_frame)
        layout.addWidget(chk_pos)
        layout.addWidget(chk_vel)
        layout.addWidget(chk_acc)
        layout.addWidget(chk_2d)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.Accepted:
            try:
                self._busy_begin('export_mat', 'Exporting MAT')
                options = {
                    'id': chk_id.isChecked(),
                    'frame': chk_frame.isChecked(),
                    'pos': chk_pos.isChecked(),
                    'vel': chk_vel.isChecked(),
                    'acc': chk_acc.isChecked(),
                    'img2d': chk_2d.isChecked()
                }
                self.processor.save_mat(path, options)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._busy_end('export_mat')
    
    def _replot_check(self):
        """Update check plots with current filter width settings."""
        try:
            fps = self.fps_spin.value()
            w_vel = self.filter_width_spin.value()
            w_acc = self.acc_filter_width_spin.value()
            
            data = self.processor.get_comparison_data(w_vel, w_acc, fps)
            
            if data:
                frames = data['frames']
                
                # --- Plot Acc Check ---
                self.check_fig.clear()
                ax_c = self.check_fig.add_subplot(111)
                ax_c.plot(frames, data['acc_raw'], color='#555', linewidth=0.5, alpha=0.7, label='Raw')
                ax_c.plot(frames, data['acc_filt'], color='red', linewidth=1.5, label=f'Filt (w={w_acc})')
                ax_c.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=6)
                ax_c.set_title("Acc Signal Check", color='white', fontsize=9)
                ax_c.set_xlabel("Frame", color='#aaa', fontsize=7)
                ax_c.set_ylabel("Acc Magnitude (mm/s²)", color='#aaa', fontsize=7)
                ax_c.grid(True, color='#333')
                ax_c.set_facecolor('black')
                ax_c.tick_params(colors='white', labelsize=6)
                self.check_canvas.draw()
                
                # --- Plot Vel Check ---
                self.vel_check_fig.clear()
                ax_vc = self.vel_check_fig.add_subplot(111)
                ax_vc.plot(frames, data['vel_raw'], color='#555', linewidth=0.5, alpha=0.7, label='Raw')
                ax_vc.plot(frames, data['vel_filt'], color='cyan', linewidth=1.5, label=f'Filt (w={w_vel})')
                ax_vc.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=6)
                ax_vc.set_title("Vel Signal Check", color='white', fontsize=9)
                ax_vc.set_xlabel("Frame", color='#aaa', fontsize=7)
                ax_vc.set_ylabel("Vel Magnitude (mm/s)", color='#aaa', fontsize=7)
                ax_vc.grid(True, color='#333')
                ax_vc.set_facecolor('black')
                ax_vc.tick_params(colors='white', labelsize=6)
                self.vel_check_canvas.draw()
                
                # --- Plot Pos Check ---
                self.pos_check_fig.clear()
                ax_p = self.pos_check_fig.add_subplot(111)
                
                # Plot Raw Path (Lighter Grey, Thicker)
                raw_x = data['pos_raw'][:, 0]
                raw_y = data['pos_raw'][:, 1]
                ax_p.plot(raw_x, raw_y, color='#888', linewidth=1.0, alpha=0.8, label='Raw Path')
                
                # Plot Filtered Path (Scatter colored by Speed)
                filt_x = data['pos_filt'][:, 0]
                filt_y = data['pos_filt'][:, 1]
                speed = data['vel_filt']
                
                sc = ax_p.scatter(filt_x, filt_y, c=speed, cmap='plasma', s=3, zorder=10, label='Filt')
                
                ax_p.set_title("Pos Check (X-Y)", color='white', fontsize=9)
                ax_p.set_xlabel("X (mm)", color='#aaa', fontsize=7)
                ax_p.set_ylabel("Y (mm)", color='#aaa', fontsize=7)
                ax_p.grid(True, color='#333')
                ax_p.set_facecolor('black')
                ax_p.tick_params(colors='white', labelsize=6)
                ax_p.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=6)
                ax_p.axis('equal')
                
                self.pos_check_canvas.draw()
                
                # Re-enable interaction for new axes
                self._enable_mouse_interaction(self.pos_check_canvas, ax_p)
                self._enable_mouse_interaction(self.vel_check_canvas, ax_vc)
                self._enable_mouse_interaction(self.check_canvas, ax_c)

        except Exception as e:
            QMessageBox.warning(self, "Replot Failed", str(e))
    
    def _on_export_finished(self, success, msg):
        if success:
            QMessageBox.information(self, "Export", msg)
        else:
            QMessageBox.critical(self, "Export Error", msg)
        self._busy_end('export_mat')
            
