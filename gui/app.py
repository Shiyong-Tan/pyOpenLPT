"""
OpenLPT Main Application Window
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow importing 'modules'
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QLabel, QFrame, QStatusBar, QToolBar, 
    QPushButton, QProgressBar, QSizePolicy, QToolButton, QButtonGroup
)
from PySide6.QtCore import Qt, QSize, Slot
from PySide6.QtGui import QIcon, QFont, QAction, QColor, QPainter
import qtawesome as qta

# Import views
# from views.camera_calibration_view import CameraCalibrationView # Replaced by modules
from views.image_preprocessing_view import ImagePreprocessingView
from views.tracking_settings_view import TrackingSettingsView
from views.tracking_view import TrackingView
from views.results_view import ResultsView
from modules.camera_calibration import CameraCalibrationView


class OpenLPTMainWindow(QMainWindow):
    """Main application window for OpenLPT GUI."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenLPT - 3D particle tracking system")
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "assets" / "icon.png")))
        self.setGeometry(100, 100, 1500, 900)
        self.setMinimumSize(1200, 700)
        
        # Load stylesheet
        self._load_stylesheet()
        
        # Setup UI
        # Setup UI
        self._setup_ui()
        # self._setup_toolbar() # Removed as requested
        self._setup_statusbar()
        self._busy_tokens = set()
        
        # Apply Windows Dark Title Bar
        self._apply_dark_title_bar()
        
        # Check for updates (async)
        self._check_for_updates()

    def _apply_dark_title_bar(self):
        """Force Windows to use dark title bar."""
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            try:
                hwnd = int(self.winId())
                
                # 1. Enable Dark Mode (Windows 10 1903+)
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                value = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, 
                    DWMWA_USE_IMMERSIVE_DARK_MODE, 
                    ctypes.byref(value), 
                    ctypes.sizeof(value)
                )

                # 2. Set Title Bar Color to Black (Windows 11+)
                # DWMWA_CAPTION_COLOR = 35
                # COLORREF is 0x00BBGGRR. Black is 0x00000000.
                DWMWA_CAPTION_COLOR = 35
                black_color = 0x00000000
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_CAPTION_COLOR,
                    ctypes.byref(ctypes.c_int(black_color)),
                    ctypes.sizeof(ctypes.c_int)
                )

                # 3. Set Title Text Color to White (Windows 11+)
                # DWMWA_TEXT_COLOR = 36
                # White is 0x00FFFFFF
                DWMWA_TEXT_COLOR = 36
                white_color = 0x00FFFFFF
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_TEXT_COLOR,
                    ctypes.byref(ctypes.c_int(white_color)),
                    ctypes.sizeof(ctypes.c_int)
                )
            except Exception as e:
                # Fail silently if DWM API is not supported (e.g. older Windows)
                print(f"DWM Customization Warning: {e}")
        
    def _load_stylesheet(self):
        """Load the QSS stylesheet."""
        style_path = Path(__file__).parent / "style.qss"
        if style_path.exists():
            with open(style_path, 'r', encoding='utf-8') as f:
                self.setStyleSheet(f.read())
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # === Left Navigation Sidebar ===
        nav_widget = QWidget()
        nav_widget.setFixedWidth(150)
        nav_widget.setStyleSheet("background-color: #000000;")
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        
        # Logo/Title
        logo_label = QLabel("OpenLPT")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("""
            QLabel {
                font-family: "Segoe UI", sans-serif;
                font-size: 28px;
                font-weight: 800;
                color: #00d4ff;
                padding: 24px 0px;
                background-color: transparent;
            }
        """)
        nav_layout.addWidget(logo_label)
        
        # Navigation Container
        nav_container = QWidget()
        nav_container.setObjectName("navContainer")
        nav_buttons_layout = QVBoxLayout(nav_container)
        nav_buttons_layout.setContentsMargins(10, 10, 10, 10)
        nav_buttons_layout.setSpacing(12)
        
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_group.idClicked.connect(self._on_nav_clicked)
        
        # Add navigation items
        # Format: (icon_name, label_text, index)
        nav_items = [
            ("fa5s.camera", "Calibration", 0),
            ("fa5s.layer-group", "Preprocessing", 1),
            ("fa5s.sliders-h", "Settings", 2),
            ("fa5s.crosshairs", "Tracking", 3),
            ("fa5s.chart-line", "Results", 4),
            ("fa5s.book", "Tutorial", 5),
        ]
        
        for icon_name, text, idx in nav_items:
            # Create modern icon using qtawesome
            icon = qta.icon(icon_name, color="#8a9aae", color_active="#00d4ff", color_on="#00d4ff")
            btn = self._create_nav_button(icon, text)
            self.nav_group.addButton(btn, idx)
            nav_buttons_layout.addWidget(btn)
        
        # Select first item default
        self.nav_group.button(0).setChecked(True)
        
        nav_buttons_layout.addStretch()
        nav_layout.addWidget(nav_container)
        
        main_layout.addWidget(nav_widget)
        
        # === Content Area ===
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(20)
        
        # Stacked widget for different views
        self.stack = QStackedWidget()
        
        # Initialize views dict
        self.views = {}
        
        # Initialize views with dependencies
        self.calib_view = CameraCalibrationView()
        self.views["preprocessing"] = ImagePreprocessingView()
        self.views["settings"] = TrackingSettingsView(
            calibration_view=self.calib_view,
            preprocessing_view=self.views["preprocessing"]
        )
        self.views["tracking"] = TrackingView(settings_view=self.views["settings"])
        self.views["results"] = ResultsView(settings_view=self.views["settings"])
        
        # Add to stack
        self.stack.addWidget(self.calib_view) # 0
        self.stack.addWidget(self.views["preprocessing"]) # 1
        self.stack.addWidget(self.views["settings"])      # 2
        self.stack.addWidget(self.views["tracking"])      # 3
        self.stack.addWidget(self.views["results"])       # 4
        
        content_layout.addWidget(self.stack)
        main_layout.addWidget(content_widget, stretch=1)
    
    def _create_nav_button(self, icon: QIcon, label_text: str) -> QToolButton:
        """Create a styled navigation button."""
        btn = QToolButton()
        btn.setText(label_text)
        btn.setIcon(icon)
        btn.setIconSize(QSize(32, 32))  # Large icons
        btn.setCheckable(True)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn.setFixedHeight(85)  # Square-ish vertical button
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    def _on_nav_clicked(self, index: int):
        """Handle navigation selection change."""
        if index == 5: # Tutorial
            # Open tutorial URL and keep previous selection
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            
            guide_path = Path(__file__).parent / "docs" / "index.html"
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(guide_path)))
            
            # Revert selection to previous valid index or keep current if valid
            current = self.stack.currentIndex()
            if self.nav_group.button(current):
                self.nav_group.button(current).setChecked(True)
            return

        self.stack.setCurrentIndex(index)
        
        # Auto-fill Project Path in Results View from Tracking View
        if index == 4: # Results View
            try:
                tracking_view = self.views.get("tracking")
                results_view = self.views.get("results")
                
                if tracking_view and results_view:
                    # Get path from Tracking View
                    current_path = tracking_view.proj_path_edit.text().strip()
                    
                    if current_path:
                        # Only update if Results View is empty or different? 
                        # Better to update always to ensure sync, or check if empty.
                        # User might have changed it manually in Results, but likely wants the active project.
                        # Let's update only if Results is empty or user requests it? 
                        # User request: "Auto read from tracking module".
                        # So just overwrite or fill if empty.
                        # Safer: Fill if empty OR if it matches the 'default' empty state.
                        # Or just overwrite. User can change it back if needed.
                        results_view.proj_path_edit.setText(current_path)
            except Exception as e:
                print(f"Error syncing project path: {e}")
    
    def _setup_toolbar(self):
        """Setup the top toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Add toolbar actions
        toolbar.addWidget(self._create_toolbar_button("📂", "Open Project"))
        toolbar.addWidget(self._create_toolbar_button("💾", "Save"))
        toolbar.addSeparator()
        toolbar.addWidget(self._create_toolbar_button("▶️", "Run"))
        toolbar.addWidget(self._create_toolbar_button("⏸️", "Pause"))
        toolbar.addWidget(self._create_toolbar_button("⏹️", "Stop"))
        toolbar.addSeparator()
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        toolbar.addWidget(self._create_toolbar_button("⚙️", "Settings"))
        toolbar.addWidget(self._create_toolbar_button("❓", "Help"))
    
    def _create_toolbar_button(self, emoji: str, tooltip: str) -> QPushButton:
        """Create a toolbar button with emoji icon."""
        btn = QPushButton(emoji)
        btn.setToolTip(tooltip)
        btn.setFixedSize(40, 40)
        btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 6px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #0f3460;
            }
            QPushButton:pressed {
                background-color: #0077b6;
            }
        """)
        return btn
    
    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #00ff88;")
        self.statusbar.addWidget(self.status_label)
        
        

        # Credit label
        credit_label = QLabel("Designed by Shiyong Tan @ Ni Research Lab")
        credit_label.setStyleSheet("color: #FFFFFF; padding-left: 20px; font-size: 11px;")
        self.statusbar.addPermanentWidget(credit_label)
    
    
    def _on_nav_changed(self, index: int):
        """Removed: Using QButtonGroup click handler instead."""
        pass
    
    def update_status(self, message: str, is_success: bool = True):
        """Update the status bar message."""
        color = "#00ff88" if is_success else "#e63946"
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(message)

    def begin_busy(self, task_name: str = ""):
        """Mark app status as busy and return a token."""
        token = object()
        self._busy_tokens.add(token)
        self.status_label.setStyleSheet("color: #ff9800;")
        self.status_label.setText("Busy")
        return token

    def end_busy(self, token):
        """Release busy token and restore ready state when idle."""
        if token in self._busy_tokens:
            self._busy_tokens.remove(token)
        if not self._busy_tokens:
            self.status_label.setStyleSheet("color: #00ff88;")
            self.status_label.setText("Ready")
    
    
    def _check_for_updates(self):
        """Check for updates asynchronously on startup."""
        try:
            from utils.update_checker import check_for_updates_async
            print("[App] Starting update check...")
            check_for_updates_async(self._on_update_check_complete)
        except Exception as e:
            print(f"Update check failed: {e}")
    
    def _on_update_check_complete(self, result: dict):
        """Handle update check result (called from background thread)."""
        print(f"[App] Update check complete. Available: {result.get('available')}")
        if result.get("available"):
            # Store result and use thread-safe method to show dialog
            self._update_result = result
            # Use QMetaObject.invokeMethod for thread-safe call
            from PySide6.QtCore import QMetaObject, Qt, Q_ARG
            QMetaObject.invokeMethod(self, "_show_update_dialog_slot", Qt.QueuedConnection)
    
    @Slot()
    def _show_update_dialog_slot(self):
        """Slot to show update dialog (called from main thread)."""
        print("[App] _show_update_dialog_slot called")
        if hasattr(self, '_update_result'):
            self._show_update_dialog(self._update_result)
    
    def _show_update_dialog(self, result: dict):
        """Show update available dialog."""
        print("[App] Showing update dialog...")
        try:
            from PySide6.QtWidgets import QMessageBox
            
            current = result.get("current", "?")
            latest = result.get("latest", "?")
            url = result.get("url", "")
            notes = result.get("notes", "")
            
            msg = QMessageBox(self)
            msg.setAttribute(Qt.WA_StyledBackground, True) # Force QSS on Mac
            msg.setWindowTitle("Update Available")
            msg.setIcon(QMessageBox.Information)
            msg.setText(f"<h3>A new version of OpenLPT is available!</h3>")
            msg.setInformativeText(
                f"<p>Current version: <b>{current}</b><br>"
                f"Latest version: <b>{latest}</b></p>"
                f"<p>Click <b>Update Now</b> to automatically update and restart.</p>"
                f"<p>Or visit the <a href='{url}'>releases page</a> for manual instructions.</p>"
            )
            
            if notes:
                msg.setDetailedText(f"Release Notes:\n\n{notes}")
            
            # Add custom buttons
            update_btn = msg.addButton("Update Now", QMessageBox.AcceptRole)
            cancel_btn = msg.addButton("Later", QMessageBox.RejectRole)
            
            msg.setDefaultButton(update_btn)
            msg.exec()
            
            if msg.clickedButton() == update_btn:
                print("[App] User clicked Update Now")
                from utils.auto_updater import run_auto_update
                # Project root assumed to be parent of 'gui' folder
                project_root = Path(__file__).resolve().parent.parent
                run_auto_update(project_root)
            
            print("[App] Update dialog closed.")
        except Exception as e:
            print(f"[App] Error showing dialog: {e}")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for better cross-platform look
    
    # Set app icon (Critical for Mac Dock)
    icon_path = Path(__file__).parent / "assets" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = OpenLPTMainWindow()
    window.showMaximized()
    
    # Check for shortcut creation (after window show so we have context, though MessageBox works independently)
    try:
        from create_shortcut import check_and_create_shortcut
        from PySide6.QtWidgets import QMessageBox
        
        shortcut_status = check_and_create_shortcut()
        
        if shortcut_status == 1:
            QMessageBox.information(
                window,
                "Shortcut Created",
                "A shortcut for OpenLPT has been created on your desktop."
            )
        elif shortcut_status == -2:
             QMessageBox.warning(
                window,
                "Shortcut Creation Failed",
                "Failed to create desktop shortcut because your desktop path contains non-English characters.\n\n"
                "Please check the README or create the shortcut manually."
            )
    except Exception as e:
        print(f"Failed to check shortcut: {e}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
