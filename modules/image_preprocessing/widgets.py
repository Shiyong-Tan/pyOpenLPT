from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QSpinBox, QAbstractSpinBox
)
from PySide6.QtCore import Qt, Signal, QPoint, QEvent
from PySide6.QtGui import QPainter, QColor, QPen, QLinearGradient, QFontMetrics


class RangeSlider(QWidget):
    """
    A visual slider with two draggable handles for min/max range selection.
    Includes QSpinBox for precise input of min and max values.
    """
    
    rangeChanged = Signal(int, int)
    
    def __init__(self, min_val=0, max_val=255, initial_min=0, initial_max=255, parent=None):
        super().__init__(parent)
        
        self._min_range = min_val
        self._max_range = max_val
        self._min_val = initial_min
        self._max_val = initial_max
        
        self._handle_radius = 8
        self._track_height = 6
        self._dragging_min = False
        self._dragging_max = False
        
        self.setMinimumHeight(32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Layout: [Min SpinBox] [Slider Canvas] [Max SpinBox]
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        widest_text = max(
            str(min_val),
            str(max_val),
            str(initial_min),
            str(initial_max),
            key=len,
        )
        text_width = QFontMetrics(self.font()).horizontalAdvance(widest_text)
        spin_w = max(38, int((text_width + 14) * 0.9))

        # Min SpinBox
        self.min_spin = QSpinBox()
        self.min_spin.setRange(min_val, max_val)
        self.min_spin.setValue(initial_min)
        self.min_spin.setFixedWidth(spin_w)
        self.min_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.min_spin.setKeyboardTracking(False)
        self.min_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.min_spin.setStyleSheet("""
            QSpinBox {
                background-color: #121924;
                color: #00d4ff;
                border: 1px solid #2f3e57;
                border-radius: 4px;
                padding: 1px 4px;
                font-size: 10px;
                font-weight: 500;
            }
        """)
        self.min_spin.editingFinished.connect(self._on_min_spin_changed)
        self.min_spin.installEventFilter(self)
        
        # Max SpinBox
        self.max_spin = QSpinBox()
        self.max_spin.setRange(min_val, max_val)
        self.max_spin.setValue(initial_max)
        self.max_spin.setFixedWidth(spin_w)
        self.max_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.max_spin.setKeyboardTracking(False)
        self.max_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.max_spin.setStyleSheet("""
            QSpinBox {
                background-color: #121924;
                color: #00d4ff;
                border: 1px solid #2f3e57;
                border-radius: 4px;
                padding: 1px 4px;
                font-size: 10px;
                font-weight: 500;
            }
        """)
        self.max_spin.editingFinished.connect(self._on_max_spin_changed)
        self.max_spin.installEventFilter(self)
        
        self.slider_canvas = _SliderCanvas(self)
        
        layout.addWidget(self.min_spin)
        layout.addWidget(self.slider_canvas, 1) # Expand slider
        layout.addWidget(self.max_spin)
        
    def _on_min_spin_changed(self):
        val = self.min_spin.value()
        if val > self.max_spin.value():
            self.min_spin.blockSignals(True)
            self.min_spin.setValue(self.max_spin.value())
            self.min_spin.blockSignals(False)
            val = self.max_spin.value()
            
        self.setMinValue(val)
        
    def _on_max_spin_changed(self):
        val = self.max_spin.value()
        if val < self.min_spin.value():
            self.max_spin.blockSignals(True)
            self.max_spin.setValue(self.min_spin.value())
            self.max_spin.blockSignals(False)
            val = self.min_spin.value()
            
        self.setMaxValue(val)

    def eventFilter(self, obj, event):
        if obj in (self.min_spin, self.max_spin) and event.type() == QEvent.Type.FocusIn:
            obj.selectAll()
        return super().eventFilter(obj, event)
    
    def value(self):
        return (self._min_val, self._max_val)
    
    def minValue(self):
        return self._min_val
    
    def maxValue(self):
        return self._max_val
    
    def setMinValue(self, val):
        old_val = self._min_val
        self._min_val = max(self._min_range, min(val, self._max_val))
        
        # Update spinbox if changed internally or clamped
        if self.min_spin.value() != self._min_val:
            self.min_spin.blockSignals(True)
            self.min_spin.setValue(self._min_val)
            self.min_spin.blockSignals(False)
            
        self.slider_canvas.update()
        
        if old_val != self._min_val:
            self.rangeChanged.emit(self._min_val, self._max_val)
    
    def setMaxValue(self, val):
        old_val = self._max_val
        self._max_val = min(self._max_range, max(val, self._min_val))
        
        # Update spinbox if changed internally or clamped
        if self.max_spin.value() != self._max_val:
            self.max_spin.blockSignals(True)
            self.max_spin.setValue(self._max_val)
            self.max_spin.blockSignals(False)
            
        self.slider_canvas.update()
        
        if old_val != self._max_val:
            self.rangeChanged.emit(self._min_val, self._max_val)

    def setRange(self, min_val, max_val):
        min_val = int(min_val)
        max_val = int(max_val)
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        self._min_range = min_val
        self._max_range = max_val
        self.min_spin.setRange(min_val, max_val)
        self.max_spin.setRange(min_val, max_val)

        old_min, old_max = self._min_val, self._max_val
        self._min_val = max(self._min_range, min(self._min_val, self._max_range))
        self._max_val = max(self._min_range, min(self._max_val, self._max_range))
        if self._min_val > self._max_val:
            self._min_val = self._max_val

        if self.min_spin.value() != self._min_val:
            self.min_spin.blockSignals(True)
            self.min_spin.setValue(self._min_val)
            self.min_spin.blockSignals(False)
        if self.max_spin.value() != self._max_val:
            self.max_spin.blockSignals(True)
            self.max_spin.setValue(self._max_val)
            self.max_spin.blockSignals(False)

        self.slider_canvas.update()
        if old_min != self._min_val or old_max != self._max_val:
            self.rangeChanged.emit(self._min_val, self._max_val)


class _SliderCanvas(QWidget):
    """Internal canvas for drawing the slider track and handles."""
    
    def __init__(self, parent_slider):
        super().__init__(parent_slider)
        self._parent = parent_slider
        self._dragging_min = False
        self._dragging_max = False
        self.setMouseTracking(True)
        self.setMinimumWidth(80)
    
    def _val_to_x(self, val):
        margin = self._parent._handle_radius
        width = self.width() - 2 * margin
        if width <= 0: return margin
        denom = self._parent._max_range - self._parent._min_range
        if denom <= 0:
            return margin
        ratio = (val - self._parent._min_range) / denom
        return int(margin + ratio * width)
    
    def _x_to_val(self, x):
        margin = self._parent._handle_radius
        width = self.width() - 2 * margin
        if width <= 0: return self._parent._min_range
        denom = self._parent._max_range - self._parent._min_range
        if denom <= 0:
            return self._parent._min_range
        ratio = (x - margin) / width
        ratio = max(0.0, min(1.0, ratio))
        return int(self._parent._min_range + ratio * denom)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        h = self.height()
        w = self.width()
        r = self._parent._handle_radius
        track_h = self._parent._track_height
        
        min_x = self._val_to_x(self._parent._min_val)
        max_x = self._val_to_x(self._parent._max_val)
        
        track_y = h // 2 - track_h // 2
        
        # Draw track background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(60, 70, 80))
        painter.drawRoundedRect(r, track_y, w - 2*r, track_h, track_h//2, track_h//2)
        
        # Draw selected range
        gradient = QLinearGradient(min_x, 0, max_x, 0)
        gradient.setColorAt(0, QColor(0, 180, 220))
        gradient.setColorAt(1, QColor(0, 220, 255))
        painter.setBrush(gradient)
        if max_x > min_x:
            painter.drawRoundedRect(min_x, track_y, max_x - min_x, track_h, track_h//2, track_h//2)
        
        # Draw handles
        painter.setBrush(QColor(0, 212, 255))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(QPoint(min_x, h // 2), r, r)
        painter.drawEllipse(QPoint(max_x, h // 2), r, r)
        
        painter.end()
    
    def mousePressEvent(self, event):
        x = event.pos().x()
        min_x = self._val_to_x(self._parent._min_val)
        max_x = self._val_to_x(self._parent._max_val)
        r = self._parent._handle_radius
        
        if abs(x - min_x) <= r + 5:
            self._dragging_min = True
        elif abs(x - max_x) <= r + 5:
            self._dragging_max = True
        elif min_x < x < max_x:
            # Click middle: move closest
            if x - min_x < max_x - x:
                self._dragging_min = True
                self._update_val_from_mouse(x, is_min=True)
            else:
                self._dragging_max = True
                self._update_val_from_mouse(x, is_min=False)
        else:
            # Click outside: move closest
            if x < min_x:
                self._dragging_min = True
                self._update_val_from_mouse(x, is_min=True)
            elif x > max_x:
                self._dragging_max = True
                self._update_val_from_mouse(x, is_min=False)
    
    def mouseMoveEvent(self, event):
        x = event.pos().x()
        if self._dragging_min:
            self._update_val_from_mouse(x, is_min=True)
        elif self._dragging_max:
            self._update_val_from_mouse(x, is_min=False)
    
    def mouseReleaseEvent(self, event):
        self._dragging_min = False
        self._dragging_max = False
        
    def _update_val_from_mouse(self, x, is_min):
        val = self._x_to_val(x)
        if is_min:
            self._parent.setMinValue(val)
        else:
            self._parent.setMaxValue(val)

from PySide6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QProgressBar, QLabel
from PySide6.QtCore import Qt, Signal

class ProcessingDialog(QDialog):
    """Dialog to show progress and allow Pause/Resume/Stop."""
    
    pause_signal = Signal(bool) # True = Pause, False = Resume
    stop_signal = Signal()
    
    def __init__(self, parent=None, title="Processing..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(400, 150)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint) # Disable close button
        
        layout = QVBoxLayout(self)
        
        self.status_label = QLabel("Initializing...")
        # self.status_label.setStyleSheet("color: white; font-size: 14px;") # Inherits from parent theme usually
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                background-color: #222;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #00d4ff;
                width: 10px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.setStyleSheet("background-color: #e6b800; color: black; font-weight: bold; padding: 5px;")
        self.btn_pause.clicked.connect(self._toggle_pause)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: #d93025; color: white; font-weight: bold; padding: 5px;")
        self.btn_stop.clicked.connect(self._stop_processing)
        
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_stop)
        
        layout.addLayout(btn_layout)
        
        self._is_paused = False
        
    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Processing Frame: {current}/{total}")
        
    def _toggle_pause(self):
        self._is_paused = not self._is_paused
        self.btn_pause.setText("Resume" if self._is_paused else "Pause")
        self.status_label.setText("Paused" if self._is_paused else self.status_label.text())
        self.pause_signal.emit(self._is_paused)
        
    def _stop_processing(self):
        self.status_label.setText("Stopping...")
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.stop_signal.emit()
