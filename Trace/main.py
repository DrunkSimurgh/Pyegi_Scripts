from PyegiInitializer import *
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 2023
Updated on Sat Mar 21 2026

@author: Drunk Simurgh and ChatGPT

Behavior:
- Draw with mouse on the preview image.
- Normal subtitle lines are animated along the first drawn stroke.
- Shape lines (\p...) are progressively revealed using all drawn strokes.

Notes:
- Left mouse button: draw a stroke
- Release mouse: ends current stroke
- Close the window when finished to generate output
"""


from pyonfx import *
import numpy as np
import sys
import time
import re
from pathlib import Path

import Pyegi

from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsView,
    QGraphicsScene,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QMessageBox,
    QCheckBox,
    QDoubleSpinBox,
)
from PyQt6.QtGui import QPixmap, QPen
from PyQt6.QtCore import Qt


SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE = SCRIPT_DIR / "reference.png"


def apply_dark_theme(app):
    app.setStyleSheet("""
        QWidget {
            background-color: #1e1e1e;
            color: #e6e6e6;
            font-size: 11pt;
        }

        QMainWindow, QGraphicsView {
            background-color: #1e1e1e;
        }

        QLabel {
            color: #f0f0f0;
        }

        QPushButton {
            background-color: #2d2d30;
            border: 1px solid #5a5a62;
            border-radius: 8px;
            padding: 8px 14px;
            color: #f0f0f0;
        }
        QPushButton:hover {
            background-color: #3a3a3f;
        }
        QPushButton:pressed {
            background-color: #252526;
        }

        QRadioButton, QCheckBox {
            spacing: 10px;
            color: #f3f3f3;
        }

        QRadioButton::indicator, QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }

        QRadioButton::indicator:unchecked {
            border: 2px solid #cfcfcf;
            border-radius: 9px;
            background: #2a2a2a;
        }

        QRadioButton::indicator:checked {
            border: 2px solid #7aa2ff;
            border-radius: 9px;
            background: #7aa2ff;
        }

        QCheckBox::indicator:unchecked {
            border: 2px solid #cfcfcf;
            border-radius: 4px;
            background: #2a2a2a;
        }

        QCheckBox::indicator:checked {
            border: 2px solid #7aa2ff;
            border-radius: 4px;
            background: #7aa2ff;
        }
        
        QDoubleSpinBox {
            background-color: #2d2d30;
            border: 1px solid #5a5a62;
            border-radius: 6px;
            padding: 4px 6px;
            color: #f0f0f0;
            min-height: 24px;
        }

        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            background-color: #3a3a3f;
            border-left: 1px solid #5a5a62;
            width: 18px;
        }

        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #4a4a50;
        }
    """)


def is_shape_line(text):
    return re.search(r"\\p\d+", text) is not None


def dedupe_close_samples(samples, min_dist=1.0, keep_last=True):
    """
    samples: [(x, y, t_ms), ...]
    Removes consecutive points that are too close together.
    """
    if not samples:
        return samples

    out = [samples[0]]
    for s in samples[1:]:
        x, y, _ = s
        px, py, _ = out[-1]
        dx = x - px
        dy = y - py
        if (dx * dx + dy * dy) ** 0.5 >= min_dist:
            out.append(s)

    if keep_last and len(samples) > 1 and out[-1] != samples[-1]:
        out.append(samples[-1])

    return out


def smooth_samples(samples, cutoff_frequency=1, sampling_frequency=10):
    """
    Smooth x/y while preserving timestamps.
    """
    if len(samples) < 3:
        return samples

    x = np.array([p[0] for p in samples], dtype=float)
    y = np.array([p[1] for p in samples], dtype=float)
    t = [p[2] for p in samples]

    x2 = apply_sinc_lowpass_filter(x, cutoff_frequency, sampling_frequency)
    y2 = apply_sinc_lowpass_filter(y, cutoff_frequency, sampling_frequency)

    n = min(len(samples), len(x2), len(y2))
    return [(float(x2[i]), float(y2[i]), t[i]) for i in range(n)]


def apply_sinc_lowpass_filter(data, cutoff_frequency=1, sampling_frequency=10):
    data = np.asarray(data, dtype=float)

    if len(data) < 3:
        return data

    filter_length = int(4 * sampling_frequency / cutoff_frequency)
    if filter_length % 2 == 0:
        filter_length += 1

    max_len = max(3, len(data) - 1 if len(data) % 2 == 0 else len(data))
    filter_length = min(filter_length, max_len)
    if filter_length % 2 == 0:
        filter_length -= 1
    if filter_length < 3:
        return data

    sinc_filter = np.sinc(
        2 * cutoff_frequency / sampling_frequency
        * (np.arange(filter_length) - filter_length / 2)
    )
    window = np.blackman(filter_length)
    sinc_filter *= window
    sinc_filter /= np.sum(sinc_filter)

    pad_length = filter_length // 2
    padded_data = np.pad(data, pad_width=pad_length, mode="reflect")
    filtered_data = np.convolve(padded_data, sinc_filter, mode="valid")
    return filtered_data


def remap_times(samples, output_duration_ms=None):
    """
    Returns timestamps normalized from 0.
    If output_duration_ms is provided, scales to that duration.
    """
    if not samples:
        return []

    raw_times = [s[2] for s in samples]
    t0 = raw_times[0]
    times = [t - t0 for t in raw_times]
    total = times[-1] if times else 0

    if output_duration_ms is None:
        return times

    if total <= 0:
        if len(times) <= 1:
            return [0]
        return [
            int(round(i * output_duration_ms / (len(times) - 1)))
            for i in range(len(times))
        ]

    scale = output_duration_ms / total
    return [int(round(t * scale)) for t in times]


class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TraceFX Setup")
        self.setMinimumWidth(420)

        self.mode = "move"
        self.timing = "line"
        self.smoothing = True
        self.accepted = False

        layout = QVBoxLayout(self)

        title = QLabel("TraceFX")
        title.setStyleSheet("font-size: 18pt; font-weight: 600;")
        layout.addWidget(title)

        subtitle = QLabel("Please choose how the drawing should be converted into subtitle lines.")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        layout.addWidget(QLabel("Mode"))
        self.mode_move = QRadioButton("Move mode — text follows the drawn path")
        self.mode_draw = QRadioButton("Draw mode — shape is progressively drawn")
        self.mode_move.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.mode_move)
        self.mode_group.addButton(self.mode_draw)

        layout.addWidget(self.mode_move)
        layout.addWidget(self.mode_draw)

        layout.addSpacing(10)

        layout.addWidget(QLabel("Timing"))
        self.timing_line = QRadioButton("Fit process to the subtitle line duration")
        self.timing_user = QRadioButton("Use the original drawing pace")
        self.timing_line.setChecked(True)

        self.timing_group = QButtonGroup(self)
        self.timing_group.addButton(self.timing_line)
        self.timing_group.addButton(self.timing_user)

        layout.addWidget(self.timing_line)
        layout.addWidget(self.timing_user)

        # Smoothing section
        layout.addSpacing(10)

        self.smoothing_box = QCheckBox("Enable smoothing")
        self.smoothing_box.setChecked(False)
        layout.addWidget(self.smoothing_box)

        self.smoothing_panel = QWidget()
        self.smoothing_panel_layout = QVBoxLayout(self.smoothing_panel)
        self.smoothing_panel_layout.setContentsMargins(18, 6, 0, 0)
        self.smoothing_panel_layout.setSpacing(8)

        # Down-sampling
        self.smoothing_panel_layout.addWidget(QLabel("Down-sampling"))

        self.downsample_none = QRadioButton("Off")
        self.downsample_distance = QRadioButton("Distance-based")
        self.downsample_distance.setChecked(True)

        self.downsample_group = QButtonGroup(self)
        self.downsample_group.addButton(self.downsample_none)
        self.downsample_group.addButton(self.downsample_distance)

        self.smoothing_panel_layout.addWidget(self.downsample_none)
        self.smoothing_panel_layout.addWidget(self.downsample_distance)

        downsample_row = QHBoxLayout()
        downsample_row.addWidget(QLabel("Minimum distance between kept points:"))
        self.downsample_distance_spin = QDoubleSpinBox()
        self.downsample_distance_spin.setRange(0.1, 100.0)
        self.downsample_distance_spin.setDecimals(2)
        self.downsample_distance_spin.setSingleStep(0.25)
        self.downsample_distance_spin.setValue(2.0)
        self.downsample_distance_spin.setSuffix(" px")
        downsample_row.addWidget(self.downsample_distance_spin)
        downsample_row.addStretch()
        self.smoothing_panel_layout.addLayout(downsample_row)

        # Interpolation
        self.interp_box = QCheckBox("Interpolate to even spacing")
        self.interp_box.setChecked(True)
        self.smoothing_panel_layout.addWidget(self.interp_box)

        interp_row = QHBoxLayout()
        interp_row.addWidget(QLabel("Interpolation spacing:"))
        self.interp_step_spin = QDoubleSpinBox()
        self.interp_step_spin.setRange(0.1, 100.0)
        self.interp_step_spin.setDecimals(2)
        self.interp_step_spin.setSingleStep(0.25)
        self.interp_step_spin.setValue(3.0)
        self.interp_step_spin.setSuffix(" px")
        interp_row.addWidget(self.interp_step_spin)
        interp_row.addStretch()
        self.smoothing_panel_layout.addLayout(interp_row)

        # Low-pass
        self.lowpass_box = QCheckBox("Apply low-pass filter")
        self.lowpass_box.setChecked(True)
        self.smoothing_panel_layout.addWidget(self.lowpass_box)

        lowpass_row = QHBoxLayout()
        lowpass_row.addWidget(QLabel("Low-pass strength:"))
        self.lowpass_cutoff_spin = QDoubleSpinBox()
        self.lowpass_cutoff_spin.setRange(0.1, 20.0)
        self.lowpass_cutoff_spin.setDecimals(2)
        self.lowpass_cutoff_spin.setSingleStep(0.1)
        self.lowpass_cutoff_spin.setValue(1.0)
        lowpass_row.addWidget(self.lowpass_cutoff_spin)
        lowpass_row.addStretch()
        self.smoothing_panel_layout.addLayout(lowpass_row)

        layout.addWidget(self.smoothing_panel)
        self.smoothing_panel.setVisible(False)

        self.smoothing_box.toggled.connect(self.on_smoothing_toggled)

        # Optional: disable parameter widgets when their parent option is off
        self.downsample_none.toggled.connect(
            lambda checked: self.downsample_distance_spin.setEnabled(not checked)
        )
        self.interp_box.toggled.connect(self.interp_step_spin.setEnabled)
        self.lowpass_box.toggled.connect(self.lowpass_cutoff_spin.setEnabled)

        # Set initial enabled/disabled state
        self.downsample_distance_spin.setEnabled(self.downsample_distance.isChecked())
        self.interp_step_spin.setEnabled(self.interp_box.isChecked())
        self.lowpass_cutoff_spin.setEnabled(self.lowpass_box.isChecked())

        layout.addSpacing(14)

        if not REFERENCE_IMAGE.exists():
            warn = QLabel(
                f'Image not found:\n{REFERENCE_IMAGE}\n\n'
                'Please place "reference.png" next to this script.'
            )
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #ffb86c;")
            layout.addWidget(warn)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.cancel_button = QPushButton("Cancel")
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.start_button.clicked.connect(self.on_start)
        self.cancel_button.clicked.connect(self.close)

    def on_start(self):
        self.mode = "move" if self.mode_move.isChecked() else "draw"
        self.timing = "line" if self.timing_line.isChecked() else "user"
        self.smoothing = self.smoothing_box.isChecked()

        self.downsample_mode = "none" if self.downsample_none.isChecked() else "distance"
        self.downsample_distance_value = float(self.downsample_distance_spin.value())

        self.interpolate = self.interp_box.isChecked()
        self.interp_step_value = float(self.interp_step_spin.value())

        self.lowpass = self.lowpass_box.isChecked()
        self.lowpass_cutoff_value = float(self.lowpass_cutoff_spin.value())

        if not REFERENCE_IMAGE.exists():
            QMessageBox.critical(
                self,
                "Missing image",
                f'Could not find:\n{REFERENCE_IMAGE}\n\n'
                'Please put "reference.png" in the same folder as this script.'
            )
            return

        self.accepted = True
        self.close()
    
    def on_smoothing_toggled(self, checked):
        self.smoothing_panel.setVisible(checked)

        # Recalculate layout and shrink/grow window to fit contents
        self.layout().activate()
        self.adjustSize()


class DrawView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.scene_ref = scene
        self.all_strokes = []
        self.current_stroke = []
        self.is_drawing = False
        self.base_time = None

    def _now_ms(self):
        if self.base_time is None:
            self.base_time = time.perf_counter()
        return int(round((time.perf_counter() - self.base_time) * 1000.0))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True
            self.current_stroke = []

            point = self.mapToScene(event.pos())
            self.current_stroke.append((point.x(), point.y(), self._now_ms()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            point = self.mapToScene(event.pos())
            sample = (point.x(), point.y(), self._now_ms())

            if self.current_stroke:
                px, py, _ = self.current_stroke[-1]
                self.scene_ref.addLine(
                    px, py, sample[0], sample[1],
                    QPen(Qt.GlobalColor.red, 3)
                )

            self.current_stroke.append(sample)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.current_stroke = dedupe_close_samples(self.current_stroke, min_dist=1.0)
            if self.current_stroke:
                self.all_strokes.append(self.current_stroke)
            self.current_stroke = []
            event.accept()
            return
        super().mouseReleaseEvent(event)


class DrawWindow(QMainWindow):
    def __init__(self, mode_text):
        super().__init__()
        self.setWindowTitle(f"TraceFX - {mode_text}")
        self.setGeometry(100, 100, 1280, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.scene = QGraphicsScene()
        self.view = DrawView(self.scene)
        layout.addWidget(self.view)

        controls = QHBoxLayout()
        self.info_label = QLabel(
            "Left-click and drag to draw. Release to end a stroke. Press Continue when finished."
        )
        self.continue_button = QPushButton("Continue")
        controls.addWidget(self.info_label)
        controls.addStretch()
        controls.addWidget(self.continue_button)
        layout.addLayout(controls)

        image = QPixmap(str(REFERENCE_IMAGE))
        self.pixmap_item = self.scene.addPixmap(image)
        self.view.setSceneRect(self.pixmap_item.boundingRect())
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        self.continue_clicked = False
        self.continue_button.clicked.connect(self.on_continue)

    def on_continue(self):
        self.continue_clicked = True
        self.close()

    @property
    def all_strokes(self):
        strokes = list(self.view.all_strokes)
        if self.view.current_stroke:
            strokes.append(self.view.current_stroke)
        return strokes

def stroke_length(samples):
    if len(samples) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(samples)):
        x1, y1, _ = samples[i - 1]
        x2, y2, _ = samples[i]
        total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return total


def interpolate_stroke(samples, step=3.0):
    """
    Resample a stroke to roughly even spatial spacing.
    Timestamps are linearly interpolated too.
    """
    if len(samples) < 2:
        return samples

    pts = [(float(x), float(y), float(t)) for x, y, t in samples]

    seg_lengths = [0.0]
    for i in range(1, len(pts)):
        x1, y1, _ = pts[i - 1]
        x2, y2, _ = pts[i]
        seg_lengths.append(
            seg_lengths[-1] + ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        )

    total_len = seg_lengths[-1]
    if total_len <= 0:
        return samples

    targets = np.arange(0.0, total_len + step, step)
    if targets[-1] > total_len:
        targets[-1] = total_len

    out = []
    j = 0
    for target in targets:
        while j < len(seg_lengths) - 2 and seg_lengths[j + 1] < target:
            j += 1

        d1 = seg_lengths[j]
        d2 = seg_lengths[j + 1]

        x1, y1, t1 = pts[j]
        x2, y2, t2 = pts[j + 1]

        if d2 <= d1:
            alpha = 0.0
        else:
            alpha = (target - d1) / (d2 - d1)

        x = x1 + alpha * (x2 - x1)
        y = y1 + alpha * (y2 - y1)
        t = t1 + alpha * (t2 - t1)

        out.append((x, y, int(round(t))))

    return out

def preprocess_strokes(
    strokes,
    smoothing=False,
    downsample_mode="distance",
    downsample_distance_value=2.0,
    interpolate=False,
    interp_step_value=3.0,
    lowpass=False,
    lowpass_cutoff_value=1.0,
):
    out = []

    for stroke in strokes:
        s = list(stroke)

        # Always do a tiny dedupe first
        s = dedupe_close_samples(s, min_dist=0.5)

        if smoothing:
            if downsample_mode == "distance":
                s = dedupe_close_samples(s, min_dist=downsample_distance_value)

            if interpolate and len(s) >= 2:
                s = interpolate_stroke(s, step=interp_step_value)

            if lowpass and len(s) >= 3:
                s = smooth_samples(
                    s,
                    cutoff_frequency=lowpass_cutoff_value,
                    sampling_frequency=10,
                )

            s = dedupe_close_samples(s, min_dist=0.5, keep_last=True)

        if len(s) >= 1:
            out.append(s)

    return out


def shape_from_full_stroke(samples):
    if not samples:
        return ""

    x = [p[0] for p in samples]
    y = [p[1] for p in samples]

    shape = "m %.2f %.2f l " % (x[0], y[0])
    for xx, yy in zip(x[1:], y[1:]):
        shape += "%.2f %.2f " % (xx, yy)
    for xx, yy in zip(reversed(x), reversed(y)):
        shape += "%.2f %.2f " % (xx, yy)

    return shape.strip()


def shape_from_partial_stroke(samples, i):
    part = samples[: i + 1]
    if not part:
        return ""

    x = [p[0] for p in part]
    y = [p[1] for p in part]

    shape = "m %.2f %.2f l " % (x[0], y[0])
    for xx, yy in zip(x[1:], y[1:]):
        shape += "%.2f %.2f " % (xx, yy)
    for xx, yy in zip(reversed(x), reversed(y)):
        shape += "%.2f %.2f " % (xx, yy)

    return shape.strip()


def build_accumulated_shape(strokes, stroke_index, point_index):
    """
    Build one shape string that preserves all previous strokes completely,
    and only partially reveals the current stroke.
    """
    parts = []

    for i in range(stroke_index):
        full_shape = shape_from_full_stroke(strokes[i])
        if full_shape:
            parts.append(full_shape)

    current_shape = shape_from_partial_stroke(strokes[stroke_index], point_index)
    if current_shape:
        parts.append(current_shape)

    return " ".join(parts).strip()


def sub_move(line, l, strokes, timing_mode):
    """
    Move text along all drawn strokes in sequence.
    """
    if not strokes:
        return

    flat = []
    for stroke in strokes:
        flat.extend(stroke)

    if len(flat) < 2:
        return

    if timing_mode == "line":
        rel_times = remap_times(flat, max(1, int(round(line.end_time - line.start_time))))
        max_end = line.end_time
    else:
        rel_times = remap_times(flat, None)
        max_end = None

    for i in range(len(flat) - 1):
        x1, y1, _ = flat[i]
        x2, y2, _ = flat[i + 1]

        start_t = line.start_time + rel_times[i]
        end_t = line.start_time + rel_times[i + 1]

        if end_t <= start_t:
            end_t = start_t + 1

        if max_end is not None:
            start_t = min(start_t, max_end)
            end_t = min(end_t, max_end)
            if end_t <= start_t:
                end_t = start_t + 1

        angle = np.rad2deg(np.arctan2(y1 - y2, x2 - x1))

        l.start_time = int(round(start_t))
        l.end_time = int(round(end_t))
        l.text = "{\\an5\\pos(%.2f,%.2f)\\fr%.2f}%s" % (
            x1,
            y1,
            angle,
            line.raw_text,
        )
        Pyegi.send_line(l)


def sub_draw(line, l, strokes, timing_mode):
    if not strokes:
        return

    flat = []
    for s in strokes:
        flat.extend(s)

    if len(flat) < 1:
        return

    if timing_mode == "line":
        total_duration = max(1, int(round(line.end_time - line.start_time)))
        flat_rel = remap_times(flat, total_duration)
        max_end = line.end_time
    else:
        flat_rel = remap_times(flat, None)
        max_end = None

    idx = 0
    for stroke_index, stroke in enumerate(strokes):
        n = len(stroke)
        if n == 0:
            continue

        for point_index in range(n):
            start_t = line.start_time + flat_rel[idx]

            if idx + 1 < len(flat_rel):
                end_t = line.start_time + flat_rel[idx + 1]
            else:
                end_t = line.end_time if timing_mode == "line" else start_t + 1

            if end_t <= start_t:
                end_t = start_t + 1

            if max_end is not None:
                start_t = min(start_t, max_end)
                end_t = min(end_t, max_end)
                if end_t <= start_t:
                    end_t = start_t + 1

            shape = build_accumulated_shape(strokes, stroke_index, point_index)

            l.start_time = int(round(start_t))
            l.end_time = int(round(end_t))
            l.text = "{\\an7\\pos(0,0)\\p1}%s" % shape
            Pyegi.send_line(l)

            idx += 1


if __name__ == "__main__":
    app = QApplication([])
    apply_dark_theme(app)

    settings = SettingsWindow()
    settings.show()
    app.exec()

    if not settings.accepted:
        raise SystemExit

    draw_window = DrawWindow(
        "Move Mode" if settings.mode == "move" else "Draw Mode"
    )
    draw_window.show()
    app.exec()

    if not draw_window.continue_clicked:
        raise SystemExit

    strokes = preprocess_strokes(
        draw_window.all_strokes,
        smoothing=settings.smoothing,
        downsample_mode=settings.downsample_mode,
        downsample_distance_value=settings.downsample_distance_value,
        interpolate=settings.interpolate,
        interp_step_value=settings.interp_step_value,
        lowpass=settings.lowpass,
        lowpass_cutoff_value=settings.lowpass_cutoff_value,
    )
    if not strokes:
        QMessageBox.critical(None, "TraceFX", "No strokes were drawn.")
        raise SystemExit

    io = Ass(Pyegi.get_input_file_path(), extended=True)
    meta, styles, lines = io.get_data()

    for line in lines:
        l = line.copy()

        if settings.mode == "move":
            sub_move(line, l, strokes, settings.timing)
        else:
            if not is_shape_line(line.raw_text):
                # Still allow draw mode even on normal lines by outputting the drawn vector shape.
                sub_draw(line, l, strokes, settings.timing)
            else:
                sub_draw(line, l, strokes, settings.timing)

    Pyegi.create_output_file()
