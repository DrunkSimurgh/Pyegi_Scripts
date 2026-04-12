from PyegiInitializer import *
# -*- coding: utf-8 -*-
"""
TraceFX
PyQt6 + Pyegi + PyonFX

Modes:
- Move: move subtitle text along traced path
- Draw: progressively draw the user's traced shape
- Trace Text: reveal the original subtitle text using a traced clip brush

Preview background:
- If PREVIEW_SOURCE is a valid video, use PREVIEW_FRAME from that video
- Otherwise use FALLBACK_IMAGE_NAME resized to the ASS script resolution

Put the fallback image in the same folder as this script.
"""


from pyonfx import *
import numpy as np
import sys
import time
import re
from pathlib import Path
from math import cos, sin, pi


# Optional dependency for preview video frame extraction
try:
    import cv2
except Exception:
    cv2 = None

# file = Path(__file__).resolve()
# sys.path.append(str(file.parents[2] / "Pyegi"))
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
    QFormLayout,
)
from PyQt6.QtGui import (
    QPixmap,
    QPen,
    QImage,
    QColor,
    QPainterPath,
    QBrush,
)
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsPathItem


DEBUG_LOG = Path(__file__).resolve().with_name("tracefx_debug.log")

def debug_write(msg):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")
    except Exception:
        pass

# =========================================================
# User-editable config at top of script
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# If this is a valid video path, PREVIEW_FRAME will be used.
# Otherwise the fallback image below will be used.
# PREVIEW_SOURCE = r"C:\path\to\preview.mp4"
# PREVIEW_FRAME = 120

project_properties_table = Pyegi.get_project_properties()
PROJECT_VIDEO_PATH = project_properties_table.get("video_file")
# FALLBACK_VIDEO_PATH = PROJECT_VIDEO_PATH
PREVIEW_SOURCE = PROJECT_VIDEO_PATH

# EXTERNAL_VIDEO_DEFAULT = ""
# USE_PYEGI_INPUT_PATH = True

raw_video_position = project_properties_table.get("video_position")
PREVIEW_FRAME = raw_video_position

FALLBACK_IMAGE_NAME = "reference.png"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

# Brush radius for Trace Text mode reveal
TRACE_TEXT_BRUSH_RADIUS = 8.0

TRACE_TEXT_RADIUS_COMPENSATION = 0.15

TRACE_TEXT_OUTPUT_RADIUS_MULTIPLIER = 1.45

# =========================================================
# Theme
# =========================================================

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


# =========================================================
# Utility
# =========================================================

def is_shape_line(text):
    return re.search(r"\\p\d+", text) is not None


def dedupe_close_samples(samples, min_dist=1.0, keep_last=True):
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


def smooth_samples(samples, cutoff_frequency=1, sampling_frequency=10):
    if len(samples) < 3:
        return samples

    x = np.array([p[0] for p in samples], dtype=float)
    y = np.array([p[1] for p in samples], dtype=float)
    t = [p[2] for p in samples]

    x2 = apply_sinc_lowpass_filter(x, cutoff_frequency, sampling_frequency)
    y2 = apply_sinc_lowpass_filter(y, cutoff_frequency, sampling_frequency)

    n = min(len(samples), len(x2), len(y2))
    return [(float(x2[i]), float(y2[i]), t[i]) for i in range(n)]


def interpolate_stroke(samples, step=3.0):
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
    if len(targets) == 0:
        return samples
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

        alpha = 0.0 if d2 <= d1 else (target - d1) / (d2 - d1)

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


def remap_times(samples, output_duration_ms=None):
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


def qimage_from_numpy_rgb(rgb_array):
    h, w, ch = rgb_array.shape
    bytes_per_line = ch * w
    return QImage(
        rgb_array.data,
        w,
        h,
        bytes_per_line,
        QImage.Format.Format_RGB888
    ).copy()


def resize_rgb_image(rgb_array, width, height):
    if cv2 is not None:
        resized = cv2.resize(rgb_array, (width, height), interpolation=cv2.INTER_AREA)
        return resized

    # Fallback via Qt if cv2 is unavailable
    qimg = qimage_from_numpy_rgb(rgb_array)
    qimg = qimg.scaled(
        width, height,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )
    ptr = qimg.bits()
    ptr.setsize(qimg.sizeInBytes())
    arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)
    return arr[:, :, :3].copy()


def prepare_preview_background(playresx, playresy):
    fallback_path = SCRIPT_DIR / FALLBACK_IMAGE_NAME
    source_path = Path(PREVIEW_SOURCE)

    # Case 1: valid video
    if (
        source_path.exists()
        and source_path.suffix.lower() in VIDEO_EXTENSIONS
        and cv2 is not None
    ):
        cap = cv2.VideoCapture(str(source_path))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, PREVIEW_FRAME)
            ok, frame_bgr = cap.read()
            cap.release()
            if ok and frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = resize_rgb_image(frame_rgb, playresx, playresy)
                return QPixmap.fromImage(qimage_from_numpy_rgb(frame_rgb))

    # Case 2: fallback image resized to script resolution
    if fallback_path.exists():
        pix = QPixmap(str(fallback_path))
        return pix.scaled(
            playresx, playresy,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

    # Final fallback: blank pixmap
    blank = QPixmap(playresx, playresy)
    blank.fill(QColor("#202020"))
    return blank


def parse_override_alignment(raw_text):
    matches = re.findall(r"\\an(\d+)", raw_text or "")
    if matches:
        try:
            v = int(matches[-1])
            if 1 <= v <= 9:
                return v
        except Exception:
            pass
    return None


def parse_override_pos(raw_text):
    m = re.search(r"\\pos\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", raw_text or "")
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None


def parse_override_alignment(raw_text):
    matches = re.findall(r"\\an(\d+)", raw_text or "")
    if matches:
        try:
            v = int(matches[-1])
            if 1 <= v <= 9:
                return v
        except Exception:
            pass
    return None


def parse_override_pos(raw_text):
    m = re.search(r"\\pos\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", raw_text or "")
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None


def get_line_alignment(line):
    an = parse_override_alignment(getattr(line, "raw_text", ""))
    if an is not None:
        return an

    style = getattr(line, "style", None)
    if style is not None:
        try:
            v = int(getattr(style, "alignment", 7))
            if 1 <= v <= 9:
                return v
        except Exception:
            pass

    return 7


def _safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def get_line_margins(line):
    """
    Prefer line margins if present and nonzero.
    Otherwise fall back to style margins.
    """
    style = getattr(line, "style", None)

    style_l = _safe_int(getattr(style, "margin_l", getattr(style, "margin_lft", 0)), 0) if style else 0
    style_r = _safe_int(getattr(style, "margin_r", getattr(style, "margin_rgt", 0)), 0) if style else 0
    style_v = _safe_int(getattr(style, "margin_v", 0), 0) if style else 0

    line_l = _safe_int(getattr(line, "margin_l", 0), 0)
    line_r = _safe_int(getattr(line, "margin_r", 0), 0)
    line_v = _safe_int(getattr(line, "margin_v", 0), 0)

    ml = line_l if line_l != 0 else style_l
    mr = line_r if line_r != 0 else style_r
    mv = line_v if line_v != 0 else style_v

    return ml, mr, mv


def default_anchor_from_alignment(alignment, playresx, playresy, margin_l, margin_r, margin_v):
    """
    ASS-like default anchor position for non-\\pos lines, including margins.
    """
    # horizontal
    col = (alignment - 1) % 3
    if col == 0:          # left
        ax = float(margin_l)
    elif col == 1:        # center
        ax = playresx / 2.0
    else:                 # right
        ax = float(playresx - margin_r)

    # vertical
    row = (alignment - 1) // 3
    if row == 0:          # bottom row: 1,2,3
        ay = float(playresy - margin_v)
    elif row == 1:        # middle row: 4,5,6
        ay = playresy / 2.0
    else:                 # top row: 7,8,9
        ay = float(margin_v)

    return ax, ay


def anchor_offset_from_alignment(alignment, width, height):
    """
    Offset from top-left of the text block to the alignment anchor.
    """
    col = (alignment - 1) % 3
    row = (alignment - 1) // 3

    if col == 0:
        ox = 0.0
    elif col == 1:
        ox = width / 2.0
    else:
        ox = width

    if row == 0:
        oy = height
    elif row == 1:
        oy = height / 2.0
    else:
        oy = 0.0

    return ox, oy


def translate_guide_path_to_line_position(path, line, playresx, playresy):
    if path is None or path.isEmpty():
        return path

    alignment = get_line_alignment(line)
    pos = parse_override_pos(getattr(line, "raw_text", ""))

    if pos is not None:
        anchor_x, anchor_y = pos
    else:
        margin_l, margin_r, margin_v = get_line_margins(line)
        anchor_x, anchor_y = default_anchor_from_alignment(
            alignment,
            playresx,
            playresy,
            margin_l,
            margin_r,
            margin_v,
        )

    bounds = path.boundingRect()
    shape_w = bounds.width()
    shape_h = bounds.height()

    anchor_ox, anchor_oy = anchor_offset_from_alignment(alignment, shape_w, shape_h)

    target_top_left_x = anchor_x - anchor_ox
    target_top_left_y = anchor_y - anchor_oy

    dx = target_top_left_x - bounds.x()
    dy = target_top_left_y - bounds.y()

    translated = QPainterPath(path)
    translated.translate(dx, dy)
    return translated

def qt_keep_aspect_ratio():
    if hasattr(Qt, "AspectRatioMode"):
        return Qt.AspectRatioMode.KeepAspectRatio
    return Qt.KeepAspectRatio


def qt_ignore_aspect_ratio():
    if hasattr(Qt, "AspectRatioMode"):
        return Qt.AspectRatioMode.IgnoreAspectRatio
    return Qt.IgnoreAspectRatio


def qt_smooth_transformation():
    if hasattr(Qt, "TransformationMode"):
        return Qt.TransformationMode.SmoothTransformation
    return Qt.SmoothTransformation


def qt_red():
    if hasattr(Qt, "GlobalColor"):
        return Qt.GlobalColor.red
    return Qt.red


def qt_left_button():
    if hasattr(Qt, "MouseButton"):
        return Qt.MouseButton.LeftButton
    return Qt.LeftButton

# =========================================================
# ASS drawing helpers
# =========================================================

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
    parts = []

    for i in range(stroke_index):
        full_shape = shape_from_full_stroke(strokes[i])
        if full_shape:
            parts.append(full_shape)

    current_shape = shape_from_partial_stroke(strokes[stroke_index], point_index)
    if current_shape:
        parts.append(current_shape)

    return " ".join(parts).strip()


def ass_circle_path(cx, cy, r, steps=18):
    pts = []
    for i in range(steps):
        a = 2 * pi * i / steps
        x = cx + r * cos(a)
        y = cy + r * sin(a)
        pts.append((x, y))

    if not pts:
        return ""

    s = "m %.2f %.2f l " % (pts[0][0], pts[0][1])
    for x, y in pts[1:]:
        s += "%.2f %.2f " % (x, y)
    s += "%.2f %.2f " % (pts[0][0], pts[0][1])
    return s.strip()


def build_trace_text_clip(strokes, stroke_index, point_index, radius, steps):
    parts = []

    for i in range(stroke_index + 1):
        stroke = strokes[i]
        if i == stroke_index:
            samples = stroke[: point_index + 1]
        else:
            samples = stroke

        if len(samples) == 0:
            continue

        # Densify the stroke so the clip behaves more like a continuous painted line
        dense_step = max(0.35, radius * 0.20)
        dense_samples = interpolate_stroke(samples, step=dense_step) if len(samples) >= 2 else samples

        for x, y, _ in dense_samples:
            parts.append(ass_circle_path(x, y, radius, steps))

    return " ".join(parts).strip()


# =========================================================
# Best-effort PyonFX text-shape guide preview
# =========================================================

def get_text_shape_ass(line):
    """
    Best-effort wrapper around PyonFX text-to-shape conversion.
    """
    try:
        shape = Convert.text_to_shape(line)
        if isinstance(shape, str):
            return shape
        return str(shape)
    except Exception:
        return None


def ass_draw_to_qpainterpath(draw_str):
    """
    Very small ASS drawing parser for preview.
    Supports m, l, b commands.
    Good enough for many text shapes.
    """
    if not draw_str:
        return None

    tokens = draw_str.strip().split()
    if not tokens:
        return None

    path = QPainterPath()
    i = 0
    cmd = None

    def is_cmd(tok):
        return tok in {"m", "n", "l", "b", "s", "p", "c"}

    while i < len(tokens):
        tok = tokens[i]
        if is_cmd(tok):
            cmd = tok
            i += 1
            continue

        if cmd in {"m", "n", "l"}:
            if i + 1 >= len(tokens):
                break
            try:
                x = float(tokens[i])
                y = float(tokens[i + 1])
            except Exception:
                break

            if cmd in {"m", "n"}:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
            i += 2
            continue

        if cmd == "b":
            if i + 5 >= len(tokens):
                break
            try:
                x1, y1 = float(tokens[i]), float(tokens[i + 1])
                x2, y2 = float(tokens[i + 2]), float(tokens[i + 3])
                x3, y3 = float(tokens[i + 4]), float(tokens[i + 5])
            except Exception:
                break
            path.cubicTo(x1, y1, x2, y2, x3, y3)
            i += 6
            continue

        i += 1

    return path


# =========================================================
# Setup window
# =========================================================

class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TraceFX Setup")
        self.setMinimumWidth(460)

        self.mode = "move"
        self.timing = "line"
        self.smoothing = False
        self.accepted = False

        layout = QVBoxLayout(self)

        title = QLabel("TraceFX")
        title.setStyleSheet("font-size: 18pt; font-weight: 600;")
        layout.addWidget(title)

        subtitle = QLabel("Choose mode, timing, and optional smoothing.")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        layout.addWidget(QLabel("Mode"))
        self.mode_move = QRadioButton("Move mode — text follows the traced path")
        self.mode_draw = QRadioButton("Draw mode — draw the traced shape")
        self.mode_trace_text = QRadioButton("Trace Text mode — reveal the original subtitle text from the trace")
        self.mode_move.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.mode_move)
        self.mode_group.addButton(self.mode_draw)
        self.mode_group.addButton(self.mode_trace_text)

        layout.addWidget(self.mode_move)
        layout.addWidget(self.mode_draw)
        layout.addWidget(self.mode_trace_text)

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

        layout.addSpacing(10)

        self.smoothing_box = QCheckBox("Enable smoothing")
        self.smoothing_box.setChecked(False)
        layout.addWidget(self.smoothing_box)

        self.smoothing_panel = QWidget()
        self.smoothing_panel_layout = QVBoxLayout(self.smoothing_panel)
        self.smoothing_panel_layout.setContentsMargins(18, 6, 0, 0)
        self.smoothing_panel_layout.setSpacing(8)

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

        self.lowpass_box = QCheckBox("Apply low-pass filter")
        self.lowpass_box.setChecked(True)
        self.smoothing_panel_layout.addWidget(self.lowpass_box)

        lowpass_row = QHBoxLayout()
        lowpass_row.addWidget(QLabel("Low-pass cutoff:"))
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

        self.downsample_none.toggled.connect(
            lambda checked: self.downsample_distance_spin.setEnabled(not checked)
        )
        self.interp_box.toggled.connect(self.interp_step_spin.setEnabled)
        self.lowpass_box.toggled.connect(self.lowpass_cutoff_spin.setEnabled)

        self.downsample_distance_spin.setEnabled(self.downsample_distance.isChecked())
        self.interp_step_spin.setEnabled(self.interp_box.isChecked())
        self.lowpass_cutoff_spin.setEnabled(self.lowpass_box.isChecked())

        layout.addSpacing(14)

        # preview_info = QLabel(
        #     f"Preview source:\n{PREVIEW_SOURCE}\n"
        #     f"Preview frame: {PREVIEW_FRAME}\n"
        #     f'Fallback image: {SCRIPT_DIR / FALLBACK_IMAGE_NAME}'
        # )
        # preview_info.setWordWrap(True)
        # preview_info.setStyleSheet("color: #c8c8c8;")
        # layout.addWidget(preview_info)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.cancel_button = QPushButton("Cancel")
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.start_button.clicked.connect(self.on_start)
        self.cancel_button.clicked.connect(self.close)

    def on_smoothing_toggled(self, checked):
        self.smoothing_panel.setVisible(checked)
        self.layout().activate()
        self.adjustSize()
        self.resize(max(self.width(), 460), self.height())

    def on_start(self):
        self.mode = (
            "move" if self.mode_move.isChecked()
            else "draw" if self.mode_draw.isChecked()
            else "trace_text"
        )
        self.timing = "line" if self.timing_line.isChecked() else "user"

        self.smoothing = self.smoothing_box.isChecked()
        self.downsample_mode = "none" if self.downsample_none.isChecked() else "distance"
        self.downsample_distance_value = float(self.downsample_distance_spin.value())
        self.interpolate = self.interp_box.isChecked()
        self.interp_step_value = float(self.interp_step_spin.value())
        self.lowpass = self.lowpass_box.isChecked()
        self.lowpass_cutoff_value = float(self.lowpass_cutoff_spin.value())

        self.accepted = True
        self.close()


# =========================================================
# Drawing view/window with undo/redo
# =========================================================

class DrawView(QGraphicsView):
    def __init__(self, scene, redraw_callback=None):
        super().__init__(scene)
        self.scene_ref = scene
        self.redraw_callback = redraw_callback

        self.all_strokes = []
        self.redo_stack = []
        self.current_stroke = []
        self.is_drawing = False
        self.base_time = None

        self.preview_pen_width = 3.0

        # Zoom state
        self.zoom_factor = 1.0
        self.zoom_step = 1.15
        self.min_zoom = 0.05
        self.max_zoom = 50.0

        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

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
                pen = QPen(Qt.GlobalColor.red, self.preview_pen_width)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

                self.scene_ref.addLine(
                    px, py, sample[0], sample[1],
                    pen
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
                self.redo_stack.clear()
            self.current_stroke = []
            if self.redraw_callback:
                self.redraw_callback()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Ctrl + wheel = zoom
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier:
            angle = event.angleDelta().y()
            if angle > 0:
                self.zoom_in()
            elif angle < 0:
                self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    def zoom_in(self):
        self._apply_zoom(self.zoom_step)

    def zoom_out(self):
        self._apply_zoom(1.0 / self.zoom_step)

    def _apply_zoom(self, factor):
        new_zoom = self.zoom_factor * factor
        if new_zoom < self.min_zoom or new_zoom > self.max_zoom:
            return

        self.scale(factor, factor)
        self.zoom_factor = new_zoom

    def reset_zoom_to_fit(self, target_item=None):
        self.resetTransform()
        self.zoom_factor = 1.0
        if target_item is not None:
            self.fitInView(target_item, Qt.AspectRatioMode.KeepAspectRatio)

    def undo(self):
        if self.current_stroke:
            self.current_stroke = []
            if self.redraw_callback:
                self.redraw_callback()
            return

        if self.all_strokes:
            self.redo_stack.append(self.all_strokes.pop())
            if self.redraw_callback:
                self.redraw_callback()

    def redo(self):
        if self.redo_stack:
            self.all_strokes.append(self.redo_stack.pop())
            if self.redraw_callback:
                self.redraw_callback()

class DrawWindow(QMainWindow):
    def __init__(self, mode_text, background_pixmap, guide_path=None, mode_key="move", parent_window=None):
        super().__init__()
        self.mode_key = mode_key
        self.setWindowTitle(f"TraceFX - {mode_text}")
        self.setGeometry(100, 100, 1280, 860)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.trace_marker_radius = TRACE_TEXT_BRUSH_RADIUS
        self.trace_marker_smoothness = 18
        self.trace_guide_y_offset = 17.0
        self.trace_marker_width = TRACE_TEXT_BRUSH_RADIUS * 0.75

        self.scene = QGraphicsScene()
        self.view = DrawView(self.scene, redraw_callback=self.rebuild_scene)
        layout.addWidget(self.view)

        # Trace Text controls
        # self.trace_marker_radius = TRACE_TEXT_BRUSH_RADIUS
        if self.mode_key == "trace_text":
            # self.view.preview_pen_width = max(1.0, self.trace_marker_radius * 2.0)

            # marker_width = float(self.trace_marker_radius * 2.0 / TRACE_TEXT_RADIUS_COMPENSATION)
            # self.view.preview_pen_width = marker_width
            self.view.preview_pen_width = self.trace_marker_width
        self.trace_marker_smoothness = 18

        if self.mode_key == "trace_text":
            trace_panel = QWidget()
            trace_layout = QFormLayout(trace_panel)

            self.marker_radius_spin = QDoubleSpinBox()
            self.marker_radius_spin.setRange(0.5, 50.0)
            self.marker_radius_spin.setDecimals(2)
            self.marker_radius_spin.setSingleStep(0.25)
            # self.marker_radius_spin.setValue(TRACE_TEXT_BRUSH_RADIUS)
            self.marker_radius_spin.setValue(self.trace_marker_width)
            self.marker_radius_spin.setSuffix(" px")

            self.marker_smoothness_spin = QDoubleSpinBox()
            self.marker_smoothness_spin.setRange(6.0, 64.0)
            self.marker_smoothness_spin.setDecimals(0)
            self.marker_smoothness_spin.setSingleStep(1.0)
            self.marker_smoothness_spin.setValue(18.0)

            trace_layout.addRow("Marker thickness:", self.marker_radius_spin)
            trace_layout.addRow("Marker smoothness:", self.marker_smoothness_spin)

            offset_row = QHBoxLayout()
            offset_row.addWidget(QLabel("Guide Y offset:"))
            self.guide_y_offset_spin = QDoubleSpinBox()
            self.guide_y_offset_spin.setRange(-200.0, 200.0)
            self.guide_y_offset_spin.setDecimals(1)
            self.guide_y_offset_spin.setSingleStep(1.0)
            self.guide_y_offset_spin.setValue(17.0)
            self.guide_y_offset_spin.setSuffix(" px")
            offset_row.addWidget(self.guide_y_offset_spin)
            offset_row.addStretch()
            trace_layout.addRow("Guide Y offset:", self.guide_y_offset_spin)

            self.guide_y_offset_spin.valueChanged.connect(self.on_trace_controls_changed)

            self.marker_radius_spin.valueChanged.connect(self.on_trace_controls_changed)
            self.marker_smoothness_spin.valueChanged.connect(self.on_trace_controls_changed)

            layout.addWidget(trace_panel)

        controls = QHBoxLayout()
        self.info_label = QLabel(
            "Left-click and drag to draw. Release to end a stroke. Ctrl + mouse wheel also zooms."
        )

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_in_button = QPushButton("Zoom In")
        self.reset_zoom_button = QPushButton("Reset Zoom")

        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.clear_button = QPushButton("Clear")
        self.continue_button = QPushButton("Continue")

        controls.addWidget(self.info_label)
        controls.addStretch()
        controls.addWidget(self.zoom_out_button)
        controls.addWidget(self.zoom_in_button)
        controls.addWidget(self.reset_zoom_button)
        controls.addWidget(self.undo_button)
        controls.addWidget(self.redo_button)
        controls.addWidget(self.clear_button)
        controls.addWidget(self.continue_button)
        layout.addLayout(controls)

        self.background_pixmap = background_pixmap
        self.guide_path = guide_path
        self.continue_clicked = False
        self._did_initial_fit = False

        self.zoom_in_button.clicked.connect(self.on_zoom_in)
        self.zoom_out_button.clicked.connect(self.on_zoom_out)
        self.reset_zoom_button.clicked.connect(self.on_reset_zoom)

        self.undo_button.clicked.connect(self.on_undo)
        self.redo_button.clicked.connect(self.on_redo)
        self.clear_button.clicked.connect(self.on_clear)
        self.continue_button.clicked.connect(self.on_continue)

        self.trace_guide_y_offset = 17.0

        self.rebuild_scene()

        # Open on same screen as first window
        if parent_window is not None:
            try:
                screen = parent_window.windowHandle().screen() if parent_window.windowHandle() else None
                if screen is None:
                    from PyQt6.QtGui import QGuiApplication
                    screen = QGuiApplication.screenAt(parent_window.frameGeometry().center())
                if screen is not None:
                    geo = screen.availableGeometry()
                    self.move(
                        geo.x() + (geo.width() - self.width()) // 2,
                        geo.y() + (geo.height() - self.height()) // 2,
                    )
            except Exception:
                pass
    
    def get_effective_scene_marker_width(self):
        return float(getattr(self, "trace_marker_width", self.view.preview_pen_width))

    def on_zoom_in(self):
        self.view.zoom_in()

    def on_zoom_out(self):
        self.view.zoom_out()

    def on_reset_zoom(self):
        self.view.reset_zoom_to_fit(self.pixmap_item)
    
    def on_trace_controls_changed(self):
        if self.mode_key == "trace_text":
            marker_width = float(self.marker_radius_spin.value())

            self.trace_marker_width = marker_width
            self.trace_marker_radius = marker_width / 2.0
            self.trace_marker_smoothness = int(round(self.marker_smoothness_spin.value()))
            self.trace_guide_y_offset = float(self.guide_y_offset_spin.value())

            # Preview pen width is in scene units, same as subtitle coordinates.
            self.view.preview_pen_width = marker_width

            self.rebuild_scene()

    def fit_background(self):
        if hasattr(self, "pixmap_item") and self.pixmap_item is not None:
            self.view.reset_zoom_to_fit(self.pixmap_item)

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_background()
        self._did_initial_fit = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._did_initial_fit and self.view.zoom_factor == 1.0:
            self.fit_background()

    def rebuild_scene(self):
        old_transform = self.view.transform()
        old_scene_rect = self.view.sceneRect()

        self.scene.clear()

        self.pixmap_item = self.scene.addPixmap(self.background_pixmap)

        if self.guide_path is not None and not self.guide_path.isEmpty():
            guide_path = QPainterPath(self.guide_path)

            if self.mode_key == "trace_text":
                guide_path.translate(0.0, getattr(self, "trace_guide_y_offset", -10.0))

            guide_item = QGraphicsPathItem(guide_path)
            pen = QPen(QColor(235, 250, 255, 255), 0.6)
            brush = QBrush(QColor(140, 230, 255, 235))
            guide_item.setPen(pen)
            guide_item.setBrush(brush)
            self.scene.addItem(guide_item)

        for stroke in self.view.all_strokes:
            for i in range(1, len(stroke)):
                x1, y1, _ = stroke[i - 1]
                x2, y2, _ = stroke[i]
                pen_width = self.view.preview_pen_width
                pen = QPen(Qt.GlobalColor.red, pen_width)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                self.scene.addLine(x1, y1, x2, y2, pen)

        self.view.setSceneRect(self.pixmap_item.boundingRect())

        # Restore current zoom/pan state after rebuild
        self.view.setTransform(old_transform)
        if not old_scene_rect.isNull():
            self.view.setSceneRect(self.pixmap_item.boundingRect())

    def on_undo(self):
        self.view.undo()

    def on_redo(self):
        self.view.redo()

    def on_clear(self):
        self.view.all_strokes.clear()
        self.view.redo_stack.clear()
        self.view.current_stroke = []
        self.rebuild_scene()

    def on_continue(self):
        self.continue_clicked = True
        self.close()

    @property
    def all_strokes(self):
        strokes = list(self.view.all_strokes)
        if self.view.current_stroke:
            strokes.append(self.view.current_stroke)
        return strokes


# =========================================================
# Generators
# =========================================================

def sub_move(line, l, strokes, timing_mode):
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


def sub_trace_text(line, l, strokes, timing_mode, brush_radius=TRACE_TEXT_BRUSH_RADIUS, brush_steps=18):
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

            clip_shape = build_trace_text_clip(
                strokes, stroke_index, point_index, brush_radius, brush_steps
            )

            l.start_time = int(round(start_t))
            l.end_time = int(round(end_t))
            l.text = "{\\clip(%s)}%s" % (clip_shape, line.raw_text)
            Pyegi.send_line(l)

            idx += 1


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    app = QApplication([])
    apply_dark_theme(app)
    
    io = Ass(Pyegi.get_input_file_path(), extended=True)
    meta, styles, lines = io.get_data()

    if not lines:
        raise RuntimeError("No subtitle lines found.")

    # Best-effort script resolution lookup
    playresx = int(getattr(meta, "play_res_x", 1280))
    playresy = int(getattr(meta, "play_res_y", 720))

    background_pixmap = prepare_preview_background(playresx, playresy)

    # First non-empty line as preview guide for Trace Text mode
    guide_line = None
    for line in lines:
        if getattr(line, "raw_text", "").strip():
            guide_line = line
            break

    guide_path = None
    if guide_line is not None:
        shape_ass = get_text_shape_ass(guide_line)
        debug_write(f"Guide shape available: {shape_ass is not None}")
        if shape_ass:
            raw_path = ass_draw_to_qpainterpath(shape_ass)
            if raw_path is not None:
                guide_path = translate_guide_path_to_line_position(
                    raw_path,
                    guide_line,
                    playresx,
                    playresy,
                )


    settings = SettingsWindow()
    settings.show()
    app.exec()

    if not settings.accepted:
        raise SystemExit

    mode_label = {
        "move": "Move Mode",
        "draw": "Draw Mode",
        "trace_text": "Trace Text Mode",
    }[settings.mode]

    draw_window = DrawWindow(
        mode_label,
        background_pixmap,
        guide_path=guide_path if settings.mode == "trace_text" else None,
        mode_key=settings.mode,
        parent_window=settings,
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

    for line in lines:
        l = line.copy()

        if settings.mode == "move":
            sub_move(line, l, strokes, settings.timing)
        elif settings.mode == "draw":
            sub_draw(line, l, strokes, settings.timing)
        else:
            marker_width = draw_window.get_effective_scene_marker_width()
            brush_radius = (marker_width / 2.0) * TRACE_TEXT_OUTPUT_RADIUS_MULTIPLIER

            sub_trace_text(
                line,
                l,
                strokes,
                settings.timing,
                brush_radius=brush_radius,
                brush_steps=getattr(draw_window, "trace_marker_smoothness", 18),
            )

    Pyegi.create_output_file()
