from PyegiInitializer import *
# -*- coding: utf-8 -*-
"""
Pyegi Selection Tool

A Pyegi-oriented script that provides:
1) A first window for choosing the working frame source and output mode.
2) A second window for interactive image selection with multiple tools.
3) Output generation as either ASS \clip/\iclip or a filled ASS vector shape
   using the average color of the selected area.


What this script assumes
------------------------
This script focuses on:
- asking which frame to use per line:
    - current video position
    - first frame of the subtitle line
- asking what output to generate:
    - clip/iclip tag from selection mask
    - filled ASS vector shape colored by average selected color
- allowing the user to interactively create and edit a selection mask

Dependencies
------------
- Python 3.10
- numpy
- opencv-python
- PyQt6

Selection tools included
------------------------
- Rectangular marquee              [M]
- Elliptical marquee               [U]
- Polygon / straight-line lasso    [L]
- Freehand lasso                   [F]
- Magic wand                       [W]
- Brush add                        [B]
- Brush subtract                   [E]
- Expand selection                 [Alt+]]
- Contract selection               [Alt+[]
- Fill islands                     [Ctrl+Shift+F]
- Pan                              [Space drag or Middle drag]
- Zoom                             [Ctrl+Wheel]
- Reset the view to fit            [0]
- Undo selection                   [Ctrl+Z]
- Redo selection                   [Ctrl+Y / Ctrl+Shift+Z]
- Polygon lasso: finish polygon    [Right-click]
- Zoom                             [Ctrl+wheel]
- Pan                              [Space+drag or middle-drag]

Modifiers
---------
- Shift  : add to current selection
- Alt    : subtract from current selection
- Ctrl+D : deselect all
- Enter  : accept current line/result
- Esc    : cancel current drag / finish nothing
- [ / ]  : decrease / increase brush size
- - / =  : decrease / increase magic wand tolerance

Magic wand options
------------------
- tolerance                : per-pixel color distance threshold
- contiguous only          : whether selection must be connected to clicked seed
- use diagonal connectivity: 4-neighbor vs 8-neighbor style connectivity
- feather radius           : softens mask edges slightly before vectorization/output
- minimum island area      : removes tiny islands from result

Notes
-----
- ASS clip tags are generated as vector clips, not rectangular clips.
- ASS filled shape output uses the average BGR color of the selected pixels.
"""


from pyonfx import *
import numpy as np
import re
import sys
from pathlib import Path
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None

import Pyegi

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QMessageBox,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSplitter,
    QScrollArea,
    QSizePolicy,
)
from PyQt6.QtGui import (
    QPixmap,
    QPen,
    QImage,
    QColor,
    QPainter,
    QPainterPath,
    QBrush,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal


project_properties_table = Pyegi.get_project_properties()
PROJECT_VIDEO_PATH = project_properties_table.get("video_file")
CURRENT_VIDEO_POSITION = project_properties_table.get("video_position", 0)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def apply_dark_theme(app):
    app.setStyleSheet("""
        QWidget {
            background-color: #1e1e1e;
            color: #e6e6e6;
            font-size: 11pt;
        }

        QMainWindow {
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

        QDoubleSpinBox, QSpinBox, QComboBox {
            background-color: #2d2d30;
            border: 1px solid #5a5a62;
            border-radius: 6px;
            padding: 4px 6px;
            color: #f0f0f0;
            min-height: 24px;
        }

        QComboBox QAbstractItemView {
            background-color: #2d2d30;
            color: #f0f0f0;
            selection-background-color: #3a3a3f;
            selection-color: #ffffff;
        }

        QGroupBox {
            border: 1px solid #5a5a62;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px 0 4px;
            color: #f0f0f0;
        }
    """)



@dataclass
class ConfigResult:
    frame_source: str
    output_mode: str
    clip_mode: str
    tool_default: str
    contour_simplify: float
    feather_radius: int


@dataclass
class WandOptions:
    tolerance: int = 32
    contiguous: bool = True
    diagonal: bool = True
    min_island_area: int = 16
    feather_radius: int = 0


@dataclass
class BrushOptions:
    radius: int = 12


@dataclass
class SelectionResult:
    mask: np.ndarray
    average_bgr: Tuple[int, int, int]
    ass_path: str


class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Selection Tool Setup")
        self.setMinimumWidth(460)

        self.accepted = False
        self.frame_source = "current"
        self.output_mode = "clip"
        self.clip_mode = "clip"

        layout = QVBoxLayout(self)

        title = QLabel("Selection Tool")
        title.setStyleSheet("font-size: 18pt; font-weight: 600;")
        layout.addWidget(title)

        subtitle = QLabel("Choose the working frame and output type.")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        layout.addWidget(QLabel("Working frame"))
        self.rb_current = QRadioButton("Use current frame / video position")
        self.rb_line_start = QRadioButton("Use first frame of subtitle line")
        self.rb_current.setChecked(True)

        self.frame_group = QButtonGroup(self)
        self.frame_group.addButton(self.rb_current)
        self.frame_group.addButton(self.rb_line_start)

        layout.addWidget(self.rb_current)
        layout.addWidget(self.rb_line_start)

        layout.addSpacing(10)

        form = QFormLayout()

        self.output_combo = QComboBox()
        self.output_combo.addItems([
            "Clip tag",
            "Inverse clip tag",
            "Shape with average selected color",
        ])

        form.addRow("Output", self.output_combo)
        
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.cancel_button = QPushButton("Cancel")
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

        self.start_button.clicked.connect(self.on_start)
        self.cancel_button.clicked.connect(self.close)

    def on_start(self):
        self.frame_source = "current" if self.rb_current.isChecked() else "line_start"

        output_index = self.output_combo.currentIndex()
        if output_index == 0:
            self.output_mode = "clip"
            self.clip_mode = "clip"
        elif output_index == 1:
            self.output_mode = "clip"
            self.clip_mode = "iclip"
        else:
            self.output_mode = "shape"
            self.clip_mode = "clip"

        self.initial_tool = "Magic Wand"
        self.contour_simplify = 1.5
        self.global_feather = 0
        self.accepted = True
        self.close()


def event_pos_qpoint(event):
    p = event.position()
    return QPoint(int(round(p.x())), int(round(p.y())))


def mouse_button(event, button):
    return event.button() == button

def get_video_path():
    if not PROJECT_VIDEO_PATH:
        return None
    p = Path(PROJECT_VIDEO_PATH)
    if not p.exists():
        return None
    if p.suffix.lower() not in VIDEO_EXTENSIONS:
        return None
    return p


def get_frame_at_number(video_path, frame_number, playresx, playresy):
    if cv2 is None or video_path is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        frame_number = max(0, int(frame_number))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame_bgr = cap.read()
    finally:
        cap.release()

    if not ok or frame_bgr is None:
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (playresx, playresy), interpolation=cv2.INTER_AREA)
    return frame_rgb


def ms_to_frame_number(ms, fps):
    return int(round((float(ms) / 1000.0) * float(fps)))


def get_video_fps(video_path):
    if cv2 is None or video_path is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    if fps is None or fps <= 0:
        return None
    return fps


def get_preview_frame_for_line(line, frame_source_mode, playresx, playresy):
    video_path = get_video_path()
    if video_path is None:
        return None

    if frame_source_mode == "current":
        frame_no = int(CURRENT_VIDEO_POSITION or 0)
        return get_frame_at_number(video_path, frame_no, playresx, playresy)

    fps = get_video_fps(video_path)
    if fps is None:
        return None

    frame_no = ms_to_frame_number(line.start_time, fps)
    return get_frame_at_number(video_path, frame_no, playresx, playresy)


def bgr_to_ass_bgr_tag(bgr: Tuple[int, int, int]) -> str:
    b, g, r = bgr
    return "&H%02X%02X%02X&" % (b, g, r)


def qimage_from_bgr(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(
        rgb.data,
        w,
        h,
        bytes_per_line,
        QImage.Format.Format_RGB888
    ).copy()


def ensure_mask_uint8(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = mask * 255
    return mask


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = ensure_mask_uint8(mask)
    if radius <= 0:
        return (mask > 0).astype(np.uint8) * 255
    k = radius * 2 + 1
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    return (blurred >= 127).astype(np.uint8) * 255


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = ensure_mask_uint8(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out


def average_color_under_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int]:
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return (255, 255, 255)
    pixels = frame_bgr[mask_bool]
    mean = pixels.mean(axis=0)
    return tuple(int(round(v)) for v in mean.tolist())


def ass_path_from_mask(mask: np.ndarray, simplify_epsilon: float = 1.5) -> str:
    mask = ensure_mask_uint8(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""

    parts = []
    for contour in contours:
        if len(contour) < 3:
            continue
        approx = cv2.approxPolyDP(contour, simplify_epsilon, True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts) < 3:
            continue
        cmd = ["m %d %d" % (pts[0][0], pts[0][1])]
        for x, y in pts[1:]:
            cmd.append("l %d %d" % (x, y))
        cmd.append("c")
        parts.append(" ".join(cmd))
    return " ".join(parts)

def parse_ass_draw_path_simple(path_text):
    """
    Parse only the simple ASS vector syntax this script produces:
    m x y l x y ... c
    Multiple subpaths are allowed.
    Returns a list of polygons, each polygon is a list of (x, y).
    """
    if not path_text:
        return []

    tokens = path_text.strip().split()
    polygons = []
    current = []
    cmd = None
    i = 0

    while i < len(tokens):
        tok = tokens[i].lower()

        if tok in ("m", "n", "l", "c"):
            cmd = tok
            i += 1

            if cmd == "c":
                if len(current) >= 3:
                    polygons.append(current)
                current = []
            continue

        if cmd in ("m", "n", "l"):
            if i + 1 >= len(tokens):
                break
            try:
                x = int(round(float(tokens[i])))
                y = int(round(float(tokens[i + 1])))
            except Exception:
                break

            if cmd in ("m", "n"):
                if len(current) >= 3:
                    polygons.append(current)
                current = [(x, y)]
            else:
                current.append((x, y))

            i += 2
            continue

        # Unsupported command for this importer.
        return []

    if len(current) >= 3:
        polygons.append(current)

    return polygons


def mask_from_ass_draw_path(path_text, width, height):
    polygons = parse_ass_draw_path_simple(path_text)
    mask = np.zeros((height, width), dtype=np.uint8)

    if not polygons:
        return mask

    pts_list = []
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        if len(pts) >= 3:
            pts_list.append(pts)

    if pts_list:
        cv2.fillPoly(mask, pts_list, 255)

    return mask

def extract_matching_vector_clip_path(text, clip_mode):
    """
    Returns the vector path from a matching \\clip(...) or \\iclip(...)
    only if it looks like the kind of vector clip this script produces.
    Returns None otherwise.
    """
    if not text:
        return None

    tag_name = "iclip" if clip_mode == "iclip" else "clip"
    pattern = r"\\%s\((.*?)\)" % tag_name
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None

    # Use the last matching tag, consistent with typical override behavior.
    path = matches[-1].strip()

    # Reject rectangular clips like x1,y1,x2,y2.
    if re.fullmatch(r"\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*", path):
        return None

    # Accept only simple paths this script can round-trip.
    polys = parse_ass_draw_path_simple(path)
    if not polys:
        return None

    return path


def extract_generated_shape_path(text):
    """
    Returns the vector path only if the line looks like a shape line this script
    itself could have produced: override block contains \\p1, body is an m/l/c path.
    """
    if not text:
        return None

    m = re.match(r"^\{([^}]*)\}(.*)$", text, flags=re.DOTALL)
    if not m:
        return None

    tags = m.group(1)
    body = m.group(2).strip()

    if re.search(r"\\p1\b", tags) is None:
        return None

    polys = parse_ass_draw_path_simple(body)
    if not polys:
        return None

    return body


def build_initial_mask_for_line(line, output_mode, clip_mode, width, height):
    text = getattr(line, "raw_text", "") or ""

    if output_mode == "clip":
        path = extract_matching_vector_clip_path(text, clip_mode)
        if path is None:
            return None
        return mask_from_ass_draw_path(path, width, height)

    if output_mode == "shape":
        path = extract_generated_shape_path(text)
        if path is None:
            return None
        return mask_from_ass_draw_path(path, width, height)

    return None

def replace_matching_vector_clip(text, clip_mode, new_path):
    """
    Replace the last matching vector \\clip(...) or \\iclip(...) in the line.
    Only replaces vector clips, not rectangular clips.
    Returns None if no matching replaceable tag is found.
    """
    if not text or not new_path:
        return None

    tag_name = "iclip" if clip_mode == "iclip" else "clip"
    pattern = r"(\\%s\((.*?)\))" % tag_name

    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
    if not matches:
        return None

    # Walk from the end so we replace the last matching usable one.
    for m in reversed(matches):
        full_tag = m.group(1)
        inner = m.group(2).strip()

        # Skip rectangular clips.
        if re.fullmatch(r"\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*", inner):
            continue

        # Only replace clips we can parse / that our code can produce.
        if not parse_ass_draw_path_simple(inner):
            continue

        replacement = "\\%s(%s)" % (tag_name, new_path)
        return text[:m.start()] + replacement + text[m.end():]

    return None

def replace_generated_shape_line(text, new_path, avg_bgr):
    """
    Replace an existing generated-shape style line:
    { ... \\p1 ... }<simple m/l/c path>
    Returns None if the line is not a replaceable generated shape.
    """
    if not text or not new_path:
        return None

    m = re.match(r"^\{([^}]*)\}(.*)$", text, flags=re.DOTALL)
    if not m:
        return None

    tags = m.group(1)
    body = m.group(2).strip()

    if re.search(r"\\p1\b", tags) is None:
        return None

    if not parse_ass_draw_path_simple(body):
        return None

    color_tag = bgr_to_ass_bgr_tag(avg_bgr)

    new_tags = tags

    if re.search(r"\\1c&H[0-9A-Fa-f]+&", new_tags):
        new_tags = re.sub(r"\\1c&H[0-9A-Fa-f]+&", r"\\1c%s" % color_tag, new_tags, count=1)
    else:
        new_tags += r"\1c%s" % color_tag

    return "{%s}%s" % (new_tags, new_path)

def insert_tag_into_first_override_block(text, tag):
    if not tag:
        return text

    if text.startswith("{"):
        close = text.find("}")
        if close != -1:
            return text[:close] + tag + text[close:]

    return "{%s}%s" % (tag, text)

def make_vector_clip_tag(path: str, inverse: bool = False) -> str:
    if not path:
        return ""
    tag_name = "iclip" if inverse else "clip"
    return "\\%s(%s)" % (tag_name, path)


def make_shape_line_text(ass_path: str, avg_bgr: Tuple[int, int, int]) -> str:
    color_tag = bgr_to_ass_bgr_tag(avg_bgr)
    return "{\\an7\\bord0\\shad0\\1c%s\\p1}%s" % (color_tag, ass_path)


def combine_masks(base: np.ndarray, incoming: np.ndarray, mode: str) -> np.ndarray:
    base = ensure_mask_uint8(base)
    incoming = ensure_mask_uint8(incoming)
    if mode == "replace":
        return incoming.copy()
    if mode == "add":
        return np.maximum(base, incoming)
    if mode == "subtract":
        out = base.copy()
        out[incoming > 0] = 0
        return out
    return incoming.copy()

def expand_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = ensure_mask_uint8(mask)
    if radius <= 0:
        return mask.copy()
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def contract_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = ensure_mask_uint8(mask)
    if radius <= 0:
        return mask.copy()
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, kernel, iterations=1)


def fill_mask_islands(mask: np.ndarray) -> np.ndarray:
    mask = ensure_mask_uint8(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if contours:
        cv2.drawContours(out, contours, -1, 255, thickness=-1)
    return out

def seed_magic_wand(frame_bgr: np.ndarray, seed_xy: Tuple[int, int], opts: WandOptions) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    sx, sy = seed_xy
    sx = int(np.clip(sx, 0, w - 1))
    sy = int(np.clip(sy, 0, h - 1))
    seed = frame_bgr[sy, sx].astype(np.int16)

    diff = frame_bgr.astype(np.int16) - seed[None, None, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    similar = (dist <= opts.tolerance).astype(np.uint8)

    if opts.contiguous:
        visited = np.zeros((h, w), dtype=np.uint8)
        out = np.zeros((h, w), dtype=np.uint8)
        stack = [(sx, sy)]
        neighbors4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        neighbors8 = neighbors4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        neighbors = neighbors8 if opts.diagonal else neighbors4

        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            if visited[y, x]:
                continue
            visited[y, x] = 1
            if similar[y, x] == 0:
                continue
            out[y, x] = 255
            for dx, dy in neighbors:
                stack.append((x + dx, y + dy))
        mask = out
    else:
        mask = similar * 255

    if opts.min_island_area > 1:
        mask = remove_small_components(mask, opts.min_island_area)
    if opts.feather_radius > 0:
        mask = feather_mask(mask, opts.feather_radius)
    return mask


def rasterize_polygon(points: List[Tuple[int, int]], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def rasterize_rect(p0: Tuple[int, int], p1: Tuple[int, int], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0 = p0
    x1, y1 = p1
    xa, xb = sorted([int(x0), int(x1)])
    ya, yb = sorted([int(y0), int(y1)])
    xa = int(np.clip(xa, 0, w - 1))
    xb = int(np.clip(xb, 0, w - 1))
    ya = int(np.clip(ya, 0, h - 1))
    yb = int(np.clip(yb, 0, h - 1))
    cv2.rectangle(mask, (xa, ya), (xb, yb), 255, thickness=-1)
    return mask


def rasterize_ellipse(p0: Tuple[int, int], p1: Tuple[int, int], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0 = p0
    x1, y1 = p1
    cx = int(round((x0 + x1) / 2.0))
    cy = int(round((y0 + y1) / 2.0))
    rx = int(round(abs(x1 - x0) / 2.0))
    ry = int(round(abs(y1 - y0) / 2.0))
    if rx > 0 and ry > 0:
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return mask


def rasterize_brush(points: List[Tuple[int, int]], radius: int, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if not points:
        return mask
    for i, pt in enumerate(points):
        p = (int(pt[0]), int(pt[1]))
        cv2.circle(mask, p, radius, 255, -1)
        if i > 0:
            prev = (int(points[i - 1][0]), int(points[i - 1][1]))
            cv2.line(mask, prev, p, 255, thickness=max(1, radius * 2))
    return mask


def mod_mode_from_keyboard() -> str:
    mods = QApplication.keyboardModifiers()
    shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
    alt = bool(mods & Qt.KeyboardModifier.AltModifier)
    if shift and not alt:
        return "add"
    if alt and not shift:
        return "subtract"
    return "replace"


class CanvasWidget(QWidget):
    selectionChanged = pyqtSignal()
    statusChanged = pyqtSignal(str)

    def __init__(self, frame_bgr: np.ndarray, initial_mask=None, parent=None):
        super().__init__(parent)
        self.frame_bgr = frame_bgr.copy()
        self.h, self.w = self.frame_bgr.shape[:2]
        self.base_qimage = qimage_from_bgr(self.frame_bgr)

        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.last_mouse_pos = None
        self.panning = False
        self.space_down = False

        self.selection_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.preview_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        if initial_mask is not None:
            if initial_mask.shape[:2] == (self.h, self.w):
                self.selection_mask = ensure_mask_uint8(initial_mask.copy())

        self.undo_stack = []
        self.redo_stack = []
        self.edit_undo_stack = []
        self.edit_redo_stack = []

        self.tool = "Magic Wand"
        self.wand_options = WandOptions()
        self.brush_options = BrushOptions()

        self.drag_start_img = None
        self.drag_current_img = None
        self.poly_points = []
        self.freehand_points = []

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 180)

    def expand_selection(self, radius: int = 1):
        if radius <= 0:
            return
        self.push_undo_state()
        self.selection_mask = expand_mask(self.selection_mask, radius)
        self.clear_active_edit()
        self.selectionChanged.emit()
        self.update()
        self.statusChanged.emit("Expanded selection by %d px" % radius)


    def contract_selection(self, radius: int = 1):
        if radius <= 0:
            return
        self.push_undo_state()
        self.selection_mask = contract_mask(self.selection_mask, radius)
        self.clear_active_edit()
        self.selectionChanged.emit()
        self.update()
        self.statusChanged.emit("Contracted selection by %d px" % radius)


    def fill_islands(self):
        self.push_undo_state()
        self.selection_mask = fill_mask_islands(self.selection_mask)
        self.clear_active_edit()
        self.selectionChanged.emit()
        self.update()
        self.statusChanged.emit("Filled islands")

    def push_undo_state(self):
        self.undo_stack.append(self.selection_mask.copy())
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack = []


    def push_edit_state(self):
        state = {
            "poly_points": list(self.poly_points),
            "freehand_points": list(self.freehand_points),
            "preview_mask": self.preview_mask.copy(),
            "drag_start_img": self.drag_start_img,
            "drag_current_img": self.drag_current_img,
            "tool": self.tool,
        }
        self.edit_undo_stack.append(state)
        if len(self.edit_undo_stack) > 100:
            self.edit_undo_stack.pop(0)
        self.edit_redo_stack = []


    def has_active_edit(self):
        return bool(self.poly_points or self.freehand_points or np.any(self.preview_mask))


    def clear_active_edit(self):
        self.preview_mask[:] = 0
        self.drag_start_img = None
        self.drag_current_img = None
        self.poly_points = []
        self.freehand_points = []
        self.edit_undo_stack = []
        self.edit_redo_stack = []
        self.update()


    def undo(self):
        if self.has_active_edit() and self.edit_undo_stack:
            current = {
                "poly_points": list(self.poly_points),
                "freehand_points": list(self.freehand_points),
                "preview_mask": self.preview_mask.copy(),
                "drag_start_img": self.drag_start_img,
                "drag_current_img": self.drag_current_img,
                "tool": self.tool,
            }
            self.edit_redo_stack.append(current)

            prev = self.edit_undo_stack.pop()
            self.poly_points = list(prev["poly_points"])
            self.freehand_points = list(prev["freehand_points"])
            self.preview_mask = prev["preview_mask"].copy()
            self.drag_start_img = prev["drag_start_img"]
            self.drag_current_img = prev["drag_current_img"]
            self.update()
            self.statusChanged.emit("Undo edit")
            return

        if not self.undo_stack:
            self.statusChanged.emit("Nothing to undo")
            return

        self.redo_stack.append(self.selection_mask.copy())
        self.selection_mask = self.undo_stack.pop()
        self.clear_active_edit()
        self.selectionChanged.emit()
        self.update()
        self.statusChanged.emit("Undo selection")


    def redo(self):
        if self.has_active_edit() and self.edit_redo_stack:
            current = {
                "poly_points": list(self.poly_points),
                "freehand_points": list(self.freehand_points),
                "preview_mask": self.preview_mask.copy(),
                "drag_start_img": self.drag_start_img,
                "drag_current_img": self.drag_current_img,
                "tool": self.tool,
            }
            self.edit_undo_stack.append(current)

            nxt = self.edit_redo_stack.pop()
            self.poly_points = list(nxt["poly_points"])
            self.freehand_points = list(nxt["freehand_points"])
            self.preview_mask = nxt["preview_mask"].copy()
            self.drag_start_img = nxt["drag_start_img"]
            self.drag_current_img = nxt["drag_current_img"]
            self.update()
            self.statusChanged.emit("Redo edit")
            return

        if not self.redo_stack:
            self.statusChanged.emit("Nothing to redo")
            return

        self.undo_stack.append(self.selection_mask.copy())
        self.selection_mask = self.redo_stack.pop()
        self.clear_active_edit()
        self.selectionChanged.emit()
        self.update()
        self.statusChanged.emit("Redo selection")

    def push_undo_state(self):
        self.undo_stack.append(self.selection_mask.copy())
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack = []


    def undo(self):
        if not self.undo_stack:
            self.statusChanged.emit("Nothing to undo")
            return

        self.redo_stack.append(self.selection_mask.copy())
        self.selection_mask = self.undo_stack.pop()
        self.preview_mask[:] = 0
        self.drag_start_img = None
        self.drag_current_img = None
        self.poly_points = []
        self.freehand_points = []
        self.selectionChanged.emit()
        self.update()
        self.statusChanged.emit("Undo")


    def redo(self):
        if not self.redo_stack:
            self.statusChanged.emit("Nothing to redo")
            return

        self.undo_stack.append(self.selection_mask.copy())
        self.selection_mask = self.redo_stack.pop()
        self.preview_mask[:] = 0
        self.drag_start_img = None
        self.drag_current_img = None
        self.poly_points = []
        self.freehand_points = []
        self.selectionChanged.emit()
        self.update()
        self.statusChanged.emit("Redo")

    def fit_image_to_widget(self):
        avail_w = max(1, self.width())
        avail_h = max(1, self.height())

        sx = avail_w / float(self.w)
        sy = avail_h / float(self.h)
        self.scale_factor = max(0.05, min(30.0, min(sx, sy)))

        drawn_w = self.w * self.scale_factor
        drawn_h = self.h * self.scale_factor

        self.offset = QPoint(
            int(round((avail_w - drawn_w) / 2.0)),
            int(round((avail_h - drawn_h) / 2.0)),
        )
        self.update()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, "_did_initial_fit"):
            self._did_initial_fit = False
        if not self._did_initial_fit:
            self.fit_image_to_widget()
            self._did_initial_fit = True

    def image_to_view(self, x: int, y: int) -> QPoint:
        return QPoint(
            int(round(x * self.scale_factor)) + self.offset.x(),
            int(round(y * self.scale_factor)) + self.offset.y(),
        )

    def view_to_image(self, pos: QPoint) -> Tuple[int, int]:
        x = int(math.floor((pos.x() - self.offset.x()) / self.scale_factor))
        y = int(math.floor((pos.y() - self.offset.y()) / self.scale_factor))
        x = int(np.clip(x, 0, self.w - 1))
        y = int(np.clip(y, 0, self.h - 1))
        return x, y

    def set_tool(self, tool: str):
        self.tool = tool
        self.preview_mask[:] = 0
        self.drag_start_img = None
        self.drag_current_img = None
        self.poly_points = []
        self.freehand_points = []
        self.update()
        self.statusChanged.emit("Tool: %s" % tool)

    def clear_selection(self):
        if np.any(self.selection_mask):
            self.push_undo_state()
        self.selection_mask[:] = 0
        self.clear_active_edit()
        self.selectionChanged.emit()
        self.update()
        self.statusChanged.emit("Selection cleared")

    def finish_polygon(self):
        if len(self.poly_points) >= 3:
            incoming = rasterize_polygon(self.poly_points, (self.h, self.w))
            self._commit_incoming(incoming, mod_mode_from_keyboard())
            self.poly_points = []
            self.update()

    def _commit_incoming(self, incoming: np.ndarray, mode: str):
        self.push_undo_state()
        self.selection_mask = combine_masks(self.selection_mask, incoming, mode)
        self.preview_mask[:] = 0
        self.drag_start_img = None
        self.drag_current_img = None
        self.poly_points = []
        self.freehand_points = []
        self.edit_undo_stack = []
        self.edit_redo_stack = []
        self.selectionChanged.emit()
        self.update()

    def _paint_mask_overlay(self, painter: QPainter, mask: np.ndarray, color: QColor):
        if not np.any(mask):
            return
        rgba = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        rgba[mask > 0] = [color.red(), color.green(), color.blue(), color.alpha()]
        qimg = QImage(
            rgba.data,
            self.w,
            self.h,
            self.w * 4,
            QImage.Format.Format_RGBA8888
        ).copy()
        target = QRect(
            self.offset.x(),
            self.offset.y(),
            int(round(self.w * self.scale_factor)),
            int(round(self.h * self.scale_factor)),
        )
        painter.drawImage(target, qimg)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        target_rect = QRect(
            self.offset.x(),
            self.offset.y(),
            int(round(self.w * self.scale_factor)),
            int(round(self.h * self.scale_factor)),
        )
        painter.drawImage(target_rect, self.base_qimage)

        self._paint_mask_overlay(painter, self.selection_mask, QColor(0, 180, 255, 90))
        self._paint_mask_overlay(painter, self.preview_mask, QColor(255, 200, 0, 90))

        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)

        if self.tool == "Polygon Lasso" and self.poly_points:
            path = QPainterPath()
            p0 = self.image_to_view(self.poly_points[0][0], self.poly_points[0][1])
            path.moveTo(float(p0.x()), float(p0.y()))
            for pt in self.poly_points[1:]:
                pv = self.image_to_view(pt[0], pt[1])
                path.lineTo(float(pv.x()), float(pv.y()))
            if self.drag_current_img is not None:
                pv = self.image_to_view(self.drag_current_img[0], self.drag_current_img[1])
                path.lineTo(float(pv.x()), float(pv.y()))
            painter.drawPath(path)

        if self.tool in ("Freehand Lasso", "Brush Add", "Brush Subtract") and self.freehand_points:
            path = QPainterPath()
            first = self.freehand_points[0]
            pv = self.image_to_view(first[0], first[1])
            path.moveTo(float(pv.x()), float(pv.y()))
            for pt in self.freehand_points[1:]:
                pv = self.image_to_view(pt[0], pt[1])
                path.lineTo(float(pv.x()), float(pv.y()))
            painter.drawPath(path)

        if self.tool in ("Rectangular Marquee", "Elliptical Marquee") and self.drag_start_img and self.drag_current_img:
            p0 = self.image_to_view(self.drag_start_img[0], self.drag_start_img[1])
            p1 = self.image_to_view(self.drag_current_img[0], self.drag_current_img[1])
            rect = QRect(p0, p1).normalized()
            if self.tool == "Rectangular Marquee":
                painter.drawRect(rect)
            else:
                painter.drawEllipse(rect)


    def mousePressEvent(self, event):
        self.setFocus()
        self.last_mouse_pos = event_pos_qpoint(event)

        if mouse_button(event, Qt.MouseButton.MiddleButton) or (
            mouse_button(event, Qt.MouseButton.LeftButton) and self.space_down
        ):
            self.panning = True
            return

        pos = event_pos_qpoint(event)
        ix, iy = self.view_to_image(pos)

        if mouse_button(event, Qt.MouseButton.LeftButton):
            if self.tool == "Magic Wand":
                incoming = seed_magic_wand(self.frame_bgr, (ix, iy), self.wand_options)
                self._commit_incoming(incoming, mod_mode_from_keyboard())
                self.statusChanged.emit(
                    "Magic wand at (%d, %d) | tol=%d | mode=%s" %
                    (ix, iy, self.wand_options.tolerance, mod_mode_from_keyboard())
                )
                return

            if self.tool in ("Rectangular Marquee", "Elliptical Marquee"):
                self.drag_start_img = (ix, iy)
                self.drag_current_img = (ix, iy)
                return

            if self.tool == "Polygon Lasso":
                self.push_edit_state()
                self.poly_points.append((ix, iy))
                self.drag_current_img = (ix, iy)
                self.update()
                return

            if self.tool in ("Freehand Lasso", "Brush Add", "Brush Subtract"):
                self.push_edit_state()
                self.freehand_points = [(ix, iy)]
                return

        if mouse_button(event, Qt.MouseButton.RightButton) and self.tool == "Polygon Lasso":
            self.finish_polygon()
            return

    def mouseMoveEvent(self, event):
        if self.panning and self.last_mouse_pos is not None:
            pos = event_pos_qpoint(event)
            delta = pos - self.last_mouse_pos
            self.offset += delta
            self.last_mouse_pos = pos
            self.update()
            return

        pos = event_pos_qpoint(event)
        ix, iy = self.view_to_image(pos)
        self.drag_current_img = (ix, iy)

        if self.tool == "Polygon Lasso" and self.poly_points:
            self.update()
            return

        if self.tool in ("Rectangular Marquee", "Elliptical Marquee") and self.drag_start_img:
            if self.tool == "Rectangular Marquee":
                self.preview_mask = rasterize_rect(self.drag_start_img, (ix, iy), (self.h, self.w))
            else:
                self.preview_mask = rasterize_ellipse(self.drag_start_img, (ix, iy), (self.h, self.w))
            self.update()
            return

        if self.tool == "Freehand Lasso" and self.freehand_points:
            self.push_edit_state()
            self.freehand_points.append((ix, iy))
            self.preview_mask = rasterize_polygon(self.freehand_points, (self.h, self.w))
            self.update()
            return

        if self.tool in ("Brush Add", "Brush Subtract") and self.freehand_points:
            self.push_edit_state()
            self.freehand_points.append((ix, iy))
            self.preview_mask = rasterize_brush(
                self.freehand_points,
                self.brush_options.radius,
                (self.h, self.w),
            )
            self.update()
            return

    def mouseReleaseEvent(self, event):
        if self.panning:
            self.panning = False
            return

        if mouse_button(event, Qt.MouseButton.LeftButton):
            if self.tool in ("Rectangular Marquee", "Elliptical Marquee") and self.drag_start_img and self.drag_current_img:
                incoming = self.preview_mask.copy()
                self._commit_incoming(incoming, mod_mode_from_keyboard())
                self.drag_start_img = None
                self.drag_current_img = None
                return

            if self.tool == "Freehand Lasso" and self.freehand_points:
                incoming = rasterize_polygon(self.freehand_points, (self.h, self.w))
                self._commit_incoming(incoming, mod_mode_from_keyboard())
                self.freehand_points = []
                return

            if self.tool == "Brush Add" and self.freehand_points:
                incoming = rasterize_brush(self.freehand_points, self.brush_options.radius, (self.h, self.w))
                self._commit_incoming(incoming, "add")
                self.freehand_points = []
                return

            if self.tool == "Brush Subtract" and self.freehand_points:
                incoming = rasterize_brush(self.freehand_points, self.brush_options.radius, (self.h, self.w))
                self._commit_incoming(incoming, "subtract")
                self.freehand_points = []
                return

    def wheelEvent(self, event):
        mods = QApplication.keyboardModifiers()
        if not (mods & Qt.KeyboardModifier.ControlModifier):
            event.ignore()
            return

        angle = event.angleDelta().y()
        old_scale = self.scale_factor
        if angle > 0:
            self.scale_factor *= 1.1
        elif angle < 0:
            self.scale_factor /= 1.1

        self.scale_factor = max(0.05, min(30.0, self.scale_factor))

        cursor_pos = event_pos_qpoint(event)
        ix, iy = self.view_to_image(cursor_pos)

        self.offset = QPoint(
            int(round(cursor_pos.x() - ix * self.scale_factor)),
            int(round(cursor_pos.y() - iy * self.scale_factor)),
        )

        if abs(old_scale - self.scale_factor) > 1e-9:
            self.update()

        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.space_down = True
            event.accept()
            return
        if key == Qt.Key.Key_BracketRight:
            self.brush_options.radius = max(1, self.brush_options.radius - 1)
            self.statusChanged.emit("Brush radius: %d" % self.brush_options.radius)
            event.accept()
            return
        if key == Qt.Key.Key_BracketRight:
            self.brush_options.radius += 1
            self.statusChanged.emit("Brush radius: %d" % self.brush_options.radius)
            event.accept()
            return
        if key == Qt.Key.Key_Minus:
            self.wand_options.tolerance = max(0, self.wand_options.tolerance - 1)
            self.statusChanged.emit("Wand tolerance: %d" % self.wand_options.tolerance)
            event.accept()
            return
        if key == Qt.Key.Key_Equal:
            self.wand_options.tolerance = min(442, self.wand_options.tolerance + 1)
            self.statusChanged.emit("Wand tolerance: %d" % self.wand_options.tolerance)
            event.accept()
            return
        if key == Qt.Key.Key_Escape:
            self.preview_mask[:] = 0
            self.drag_start_img = None
            self.drag_current_img = None
            self.freehand_points = []
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space_down = False
            event.accept()
            return
        super().keyReleaseEvent(event)

class SelectionWindow(QMainWindow):
    def __init__(
        self,
        frame_rgb: np.ndarray,
        line,
        output_mode: str,
        clip_mode: str,
        initial_tool: str,
        contour_simplify: float,
        global_feather: int,
        initial_mask=None,
        parent_window=None,
    ):
        super().__init__()
        self.setWindowTitle("Pyegi Selection Editor")
        screen = None
        if parent_window is not None and parent_window.windowHandle() is not None:
            screen = parent_window.windowHandle().screen()

        if screen is None:
            from PyQt6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen()

        if screen is not None:
            geo = screen.availableGeometry()
            self.resize(
                max(880, int(geo.width() * 0.82)),
                max(620, int(geo.height() * 0.82)),
            )
        else:
            self.resize(1200, 800)

        self.line = line
        self.output_mode = output_mode
        self.clip_mode = clip_mode
        self.initial_tool = initial_tool
        self.contour_simplify = contour_simplify
        self.global_feather = global_feather

        self.accepted = False
        self.selection_result = None

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.canvas = CanvasWidget(frame_bgr, initial_mask=initial_mask)
        self.canvas.set_tool(initial_tool)
        self.canvas.statusChanged.connect(self._set_status)
        self.canvas.selectionChanged.connect(self._sync_stats)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        splitter = QSplitter()
        outer.addWidget(splitter)

        splitter.addWidget(self.canvas)

        sidebar = QWidget()
        sidebar.setMinimumWidth(260)
        sidebar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(8, 8, 8, 8)
        side.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(sidebar)
        scroll.setMinimumWidth(280)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        splitter.addWidget(scroll)
        splitter.setSizes([1000, 220])

        info_box = QGroupBox("Info")
        info_form = QFormLayout(info_box)
        line_text = getattr(line, "raw_text", "")
        if len(line_text) > 60:
            line_text = line_text[:57] + "..."
        self.line_label = QLabel(line_text or "(empty line)")
        self.selection_info = QLabel("0 px selected")
        info_form.addRow("Line", self.line_label)
        info_form.addRow("Selection", self.selection_info)
        side.addWidget(info_box)

        tool_box = QGroupBox("Tools")
        tool_form = QFormLayout(tool_box)
        self.tool_combo = QComboBox()
        self.tool_combo.addItems([
            "Magic Wand",
            "Rectangular Marquee",
            "Elliptical Marquee",
            "Polygon Lasso",
            "Freehand Lasso",
            "Brush Add",
            "Brush Subtract",
        ])
        self.tool_combo.setCurrentText(initial_tool)
        self.tool_combo.currentTextChanged.connect(self.canvas.set_tool)
        tool_form.addRow("Active", self.tool_combo)
        side.addWidget(tool_box)

        wand_box = QGroupBox("Magic Wand")
        wand_form = QFormLayout(wand_box)
        self.wand_tol = QSpinBox()
        self.wand_tol.setRange(0, 442)
        self.wand_tol.setValue(self.canvas.wand_options.tolerance)
        self.wand_tol.valueChanged.connect(self._wand_changed)

        self.wand_contig = QCheckBox("Contiguous only")
        self.wand_contig.setChecked(self.canvas.wand_options.contiguous)
        self.wand_contig.stateChanged.connect(self._wand_changed)

        self.wand_diag = QCheckBox("Diagonal connectivity")
        self.wand_diag.setChecked(self.canvas.wand_options.diagonal)
        self.wand_diag.stateChanged.connect(self._wand_changed)

        self.wand_island = QSpinBox()
        self.wand_island.setRange(0, 100000)
        self.wand_island.setValue(self.canvas.wand_options.min_island_area)
        self.wand_island.valueChanged.connect(self._wand_changed)

        self.wand_feather = QSpinBox()
        self.wand_feather.setRange(0, 50)
        self.wand_feather.setValue(self.canvas.wand_options.feather_radius)
        self.wand_feather.valueChanged.connect(self._wand_changed)

        wand_form.addRow("Tolerance", self.wand_tol)
        wand_form.addRow("", self.wand_contig)
        wand_form.addRow("", self.wand_diag)
        wand_form.addRow("Min island area", self.wand_island)
        wand_form.addRow("Feather radius", self.wand_feather)
        side.addWidget(wand_box)

        brush_box = QGroupBox("Brush")
        brush_form = QFormLayout(brush_box)
        self.brush_radius = QSpinBox()
        self.brush_radius.setRange(1, 500)
        self.brush_radius.setValue(self.canvas.brush_options.radius)
        self.brush_radius.valueChanged.connect(self._brush_changed)
        brush_form.addRow("Radius", self.brush_radius)
        side.addWidget(brush_box)

        morph_box = QGroupBox("Selection Ops")
        morph_form = QFormLayout(morph_box)

        self.morph_radius = QSpinBox()
        self.morph_radius.setRange(1, 200)
        self.morph_radius.setValue(2)

        self.btn_expand = QPushButton("Expand")
        self.btn_contract = QPushButton("Contract")
        self.btn_fill = QPushButton("Fill Islands")

        self.btn_expand.clicked.connect(lambda: self.canvas.expand_selection(int(self.morph_radius.value())))
        self.btn_contract.clicked.connect(lambda: self.canvas.contract_selection(int(self.morph_radius.value())))
        self.btn_fill.clicked.connect(self.canvas.fill_islands)

        morph_form.addRow("Radius", self.morph_radius)
        morph_buttons = QVBoxLayout()
        morph_buttons.setSpacing(6)
        morph_buttons.addWidget(self.btn_expand)
        morph_buttons.addWidget(self.btn_contract)
        morph_buttons.addWidget(self.btn_fill)

        morph_form.addRow("Radius", self.morph_radius)
        morph_form.addRow(morph_buttons)

        side.addWidget(morph_box)

        export_box = QGroupBox("Export")
        export_form = QFormLayout(export_box)

        self.export_simplify = QDoubleSpinBox()
        self.export_simplify.setRange(0.1, 20.0)
        self.export_simplify.setDecimals(2)
        self.export_simplify.setSingleStep(0.25)
        self.export_simplify.setValue(contour_simplify)

        self.export_feather = QSpinBox()
        self.export_feather.setRange(0, 50)
        self.export_feather.setValue(global_feather)

        export_form.addRow("Contour simplify", self.export_simplify)
        export_form.addRow("Global feather", self.export_feather)

        side.addWidget(export_box)

        shortcuts_box = QGroupBox("Shortcuts")
        shortcuts_layout = QVBoxLayout(shortcuts_box)

        self.shortcuts_label = QLabel(
            "W: wand\n"
            "M: rectangular marquee\n"
            "U: elliptical marquee\n"
            "L: polygon lasso\n"
            "F: freehand lasso\n"
            "B: brush add\n"
            "E: brush subtract\n"
            "Shift: add\n"
            "Alt: subtract\n"
            "Ctrl+D: clear selection\n"
            "Ctrl+Z: undo\n"
            "Ctrl+Y / Ctrl+Shift+Z: redo\n"
            "Enter: accept\n"
            "Right-click polygon lasso: finish polygon\n"
            "Ctrl+wheel: zoom\n"
            "Space+drag or middle-drag: pan\n"
            "[ / ]: brush size\n"
            "- / =: wand tolerance\n"
            "Alt+]: expand selection\n"
            "Alt+[: contract selection\n"
            "Ctrl+Shift+F: fill islands\n"
            "0: fit view"
        )
        self.shortcuts_label.setWordWrap(True)
        self.shortcuts_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.shortcuts_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)

        shortcuts_layout.addWidget(self.shortcuts_label)
        side.addWidget(shortcuts_box)

        button_panel = QVBoxLayout()
        button_panel.setSpacing(6)

        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        self.btn_undo.clicked.connect(self.canvas.undo)
        self.btn_redo.clicked.connect(self.canvas.redo)
        self.btn_clear = QPushButton("Clear")
        self.btn_finish_poly = QPushButton("Finish Polygon")
        self.btn_accept = QPushButton("Accept")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_clear.clicked.connect(self.canvas.clear_selection)
        self.btn_finish_poly.clicked.connect(self.canvas.finish_polygon)
        self.btn_accept.clicked.connect(self.accept_result)
        self.btn_cancel.clicked.connect(self.close)
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self.canvas.fit_image_to_widget)
        buttons_row1 = QHBoxLayout()
        buttons_row1.setSpacing(6)
        buttons_row1.addWidget(self.btn_undo)
        buttons_row1.addWidget(self.btn_redo)
        buttons_row1.addWidget(self.btn_fit)
        buttons_row1.addWidget(self.btn_clear)

        buttons_row2 = QHBoxLayout()
        buttons_row2.setSpacing(6)
        buttons_row2.addWidget(self.btn_finish_poly)
        buttons_row2.addWidget(self.btn_accept)
        buttons_row2.addWidget(self.btn_cancel)

        button_panel.addLayout(buttons_row1)
        button_panel.addLayout(buttons_row2)

        side.addLayout(button_panel)

        for btn in [
            self.btn_expand,
            self.btn_contract,
            self.btn_fill,
            self.btn_undo,
            self.btn_redo,
            self.btn_fit,
            self.btn_clear,
            self.btn_finish_poly,
            self.btn_accept,
            self.btn_cancel,
        ]:
            btn.setMinimumHeight(32)
            btn.setMinimumWidth(90)
        
        for w in [
            self.tool_combo,
            self.wand_tol,
            self.wand_island,
            self.wand_feather,
            self.brush_radius,
            self.morph_radius,
        ]:
            w.setMinimumHeight(28)
        
        for box in [
            info_box,
            tool_box,
            wand_box,
            brush_box,
            morph_box,
        ]:
            box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        shortcuts_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        side.addStretch(1)

        self.status_label = QLabel("Ready")
        side.addWidget(self.status_label)

        QShortcut(QKeySequence("W"), self, activated=lambda: self.tool_combo.setCurrentText("Magic Wand"))
        QShortcut(QKeySequence("M"), self, activated=lambda: self.tool_combo.setCurrentText("Rectangular Marquee"))
        QShortcut(QKeySequence("U"), self, activated=lambda: self.tool_combo.setCurrentText("Elliptical Marquee"))
        QShortcut(QKeySequence("L"), self, activated=lambda: self.tool_combo.setCurrentText("Polygon Lasso"))
        QShortcut(QKeySequence("F"), self, activated=lambda: self.tool_combo.setCurrentText("Freehand Lasso"))
        QShortcut(QKeySequence("B"), self, activated=lambda: self.tool_combo.setCurrentText("Brush Add"))
        QShortcut(QKeySequence("E"), self, activated=lambda: self.tool_combo.setCurrentText("Brush Subtract"))
        QShortcut(QKeySequence("0"), self, activated=self.canvas.fit_image_to_widget)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.canvas.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.canvas.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.canvas.redo)
        QShortcut(QKeySequence("Ctrl+D"), self, activated=self.canvas.clear_selection)
        QShortcut(QKeySequence("Return"), self, activated=self.accept_result)
        QShortcut(QKeySequence("Enter"), self, activated=self.accept_result)
        QShortcut(
            QKeySequence("Alt+]"),
            self,
            activated=lambda: self.canvas.expand_selection(int(self.morph_radius.value()))
        )
        QShortcut(
            QKeySequence("Alt+["),
            self,
            activated=lambda: self.canvas.contract_selection(int(self.morph_radius.value()))
        )

        QShortcut(QKeySequence("Ctrl+Shift+F"), self, activated=self.canvas.fill_islands)

        self._sync_stats()

        if parent_window is not None:
            try:
                screen = parent_window.windowHandle().screen() if parent_window.windowHandle() else None
                if screen is not None:
                    geo = screen.availableGeometry()
                    self.move(
                        geo.x() + (geo.width() - self.width()) // 2,
                        geo.y() + (geo.height() - self.height()) // 2,
                    )
            except Exception:
                pass

    def _set_status(self, text: str):
        self.status_label.setText(text)
        self.brush_radius.blockSignals(True)
        self.brush_radius.setValue(self.canvas.brush_options.radius)
        self.brush_radius.blockSignals(False)

        self.wand_tol.blockSignals(True)
        self.wand_tol.setValue(self.canvas.wand_options.tolerance)
        self.wand_tol.blockSignals(False)

    def _sync_stats(self):
        if np.any(self.canvas.selection_mask):
            self.status_label.setText("Loaded existing selection from line")
        count = int(np.count_nonzero(self.canvas.selection_mask))
        self.selection_info.setText("%d px selected" % count)

    def _wand_changed(self):
        self.canvas.wand_options.tolerance = int(self.wand_tol.value())
        self.canvas.wand_options.contiguous = bool(self.wand_contig.isChecked())
        self.canvas.wand_options.diagonal = bool(self.wand_diag.isChecked())
        self.canvas.wand_options.min_island_area = int(self.wand_island.value())
        self.canvas.wand_options.feather_radius = int(self.wand_feather.value())

    def _brush_changed(self):
        self.canvas.brush_options.radius = int(self.brush_radius.value())

    def accept_result(self):
        mask = self.canvas.selection_mask.copy()
        global_feather = int(self.export_feather.value())
        if global_feather > 0:
            mask = feather_mask(mask, global_feather)

        if not np.any(mask):
            QMessageBox.warning(self, "No selection", "There is no selection to export.")
            return

        avg_bgr = average_color_under_mask(self.canvas.frame_bgr, mask)
        ass_path = ass_path_from_mask(mask, float(self.export_simplify.value()))
        if not ass_path:
            QMessageBox.warning(self, "No vector path", "The selection could not be converted into an ASS path.")
            return

        self.selection_result = SelectionResult(
            mask=mask,
            average_bgr=avg_bgr,
            ass_path=ass_path,
        )
        self.accepted = True
        self.close()

# ------------------------------ Main flow ------------------------------


if __name__ == "__main__":
    app = QApplication([])
    apply_dark_theme(app)

    io = Ass(Pyegi.get_input_file_path(), extended=True)
    meta, styles, lines = io.get_data()

    if not lines:
        raise RuntimeError("No subtitle lines found.")

    playresx = int(getattr(meta, "play_res_x", 1280))
    playresy = int(getattr(meta, "play_res_y", 720))

    settings = SettingsWindow()
    settings.show()
    app.exec()

    if not settings.accepted:
        raise SystemExit
    
    for line in lines:
        preview_rgb = get_preview_frame_for_line(
            line,
            settings.frame_source,
            playresx,
            playresy,
        )

        if preview_rgb is None:
            QMessageBox.critical(
                None,
                "Selection Tool",
                "Could not load the requested preview frame from the project video."
            )
            raise SystemExit
        
        initial_mask = build_initial_mask_for_line(
            line,
            settings.output_mode,
            settings.clip_mode,
            preview_rgb.shape[1],
            preview_rgb.shape[0],
        )

        editor = SelectionWindow(
            frame_rgb=preview_rgb,
            line=line,
            output_mode=settings.output_mode,
            clip_mode=settings.clip_mode,
            initial_tool="Magic Wand",
            contour_simplify=settings.contour_simplify,
            global_feather=settings.global_feather,
            initial_mask=initial_mask,
            parent_window=settings,
        )
        editor.show()
        app.exec()

        if not editor.accepted:
            raise SystemExit

        result = editor.selection_result
        if result is None or not np.any(result.mask):
            QMessageBox.critical(None, "Selection Tool", "No area was selected.")
            raise SystemExit

        l = line.copy()

        if settings.output_mode == "clip":
            replaced = replace_matching_vector_clip(
                line.raw_text,
                settings.clip_mode,
                result.ass_path,
            )

            if replaced is not None:
                l.text = replaced
            else:
                clip_tag = make_vector_clip_tag(
                    result.ass_path,
                    inverse=(settings.clip_mode == "iclip"),
                )
                l.text = insert_tag_into_first_override_block(line.raw_text, clip_tag)

        else:
            replaced = replace_generated_shape_line(
                line.raw_text,
                result.ass_path,
                result.average_bgr,
            )

            if replaced is not None:
                l.text = replaced
            else:
                color_tag = bgr_to_ass_bgr_tag(result.average_bgr)
                l.text = "{\\an7\\pos(0,0)\\bord0\\shad0\\1c%s\\p1}%s" % (
                    color_tag,
                    result.ass_path,
                )

        Pyegi.send_line(l)

    Pyegi.create_output_file()
