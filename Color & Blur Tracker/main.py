from __future__ import annotations
from PyegiInitializer import *
# -*- coding: utf-8 -*-

"""
Color & Blur Tracker

Created on Fri Sep 24 2021
Updated on Sat Apr 4  2026

@author: Drunk Simurgh and ChatGPT
"""

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import sys
from PyQt6 import QtCore, QtWidgets, QtGui

_app = QtWidgets.QApplication.instance()
if _app is None:
    _app = QtWidgets.QApplication(sys.argv)

from pyonfx import *
import Pyegi

from fractions import Fraction

try:
    from video_timestamps import VideoTimestamps, FPSTimestamps, RoundingMethod, TimeType
    _VIDEO_TIMESTAMPS_IMPORT_ERROR = None
except Exception as exc:
    VideoTimestamps = None
    FPSTimestamps = None
    RoundingMethod = None
    TimeType = None
    _VIDEO_TIMESTAMPS_IMPORT_ERROR = exc

# -----------------------------------------------------------------------------
# Pyegi bridge / placeholders
# -----------------------------------------------------------------------------
project_properties_table = Pyegi.get_project_properties()
PROJECT_VIDEO_PATH = project_properties_table.get("video_file")
FALLBACK_VIDEO_PATH = PROJECT_VIDEO_PATH

EXTERNAL_VIDEO_DEFAULT = ""
USE_PYEGI_INPUT_PATH = True

raw_video_position = project_properties_table.get("video_position")
try:
    PROJECT_VIDEO_POSITION = int(raw_video_position) if raw_video_position is not None else None
except Exception:
    PROJECT_VIDEO_POSITION = None

# -----------------------------------------------------------------------------
# UI data models
# -----------------------------------------------------------------------------
@dataclass
class SegmentationSettings:
    k_clusters: int = 5
    downsample_max_dim: int = 220
    gaussian_blur: int = 3
    min_component_area: int = 20
    morph_open: int = 1
    morph_close: int = 1

@dataclass
class TrackerSettings:
    track_color: bool = True
    track_blur: bool = True
    use_external_trimmed_video: bool = False
    external_trimmed_video_path: str = ""
    use_tracking_data: bool = False
    tracking_input_mode: str = "paste"
    tracking_text: str = ""
    tracking_file_path: str = ""
    use_clip_as_roi: bool = True
    semi_supervised: bool = False
    calculation_scope: str = "per_line"
    timeout_seconds: int = 0
    blur_gain: float = 0.08
    blur_gamma: float = 0.85
    color_change_threshold: float = 10.0
    blur_change_threshold: float = 0.35
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    cluster_size_tolerance_ratio: float = 0.35
    remove_clip_from_output: bool = True
    use_video_position_as_reference: bool = False
    project_video_position: Optional[int] = None
    static_tracking: bool = False
    motion_tracked_segment: bool = False
    track_alpha: bool = False
    alpha_inner_erode: int = 1
    alpha_bg_ring: int = 6
    alpha_min_channel_sep: float = 18.0
    alpha_max_residual: float = 18.0
    alpha_stable_color_threshold: float = 12.0

@dataclass
class TrackingFrame:
    frame: int
    x: Optional[float] = None
    y: Optional[float] = None
    scale_x: Optional[float] = None
    scale_y: Optional[float] = None
    rotation: Optional[float] = None

@dataclass
class TrackingData:
    fps: float = 23.976
    width: Optional[int] = None
    height: Optional[int] = None
    frames: List[TrackingFrame] = field(default_factory=list)

@dataclass
class AnalysisSample:
    start_ms: int
    end_ms: int
    primary_color: Optional[str] = None
    blur_value: Optional[float] = None
    alpha_value: Optional[int] = None
    confidence: float = 1.0
    tracking_frame_index: Optional[int] = None

@dataclass
class ReferenceTarget:
    mean_lab: np.ndarray
    std_lab: np.ndarray
    mean_bgr: np.ndarray
    rel_cx: float
    rel_cy: float
    rel_area: float
    bbox_rel: Tuple[float, float, float, float]
    last_bbox_px: Optional[Tuple[int, int, int, int]] = None

@dataclass
class ClusterReference:
    mean_bgr: np.ndarray
    mean_lab: np.ndarray
    pixel_count: int

@dataclass
class SegmentMetrics:
    pixel_count: int
    perimeter: float
    bbox_area: int
    bbox_w: int
    bbox_h: int
    aspect_ratio: float
    fill_ratio: float
    centroid_x: float
    centroid_y: float
    mean_bgr: np.ndarray

@dataclass
class TrackingTransformReference:
    ref_x: float
    ref_y: float
    ref_scale_x: float
    ref_scale_y: float
    ref_rotation: float
    base_pos_x: float
    base_pos_y: float
    base_fscx: float
    base_fscy: float
    base_frz: float

@dataclass
class AlphaReference:
    fg_bgr: np.ndarray
    ref_inner_mask_full: np.ndarray

# -----------------------------------------------------------------------------
# Qt UI
# -----------------------------------------------------------------------------
class UserCancelledSelection(Exception):
    pass

class SegmentSelectionDialog(QtWidgets.QDialog):
    def __init__(self, roi_bgr: np.ndarray, seg: SegmentationSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select target segment")
        self.roi_bgr = roi_bgr
        self.seg = seg
        self.candidates = build_candidate_segments(roi_bgr, seg)
        self.selected_index = 0
        self.setMinimumSize(900, 700)
        self._build_ui()
        self._apply_theme()
        self._refresh_preview()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "Choose the actual text/shape mask. White = selected segment, black = everything else."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.preview = QtWidgets.QLabel()
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(520)
        layout.addWidget(self.preview, 1)

        row = QtWidgets.QHBoxLayout()

        self.combo = QtWidgets.QComboBox()
        for i, mask in enumerate(self.candidates):
            area = int((mask > 0).sum())
            self.combo.addItem(f"Segment {i + 1} — area {area}", i)
        self.combo.currentIndexChanged.connect(self._on_combo_changed)

        row.addWidget(QtWidgets.QLabel("Segment"))
        row.addWidget(self.combo, 1)
        layout.addLayout(row)

        self.status_label = QtWidgets.QLabel("Choose a segment and click OK.")
        layout.addWidget(self.status_label)

        self.metric_label = QtWidgets.QLabel("Color: —    Blur: —")
        layout.addWidget(self.metric_label)

        self.color_patch = QtWidgets.QLabel()
        self.color_patch.setFixedHeight(28)
        self.color_patch.setStyleSheet("background:#000; border:1px solid #666; border-radius:6px;")
        layout.addWidget(self.color_patch)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.cancel_button = self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self._on_ok_clicked)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _on_ok_clicked(self):
        self.selected_index = self.combo.currentIndex()
        self.combo.setEnabled(False)
        self.ok_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self._confirmed = True

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QDialog { background: #0f1117; color: #eef3ff; }
            QLabel { color: #eef3ff; font-size: 13px; }
            QComboBox {
                background: #0c0f15;
                color: #f5f8ff;
                border: 1px solid #343b4f;
                border-radius: 10px;
                padding: 8px;
            }
            QPushButton {
                background: #2f6fed;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 14px;
                font-weight: 600;
            }
            """
        )

    def _on_combo_changed(self, idx: int):
        self.selected_index = idx
        self._refresh_preview()

    def _refresh_preview(self):
        if not self.candidates:
            self.preview.setText("No candidate segments found.")
            return

        mask_small = self.candidates[self.selected_index]
        mask = upscale_mask(mask_small, self.roi_bgr.shape[:2])

        # left: original ROI with blue outline around chosen segment
        left = self.roi_bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, contours, -1, (255, 0, 0), 2)  # blue in BGR

        # right: black/white mask with same blue outline
        right = np.zeros_like(self.roi_bgr)
        right[mask > 0] = (255, 255, 255)
        cv2.drawContours(right, contours, -1, (255, 0, 0), 2)

        gap = np.full((left.shape[0], 16, 3), 24, dtype=np.uint8)
        vis = np.concatenate([left, gap, right], axis=1)

        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)

        self.preview.setPixmap(
            pix.scaled(
                self.preview.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview()

    def get_selected_mask(self) -> Optional[np.ndarray]:
        if not self.candidates:
            return None
        mask_small = self.candidates[self.selected_index]
        return upscale_mask(mask_small, self.roi_bgr.shape[:2])
    
    def set_processing_status(self, processed_count: int, total_frames: int):
        self.status_label.setText(
            f"Processing {processed_count}/{total_frames}"
        )

    def set_result_info(self, color_ass: Optional[str], blur_value: Optional[float]):
        color_text = color_ass if color_ass is not None else "—"
        blur_text = f"{blur_value:.3f}" if blur_value is not None else "—"
        self.metric_label.setText(f"Color: {color_text}    Blur: {blur_text}")

        if color_ass is not None:
            hex_part = color_ass.replace("&H", "").replace("&", "")
            if len(hex_part) == 6:
                b = int(hex_part[0:2], 16)
                g = int(hex_part[2:4], 16)
                r = int(hex_part[4:6], 16)
                self.color_patch.setStyleSheet(
                    f"background: rgb({r},{g},{b}); border:1px solid #aaa; border-radius:6px;"
                )

    def show_frame_result(self, roi_bgr: np.ndarray, chosen_mask: Optional[np.ndarray]):
        left = roi_bgr.copy()

        if chosen_mask is None:
            right = np.zeros_like(roi_bgr)
        else:
            contours, _ = cv2.findContours(chosen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(left, contours, -1, (255, 0, 0), 2)

            right = np.zeros_like(roi_bgr)
            right[chosen_mask > 0] = (255, 255, 255)
            cv2.drawContours(right, contours, -1, (255, 0, 0), 2)

        gap = np.full((left.shape[0], 16, 3), 24, dtype=np.uint8)
        vis = np.concatenate([left, gap, right], axis=1)

        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)

        self.preview.setPixmap(
            pix.scaled(
                self.preview.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

        QtWidgets.QApplication.processEvents()

class SetupDialog(QtWidgets.QDialog):
    def __init__(self, default_dir: str = "", parent=None):
        super().__init__(parent)
        self.default_dir = default_dir
        self.setWindowTitle("Pyegi Color / Blur Tracker")
        self.setMinimumWidth(760)
        self.resize(820, 720)
        self._build_ui()
        self._apply_theme()
        self._sync_widgets()

    def _build_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(10)

        title = QtWidgets.QLabel("Color / Blur Tracker")
        title.setObjectName("title")

        subtitle = QtWidgets.QLabel(
            "Analyze selected ASS lines using subtitle's video or an optional trimmed external video."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitle")

        header_card = QtWidgets.QFrame()
        header_card.setObjectName("headerCard")
        header_layout = QtWidgets.QVBoxLayout(header_card)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(4)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main.addWidget(header_card)

        # ------------------------------------------------------------------
        # Tracking / analysis mode
        # ------------------------------------------------------------------
        mode_box = QtWidgets.QGroupBox("Tracking")
        mode_layout = QtWidgets.QGridLayout(mode_box)
        mode_layout.setContentsMargins(12, 12, 12, 12)
        mode_layout.setHorizontalSpacing(10)
        mode_layout.setVerticalSpacing(8)

        self.cb_color = QtWidgets.QCheckBox("Track color")
        self.cb_color.setChecked(True)

        self.cb_blur = QtWidgets.QCheckBox("Track blur")
        self.cb_blur.setChecked(True)

        self.cb_clip = QtWidgets.QCheckBox("Use \\clip / \\iclip as ROI when present")
        self.cb_clip.setChecked(True)

        self.cb_semisup = QtWidgets.QCheckBox("Semi-supervised target selection")
        self.cb_semisup.setChecked(True)

        self.scope_combo = QtWidgets.QComboBox()
        self.scope_combo.addItem("Do one new calculation for each selected line", "per_line")
        self.scope_combo.addItem("Do one calculation and apply it to all selected lines", "all_selected_lines")

        self.cb_use_video_position_ref = QtWidgets.QCheckBox(
            "Use current video position as reference frame"
        )
        self.cb_use_video_position_ref.setChecked(False)
        self.cb_use_video_position_ref.setToolTip(
            "Use current absolute video frame as the reference frame for segmentation and tracking anchoring."
        )

        self.cb_track_alpha = QtWidgets.QCheckBox("Track alpha (experimental)")
        self.cb_track_alpha.setChecked(False)
        self.cb_track_alpha.setToolTip(
            "Estimate fill alpha from the tracked segment using an inner mask and local background ring."
        )

        mode_layout.addWidget(self.cb_color, 0, 0)
        mode_layout.addWidget(self.cb_blur, 0, 1)
        mode_layout.addWidget(self.cb_clip, 1, 0, 1, 2)
        mode_layout.addWidget(self.cb_semisup, 2, 0, 1, 2)
        mode_layout.addWidget(QtWidgets.QLabel("Calculation scope"), 3, 0)
        mode_layout.addWidget(self.scope_combo, 3, 1)
        mode_layout.addWidget(self.cb_use_video_position_ref, 4, 0)
        mode_layout.addWidget(self.cb_track_alpha, 5, 0, 1, 2)

        main.addWidget(mode_box)

        # ------------------------------------------------------------------
        # Video source
        # ------------------------------------------------------------------
        video_box = QtWidgets.QGroupBox("Video source")
        video_layout = QtWidgets.QGridLayout(video_box)
        video_layout.setContentsMargins(12, 12, 12, 12)
        video_layout.setHorizontalSpacing(10)
        video_layout.setVerticalSpacing(8)

        self.cb_external_video = QtWidgets.QCheckBox(
            "Use external trimmed video instead of Pyegi's default video"
        )
        self.le_external_video = QtWidgets.QLineEdit(EXTERNAL_VIDEO_DEFAULT)
        self.btn_browse_video = QtWidgets.QPushButton("Browse…")

        video_layout.addWidget(self.cb_external_video, 0, 0, 1, 3)
        video_layout.addWidget(QtWidgets.QLabel("External video"), 1, 0)
        video_layout.addWidget(self.le_external_video, 1, 1)
        video_layout.addWidget(self.btn_browse_video, 1, 2)

        main.addWidget(video_box)

        # ------------------------------------------------------------------
        # Motion / tracking data
        # ------------------------------------------------------------------
        motion_box = QtWidgets.QGroupBox("Motion / Mocha data")
        motion_layout = QtWidgets.QGridLayout(motion_box)
        motion_layout.setContentsMargins(12, 12, 12, 12)
        motion_layout.setHorizontalSpacing(10)
        motion_layout.setVerticalSpacing(8)

        self.cb_use_tracking = QtWidgets.QCheckBox(
            "Use Aegisub-Motion compatible tracking data"
        )
        self.rb_paste = QtWidgets.QRadioButton("Paste tracking data")
        self.rb_file = QtWidgets.QRadioButton("Load tracking data from file")
        self.rb_paste.setChecked(True)

        self.te_tracking = QtWidgets.QPlainTextEdit()
        self.te_tracking.setPlaceholderText("Paste Adobe After Effects 6.0 Keyframe Data here…")
        self.te_tracking.setMinimumHeight(100)
        self.te_tracking.setMaximumHeight(140)

        self.le_tracking_file = QtWidgets.QLineEdit()
        self.btn_browse_tracking = QtWidgets.QPushButton("Browse…")

        self.cb_remove_clip = QtWidgets.QCheckBox("Remove \\clip / \\iclip from output")
        self.cb_remove_clip.setChecked(True)
        self.cb_remove_clip.setToolTip("Remove clip tags from generated output lines.")

        self.cb_static_tracking = QtWidgets.QCheckBox(
            "Static tracking from reference segment"
        )
        self.cb_static_tracking.setChecked(False)
        self.cb_static_tracking.setToolTip(
            "Use the chosen reference-frame segment unchanged for all frames."
        )

        self.cb_motion_tracked_segment = QtWidgets.QCheckBox(
            "Track reference segment using motion data"
        )
        self.cb_motion_tracked_segment.setChecked(False)
        self.cb_motion_tracked_segment.setToolTip(
            "Transform the chosen reference-frame segment using the loaded motion tracking data."
        )

        self.cb_static_tracking.toggled.connect(self._on_static_tracking_toggled)
        self.cb_motion_tracked_segment.toggled.connect(self._on_motion_segment_toggled)
        self.cb_use_tracking.toggled.connect(self._on_use_tracking_toggled)
        
        self.cb_static_tracking.toggled.connect(self._on_static_tracking_toggled)
        self.cb_motion_tracked_segment.toggled.connect(self._on_motion_segment_toggled)
        self.cb_use_tracking.toggled.connect(self._on_use_tracking_toggled)

        motion_layout.addWidget(self.cb_use_tracking, 0, 0, 1, 3)
        motion_layout.addWidget(self.rb_paste, 1, 0)
        motion_layout.addWidget(self.rb_file, 1, 1)
        motion_layout.addWidget(QtWidgets.QLabel("Tracking text"), 2, 0)
        motion_layout.addWidget(self.te_tracking, 2, 1, 2, 2)
        motion_layout.addWidget(QtWidgets.QLabel("Tracking file"), 4, 0)
        motion_layout.addWidget(self.le_tracking_file, 4, 1)
        motion_layout.addWidget(self.btn_browse_tracking, 4, 2)
        motion_layout.addWidget(self.cb_static_tracking, 5, 0, 1, 2)
        motion_layout.addWidget(self.cb_motion_tracked_segment, 6, 0, 1, 2)
        motion_layout.addWidget(self.cb_remove_clip, 7, 0)
        
        main.addWidget(motion_box)

        # ------------------------------------------------------------------
        # Advanced options toggle
        # ------------------------------------------------------------------
        self.cb_show_advanced = QtWidgets.QCheckBox("Show advanced options")
        self.cb_show_advanced.setChecked(False)
        main.addWidget(self.cb_show_advanced)

        self.adv_box = QtWidgets.QGroupBox("Advanced")
        self.adv_box.setObjectName("advancedBox")

        adv_outer = QtWidgets.QVBoxLayout(self.adv_box)
        adv_outer.setContentsMargins(12, 10, 12, 12)
        adv_outer.setSpacing(8)

        # compact two-column grid for advanced controls
        self.adv_grid = QtWidgets.QGridLayout()
        self.adv_grid.setHorizontalSpacing(10)
        self.adv_grid.setVerticalSpacing(6)
        adv_outer.addLayout(self.adv_grid)

        self.sb_timeout = QtWidgets.QSpinBox()
        self.sb_timeout.setRange(0, 60 * 60)
        self.sb_timeout.setSuffix(" s")
        self.sb_timeout.setValue(0)
        self.sb_timeout.setToolTip("Soft timeout for the whole processing run. 0 disables the timeout.")

        self.dsb_color_threshold = QtWidgets.QDoubleSpinBox()
        self.dsb_color_threshold.setRange(0.0, 200.0)
        self.dsb_color_threshold.setValue(10.0)
        self.dsb_color_threshold.setDecimals(2)
        self.dsb_color_threshold.setToolTip("Neighboring samples with smaller color difference will be merged.")

        self.dsb_blur_threshold = QtWidgets.QDoubleSpinBox()
        self.dsb_blur_threshold.setRange(0.0, 50.0)
        self.dsb_blur_threshold.setValue(0.35)
        self.dsb_blur_threshold.setDecimals(3)
        self.dsb_blur_threshold.setToolTip("Neighboring samples with smaller blur difference will be merged.")

        self.sb_k_clusters = QtWidgets.QSpinBox()
        self.sb_k_clusters.setRange(2, 12)
        self.sb_k_clusters.setValue(5)
        self.sb_k_clusters.setToolTip("Number of K-means clusters used for segmentation candidates.")

        self.sb_downsample = QtWidgets.QSpinBox()
        self.sb_downsample.setRange(32, 800)
        self.sb_downsample.setValue(220)
        self.sb_downsample.setToolTip("Maximum ROI size used during segmentation preprocessing.")

        self.sb_gaussian = QtWidgets.QSpinBox()
        self.sb_gaussian.setRange(0, 25)
        self.sb_gaussian.setValue(3)
        self.sb_gaussian.setToolTip("Gaussian blur kernel size used before segmentation. 0 disables it.")

        self.sb_min_area = QtWidgets.QSpinBox()
        self.sb_min_area.setRange(1, 50000)
        self.sb_min_area.setValue(20)
        self.sb_min_area.setToolTip("Ignore connected components smaller than this area.")

        self.sb_morph_open = QtWidgets.QSpinBox()
        self.sb_morph_open.setRange(0, 10)
        self.sb_morph_open.setValue(1)
        self.sb_morph_open.setToolTip("How many opening passes to apply to segmentation masks.")

        self.sb_morph_close = QtWidgets.QSpinBox()
        self.sb_morph_close.setRange(0, 10)
        self.sb_morph_close.setValue(1)
        self.sb_morph_close.setToolTip("How many closing passes to apply to segmentation masks.")

        self.dsb_cluster_tol = QtWidgets.QDoubleSpinBox()
        self.dsb_cluster_tol.setRange(0.01, 1.0)
        self.dsb_cluster_tol.setDecimals(3)
        self.dsb_cluster_tol.setValue(0.35)
        self.dsb_cluster_tol.setToolTip("How closely a cluster size must match the reference target size.")

        self.dsb_blur_gain = QtWidgets.QDoubleSpinBox()
        self.dsb_blur_gain.setRange(0.001, 10.0)
        self.dsb_blur_gain.setDecimals(4)
        self.dsb_blur_gain.setValue(0.04)
        self.dsb_blur_gain.setToolTip("Gain used in the heuristic mapping from measured blur to ASS blur.")

        self.dsb_blur_gamma = QtWidgets.QDoubleSpinBox()
        self.dsb_blur_gamma.setRange(0.05, 3.0)
        self.dsb_blur_gamma.setDecimals(3)
        self.dsb_blur_gamma.setValue(1.3)
        self.dsb_blur_gamma.setToolTip("Gamma used in the heuristic mapping from measured blur to ASS blur.")

        self.sb_alpha_inner_erode = QtWidgets.QSpinBox()
        self.sb_alpha_inner_erode.setRange(0, 20)
        self.sb_alpha_inner_erode.setValue(1)
        self.sb_alpha_inner_erode.setToolTip(
            "Erode the tracked mask inward by this many pixels before estimating alpha."
        )

        self.sb_alpha_bg_ring = QtWidgets.QSpinBox()
        self.sb_alpha_bg_ring.setRange(1, 50)
        self.sb_alpha_bg_ring.setValue(6)
        self.sb_alpha_bg_ring.setToolTip(
            "Thickness in pixels of the local background ring around the inner mask."
        )

        self.dsb_alpha_min_channel_sep = QtWidgets.QDoubleSpinBox()
        self.dsb_alpha_min_channel_sep.setRange(1.0, 255.0)
        self.dsb_alpha_min_channel_sep.setDecimals(1)
        self.dsb_alpha_min_channel_sep.setValue(18.0)
        self.dsb_alpha_min_channel_sep.setToolTip(
            "Minimum foreground/background channel difference required to use that channel for alpha solving."
        )

        self.dsb_alpha_max_residual = QtWidgets.QDoubleSpinBox()
        self.dsb_alpha_max_residual.setRange(1.0, 255.0)
        self.dsb_alpha_max_residual.setDecimals(1)
        self.dsb_alpha_max_residual.setValue(18.0)
        self.dsb_alpha_max_residual.setToolTip(
            "Maximum allowed residual between observed and predicted color for alpha estimation."
        )

        row = 0
        self._add_advanced_row(row, "Timeout", self.sb_timeout, "", None); row += 1
        self._add_advanced_row(row, "Color merge threshold", self.dsb_color_threshold, "Blur merge threshold", self.dsb_blur_threshold); row += 1
        self._add_advanced_row(row, "Segmentation K", self.sb_k_clusters, "Segmentation max dim", self.sb_downsample); row += 1
        self._add_advanced_row(row, "Gaussian blur", self.sb_gaussian, "Min component area", self.sb_min_area); row += 1
        self._add_advanced_row(row, "Morph open", self.sb_morph_open, "Morph close", self.sb_morph_close); row += 1
        self._add_advanced_row(row, "Cluster size tolerance", self.dsb_cluster_tol, "ASS blur gain", self.dsb_blur_gain); row += 1
        self._add_advanced_row(row, "ASS blur gamma", self.dsb_blur_gamma, "", None); row += 1
        self._add_advanced_row(row, "Alpha inner erode", self.sb_alpha_inner_erode, "Alpha bg ring", self.sb_alpha_bg_ring); row += 1
        self._add_advanced_row(row, "Alpha min channel sep", self.dsb_alpha_min_channel_sep, "Alpha max residual", self.dsb_alpha_max_residual); row += 1


        main.addWidget(self.adv_box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        main.addWidget(buttons)

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.cb_use_tracking.toggled.connect(self._sync_widgets)
        self.cb_external_video.toggled.connect(self._sync_widgets)
        self.rb_paste.toggled.connect(self._sync_widgets)
        self.cb_show_advanced.toggled.connect(self._toggle_advanced)
        self.btn_browse_video.clicked.connect(self._browse_video)
        self.btn_browse_tracking.clicked.connect(self._browse_tracking)

        self._toggle_advanced(False)

    def _on_use_tracking_toggled(self, checked: bool):
        if checked:
            self.cb_motion_tracked_segment.setEnabled(True)
            if not self.cb_static_tracking.isChecked() and not self.cb_motion_tracked_segment.isChecked():
                self.cb_motion_tracked_segment.setChecked(True)
        else:
            self.cb_motion_tracked_segment.setChecked(False)
            self.cb_motion_tracked_segment.setEnabled(False)

        self._sync_widgets()

    def _on_static_tracking_toggled(self, checked: bool):
        if checked and self.cb_motion_tracked_segment.isChecked():
            self.cb_motion_tracked_segment.blockSignals(True)
            self.cb_motion_tracked_segment.setChecked(False)
            self.cb_motion_tracked_segment.blockSignals(False)

    def _on_motion_segment_toggled(self, checked: bool):
        if checked and self.cb_static_tracking.isChecked():
            self.cb_static_tracking.blockSignals(True)
            self.cb_static_tracking.setChecked(False)
            self.cb_static_tracking.blockSignals(False)

    def _add_advanced_row(self, row: int, left_label: str, left_widget, right_label: str, right_widget):
        left_lbl = QtWidgets.QLabel(left_label)
        left_lbl.setProperty("formLabel", True)
        self.adv_grid.addWidget(left_lbl, row, 0)
        self.adv_grid.addWidget(left_widget, row, 1)

        if right_widget is not None:
            right_lbl = QtWidgets.QLabel(right_label)
            right_lbl.setProperty("formLabel", True)
            self.adv_grid.addWidget(right_lbl, row, 2)
            self.adv_grid.addWidget(right_widget, row, 3)

    def _toggle_advanced(self, checked: bool):
        self.adv_box.setVisible(checked)
        self.adjustSize()

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QDialog {
                background: #0b1020;
                color: #eaf1ff;
            }

            QFrame#headerCard {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #151d35,
                    stop:1 #101726
                );
                border: 1px solid #27314a;
                border-radius: 16px;
            }

            QGroupBox {
                border: 1px solid #27314a;
                border-radius: 14px;
                margin-top: 10px;
                padding: 10px 10px 10px 10px;
                font-weight: 600;
                color: #f5f8ff;
                background: #121a2b;
            }

            QGroupBox#advancedBox {
                background: #101726;
                border: 1px solid #22304a;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #ffffff;
            }

            QLabel {
                color: #eaf1ff;
                font-size: 12px;
            }

            QLabel[formLabel="true"] {
                color: #b8c5e2;
                font-size: 12px;
            }

            QLabel#title {
                font-size: 21px;
                font-weight: 700;
                color: #ffffff;
            }

            QLabel#subtitle {
                color: #aebcd9;
                font-size: 12px;
            }

            QCheckBox, QRadioButton {
                color: #eef4ff;
                font-size: 12px;
                spacing: 6px;
            }

            QCheckBox::indicator, QRadioButton::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #7f93ba;
                background: #0f1524;
                border-radius: 7px;
            }

            QRadioButton::indicator {
                border-radius: 7px;
            }

            QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                background: #4f8cff;
                border: 1px solid #9ec0ff;
            }

            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #0d1322;
                color: #f6f9ff;
                border: 1px solid #32415f;
                border-radius: 9px;
                padding: 6px 8px;
                min-height: 18px;
                selection-background-color: #4f8cff;
                selection-color: white;
            }

            QPlainTextEdit {
                padding-top: 8px;
                padding-bottom: 8px;
            }

            QComboBox::drop-down {
                border: none;
                width: 20px;
            }

            QAbstractSpinBox {
                color: #f6f9ff;
            }

            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4f8cff,
                    stop:1 #3f73e6
                );
                color: white;
                border: none;
                border-radius: 9px;
                padding: 7px 12px;
                font-size: 12px;
                font-weight: 600;
                min-height: 18px;
            }

            QPushButton:hover {
                background: #5d97ff;
            }

            QPushButton:disabled {
                background: #243149;
                color: #8d9bb7;
            }

            QDialogButtonBox QPushButton {
                min-width: 96px;
            }

            QToolTip {
                background-color: #162038;
                color: #eef4ff;
                border: 1px solid #4a618e;
                padding: 6px;
            }
            """
        )

    def _browse_video(self):
        start_dir = self.default_dir or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose trimmed video",
            start_dir,
            "Video Files (*.mkv *.mp4 *.avi *.mov *.webm);;All Files (*)",
        )
        if path:
            self.le_external_video.setText(path)
            self.cb_external_video.setChecked(True)

    def _browse_tracking(self):
        start_dir = self.default_dir or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose tracking data file",
            start_dir,
            "Text Files (*.txt *.log *.data);;All Files (*)",
        )
        if path:
            self.le_tracking_file.setText(path)
            self.rb_file.setChecked(True)
            try:
                self.te_tracking.setPlainText(
                    Path(path).read_text(encoding="utf-8", errors="replace")
                )
            except Exception:
                pass

    def _sync_widgets(self):
        use_tracking = self.cb_use_tracking.isChecked()
        use_file = self.rb_file.isChecked()
        use_external = self.cb_external_video.isChecked()

        self.rb_paste.setEnabled(use_tracking)
        self.rb_file.setEnabled(use_tracking)
        self.te_tracking.setEnabled(use_tracking and not use_file)
        self.le_tracking_file.setEnabled(use_tracking and use_file)
        self.btn_browse_tracking.setEnabled(use_tracking and use_file)

        self.le_external_video.setEnabled(use_external)
        self.btn_browse_video.setEnabled(use_external)

        self.cb_motion_tracked_segment.setEnabled(use_tracking)
        if not use_tracking:
            self.cb_motion_tracked_segment.setChecked(False)

        self._toggle_advanced(self.cb_show_advanced.isChecked())
        
    def get_settings(self) -> TrackerSettings:
        return TrackerSettings(
            track_color=self.cb_color.isChecked(),
            track_blur=self.cb_blur.isChecked(),
            use_external_trimmed_video=self.cb_external_video.isChecked(),
            external_trimmed_video_path=self.le_external_video.text().strip(),
            use_tracking_data=self.cb_use_tracking.isChecked(),
            tracking_input_mode="file" if self.rb_file.isChecked() else "paste",
            tracking_text=self.te_tracking.toPlainText(),
            tracking_file_path=self.le_tracking_file.text().strip(),
            use_clip_as_roi=self.cb_clip.isChecked(),
            semi_supervised=self.cb_semisup.isChecked(),
            calculation_scope=self.scope_combo.currentData(),
            timeout_seconds=self.sb_timeout.value(),
            blur_gain=self.dsb_blur_gain.value(),
            blur_gamma=self.dsb_blur_gamma.value(),
            color_change_threshold=self.dsb_color_threshold.value(),
            blur_change_threshold=self.dsb_blur_threshold.value(),
            segmentation=SegmentationSettings(
                k_clusters=self.sb_k_clusters.value(),
                downsample_max_dim=self.sb_downsample.value(),
                gaussian_blur=self.sb_gaussian.value(),
                min_component_area=self.sb_min_area.value(),
                morph_open=self.sb_morph_open.value(),
                morph_close=self.sb_morph_close.value(),
            ),
            cluster_size_tolerance_ratio=self.dsb_cluster_tol.value(),
            remove_clip_from_output=self.cb_remove_clip.isChecked(),
            use_video_position_as_reference=self.cb_use_video_position_ref.isChecked(),
            project_video_position=PROJECT_VIDEO_POSITION,
            static_tracking=self.cb_static_tracking.isChecked(),
            motion_tracked_segment=self.cb_motion_tracked_segment.isChecked(),
            track_alpha=self.cb_track_alpha.isChecked(),
            alpha_inner_erode=self.sb_alpha_inner_erode.value(),
            alpha_bg_ring=self.sb_alpha_bg_ring.value(),
            alpha_min_channel_sep=self.dsb_alpha_min_channel_sep.value(),
            alpha_max_residual=self.dsb_alpha_max_residual.value(),
        )

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def build_alpha_reference(
    reference_frame_bgr: np.ndarray,
    reference_full_mask: np.ndarray,
    settings: TrackerSettings,
) -> Optional[AlphaReference]:
    inner = erode_mask(reference_full_mask, settings.alpha_inner_erode)
    fg_bgr = mean_bgr_from_mask(reference_frame_bgr, inner)
    if fg_bgr is None:
        return None
    return AlphaReference(
        fg_bgr=fg_bgr,
        ref_inner_mask_full=inner,
    )

def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode((mask > 0).astype(np.uint8) * 255, kernel, iterations=iterations)

def make_background_ring(mask: np.ndarray, ring_px: int) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if ring_px <= 0:
        return np.zeros_like(mask_u8)
    kernel = np.ones((ring_px * 2 + 1, ring_px * 2 + 1), np.uint8)
    dil = cv2.dilate(mask_u8, kernel, iterations=1)
    ring = cv2.subtract(dil, mask_u8)
    return ring

def mean_bgr_from_mask(image_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    m = mask > 0
    if image_bgr.size == 0 or not np.any(m):
        return None
    return image_bgr[m].mean(axis=0).astype(np.float32)

def ass_alpha_from_opacity(opacity_0_1: float) -> int:
    opacity_0_1 = max(0.0, min(1.0, float(opacity_0_1)))
    return int(round((1.0 - opacity_0_1) * 255.0))

def estimate_alpha_from_composite(
    fg_bgr: np.ndarray,
    bg_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    min_channel_sep: float,
) -> Tuple[Optional[float], float]:
    """
    Solve C = a*F + (1-a)*B for scalar a per channel, then combine robustly.
    Returns:
        opacity (0..1) or None,
        residual error in BGR space
    """
    alphas = []
    for ch in range(3):
        denom = float(fg_bgr[ch] - bg_bgr[ch])
        if abs(denom) < min_channel_sep:
            continue
        a = float((cur_bgr[ch] - bg_bgr[ch]) / denom)
        alphas.append(a)

    if not alphas:
        return None, 1e9

    opacity = float(np.median(alphas))
    opacity = max(0.0, min(1.0, opacity))

    pred = opacity * fg_bgr + (1.0 - opacity) * bg_bgr
    residual = float(np.linalg.norm(cur_bgr - pred))
    return opacity, residual

def estimate_frame_alpha(
    frame_bgr: np.ndarray,
    current_full_mask: np.ndarray,
    alpha_ref: AlphaReference,
    settings: TrackerSettings,
) -> Tuple[Optional[int], float]:
    cur_inner = erode_mask(current_full_mask, settings.alpha_inner_erode)
    ring = make_background_ring(cur_inner, settings.alpha_bg_ring)

    cur_fg = mean_bgr_from_mask(frame_bgr, cur_inner)
    cur_bg = mean_bgr_from_mask(frame_bgr, ring)

    if cur_fg is None or cur_bg is None:
        return None, 0.0

    opacity, residual = estimate_alpha_from_composite(
        alpha_ref.fg_bgr,
        cur_bg,
        cur_fg,
        settings.alpha_min_channel_sep,
    )
    if opacity is None:
        return None, 0.0

    if residual > settings.alpha_max_residual:
        return None, 0.0

    ass_alpha = ass_alpha_from_opacity(opacity)
    confidence = max(0.0, min(1.0, 1.0 - residual / max(settings.alpha_max_residual, 1e-6)))
    return ass_alpha, confidence

def compute_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

def analyze_color_from_mask(roi_bgr: np.ndarray, mask: np.ndarray) -> Tuple[Optional[str], float]:
    mask_bool = mask > 0
    if roi_bgr.size == 0 or not np.any(mask_bool):
        return None, 0.0

    mean_bgr = roi_bgr[mask_bool].mean(axis=0)
    return ass_color_from_bgr(mean_bgr), 1.0

def analyze_blur_from_mask(roi_bgr: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = mask > 0
    if roi_bgr.size == 0 or not np.any(mask_bool):
        return 0.0

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    masked_vals = lap[mask_bool]
    if masked_vals.size == 0:
        return 0.0

    lap_var = float(masked_vals.var())
    blur_proxy = 150.0 / math.sqrt(lap_var + 1.0)
    return max(0.0, blur_proxy)

def transform_mask_with_tracking(
    reference_mask: np.ndarray,
    tracking_frame: TrackingFrame,
    tracking_ref: TrackingTransformReference,
    output_shape: Tuple[int, int],
) -> np.ndarray:
    cur_x = float(tracking_frame.x if tracking_frame.x is not None else tracking_ref.ref_x)
    cur_y = float(tracking_frame.y if tracking_frame.y is not None else tracking_ref.ref_y)

    cur_sx = float(tracking_frame.scale_x if tracking_frame.scale_x is not None else tracking_ref.ref_scale_x)
    cur_sy = float(tracking_frame.scale_y if tracking_frame.scale_y is not None else tracking_ref.ref_scale_y)
    cur_rot = float(tracking_frame.rotation if tracking_frame.rotation is not None else tracking_ref.ref_rotation)

    sx_ratio = (cur_sx / tracking_ref.ref_scale_x) if abs(tracking_ref.ref_scale_x) > 1e-9 else 1.0
    sy_ratio = (cur_sy / tracking_ref.ref_scale_y) if abs(tracking_ref.ref_scale_y) > 1e-9 else 1.0
    rot_diff = cur_rot - tracking_ref.ref_rotation

    rad = math.radians(-rot_diff)
    c = math.cos(rad)
    s = math.sin(rad)

    a00 = c * sx_ratio
    a01 = -s * sy_ratio
    a10 = s * sx_ratio
    a11 = c * sy_ratio

    tx = cur_x - a00 * tracking_ref.ref_x - a01 * tracking_ref.ref_y
    ty = cur_y - a10 * tracking_ref.ref_x - a11 * tracking_ref.ref_y

    M = np.array([[a00, a01, tx], [a10, a11, ty]], dtype=np.float32)

    warped = cv2.warpAffine(
        reference_mask,
        M,
        (output_shape[1], output_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return warped

def get_reference_local_index(
    frame_spans: List[Tuple[int, int, int]],
    settings: TrackerSettings,
) -> int:
    if (
        not settings.use_video_position_as_reference
        or settings.project_video_position is None
        or not frame_spans
    ):
        return 0

    target_abs = int(settings.project_video_position)

    for local_i, (abs_frame, _s_ms, _e_ms) in enumerate(frame_spans):
        if abs_frame == target_abs:
            return local_i

    # fallback to closest frame inside the line
    return min(
        range(len(frame_spans)),
        key=lambda i: abs(frame_spans[i][0] - target_abs)
    )

def normalize_filesystem_path(path_str: str, base_dir: Optional[Path] = None) -> str:
    if not path_str:
        return ""

    p = Path(path_str)

    # If the path is already absolute, normalize it.
    if p.is_absolute():
        try:
            return str(p.resolve(strict=False))
        except Exception:
            return str(p)

    # Otherwise resolve it relative to the provided base dir.
    if base_dir is None:
        base_dir = Path.cwd()

    try:
        return str((base_dir / p).resolve(strict=False))
    except Exception:
        return str(base_dir / p)


def get_path_resolution_base(input_path: Optional[str] = None) -> Path:
    if input_path:
        try:
            return Path(input_path).resolve(strict=False).parent
        except Exception:
            return Path(input_path).parent

    if PROJECT_VIDEO_PATH:
        try:
            return Path(PROJECT_VIDEO_PATH).resolve(strict=False).parent
        except Exception:
            return Path(PROJECT_VIDEO_PATH).parent

    return Path.cwd()

def choose_timestamp_video_path(settings: TrackerSettings, input_path: Optional[str] = None) -> str:
    """
    Timestamp source should prefer the original project video, because subtitle
    line times are in that timeline. External trimmed video is only for analysis
    frames unless the original project video path is unavailable.
    """
    base_dir = get_path_resolution_base(input_path)

    if PROJECT_VIDEO_PATH:
        return normalize_filesystem_path(PROJECT_VIDEO_PATH, base_dir)

    if settings.use_external_trimmed_video and settings.external_trimmed_video_path:
        return normalize_filesystem_path(settings.external_trimmed_video_path, base_dir)

    return choose_video_path(settings, input_path)


def guess_rounding_method_from_path(video_path: str):
    """
    VideoTimestamps docs note that mkv timestamps are typically rounded, while
    PTS are generally floored for other formats.
    """
    suffix = Path(video_path).suffix.lower()
    if suffix == ".mkv":
        return RoundingMethod.ROUND
    return RoundingMethod.FLOOR


def build_precise_timestamps(video_path: str):
    if not video_path:
        return None

    if VideoTimestamps is None:
        raise RuntimeError(
            "video_timestamps is required for precise frame timing but could not be imported: "
            f"{_VIDEO_TIMESTAMPS_IMPORT_ERROR}"
        )

    # Primary path: exact timestamps from the actual video file
    try:
        return VideoTimestamps.from_video_file(Path(video_path))
    except Exception:
        pass

    # Library-based fallback: CFR timestamps via FPSTimestamps
    # Still better than hand-rolled ms<->fps math because it keeps all timing
    # conversion logic inside the same timestamps library.
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for timestamp fallback: {video_path}")

        fps_value = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps_value <= 0:
            raise RuntimeError(f"Invalid FPS for timestamp fallback: {video_path}")

        fps_fraction = Fraction(str(fps_value)).limit_denominator(1_000_000)
        rounding = guess_rounding_method_from_path(video_path)
        return FPSTimestamps(rounding, Fraction(1000), fps_fraction)
    finally:
        cap.release()


def install_precise_meta_timestamps(meta, settings: TrackerSettings, input_path: Optional[str] = None):
    timestamp_video_path = choose_timestamp_video_path(settings, input_path)
    if not timestamp_video_path:
        return

    meta.timestamps = build_precise_timestamps(timestamp_video_path)

def extract_org(text: str) -> Optional[Tuple[float, float]]:
    m = re.search(r"\\org\(([-\d.]+)\s*,\s*([-\d.]+)\)", text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def with_org(text: str, x: float, y: float) -> str:
    return replace_or_insert_tag(text, "org", f"({x:.3f},{y:.3f})")


def transform_point_with_tracking(
    x: float,
    y: float,
    tracking_frame: TrackingFrame,
    tracking_ref: TrackingTransformReference,
) -> Tuple[float, float]:
    cur_x = float(tracking_frame.x if tracking_frame.x is not None else tracking_ref.ref_x)
    cur_y = float(tracking_frame.y if tracking_frame.y is not None else tracking_ref.ref_y)

    cur_sx = float(tracking_frame.scale_x if tracking_frame.scale_x is not None else tracking_ref.ref_scale_x)
    cur_sy = float(tracking_frame.scale_y if tracking_frame.scale_y is not None else tracking_ref.ref_scale_y)
    cur_rot = float(tracking_frame.rotation if tracking_frame.rotation is not None else tracking_ref.ref_rotation)

    sx_ratio = (cur_sx / tracking_ref.ref_scale_x) if abs(tracking_ref.ref_scale_x) > 1e-9 else 1.0
    sy_ratio = (cur_sy / tracking_ref.ref_scale_y) if abs(tracking_ref.ref_scale_y) > 1e-9 else 1.0
    rot_diff = cur_rot - tracking_ref.ref_rotation

    # Aegisub-Motion-style: transform the vector relative to the tracked start point
    dx = (x - tracking_ref.ref_x) * sx_ratio
    dy = (y - tracking_ref.ref_y) * sy_ratio

    # Equivalent to alpha - zRotationDiff in Aegisub-Motion
    rad = math.radians(-rot_diff)
    c = math.cos(rad)
    s = math.sin(rad)

    rdx = dx * c - dy * s
    rdy = dx * s + dy * c

    return cur_x + rdx, cur_y + rdy

def ms_to_frame_index(ms: int, fps: float) -> int:
    # CFR-only fallback.
    return max(0, int(round((ms / 1000.0) * fps)))

def get_line_frames(line, meta, video_fps: float) -> List[Tuple[int, int, int]]:
    """
    Returns:
        [(frame_index_in_original_timeline, start_ms, end_ms), ...]

    Preferred behavior:
    - If meta.timestamps is a VideoTimestamps/FPSTimestamps-like object, use it
      directly for exact frame boundaries.
    - Only fall back to FrameUtility/CFR math if precise timestamps are unavailable.
    """
    out: List[Tuple[int, int, int]] = []

    timestamps = getattr(meta, "timestamps", None)
    if (
        timestamps is not None
        and TimeType is not None
        and hasattr(timestamps, "time_to_frame")
        and hasattr(timestamps, "frame_to_time")
    ):
        try:
            line_start_ms = int(round(float(line.start_time)))

            frame_idx = timestamps.time_to_frame(line_start_ms, TimeType.START, 3)

            FU = FrameUtility(line.start_time, line.end_time, timestamps)

            for s, e, i, _ in FU:
                t1 = s
                t2 = e
                out.append((frame_idx + i - 1, t1, t2))

            if out:
                return out
        except Exception:
            pass

    # Secondary fallback: whatever FrameUtility can do
    try:
        fu_rows = list(FrameUtility(line.start_time, line.end_time, meta.timestamps))
        if fu_rows:
            first_s = int(round(float(fu_rows[0][0])))
            first_frame = ms_to_frame_index(first_s, video_fps)

            for local_i, (s, e, _i, _n) in enumerate(fu_rows):
                out.append((
                    first_frame + local_i,
                    int(round(float(s))),
                    int(round(float(e))),
                ))
            return out
    except Exception:
        pass

    # Final CFR fallback
    start_f = ms_to_frame_index(int(round(line.start_time)), video_fps)
    end_f = max(start_f + 1, ms_to_frame_index(int(round(line.end_time)), video_fps))

    for idx in range(start_f, end_f):
        s = int(round(idx * 1000.0 / video_fps))
        e = int(round((idx + 1) * 1000.0 / video_fps))
        out.append((idx, s, e))

    return out

def get_tracking_frame_index_for_span(
    local_index: int,
    total_spans: int,
    tracking: Optional[TrackingData],
) -> Optional[int]:
    if not tracking or not tracking.frames:
        return None

    if len(tracking.frames) == total_spans:
        return local_index

    if total_spans <= 1:
        return 0

    mapped = int(round(
        local_index * (len(tracking.frames) - 1) / max(1, total_spans - 1)
    ))
    return max(0, min(mapped, len(tracking.frames) - 1))


def get_tracking_frame_for_span(
    local_index: int,
    total_spans: int,
    tracking: Optional[TrackingData],
) -> Optional[TrackingFrame]:
    mapped = get_tracking_frame_index_for_span(local_index, total_spans, tracking)
    if mapped is None:
        return None
    return tracking.frames[mapped]

def remove_tag(text: str, tag: str) -> str:
    block_match = re.match(r"(\{[^}]*\})(.*)", text, flags=re.DOTALL)
    if not block_match:
        return text

    block, tail = block_match.groups()
    pattern = rf"\\{re.escape(tag)}(?:\([^)]*\)|[^\\}} ]+)"
    block = re.sub(pattern, "", block, count=1)

    # optional cleanup for repeated backslashes/spaces is minimal here
    return block + tail

def extract_pos(text: str) -> Optional[Tuple[float, float]]:
    m = re.search(r"\\pos\(([-\d.]+)\s*,\s*([-\d.]+)\)", text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

def extract_scalar_tag(text: str, tag: str) -> Optional[float]:
    m = re.search(rf"\\{re.escape(tag)}([-\d.]+)", text)
    if not m:
        return None
    return float(m.group(1))

def build_tracking_transform_reference(
    base_line,
    tracking: Optional[TrackingData],
    reference_tracking_index: int = 0,
) -> Optional[TrackingTransformReference]:
    if not tracking or not tracking.frames:
        return None

    reference_tracking_index = max(0, min(reference_tracking_index, len(tracking.frames) - 1))
    first = tracking.frames[reference_tracking_index]

    raw_text = getattr(base_line, "raw_text", "") or getattr(base_line, "text", "")

    pos = extract_pos(raw_text)
    if pos is not None:
        base_pos_x, base_pos_y = pos
    else:
        base_pos_x = float(getattr(base_line, "center", 0.0))
        base_pos_y = float(getattr(base_line, "middle", 0.0))

    base_fscx = extract_scalar_tag(raw_text, "fscx")
    if base_fscx is None:
        base_fscx = 100.0

    base_fscy = extract_scalar_tag(raw_text, "fscy")
    if base_fscy is None:
        base_fscy = 100.0

    base_frz = extract_scalar_tag(raw_text, "frz")
    if base_frz is None:
        base_frz = 0.0

    return TrackingTransformReference(
        ref_x=float(first.x if first.x is not None else 0.0),
        ref_y=float(first.y if first.y is not None else 0.0),
        ref_scale_x=float(first.scale_x if first.scale_x is not None else 100.0),
        ref_scale_y=float(first.scale_y if first.scale_y is not None else 100.0),
        ref_rotation=float(first.rotation if first.rotation is not None else 0.0),
        base_pos_x=base_pos_x,
        base_pos_y=base_pos_y,
        base_fscx=base_fscx,
        base_fscy=base_fscy,
        base_frz=base_frz,
    )

def map_raw_blur_to_ass(raw_blur: float, settings: TrackerSettings) -> float:
    if raw_blur <= 0:
        return 0.0
    return max(0.0, settings.blur_gain * (raw_blur ** settings.blur_gamma))

def remove_clip_tags(text: str) -> str:
    text = re.sub(r"\\clip\([^)]*\)", "", text)
    text = re.sub(r"\\iclip\([^)]*\)", "", text)
    return text

def get_cluster_reference_from_first_frame(
    line,
    meta,
    video: VideoSource,
    settings: TrackerSettings,
) -> Optional[ClusterReference]:
    frame_spans = get_line_frames(line, meta, video.fps)
    if not frame_spans:
        return None

    first_video_frame_index, _s_ms, _e_ms = frame_spans[0]
    mapped_frame = 0 if settings.use_external_trimmed_video else first_video_frame_index

    frame = video.read_frame_at_index(mapped_frame)
    x1, y1, x2, y2 = derive_roi(line, frame, settings.use_clip_as_roi)
    roi = frame[y1:y2, x1:x2]

    while True:
        dlg = SegmentSelectionDialog(roi, settings.segmentation)
        result = dlg.exec()

        if result == QtWidgets.QDialog.DialogCode.Accepted:
            mask = dlg.get_selected_mask()
            if mask is None:
                return None
            return cluster_reference_from_mask(roi, mask)

        raise UserCancelledSelection("User cancelled segment selection")

def ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    return app


def ass_color_from_bgr(bgr: Sequence[float]) -> str:
    b, g, r = [int(max(0, min(255, round(v)))) for v in bgr]
    return f"&H{b:02X}{g:02X}{r:02X}&"


def delta_e_like(c1_bgr: Sequence[float], c2_bgr: Sequence[float]) -> float:
    a = np.uint8([[list(c1_bgr)]])
    b = np.uint8([[list(c2_bgr)]])
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.linalg.norm(la - lb))


def parse_clip_rectangle(text: str) -> Optional[Tuple[int, int, int, int]]:
    m = re.search(
        r"\\clip\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
        text,
    )
    if not m:
        m = re.search(
            r"\\iclip\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
            text,
        )
    if not m:
        return None

    x1, y1, x2, y2 = [int(round(float(v))) for v in m.groups()]
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def replace_or_insert_tag(text: str, tag: str, value: str) -> str:
    block_match = re.match(r"(\{[^}]*\})(.*)", text, flags=re.DOTALL)
    if block_match:
        block, tail = block_match.groups()
    else:
        block, tail = "{}", text

    pattern = rf"\\{re.escape(tag)}(?:\([^)]*\)|[^\\}} ]+)"

    replacement = f"\\{tag}{value}"

    if re.search(pattern, block):
        block = re.sub(
            pattern,
            lambda m: replacement,
            block,
            count=1,
        )
    else:
        block = block[:-1] + replacement + "}"

    return block + tail

def with_pos(text: str, x: float, y: float) -> str:
    return replace_or_insert_tag(text, "pos", f"({x:.3f},{y:.3f})")


def with_scalar_tag(text: str, tag: str, number: float) -> str:
    val = f"{number:.3f}".rstrip("0").rstrip(".")
    return replace_or_insert_tag(text, tag, val)


def with_color_tag(text: str, tag: str, color_ass: str) -> str:
    return replace_or_insert_tag(text, tag, color_ass)


def choose_video_path(settings: TrackerSettings, input_path: Optional[str] = None) -> str:
    base_dir = get_path_resolution_base(input_path)

    if settings.use_external_trimmed_video and settings.external_trimmed_video_path:
        return normalize_filesystem_path(settings.external_trimmed_video_path, base_dir)

    if USE_PYEGI_INPUT_PATH:
        try:
            return normalize_filesystem_path(FALLBACK_VIDEO_PATH, base_dir)
        except Exception:
            pass

    return normalize_filesystem_path(FALLBACK_VIDEO_PATH, base_dir)


def soft_timeout(start_ts: float, timeout_seconds: int):
    if timeout_seconds > 0 and (time.monotonic() - start_ts) > timeout_seconds:
        raise TimeoutError(f"Timed out after {timeout_seconds} seconds")

def cluster_reference_from_mask(roi_bgr: np.ndarray, mask: np.ndarray) -> ClusterReference:
    mask_bool = mask > 0
    pixels_bgr = roi_bgr[mask_bool]
    mean_bgr = pixels_bgr.mean(axis=0).astype(np.float32)

    mean_patch = np.uint8([[mean_bgr]])
    mean_lab = cv2.cvtColor(mean_patch, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    return ClusterReference(
        mean_bgr=mean_bgr,
        mean_lab=mean_lab,
        pixel_count=int(mask_bool.sum()),
    )

def compute_segment_metrics(mask: np.ndarray, roi_bgr: np.ndarray) -> Optional[SegmentMetrics]:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    bbox_w = int(x2 - x1 + 1)
    bbox_h = int(y2 - y1 + 1)
    bbox_area = int(bbox_w * bbox_h)

    pixel_count = int((mask_u8 > 0).sum())
    fill_ratio = pixel_count / max(1, bbox_area)
    aspect_ratio = bbox_w / max(1, bbox_h)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0.0
    if contours:
        perimeter = float(max(cv2.arcLength(c, True) for c in contours))

    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    mean_bgr = roi_bgr[mask_u8 > 0].mean(axis=0).astype(np.float32)

    return SegmentMetrics(
        pixel_count=pixel_count,
        perimeter=perimeter,
        bbox_area=bbox_area,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        aspect_ratio=float(aspect_ratio),
        fill_ratio=float(fill_ratio),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        mean_bgr=mean_bgr,
    )

# -----------------------------------------------------------------------------
# Tracking data parsing
# -----------------------------------------------------------------------------
def set_processing_status(self, processed_count: int, total_frames: int):
    self.status_label.setText(
        f"Processing {processed_count}/{total_frames}"
    )

def set_result_info(self, color_ass: Optional[str], blur_value: Optional[float]):
    color_text = color_ass if color_ass is not None else "—"
    blur_text = f"{blur_value:.3f}" if blur_value is not None else "—"
    self.metric_label.setText(f"Color: {color_text}    Blur: {blur_text}")

    if color_ass is not None:
        hex_part = color_ass.replace("&H", "").replace("&", "")
        if len(hex_part) == 6:
            b = int(hex_part[0:2], 16)
            g = int(hex_part[2:4], 16)
            r = int(hex_part[4:6], 16)
            self.color_patch.setStyleSheet(
                f"background: rgb({r},{g},{b}); border:1px solid #aaa; border-radius:6px;"
            )

def show_frame_result(self, roi_bgr: np.ndarray, chosen_mask: Optional[np.ndarray]):
    left = roi_bgr.copy()

    if chosen_mask is None:
        right = np.zeros_like(roi_bgr)
    else:
        contours, _ = cv2.findContours(chosen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, contours, -1, (255, 0, 0), 2)

        right = np.zeros_like(roi_bgr)
        right[chosen_mask > 0] = (255, 255, 255)
        cv2.drawContours(right, contours, -1, (255, 0, 0), 2)

    gap = np.full((left.shape[0], 16, 3), 24, dtype=np.uint8)
    vis = np.concatenate([left, gap, right], axis=1)

    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
    pix = QtGui.QPixmap.fromImage(qimg)

    self.preview.setPixmap(
        pix.scaled(
            self.preview.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
    )

    QtWidgets.QApplication.processEvents()

def get_initial_selection_with_dialog(line, meta, video, settings):
    frame_spans = get_line_frames(line, meta, video.fps)
    if not frame_spans:
        return None, None, None, None, None, 0

    ref_local_index = get_reference_local_index(frame_spans, settings)
    ref_video_frame_index, _, _ = frame_spans[ref_local_index]

    if settings.use_external_trimmed_video:
        mapped_frame = map_external_trimmed_frame(ref_local_index, len(frame_spans), video)
    else:
        mapped_frame = ref_video_frame_index

    frame = video.read_frame_at_index(mapped_frame)

    x1, y1, x2, y2 = derive_roi(line, frame, settings.use_clip_as_roi)
    roi = frame[y1:y2, x1:x2]

    dlg = SegmentSelectionDialog(roi, settings.segmentation)
    dlg.show()

    while not getattr(dlg, "_confirmed", False):
        if not dlg.isVisible():
            return None, None, None, None, None, ref_local_index
        QtWidgets.QApplication.processEvents()

    mask = dlg.get_selected_mask()
    if mask is None:
        dlg.close()
        return None, None, None, None, None, ref_local_index

    ref_metrics = compute_segment_metrics(mask, roi)

    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask

    return dlg, ref_metrics, frame_spans, (x1, y1, x2, y2), full_mask, ref_local_index

def analyze_color_with_cluster_reference(
    roi_bgr: np.ndarray,
    ref: ClusterReference,
    k_clusters: int,
    tolerance_ratio: float,
) -> Tuple[Optional[str], float]:
    if roi_bgr.size == 0:
        return None, 0.0

    small = roi_bgr
    max_dim = max(roi_bgr.shape[:2])
    if max_dim > 140:
        scale = 140.0 / max_dim
        small = cv2.resize(
            roi_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        15,
        1.0,
    )
    _compact, labels, centers = cv2.kmeans(
        pixels,
        k_clusters,
        None,
        criteria,
        2,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(small.shape[:2])

    scale_area = (small.shape[0] * small.shape[1]) / float(roi_bgr.shape[0] * roi_bgr.shape[1])
    target_count = max(1.0, ref.pixel_count * scale_area)

    candidates = []
    for idx in range(centers.shape[0]):
        count = int((labels == idx).sum())
        size_ratio = abs(count - target_count) / max(target_count, 1.0)

        if size_ratio <= tolerance_ratio:
            center_lab = centers[idx].astype(np.float32)
            color_dist = float(np.linalg.norm(center_lab - ref.mean_lab))
            candidates.append((size_ratio, color_dist, idx, count))

    if not candidates:
        # fallback: pick by combined size + color closeness
        for idx in range(centers.shape[0]):
            count = int((labels == idx).sum())
            size_ratio = abs(count - target_count) / max(target_count, 1.0)
            center_lab = centers[idx].astype(np.float32)
            color_dist = float(np.linalg.norm(center_lab - ref.mean_lab))
            candidates.append((size_ratio, color_dist, idx, count))

    candidates.sort(key=lambda x: (x[0], x[1]))
    best_size_ratio, best_color_dist, best_idx, best_count = candidates[0]

    best_mask = labels == best_idx
    mean_bgr = small[best_mask].mean(axis=0)
    confidence = max(0.0, min(1.0, 1.0 - (best_size_ratio * 0.7 + best_color_dist / 80.0)))

    return ass_color_from_bgr(mean_bgr), confidence

def load_tracking_text(settings: TrackerSettings) -> str:
    if not settings.use_tracking_data:
        return ""
    if settings.tracking_input_mode == "file" and settings.tracking_file_path:
        return Path(settings.tracking_file_path).read_text(
            encoding="utf-8", errors="replace"
        )
    return settings.tracking_text or ""

def parse_ae_keyframe_data(text: str) -> TrackingData:
    if not text.strip():
        raise ValueError("Tracking data is empty")

    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    fps = 23.976
    width = None
    height = None

    m = re.search(r"Units Per Second\s+([0-9.]+)", text)
    if m:
        fps = float(m.group(1))

    m = re.search(r"Source Width\s+([0-9]+)", text)
    if m:
        width = int(m.group(1))

    m = re.search(r"Source Height\s+([0-9]+)", text)
    if m:
        height = int(m.group(1))

    def extract_section(section_name: str) -> List[str]:
        lines = text.splitlines()

        start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == section_name:
                start_idx = i
                break

        if start_idx is None:
            return []

        # skip the header line after the section name
        i = start_idx + 1
        while i < len(lines) and not lines[i].strip():
            i += 1

        # this should be the "Frame ..." header row
        if i < len(lines):
            i += 1

        rows = []
        while i < len(lines):
            stripped = lines[i].strip()

            if not stripped:
                break
            if stripped in ("Position", "Scale", "Rotation", "End of Keyframe Data"):
                break

            rows.append(lines[i])
            i += 1

        return rows

    frames: Dict[int, TrackingFrame] = {}

    position_rows = extract_section("Position")
    for line in position_rows:
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) >= 3 and parts[0].isdigit():
            fr = int(parts[0])
            item = frames.setdefault(fr, TrackingFrame(frame=fr))
            item.x = float(parts[1])
            item.y = float(parts[2])

    scale_rows = extract_section("Scale")
    for line in scale_rows:
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) >= 3 and parts[0].isdigit():
            fr = int(parts[0])
            item = frames.setdefault(fr, TrackingFrame(frame=fr))
            item.scale_x = float(parts[1])
            item.scale_y = float(parts[2])

    rotation_rows = extract_section("Rotation")
    for line in rotation_rows:
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) >= 2 and parts[0].isdigit():
            fr = int(parts[0])
            item = frames.setdefault(fr, TrackingFrame(frame=fr))
            item.rotation = float(parts[1])

    if not frames:
        raise ValueError("Could not parse any keyframe rows from tracking data")

    ordered = [frames[k] for k in sorted(frames)]
    return TrackingData(fps=fps, width=width, height=height, frames=ordered)

# -----------------------------------------------------------------------------
# Video helpers
# -----------------------------------------------------------------------------
class VideoSource:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 23.976
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame_at_index(self, frame_index: int) -> np.ndarray:
        frame_index = max(0, min(frame_index, max(0, self.frame_count - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {self.path}")
        return frame

    def release(self):
        self.cap.release()


def map_external_trimmed_frame(local_index: int, total_local: int, source: VideoSource) -> int:
    if total_local <= 1:
        return 0
    ratio = local_index / float(total_local - 1)
    return int(round(ratio * max(0, source.frame_count - 1)))


# -----------------------------------------------------------------------------
# ROI + analysis
# -----------------------------------------------------------------------------
def build_candidate_segments(
    roi_bgr: np.ndarray,
    seg,
) -> List[np.ndarray]:
    if roi_bgr.size == 0:
        return []

    small = roi_bgr
    h, w = roi_bgr.shape[:2]
    max_dim = max(h, w)
    scale = 1.0
    if max_dim > seg.downsample_max_dim:
        scale = seg.downsample_max_dim / float(max_dim)
        small = cv2.resize(
            roi_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    work = small
    if seg.gaussian_blur > 0:
        k = seg.gaussian_blur
        if k % 2 == 0:
            k += 1
        work = cv2.GaussianBlur(work, (k, k), 0)

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0,
    )
    _compact, labels, centers = cv2.kmeans(
        pixels,
        seg.k_clusters,
        None,
        criteria,
        2,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(work.shape[:2])

    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((3, 3), np.uint8)

    candidates = []

    for idx in range(centers.shape[0]):
        mask = (labels == idx).astype(np.uint8) * 255

        for _ in range(seg.morph_open):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        for _ in range(seg.morph_close):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for comp_id in range(1, num_labels):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < seg.min_component_area:
                continue
            comp_mask = np.where(cc_labels == comp_id, 255, 0).astype(np.uint8)
            candidates.append(comp_mask)

    # upscale back to roi size
    out = []
    for m in candidates:
        up = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(up)

    out.sort(key=lambda m: int((m > 0).sum()), reverse=True)
    return out[:24]

def segment_distance(ref: SegmentMetrics, cur: SegmentMetrics) -> float:
    size_term = abs(cur.pixel_count - ref.pixel_count) / max(1, ref.pixel_count)
    perim_term = abs(cur.perimeter - ref.perimeter) / max(1.0, ref.perimeter)
    bbox_term = abs(cur.bbox_area - ref.bbox_area) / max(1, ref.bbox_area)
    aspect_term = abs(cur.aspect_ratio - ref.aspect_ratio)
    fill_term = abs(cur.fill_ratio - ref.fill_ratio)
    centroid_term = math.hypot(cur.centroid_x - ref.centroid_x, cur.centroid_y - ref.centroid_y) / 100.0

    return (
        2.5 * size_term
        + 1.2 * perim_term
        + 1.5 * bbox_term
        + 1.0 * aspect_term
        + 1.0 * fill_term
        + 0.8 * centroid_term
    )

def choose_best_segment_by_metrics(
    roi_bgr: np.ndarray,
    candidate_masks: List[np.ndarray],
    ref_metrics: SegmentMetrics,
) -> Tuple[Optional[np.ndarray], Optional[SegmentMetrics], float]:
    best_mask = None
    best_metrics = None
    best_score = None

    for mask in candidate_masks:
        metrics = compute_segment_metrics(mask, roi_bgr)
        if metrics is None:
            continue
        score = segment_distance(ref_metrics, metrics)
        if best_score is None or score < best_score:
            best_score = score
            best_mask = mask
            best_metrics = metrics

    if best_mask is None or best_metrics is None:
        return None, None, 0.0

    confidence = max(0.0, min(1.0, 1.0 - best_score / 6.0))
    return best_mask, best_metrics, confidence

def upscale_mask(mask_small: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    th, tw = target_shape
    return cv2.resize(mask_small, (tw, th), interpolation=cv2.INTER_NEAREST)


def reference_target_from_mask(roi_bgr: np.ndarray, mask: np.ndarray) -> ReferenceTarget:
    mask_bool = mask > 0
    pixels_bgr = roi_bgr[mask_bool]
    pixels_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)[mask_bool]

    mean_lab = pixels_lab.mean(axis=0).astype(np.float32)
    std_lab = pixels_lab.std(axis=0).astype(np.float32) + 1.0
    mean_bgr = pixels_bgr.mean(axis=0).astype(np.float32)

    h, w = roi_bgr.shape[:2]
    ys, xs = np.where(mask_bool)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

    cx = xs.mean() / max(1, w - 1)
    cy = ys.mean() / max(1, h - 1)
    area = float(mask_bool.mean())

    return ReferenceTarget(
        mean_lab=mean_lab,
        std_lab=std_lab,
        mean_bgr=mean_bgr,
        rel_cx=float(cx),
        rel_cy=float(cy),
        rel_area=float(area),
        bbox_rel=(
            x1 / max(1, w - 1),
            y1 / max(1, h - 1),
            x2 / max(1, w - 1),
            y2 / max(1, h - 1),
        ),
        last_bbox_px=(int(x1), int(y1), int(x2), int(y2)),
    )


def analyze_color_with_reference(
    roi_bgr: np.ndarray,
    ref: ReferenceTarget,
) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
    if roi_bgr.size == 0:
        return None, 0.0, None

    h, w = roi_bgr.shape[:2]

    # local search window around previous match or initial bbox
    if ref.last_bbox_px is not None:
        bx1, by1, bx2, by2 = ref.last_bbox_px
    else:
        bx1 = int(ref.bbox_rel[0] * max(1, w - 1))
        by1 = int(ref.bbox_rel[1] * max(1, h - 1))
        bx2 = int(ref.bbox_rel[2] * max(1, w - 1))
        by2 = int(ref.bbox_rel[3] * max(1, h - 1))

    pad_x = max(6, int((bx2 - bx1) * 0.5))
    pad_y = max(6, int((by2 - by1) * 0.5))

    sx1 = max(0, bx1 - pad_x)
    sy1 = max(0, by1 - pad_y)
    sx2 = min(w, bx2 + pad_x)
    sy2 = min(h, by2 + pad_y)

    search = roi_bgr[sy1:sy2, sx1:sx2]
    if search.size == 0:
        return None, 0.0, None

    lab = cv2.cvtColor(search, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = (lab - ref.mean_lab.reshape(1, 1, 3)) / ref.std_lab.reshape(1, 1, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    mask = (dist < 2.6).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    best_comp = None
    best_score = None

    expected_area = max(1.0, ref.rel_area * search.shape[0] * search.shape[1])

    for comp_id in range(1, num_labels):
        area = stats[comp_id, cv2.CC_STAT_AREA]
        if area < 8:
            continue

        cx, cy = centroids[comp_id]
        area_pen = abs(area - expected_area) / max(expected_area, 1.0)
        pos_pen = math.hypot(
            cx - (bx1 + bx2) / 2 + sx1,
            cy - (by1 + by2) / 2 + sy1,
        ) / max(w, h)

        score = area_pen + pos_pen * 2.0
        if best_score is None or score < best_score:
            best_score = score
            best_comp = comp_id

    if best_comp is None:
        mean_bgr = search.reshape(-1, 3).mean(axis=0)
        return ass_color_from_bgr(mean_bgr), 0.1, None

    best_mask_local = (cc_labels == best_comp)
    ys, xs = np.where(best_mask_local)
    mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()

    # update reference for next frame
    ref.last_bbox_px = (
        int(sx1 + mx1),
        int(sy1 + my1),
        int(sx1 + mx2),
        int(sy1 + my2),
    )

    mean_bgr = search[best_mask_local].mean(axis=0)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[sy1:sy2, sx1:sx2][best_mask_local] = 255

    confidence = max(0.0, min(1.0, 1.0 - float(best_score)))
    return ass_color_from_bgr(mean_bgr), confidence, full_mask

def derive_roi(line, frame: np.ndarray, use_clip: bool) -> Tuple[int, int, int, int]:
    h, w = frame.shape[:2]

    if use_clip:
        clip = parse_clip_rectangle(getattr(line, "raw_text", ""))
        if clip is not None:
            x1, y1, x2, y2 = clip
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                return x1, y1, x2, y2

    left = int(max(0, math.floor(getattr(line, "left", 0))))
    top = int(max(0, math.floor(getattr(line, "top", 0))))
    right = int(min(w, math.ceil(getattr(line, "right", w))))
    bottom = int(min(h, math.ceil(getattr(line, "bottom", h))))

    if right <= left or bottom <= top:
        return 0, 0, w, h
    return left, top, right, bottom


def analyze_color(roi_bgr: np.ndarray) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
    if roi_bgr.size == 0:
        return None, 0.0, None

    small = roi_bgr
    max_dim = max(roi_bgr.shape[:2])
    if max_dim > 220:
        scale = 220.0 / max_dim
        small = cv2.resize(
            roi_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        25,
        1.0,
    )
    _compact, labels, centers = cv2.kmeans(
        pixels,
        4,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(small.shape[:2])

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grad = cv2.Laplacian(gray, cv2.CV_32F)

    best_score = -1e9
    best_idx = None

    for idx, center in enumerate(centers):
        mask = labels == idx
        area = float(mask.mean())
        if area < 0.005:
            continue
        edge_strength = float(np.mean(np.abs(grad[mask]))) if np.any(mask) else 0.0
        score = edge_strength * 1.5 - abs(area - 0.12) * 25.0
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        return None, 0.0, None

    mask = (labels == best_idx).astype(np.uint8) * 255
    center_lab = np.uint8([[centers[best_idx]]])
    center_bgr = cv2.cvtColor(center_lab, cv2.COLOR_LAB2BGR)[0, 0]
    color_ass = ass_color_from_bgr(center_bgr)
    confidence = max(0.0, min(1.0, (best_score + 20.0) / 40.0))
    return color_ass, confidence, mask


def analyze_blur(roi_bgr: np.ndarray) -> float:
    if roi_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_proxy = 150.0 / (math.sqrt(lap_var + 1.0))
    return max(0.0, blur_proxy)


# -----------------------------------------------------------------------------
# Line generation + modification
# -----------------------------------------------------------------------------
def clone_line(line):
    return line.copy() if hasattr(line, "copy") else line

def apply_tracking_to_text(
    text: str,
    base_line,
    tracking_frame: Optional[TrackingFrame],
    tracking_ref: Optional[TrackingTransformReference],
) -> str:
    if not tracking_frame or not tracking_ref:
        return text

    raw_text = getattr(base_line, "raw_text", "") or getattr(base_line, "text", "")

    # transform \pos with tracked-point-relative math
    base_pos = extract_pos(raw_text)
    if base_pos is not None:
        px, py = transform_point_with_tracking(
            base_pos[0], base_pos[1], tracking_frame, tracking_ref
        )
        text = with_pos(text, px, py)
    else:
        px, py = transform_point_with_tracking(
            tracking_ref.base_pos_x, tracking_ref.base_pos_y, tracking_frame, tracking_ref
        )
        text = with_pos(text, px, py)

    # transform \org too when present, otherwise rotations can look wrong
    base_org = extract_org(raw_text)
    if base_org is not None:
        ox, oy = transform_point_with_tracking(
            base_org[0], base_org[1], tracking_frame, tracking_ref
        )
        text = with_org(text, ox, oy)

    cur_sx = float(tracking_frame.scale_x if tracking_frame.scale_x is not None else tracking_ref.ref_scale_x)
    cur_sy = float(tracking_frame.scale_y if tracking_frame.scale_y is not None else tracking_ref.ref_scale_y)
    cur_rot = float(tracking_frame.rotation if tracking_frame.rotation is not None else tracking_ref.ref_rotation)

    new_fscx = (
        tracking_ref.base_fscx * (cur_sx / tracking_ref.ref_scale_x)
        if abs(tracking_ref.ref_scale_x) > 1e-9
        else tracking_ref.base_fscx
    )
    new_fscy = (
        tracking_ref.base_fscy * (cur_sy / tracking_ref.ref_scale_y)
        if abs(tracking_ref.ref_scale_y) > 1e-9
        else tracking_ref.base_fscy
    )
    new_frz = tracking_ref.base_frz + (cur_rot - tracking_ref.ref_rotation)

    if abs(new_fscx - 100.0) < 0.01:
        text = remove_tag(text, "fscx")
    else:
        text = with_scalar_tag(text, "fscx", new_fscx)

    if abs(new_fscy - 100.0) < 0.01:
        text = remove_tag(text, "fscy")
    else:
        text = with_scalar_tag(text, "fscy", new_fscy)

    if abs(new_frz) < 0.01:
        text = remove_tag(text, "frz")
    else:
        text = with_scalar_tag(text, "frz", new_frz)

    return text

def build_frame_lines_for_source_line(
    line,
    meta,
    video: VideoSource,
    settings: TrackerSettings,
    tracking: Optional[TrackingData],
    start_ts: float,
    ref_metrics=None,
    progress_dialog=None,
    reference_full_mask: Optional[np.ndarray] = None,
    reference_tracking_ref: Optional[TrackingTransformReference] = None,
    alpha_ref: Optional[AlphaReference] = None,
) -> List[AnalysisSample]:
    frame_spans = get_line_frames(line, meta, video.fps)
    results: List[AnalysisSample] = []

    for local_i, (video_frame_index, s_ms, e_ms) in enumerate(frame_spans):
        soft_timeout(start_ts, settings.timeout_seconds)

        if settings.use_external_trimmed_video:
            mapped_frame = map_external_trimmed_frame(local_i, len(frame_spans), video)
        else:
            mapped_frame = video_frame_index

        frame = video.read_frame_at_index(mapped_frame)
        x1, y1, x2, y2 = derive_roi(line, frame, settings.use_clip_as_roi)
        roi = frame[y1:y2, x1:x2]

        chosen_mask = None
        current_full_mask = None
        confidence = 1.0
        color = None
        blur = None
        alpha_value = None

        use_reference_mask_mode = (
            reference_full_mask is not None
            and (settings.static_tracking or settings.motion_tracked_segment)
        )

        if use_reference_mask_mode:
            if settings.motion_tracked_segment and tracking is not None and reference_tracking_ref is not None:
                current_tf = get_tracking_frame_for_span(local_i, len(frame_spans), tracking)
                if current_tf is not None:
                    current_full_mask = transform_mask_with_tracking(
                        reference_full_mask,
                        current_tf,
                        reference_tracking_ref,
                        frame.shape[:2],
                    )
                else:
                    current_full_mask = reference_full_mask.copy()
            else:
                current_full_mask = reference_full_mask.copy()

            chosen_mask = current_full_mask[y1:y2, x1:x2]

            if settings.track_color:
                color, confidence = analyze_color_from_mask(roi, chosen_mask)

            if settings.track_blur:
                raw_blur = analyze_blur_from_mask(roi, chosen_mask)
                blur = map_raw_blur_to_ass(raw_blur, settings)

        else:
            if settings.track_color:
                if settings.semi_supervised and ref_metrics is not None:
                    candidate_masks = build_candidate_segments(roi, settings.segmentation)
                    chosen_mask, chosen_metrics, confidence = choose_best_segment_by_metrics(
                        roi,
                        candidate_masks,
                        ref_metrics,
                    )
                    if chosen_metrics is not None:
                        color = ass_color_from_bgr(chosen_metrics.mean_bgr)
                        current_full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        current_full_mask[y1:y2, x1:x2] = chosen_mask
                else:
                    color, confidence, _mask = analyze_color(roi)
                    if _mask is not None:
                        chosen_mask = _mask
                        current_full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        current_full_mask[y1:y2, x1:x2] = chosen_mask

            if settings.track_blur:
                raw_blur = analyze_blur(roi)
                blur = map_raw_blur_to_ass(raw_blur, settings)

        if settings.track_alpha and current_full_mask is not None and alpha_ref is not None:
            alpha_value, alpha_conf = estimate_frame_alpha(
                frame,
                current_full_mask,
                alpha_ref,
                settings,
            )
            if alpha_value is not None:
                confidence = min(confidence, alpha_conf)

        if progress_dialog is not None:
            progress_dialog.set_processing_status(local_i + 1, len(frame_spans))
            progress_dialog.show_frame_result(roi, chosen_mask)
            progress_dialog.set_result_info(color, blur)

        tracking_frame_index = get_tracking_frame_index_for_span(
            local_i,
            len(frame_spans),
            tracking,
        )

        results.append(
            AnalysisSample(
                start_ms=s_ms,
                end_ms=e_ms,
                primary_color=color,
                blur_value=blur,
                alpha_value=alpha_value,
                confidence=confidence,
                tracking_frame_index=tracking_frame_index,
            )
        )

    if tracking is not None:
        return results
    return merge_neighboring_samples(results, settings)

def _ass_color_to_bgr_tuple(color_ass: str) -> Tuple[int, int, int]:
    hex_part = color_ass.replace("&H", "").replace("&", "")
    if len(hex_part) != 6:
        return 0, 0, 0
    return (
        int(hex_part[0:2], 16),
        int(hex_part[2:4], 16),
        int(hex_part[4:6], 16),
    )


def merge_neighboring_samples(
    samples: List[AnalysisSample],
    settings: TrackerSettings,
) -> List[AnalysisSample]:
    if not samples:
        return []

    merged = [samples[0]]

    for sample in samples[1:]:
        prev = merged[-1]

        same_color = True
        if settings.track_color:
            if prev.primary_color is None or sample.primary_color is None:
                same_color = prev.primary_color == sample.primary_color
            else:
                prev_bgr = _ass_color_to_bgr_tuple(prev.primary_color)
                samp_bgr = _ass_color_to_bgr_tuple(sample.primary_color)
                same_color = (
                    delta_e_like(prev_bgr, samp_bgr)
                    <= settings.color_change_threshold
                )

        same_blur = True
        if settings.track_blur:
            if prev.blur_value is None or sample.blur_value is None:
                same_blur = prev.blur_value == sample.blur_value
            else:
                same_blur = (
                    abs(prev.blur_value - sample.blur_value)
                    <= settings.blur_change_threshold
                )

        if same_color and same_blur:
            prev.end_ms = sample.end_ms
            prev.confidence = min(prev.confidence, sample.confidence)
        else:
            merged.append(sample)

    return merged


def with_alpha_tag(text: str, tag: str, alpha_byte: int) -> str:
    alpha_byte = max(0, min(255, int(round(alpha_byte))))
    return replace_or_insert_tag(text, tag, f"&H{alpha_byte:02X}&")

def create_output_line(
    base_line,
    sample: AnalysisSample,
    tracking_frame: Optional[TrackingFrame],
    settings: TrackerSettings,
    tracking_ref: Optional[TrackingTransformReference]
):
    l = clone_line(base_line)
    l.start_time = sample.start_ms
    l.end_time = sample.end_ms

    text = getattr(base_line, "raw_text", "")
    if settings.remove_clip_from_output:
        text = remove_clip_tags(text)

    text = apply_tracking_to_text(text, base_line, tracking_frame, tracking_ref)

    if sample.primary_color is not None:
        text = with_color_tag(text, "1c", sample.primary_color)
    if sample.blur_value is not None:
        text = with_scalar_tag(text, "blur", sample.blur_value)
    if sample.alpha_value is not None:
        text = with_alpha_tag(text, "alpha", sample.alpha_value)

    l.text = text
    return l

def emit_generated_lines(lines_out: Sequence):
    if hasattr(Pyegi, "send_line"):
        for l in lines_out:
            Pyegi.send_line(l)
        if hasattr(Pyegi, "create_output_file"):
            Pyegi.create_output_file()
        return

    io = Ass("in.ass")
    for l in lines_out:
        io.write_line(l)
    io.save()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def run_gui(default_dir: str) -> Optional[TrackerSettings]:
    ensure_qapp()
    dlg = SetupDialog(default_dir=default_dir)
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None

    settings = dlg.get_settings()
    if not settings.track_color and not settings.track_blur:
        raise ValueError("At least one of color or blur tracking must be enabled")
    return settings


def resolve_selected_lines(lines) -> List:
    return list(lines)

def process_group(
    source_lines: Sequence,
    meta,
    video: VideoSource,
    settings: TrackerSettings,
    tracking: Optional[TrackingData],
    start_ts: float,
) -> List:
    if not source_lines:
        return []

    all_output = []

    # Shared analysis for all selected lines
    if settings.calculation_scope == "all_selected_lines":
        ref_line = source_lines[0]

        dlg = None
        ref_metrics = None
        ref_frame_spans = None
        ref_full_mask = None
        ref_local_index = 0

        if settings.semi_supervised:
            (
                dlg,
                ref_metrics,
                ref_frame_spans,
                _roi_box,
                ref_full_mask,
                ref_local_index,
            ) = get_initial_selection_with_dialog(
                ref_line, meta, video, settings
            )
            if dlg is None or ref_metrics is None:
                return []
        else:
            ref_frame_spans = get_line_frames(ref_line, meta, video.fps)
            ref_local_index = get_reference_local_index(ref_frame_spans, settings)

        ref_tracking_index = get_tracking_frame_index_for_span(
            ref_local_index,
            len(ref_frame_spans) if ref_frame_spans else 0,
            tracking,
        )

        reference_tracking_ref = build_tracking_transform_reference(
            ref_line,
            tracking,
            0 if ref_tracking_index is None else ref_tracking_index,
        )

        alpha_ref = None
        if settings.track_alpha and ref_full_mask is not None:
            ref_video_frame_index, _, _ = ref_frame_spans[ref_local_index]
            if settings.use_external_trimmed_video:
                ref_mapped_frame = map_external_trimmed_frame(ref_local_index, len(ref_frame_spans), video)
            else:
                ref_mapped_frame = ref_video_frame_index
            ref_frame_bgr = video.read_frame_at_index(ref_mapped_frame)
            alpha_ref = build_alpha_reference(ref_frame_bgr, ref_full_mask, settings)

        shared_samples = build_frame_lines_for_source_line(
            ref_line,
            meta,
            video,
            settings,
            tracking,
            start_ts,
            ref_metrics=ref_metrics,
            progress_dialog=dlg,
            reference_full_mask=ref_full_mask,
            reference_tracking_ref=reference_tracking_ref,
            alpha_ref=alpha_ref,
        )

        if dlg is not None:
            dlg.status_label.setText("Finished.")
            dlg.cancel_button.setText("Close")
            dlg.cancel_button.setEnabled(True)
            QtWidgets.QApplication.processEvents()

        for line in source_lines:
            total_spans = len(shared_samples)

            shared_frame_spans = get_line_frames(line, meta, video.fps)
            shared_ref_local_index = get_reference_local_index(shared_frame_spans, settings)
            shared_ref_tracking_index = get_tracking_frame_index_for_span(
                shared_ref_local_index,
                len(shared_frame_spans),
                tracking,
            )

            line_tracking_ref = build_tracking_transform_reference(
                line,
                tracking,
                0 if shared_ref_tracking_index is None else shared_ref_tracking_index,
            )

            for idx, sample in enumerate(shared_samples):
                tf = None
                if tracking is not None and sample.tracking_frame_index is not None:
                    tf = tracking.frames[sample.tracking_frame_index]
                all_output.append(
                    create_output_line(line, sample, tf, settings, line_tracking_ref)
                )

        return all_output

    # Per-line analysis
    for line in source_lines:
        dlg = None
        ref_metrics = None
        frame_spans_for_ref = None
        ref_full_mask = None
        ref_local_index = 0

        if settings.semi_supervised:
            (
                dlg,
                ref_metrics,
                frame_spans_for_ref,
                _roi_box,
                ref_full_mask,
                ref_local_index,
            ) = get_initial_selection_with_dialog(
                line, meta, video, settings
            )
            if dlg is None or ref_metrics is None:
                continue
        else:
            frame_spans_for_ref = get_line_frames(line, meta, video.fps)
            ref_local_index = get_reference_local_index(frame_spans_for_ref, settings)

        ref_tracking_index = get_tracking_frame_index_for_span(
            ref_local_index,
            len(frame_spans_for_ref) if frame_spans_for_ref else 0,
            tracking,
        )

        reference_tracking_ref = build_tracking_transform_reference(
            line,
            tracking,
            0 if ref_tracking_index is None else ref_tracking_index,
        )

        alpha_ref = None
        if settings.track_alpha and ref_full_mask is not None:
            ref_video_frame_index, _, _ = frame_spans_for_ref[ref_local_index]
            if settings.use_external_trimmed_video:
                ref_mapped_frame = map_external_trimmed_frame(ref_local_index, len(frame_spans_for_ref), video)
            else:
                ref_mapped_frame = ref_video_frame_index
            ref_frame_bgr = video.read_frame_at_index(ref_mapped_frame)
            alpha_ref = build_alpha_reference(ref_frame_bgr, ref_full_mask, settings)

        samples = build_frame_lines_for_source_line(
            line,
            meta,
            video,
            settings,
            tracking,
            start_ts,
            ref_metrics=ref_metrics,
            progress_dialog=dlg,
            reference_full_mask=ref_full_mask,
            reference_tracking_ref=reference_tracking_ref,
            alpha_ref=alpha_ref,
        )

        if dlg is not None:
            dlg.status_label.setText("Finished.")
            dlg.cancel_button.setText("Close")
            dlg.cancel_button.setEnabled(True)
            QtWidgets.QApplication.processEvents()

        total_spans = len(samples)

        frame_spans = get_line_frames(line, meta, video.fps)
        ref_local_index = get_reference_local_index(frame_spans, settings)
        ref_tracking_index = get_tracking_frame_index_for_span(
            ref_local_index,
            len(frame_spans),
            tracking,
        )

        line_tracking_ref = build_tracking_transform_reference(
            line,
            tracking,
            0 if ref_tracking_index is None else ref_tracking_index,
        )

        for idx, sample in enumerate(samples):
            tf = get_tracking_frame_for_span(idx, total_spans, tracking)
            all_output.append(
                create_output_line(line, sample, tf, settings, line_tracking_ref)
            )

    return all_output

def main():
    input_path = (
        Pyegi.get_input_file_path()
        if (USE_PYEGI_INPUT_PATH and hasattr(Pyegi, "get_input_file_path"))
        else "in.ass"
    )

    io = Ass(input_path, extended=True)
    meta, styles, lines = io.get_data()
    selected_lines = resolve_selected_lines(lines)

    try:
        pyegi_video = (
            FALLBACK_VIDEO_PATH
            # Pyegi.get_video_file_path()
            # if hasattr(Pyegi, "get_video_file_path")
            # else ""
        )
        default_dir = str(Path(pyegi_video).parent) if pyegi_video else str(Path(input_path).parent)
    except Exception:
        default_dir = str(Path(input_path).parent)

    settings = run_gui(default_dir)
    if settings is None:
        return
    
    install_precise_meta_timestamps(meta, settings, input_path)

    tracking = None
    if settings.use_tracking_data:
        tracking_text = load_tracking_text(settings)
        tracking = parse_ae_keyframe_data(tracking_text)

    video_path = choose_video_path(settings, input_path)
    if not video_path:
        raise RuntimeError("No usable video path is available")

    start_ts = time.monotonic()
    video = VideoSource(video_path)
    try:
        output_lines = process_group(
            selected_lines,
            meta,
            video,
            settings,
            tracking,
            start_ts,
        )
    except UserCancelledSelection:
        return
    finally:
        video.release()

    emit_generated_lines(output_lines)


if __name__ == "__main__":
    main()
