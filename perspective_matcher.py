# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Seghier Mohamed Abdellatif and IngeTrazo contributors.
"""Perspective Matcher — Camera calibration and photo perspective matching for IngeTrazo.

Aligns IngeTrazo's 3D viewport camera to an architectural photograph, identical
in function to SketchUp's "Match New Photo" and Blender's "Perspective Plotter" / fSpy.

Features:
- Overlay reference photograph with adjustable opacity.
- Interactive vanishing lines:
    * 2 Red lines for the X axis -> Vanishing Point Vx.
    * 2 Green lines for the Y axis -> Vanishing Point Vy.
    * 2 Blue lines for the Z axis -> Vanishing Point Vz (in 3-point mode).
- Draggable Origin pin (0, 0, 0) with 3D coordinate axis triad gizmo.
- Horizon line visualization.
- Real-time projective geometry solver:
    * Solves camera focal length (mm, 35mm eq.) and Field of View (FOV).
    * Computes 3D camera rotation matrix (Pitch, Yaw, Roll).
    * Calculates camera position and eye distance relative to world origin.
- 2-Point and 3-Point perspective modes.
- Saves calibration and guide configuration inside the IngeTrazo document (.igz).
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QFont, QImage, QPainter, QPen,
    QPixmap, QVector3D
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QVBoxLayout, QWidget
)

from core.i18n import current_language


# ---- Internationalization Helper -------------------------------------------
_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "es": {
        "Perspective Match": "Ajuste de Perspectiva",
        "Match Photo": "Ajustar Foto",
        "Load Photograph…": "Cargar Fotografía…",
        "Clear Photo": "Quitar Foto",
        "Opacity:": "Opacidad:",
        "Perspective Mode:": "Modo de Perspectiva:",
        "2-Point (Verticals stay vertical)": "2 Puntos (Verticales rectas)",
        "3-Point (Tilted camera)": "3 Puntos (Cámara inclinada)",
        "Camera Distance (m):": "Distancia de Cámara (m):",
        "Show Reference Guides": "Mostrar Guías de Referencia",
        "Show Background Photo": "Mostrar Foto de Fondo",
        "Lock Viewport Camera": "Bloquear Cámara de Vista",
        "Align Camera Now": "Alinear Cámara Ahora",
        "Reset Default Guides": "Restablecer Guías",
        "Invert X Axis": "Invertir Eje X",
        "Invert Y Axis": "Invertir Eje Y",
        "Camera Calibration": "Calibración de Cámara",
        "Focal Length:": "Distancia Focal:",
        "Field of View (FOV):": "Campo de Visión (FOV):",
        "Camera Pitch / Yaw:": "Inclinación / Giro:",
        "Status:": "Estado:",
        "Ready": "Listo",
        "Lines converging properly": "Líneas convergiendo correctamente",
        "Lines nearly parallel (adjust guides)": "Líneas casi paralelas (ajustar guías)",
        "Invalid vanishing point configuration": "Configuración de puntos de fuga no válida",
    }
}


def _t(text: str) -> str:
    lang = current_language()
    return _TRANSLATIONS.get(lang, {}).get(text, text)


# ---- Mathematical Projective Geometry Solver -------------------------------
def intersect_lines_2d(
    p1: Tuple[float, float], p2: Tuple[float, float],
    p3: Tuple[float, float], p4: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """Calculates the 2D intersection point of line (p1, p2) and line (p3, p4).
    Returns (x, y) or None if lines are parallel."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    a1, b1, c1 = y1 - y2, x2 - x1, x1 * y2 - y1 * x2
    a2, b2, c2 = y3 - y4, x4 - x3, x3 * y4 - y3 * x4

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    return ((b1 * c2 - b2 * c1) / det, (c1 * a2 - c2 * a1) / det)


class PerspectiveMatchData:
    """Holds calibration state and relative guide coordinates in [0.0, 1.0]."""

    def __init__(self) -> None:
        self.image_path: str = ""
        self.image_opacity: float = 0.55
        self.show_image: bool = True
        self.show_guides: bool = True
        self.camera_locked: bool = False
        self.mode: str = "2point"  # "2point" or "3point"
        self.distance: float = 20.0  # meters from origin
        self.invert_x: bool = False
        self.invert_y: bool = False

        # Relative handle coordinates (rx, ry) in [0.0, 1.0] across viewport
        self.x1_a = [0.38, 0.44]
        self.x1_b = [0.18, 0.49]
        self.x2_a = [0.38, 0.65]
        self.x2_b = [0.18, 0.74]

        self.y1_a = [0.62, 0.44]
        self.y1_b = [0.82, 0.49]
        self.y2_a = [0.62, 0.65]
        self.y2_b = [0.82, 0.74]

        self.z1_a = [0.38, 0.25]
        self.z1_b = [0.38, 0.72]
        self.z2_a = [0.62, 0.25]
        self.z2_b = [0.62, 0.72]

        self.origin = [0.50, 0.68]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_opacity": self.image_opacity,
            "show_image": self.show_image,
            "show_guides": self.show_guides,
            "camera_locked": self.camera_locked,
            "mode": self.mode,
            "distance": self.distance,
            "invert_x": self.invert_x,
            "invert_y": self.invert_y,
            "x1_a": list(self.x1_a),
            "x1_b": list(self.x1_b),
            "x2_a": list(self.x2_a),
            "x2_b": list(self.x2_b),
            "y1_a": list(self.y1_a),
            "y1_b": list(self.y1_b),
            "y2_a": list(self.y2_a),
            "y2_b": list(self.y2_b),
            "z1_a": list(self.z1_a),
            "z1_b": list(self.z1_b),
            "z2_a": list(self.z2_a),
            "z2_b": list(self.z2_b),
            "origin": list(self.origin),
        }

    def from_dict(self, d: Dict[str, Any]) -> None:
        if not isinstance(d, dict):
            return
        self.image_path = str(d.get("image_path", self.image_path))
        self.image_opacity = float(d.get("image_opacity", self.image_opacity))
        self.show_image = bool(d.get("show_image", self.show_image))
        self.show_guides = bool(d.get("show_guides", self.show_guides))
        self.camera_locked = bool(d.get("camera_locked", self.camera_locked))
        self.mode = str(d.get("mode", self.mode))
        self.distance = float(d.get("distance", self.distance))
        self.invert_x = bool(d.get("invert_x", self.invert_x))
        self.invert_y = bool(d.get("invert_y", self.invert_y))

        for k in ("x1_a", "x1_b", "x2_a", "x2_b", "y1_a", "y1_b", "y2_a", "y2_b",
                  "z1_a", "z1_b", "z2_a", "z2_b", "origin"):
            if k in d and len(d[k]) == 2:
                setattr(self, k, [float(d[k][0]), float(d[k][1])])


class SolvedCameraParams:
    """Output results from the vanishing point perspective solver."""

    def __init__(self) -> None:
        self.valid: bool = False
        self.status_msg: str = "Ready"
        self.focal_px: float = 800.0
        self.focal_35mm: float = 35.0
        self.fov_deg: float = 45.0
        self.pitch_deg: float = 20.0
        self.yaw_deg: float = -45.0
        self.eye: QVector3D = QVector3D(15.0, 15.0, 10.0)
        self.target: QVector3D = QVector3D(0.0, 0.0, 0.0)
        self.forward: QVector3D = QVector3D(-0.7, -0.7, -0.3)
        self.vp_x: Optional[Tuple[float, float]] = None
        self.vp_y: Optional[Tuple[float, float]] = None
        self.vp_z: Optional[Tuple[float, float]] = None
        self.horizon: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None


def solve_perspective(
    data: PerspectiveMatchData,
    view_width: float,
    view_height: float
) -> SolvedCameraParams:
    """Solves focal length, FOV, camera rotation, and eye position from guides."""
    res = SolvedCameraParams()
    W, H = max(view_width, 100.0), max(view_height, 100.0)
    cx, cy = W / 2.0, H / 2.0

    # Convert relative handle coordinates to screen pixels
    def to_px(rel_pt: List[float]) -> Tuple[float, float]:
        return (rel_pt[0] * W, rel_pt[1] * H)

    p_x1_a, p_x1_b = to_px(data.x1_a), to_px(data.x1_b)
    p_x2_a, p_x2_b = to_px(data.x2_a), to_px(data.x2_b)

    p_y1_a, p_y1_b = to_px(data.y1_a), to_px(data.y1_b)
    p_y2_a, p_y2_b = to_px(data.y2_a), to_px(data.y2_b)

    # Calculate 2D vanishing points
    vp_x = intersect_lines_2d(p_x1_a, p_x1_b, p_x2_a, p_x2_b)
    vp_y = intersect_lines_2d(p_y1_a, p_y1_b, p_y2_a, p_y2_b)
    res.vp_x = vp_x
    res.vp_y = vp_y

    if vp_x is None or vp_y is None:
        res.status_msg = _t("Lines nearly parallel (adjust guides)")
        return res

    # 2D Screen to Camera coordinate offsets:
    # Screen has Y down; standard camera coordinate frame has Y up.
    du_x = vp_x[0] - cx
    dv_x = -(vp_x[1] - cy)
    du_y = vp_y[0] - cx
    dv_y = -(vp_y[1] - cy)

    # Vanishing Point Orthogonality condition:
    # dot(ray_x, ray_y) = du_x * du_y + dv_x * dv_y + f^2 = 0
    dot_prod = du_x * du_y + dv_x * dv_y

    if dot_prod >= -1.0:
        # Lines do not converge towards opposite sides of the optical center
        res.status_msg = _t("Invalid vanishing point configuration")
        f = (H / 2.0) / math.tan(math.radians(45.0) / 2.0)
    else:
        f = math.sqrt(-dot_prod)
        res.status_msg = _t("Lines converging properly")
        res.valid = True

    f = max(50.0, min(100000.0, f))
    res.focal_px = f

    # Vertical FOV in degrees (IngeTrazo's OrbitCamera uses vertical FOV)
    fov_deg = math.degrees(2.0 * math.atan((H / 2.0) / f))
    fov_deg = max(5.0, min(140.0, fov_deg))
    res.fov_deg = fov_deg

    # 35mm equivalent focal length (24mm sensor height)
    res.focal_35mm = 24.0 / (2.0 * math.tan(math.radians(fov_deg) / 2.0))

    # Construct 3D rays from camera center to vanishing points
    rx_x, rx_y, rx_z = du_x, dv_x, -f
    len_x = math.hypot(rx_x, rx_y, rx_z)
    vx = [rx_x / len_x, rx_y / len_x, rx_z / len_x]

    ry_x, ry_y, ry_z = du_y, dv_y, -f
    len_y = math.hypot(ry_x, ry_y, ry_z)
    vy = [ry_x / len_y, ry_y / len_y, ry_z / len_y]

    # Invert axis directions if requested
    if data.invert_x or du_x > 0:
        vx = [-vx[0], -vx[1], -vx[2]]
    if data.invert_y or du_y < 0:
        vy = [-vy[0], -vy[1], -vy[2]]

    # Gram-Schmidt orthonormalize vy against vx:
    dot_xy = vx[0] * vy[0] + vx[1] * vy[1] + vx[2] * vy[2]
    vy = [vy[0] - dot_xy * vx[0], vy[1] - dot_xy * vx[1], vy[2] - dot_xy * vx[2]]
    len_y2 = math.hypot(vy[0], vy[1], vy[2])
    if len_y2 > 1e-6:
        vy = [vy[0] / len_y2, vy[1] / len_y2, vy[2] / len_y2]

    # World Z axis: cross product vx x vy
    vz = [
        vx[1] * vy[2] - vx[2] * vy[1],
        vx[2] * vy[0] - vx[0] * vy[2],
        vx[0] * vy[1] - vx[1] * vy[0]
    ]
    len_z = math.hypot(vz[0], vz[1], vz[2])
    if len_z > 1e-6:
        vz = [vz[0] / len_z, vz[1] / len_z, vz[2] / len_z]

    # Ensure Z points upwards (positive Y in camera coordinate space)
    if vz[1] < 0:
        vz = [-vz[0], -vz[1], -vz[2]]
        vy = [
            vz[1] * vx[2] - vz[2] * vx[1],
            vz[2] * vx[0] - vz[0] * vx[2],
            vz[0] * vx[1] - vz[1] * vx[0]
        ]

    # World-to-Camera rotation matrix columns are vx, vy, vz:
    # R_wc = [vx, vy, vz].
    # Camera-to-World rotation is R_cw = R_wc^T.
    # Therefore, Camera Forward (-Z_cam) in world is -R_cw[:, 2] = -[vx[2], vy[2], vz[2]].
    fwd_x = -vx[2]
    fwd_y = -vy[2]
    fwd_z = -vz[2]
    len_fwd = math.hypot(fwd_x, fwd_y, fwd_z)
    if len_fwd > 1e-6:
        fwd_x /= len_fwd
        fwd_y /= len_fwd
        fwd_z /= len_fwd
    res.forward = QVector3D(fwd_x, fwd_y, fwd_z)

    # Pitch & Yaw from forward vector:
    # In IngeTrazo: forward = - (cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), sin(pitch))
    pitch_rad = math.asin(max(-1.0, min(1.0, -fwd_z)))
    yaw_rad = math.atan2(-fwd_y, -fwd_x)
    res.pitch_deg = math.degrees(pitch_rad)
    res.yaw_deg = math.degrees(yaw_rad)

    # Origin placement:
    # The origin handle projects the world origin (0, 0, 0) onto (u0, v0).
    u0, v0 = to_px(data.origin)
    du0 = u0 - cx
    dv0 = -(v0 - cy)
    r0_cam = [du0, dv0, -f]
    len_r0 = math.hypot(r0_cam[0], r0_cam[1], r0_cam[2])
    r0_cam = [r0_cam[0] / len_r0, r0_cam[1] / len_r0, r0_cam[2] / len_r0]

    # Transform r0 from camera frame to world frame:
    # r0_world = R_cw * r0_cam = [vx; vy; vz]^T * r0_cam
    r0_wx = vx[0] * r0_cam[0] + vx[1] * r0_cam[1] + vx[2] * r0_cam[2]
    r0_wy = vy[0] * r0_cam[0] + vy[1] * r0_cam[1] + vy[2] * r0_cam[2]
    r0_wz = vz[0] * r0_cam[0] + vz[1] * r0_cam[1] + vz[2] * r0_cam[2]

    # Eye position: eye = origin_world - distance * r0_world = -dist * r0_world
    dist = max(0.1, data.distance)
    eye_x = -dist * r0_wx
    eye_y = -dist * r0_wy
    eye_z = -dist * r0_wz
    res.eye = QVector3D(eye_x, eye_y, eye_z)
    res.target = QVector3D(0.0, 0.0, 0.0)

    # Horizon line segment (connecting Vx and Vy)
    if vp_x is not None and vp_y is not None:
        res.horizon = (vp_x, vp_y)

    return res


# ---- Viewport Event Filter for Dragging Guides -----------------------------
class PerspectiveEventFilter(QObject):
    """Filters mouse events on IngeTrazo's viewport to allow dragging guide handles."""

    HANDLE_RADIUS = 7.0
    HIT_RADIUS = 14.0

    def __init__(self, plugin: "PerspectiveMatcherPlugin") -> None:
        super().__init__()
        self.plugin = plugin
        self.active_handle: Optional[str] = None
        self.hovered_handle: Optional[str] = None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if not self.plugin.data.show_guides or self.plugin.data.camera_locked:
            return False

        t = event.type()
        vp = self.plugin.app.viewport

        if t == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                pos = event.position()
                hit = self._hit_test(pos.x(), pos.y(), vp.width(), vp.height())
                if hit is not None:
                    self.active_handle = hit
                    vp.setCursor(Qt.ClosedHandCursor)
                    vp.update()
                    return True  # Consume event to prevent canvas orbit/selection

        elif t == QEvent.MouseMove:
            pos = event.position()
            w, h = max(vp.width(), 100), max(vp.height(), 100)

            if self.active_handle is not None and event.buttons() & Qt.LeftButton:
                # Update handle position in relative [0.0, 1.0] coordinates
                rx = max(0.01, min(0.99, pos.x() / w))
                ry = max(0.01, min(0.99, pos.y() / h))
                setattr(self.plugin.data, self.active_handle, [rx, ry])

                # Live solve & align camera
                self.plugin.update_camera_from_guides()
                vp.update()
                return True

            # Hover test for changing cursor
            hit = self._hit_test(pos.x(), pos.y(), w, h)
            if hit != self.hovered_handle:
                self.hovered_handle = hit
                if hit is not None:
                    vp.setCursor(Qt.PointingHandCursor)
                else:
                    vp.unsetCursor()
                vp.update()

        elif t == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton and self.active_handle is not None:
                self.active_handle = None
                vp.unsetCursor()
                self.plugin.save_state()
                vp.update()
                return True

        return False

    def _hit_test(self, px: float, py: float, w: int, h: int) -> Optional[str]:
        handles = [
            "x1_a", "x1_b", "x2_a", "x2_b",
            "y1_a", "y1_b", "y2_a", "y2_b",
            "origin"
        ]
        if self.plugin.data.mode == "3point":
            handles.extend(["z1_a", "z1_b", "z2_a", "z2_b"])

        for name in handles:
            rel = getattr(self.plugin.data, name, None)
            if rel is None:
                continue
            hx, hy = rel[0] * w, rel[1] * h
            if math.hypot(px - hx, py - hy) <= self.HIT_RADIUS:
                return name
        return None


# ---- Side Tray UI Panel ---------------------------------------------------
class PerspectiveMatcherPanel(QWidget):
    """Side panel for controlling perspective match photo, guides, and parameters."""

    def __init__(self, plugin: "PerspectiveMatcherPlugin") -> None:
        super().__init__()
        self.plugin = plugin
        self._init_ui()

    def _init_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        # --- Photo Section ---
        photo_grp = QGroupBox(_t("Reference Photograph"))
        photo_lay = QVBoxLayout(photo_grp)
        photo_lay.setSpacing(6)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton(_t("Load Photograph…"))
        self.btn_load.clicked.connect(self._on_load_photo)
        self.btn_clear = QPushButton(_t("Clear Photo"))
        self.btn_clear.clicked.connect(self._on_clear_photo)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_clear)
        photo_lay.addLayout(btn_row)

        self.lbl_path = QLabel("")
        self.lbl_path.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_path.setWordWrap(True)
        photo_lay.addWidget(self.lbl_path)

        # Opacity slider
        opac_row = QHBoxLayout()
        opac_row.addWidget(QLabel(_t("Opacity:")))
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(int(self.plugin.data.image_opacity * 100))
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        self.lbl_opacity = QLabel(f"{int(self.plugin.data.image_opacity * 100)}%")
        opac_row.addWidget(self.slider_opacity)
        opac_row.addWidget(self.lbl_opacity)
        photo_lay.addLayout(opac_row)

        self.chk_show_photo = QCheckBox(_t("Show Background Photo"))
        self.chk_show_photo.setChecked(self.plugin.data.show_image)
        self.chk_show_photo.toggled.connect(self._on_toggle_show_photo)
        photo_lay.addWidget(self.chk_show_photo)

        lay.addWidget(photo_grp)

        # --- Perspective Solver Section ---
        solve_grp = QGroupBox(_t("Perspective & Guides"))
        solve_lay = QVBoxLayout(solve_grp)
        solve_lay.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel(_t("Perspective Mode:")))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem(_t("2-Point (Verticals stay vertical)"), "2point")
        self.combo_mode.addItem(_t("3-Point (Tilted camera)"), "3point")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.combo_mode)
        solve_lay.addLayout(mode_row)

        # Distance to origin
        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel(_t("Camera Distance (m):")))
        self.spin_dist = QDoubleSpinBox()
        self.spin_dist.setRange(0.5, 5000.0)
        self.spin_dist.setSingleStep(1.0)
        self.spin_dist.setValue(self.plugin.data.distance)
        self.spin_dist.valueChanged.connect(self._on_dist_changed)
        dist_row.addWidget(self.spin_dist)
        solve_lay.addLayout(dist_row)

        # Invert axes
        inv_row = QHBoxLayout()
        self.chk_inv_x = QCheckBox(_t("Invert X Axis"))
        self.chk_inv_x.setChecked(self.plugin.data.invert_x)
        self.chk_inv_x.toggled.connect(self._on_toggle_inv_x)
        self.chk_inv_y = QCheckBox(_t("Invert Y Axis"))
        self.chk_inv_y.setChecked(self.plugin.data.invert_y)
        self.chk_inv_y.toggled.connect(self._on_toggle_inv_y)
        inv_row.addWidget(self.chk_inv_x)
        inv_row.addWidget(self.chk_inv_y)
        solve_lay.addLayout(inv_row)

        self.chk_show_guides = QCheckBox(_t("Show Reference Guides"))
        self.chk_show_guides.setChecked(self.plugin.data.show_guides)
        self.chk_show_guides.toggled.connect(self._on_toggle_show_guides)
        solve_lay.addWidget(self.chk_show_guides)

        self.chk_lock_cam = QCheckBox(_t("Lock Viewport Camera"))
        self.chk_lock_cam.setChecked(self.plugin.data.camera_locked)
        self.chk_lock_cam.toggled.connect(self._on_toggle_lock_cam)
        solve_lay.addWidget(self.chk_lock_cam)

        btn_row2 = QHBoxLayout()
        self.btn_align = QPushButton(_t("Align Camera Now"))
        self.btn_align.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 5px;")
        self.btn_align.clicked.connect(self.plugin.update_camera_from_guides)
        self.btn_reset = QPushButton(_t("Reset Default Guides"))
        self.btn_reset.clicked.connect(self._on_reset_guides)
        btn_row2.addWidget(self.btn_align)
        btn_row2.addWidget(self.btn_reset)
        solve_lay.addLayout(btn_row2)

        lay.addWidget(solve_grp)

        # --- Readout & Calibration Info ---
        info_grp = QGroupBox(_t("Camera Calibration"))
        info_lay = QFormLayout(info_grp)
        info_lay.setContentsMargins(8, 8, 8, 8)
        info_lay.setSpacing(4)

        self.lbl_focal = QLabel("35.0 mm")
        self.lbl_fov = QLabel("45.0°")
        self.lbl_orient = QLabel("Pitch: 20° | Yaw: -45°")
        self.lbl_status = QLabel(_t("Ready"))
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

        info_lay.addRow(_t("Focal Length:"), self.lbl_focal)
        info_lay.addRow(_t("Field of View (FOV):"), self.lbl_fov)
        info_lay.addRow(_t("Camera Pitch / Yaw:"), self.lbl_orient)
        info_lay.addRow(_t("Status:"), self.lbl_status)

        lay.addWidget(info_grp)
        lay.addStretch(1)

        self.refresh_ui()

    def refresh_ui(self) -> None:
        self.lbl_path.setText(os.path.basename(self.plugin.data.image_path) or "")
        self.slider_opacity.setValue(int(self.plugin.data.image_opacity * 100))
        self.lbl_opacity.setText(f"{int(self.plugin.data.image_opacity * 100)}%")
        self.chk_show_photo.setChecked(self.plugin.data.show_image)
        self.chk_show_guides.setChecked(self.plugin.data.show_guides)
        self.chk_lock_cam.setChecked(self.plugin.data.camera_locked)
        self.spin_dist.setValue(self.plugin.data.distance)
        self.chk_inv_x.setChecked(self.plugin.data.invert_x)
        self.chk_inv_y.setChecked(self.plugin.data.invert_y)
        idx = self.combo_mode.findData(self.plugin.data.mode)
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)

    def update_readouts(self, solved: SolvedCameraParams) -> None:
        self.lbl_focal.setText(f"{solved.focal_35mm:.1f} mm (35mm eq.)")
        self.lbl_fov.setText(f"{solved.fov_deg:.1f}°")
        self.lbl_orient.setText(f"Pitch: {solved.pitch_deg:.1f}° | Yaw: {solved.yaw_deg:.1f}°")
        self.lbl_status.setText(solved.status_msg)
        if solved.valid:
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet("color: #FF9800; font-weight: bold;")

    def _on_load_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, _t("Load Photograph…"), "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All Files (*)"
        )
        if path:
            self.plugin.load_image(path)
            self.refresh_ui()

    def _on_clear_photo(self) -> None:
        self.plugin.clear_image()
        self.refresh_ui()

    def _on_opacity_changed(self, val: int) -> None:
        self.plugin.data.image_opacity = val / 100.0
        self.lbl_opacity.setText(f"{val}%")
        self.plugin.app.viewport.update()
        self.plugin.save_state()

    def _on_toggle_show_photo(self, checked: bool) -> None:
        self.plugin.data.show_image = checked
        self.plugin.app.viewport.update()
        self.plugin.save_state()

    def _on_toggle_show_guides(self, checked: bool) -> None:
        self.plugin.data.show_guides = checked
        self.plugin.app.viewport.update()
        self.plugin.save_state()

    def _on_toggle_lock_cam(self, checked: bool) -> None:
        self.plugin.data.camera_locked = checked
        self.plugin.save_state()

    def _on_mode_changed(self) -> None:
        self.plugin.data.mode = self.combo_mode.currentData()
        self.plugin.update_camera_from_guides()
        self.plugin.save_state()

    def _on_dist_changed(self, val: float) -> None:
        self.plugin.data.distance = val
        self.plugin.update_camera_from_guides()
        self.plugin.save_state()

    def _on_toggle_inv_x(self, checked: bool) -> None:
        self.plugin.data.invert_x = checked
        self.plugin.update_camera_from_guides()
        self.plugin.save_state()

    def _on_toggle_inv_y(self, checked: bool) -> None:
        self.plugin.data.invert_y = checked
        self.plugin.update_camera_from_guides()
        self.plugin.save_state()

    def _on_reset_guides(self) -> None:
        d = PerspectiveMatchData()
        d.image_path = self.plugin.data.image_path
        d.image_opacity = self.plugin.data.image_opacity
        d.show_image = self.plugin.data.show_image
        self.plugin.data = d
        self.refresh_ui()
        self.plugin.update_camera_from_guides()
        self.plugin.save_state()


# ---- Core Perspective Matcher Plugin --------------------------------------
class PerspectiveMatcherPlugin:
    """Master controller connecting IngeTrazo viewport, side panel, and solver."""

    def __init__(self, app) -> None:
        self.app = app
        self.data = PerspectiveMatchData()
        self.pixmap: Optional[QPixmap] = None
        self.solved = SolvedCameraParams()

        # Load persisted document data
        saved = app.document_data(default=None)
        if saved:
            self.data.from_dict(saved)
            if self.data.image_path and os.path.isfile(self.data.image_path):
                self.pixmap = QPixmap(self.data.image_path)

        self.panel = PerspectiveMatcherPanel(self)
        self.filter = PerspectiveEventFilter(self)
        app.viewport.installEventFilter(self.filter)

    def load_image(self, path: str) -> None:
        if os.path.isfile(path):
            self.data.image_path = path
            self.pixmap = QPixmap(path)
            self.save_state()
            self.app.viewport.update()

    def clear_image(self) -> None:
        self.data.image_path = ""
        self.pixmap = None
        self.save_state()
        self.app.viewport.update()

    def save_state(self) -> None:
        self.app.set_document_data(self.data.to_dict())

    def update_camera_from_guides(self) -> None:
        """Solves perspective from current guides and sets IngeTrazo's OrbitCamera."""
        vp = self.app.viewport
        w, h = max(vp.width(), 100), max(vp.height(), 100)
        self.solved = solve_perspective(self.data, w, h)
        self.panel.update_readouts(self.solved)

        if not self.solved.valid:
            vp.update()
            return

        cam = vp.camera
        cam.fov_deg = self.solved.fov_deg
        cam.distance = (self.solved.target - self.solved.eye).length()
        direction = self.solved.target - self.solved.eye
        cam.look_from(self.solved.eye, direction)
        cam.two_point = (self.data.mode == "2point")
        vp.update()

    def draw_overlay(self, viewport, painter: QPainter) -> None:
        """Draws background photo and perspective matching guide lines."""
        w, h = viewport.width(), viewport.height()

        # 1. Draw Background Photograph
        if self.data.show_image and self.pixmap is not None and not self.pixmap.isNull():
            painter.save()
            painter.setOpacity(self.data.image_opacity)
            scaled = self.pixmap.scaled(
                w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            dx = (w - scaled.width()) / 2
            dy = (h - scaled.height()) / 2
            painter.drawPixmap(int(dx), int(dy), scaled)
            painter.restore()

        # 2. Draw Guides
        if not self.data.show_guides:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        def to_px(rel_pt: List[float]) -> QPointF:
            return QPointF(rel_pt[0] * w, rel_pt[1] * h)

        # Style constants
        COLOR_X = QColor(255, 59, 48)     # Bright Red
        COLOR_Y = QColor(52, 199, 89)     # Bright Green
        COLOR_Z = QColor(0, 122, 255)     # Bright Blue
        COLOR_ORIGIN = QColor(255, 149, 0) # Gold / Orange
        COLOR_HORIZON = QColor(255, 214, 10, 180)

        # Helper to draw line pair with vanishing point ray
        def draw_line_pair(
            p1_a: QPointF, p1_b: QPointF,
            p2_a: QPointF, p2_b: QPointF,
            color: QColor,
            vp_pt: Optional[Tuple[float, float]],
            name1: str, name2: str,
            handle_a1: str, handle_b1: str,
            handle_a2: str, handle_b2: str
        ) -> None:
            # Solid line segments
            pen = QPen(color, 2.0, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(p1_a, p1_b)
            painter.drawLine(p2_a, p2_b)

            # Dashed rays towards vanishing point
            if vp_pt is not None:
                vp_q = QPointF(vp_pt[0], vp_pt[1])
                dash_pen = QPen(QColor(color.red(), color.green(), color.blue(), 140), 1.0, Qt.DashLine)
                painter.setPen(dash_pen)
                painter.drawLine(p1_b, vp_q)
                painter.drawLine(p2_b, vp_q)

            # Draggable Endpoints
            draw_handle(p1_a, color, handle_a1)
            draw_handle(p1_b, color, handle_b1)
            draw_handle(p2_a, color, handle_a2)
            draw_handle(p2_b, color, handle_b2)

        def draw_handle(pt: QPointF, color: QColor, handle_name: str) -> None:
            is_hover = (self.filter.hovered_handle == handle_name or
                        self.filter.active_handle == handle_name)
            radius = 8.5 if is_hover else 6.0

            # Outer white ring
            painter.setPen(QPen(QColor(255, 255, 255, 240), 2.0))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(pt, radius, radius)

            # Center white dot
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(Qt.white))
            painter.drawEllipse(pt, 2.0, 2.0)

        # Draw X lines (Red)
        px1_a, px1_b = to_px(self.data.x1_a), to_px(self.data.x1_b)
        px2_a, px2_b = to_px(self.data.x2_a), to_px(self.data.x2_b)
        draw_line_pair(
            px1_a, px1_b, px2_a, px2_b, COLOR_X, self.solved.vp_x,
            "X1", "X2", "x1_a", "x1_b", "x2_a", "x2_b"
        )

        # Draw Y lines (Green)
        py1_a, py1_b = to_px(self.data.y1_a), to_px(self.data.y1_b)
        py2_a, py2_b = to_px(self.data.y2_a), to_px(self.data.y2_b)
        draw_line_pair(
            py1_a, py1_b, py2_a, py2_b, COLOR_Y, self.solved.vp_y,
            "Y1", "Y2", "y1_a", "y1_b", "y2_a", "y2_b"
        )

        # Draw Z lines (Blue) if 3-point mode
        if self.data.mode == "3point":
            pz1_a, pz1_b = to_px(self.data.z1_a), to_px(self.data.z1_b)
            pz2_a, pz2_b = to_px(self.data.z2_a), to_px(self.data.z2_b)
            draw_line_pair(
                pz1_a, pz1_b, pz2_a, pz2_b, COLOR_Z, self.solved.vp_z,
                "Z1", "Z2", "z1_a", "z1_b", "z2_a", "z2_b"
            )

        # Draw Horizon Line between Vx and Vy
        if self.solved.horizon is not None:
            (vx1, vy1), (vx2, vy2) = self.solved.horizon
            pen_h = QPen(COLOR_HORIZON, 1.5, Qt.DashDotLine)
            painter.setPen(pen_h)
            painter.drawLine(QPointF(vx1, vy1), QPointF(vx2, vy2))

        # Draw Origin Handle with Axis Crosshair
        p_orig = to_px(self.data.origin)
        draw_handle(p_orig, COLOR_ORIGIN, "origin")

        # 3D Origin Axes Gizmo projected in 2D
        arm = 18.0
        painter.setPen(QPen(COLOR_X, 2.5))
        painter.drawLine(p_orig, QPointF(p_orig.x() - arm, p_orig.y() + arm * 0.4))
        painter.setPen(QPen(COLOR_Y, 2.5))
        painter.drawLine(p_orig, QPointF(p_orig.x() + arm, p_orig.y() + arm * 0.4))
        painter.setPen(QPen(COLOR_Z, 2.5))
        painter.drawLine(p_orig, QPointF(p_orig.x(), p_orig.y() - arm * 1.2))

        # HUD Badge at Top-Left
        badge_rect = QRectF(12, 12, 280, 26)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(20, 22, 28, 190)))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)

        font = QFont("Segoe UI", 9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(240, 240, 240)))
        status_txt = f"🎯 f: {self.solved.focal_35mm:.1f}mm | FOV: {self.solved.fov_deg:.1f}° | P: {self.solved.pitch_deg:.0f}°"
        painter.drawText(badge_rect.adjusted(8, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, status_txt)

        painter.restore()


# ---- Entry point -----------------------------------------------------------
_GLOBAL_PLUGIN: Optional[PerspectiveMatcherPlugin] = None


def setup(app) -> None:
    """Extension entry point for IngeTrazo (views/extension_api.py)."""
    global _GLOBAL_PLUGIN

    plugin = PerspectiveMatcherPlugin(app)
    _GLOBAL_PLUGIN = plugin

    # 1. Add side panel tab to IngeTrazo
    dock = app.add_panel(_t("Perspective Match"), plugin.panel)

    # 2. Add viewport drawing overlay
    app.add_overlay(lambda vp, painter: plugin.draw_overlay(vp, painter))

    # 3. Add entry to Extensions menu
    app.add_menu_action(
        _t("Perspective Matcher…"),
        lambda: app.show_panel(dock),
        shortcut="Ctrl+Shift+M",
        tip="Match viewport perspective to reference photograph (SketchUp / Blender style)."
    )

    # 4. Sync document changes
    app.on_document_changed(lambda: plugin.panel.refresh_ui())
