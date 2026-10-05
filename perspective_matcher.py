# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Seghier Mohamed Abdellatif and IngeTrazo contributors.
"""Perspective Matcher — Camera calibration and photo perspective matching for IngeTrazo.

Aligns IngeTrazo's 3D viewport camera to an architectural photograph, identical
in function to SketchUp's "Match New Photo" and Blender's "Perspective Plotter" / fSpy.

Features:
- Master Enable / Disable Toggle (always disabled by default, zero viewport interference).
- SketchUp-style Scene Tabs on Viewport with clickable buttons:
    * [ 📷 Perspective Match ] (locks/matches photo camera)
    * [ 🌐 Default 3D ] (returns to free 3D orbit)
    * [ ➕ Add View ] (saves current viewport view)
- Precision Loupe (magnifying zoom glass on handle drag with [Shift] 0.2x slow-motion micro-adjustment).
- Overlay reference photograph with adjustable opacity.
- Interactive vanishing lines:
    * 2 Red lines for the X axis -> Vanishing Point Vx.
    * 2 Green lines for the Y axis -> Vanishing Point Vy.
    * 2 Blue lines for the Z axis -> Vanishing Point Vz (in 3-point mode).
- Draggable Origin pin (0, 0, 0) with true 3D perspective axes projection.
- Horizon line visualization.
- Real-time projective geometry solver:
    * Solves camera focal length (mm, 35mm eq.) and Field of View (FOV).
    * Computes 3D camera rotation matrix (Pitch, Yaw, Roll).
    * Places camera eye such that (0, 0, 0) projects exactly onto the Origin handle.
- 2-Point and 3-Point perspective modes.
- Saves calibration and saved views configuration inside the IngeTrazo document (.igz).
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen,
    QPixmap, QVector3D
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget
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
        "Invert Z Axis": "Invertir Eje Z",
        "Swap X ⇄ Y": "Intercambiar X ⇄ Y",
        "Camera Calibration": "Calibración de Cámara",
        "Focal Length:": "Distancia Focal:",
        "Field of View (FOV):": "Campo de Visión (FOV):",
        "Camera Pitch / Yaw:": "Inclinación / Giro:",
        "Status:": "Estado:",
        "Ready": "Listo",
        "Lines converging properly": "Líneas convergiendo correctamente",
        "Lines nearly parallel (adjust guides)": "Líneas casi paralelas (ajustar guías)",
        "Invalid vanishing point configuration": "Configuración de puntos de fuga no válida",
        "Saved Views & Scenes": "Vistas Guardadas y Escenas",
        "Save Current": "Guardar Actual",
        "Update Camera": "Actualizar Cámara",
        "Delete": "Eliminar",
        "Rename": "Renombrar",
        "Default 3D": "3D Predeterminado",
        "Enable Perspective Match": "Activar Ajuste de Perspectiva",
        "Perspective Match: ACTIVE": "Ajuste de Perspectiva: ACTIVO",
        "Perspective Match: OFF": "Ajuste de Perspectiva: INACTIVO",
    }
}


def _t(text: str) -> str:
    lang = current_language()
    return _TRANSLATIONS.get(lang, {}).get(text, text)


# ---- Camera State Capture & Restore Helpers --------------------------------
def get_camera_state(cam) -> Dict[str, Any]:
    """Captures camera attributes (eye, forward, distance, fov) for view restoration."""
    state: Dict[str, Any] = {}
    if hasattr(cam, "fov_deg"):
        try:
            state["fov_deg"] = float(cam.fov_deg)
        except Exception:
            pass
    if hasattr(cam, "distance"):
        try:
            state["distance"] = float(cam.distance)
        except Exception:
            pass
    if hasattr(cam, "two_point"):
        try:
            state["two_point"] = bool(cam.two_point)
        except Exception:
            pass
    if hasattr(cam, "pitch"):
        try:
            state["pitch"] = float(cam.pitch)
        except Exception:
            pass
    if hasattr(cam, "yaw"):
        try:
            state["yaw"] = float(cam.yaw)
        except Exception:
            pass
    if hasattr(cam, "eye"):
        v = cam.eye
        state["eye"] = [float(v.x()), float(v.y()), float(v.z())] if hasattr(v, "x") else list(v)
    if hasattr(cam, "forward"):
        v = cam.forward
        state["forward"] = [float(v.x()), float(v.y()), float(v.z())] if hasattr(v, "x") else list(v)
    if hasattr(cam, "target"):
        v = cam.target
        state["target"] = [float(v.x()), float(v.y()), float(v.z())] if hasattr(v, "x") else list(v)
    return state


def apply_camera_state(cam, state: Dict[str, Any]) -> None:
    """Restores camera attributes to IngeTrazo's OrbitCamera."""
    if not isinstance(state, dict):
        return
    if "fov_deg" in state and hasattr(cam, "fov_deg"):
        cam.fov_deg = float(state["fov_deg"])
    if "distance" in state and hasattr(cam, "distance"):
        cam.distance = float(state["distance"])
    if "two_point" in state and hasattr(cam, "two_point"):
        cam.two_point = bool(state["two_point"])
    if "pitch" in state and hasattr(cam, "pitch"):
        try:
            cam.pitch = float(state["pitch"])
        except Exception:
            pass
    if "yaw" in state and hasattr(cam, "yaw"):
        try:
            cam.yaw = float(state["yaw"])
        except Exception:
            pass

    # Prefer look_from with eye and forward
    if "eye" in state and "forward" in state and hasattr(cam, "look_from"):
        try:
            eye = QVector3D(*state["eye"])
            fwd = QVector3D(*state["forward"])
            cam.look_from(eye, fwd)
            return
        except Exception:
            pass

    if "eye" in state and hasattr(cam, "eye"):
        try:
            cam.eye = QVector3D(*state["eye"])
        except Exception:
            pass
    if "target" in state and hasattr(cam, "target"):
        try:
            cam.target = QVector3D(*state["target"])
        except Exception:
            pass


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
    """Holds calibration state, guide positions, and saved scene views."""

    def __init__(self) -> None:
        # Master Toggle: ALWAYS False by default per architectural rule
        self.enabled: bool = False

        self.image_path: str = ""
        self.image_opacity: float = 0.55
        self.show_image: bool = True
        self.show_guides: bool = True
        self.camera_locked: bool = False
        self.mode: str = "3point"  # "2point" or "3point"
        self.distance: float = 30.0  # meters from origin
        self.invert_x: bool = False
        self.invert_y: bool = False
        self.invert_z: bool = False
        self.swap_xy: bool = False

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

        self.origin = [0.50, 0.72]

        # Saved Views / Scenes System
        self.active_view_index: int = 1  # Start on Default 3D since match is disabled by default
        self.saved_views: List[Dict[str, Any]] = [
            {
                "name": "Perspective Match",
                "type": "match",
            },
            {
                "name": "Default 3D",
                "type": "orbit",
                "camera": {
                    "fov_deg": 45.0,
                    "distance": 35.0,
                    "two_point": False,
                    "eye": [28.0, 28.0, 20.0],
                    "forward": [-0.65, -0.65, -0.4],
                    "target": [0.0, 0.0, 0.0],
                }
            }
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "active_view_index": self.active_view_index,
            "saved_views": list(self.saved_views),
            "image_path": self.image_path,
            "image_opacity": self.image_opacity,
            "show_image": self.show_image,
            "show_guides": self.show_guides,
            "camera_locked": self.camera_locked,
            "mode": self.mode,
            "distance": self.distance,
            "invert_x": self.invert_x,
            "invert_y": self.invert_y,
            "invert_z": self.invert_z,
            "swap_xy": self.swap_xy,
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
        # ALWAYS disabled by default when loading documents
        self.enabled = False

        self.image_path = str(d.get("image_path", self.image_path))
        self.image_opacity = float(d.get("image_opacity", self.image_opacity))
        self.show_image = bool(d.get("show_image", self.show_image))
        self.show_guides = bool(d.get("show_guides", self.show_guides))
        self.camera_locked = bool(d.get("camera_locked", self.camera_locked))
        self.mode = str(d.get("mode", self.mode))
        self.distance = float(d.get("distance", self.distance))
        self.invert_x = bool(d.get("invert_x", self.invert_x))
        self.invert_y = bool(d.get("invert_y", self.invert_y))
        self.invert_z = bool(d.get("invert_z", self.invert_z))
        self.swap_xy = bool(d.get("swap_xy", self.swap_xy))

        for k in ("x1_a", "x1_b", "x2_a", "x2_b", "y1_a", "y1_b", "y2_a", "y2_b",
                  "z1_a", "z1_b", "z2_a", "z2_b", "origin"):
            if k in d and len(d[k]) == 2:
                setattr(self, k, [float(d[k][0]), float(d[k][1])])

        raw_views = d.get("saved_views", None)
        if isinstance(raw_views, list) and len(raw_views) >= 2:
            self.saved_views = raw_views
            self.active_view_index = min(len(self.saved_views) - 1, int(d.get("active_view_index", 1)))


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
        self.v_x_cam: List[float] = [1.0, 0.0, 0.0]
        self.v_y_cam: List[float] = [0.0, 1.0, 0.0]
        self.v_z_cam: List[float] = [0.0, 0.0, 1.0]
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

    # Calculate Z vanishing point (in 3-point mode)
    p_z1_a, p_z1_b = to_px(data.z1_a), to_px(data.z1_b)
    p_z2_a, p_z2_b = to_px(data.z2_a), to_px(data.z2_b)
    vp_z = intersect_lines_2d(p_z1_a, p_z1_b, p_z2_a, p_z2_b)
    res.vp_z = vp_z

    if vp_x is None or vp_y is None:
        res.status_msg = _t("Lines nearly parallel (adjust guides)")
        return res

    # 2D Screen to Camera coordinate offsets (Y inverted in camera frame)
    du_x = vp_x[0] - cx
    dv_x = -(vp_x[1] - cy)
    du_y = vp_y[0] - cx
    dv_y = -(vp_y[1] - cy)

    # Vanishing Point Orthogonality condition:
    dot_prod = du_x * du_y + dv_x * dv_y

    if dot_prod >= -1.0:
        res.status_msg = _t("Invalid vanishing point configuration")
        f = (H / 2.0) / math.tan(math.radians(45.0) / 2.0)
    else:
        f = math.sqrt(-dot_prod)
        res.status_msg = _t("Lines converging properly")
        res.valid = True

    f = max(50.0, min(100000.0, f))
    res.focal_px = f

    fov_deg = math.degrees(2.0 * math.atan((H / 2.0) / f))
    fov_deg = max(5.0, min(140.0, fov_deg))
    res.fov_deg = fov_deg

    res.focal_35mm = 24.0 / (2.0 * math.tan(math.radians(fov_deg) / 2.0))

    rx = [du_x, dv_x, -f]
    len_x = math.hypot(*rx)
    rx = [c / len_x for c in rx]

    ry = [du_y, dv_y, -f]
    len_y = math.hypot(*ry)
    ry = [c / len_y for c in ry]

    if data.swap_xy:
        rx, ry = ry, rx

    if data.invert_x:
        rx = [-c for c in rx]
    if data.invert_y:
        ry = [-c for c in ry]

    vx = rx
    vy = ry

    dot_xy = vx[0] * vy[0] + vx[1] * vy[1] + vx[2] * vy[2]
    vy = [vy[0] - dot_xy * vx[0], vy[1] - dot_xy * vx[1], vy[2] - dot_xy * vx[2]]
    len_vy = math.hypot(*vy)
    if len_vy > 1e-6:
        vy = [c / len_vy for c in vy]

    if data.mode == "3point" and vp_z is not None:
        du_z = vp_z[0] - cx
        dv_z = -(vp_z[1] - cy)
        rz = [du_z, dv_z, -f]
        len_z = math.hypot(*rz)
        vz = [c / len_z for c in rz]
        if data.invert_z or dv_z < 0:
            vz = [-c for c in vz]
        dot_xz = vx[0] * vz[0] + vx[1] * vz[1] + vx[2] * vz[2]
        dot_yz = vy[0] * vz[0] + vy[1] * vz[1] + vy[2] * vz[2]
        vz = [vz[0] - dot_xz * vx[0] - dot_yz * vy[0],
              vz[1] - dot_xz * vx[1] - dot_yz * vy[1],
              vz[2] - dot_xz * vx[2] - dot_yz * vy[2]]
        len_vz = math.hypot(*vz)
        if len_vz > 1e-6:
            vz = [c / len_vz for c in vz]
    else:
        vz = [
            vx[1] * vy[2] - vx[2] * vy[1],
            vx[2] * vy[0] - vx[0] * vy[2],
            vx[0] * vy[1] - vx[1] * vy[0]
        ]
        len_vz = math.hypot(*vz)
        if len_vz > 1e-6:
            vz = [c / len_vz for c in vz]
        if vz[1] < 0:
            vz = [-c for c in vz]
            vy = [
                vz[1] * vx[2] - vz[2] * vx[1],
                vz[2] * vx[0] - vz[0] * vx[2],
                vz[0] * vx[1] - vz[1] * vx[0]
            ]

    res.v_x_cam = vx
    res.v_y_cam = vy
    res.v_z_cam = vz

    fwd_x = -vx[2]
    fwd_y = -vy[2]
    fwd_z = -vz[2]
    len_fwd = math.hypot(fwd_x, fwd_y, fwd_z)
    if len_fwd > 1e-6:
        fwd_x /= len_fwd
        fwd_y /= len_fwd
        fwd_z /= len_fwd
    res.forward = QVector3D(fwd_x, fwd_y, fwd_z)

    pitch_rad = math.asin(max(-1.0, min(1.0, -fwd_z)))
    yaw_rad = math.atan2(-fwd_y, -fwd_x)
    res.pitch_deg = math.degrees(pitch_rad)
    res.yaw_deg = math.degrees(yaw_rad)

    u0, v0 = to_px(data.origin)
    du0 = u0 - cx
    dv0 = -(v0 - cy)
    r0_cam = [du0, dv0, -f]
    len_r0 = math.hypot(*r0_cam)
    r0_cam = [c / len_r0 for c in r0_cam]

    r0_wx = vx[0] * r0_cam[0] + vx[1] * r0_cam[1] + vx[2] * r0_cam[2]
    r0_wy = vy[0] * r0_cam[0] + vy[1] * r0_cam[1] + vy[2] * r0_cam[2]
    r0_wz = vz[0] * r0_cam[0] + vz[1] * r0_cam[1] + vz[2] * r0_cam[2]

    dist = max(0.5, data.distance)
    eye_x = -dist * r0_wx
    eye_y = -dist * r0_wy
    eye_z = -dist * r0_wz
    res.eye = QVector3D(eye_x, eye_y, eye_z)
    res.target = res.eye + res.forward * dist

    if vp_x is not None and vp_y is not None:
        res.horizon = (vp_x, vp_y)

    return res


# ---- Viewport Event Filter for Dragging Guides & Scene Tabs ----------------
class PerspectiveEventFilter(QObject):
    """Filters mouse and key events on IngeTrazo's viewport."""

    HANDLE_RADIUS = 7.0
    HIT_RADIUS = 15.0

    def __init__(self, plugin: "PerspectiveMatcherPlugin") -> None:
        super().__init__()
        self.plugin = plugin
        self.active_handle: Optional[str] = None
        self.hovered_handle: Optional[str] = None
        self.hovered_tab_index: Optional[int] = None
        self.hovered_add_btn: bool = False
        self.hovered_toggle_btn: bool = False
        self.is_shift_held: bool = False
        self.last_mouse_pos: Optional[QPointF] = None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        t = event.type()
        vp = self.plugin.app.viewport

        # Track Shift key for precision dragging and loupe display
        if t == QEvent.KeyPress:
            if event.key() == Qt.Key_Shift:
                self.is_shift_held = True
                if self.active_handle is not None:
                    vp.update()
        elif t == QEvent.KeyRelease:
            if event.key() == Qt.Key_Shift:
                self.is_shift_held = False
                if self.active_handle is not None:
                    vp.update()

        # 1. Top tabs bar / master toggle interactions (ALWAYS ACTIVE)
        if t == QEvent.MouseMove and not (event.buttons() & Qt.LeftButton):
            pos = event.position()
            hit_tab = self._hit_test_tabs(pos.x(), pos.y())
            hit_add = (self.plugin.add_view_rect is not None and self.plugin.add_view_rect.contains(pos))
            hit_toggle = (self.plugin.master_toggle_rect is not None and self.plugin.master_toggle_rect.contains(pos))

            changed = False
            if hit_tab != self.hovered_tab_index:
                self.hovered_tab_index = hit_tab
                changed = True
            if hit_add != self.hovered_add_btn:
                self.hovered_add_btn = hit_add
                changed = True
            if hit_toggle != self.hovered_toggle_btn:
                self.hovered_toggle_btn = hit_toggle
                changed = True

            if hit_tab is not None or hit_add or hit_toggle:
                vp.setCursor(Qt.PointingHandCursor)
                if changed:
                    vp.update()
                return False
            else:
                if self.hovered_handle is None and (self.hovered_tab_index is not None or self.hovered_add_btn or self.hovered_toggle_btn):
                    vp.unsetCursor()
                if changed:
                    vp.update()

        elif t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            pos = event.position()

            # Master toggle button click
            if self.plugin.master_toggle_rect is not None and self.plugin.master_toggle_rect.contains(pos):
                self.plugin.toggle_enabled()
                vp.update()
                return True

            # Scene tabs click
            tab_idx = self._hit_test_tabs(pos.x(), pos.y())
            if tab_idx is not None:
                self.plugin.switch_to_view(tab_idx)
                vp.update()
                return True

            # Add view button click
            if self.plugin.add_view_rect is not None and self.plugin.add_view_rect.contains(pos):
                self.plugin.add_current_view()
                vp.update()
                return True

        # 2. Guide Handles Dragging (ONLY ACTIVE WHEN MATCH IS ENABLED AND CAMERA UNLOCKED)
        if not self.plugin.data.enabled or not self.plugin.data.show_guides or self.plugin.data.camera_locked:
            return False

        if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            pos = event.position()
            hit = self._hit_test(pos.x(), pos.y(), vp.width(), vp.height())
            if hit is not None:
                self.active_handle = hit
                self.last_mouse_pos = pos
                self.is_shift_held = bool(event.modifiers() & Qt.ShiftModifier)
                vp.setCursor(Qt.ClosedHandCursor)
                vp.update()
                return True

        elif t == QEvent.MouseMove:
            pos = event.position()
            w, h = max(vp.width(), 100), max(vp.height(), 100)

            if self.active_handle is not None and (event.buttons() & Qt.LeftButton):
                self.is_shift_held = bool(event.modifiers() & Qt.ShiftModifier)

                if self.is_shift_held and self.last_mouse_pos is not None:
                    # Precision slow-motion mode (0.2x speed delta)
                    dx = pos.x() - self.last_mouse_pos.x()
                    dy = pos.y() - self.last_mouse_pos.y()
                    cur = getattr(self.plugin.data, self.active_handle)
                    rx = max(0.005, min(0.995, cur[0] + (dx * 0.2) / w))
                    ry = max(0.005, min(0.995, cur[1] + (dy * 0.2) / h))
                else:
                    # Direct 1:1 positioning
                    rx = max(0.005, min(0.995, pos.x() / w))
                    ry = max(0.005, min(0.995, pos.y() / h))

                self.last_mouse_pos = pos
                setattr(self.plugin.data, self.active_handle, [rx, ry])

                # Live solve & align camera
                self.plugin.update_camera_from_guides()
                vp.update()
                return True

            # Hover test for guide handles
            hit = self._hit_test(pos.x(), pos.y(), w, h)
            if hit != self.hovered_handle:
                self.hovered_handle = hit
                if hit is not None:
                    vp.setCursor(Qt.PointingHandCursor)
                elif not (self.hovered_tab_index is not None or self.hovered_add_btn or self.hovered_toggle_btn):
                    vp.unsetCursor()
                vp.update()

        elif t == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self.active_handle is not None:
                self.active_handle = None
                self.last_mouse_pos = None
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

    def _hit_test_tabs(self, px: float, py: float) -> Optional[int]:
        for rect, idx in self.plugin.scene_tab_rects:
            if rect.contains(px, py):
                return idx
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

        # --- Master Enable / Disable Toggle ---
        self.btn_master_toggle = QPushButton()
        self.btn_master_toggle.setCheckable(False)
        self.btn_master_toggle.clicked.connect(self._on_master_toggle_clicked)
        lay.addWidget(self.btn_master_toggle)

        # --- Saved Views & Scenes Section ---
        scenes_grp = QGroupBox(_t("Saved Views & Scenes"))
        scenes_lay = QVBoxLayout(scenes_grp)
        scenes_lay.setSpacing(6)

        self.views_list = QListWidget()
        self.views_list.currentRowChanged.connect(self._on_view_selected)
        self.views_list.itemDoubleClicked.connect(lambda _i: self._on_rename_view())
        scenes_lay.addWidget(self.views_list)

        views_btn_row = QHBoxLayout()
        self.btn_add_view = QPushButton(_t("Save Current"))
        self.btn_add_view.clicked.connect(self._on_add_view)
        self.btn_update_view = QPushButton(_t("Update Camera"))
        self.btn_update_view.clicked.connect(self._on_update_view)
        self.btn_del_view = QPushButton(_t("Delete"))
        self.btn_del_view.clicked.connect(self._on_delete_view)
        views_btn_row.addWidget(self.btn_add_view)
        views_btn_row.addWidget(self.btn_update_view)
        views_btn_row.addWidget(self.btn_del_view)
        scenes_lay.addLayout(views_btn_row)

        lay.addWidget(scenes_grp)

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

        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel(_t("Camera Distance (m):")))
        self.spin_dist = QDoubleSpinBox()
        self.spin_dist.setRange(0.5, 5000.0)
        self.spin_dist.setSingleStep(1.0)
        self.spin_dist.setValue(self.plugin.data.distance)
        self.spin_dist.valueChanged.connect(self._on_dist_changed)
        dist_row.addWidget(self.spin_dist)
        solve_lay.addLayout(dist_row)

        inv_row1 = QHBoxLayout()
        self.chk_inv_x = QCheckBox(_t("Invert X Axis"))
        self.chk_inv_x.setChecked(self.plugin.data.invert_x)
        self.chk_inv_x.toggled.connect(self._on_toggle_inv_x)
        self.chk_inv_y = QCheckBox(_t("Invert Y Axis"))
        self.chk_inv_y.setChecked(self.plugin.data.invert_y)
        self.chk_inv_y.toggled.connect(self._on_toggle_inv_y)
        inv_row1.addWidget(self.chk_inv_x)
        inv_row1.addWidget(self.chk_inv_y)
        solve_lay.addLayout(inv_row1)

        inv_row2 = QHBoxLayout()
        self.chk_inv_z = QCheckBox(_t("Invert Z Axis"))
        self.chk_inv_z.setChecked(self.plugin.data.invert_z)
        self.chk_inv_z.toggled.connect(self._on_toggle_inv_z)
        self.chk_swap_xy = QCheckBox(_t("Swap X ⇄ Y"))
        self.chk_swap_xy.setChecked(self.plugin.data.swap_xy)
        self.chk_swap_xy.toggled.connect(self._on_toggle_swap_xy)
        inv_row2.addWidget(self.chk_inv_z)
        inv_row2.addWidget(self.chk_swap_xy)
        solve_lay.addLayout(inv_row2)

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
        # Refresh Master Toggle visual state
        if self.plugin.data.enabled:
            self.btn_master_toggle.setText(_t("🟢 Perspective Match: ACTIVE"))
            self.btn_master_toggle.setStyleSheet(
                "background-color: #1B5E20; color: #FFFFFF; font-weight: bold; "
                "font-size: 12px; padding: 7px; border: 1px solid #4CAF50; border-radius: 4px;"
            )
        else:
            self.btn_master_toggle.setText(_t("⚪ Perspective Match: OFF (Click to Enable)"))
            self.btn_master_toggle.setStyleSheet(
                "background-color: #263238; color: #CFD8DC; font-weight: bold; "
                "font-size: 12px; padding: 7px; border: 1px solid #546E7A; border-radius: 4px;"
            )

        # Refresh Views List
        self.views_list.blockSignals(True)
        self.views_list.clear()
        for idx, view in enumerate(self.plugin.data.saved_views):
            vtype = view.get("type", "custom")
            icon = "📷 " if vtype == "match" else "🌐 "
            vname = view.get("name", f"Scene {idx}")
            item = QListWidgetItem(f"{icon}{vname}")
            self.views_list.addItem(item)
        if 0 <= self.plugin.data.active_view_index < self.views_list.count():
            self.views_list.setCurrentRow(self.plugin.data.active_view_index)
        self.views_list.blockSignals(False)

        self.lbl_path.setText(os.path.basename(self.plugin.data.image_path) or "")

        for w in (self.slider_opacity, self.chk_show_photo, self.chk_show_guides,
                  self.chk_lock_cam, self.spin_dist, self.chk_inv_x, self.chk_inv_y,
                  self.chk_inv_z, self.chk_swap_xy, self.combo_mode):
            w.blockSignals(True)

        self.slider_opacity.setValue(int(self.plugin.data.image_opacity * 100))
        self.lbl_opacity.setText(f"{int(self.plugin.data.image_opacity * 100)}%")
        self.chk_show_photo.setChecked(self.plugin.data.show_image)
        self.chk_show_guides.setChecked(self.plugin.data.show_guides)
        self.chk_lock_cam.setChecked(self.plugin.data.camera_locked)
        self.spin_dist.setValue(self.plugin.data.distance)
        self.chk_inv_x.setChecked(self.plugin.data.invert_x)
        self.chk_inv_y.setChecked(self.plugin.data.invert_y)
        self.chk_inv_z.setChecked(self.plugin.data.invert_z)
        self.chk_swap_xy.setChecked(self.plugin.data.swap_xy)
        idx = self.combo_mode.findData(self.plugin.data.mode)
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)

        for w in (self.slider_opacity, self.chk_show_photo, self.chk_show_guides,
                  self.chk_lock_cam, self.spin_dist, self.chk_inv_x, self.chk_inv_y,
                  self.chk_inv_z, self.chk_swap_xy, self.combo_mode):
            w.blockSignals(False)

    def update_readouts(self, solved: SolvedCameraParams) -> None:
        self.lbl_focal.setText(f"{solved.focal_35mm:.1f} mm (35mm eq.)")
        self.lbl_fov.setText(f"{solved.fov_deg:.1f}°")
        self.lbl_orient.setText(f"Pitch: {solved.pitch_deg:.1f}° | Yaw: {solved.yaw_deg:.1f}°")
        self.lbl_status.setText(solved.status_msg)
        if solved.valid:
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet("color: #FF9800; font-weight: bold;")

    def _on_master_toggle_clicked(self) -> None:
        self.plugin.toggle_enabled()

    def _on_view_selected(self, row: int) -> None:
        if row >= 0 and row != self.plugin.data.active_view_index:
            self.plugin.switch_to_view(row)

    def _on_add_view(self) -> None:
        self.plugin.add_current_view()

    def _on_update_view(self) -> None:
        self.plugin.update_active_view_camera()

    def _on_delete_view(self) -> None:
        self.plugin.delete_view(self.plugin.data.active_view_index)

    def _on_rename_view(self) -> None:
        self.plugin.rename_view(self.plugin.data.active_view_index)

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

    def _on_toggle_inv_z(self, checked: bool) -> None:
        self.plugin.data.invert_z = checked
        self.plugin.update_camera_from_guides()
        self.plugin.save_state()

    def _on_toggle_swap_xy(self, checked: bool) -> None:
        self.plugin.data.swap_xy = checked
        self.plugin.update_camera_from_guides()
        self.plugin.save_state()

    def _on_reset_guides(self) -> None:
        d = PerspectiveMatchData()
        d.image_path = self.plugin.data.image_path
        d.image_opacity = self.plugin.data.image_opacity
        d.show_image = self.plugin.data.show_image
        d.saved_views = self.plugin.data.saved_views
        d.active_view_index = self.plugin.data.active_view_index
        d.enabled = self.plugin.data.enabled
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

        # Cache of hit rects for viewport drawing & event filter
        self.scene_tab_rects: List[Tuple[QRectF, int]] = []
        self.add_view_rect: Optional[QRectF] = None
        self.master_toggle_rect: Optional[QRectF] = None

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

    def toggle_enabled(self, state: Optional[bool] = None) -> None:
        """Toggles the master perspective match mode on or off."""
        if state is None:
            self.data.enabled = not self.data.enabled
        else:
            self.data.enabled = bool(state)

        if self.data.enabled:
            # Switch to Perspective Match view
            self.data.active_view_index = 0
            self.update_camera_from_guides()
        else:
            # Revert to standard orbit view if coming from match view
            if self.data.active_view_index == 0 and len(self.data.saved_views) > 1:
                self.switch_to_view(1)

        self.save_state()
        self.panel.refresh_ui()
        self.app.viewport.update()

    def switch_to_view(self, index: int) -> None:
        """Switches active camera to a saved view / scene."""
        if index < 0 or index >= len(self.data.saved_views):
            return

        self.data.active_view_index = index
        view_data = self.data.saved_views[index]
        vtype = view_data.get("type", "custom")

        if vtype == "match":
            # Enable match mode and align camera to photo
            self.data.enabled = True
            self.update_camera_from_guides()
        else:
            # Standard or custom 3D orbit view: disable match guides/photo
            self.data.enabled = False
            cam = self.app.viewport.camera
            if "camera" in view_data:
                apply_camera_state(cam, view_data["camera"])
            self.app.viewport.update()

        self.save_state()
        self.panel.refresh_ui()
        self.app.viewport.update()

    def add_current_view(self) -> None:
        """Captures current camera and saves it as a new scene tab."""
        cam = self.app.viewport.camera
        cam_state = get_camera_state(cam)
        scene_num = len(self.data.saved_views)
        new_name = f"Scene {scene_num}"

        self.data.saved_views.append({
            "name": new_name,
            "type": "custom",
            "camera": cam_state
        })
        self.data.active_view_index = len(self.data.saved_views) - 1
        self.data.enabled = False  # Custom scene is a normal orbit view

        self.save_state()
        self.panel.refresh_ui()
        self.app.viewport.update()

    def update_active_view_camera(self) -> None:
        """Overwrites the selected view with current viewport camera."""
        idx = self.data.active_view_index
        if idx <= 0:
            # View 0 is dynamically solved from guides
            self.update_camera_from_guides()
            return

        cam = self.app.viewport.camera
        cam_state = get_camera_state(cam)
        self.data.saved_views[idx]["camera"] = cam_state
        self.save_state()
        self.app.viewport.update()

    def delete_view(self, index: int) -> None:
        """Deletes a custom saved scene view (cannot delete default views)."""
        if index <= 1:
            return  # Protect "Perspective Match" and "Default 3D"

        del self.data.saved_views[index]
        if self.data.active_view_index >= len(self.data.saved_views):
            self.data.active_view_index = len(self.data.saved_views) - 1

        self.switch_to_view(self.data.active_view_index)

    def rename_view(self, index: int) -> None:
        """Renames a saved scene view."""
        if index < 0 or index >= len(self.data.saved_views):
            return
        curr_name = self.data.saved_views[index].get("name", "")
        new_name, ok = QInputDialog.getText(
            self.panel, _t("Rename"), _t("Name:"), QLineEdit.Normal, curr_name
        )
        if ok and new_name.strip():
            self.data.saved_views[index]["name"] = new_name.strip()
            self.save_state()
            self.panel.refresh_ui()
            self.app.viewport.update()

    def update_camera_from_guides(self) -> None:
        """Solves perspective from current guides and sets IngeTrazo's OrbitCamera."""
        vp = self.app.viewport
        w, h = max(vp.width(), 100), max(vp.height(), 100)
        self.solved = solve_perspective(self.data, w, h)
        if self.panel is not None:
            self.panel.update_readouts(self.solved)

        if not self.solved.valid or not self.data.enabled:
            vp.update()
            return

        cam = vp.camera
        cam.fov_deg = self.solved.fov_deg
        cam.distance = self.data.distance
        cam.look_from(self.solved.eye, self.solved.forward)
        cam.two_point = (self.data.mode == "2point")
        vp.update()

    def _draw_scene_tabs(self, viewport, painter: QPainter) -> None:
        """Draws SketchUp-style Scene Tabs Bar and Master Toggle at the top of the viewport."""
        w, h = viewport.width(), viewport.height()
        self.scene_tab_rects = []
        self.add_view_rect = None
        self.master_toggle_rect = None

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        font = QFont("Segoe UI", 9)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()

        x = 12.0
        y = 10.0
        tab_h = 28.0

        # 1. Master Toggle Pill Button
        toggle_text = "🟢 PERSPECTIVE MATCH: ON" if self.data.enabled else "⚪ PERSPECTIVE MATCH: OFF"
        tw = fm.horizontalAdvance(toggle_text) + 24.0
        t_rect = QRectF(x, y, tw, tab_h)
        self.master_toggle_rect = t_rect

        is_toggle_hover = self.filter.hovered_toggle_btn
        painter.setPen(Qt.NoPen)
        if self.data.enabled:
            bg_col = QColor(24, 134, 75, 230) if not is_toggle_hover else QColor(30, 160, 90, 245)
            border_col = QColor(100, 255, 160, 180)
        else:
            bg_col = QColor(36, 40, 50, 200) if not is_toggle_hover else QColor(48, 54, 68, 230)
            border_col = QColor(120, 130, 150, 120)

        painter.setBrush(QBrush(bg_col))
        painter.drawRoundedRect(t_rect, 14.0, 14.0)
        painter.setPen(QPen(border_col, 1.2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(t_rect, 14.0, 14.0)

        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(t_rect, Qt.AlignCenter, toggle_text)

        x += tw + 12.0

        # Vertical separator
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1.5))
        painter.drawLine(QPointF(x, y + 4), QPointF(x, y + tab_h - 4))
        x += 12.0

        # 2. Scene Tabs
        for idx, view in enumerate(self.data.saved_views):
            vname = view.get("name", f"Scene {idx}")
            vtype = view.get("type", "custom")
            icon = "📷 " if vtype == "match" else "🌐 "
            tab_label = f"{icon}{vname}"
            tab_w = fm.horizontalAdvance(tab_label) + 26.0
            rect = QRectF(x, y, tab_w, tab_h)
            self.scene_tab_rects.append((rect, idx))

            is_active = (idx == self.data.active_view_index)
            is_hover = (idx == self.filter.hovered_tab_index)

            painter.setPen(Qt.NoPen)
            if is_active:
                painter.setBrush(QBrush(QColor(0, 122, 204, 235)))
            elif is_hover:
                painter.setBrush(QBrush(QColor(48, 54, 68, 220)))
            else:
                painter.setBrush(QBrush(QColor(24, 26, 32, 190)))

            painter.drawRoundedRect(rect, 4.0, 4.0)

            # Bottom accent indicator on active tab
            if is_active:
                painter.setPen(QPen(QColor(60, 180, 255), 2.5))
                painter.drawLine(QPointF(x + 2, y + tab_h - 1), QPointF(x + tab_w - 2, y + tab_h - 1))
            else:
                painter.setPen(QPen(QColor(255, 255, 255, 30), 1.0))
                painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setPen(QPen(QColor(255, 255, 255) if (is_active or is_hover) else QColor(190, 195, 205)))
            painter.drawText(rect, Qt.AlignCenter, tab_label)

            x += tab_w + 6.0

        # 3. Add Scene Button [+]
        add_btn_w = 28.0
        add_rect = QRectF(x, y, add_btn_w, tab_h)
        self.add_view_rect = add_rect

        is_add_hover = self.filter.hovered_add_btn
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 122, 204, 210) if is_add_hover else QColor(30, 34, 44, 190)))
        painter.drawRoundedRect(add_rect, 4.0, 4.0)
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(add_rect, 4.0, 4.0)

        painter.setPen(QPen(Qt.white))
        painter.drawText(add_rect, Qt.AlignCenter, "➕")

        painter.restore()

    def _draw_loupe(self, viewport, painter: QPainter, hx: float, hy: float, handle_name: str) -> None:
        """Draws magnifying zoom loupe around active handle during dragging."""
        w, h = viewport.width(), viewport.height()
        radius = 75.0

        # Optimal loupe center offset (+55, -95)
        lx = hx + 55.0
        ly = hy - 95.0

        if lx + radius > w - 15:
            lx = hx - 55.0 - 2 * radius
        if lx - radius < 15:
            lx = 15 + radius

        if ly - radius < 48:
            ly = hy + 55.0 + radius
        if ly + radius > h - 15:
            ly = h - 15 - radius

        loupe_center = QPointF(lx, ly)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Draw leader line from handle to loupe
        ldx = lx - hx
        ldy = ly - hy
        llen = math.hypot(ldx, ldy)
        if llen > radius:
            edge_x = lx - (ldx / llen) * radius
            edge_y = ly - (ldy / llen) * radius
            leader_pen = QPen(QColor(255, 255, 255, 120), 1.5, Qt.DashLine)
            painter.setPen(leader_pen)
            painter.drawLine(QPointF(hx, hy), QPointF(edge_x, edge_y))

        # Handle color
        handle_color = QColor(255, 149, 0)
        if handle_name.startswith("x"):
            handle_color = QColor(255, 59, 48)
        elif handle_name.startswith("y"):
            handle_color = QColor(52, 199, 89)
        elif handle_name.startswith("z"):
            handle_color = QColor(0, 122, 255)

        # Circular clip path
        clip = QPainterPath()
        clip.addEllipse(loupe_center, radius, radius)
        painter.setClipPath(clip)

        # Dark background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(18, 20, 26)))
        painter.drawRect(QRectF(lx - radius, ly - radius, radius * 2, radius * 2))

        # Magnified photograph
        zoom = 4.0
        if self.pixmap is not None and not self.pixmap.isNull():
            img_w = self.pixmap.width()
            img_h = self.pixmap.height()
            if img_w > 0 and img_h > 0:
                scale = min(w / img_w, h / img_h)
                dx = (w - img_w * scale) / 2.0
                dy = (h - img_h * scale) / 2.0

                img_px = (hx - dx) / scale
                img_py = (hy - dy) / scale

                target_scale = scale * zoom
                pix_x = lx - img_px * target_scale
                pix_y = ly - img_py * target_scale
                pix_w = img_w * target_scale
                pix_h = img_h * target_scale

                painter.setOpacity(1.0)
                painter.drawPixmap(
                    QRectF(pix_x, pix_y, pix_w, pix_h),
                    self.pixmap,
                    QRectF(0, 0, img_w, img_h)
                )

        # Pixel grid overlay
        grid_pen = QPen(QColor(255, 255, 255, 20), 1.0)
        painter.setPen(grid_pen)
        step = 16.0
        for gx in range(int(lx - radius), int(lx + radius + step), int(step)):
            painter.drawLine(QPointF(gx, ly - radius), QPointF(gx, ly + radius))
        for gy in range(int(ly - radius), int(ly + radius + step), int(step)):
            painter.drawLine(QPointF(lx - radius, gy), QPointF(lx + radius, gy))

        # Center crosshairs
        painter.setPen(QPen(QColor(0, 0, 0, 200), 2.0))
        painter.drawLine(QPointF(lx - 20, ly), QPointF(lx + 20, ly))
        painter.drawLine(QPointF(lx, ly - 20), QPointF(lx, ly + 20))
        painter.setPen(QPen(QColor(255, 255, 255, 240), 1.0))
        painter.drawLine(QPointF(lx - 20, ly), QPointF(lx + 20, ly))
        painter.drawLine(QPointF(lx, ly - 20), QPointF(lx, ly + 20))

        # Center pinpoint
        painter.setBrush(QBrush(handle_color))
        painter.setPen(QPen(Qt.white, 1.0))
        painter.drawEllipse(loupe_center, 2.5, 2.5)

        painter.setClipping(False)

        # Outer border rings
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, 180), 4.5))
        painter.drawEllipse(loupe_center, radius + 1.5, radius + 1.5)

        painter.setPen(QPen(handle_color, 3.0))
        painter.drawEllipse(loupe_center, radius, radius)

        painter.setPen(QPen(QColor(255, 255, 255, 200), 1.0))
        painter.drawEllipse(loupe_center, radius - 2.0, radius - 2.0)

        # Dynamic status badge below loupe
        badge_w = 170.0
        badge_h = 22.0
        badge_x = lx - badge_w / 2.0
        badge_y = ly + radius + 6.0
        if badge_y + badge_h > h - 5:
            badge_y = ly - radius - badge_h - 6.0
        badge_rect = QRectF(badge_x, badge_y, badge_w, badge_h)

        painter.setPen(Qt.NoPen)
        if self.filter.is_shift_held:
            painter.setBrush(QBrush(QColor(16, 140, 72, 230)))
        else:
            painter.setBrush(QBrush(QColor(24, 26, 32, 220)))
        painter.drawRoundedRect(badge_rect, 11.0, 11.0)

        painter.setPen(QPen(QColor(255, 255, 255, 80), 1.0))
        painter.drawRoundedRect(badge_rect, 11.0, 11.0)

        font = QFont("Segoe UI", 8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(Qt.white))
        if self.filter.is_shift_held:
            badge_text = "⚡ 4× PRECISION (0.2×)"
        else:
            badge_text = "🔍 4× ZOOM [Shift: Precision]"
        painter.drawText(badge_rect, Qt.AlignCenter, badge_text)

        painter.restore()

    def draw_overlay(self, viewport, painter: QPainter) -> None:
        """Draws SketchUp-style Scene Tabs, Background Photo, Vanishing Guides, and Precision Loupe."""
        w, h = viewport.width(), viewport.height()

        # 1. ALWAYS Draw Scene Tabs Bar at top of viewport
        self._draw_scene_tabs(viewport, painter)

        # 2. If Perspective Match is disabled, do not draw photo or guides!
        if not self.data.enabled:
            return

        # 3. Draw Background Photograph
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

        # 4. Draw Guides
        if not self.data.show_guides:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        def to_px(rel_pt: List[float]) -> QPointF:
            return QPointF(rel_pt[0] * w, rel_pt[1] * h)

        COLOR_X = QColor(255, 59, 48)
        COLOR_Y = QColor(52, 199, 89)
        COLOR_Z = QColor(0, 122, 255)
        COLOR_ORIGIN = QColor(255, 149, 0)
        COLOR_HORIZON = QColor(255, 214, 10, 180)

        def draw_line_pair(
            p1_a: QPointF, p1_b: QPointF,
            p2_a: QPointF, p2_b: QPointF,
            color: QColor,
            vp_pt: Optional[Tuple[float, float]],
            name1: str, name2: str,
            handle_a1: str, handle_b1: str,
            handle_a2: str, handle_b2: str
        ) -> None:
            pen = QPen(color, 2.0, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(p1_a, p1_b)
            painter.drawLine(p2_a, p2_b)

            if vp_pt is not None:
                vp_q = QPointF(vp_pt[0], vp_pt[1])
                dash_pen = QPen(QColor(color.red(), color.green(), color.blue(), 140), 1.0, Qt.DashLine)
                painter.setPen(dash_pen)
                painter.drawLine(p1_b, vp_q)
                painter.drawLine(p2_b, vp_q)

            draw_handle(p1_a, color, handle_a1)
            draw_handle(p1_b, color, handle_b1)
            draw_handle(p2_a, color, handle_a2)
            draw_handle(p2_b, color, handle_b2)

        def draw_handle(pt: QPointF, color: QColor, handle_name: str) -> None:
            is_hover = (self.filter.hovered_handle == handle_name or
                        self.filter.active_handle == handle_name)
            radius = 8.5 if is_hover else 6.0

            painter.setPen(QPen(QColor(255, 255, 255, 240), 2.0))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(pt, radius, radius)

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

        # Draw Origin Handle
        p_orig = to_px(self.data.origin)
        draw_handle(p_orig, COLOR_ORIGIN, "origin")

        # True 3D Origin Perspective Axes Gizmo
        arm = 42.0

        if self.solved.vp_x is not None:
            dx = self.solved.vp_x[0] - p_orig.x()
            dy = self.solved.vp_x[1] - p_orig.y()
            l = math.hypot(dx, dy)
            if l > 1e-4:
                sign = -1.0 if (self.data.invert_x ^ self.data.swap_xy) else 1.0
                painter.setPen(QPen(COLOR_X, 2.8))
                painter.drawLine(p_orig, QPointF(p_orig.x() + sign * arm * dx / l,
                                                p_orig.y() + sign * arm * dy / l))

        if self.solved.vp_y is not None:
            dx = self.solved.vp_y[0] - p_orig.x()
            dy = self.solved.vp_y[1] - p_orig.y()
            l = math.hypot(dx, dy)
            if l > 1e-4:
                sign = -1.0 if (self.data.invert_y ^ self.data.swap_xy) else 1.0
                painter.setPen(QPen(COLOR_Y, 2.8))
                painter.drawLine(p_orig, QPointF(p_orig.x() + sign * arm * dx / l,
                                                p_orig.y() + sign * arm * dy / l))

        if self.solved.vp_z is not None and self.data.mode == "3point":
            dx = self.solved.vp_z[0] - p_orig.x()
            dy = self.solved.vp_z[1] - p_orig.y()
            l = math.hypot(dx, dy)
            if l > 1e-4:
                sign = -1.0 if self.data.invert_z else 1.0
                painter.setPen(QPen(COLOR_Z, 2.8))
                painter.drawLine(p_orig, QPointF(p_orig.x() + sign * arm * dx / l,
                                                p_orig.y() + sign * arm * dy / l))
        else:
            painter.setPen(QPen(COLOR_Z, 2.8))
            sign = 1.0 if self.data.invert_z else -1.0
            painter.drawLine(p_orig, QPointF(p_orig.x(), p_orig.y() + sign * arm))

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255, 220)))
        painter.drawText(QPointF(p_orig.x() + 10, p_orig.y() + 14), "Origin (0,0,0)")

        # Calibration HUD Badge at Bottom-Left (moved away from top tabs)
        badge_rect = QRectF(12, h - 38, 320, 26)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(20, 22, 28, 190)))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)

        font = QFont("Segoe UI", 9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(240, 240, 240)))
        status_txt = f"🎯 f: {self.solved.focal_35mm:.1f}mm | FOV: {self.solved.fov_deg:.1f}° | P: {self.solved.pitch_deg:.1f}° | Y: {self.solved.yaw_deg:.1f}°"
        painter.drawText(badge_rect.adjusted(8, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, status_txt)

        painter.restore()

        # 5. DRAW PRECISION LOUPE IF A HANDLE IS ACTIVELY BEING DRAGGED
        if self.filter.active_handle is not None:
            rel = getattr(self.data, self.filter.active_handle, None)
            if rel is not None:
                hx = rel[0] * w
                hy = rel[1] * h
                self._draw_loupe(viewport, painter, hx, hy, self.filter.active_handle)


# ---- Entry point -----------------------------------------------------------
_GLOBAL_PLUGIN: Optional[PerspectiveMatcherPlugin] = None


def setup(app) -> None:
    """Extension entry point for IngeTrazo (views/extension_api.py)."""
    global _GLOBAL_PLUGIN

    plugin = PerspectiveMatcherPlugin(app)
    _GLOBAL_PLUGIN = plugin

    dock = app.add_panel(_t("Perspective Match"), plugin.panel)

    app.add_overlay(lambda vp, painter: plugin.draw_overlay(vp, painter))

    app.add_menu_action(
        _t("Perspective Matcher…"),
        lambda: app.show_panel(dock),
        shortcut="Ctrl+Shift+M",
        tip="Match viewport perspective to reference photograph (SketchUp / Blender style)."
    )

    app.on_document_changed(lambda: plugin.panel.refresh_ui())
