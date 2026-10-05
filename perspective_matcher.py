# SPDX-License-Identifier: GPL-3.0-or-later
# Perspective Matcher Plugin for IngeTrazo
"""
Perspective Matcher Extension for IngeTrazo (API v2).
Calibrates camera focal length, FOV, rotation (yaw/pitch), and eye position
from vanishing line guides aligned to a background photograph.
Supports SketchUp / Blender style perspective matching, scene tabs,
real-world scale calibration, and precision optical loupe.
"""

from __future__ import annotations

import base64
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QByteArray, QBuffer, QEvent, QIODevice, QObject, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QFont, QMatrix4x4, QPainter, QPainterPath, QPen, QPixmap, QVector3D
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget
)


def _t(text: str) -> str:
    """Translation stub for gettext / Qt tr."""
    return text


# ---- Camera State Helpers --------------------------------------------------
def get_camera_state(cam) -> Dict[str, Any]:
    """Extracts IngeTrazo's OrbitCamera state into a JSON-safe dictionary.
    Directly reads numerical attributes to avoid method-introspection bugs.
    """
    state: Dict[str, Any] = {}
    if hasattr(cam, "target"):
        t = cam.target
        if hasattr(t, "x"):
            state["target"] = [float(t.x()), float(t.y()), float(t.z())]
        elif isinstance(t, (list, tuple)) and len(t) == 3:
            state["target"] = [float(t[0]), float(t[1]), float(t[2])]
    if hasattr(cam, "distance"):
        state["distance"] = float(cam.distance)
    if hasattr(cam, "yaw"):
        state["yaw"] = float(cam.yaw)
    if hasattr(cam, "pitch"):
        state["pitch"] = float(cam.pitch)
    if hasattr(cam, "fov_deg"):
        state["fov_deg"] = float(cam.fov_deg)
    if hasattr(cam, "two_point"):
        state["two_point"] = bool(cam.two_point)
    if hasattr(cam, "perspective"):
        state["perspective"] = bool(cam.perspective)
    if hasattr(cam, "up"):
        u = cam.up
        if hasattr(u, "x"):
            state["up"] = [float(u.x()), float(u.y()), float(u.z())]
        elif isinstance(u, (list, tuple)) and len(u) == 3:
            state["up"] = [float(u[0]), float(u[1]), float(u[2])]
    return state


def apply_camera_state(cam, state: Dict[str, Any]) -> None:
    """Restores camera attributes to IngeTrazo's OrbitCamera."""
    if not isinstance(state, dict):
        return
    if "target" in state:
        t = state["target"]
        cam.target = QVector3D(float(t[0]), float(t[1]), float(t[2]))
    if "distance" in state:
        cam.distance = float(state["distance"])
    if "yaw" in state:
        cam.yaw = float(state["yaw"])
    if "pitch" in state:
        cam.pitch = float(state["pitch"])
    if "fov_deg" in state:
        cam.fov_deg = float(state["fov_deg"])
    if "two_point" in state:
        cam.two_point = bool(state["two_point"])
    if "perspective" in state:
        cam.perspective = bool(state["perspective"])
    if "up" in state and isinstance(state["up"], (list, tuple)) and len(state["up"]) == 3:
        u = state["up"]
        cam.up = QVector3D(float(u[0]), float(u[1]), float(u[2]))
    else:
        cam.up = QVector3D(0.0, 0.0, 1.0)


# ---- Real-World Scale & Geometry Manipulation Helpers ----------------------
def get_selected_edge_info(scene) -> Optional[Tuple[float, str]]:
    """Inspects scene.selection and returns (current_length_in_meters, description)."""
    if scene is None or not hasattr(scene, "selection") or not scene.selection:
        return None

    # 1. Direct Edge entities
    edges = [e for e in scene.selection if hasattr(e, "length") or (hasattr(e, "a") and hasattr(e, "b"))]
    if edges:
        e = edges[0]
        if hasattr(e, "length") and callable(e.length):
            length = float(e.length())
        else:
            p1 = QVector3D(e.a.x(), e.a.y(), e.a.z()) if hasattr(e.a, "x") else QVector3D(e.a)
            p2 = QVector3D(e.b.x(), e.b.y(), e.b.z()) if hasattr(e.b, "x") else QVector3D(e.b)
            dx = float(p2.x() - p1.x())
            dy = float(p2.y() - p1.y())
            dz = float(p2.z() - p1.z())
            length = math.hypot(dx, dy, dz)
        return (length, f"Edge ({length:.3f} m)")

    # 2. Two selected vertices or guide points
    pts = []
    for ent in scene.selection:
        if hasattr(ent, "position") and hasattr(ent.position, "x"):
            pts.append(QVector3D(ent.position))
        elif hasattr(ent, "point") and hasattr(ent.point, "x"):
            pts.append(QVector3D(ent.point))
        elif hasattr(ent, "x") and hasattr(ent, "y") and hasattr(ent, "z"):
            pts.append(QVector3D(ent.x(), ent.y(), ent.z()))

    if len(pts) == 2:
        dx = float(pts[1].x() - pts[0].x())
        dy = float(pts[1].y() - pts[0].y())
        dz = float(pts[1].z() - pts[0].z())
        length = math.hypot(dx, dy, dz)
        return (length, f"2 Points ({length:.3f} m)")

    # 3. Single selected Group/Component or Face
    if len(scene.selection) == 1:
        ent = next(iter(scene.selection))
        if hasattr(ent, "bounds") and callable(ent.bounds):
            try:
                b_min, b_max = ent.bounds()
                dx = float(b_max.x() - b_min.x())
                dy = float(b_max.y() - b_min.y())
                dz = float(b_max.z() - b_min.z())
                diag = math.hypot(dx, dy, dz)
                if diag > 1e-4:
                    return (diag, f"Object Bounding Box ({diag:.3f} m)")
            except Exception:
                pass

    return None


def scale_scene_geometry(scene, factor: float, anchor: Optional[QVector3D] = None, viewport=None) -> None:
    """Scales all geometry in scene (loose meshes, all groups, components, billboard figures,
    guides, dimensions, and image planes) uniformly about anchor (0, 0, 0).
    Properly updates mesh registries, chunk dirty flags, and purges viewport caches."""
    if scene is None or abs(factor - 1.0) < 1e-7:
        return

    if anchor is None:
        anchor = QVector3D(0.0, 0.0, 0.0)

    # 1. Scale loose mesh vertices using place_vertex for registry integrity
    if hasattr(scene, "mesh") and scene.mesh is not None:
        mesh = scene.mesh
        if hasattr(mesh, "vertices"):
            for v in list(mesh.vertices):
                if hasattr(v, "position") and hasattr(v.position, "x"):
                    pos = v.position
                    new_pos = anchor + (pos - anchor) * factor
                    if hasattr(mesh, "place_vertex"):
                        mesh.place_vertex(v, new_pos)
                    else:
                        v.position = new_pos
            if hasattr(mesh, "_chunk_dirty"):
                mesh._chunk_dirty = True

    # 2. Scale groups (classic groups, component instances, billboard figures)
    if hasattr(scene, "groups") and scene.groups:
        for g in scene.groups:
            _scale_group_recursive(g, factor, anchor)

    # 3. Scale guide points and segments if present
    if hasattr(scene, "guides") and scene.guides:
        for guide in scene.guides:
            if hasattr(guide, "point") and hasattr(guide.point, "x"):
                p = guide.point
                guide.point = anchor + (p - anchor) * factor
            if hasattr(guide, "origin") and guide.origin is not None and hasattr(guide.origin, "x"):
                orig = guide.origin
                guide.origin = anchor + (orig - anchor) * factor

    # 4. Scale dimensions if present
    if hasattr(scene, "dimensions") and scene.dimensions:
        for dim in scene.dimensions:
            if hasattr(dim, "_a") and hasattr(dim._a, "x"):
                dim._a = anchor + (dim._a - anchor) * factor
            if hasattr(dim, "_b") and hasattr(dim._b, "x"):
                dim._b = anchor + (dim._b - anchor) * factor
            if hasattr(dim, "offset") and hasattr(dim.offset, "x"):
                dim.offset = dim.offset * factor

    # 5. Scale image planes if present
    if hasattr(scene, "image_planes") and scene.image_planes:
        for im in scene.image_planes:
            if hasattr(im, "origin") and hasattr(im.origin, "x"):
                im.origin = anchor + (im.origin - anchor) * factor
            if hasattr(im, "u") and hasattr(im.u, "x"):
                im.u = im.u * factor
            if hasattr(im, "v") and hasattr(im.v, "x"):
                im.v = im.v * factor

    # 6. Bump scene versions so caches recognize mutation
    if hasattr(scene, "version"):
        scene.version += 1
    if hasattr(scene, "view_version"):
        scene.view_version += 1

    # 7. Invalidate viewport caches (vital for billboards and group chunks)
    if viewport is not None:
        try:
            if hasattr(viewport, "reset_document_caches"):
                viewport.reset_document_caches()
            else:
                for attr in ("_billboard_world", "_fp_memo", "_group_chunks", "_inst_chunks"):
                    c = getattr(viewport, attr, None)
                    if isinstance(c, dict):
                        c.clear()
                viewport._billboard_groups = None
                viewport._edges_version = -1
        except Exception:
            pass


def _scale_group_recursive(group, factor: float, anchor: QVector3D) -> None:
    """Recursively scales a Group around anchor.
    Handles component xform, classic group mesh, UVWs, local axes, and billboard textures."""
    # Component instance: compose xform with scale matrix around anchor
    if hasattr(group, "xform") and group.xform is not None:
        mat = QMatrix4x4()
        mat.translate(anchor)
        mat.scale(factor, factor, factor)
        mat.translate(-anchor)
        group.xform = mat * group.xform
    elif hasattr(group, "mesh") and group.mesh is not None:
        gmesh = group.mesh
        if hasattr(gmesh, "vertices"):
            for v in list(gmesh.vertices):
                if hasattr(v, "position") and hasattr(v.position, "x"):
                    pos = v.position
                    new_pos = anchor + (pos - anchor) * factor
                    if hasattr(gmesh, "place_vertex"):
                        gmesh.place_vertex(v, new_pos)
                    else:
                        v.position = new_pos
            if hasattr(gmesh, "_chunk_dirty"):
                gmesh._chunk_dirty = True

        # Remap UVWs & carry axes for classic groups
        try:
            from core.group import _remap_uvws, carry_axes
            m = QMatrix4x4()
            m.translate(anchor)
            m.scale(factor, factor, factor)
            m.translate(-anchor)
            _remap_uvws(gmesh, m)
            carry_axes(group, m)
        except Exception:
            pass

    # Billboard scale figure: scale texture bounding dimensions
    if getattr(group, "billboard", False) and hasattr(group, "mesh") and group.mesh is not None:
        for f in group.mesh.faces:
            if f.attrs and "texture" in f.attrs:
                t = f.attrs["texture"]
                if "sw" in t and t["sw"] is not None:
                    t["sw"] = float(t["sw"]) * factor
                if "sh" in t and t["sh"] is not None:
                    t["sh"] = float(t["sh"]) * factor

    if hasattr(group, "children") and group.children:
        for child in group.children:
            _scale_group_recursive(child, factor, anchor)


try:
    from core.history import Command
except Exception:
    class Command:  # type: ignore
        def do(self, scene) -> None: pass
        def undo(self, scene) -> None: pass


class ScaleMatchSceneCommand(Command):
    """Reversible command that scales the entire scene and perspective match distance."""

    def __init__(self, factor: float, plugin: Any, anchor: Optional[QVector3D] = None) -> None:
        self.factor = float(factor)
        self.anchor = QVector3D(anchor) if anchor is not None else QVector3D(0.0, 0.0, 0.0)
        self.plugin = plugin
        self._executed = False

    def do(self, scene) -> None:
        if self._executed:
            self.plugin._apply_scale_factor(self.factor, self.anchor, register_history=False)
        else:
            self._executed = True

    def undo(self, scene) -> None:
        inv_factor = 1.0 / self.factor if abs(self.factor) > 1e-6 else 1.0
        self.plugin._apply_scale_factor(inv_factor, self.anchor, register_history=False)


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
        self.image_b64: str = ""
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

        self.z1_a = [0.40, 0.22]
        self.z1_b = [0.38, 0.72]
        self.z2_a = [0.60, 0.22]
        self.z2_b = [0.62, 0.72]

        self.origin = [0.50, 0.72]

        # Saved Views / Scenes System
        # View 0: Perspective Match (photo alignment)
        # View 1: Perspective (free 3D orbit)
        self.active_view_index: int = 0
        self.saved_views: List[Dict[str, Any]] = [
            {
                "name": "Perspective Match",
                "type": "match",
            },
            {
                "name": "Perspective",
                "type": "orbit",
                "camera": {
                    "target": [0.0, 0.0, 0.0],
                    "distance": 35.0,
                    "yaw": -0.785398,
                    "pitch": 0.523598,
                    "fov_deg": 45.0,
                    "two_point": False,
                    "perspective": True,
                }
            }
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes all match settings, guides, views, and photo backup."""
        return {
            "has_perspective_match": True,
            "enabled": self.enabled,
            "active_view_index": self.active_view_index,
            "saved_views": list(self.saved_views),
            "image_path": self.image_path,
            "image_b64": self.image_b64,
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
        """Restores persisted configuration without resetting guides or photo."""
        if not isinstance(d, dict):
            return

        # Keep master toggle disabled by default on file load per requirement
        self.enabled = False

        self.image_path = str(d.get("image_path", self.image_path))
        self.image_b64 = str(d.get("image_b64", self.image_b64))
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
            self.active_view_index = min(len(self.saved_views) - 1, int(d.get("active_view_index", 0)))


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
        self.up_w: QVector3D = QVector3D(0.0, 0.0, 1.0)
        self.v_x_cam: List[float] = [1.0, 0.0, 0.0]
        self.v_y_cam: List[float] = [0.0, 1.0, 0.0]
        self.v_z_cam: List[float] = [0.0, 0.0, 1.0]
        self.vp_x: Optional[Tuple[float, float]] = None
        self.vp_y: Optional[Tuple[float, float]] = None
        self.vp_z: Optional[Tuple[float, float]] = None
        self.horizon: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None


def mat3_inv(m: List[List[float]]) -> Optional[List[List[float]]]:
    """Inverts a 3x3 matrix using Cramer's rule. Returns None if singular."""
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if abs(det) < 1e-12:
        return None
    invdet = 1.0 / det
    return [
        [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invdet,
         (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet,
         (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet],
        [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet,
         (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet,
         (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invdet],
        [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invdet,
         (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invdet,
         (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invdet],
    ]


def nearest_rotation_matrix(m: List[List[float]]) -> List[List[float]]:
    """Finds the nearest orthogonal rotation matrix R in SO(3) via polar decomposition.
    Uses numpy SVD if available; otherwise falls back to pure-Python Newton polar iteration."""
    try:
        import numpy as np
        M = np.array(m, dtype=float)
        U, _, Vt = np.linalg.svd(M)
        d = np.linalg.det(U @ Vt)
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        return R.tolist()
    except Exception:
        pass

    cur = [row[:] for row in m]
    for _ in range(8):
        inv = mat3_inv(cur)
        if inv is None:
            break
        next_m = [[0.0] * 3 for _ in range(3)]
        diff = 0.0
        for i in range(3):
            for j in range(3):
                val = 0.5 * (cur[i][j] + inv[j][i])
                diff = max(diff, abs(val - cur[i][j]))
                next_m[i][j] = val
        cur = next_m
        if diff < 1e-7:
            break

    det = (cur[0][0] * (cur[1][1] * cur[2][2] - cur[1][2] * cur[2][1]) -
           cur[0][1] * (cur[1][0] * cur[2][2] - cur[1][2] * cur[2][0]) +
           cur[0][2] * (cur[1][0] * cur[2][1] - cur[1][1] * cur[2][0]))
    if det < 0:
        for i in range(3):
            cur[i][2] = -cur[i][2]
    return cur


def solve_perspective(
    data: PerspectiveMatchData,
    view_width: float,
    view_height: float
) -> SolvedCameraParams:
    """Solves focal length, FOV, camera rotation, and eye position from guides.
    Supports both 2-point mode and full 3-point perspective with active Z-axis influence."""
    res = SolvedCameraParams()
    W, H = max(view_width, 100.0), max(view_height, 100.0)
    cx, cy = W / 2.0, H / 2.0

    def to_px(rel_pt: List[float]) -> Tuple[float, float]:
        return (rel_pt[0] * W, rel_pt[1] * H)

    p_x1_a, p_x1_b = to_px(data.x1_a), to_px(data.x1_b)
    p_x2_a, p_x2_b = to_px(data.x2_a), to_px(data.x2_b)

    p_y1_a, p_y1_b = to_px(data.y1_a), to_px(data.y1_b)
    p_y2_a, p_y2_b = to_px(data.y2_a), to_px(data.y2_b)

    p_z1_a, p_z1_b = to_px(data.z1_a), to_px(data.z1_b)
    p_z2_a, p_z2_b = to_px(data.z2_a), to_px(data.z2_b)

    # Calculate 2D vanishing points
    vp_x = intersect_lines_2d(p_x1_a, p_x1_b, p_x2_a, p_x2_b)
    vp_y = intersect_lines_2d(p_y1_a, p_y1_b, p_y2_a, p_y2_b)
    vp_z = intersect_lines_2d(p_z1_a, p_z1_b, p_z2_a, p_z2_b)
    res.vp_x = vp_x
    res.vp_y = vp_y
    res.vp_z = vp_z

    if vp_x is None or vp_y is None:
        res.status_msg = _t("Lines nearly parallel (adjust guides)")
        return res

    # 2D Screen to Camera coordinate offsets (Y inverted in camera frame)
    du_x = vp_x[0] - cx
    dv_x = -(vp_x[1] - cy)
    du_y = vp_y[0] - cx
    dv_y = -(vp_y[1] - cy)

    dot_xy = du_x * du_y + dv_x * dv_y

    # Solve focal length f
    if data.mode == "3point" and vp_z is not None:
        du_z = vp_z[0] - cx
        dv_z = -(vp_z[1] - cy)
        dot_yz = du_y * du_z + dv_y * dv_z
        dot_zx = du_z * du_x + dv_z * du_x  # du_z * du_x + dv_z * dv_x
        dot_zx = du_z * du_x + dv_z * dv_x

        cands: List[float] = []
        if dot_xy < -1.0:
            cands.append(-dot_xy)
        if dot_yz < -1.0:
            cands.append(-dot_yz)
        if dot_zx < -1.0:
            cands.append(-dot_zx)

        if cands:
            f = math.sqrt(sum(cands) / len(cands))
            res.status_msg = _t("3-point perspective solved")
            res.valid = True
        elif dot_xy < -1.0:
            f = math.sqrt(-dot_xy)
            res.status_msg = _t("Lines converging properly")
            res.valid = True
        else:
            f = (H / 2.0) / math.tan(math.radians(45.0) / 2.0)
            res.status_msg = _t("Invalid vanishing point configuration")
    else:
        if dot_xy >= -1.0:
            res.status_msg = _t("Invalid vanishing point configuration")
            f = (H / 2.0) / math.tan(math.radians(45.0) / 2.0)
        else:
            f = math.sqrt(-dot_xy)
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

    if data.mode == "3point":
        # Form Z ray from guides
        if vp_z is not None:
            du_z = vp_z[0] - cx
            dv_z = -(vp_z[1] - cy)
            rz = [du_z, dv_z, -f]
        else:
            # Parallel Z lines on screen
            dx_z = ((p_z1_b[0] - p_z1_a[0]) + (p_z2_b[0] - p_z2_a[0])) * 0.5
            dy_z = ((p_z1_b[1] - p_z1_a[1]) + (p_z2_b[1] - p_z2_a[1])) * 0.5
            rz = [dx_z, -dy_z, 0.0]

        len_z = math.hypot(*rz)
        if len_z > 1e-6:
            rz = [c / len_z for c in rz]
        else:
            rz = [0.0, 1.0, 0.0]

        # In camera frame, +Y is UP. For world +Z pointing upwards in scene:
        if rz[1] < 0:
            rz = [-c for c in rz]
        if data.invert_z:
            rz = [-c for c in rz]

        # Right-handed chirality test: (rx x ry) . rz > 0
        nx = rx[1] * ry[2] - rx[2] * ry[1]
        ny = rx[2] * ry[0] - rx[0] * ry[2]
        nz = rx[0] * ry[1] - rx[1] * ry[0]
        dot_n = nx * rz[0] + ny * rz[1] + nz * rz[2]
        if dot_n < 0:
            ry = [-c for c in ry]

        # Solve nearest orthogonal rotation matrix via polar decomposition
        M = [
            [rx[0], ry[0], rz[0]],
            [rx[1], ry[1], rz[1]],
            [rx[2], ry[2], rz[2]]
        ]
        R = nearest_rotation_matrix(M)
        vx = [R[0][0], R[1][0], R[2][0]]
        vy = [R[0][1], R[1][1], R[2][1]]
        vz = [R[0][2], R[1][2], R[2][2]]
    else:
        # 2-point mode: verticals stay strictly vertical
        vx = rx
        dot_xy_cam = vx[0] * ry[0] + vx[1] * ry[1] + vx[2] * ry[2]
        vy = [ry[0] - dot_xy_cam * vx[0], ry[1] - dot_xy_cam * vx[1], ry[2] - dot_xy_cam * vx[2]]
        len_vy = math.hypot(*vy)
        if len_vy > 1e-6:
            vy = [c / len_vy for c in vy]

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

    # Forward direction in world coordinates: -Row 2 of R (where -Z_cam is forward)
    fwd_x = -vx[2]
    fwd_y = -vy[2]
    fwd_z = -vz[2]
    len_fwd = math.hypot(fwd_x, fwd_y, fwd_z)
    if len_fwd > 1e-6:
        fwd_x /= len_fwd
        fwd_y /= len_fwd
        fwd_z /= len_fwd
    res.forward = QVector3D(fwd_x, fwd_y, fwd_z)

    # Up direction in world coordinates: Row 1 of R (where +Y_cam is up)
    up_x = vx[1]
    up_y = vy[1]
    up_z = vz[1]
    len_up = math.hypot(up_x, up_y, up_z)
    if len_up > 1e-6:
        up_x /= len_up
        up_y /= len_up
        up_z /= len_up
    res.up_w = QVector3D(up_x, up_y, up_z)

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
        self.hovered_scale_btn: bool = False
        self.is_shift_held: bool = False
        self.last_mouse_pos: Optional[QPointF] = None

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        t = event.type()
        vp = self.plugin.app.viewport

        # 1. Track Shift key for precision dragging and loupe display
        if t == QEvent.KeyPress:
            if event.key() == Qt.Key_Shift:
                self.is_shift_held = True
                if self.plugin.data.enabled:
                    vp.update()
        elif t == QEvent.KeyRelease:
            if event.key() == Qt.Key_Shift:
                self.is_shift_held = False
                if self.plugin.data.enabled:
                    vp.update()

        # If perspective match is disabled and camera is not locked, leave viewport interactions 100% untouched
        if not self.plugin.data.enabled and not self.plugin.data.camera_locked:
            return False

        # 2. Top Scene Tabs Bar Interactions (When Match is Enabled)
        if t == QEvent.MouseMove and not (event.buttons() & Qt.LeftButton):
            pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
            hit_tab = self._hit_test_tabs(pos.x(), pos.y())
            hit_add = (self.plugin.add_view_rect is not None and self.plugin.add_view_rect.contains(pos))
            hit_scale = (self.plugin.scale_tool_rect is not None and self.plugin.scale_tool_rect.contains(pos))

            changed = False
            if hit_tab != self.hovered_tab_index:
                self.hovered_tab_index = hit_tab
                changed = True
            if hit_add != self.hovered_add_btn:
                self.hovered_add_btn = hit_add
                changed = True
            if hit_scale != self.hovered_scale_btn:
                self.hovered_scale_btn = hit_scale
                changed = True

            if hit_tab is not None or hit_add or hit_scale:
                vp.setCursor(Qt.PointingHandCursor)
                if changed:
                    vp.update()
                return False
            else:
                if self.hovered_handle is None and (
                    self.hovered_tab_index is not None or self.hovered_add_btn or self.hovered_scale_btn
                ):
                    vp.unsetCursor()
                if changed:
                    vp.update()

        elif t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            pos = event.position() if hasattr(event, "position") else QPointF(event.pos())

            # Scene tabs click
            tab_idx = self._hit_test_tabs(pos.x(), pos.y())
            if tab_idx is not None:
                self.plugin.switch_to_view(tab_idx)
                vp.update()
                return True

            # Add view button click [+]
            if self.plugin.add_view_rect is not None and self.plugin.add_view_rect.contains(pos):
                self.plugin.add_current_view()
                vp.update()
                return True

            # Calibrate scale button click [📏]
            if self.plugin.scale_tool_rect is not None and self.plugin.scale_tool_rect.contains(pos):
                self.plugin.open_scale_dialog()
                vp.update()
                return True

        # 3. Camera Lock Enforcement (Blocks Orbit, Pan, Zoom, and Rotation)
        if self.plugin.data.camera_locked:
            # Enforce camera immutability if camera drifted
            if self.plugin.locked_camera_state is not None and hasattr(vp, "camera"):
                cur = get_camera_state(vp.camera)
                if (cur.get("yaw") != self.plugin.locked_camera_state.get("yaw") or
                    cur.get("pitch") != self.plugin.locked_camera_state.get("pitch") or
                    cur.get("distance") != self.plugin.locked_camera_state.get("distance") or
                    cur.get("target") != self.plugin.locked_camera_state.get("target")):
                    apply_camera_state(vp.camera, self.plugin.locked_camera_state)
                    vp.update()

            # Block mouse wheel zoom
            if t == QEvent.Wheel:
                self.plugin._show_status_message(_t("🔒 Camera is locked. Click 'Locked' in panel to unlock."), 2500)
                return True

            # Block Middle Mouse button orbit and pan
            if t in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseMove, QEvent.MouseButtonDblClick):
                btn = getattr(event, "button", None)
                btns = getattr(event, "buttons", None)
                is_mid = False
                if callable(btn):
                    is_mid = (btn() == Qt.MiddleButton)
                elif isinstance(btn, Qt.MouseButton):
                    is_mid = (btn == Qt.MiddleButton)
                if not is_mid and callable(btns):
                    is_mid = bool(btns() & Qt.MiddleButton)
                elif not is_mid and isinstance(btns, Qt.MouseButtons):
                    is_mid = bool(btns & Qt.MiddleButton)

                if is_mid:
                    if t == QEvent.MouseButtonPress:
                        self.plugin._show_status_message(_t("🔒 Camera is locked. Click 'Locked' in panel to unlock."), 2500)
                    return True

            # Block viewport navigation tools (Orbit, Pan, Zoom, Look, Walk)
            nav = getattr(vp, "nav_mode", None)
            if nav in ("orbit", "pan", "zoom", "zoom_window", "look", "walk"):
                if t in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseMove, QEvent.MouseButtonDblClick):
                    if t == QEvent.MouseButtonPress:
                        self.plugin._show_status_message(_t("🔒 Camera is locked. Click 'Locked' in panel to unlock."), 2500)
                    return True

            # In Match tab, if user attempts to click a guide handle while locked, notify them
            is_match_tab_check = (
                self.plugin.data.active_view_index < len(self.plugin.data.saved_views)
                and self.plugin.data.saved_views[self.plugin.data.active_view_index].get("type") == "match"
            )
            if is_match_tab_check and self.plugin.data.show_guides:
                if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
                    hit = self._hit_test(pos.x(), pos.y(), vp.width(), vp.height())
                    if hit is not None:
                        self.plugin._show_status_message(_t("🔒 Camera is locked. Click 'Locked' in panel to unlock."), 2500)
                        return True

        # 4. Handle 3D Orbit Camera Tracking (when in Perspective / Orbit tabs)
        is_match_tab = (
            self.plugin.data.active_view_index < len(self.plugin.data.saved_views)
            and self.plugin.data.saved_views[self.plugin.data.active_view_index].get("type") == "match"
        )

        if not is_match_tab:
            # User is orbiting in standard 3D perspective view
            if t == QEvent.MouseButtonRelease and event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
                curr_idx = self.plugin.data.active_view_index
                if 0 <= curr_idx < len(self.plugin.data.saved_views):
                    self.plugin.data.saved_views[curr_idx]["camera"] = get_camera_state(vp.camera)
                    self.plugin.save_state()
            return False

        # 4. Guide Handles Dragging (ONLY when in Perspective Match tab)
        if not self.plugin.data.show_guides or self.plugin.data.camera_locked:
            return False

        if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
            hit = self._hit_test(pos.x(), pos.y(), vp.width(), vp.height())
            if hit is not None:
                self.active_handle = hit
                self.last_mouse_pos = pos
                self.is_shift_held = bool(event.modifiers() & Qt.ShiftModifier)
                vp.setCursor(Qt.ClosedHandCursor)
                vp.update()
                return True

        elif t == QEvent.MouseMove:
            pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
            w, h = max(vp.width(), 100), max(vp.height(), 100)

            if self.active_handle is not None and (event.buttons() & Qt.LeftButton):
                self.is_shift_held = bool(event.modifiers() & Qt.ShiftModifier)

                if self.is_shift_held and self.last_mouse_pos is not None:
                    # Precision slow-motion mode (0.2x delta)
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
                elif not (self.hovered_tab_index is not None or self.hovered_add_btn or self.hovered_scale_btn):
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
    """Side panel for controlling perspective match photo, guides, scenes, and scale."""

    def __init__(self, plugin: "PerspectiveMatcherPlugin") -> None:
        super().__init__()
        self.plugin = plugin
        self._init_ui()

    def _init_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        # --- Master Enable / Disable Toggle Checkbox ---
        self.chk_match_enabled = QCheckBox(_t("Enable Perspective Match"))
        self.chk_match_enabled.setStyleSheet(
            "QCheckBox { font-size: 13px; font-weight: bold; color: #4CAF50; padding: 4px; }"
            if self.plugin.data.enabled else
            "QCheckBox { font-size: 13px; font-weight: bold; color: #B0BEC5; padding: 4px; }"
        )
        self.chk_match_enabled.setChecked(self.plugin.data.enabled)
        self.chk_match_enabled.toggled.connect(self._on_master_toggle_toggled)
        lay.addWidget(self.chk_match_enabled)

        # --- Saved Views & Scenes Section ---
        scenes_grp = QGroupBox(_t("Saved Views & Scenes"))
        scenes_lay = QVBoxLayout(scenes_grp)
        scenes_lay.setSpacing(6)

        self.views_list = QListWidget()
        self.views_list.currentRowChanged.connect(self._on_view_selected)
        self.views_list.itemClicked.connect(lambda item: self._on_view_selected(self.views_list.row(item)))
        self.views_list.itemDoubleClicked.connect(lambda _i: self._on_rename_view())
        scenes_lay.addWidget(self.views_list)

        views_btn_row = QHBoxLayout()
        self.btn_add_view = QPushButton(_t("Save Current"))
        self.btn_add_view.clicked.connect(self._on_add_view)
        self.btn_update_view = QPushButton(_t("Update Camera"))
        self.btn_update_view.clicked.connect(self._on_update_view)
        self.btn_lock_cam = QPushButton(_t("🔒 Lock"))
        self.btn_lock_cam.setToolTip(_t("Lock camera so it cannot move, rotate, or zoom."))
        self.btn_lock_cam.clicked.connect(self._on_lock_cam_clicked)
        self.btn_del_view = QPushButton(_t("Delete"))
        self.btn_del_view.clicked.connect(self._on_delete_view)
        views_btn_row.addWidget(self.btn_add_view)
        views_btn_row.addWidget(self.btn_update_view)
        views_btn_row.addWidget(self.btn_lock_cam)
        views_btn_row.addWidget(self.btn_del_view)
        scenes_lay.addLayout(views_btn_row)

        lay.addWidget(scenes_grp)

        # --- Real-World Scale & Distance Section ---
        scale_grp = QGroupBox(_t("Real-World Scale & Distance"))
        scale_lay = QVBoxLayout(scale_grp)
        scale_lay.setSpacing(6)

        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel(_t("Camera Distance:")))
        self.spin_dist = QDoubleSpinBox()
        self.spin_dist.setRange(0.5, 50000.0)
        self.spin_dist.setDecimals(2)
        self.spin_dist.setSingleStep(1.0)
        self.spin_dist.setValue(self.plugin.data.distance)
        self.spin_dist.setSuffix(" m")
        self.spin_dist.valueChanged.connect(self._on_dist_spin_changed)
        dist_row.addWidget(self.spin_dist)
        scale_lay.addLayout(dist_row)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel(_t("Match Scale:")))
        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setRange(10, 2000)  # 1.0m to 200.0m with 0.1m step
        self.slider_scale.setValue(min(2000, max(10, int(self.plugin.data.distance * 10))))
        self.slider_scale.sliderPressed.connect(self._on_scale_slider_pressed)
        self.slider_scale.valueChanged.connect(self._on_scale_slider_changed)
        self.slider_scale.sliderReleased.connect(self._on_scale_slider_released)
        slider_row.addWidget(self.slider_scale)
        scale_lay.addLayout(slider_row)

        self.chk_scale_scene_with_view = QCheckBox(_t("Scale Model with View (Fixed on Photo)"))
        self.chk_scale_scene_with_view.setChecked(True)
        self.chk_scale_scene_with_view.setStyleSheet("font-weight: bold; color: #64B5F6;")
        self.chk_scale_scene_with_view.setToolTip(
            _t("When enabled, adjusting distance/scale scales 3D scene geometry proportionally so the volume remains 100% fixed on top of the background photograph.")
        )
        scale_lay.addWidget(self.chk_scale_scene_with_view)

        # Selected Edge Calibration Sub-section
        self.lbl_selected_edge = QLabel(_t("Select an edge in viewport to calibrate real scale:"))
        self.lbl_selected_edge.setStyleSheet("color: #AAA; font-size: 11px;")
        self.lbl_selected_edge.setWordWrap(True)
        scale_lay.addWidget(self.lbl_selected_edge)

        cur_row = QHBoxLayout()
        cur_row.addWidget(QLabel(_t("Current Length:")))
        self.spin_current_len = QDoubleSpinBox()
        self.spin_current_len.setRange(0.001, 100000.0)
        self.spin_current_len.setDecimals(3)
        self.spin_current_len.setValue(1.0)
        self.spin_current_len.setSuffix(" m")
        cur_row.addWidget(self.spin_current_len)
        scale_lay.addLayout(cur_row)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel(_t("Real Length:")))
        self.spin_target_len = QDoubleSpinBox()
        self.spin_target_len.setRange(0.001, 100000.0)
        self.spin_target_len.setDecimals(3)
        self.spin_target_len.setValue(5.0)
        self.spin_target_len.setSuffix(" m")
        target_row.addWidget(self.spin_target_len)
        scale_lay.addLayout(target_row)

        btn_scale_row = QHBoxLayout()
        self.btn_measure_edge = QPushButton(_t("Read Edge"))
        self.btn_measure_edge.clicked.connect(self._on_read_selected_edge)

        self.btn_apply_scale = QPushButton(_t("Apply Scale"))
        self.btn_apply_scale.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 5px;")
        self.btn_apply_scale.clicked.connect(self._on_apply_scale)

        btn_scale_row.addWidget(self.btn_measure_edge)
        btn_scale_row.addWidget(self.btn_apply_scale)
        scale_lay.addLayout(btn_scale_row)

        lay.addWidget(scale_grp)

        # --- Reference Photograph Section ---
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
        # 1. Refresh Master Checkbox
        self.chk_match_enabled.blockSignals(True)
        self.chk_match_enabled.setChecked(self.plugin.data.enabled)
        if self.plugin.data.enabled:
            self.chk_match_enabled.setStyleSheet(
                "QCheckBox { font-size: 13px; font-weight: bold; color: #4CAF50; padding: 4px; }"
            )
        else:
            self.chk_match_enabled.setStyleSheet(
                "QCheckBox { font-size: 13px; font-weight: bold; color: #B0BEC5; padding: 4px; }"
            )
        self.chk_match_enabled.blockSignals(False)

        # 2. Refresh Views List
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

        # 3. Check for selected edge in viewport
        edge_info = self.plugin.get_selected_edge_info()
        if edge_info is not None:
            cur_len, desc = edge_info
            self.lbl_selected_edge.setText(f"🟢 {_t('Detected:')} {desc}")
            self.lbl_selected_edge.setStyleSheet("color: #4CAF50; font-size: 11px;")
            self.spin_current_len.blockSignals(True)
            self.spin_current_len.setValue(cur_len)
            self.spin_current_len.blockSignals(False)

        if self.plugin.pixmap is not None and not self.plugin.pixmap.isNull():
            base_name = os.path.basename(self.plugin.data.image_path) or "Photo Loaded"
            self.lbl_path.setText(f"🟢 {base_name}")
            self.lbl_path.setStyleSheet("color: #4CAF50; font-size: 11px;")
        elif self.plugin.data.image_path:
            base_name = os.path.basename(self.plugin.data.image_path)
            self.lbl_path.setText(f"⚠️ {base_name} (file not found)")
            self.lbl_path.setStyleSheet("color: #FFA726; font-size: 11px;")
        else:
            self.lbl_path.setText(_t("No photo loaded"))
            self.lbl_path.setStyleSheet("color: #888; font-size: 11px;")

        for w in (self.slider_opacity, self.chk_show_photo, self.chk_show_guides,
                  self.chk_lock_cam, self.spin_dist, self.slider_scale, self.chk_inv_x, self.chk_inv_y,
                  self.chk_inv_z, self.chk_swap_xy, self.combo_mode):
            w.blockSignals(True)

        self.slider_opacity.setValue(int(self.plugin.data.image_opacity * 100))
        self.lbl_opacity.setText(f"{int(self.plugin.data.image_opacity * 100)}%")
        self.chk_show_photo.setChecked(self.plugin.data.show_image)
        self.chk_show_guides.setChecked(self.plugin.data.show_guides)
        self.chk_lock_cam.setChecked(self.plugin.data.camera_locked)
        self.spin_dist.setValue(self.plugin.data.distance)
        self.slider_scale.setValue(min(2000, max(10, int(self.plugin.data.distance * 10))))
        self.chk_inv_x.setChecked(self.plugin.data.invert_x)
        self.chk_inv_y.setChecked(self.plugin.data.invert_y)
        self.chk_inv_z.setChecked(self.plugin.data.invert_z)
        self.chk_swap_xy.setChecked(self.plugin.data.swap_xy)
        idx = self.combo_mode.findData(self.plugin.data.mode)
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)

        for w in (self.slider_opacity, self.chk_show_photo, self.chk_show_guides,
                  self.chk_lock_cam, self.spin_dist, self.slider_scale, self.chk_inv_x, self.chk_inv_y,
                  self.chk_inv_z, self.chk_swap_xy, self.combo_mode):
            w.blockSignals(False)

        if hasattr(self, "btn_lock_cam"):
            if self.plugin.data.camera_locked:
                self.btn_lock_cam.setText(_t("🔒 Locked"))
                self.btn_lock_cam.setToolTip(_t("Camera is locked (cannot move, rotate, or zoom). Click to unlock."))
                self.btn_lock_cam.setStyleSheet(
                    "QPushButton { background-color: #E65100; color: #FFFFFF; font-weight: bold; border-radius: 4px; padding: 4px 8px; }"
                    "QPushButton:hover { background-color: #F57C00; }"
                )
            else:
                self.btn_lock_cam.setText(_t("🔓 Lock"))
                self.btn_lock_cam.setToolTip(_t("Lock camera so it cannot move, rotate, or zoom."))
                self.btn_lock_cam.setStyleSheet("")

    def update_readouts(self, solved: SolvedCameraParams) -> None:
        self.lbl_focal.setText(f"{solved.focal_35mm:.1f} mm (35mm eq.)")
        self.lbl_fov.setText(f"{solved.fov_deg:.1f}°")
        self.lbl_orient.setText(f"Pitch: {solved.pitch_deg:.1f}° | Yaw: {solved.yaw_deg:.1f}°")
        self.lbl_status.setText(solved.status_msg)
        if solved.valid:
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet("color: #FF9800; font-weight: bold;")

    def _on_master_toggle_toggled(self, checked: bool) -> None:
        self.plugin.toggle_enabled(checked)

    def _on_view_selected(self, row: int) -> None:
        if row >= 0:
            self.plugin.switch_to_view(row)

    def _on_add_view(self) -> None:
        self.plugin.add_current_view()

    def _on_update_view(self) -> None:
        self.plugin.update_active_view_camera()

    def _on_delete_view(self) -> None:
        self.plugin.delete_view(self.plugin.data.active_view_index)

    def _on_rename_view(self) -> None:
        self.plugin.rename_view(self.plugin.data.active_view_index)

    def _on_read_selected_edge(self) -> None:
        info = self.plugin.get_selected_edge_info()
        if info is not None:
            length, desc = info
            self.spin_current_len.setValue(length)
            self.lbl_selected_edge.setText(f"🟢 {_t('Detected:')} {desc}")
            self.lbl_selected_edge.setStyleSheet("color: #4CAF50; font-size: 11px;")
        else:
            self.lbl_selected_edge.setText(
                _t("No edge selected in viewport. Select an edge, or enter Current Length manually.")
            )
            self.lbl_selected_edge.setStyleSheet("color: #FFA726; font-size: 11px;")

    def _on_apply_scale(self) -> None:
        cur_len = self.spin_current_len.value()
        tgt_len = self.spin_target_len.value()
        if cur_len <= 1e-6 or tgt_len <= 1e-6:
            return

        ok, msg = self.plugin.scale_model_to_edge(cur_len, tgt_len)
        if ok:
            self.lbl_selected_edge.setText(f"✅ {msg}")
            self.lbl_selected_edge.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11px;")
            self.spin_current_len.setValue(tgt_len)
            self.refresh_ui()
        else:
            self.lbl_selected_edge.setText(f"⚠️ {msg}")
            self.lbl_selected_edge.setStyleSheet("color: #FF5252; font-size: 11px;")

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

    def _on_lock_cam_clicked(self) -> None:
        self._on_toggle_lock_cam(not self.plugin.data.camera_locked)

    def _on_toggle_lock_cam(self, checked: bool) -> None:
        self.plugin.set_camera_locked(checked)

    def _on_mode_changed(self) -> None:
        self.plugin.data.mode = self.combo_mode.currentData()
        self.plugin.update_camera_from_guides()
        self.plugin.save_state()

    def _on_scale_slider_pressed(self) -> None:
        self._drag_start_dist = max(0.5, self.plugin.data.distance)
        self._drag_last_dist = self._drag_start_dist

    def _on_scale_slider_changed(self, val: int) -> None:
        new_dist = max(0.5, val / 10.0)
        self.spin_dist.blockSignals(True)
        self.spin_dist.setValue(new_dist)
        self.spin_dist.blockSignals(False)

        if getattr(self, "_is_internal_scale_sync", False):
            return

        if self.chk_scale_scene_with_view.isChecked():
            last_dist = getattr(self, "_drag_last_dist", self.plugin.data.distance)
            if last_dist > 1e-4 and abs(new_dist - last_dist) > 1e-4:
                factor = new_dist / last_dist
                self._drag_last_dist = new_dist
                self.plugin._apply_scale_factor(factor, register_history=False)
        else:
            self.plugin.data.distance = new_dist
            self.plugin.update_camera_from_guides()
            self.plugin.save_state()

    def _on_scale_slider_released(self) -> None:
        if self.chk_scale_scene_with_view.isChecked():
            start_dist = getattr(self, "_drag_start_dist", None)
            curr_dist = self.plugin.data.distance
            if start_dist is not None and start_dist > 1e-4:
                total_factor = curr_dist / start_dist
                if abs(total_factor - 1.0) > 1e-4:
                    vp = self.plugin.app.viewport
                    if hasattr(vp, "history"):
                        try:
                            cmd = ScaleMatchSceneCommand(total_factor, self.plugin)
                            cmd._executed = True
                            vp.history.undo_stack.append(cmd)
                            vp.history.redo_stack.clear()
                        except Exception:
                            pass
        self.plugin.save_state()
        self._drag_start_dist = None
        self._drag_last_dist = None

    def _on_dist_spin_changed(self, val: float) -> None:
        if getattr(self, "_is_internal_scale_sync", False):
            return
        new_dist = max(0.5, val)
        self._is_internal_scale_sync = True
        try:
            self.slider_scale.blockSignals(True)
            self.slider_scale.setValue(min(2000, max(10, int(new_dist * 10))))
            self.slider_scale.blockSignals(False)

            old_dist = self.plugin.data.distance
            if self.chk_scale_scene_with_view.isChecked() and old_dist > 1e-4 and abs(new_dist - old_dist) > 1e-4:
                factor = new_dist / old_dist
                self.plugin._apply_scale_factor(factor, register_history=True)
            else:
                self.plugin.data.distance = new_dist
                self.plugin.update_camera_from_guides()
                self.plugin.save_state()
        finally:
            self._is_internal_scale_sync = False

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
        d.image_b64 = self.plugin.data.image_b64
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
        self._is_internal_update: bool = False
        self.data = PerspectiveMatchData()
        self.pixmap: Optional[QPixmap] = None
        self.solved = SolvedCameraParams()

        # Cache of hit rects for viewport drawing & event filter
        self.scene_tab_rects: List[Tuple[QRectF, int]] = []
        self.add_view_rect: Optional[QRectF] = None
        self.scale_tool_rect: Optional[QRectF] = None

        # Load persisted document data
        self.load_state_from_document()

        self.locked_camera_state: Optional[Dict[str, Any]] = None
        if self.data.camera_locked:
            vp = getattr(self.app, "viewport", None)
            if vp is not None and hasattr(vp, "camera"):
                self.locked_camera_state = get_camera_state(vp.camera)

        self.panel: Optional[PerspectiveMatcherPanel] = None
        self.panel = PerspectiveMatcherPanel(self)
        self.filter = PerspectiveEventFilter(self)
        app.viewport.installEventFilter(self.filter)

    def _show_status_message(self, msg: str, timeout: int = 3000) -> None:
        try:
            win = getattr(self.app, "window", None)
            if win is not None and hasattr(win, "statusBar"):
                sb = win.statusBar()
                if sb is not None:
                    sb.showMessage(msg, timeout)
        except Exception:
            pass

    def set_camera_locked(self, locked: bool) -> None:
        self.data.camera_locked = bool(locked)
        vp = getattr(self.app, "viewport", None)
        if locked:
            if vp is not None and hasattr(vp, "camera"):
                self.locked_camera_state = get_camera_state(vp.camera)
            self._show_status_message(_t("🔒 Camera locked (movement & rotation frozen)."), 3000)
        else:
            self.locked_camera_state = None
            self._show_status_message(_t("🔓 Camera unlocked."), 2000)
        self.save_state()
        if self.panel is not None:
            self.panel.refresh_ui()
        if vp is not None:
            vp.update()

    def load_state_from_document(self) -> None:
        """Loads perspective match configuration and restores photo without resetting existing state."""
        saved = self.app.document_data(default=None)
        if not saved:
            try:
                pdata = getattr(self.app.scene, "plugin_data", {}) or {}
                saved = pdata.get(self.app.key, None)
            except Exception:
                saved = None

        if saved and isinstance(saved, dict):
            currently_enabled = self.data.enabled
            self.data.from_dict(saved)
            if currently_enabled:
                self.data.enabled = True
            self._restore_photo()

    def _restore_photo(self) -> None:
        """Restores QPixmap from file path, candidate user folders, or embedded base64."""
        # 1. Try exact image_path on disk
        if self.data.image_path and os.path.isfile(self.data.image_path):
            pm = QPixmap(self.data.image_path)
            if pm is not None and not pm.isNull():
                self.pixmap = pm
                self._ensure_b64_backup()
                return

        # 2. Check candidate folders (Pictures, Downloads, Desktop, Documents)
        if self.data.image_path:
            fname = os.path.basename(self.data.image_path)
            candidate_dirs = [
                os.path.expanduser("~/Pictures"),
                os.path.expanduser("~/Downloads"),
                os.path.expanduser("~/Desktop"),
                os.path.expanduser("~/Documents"),
            ]
            for cdir in candidate_dirs:
                cpath = os.path.join(cdir, fname)
                if os.path.isfile(cpath):
                    pm = QPixmap(cpath)
                    if pm is not None and not pm.isNull():
                        self.data.image_path = cpath
                        self.pixmap = pm
                        self._ensure_b64_backup()
                        return

            # Prefix match for UUID or hash filenames (e.g. 2404fc27...)
            prefix = fname[:16] if len(fname) >= 16 else fname
            for cdir in candidate_dirs:
                if os.path.isdir(cdir):
                    try:
                        for entry in os.listdir(cdir):
                            if entry.startswith(prefix) and entry.lower().endswith(
                                (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
                            ):
                                fpath = os.path.join(cdir, entry)
                                pm = QPixmap(fpath)
                                if pm is not None and not pm.isNull():
                                    self.data.image_path = fpath
                                    self.pixmap = pm
                                    self._ensure_b64_backup()
                                    return
                    except Exception:
                        pass

        # 3. Restore from embedded base64 backup
        if self.data.image_b64:
            try:
                ba = QByteArray.fromBase64(self.data.image_b64.encode("ascii"))
                pm = QPixmap()
                pm.loadFromData(ba)
                if not pm.isNull():
                    self.pixmap = pm
                    return
            except Exception:
                pass

        self.pixmap = None

    def _ensure_b64_backup(self) -> None:
        """Ensures image_b64 contains the photo so it survives document transfers."""
        if self.data.image_path and os.path.isfile(self.data.image_path):
            try:
                if not self.data.image_b64:
                    with open(self.data.image_path, "rb") as f:
                        raw = f.read()
                        if len(raw) < 16 * 1024 * 1024:
                            self.data.image_b64 = base64.b64encode(raw).decode("ascii")
            except Exception:
                pass

    def on_document_changed(self) -> None:
        """Callback invoked whenever an existing document is opened or updated."""
        if getattr(self, "_is_internal_update", False):
            return
        self.load_state_from_document()
        if self.panel is not None:
            self.panel.refresh_ui()
        self.app.viewport.update()

    def load_image(self, path: str) -> None:
        """Loads image from disk and creates base64 backup for portability."""
        if os.path.isfile(path):
            self.data.image_path = path
            self.pixmap = QPixmap(path)
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                    if len(raw) < 16 * 1024 * 1024:
                        self.data.image_b64 = base64.b64encode(raw).decode("ascii")
            except Exception:
                pass

            # Automatically enable match view when loading a new photo
            self.data.enabled = True
            self.data.active_view_index = 0
            self.update_camera_from_guides()
            self.save_state()
            if self.panel is not None:
                self.panel.refresh_ui()
            self.app.viewport.update()

    def clear_image(self) -> None:
        self.data.image_path = ""
        self.data.image_b64 = ""
        self.pixmap = None
        self.save_state()
        self.app.viewport.update()

    def save_state(self) -> None:
        """Serializes plugin data into document undo history and direct plugin_data."""
        if getattr(self, "_is_internal_update", False):
            return
        self._is_internal_update = True
        try:
            payload = self.data.to_dict()
            self.app.set_document_data(payload)
            try:
                sc = getattr(self.app, "scene", None) or getattr(self.app.viewport, "scene", None)
                if sc is not None:
                    if not hasattr(sc, "plugin_data") or sc.plugin_data is None:
                        sc.plugin_data = {}
                    sc.plugin_data[self.app.key] = payload
            except Exception:
                pass
        finally:
            self._is_internal_update = False

    def toggle_enabled(self, state: Optional[bool] = None) -> None:
        """Toggles perspective match mode on or off."""
        if state is None:
            self.data.enabled = not self.data.enabled
        else:
            self.data.enabled = bool(state)

        if self.data.enabled:
            # Switch to Perspective Match view
            self.data.active_view_index = 0
            self.update_camera_from_guides()
        else:
            # Revert to standard 3D orbit view
            if self.data.active_view_index == 0 and len(self.data.saved_views) > 1:
                self.switch_to_view(1)

        self.save_state()
        if self.panel is not None:
            self.panel.refresh_ui()
        self.app.viewport.update()

    def switch_to_view(self, index: int) -> None:
        """Switches active camera to a saved scene tab."""
        if index < 0 or index >= len(self.data.saved_views):
            return

        # 1. Before leaving current view, save its 3D camera if not in match tab
        curr_idx = self.data.active_view_index
        if 0 <= curr_idx < len(self.data.saved_views):
            if self.data.saved_views[curr_idx].get("type") != "match":
                self.data.saved_views[curr_idx]["camera"] = get_camera_state(self.app.viewport.camera)

        # 2. Switch to target view
        self.data.active_view_index = index
        view_data = self.data.saved_views[index]
        vtype = view_data.get("type", "custom")

        if vtype == "match":
            # Match view: always ensure match mode is enabled, solve camera & display photo
            self.data.enabled = True
            self.update_camera_from_guides()
        else:
            # 3D orbit view: restore exact last saved camera (yaw, pitch, distance, target)
            cam = self.app.viewport.camera
            if "camera" in view_data:
                apply_camera_state(cam, view_data["camera"])
            else:
                cam.up = QVector3D(0.0, 0.0, 1.0)
                cam.two_point = False

        if self.data.camera_locked:
            vp = getattr(self.app, "viewport", None)
            if vp is not None and hasattr(vp, "camera"):
                self.locked_camera_state = get_camera_state(vp.camera)

        self.save_state()
        if self.panel is not None:
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
        self.switch_to_view(len(self.data.saved_views) - 1)

    def update_active_view_camera(self) -> None:
        """Overwrites the selected view with current viewport camera."""
        idx = self.data.active_view_index
        if idx == 0:
            self.update_camera_from_guides()
        else:
            cam = self.app.viewport.camera
            self.data.saved_views[idx]["camera"] = get_camera_state(cam)
            if self.data.camera_locked:
                self.locked_camera_state = get_camera_state(cam)
            self.save_state()
            self.app.viewport.update()

    def delete_view(self, index: int) -> None:
        """Deletes a custom saved scene view (cannot delete default match or perspective)."""
        if index <= 1:
            return

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
            self.panel, _t("Rename Scene"), _t("Name:"), QLineEdit.Normal, curr_name
        )
        if ok and new_name.strip():
            self.data.saved_views[index]["name"] = new_name.strip()
            self.save_state()
            if self.panel is not None:
                self.panel.refresh_ui()
            self.app.viewport.update()

    def get_selected_edge_info(self) -> Optional[Tuple[float, str]]:
        """Returns (length, description) for the active selection in viewport.scene."""
        try:
            scene = self.app.viewport.scene
        except Exception:
            return None
        return get_selected_edge_info(scene)

    def _apply_scale_factor(self, factor: float, anchor: Optional[QVector3D] = None, register_history: bool = True) -> bool:
        """Scales the entire 3D scene geometry, camera distance, and saved views uniformly.
        Because camera optical rays scale proportionally around the match origin, the 3D volume
        seen through the perspective match camera remains 100% FIXED on top of the background photo."""
        if abs(factor - 1.0) < 1e-6 or factor <= 1e-6:
            return False

        if anchor is None:
            anchor = QVector3D(0.0, 0.0, 0.0)

        vp = self.app.viewport
        scene = getattr(vp, "scene", None)
        if scene is None:
            return False

        self._is_internal_update = True
        try:
            # 1. Scale entire scene geometry uniformly (meshes, groups, billboards, guides, dimensions)
            scale_scene_geometry(scene, factor, anchor=anchor, viewport=vp)

            # 2. Scale camera distance in PerspectiveMatchData
            self.data.distance *= factor

            # 3. Scale all cameras in saved_views (both target and distance)
            for v in self.data.saved_views:
                cam_state = v.get("camera")
                if isinstance(cam_state, dict):
                    if "distance" in cam_state:
                        cam_state["distance"] = float(cam_state["distance"]) * factor
                    if "target" in cam_state and isinstance(cam_state["target"], list) and len(cam_state["target"]) == 3:
                        t = QVector3D(float(cam_state["target"][0]), float(cam_state["target"][1]), float(cam_state["target"][2]))
                        new_t = anchor + (t - anchor) * factor
                        cam_state["target"] = [new_t.x(), new_t.y(), new_t.z()]

            # 4. Re-solve perspective with updated distance
            w, h = max(vp.width(), 100), max(vp.height(), 100)
            self.solved = solve_perspective(self.data, w, h)
            if self.panel is not None:
                self.panel.update_readouts(self.solved)

            # 5. Apply directly to live viewport camera
            is_match_tab = (
                self.data.active_view_index < len(self.data.saved_views)
                and self.data.saved_views[self.data.active_view_index].get("type") == "match"
            )
            if is_match_tab or self.data.enabled:
                cam = vp.camera
                cam.fov_deg = self.solved.fov_deg
                cam.distance = self.data.distance
                if self.data.mode == "3point":
                    cam.up = self.solved.up_w
                    cam.two_point = False
                else:
                    cam.up = QVector3D(0.0, 0.0, 1.0)
                    cam.two_point = True
                cam.look_from(self.solved.eye, self.solved.forward)
            else:
                cam = vp.camera
                cam.distance *= factor
                new_t = anchor + (cam.target - anchor) * factor
                cam.target = new_t

            if self.data.camera_locked:
                self.locked_camera_state = get_camera_state(cam)

            # 6. Save state to document
            payload = self.data.to_dict()
            self.app.set_document_data(payload)
            try:
                sc = getattr(self.app, "scene", None) or getattr(self.app.viewport, "scene", None)
                if sc is not None:
                    if not hasattr(sc, "plugin_data") or sc.plugin_data is None:
                        sc.plugin_data = {}
                    sc.plugin_data[self.app.key] = payload
            except Exception:
                pass

            # 7. Register undo history if requested
            if register_history and hasattr(vp, "history"):
                try:
                    cmd = ScaleMatchSceneCommand(factor, self, anchor)
                    cmd._executed = True
                    vp.history.undo_stack.append(cmd)
                    vp.history.redo_stack.clear()
                except Exception:
                    pass

            # 8. Force full viewport redraw
            vp.update()
            return True
        finally:
            self._is_internal_update = False

    def scale_model_to_edge(self, current_len: float, target_len: float) -> Tuple[bool, str]:
        """Scales the entire 3D scene geometry and camera distance to match the real dimension.
        Preserves camera optical rays so the photo perspective match is 100% untouched!"""
        if current_len <= 1e-6 or target_len <= 1e-6:
            return False, "Invalid distance values"

        factor = target_len / current_len
        if abs(factor - 1.0) < 1e-6:
            return True, "Scale is already 1.0×"

        ok = self._apply_scale_factor(factor, anchor=QVector3D(0.0, 0.0, 0.0), register_history=True)
        if ok:
            if self.panel is not None:
                self.panel.refresh_ui()
            msg = f"Scaled entire scene by {factor:.3f}×. Edge is now {target_len:.3f} m (Photo match preserved)"
            return True, msg
        return False, "Failed to apply scene scale"

    def open_scale_dialog(self) -> None:
        """Opens interactive modal dialog to calibrate scene scale and adjust match distance."""
        info = self.get_selected_edge_info()
        init_cur = info[0] if info else 1.0

        parent_win = self.panel.window() if self.panel is not None else None
        dlg = QDialog(parent_win)
        dlg.setWindowTitle(_t("Real-World Scale & Distance Calibration"))
        dlg.setFixedWidth(400)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(12)

        hdr_lbl = QLabel(
            _t("Scale the entire scene and perspective camera together.\n"
               "The volume will remain 100% locked to the photo view.")
        )
        hdr_lbl.setStyleSheet("color: #90CAF9; font-size: 11px;")
        hdr_lbl.setWordWrap(True)
        lay.addWidget(hdr_lbl)

        info_lbl = QLabel(
            f"🟢 {_t('Detected:')} {info[1]}" if info else _t("⚪ Select an edge in viewport or enter lengths below:")
        )
        info_lbl.setStyleSheet("font-size: 11px; color: #4CAF50;" if info else "font-size: 11px; color: #FFA726;")
        info_lbl.setWordWrap(True)
        lay.addWidget(info_lbl)

        form = QFormLayout()
        form.setSpacing(8)

        cur_spin = QDoubleSpinBox()
        cur_spin.setRange(0.001, 100000.0)
        cur_spin.setDecimals(3)
        cur_spin.setValue(init_cur)
        cur_spin.setSuffix(" m")
        form.addRow(_t("Current Length:"), cur_spin)

        tgt_spin = QDoubleSpinBox()
        tgt_spin.setRange(0.001, 100000.0)
        tgt_spin.setDecimals(3)
        tgt_spin.setValue(round(init_cur, 1) if init_cur > 1.0 else 5.0)
        tgt_spin.setSuffix(" m")
        form.addRow(_t("Real Length:"), tgt_spin)
        lay.addLayout(form)

        # Interactive Scale Multiplier Section
        slider_box = QGroupBox(_t("Interactive Scale Multiplier"))
        slider_lay = QVBoxLayout(slider_box)
        slider_lay.setSpacing(6)

        initial_mult = tgt_spin.value() / max(cur_spin.value(), 1e-4)
        mult_lbl = QLabel(f"Scale Factor: {initial_mult:.3f}× (Camera Dist: {self.data.distance * initial_mult:.1f} m)")
        mult_lbl.setStyleSheet("font-weight: bold; color: #E0E0E0;")
        slider_lay.addWidget(mult_lbl)

        mult_slider = QSlider(Qt.Horizontal)
        mult_slider.setRange(10, 500)
        mult_slider.setValue(min(500, max(10, int(initial_mult * 100))))
        slider_lay.addWidget(mult_slider)
        lay.addWidget(slider_box)

        def on_spin_changed():
            c = cur_spin.value()
            t = tgt_spin.value()
            if c > 1e-6:
                ratio = t / c
                mult_lbl.setText(f"Scale Factor: {ratio:.3f}× (Camera Dist: {self.data.distance * ratio:.1f} m)")
                mult_slider.blockSignals(True)
                mult_slider.setValue(min(500, max(10, int(ratio * 100))))
                mult_slider.blockSignals(False)

        def on_slider_changed(val):
            ratio = val / 100.0
            tgt_spin.blockSignals(True)
            tgt_spin.setValue(cur_spin.value() * ratio)
            tgt_spin.blockSignals(False)
            mult_lbl.setText(f"Scale Factor: {ratio:.3f}× (Camera Dist: {self.data.distance * ratio:.1f} m)")

        cur_spin.valueChanged.connect(on_spin_changed)
        tgt_spin.valueChanged.connect(on_spin_changed)
        mult_slider.valueChanged.connect(on_slider_changed)

        btn_row = QHBoxLayout()
        apply_btn = QPushButton(_t("Apply Scale"))
        apply_btn.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 7px; border-radius: 3px;")
        cancel_btn = QPushButton(_t("Cancel"))
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)

        def on_apply():
            ok, _msg = self.scale_model_to_edge(cur_spin.value(), tgt_spin.value())
            if ok:
                if self.panel is not None:
                    self.panel.refresh_ui()
                dlg.accept()

        apply_btn.clicked.connect(on_apply)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec()

    def update_camera_from_guides(self) -> None:
        """Solves perspective from current guides and sets IngeTrazo's OrbitCamera."""
        vp = self.app.viewport
        w, h = max(vp.width(), 100), max(vp.height(), 100)
        self.solved = solve_perspective(self.data, w, h)
        if self.panel is not None:
            self.panel.update_readouts(self.solved)

        if not self.solved.valid:
            vp.update()
            return

        is_match_tab = (
            self.data.active_view_index < len(self.data.saved_views)
            and self.data.saved_views[self.data.active_view_index].get("type") == "match"
        )
        if not is_match_tab and not self.data.enabled:
            return

        cam = vp.camera
        cam.fov_deg = self.solved.fov_deg
        cam.distance = self.data.distance
        if self.data.mode == "3point":
            cam.up = self.solved.up_w
            cam.two_point = False
        else:
            cam.up = QVector3D(0.0, 0.0, 1.0)
            cam.two_point = True
        cam.look_from(self.solved.eye, self.solved.forward)
        if self.data.camera_locked:
            self.locked_camera_state = get_camera_state(cam)
        vp.update()

    def _draw_scene_tabs(self, viewport, painter: QPainter) -> None:
        """Draws SketchUp-style Scene Tabs Bar at the top of the viewport when enabled."""
        w, h = viewport.width(), viewport.height()
        self.scene_tab_rects = []
        self.add_view_rect = None
        self.scale_tool_rect = None

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        font = QFont("Segoe UI", 9)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()

        x = 12.0
        y = 10.0
        tab_h = 28.0

        # Draw Scene Tabs
        for idx, view in enumerate(self.data.saved_views):
            vname = view.get("name", f"Scene {idx}")
            vtype = view.get("type", "custom")
            icon = "📷 " if vtype == "match" else "🌐 "
            tab_label = f"{icon}{vname}"
            tab_w = fm.horizontalAdvance(tab_label) + 26.0
            rect = QRectF(x, y, tab_w, tab_h)
            self.scene_tab_rects.append((rect, idx))

            is_active = (idx == self.data.active_view_index)
            is_hover = (idx == getattr(self.filter, "hovered_tab_index", None))

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

        # Add Scene Button [+]
        add_btn_w = 28.0
        add_rect = QRectF(x, y, add_btn_w, tab_h)
        self.add_view_rect = add_rect

        is_add_hover = getattr(self.filter, "hovered_add_btn", False)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 122, 204, 210) if is_add_hover else QColor(30, 34, 44, 190)))
        painter.drawRoundedRect(add_rect, 4.0, 4.0)
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(add_rect, 4.0, 4.0)

        painter.setPen(QPen(Qt.white))
        painter.drawText(add_rect, Qt.AlignCenter, "➕")

        x += add_btn_w + 6.0

        # Calibrate Real Scale Button [📏 Scale]
        scale_label = f"📏 {_t('Scale')}"
        scale_btn_w = fm.horizontalAdvance(scale_label) + 18.0
        scale_rect = QRectF(x, y, scale_btn_w, tab_h)
        self.scale_tool_rect = scale_rect

        is_scale_hover = getattr(self.filter, "hovered_scale_btn", False)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 122, 204, 210) if is_scale_hover else QColor(30, 34, 44, 190)))
        painter.drawRoundedRect(scale_rect, 4.0, 4.0)
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(scale_rect, 4.0, 4.0)

        painter.setPen(QPen(Qt.white))
        painter.drawText(scale_rect, Qt.AlignCenter, scale_label)

        painter.restore()

    def _draw_loupe(self, viewport, painter: QPainter, hx: float, hy: float, handle_name: str) -> None:
        """Draws clean circular magnifying zoom loupe around active handle without confusing text."""
        w, h = viewport.width(), viewport.height()
        radius = 70.0

        lx = hx + 55.0
        ly = hy - 90.0

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

        # Subtle leader line from handle to loupe edge
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

        clip = QPainterPath()
        clip.addEllipse(loupe_center, radius, radius)
        painter.setClipPath(clip)

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
        painter.drawLine(QPointF(lx - 18, ly), QPointF(lx + 18, ly))
        painter.drawLine(QPointF(lx, ly - 18), QPointF(lx, ly + 18))
        painter.setPen(QPen(QColor(255, 255, 255, 240), 1.0))
        painter.drawLine(QPointF(lx - 18, ly), QPointF(lx + 18, ly))
        painter.drawLine(QPointF(lx, ly - 18), QPointF(lx, ly + 18))

        # Center pinpoint
        painter.setBrush(QBrush(handle_color))
        painter.setPen(QPen(Qt.white, 1.0))
        painter.drawEllipse(loupe_center, 2.5, 2.5)

        painter.setClipping(False)

        # Clean outer border rings (No confusing dark badges attached)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, 180), 4.0))
        painter.drawEllipse(loupe_center, radius + 1.5, radius + 1.5)

        painter.setPen(QPen(handle_color, 2.5))
        painter.drawEllipse(loupe_center, radius, radius)

        painter.setPen(QPen(QColor(255, 255, 255, 200), 1.0))
        painter.drawEllipse(loupe_center, radius - 2.0, radius - 2.0)

        painter.restore()

    def draw_overlay(self, viewport, painter: QPainter) -> None:
        """Draws SketchUp-style Scene Tabs, Background Photo, Vanishing Guides, and Bottom Help Text."""
        try:
            self._safe_draw_overlay(viewport, painter)
        except Exception:
            pass

    def _safe_draw_overlay(self, viewport, painter: QPainter) -> None:
        if not self.data.enabled:
            return

        # Enforce camera immutability when locked
        if self.data.camera_locked and self.locked_camera_state is not None:
            cam = getattr(viewport, "camera", None)
            if cam is not None:
                cur = get_camera_state(cam)
                if (cur.get("yaw") != self.locked_camera_state.get("yaw") or
                    cur.get("pitch") != self.locked_camera_state.get("pitch") or
                    cur.get("distance") != self.locked_camera_state.get("distance") or
                    cur.get("target") != self.locked_camera_state.get("target")):
                    apply_camera_state(cam, self.locked_camera_state)

        w = max(viewport.width(), 100)
        h = max(viewport.height(), 100)

        # 1. ALWAYS Draw Scene Tabs Bar at top when match mode is enabled
        try:
            self._draw_scene_tabs(viewport, painter)
        except Exception:
            pass

        # 2. Check active view tab
        is_match_tab = (
            self.data.active_view_index < len(self.data.saved_views)
            and self.data.saved_views[self.data.active_view_index].get("type") == "match"
        )

        if not is_match_tab:
            # When viewing standard Perspective (3D orbit) or custom scenes:
            # Leave viewport completely clean for normal 3D orbiting/zooming!
            return

        # 3. Draw Background Photograph (in Match view only)
        if self.data.show_image and self.pixmap is not None and not self.pixmap.isNull():
            try:
                painter.save()
                painter.setOpacity(self.data.image_opacity)
                scaled = self.pixmap.scaled(
                    w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                dx = (w - scaled.width()) / 2
                dy = (h - scaled.height()) / 2
                painter.drawPixmap(int(dx), int(dy), scaled)
                painter.restore()
            except Exception:
                pass

        # 4. Draw Reference Guides
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
            handle_a1: str, handle_b1: str,
            handle_a2: str, handle_b2: str
        ) -> None:
            pen = QPen(color, 2.0, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(p1_a, p1_b)
            painter.drawLine(p2_a, p2_b)

            if vp_pt is not None:
                vx, vy = vp_pt
                if not (math.isnan(vx) or math.isnan(vy) or math.isinf(vx) or math.isinf(vy)):
                    if abs(vx) < 50000 and abs(vy) < 50000:
                        vp_q = QPointF(vx, vy)
                        dash_pen = QPen(QColor(color.red(), color.green(), color.blue(), 140), 1.0, Qt.DashLine)
                        painter.setPen(dash_pen)
                        painter.drawLine(p1_b, vp_q)
                        painter.drawLine(p2_a, vp_q)
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
            "x1_a", "x1_b", "x2_a", "x2_b"
        )

        # Draw Y lines (Green)
        py1_a, py1_b = to_px(self.data.y1_a), to_px(self.data.y1_b)
        py2_a, py2_b = to_px(self.data.y2_a), to_px(self.data.y2_b)
        draw_line_pair(
            py1_a, py1_b, py2_a, py2_b, COLOR_Y, self.solved.vp_y,
            "y1_a", "y1_b", "y2_a", "y2_b"
        )

        # Draw Z lines (Blue) if 3-point mode
        if self.data.mode == "3point":
            pz1_a, pz1_b = to_px(self.data.z1_a), to_px(self.data.z1_b)
            pz2_a, pz2_b = to_px(self.data.z2_a), to_px(self.data.z2_b)
            draw_line_pair(
                pz1_a, pz1_b, pz2_a, pz2_b, COLOR_Z, self.solved.vp_z,
                "z1_a", "z1_b", "z2_a", "z2_b"
            )

        # Draw Horizon Line between Vx and Vy
        if self.solved.horizon is not None:
            (vx1, vy1), (vx2, vy2) = self.solved.horizon
            if not any(math.isnan(c) or math.isinf(c) for c in (vx1, vy1, vx2, vy2)):
                if all(abs(c) < 50000 for c in (vx1, vy1, vx2, vy2)):
                    pen_h = QPen(COLOR_HORIZON, 1.5, Qt.DashDotLine)
                    painter.setPen(pen_h)
                    painter.drawLine(QPointF(vx1, vy1), QPointF(vx2, vy2))

        # Draw Origin Handle
        p_orig = to_px(self.data.origin)
        draw_handle(p_orig, COLOR_ORIGIN, "origin")

        # True 3D Origin Perspective Axes Gizmo
        arm = 42.0

        # World X (Red)
        dx_x = self.solved.v_x_cam[0]
        dy_x = -self.solved.v_x_cam[1]
        l_x = math.hypot(dx_x, dy_x)
        if l_x > 1e-4:
            painter.setPen(QPen(COLOR_X, 2.8))
            painter.drawLine(p_orig, QPointF(p_orig.x() + arm * dx_x / l_x,
                                            p_orig.y() + arm * dy_x / l_x))

        # World Y (Green)
        dx_y = self.solved.v_y_cam[0]
        dy_y = -self.solved.v_y_cam[1]
        l_y = math.hypot(dx_y, dy_y)
        if l_y > 1e-4:
            painter.setPen(QPen(COLOR_Y, 2.8))
            painter.drawLine(p_orig, QPointF(p_orig.x() + arm * dx_y / l_y,
                                            p_orig.y() + arm * dy_y / l_y))

        # World Z (Blue)
        dx_z = self.solved.v_z_cam[0]
        dy_z = -self.solved.v_z_cam[1]
        l_z = math.hypot(dx_z, dy_z)
        if l_z > 1e-4:
            painter.setPen(QPen(COLOR_Z, 2.8))
            painter.drawLine(p_orig, QPointF(p_orig.x() + arm * dx_z / l_z,
                                            p_orig.y() + arm * dy_z / l_z))

        # Bottom SketchUp-Style Prompt and Guidance Bar
        guide_rect = QRectF(12, h - 36, w - 24, 26)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(18, 20, 26, 210)))
        painter.drawRoundedRect(guide_rect, 4.0, 4.0)

        font = QFont("Segoe UI", 9)
        font.setBold(True)
        painter.setFont(font)

        if getattr(self.filter, "is_shift_held", False):
            painter.setPen(QPen(QColor(76, 175, 80)))
            guide_msg = "⚡ PRECISION MODE ACTIVE (0.2×): Drag handles slowly to calibrate • Release Shift for normal speed"
        else:
            painter.setPen(QPen(QColor(230, 230, 230)))
            guide_msg = "💡 Drag red/green/blue handles to match photo • Hold Shift for precision zoom • Click [Default 3D] to orbit 3D"

        painter.drawText(guide_rect.adjusted(10, 0, -10, 0), Qt.AlignVCenter | Qt.AlignLeft, guide_msg)

        # Right-aligned camera info readout in bottom bar
        status_txt = f"🎯 f: {self.solved.focal_35mm:.1f}mm | FOV: {self.solved.fov_deg:.1f}°"
        painter.setPen(QPen(QColor(180, 190, 205)))
        painter.drawText(guide_rect.adjusted(0, 0, -12, 0), Qt.AlignVCenter | Qt.AlignRight, status_txt)

        painter.restore()

        # 5. Draw clean optical loupe during dragging (No dark text clutter)
        if getattr(self.filter, "active_handle", None) is not None:
            rel = getattr(self.data, self.filter.active_handle, None)
            if rel is not None:
                hx = rel[0] * w
                hy = rel[1] * h
                try:
                    self._draw_loupe(viewport, painter, hx, hy, self.filter.active_handle)
                except Exception:
                    pass


# ---- Extension Setup Entry Point -------------------------------------------
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

    app.on_document_changed(lambda: plugin.on_document_changed())
