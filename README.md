# IngeTrazo Perspective Matcher

**Perspective Matcher** is an extension for [IngeTrazo](https://ingelibre.github.io/ingetrazo/) that calibrates and aligns the 3D viewport camera to an architectural photograph, bringing the capabilities of **SketchUp's "Match New Photo"** and **Blender's "Perspective Plotter" / fSpy** directly into IngeTrazo.

---

## 🌟 Features

* **📷 Reference Photo Overlay**: Load any photo (PNG, JPG, WEBP) directly into the 3D viewport with an interactive opacity slider.
* **🎯 Interactive Vanishing Lines**:
  * **2 Red lines** for horizontal edges parallel to the **X-axis** $\rightarrow$ calculates Vanishing Point $V_x$.
  * **2 Green lines** for horizontal edges parallel to the **Y-axis** $\rightarrow$ calculates Vanishing Point $V_y$.
  * **2 Blue lines** for vertical edges parallel to the **Z-axis** $\rightarrow$ calculates Vanishing Point $V_z$ (in 3-Point mode).
* **📍 Draggable Origin Pin $(0, 0, 0)$**: Place world origin precisely at any key corner of the building or photo. Includes a 3D coordinate axis triad gizmo.
* **📐 Projective Geometry Solver**:
  * Calculates focal length (in pixels and 35mm equivalent).
  * Computes vertical Field of View (FOV).
  * Calculates camera orientation (Pitch, Yaw, Roll) and eye distance.
* **🏢 2-Point & 3-Point Modes**:
  * **2-Point Perspective**: Keeps verticals perfectly vertical (architectural standard).
  * **3-Point Perspective**: For photos with tilted or drone camera angles.
* **📏 Non-Destructive Scaling & Quantity Protection**:
  * **"Scale Model with View (Fixed on Photo)" is OFF by default** to safeguard user model dimensions and quantity takeoffs.
  * Adjusting the scale slider freely changes camera distance without altering model measurements.
  * When explicitly enabled, uniform scaling executes safely through IngeTrazo's transactional command history (`viewport.history.execute`) with automatic rollback protection.
  * **Remembers Original Scale**: Tracks baseline scale across sessions (saved directly into `.igz`), allowing you to restore the model at any time with the **"🔄 Reset to Original"** button.
* **💾 Document Integration**: Calibration, photo path, cumulative scale, and guide handles are saved directly inside your IngeTrazo document (`.igz`).

---

## 🚀 Installation

1. In IngeTrazo, go to **Extensions ▸ Open plugins folder**.
2. Copy `perspective_matcher.py` into that folder.
3. Restart IngeTrazo.
4. The **"Perspective Match"** tab will appear in the side panel tray, and under **Extensions ▸ Perspective Matcher…** (`Ctrl+Shift+M`).

---

## 🛠️ How to Use

1. Click **"Load Photograph…"** in the side panel to choose your reference image.
2. Adjust the **Opacity** slider to blend the photograph with your 3D model.
3. Drag the colored circle handles to align with prominent lines in the photo:
   * Red handles on horizontal edges pointing along the X axis.
   * Green handles on horizontal edges pointing along the Y axis.
4. Drag the orange **Origin Handle** to the corner of the building where you want $(0, 0, 0)$ located.
5. Calibrate dimensions using **"Read Edge"** / **"Apply Scale"**, or adjust **Match Scale** slider.
6. If needed, click **"🔄 Reset to Original"** at any time to return the model to its original dimensions.
7. Click **"🔒 Lock"** once satisfied to freeze the camera while modeling.

---

## 📄 License

GPL-3.0-or-later © 2026 Seghier Mohamed Abdellatif and IngeTrazo contributors.
