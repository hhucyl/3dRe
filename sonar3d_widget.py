"""Interactive software-rendered 3-D view for stacked sonar scans."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import QWidget

from bp_reader import PointCloud


Projected = Tuple[float, float, float]


class Sonar3DWidget(QWidget):
    """Perspective point-cloud view with mouse orbit, pan, and zoom."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(560, 420)
        self.setMouseTracking(True)
        self._cloud: Optional[PointCloud] = None
        self._yaw = -35.0
        self._pitch = 27.0
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._last_mouse = QPoint()
        self._current_height: Optional[float] = None
        self._point_size = 1.7
        self._show_contours = True
        self._model_optimized = False
        self._section_boundaries = ()
        self._surface = None
        self._show_surface = True
        self._surface_points = False
        self._confidence_colors = False
        self._mesh_cache = None
        self._mesh_cache_key = None
        self._bounds_cache = None
        self._preview = None

    def set_cloud(self, cloud: PointCloud, model_optimized: bool = False) -> None:
        self._cloud = cloud
        self._model_optimized = bool(model_optimized)
        self._bounds_cache = None
        self.reset_view()

    def set_surface(self, mesh) -> None:
        self._surface = mesh
        self._mesh_cache = None
        self._mesh_cache_key = None
        self._bounds_cache = None
        self._preview = None
        if mesh is not None:
            import numpy as np

            vertices, faces, confidence = mesh.vertices, mesh.faces, mesh.confidence
            # Vertex clustering creates a complete coarse preview (no random
            # triangle dropping). The full-resolution mesh remains exportable.
            size = mesh.voxel_size
            for _ in range(16):
                if len(faces) <= 40000:
                    break
                cells = np.floor(mesh.vertices / size).astype(np.int64)
                _, inverse = np.unique(cells, axis=0, return_inverse=True)
                counts = np.bincount(inverse)
                vertices = np.column_stack([np.bincount(inverse, weights=mesh.vertices[:, i]) / counts for i in range(3)])
                confidence = np.bincount(inverse, weights=mesh.confidence) / counts
                faces = inverse[mesh.faces]
                valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
                faces = faces[valid]
                _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
                faces = faces[unique]
                size *= 1.4
            triangles = vertices[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            self._preview = (vertices, faces, confidence[faces].mean(axis=1), normals)
        self.update()

    def set_show_surface(self, enabled: bool) -> None:
        self._show_surface = bool(enabled)
        self._bounds_cache = None
        self.update()

    def set_surface_points(self, enabled: bool) -> None:
        self._surface_points = bool(enabled)
        self.update()

    def set_confidence_colors(self, enabled: bool) -> None:
        self._confidence_colors = bool(enabled)
        self._mesh_cache_key = None
        self.update()

    def clear(self) -> None:
        self._cloud = None
        self._current_height = None
        self._model_optimized = False
        self.set_surface(None)
        self.update()

    def set_current_height(self, height: Optional[float]) -> None:
        self._current_height = height
        self.update()

    def set_structure_boundaries(self, boundaries) -> None:
        self._section_boundaries = tuple(boundaries)
        self.update()

    def set_point_size(self, size: float) -> None:
        self._point_size = max(0.5, min(5.0, float(size)))
        self.update()

    def set_show_contours(self, enabled: bool) -> None:
        self._show_contours = bool(enabled)
        self.update()

    def reset_view(self) -> None:
        self._yaw = -35.0
        self._pitch = 27.0
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def _bounds(self) -> Tuple[float, float, float, float, float, float]:
        assert self._cloud is not None
        if self._bounds_cache is not None:
            return self._bounds_cache
        if self._surface is not None and self._show_surface:
            lower = self._surface.vertices.min(axis=0)
            upper = self._surface.vertices.max(axis=0)
            self._bounds_cache = (float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1]), float(lower[2]), float(upper[2]))
            return self._bounds_cache
        finite_points = [(p[0], p[1], p[2]) for p in self._cloud.points]
        finite_points.extend(self._cloud.interpolated)
        for contour in self._cloud.contours:
            finite_points.extend((x, y, z) for x, y, z in contour if math.isfinite(x) and math.isfinite(y))
        if not finite_points:
            return (-1.0, 1.0, -1.0, 1.0, self._cloud.min_height, self._cloud.max_height)
        xs, ys, zs = zip(*finite_points)
        self._bounds_cache = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
        return self._bounds_cache

    def _projector(self):  # type: ignore[no-untyped-def]
        bounds = self._bounds()
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        center_z = (min_z + max_z) / 2.0
        span = max(max_x - min_x, max_y - min_y, max_z - min_z, 0.2)
        screen_scale = min(self.width(), self.height()) * 0.67 / span * self._zoom
        yaw = math.radians(self._yaw)
        pitch = math.radians(self._pitch)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
        origin_x = self.width() / 2.0 + self._pan.x()
        origin_y = self.height() / 2.0 + self._pan.y()

        def project(x: float, y: float, z: float) -> Projected:
            dx, dy, dz = x - center_x, y - center_y, z - center_z
            x1 = cos_yaw * dx - sin_yaw * dy
            y1 = sin_yaw * dx + cos_yaw * dy
            y2 = cos_pitch * y1 - sin_pitch * dz
            depth = sin_pitch * y1 + cos_pitch * dz
            perspective = max(0.55, min(1.55, 1.0 / (1.0 + depth / (span * 3.5))))
            return origin_x + x1 * screen_scale * perspective, origin_y - y2 * screen_scale * perspective, depth

        return project, bounds

    @staticmethod
    def _point_color(value: int, depth_ratio: float) -> QColor:
        if value < 0:
            return QColor(218, 126, 245, 190)
        strength = max(0.0, min(1.0, value / 255.0))
        depth_light = 0.72 + 0.28 * depth_ratio
        r = int((30 + 225 * max(0.0, strength - 0.62) / 0.38) * depth_light)
        g = int((105 + 150 * strength) * depth_light)
        b = int((155 + 100 * (1.0 - strength)) * depth_light)
        return QColor(min(255, r), min(255, g), min(255, b), int(60 + 190 * strength))

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#071118"))
        if self._cloud is None:
            painter.setPen(QColor("#8aa6ad"))
            painter.drawText(self.rect(), Qt.AlignCenter, "打开 BP 文件后生成三维叠加")
            return

        project, bounds = self._projector()
        self._draw_grid_and_axes(painter, project, bounds)
        showing_surface = self._surface is not None and self._show_surface
        if showing_surface:
            self._draw_surface(painter, project, bounds)

        projected_points: List[Tuple[float, float, float, int]] = []
        for x, y, z, value in (self._cloud.points if not showing_surface or self._surface_points else ()):
            sx, sy, depth = project(x, y, z)
            if -4 <= sx <= self.width() + 4 and -4 <= sy <= self.height() + 4:
                projected_points.append((depth, sx, sy, value))
        if not showing_surface or self._surface_points:
            for x, y, z in self._cloud.interpolated:
                sx, sy, depth = project(x, y, z)
                if -4 <= sx <= self.width()+4 and -4 <= sy <= self.height()+4:
                    projected_points.append((depth, sx, sy, -1))
        projected_points.sort(key=lambda item: item[0], reverse=True)
        if projected_points:
            min_depth = min(item[0] for item in projected_points)
            max_depth = max(item[0] for item in projected_points)
            depth_span = max(1e-6, max_depth - min_depth)
            for depth, sx, sy, value in projected_points:
                ratio = (depth - min_depth) / depth_span
                painter.setPen(QPen(self._point_color(value, ratio), self._point_size))
                painter.drawPoint(QPointF(sx, sy))

        # Profile contours give each water-height layer a readable outline.
        if self._show_contours and (not showing_surface or self._surface_points):
            painter.setPen(QPen(QColor(255, 205, 66, 185), 1.15))
            for contour in self._cloud.contours:
                path = QPainterPath()
                open_segment = False
                previous: Optional[Tuple[float, float, float]] = None
                for point in contour:
                    x, y, z = point
                    if not math.isfinite(x) or not math.isfinite(y):
                        open_segment = False
                        previous = None
                        continue
                    if previous and math.dist(point, previous) > max(0.15, self._horizontal_span(bounds) * 0.08):
                        open_segment = False
                    sx, sy, _ = project(x, y, z)
                    if open_segment:
                        path.lineTo(sx, sy)
                    else:
                        path.moveTo(sx, sy)
                        open_segment = True
                    previous = point
                painter.drawPath(path)

        self._draw_overlay(painter)
        painter.end()

    def _draw_surface(self, painter, project, bounds):
        import numpy as np

        key = (self.width(), self.height(), self.devicePixelRatioF(), self._yaw, self._pitch,
               self._zoom, self._pan.x(), self._pan.y(), bounds, self._confidence_colors)
        if self._mesh_cache_key != key:
            vertices, faces, confidence, normals = self._preview
            projected = np.array([project(*vertex) for vertex in vertices])
            depth = projected[faces, 2].mean(axis=1)
            triangles = projected[faces, :2]
            visible = (triangles[:, :, 0].max(axis=1) >= 0) & (triangles[:, :, 0].min(axis=1) <= self.width())
            visible &= (triangles[:, :, 1].max(axis=1) >= 0) & (triangles[:, :, 1].min(axis=1) <= self.height())
            order = np.flatnonzero(visible)
            order = order[np.argsort(-depth[order])]
            light = np.array([0.3, -0.5, 0.8])
            light /= np.linalg.norm(light)
            shade = 0.35 + 0.65 * np.abs(normals @ light)
            if self._confidence_colors:
                ratio = np.clip(confidence * 2.5, 0, 1)[:, None]
                base = np.array([225, 145, 65]) * (1 - ratio) + np.array([65, 210, 200]) * ratio
            else:
                base = np.tile(np.array([90, 191, 204]), (len(faces), 1))
            colors = np.clip(base * shade[:, None], 0, 255).astype(int)
            ratio = self.devicePixelRatioF()
            canvas = QPixmap(int(self.width() * ratio), int(self.height() * ratio))
            canvas.setDevicePixelRatio(ratio)
            canvas.fill(Qt.transparent)
            renderer = QPainter(canvas)
            # No antialiasing between adjoining triangles: avoid hairline gaps.
            renderer.setPen(Qt.NoPen)
            for index in order:
                renderer.setBrush(QColor(*map(int, colors[index])))
                renderer.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in triangles[index]]))
            renderer.end()
            self._mesh_cache = canvas
            self._mesh_cache_key = key
        painter.drawPixmap(0, 0, self._mesh_cache)

    @staticmethod
    def _horizontal_span(bounds: Tuple[float, float, float, float, float, float]) -> float:
        return max(bounds[1] - bounds[0], bounds[3] - bounds[2])

    def _draw_grid_and_axes(self, painter: QPainter, project, bounds) -> None:  # type: ignore[no-untyped-def]
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        horizontal = max(max_x - min_x, max_y - min_y, 0.2)
        cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        half = horizontal * 0.55
        grid_min_x, grid_max_x = cx - half, cx + half
        grid_min_y, grid_max_y = cy - half, cy + half

        painter.setPen(QPen(QColor(65, 130, 143, 80), 1))
        for step in range(6):
            fraction = step / 5.0
            x = grid_min_x + (grid_max_x - grid_min_x) * fraction
            y = grid_min_y + (grid_max_y - grid_min_y) * fraction
            self._line3d(painter, project, (x, grid_min_y, min_z), (x, grid_max_y, min_z))
            self._line3d(painter, project, (grid_min_x, y, min_z), (grid_max_x, y, min_z))

        # Five water-height reference rectangles communicate the Z scale.
        z_span = max_z - min_z
        if z_span > 1e-4:
            painter.setPen(QPen(QColor(70, 170, 180, 65), 1, Qt.DashLine))
            for step in range(1, 5):
                z = min_z + z_span * step / 5.0
                self._rectangle3d(painter, project, grid_min_x, grid_max_x, grid_min_y, grid_max_y, z)

        if self._current_height is not None and min_z - 0.01 <= self._current_height <= max_z + 0.01:
            painter.setPen(QPen(QColor(99, 220, 255, 210), 1.6))
            self._rectangle3d(
                painter, project, grid_min_x, grid_max_x, grid_min_y, grid_max_y, self._current_height
            )

        for height in self._section_boundaries:
            if min_z <= height <= max_z:
                painter.setPen(QPen(QColor("#f4ae60"), 1.4, Qt.DashLine))
                self._rectangle3d(painter, project, grid_min_x, grid_max_x, grid_min_y, grid_max_y, height)
                sx, sy, _ = project(grid_max_x, grid_max_y, height)
                painter.drawText(QPointF(sx + 4, sy - 4), f"结构分界 Z={height:g} m")

        origin = (grid_min_x, grid_min_y, min_z)
        axes = [
            ((grid_max_x, grid_min_y, min_z), QColor("#ef6f6c"), "X / m"),
            ((grid_min_x, grid_max_y, min_z), QColor("#67d391"), "Y / m"),
            ((grid_min_x, grid_min_y, max_z if max_z > min_z else min_z + horizontal * 0.3), QColor("#65b9ff"), "水位 Z / m"),
        ]
        for end, color, label in axes:
            painter.setPen(QPen(color, 2))
            self._line3d(painter, project, origin, end)
            sx, sy, _ = project(*end)
            painter.drawText(QPointF(sx + 5, sy - 4), label)

    @staticmethod
    def _line3d(painter: QPainter, project, start, end) -> None:  # type: ignore[no-untyped-def]
        x1, y1, _ = project(*start)
        x2, y2, _ = project(*end)
        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def _rectangle3d(self, painter: QPainter, project, min_x, max_x, min_y, max_y, z) -> None:  # type: ignore[no-untyped-def]
        corners = [(min_x, min_y, z), (max_x, min_y, z), (max_x, max_y, z), (min_x, max_y, z)]
        for index, corner in enumerate(corners):
            self._line3d(painter, project, corner, corners[(index + 1) % 4])

    def _draw_overlay(self, painter: QPainter) -> None:
        assert self._cloud is not None
        painter.setPen(QColor("#d8e8ea"))
        font = QFont(painter.font())
        font.setPointSize(10)
        painter.setFont(font)
        height_span = self._cloud.max_height - self._cloud.min_height
        height_label = "旋转后 Z" if self._model_optimized else "waterHeight"
        painter.drawText(
            QRectF(14, 12, self.width() - 28, 25),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"三维点 {len(self._cloud.points):,}  |  轮廓层 {len(self._cloud.contours):,}  |  "
            f"{height_label} {self._cloud.min_height:.3f}–{self._cloud.max_height:.3f} m "
            f"(跨度 {height_span:.3f} m)"
            + ("  |  切片 a/b/c 校正已应用" if self._model_optimized else ""),
        )
        if self._surface is not None and self._show_surface:
            mesh = self._surface
            painter.setPen(QColor("#79e5d4"))
            painter.drawText(
                QRectF(14, 40, self.width() - 28, 25), Qt.AlignLeft | Qt.AlignVCenter,
                f"连续曲面：{len(mesh.faces):,} 个三角面  |  网格 {mesh.voxel_size * 100:.1f} cm"
                + ("  |  置信度：橙色低 → 青色高" if self._confidence_colors else ""),
            )
        if self._cloud.interpolation_note:
            painter.setPen(QColor("#da7ef5"))
            painter.drawText(QRectF(14, 68 if self._surface is not None and self._show_surface else 40,
                                   self.width()-28, 25), Qt.AlignLeft | Qt.AlignVCenter,
                             f"紫色：空间插值辅助点，预览 {len(self._cloud.interpolated):,} 点（非实测）")
        painter.setPen(QColor("#78959c"))
        painter.drawText(
            QRectF(14, self.height() - 32, self.width() - 28, 20),
            Qt.AlignLeft | Qt.AlignVCenter,
            "左键拖动：旋转视角   右键拖动：平移视角   滚轮：缩放   双击：复位   蓝框仅表示探头水位",
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._last_mouse = event.pos()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        delta = event.pos() - self._last_mouse
        self._last_mouse = event.pos()
        if event.buttons() & Qt.LeftButton:
            self._yaw += delta.x() * 0.45
            self._pitch = max(-85.0, min(85.0, self._pitch + delta.y() * 0.35))
            self.update()
        elif event.buttons() & (Qt.RightButton | Qt.MiddleButton):
            self._pan += QPointF(delta.x(), delta.y())
            self.update()

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        factor = 1.12 if event.angleDelta().y() > 0 else 1.0 / 1.12
        self._zoom = max(0.2, min(8.0, self._zoom * factor))
        self.update()

    def mouseDoubleClickEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        self.reset_view()
