"""Qt radar renderer used by the .bp replay window."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PyQt5.QtWidgets import QWidget


CANVAS_SIZE = 1200


class RadarWidget(QWidget):
    cursorChanged = pyqtSignal(float, float)
    measurementChanged = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(560, 560)
        self.setMouseTracking(True)
        self._canvas = QImage(CANVAS_SIZE, CANVAS_SIZE, QImage.Format_ARGB32_Premultiplied)
        self._canvas.fill(Qt.transparent)
        self._range_m = 2.0
        self._heading_deg = 0.0
        self._current_angle = 0.0
        self._gain = 1.7
        self._north_up = False
        self._show_profile = True
        self._profile: List[float] = []
        self._palette = "green"
        self._measurement_enabled = False
        # Endpoints are fractions of the actual echo radius, so resizing does
        # not change the measured distance or the marked echo positions.
        self._measurement_points: List[Tuple[float, float]] = []
        self._measurement_preview: Optional[Tuple[float, float]] = None

    def clear(self) -> None:
        self._canvas.fill(Qt.transparent)
        self._profile = []
        self.clear_measurement()
        self.update()

    def set_measurement_enabled(self, enabled: bool) -> None:
        self._measurement_enabled = bool(enabled)
        self._measurement_preview = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def clear_measurement(self) -> None:
        self._measurement_points.clear()
        self._measurement_preview = None
        self.measurementChanged.emit()
        self.update()

    @property
    def measured_distance_m(self) -> Optional[float]:
        if len(self._measurement_points) != 2:
            return None
        (ax, ay), (bx, by) = self._measurement_points
        return math.hypot(bx-ax, by-ay) * self._range_m

    def _echo_radius(self) -> float:
        # add_scan paints to a canvas radius of 0.485 * CANVAS_SIZE, and
        # drawImage maps the full canvas into the circular content rectangle.
        return self._content_rect().width() * 0.485

    def _measurement_position(self, point) -> Optional[Tuple[float, float]]:
        center = self._content_rect().center()
        radius = self._echo_radius()
        x = (point.x()-center.x())/radius
        y = (point.y()-center.y())/radius
        return (x, y) if x*x+y*y <= 1.0 else None

    def _screen_measurement_point(self, point: Tuple[float, float]) -> QPointF:
        center = self._content_rect().center()
        radius = self._echo_radius()
        return QPointF(center.x()+point[0]*radius, center.y()+point[1]*radius)

    def set_gain(self, gain: float) -> None:
        self._gain = max(0.1, gain)

    def set_palette(self, name: str) -> None:
        if name != self._palette:
            self._palette = name
            self.clear()

    def set_north_up(self, enabled: bool) -> None:
        self._north_up = enabled
        self.update()

    def set_show_profile(self, enabled: bool) -> None:
        self._show_profile = enabled
        self.update()

    def set_profile(self, distances: Sequence[float]) -> None:
        self._profile = list(distances)
        self.update()

    def _color(self, value: int) -> QColor:
        normalized = min(1.0, max(0.0, value / 255.0) * self._gain)
        normalized = normalized ** 0.72
        alpha = int(30 + 225 * normalized)
        if self._palette == "amber":
            return QColor(int(255 * normalized), int(180 * normalized), 20, alpha)
        if self._palette == "thermal":
            r = min(255, int(510 * normalized))
            g = min(255, int(510 * max(0.0, normalized - 0.35)))
            b = min(255, int(650 * max(0.0, normalized - 0.75)))
            return QColor(r, g, b, alpha)
        return QColor(30, int(255 * normalized), int(120 + 135 * normalized), alpha)

    def add_scan(
        self,
        angle_deg: float,
        range_m: float,
        intensities: bytes,
        heading_deg: float = 0.0,
    ) -> None:
        if not intensities or range_m <= 0:
            return
        if abs(range_m - self._range_m) > 1e-6:
            self._range_m = range_m
            self._canvas.fill(Qt.transparent)
            self._profile = []
            self.clear_measurement()

        self._heading_deg = heading_deg % 360.0
        self._current_angle = angle_deg % 360.0
        draw_angle = self._current_angle + (self._heading_deg if self._north_up else 0.0)
        radians = math.radians(draw_angle - 90.0)
        cx = cy = CANVAS_SIZE / 2.0
        radius = CANVAS_SIZE * 0.485
        cos_a, sin_a = math.cos(radians), math.sin(radians)
        count = len(intensities)

        painter = QPainter(self._canvas)
        painter.setRenderHint(QPainter.Antialiasing, False)
        for index, value in enumerate(intensities):
            if value <= 1:
                continue
            r = radius * (index + 0.5) / count
            painter.setPen(QPen(self._color(value), 1.5))
            painter.drawPoint(QPointF(cx + cos_a * r, cy + sin_a * r))
        painter.end()
        self.update()

    def _content_rect(self) -> QRectF:
        side = max(1.0, min(self.width(), self.height()) - 76.0)
        return QRectF((self.width() - side) / 2.0, (self.height() - side) / 2.0, side, side)

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#071118"))

        area = self._content_rect()
        center = area.center()
        radius = area.width() / 2.0
        clip = QPainterPath()
        clip.addEllipse(area)
        painter.save()
        painter.setClipPath(clip)
        painter.drawImage(area, self._canvas)
        painter.restore()

        grid = QColor(54, 159, 167, 115)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(grid, 1, Qt.DashLine))
        for ring in range(1, 5):
            rr = radius * ring / 4.0
            painter.drawEllipse(center, rr, rr)
        painter.drawLine(QPointF(center.x() - radius, center.y()), QPointF(center.x() + radius, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - radius), QPointF(center.x(), center.y() + radius))

        # Profile distances use 0.45-degree points, matching the sonar head.
        if self._show_profile and len(self._profile) > 2 and self._range_m > 0:
            profile_path = QPainterPath()
            previous_distance: Optional[float] = None
            segment_open = False
            for index, distance in enumerate(self._profile):
                angle = index * 0.45 + (self._heading_deg if self._north_up else 0.0)
                radians = math.radians(angle - 90.0)
                valid = 0.0 < distance <= self._range_m * 1.02
                continuous = (
                    previous_distance is None
                    or abs(distance - previous_distance) <= self._range_m * 0.08
                )
                if not valid:
                    segment_open = False
                    previous_distance = None
                    continue
                rr = distance / self._range_m * self._echo_radius()
                point = QPointF(center.x() + math.cos(radians) * rr, center.y() + math.sin(radians) * rr)
                if not segment_open or not continuous:
                    profile_path.moveTo(point)
                    segment_open = True
                else:
                    profile_path.lineTo(point)
                previous_distance = distance
            painter.setPen(QPen(QColor(255, 210, 72, 225), 2.0))
            painter.drawPath(profile_path)

        # Current head line.
        head_angle = self._current_angle + (self._heading_deg if self._north_up else 0.0)
        radians = math.radians(head_angle - 90.0)
        tip = QPointF(center.x() + math.cos(radians) * radius, center.y() + math.sin(radians) * radius)
        painter.setPen(QPen(QColor(95, 255, 190, 190), 1.5))
        painter.drawLine(center, tip)

        self._draw_compass(painter, center, radius)
        self._draw_measurement(painter)
        painter.end()

    def _draw_measurement(self, painter: QPainter) -> None:
        if not self._measurement_points:
            return
        painter.save()
        painter.setPen(QPen(QColor('#fff3a6'), 2.5))
        painter.setBrush(QColor('#fff3a6'))
        start = self._screen_measurement_point(self._measurement_points[0])
        painter.drawEllipse(start, 4.0, 4.0)
        end_point = (self._measurement_points[1] if len(self._measurement_points) == 2
                     else self._measurement_preview)
        if end_point is not None:
            end = self._screen_measurement_point(end_point)
            if len(self._measurement_points) == 1:
                painter.setPen(QPen(QColor('#fff3a6'), 1.5, Qt.DashLine))
            painter.drawLine(start, end)
            if len(self._measurement_points) == 2:
                painter.setPen(QPen(QColor('#fff3a6'), 2.5))
                painter.drawEllipse(end, 4.0, 4.0)
                value = f'{self.measured_distance_m:.3f} m'
                midpoint = (start+end)/2
                font = QFont('Microsoft YaHei UI', 10)
                font.setBold(True)
                painter.setFont(font)
                bounds = painter.fontMetrics().boundingRect(value)
                bubble = QRectF(midpoint.x()+8, midpoint.y()-bounds.height()-10,
                                bounds.width()+14, bounds.height()+10)
                # Keep the readout visible beside points near the right edge.
                if bubble.right() > self.width()-4:
                    bubble.moveRight(self.width()-4)
                if bubble.top() < 4:
                    bubble.moveTop(4)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(20, 36, 44, 235))
                painter.drawRoundedRect(bubble, 4, 4)
                painter.setPen(QColor('#fff3a6'))
                painter.drawText(bubble, Qt.AlignCenter, value)
        painter.restore()

    def _draw_compass(self, painter: QPainter, center: QPointF, radius: float) -> None:
        painter.save()
        font = QFont("Microsoft YaHei UI", 9)
        painter.setFont(font)
        for degree in range(0, 360, 10):
            # In vehicle-up mode the compass ring rotates against the vehicle.
            screen_degree = degree if self._north_up else degree - self._heading_deg
            radians = math.radians(screen_degree - 90.0)
            major = degree % 90 == 0
            outer = radius + 1
            inner = radius - (12 if major else 6)
            p1 = QPointF(center.x() + math.cos(radians) * inner, center.y() + math.sin(radians) * inner)
            p2 = QPointF(center.x() + math.cos(radians) * outer, center.y() + math.sin(radians) * outer)
            painter.setPen(QPen(QColor("#f15b64") if degree == 0 else QColor("#b8d4d8"), 2 if major else 1))
            painter.drawLine(p1, p2)
            if major:
                label = {0: "N", 90: "E", 180: "S", 270: "W"}[degree]
                text_r = radius + 22
                anchor = QPointF(center.x() + math.cos(radians) * text_r, center.y() + math.sin(radians) * text_r)
                painter.drawText(QRectF(anchor.x() - 12, anchor.y() - 10, 24, 20), Qt.AlignCenter, label)

        # Fixed vehicle direction marker.
        marker = QPolygonF([QPointF(center.x(), center.y() - radius + 2),
                            QPointF(center.x() - 7, center.y() - radius - 10),
                            QPointF(center.x() + 7, center.y() - radius - 10)])
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#ffd34d"))
        painter.drawPolygon(marker)
        painter.restore()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        area = self._content_rect()
        center = area.center()
        dx = event.pos().x() - center.x()
        dy = event.pos().y() - center.y()
        distance_px = math.hypot(dx, dy)
        if distance_px <= self._echo_radius():
            distance = distance_px / self._echo_radius() * self._range_m
            angle = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
            if self._north_up:
                angle = (angle - self._heading_deg) % 360.0
            self.cursorChanged.emit(distance, angle)
        else:
            self.cursorChanged.emit(-1.0, -1.0)
        if self._measurement_enabled and len(self._measurement_points) == 1:
            preview = self._measurement_position(event.pos())
            if preview != self._measurement_preview:
                self._measurement_preview = preview
                self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if not self._measurement_enabled or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        point = self._measurement_position(event.pos())
        if point is None:
            return
        if len(self._measurement_points) == 2:
            self._measurement_points = [point]
        else:
            self._measurement_points.append(point)
        self._measurement_preview = None
        self.measurementChanged.emit()
        self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.cursorChanged.emit(-1.0, -1.0)
        self._measurement_preview = None
        self.update()
        super().leaveEvent(event)
