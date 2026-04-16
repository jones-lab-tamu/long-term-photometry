"""
Shared interactive image helpers for GUI figure inspection.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea


class InteractiveImageLabel(QLabel):
    """
    QLabel with click, drag, and wheel-zoom signals for image inspection.
    """

    clicked = Signal()
    drag_started = Signal(QPoint)
    drag_moved = Signal(QPoint)
    drag_finished = Signal()
    wheel_zoom = Signal(int, QPoint)

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._drag_start = QPoint()
        self._left_down = False
        self._dragging = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._left_down = True
            self._dragging = False
            self._drag_start = event.position().toPoint()
            self.drag_started.emit(self._drag_start)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._left_down and (event.buttons() & Qt.LeftButton):
            pos = event.position().toPoint()
            if (pos - self._drag_start).manhattanLength() >= 4:
                self._dragging = True
            self.drag_moved.emit(pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            was_drag = self._dragging or (pos - self._drag_start).manhattanLength() >= 4
            if not was_drag:
                self.clicked.emit()
            self.drag_finished.emit()
            self._left_down = False
            self._dragging = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = int(event.angleDelta().y())
        if delta:
            steps = int(delta / 120)
            if steps == 0:
                steps = 1 if delta > 0 else -1
            self.wheel_zoom.emit(steps, event.position().toPoint())
            event.accept()
            return
        super().wheelEvent(event)


class InteractiveImageController:
    """
    Shared incremental zoom + pan behavior for a QLabel inside a QScrollArea.
    """

    def __init__(
        self,
        *,
        label: InteractiveImageLabel,
        scroll_area: QScrollArea,
        set_hint_text: Callable[[str], None],
        fit_hint: str,
        zoom_hint: str,
        allow_upscale_in_fit: bool,
        zoom_step_factor: float = 1.2,
        min_scale: float = 0.05,
        max_scale: float = 20.0,
        fit_margin_px: int = 8,
        on_zoom_mode_changed: Callable[[bool], None] | None = None,
    ):
        self._label = label
        self._scroll = scroll_area
        self._set_hint_text = set_hint_text
        self._fit_hint = fit_hint
        self._zoom_hint = zoom_hint
        self._allow_upscale_in_fit = bool(allow_upscale_in_fit)
        self._zoom_step_factor = float(zoom_step_factor)
        self._min_scale = float(min_scale)
        self._max_scale = float(max_scale)
        self._fit_margin_px = int(max(0, fit_margin_px))
        self._on_zoom_mode_changed = on_zoom_mode_changed

        self._pixmap = QPixmap()
        self._zoom_mode = False
        self._current_scale = 1.0
        self._panning = False
        self._pan_start = QPoint()
        self._pan_h_start = 0
        self._pan_v_start = 0

        self._label.drag_started.connect(self._on_drag_started)
        self._label.drag_moved.connect(self._on_drag_moved)
        self._label.drag_finished.connect(self._on_drag_finished)
        self._label.wheel_zoom.connect(self._on_wheel_zoom)
        self._set_hint_text(self._fit_hint)
        self._sync_cursor()

    @property
    def zoom_mode(self) -> bool:
        return bool(self._zoom_mode)

    def clear(self, message: str, fallback_width: int = 640, fallback_height: int = 320) -> None:
        self._pixmap = QPixmap()
        self._set_zoom_mode_internal(False)
        self._current_scale = 1.0
        self._panning = False
        self._label.setPixmap(QPixmap())
        self._label.setText(message)
        viewport = self._scroll.viewport().size()
        w = int(viewport.width())
        h = int(viewport.height())
        if w < 10 or h < 10:
            w, h = int(fallback_width), int(fallback_height)
        self._label.resize(max(10, w), max(10, h))
        self._scroll.horizontalScrollBar().setValue(0)
        self._scroll.verticalScrollBar().setValue(0)
        self._sync_cursor()

    def set_pixmap(self, pixmap: QPixmap, *, reset_zoom: bool = True) -> None:
        self._pixmap = QPixmap(pixmap)
        if reset_zoom:
            self._set_zoom_mode_internal(False)
            self._current_scale = 1.0
        self.render()

    def render(self) -> None:
        if self._pixmap.isNull():
            return
        fit_scale = self._fit_scale()
        if self._zoom_mode:
            scale = max(self._min_scale, min(self._current_scale, self._max_scale))
            if self._is_fit_scale(scale, fit_scale):
                self._set_zoom_mode_internal(False)
                scale = fit_scale
        else:
            scale = fit_scale
        self._current_scale = scale

        target_w = max(1, int(round(self._pixmap.width() * scale)))
        target_h = max(1, int(round(self._pixmap.height() * scale)))
        scaled = self._pixmap.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._label.setText("")
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())
        self._sync_cursor()

    def toggle_zoom(self) -> None:
        if self._pixmap.isNull():
            return
        if self._zoom_mode:
            self._set_zoom_mode_internal(False)
            self.render()
            return
        self._set_zoom_mode_internal(True)
        # Toggle target: 100% native scale. If this equals fit-scale, apply a
        # single incremental step so toggle is still meaningful.
        fit_scale = self._fit_scale()
        target = 1.0
        if self._is_fit_scale(target, fit_scale):
            target = fit_scale * self._zoom_step_factor
        self._current_scale = max(self._min_scale, min(target, self._max_scale))
        self.render()

    def set_zoom_mode(self, enabled: bool) -> None:
        if self._pixmap.isNull():
            self._set_zoom_mode_internal(False)
            return
        want_zoom = bool(enabled)
        if not want_zoom:
            self._set_zoom_mode_internal(False)
            self.render()
            return
        self._set_zoom_mode_internal(True)
        fit_scale = self._fit_scale()
        target = 1.0
        if self._is_fit_scale(target, fit_scale):
            target = fit_scale * self._zoom_step_factor
        self._current_scale = max(self._min_scale, min(target, self._max_scale))
        self.render()

    def zoom_steps(self, step_count: int) -> None:
        if self._pixmap.isNull():
            return
        steps = int(step_count)
        if steps == 0:
            return
        fit_scale = self._fit_scale()
        current = self._current_scale if self._zoom_mode else fit_scale
        new_scale = current * (self._zoom_step_factor ** steps)
        new_scale = max(self._min_scale, min(new_scale, self._max_scale))

        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        viewport = self._scroll.viewport().size()
        old_center_x = float(hbar.value()) + (float(viewport.width()) / 2.0)
        old_center_y = float(vbar.value()) + (float(viewport.height()) / 2.0)
        ratio = new_scale / current if current > 1e-9 else 1.0

        if steps < 0 and new_scale <= fit_scale:
            self._set_zoom_mode_internal(False)
            self._current_scale = fit_scale
            self.render()
            hbar.setValue(0)
            vbar.setValue(0)
            return

        if self._is_fit_scale(new_scale, fit_scale):
            self._set_zoom_mode_internal(False)
            self._current_scale = fit_scale
            self.render()
            hbar.setValue(0)
            vbar.setValue(0)
            return

        self._set_zoom_mode_internal(True)
        self._current_scale = new_scale
        self.render()
        new_center_x = old_center_x * ratio
        new_center_y = old_center_y * ratio
        hbar.setValue(int(round(new_center_x - (float(viewport.width()) / 2.0))))
        vbar.setValue(int(round(new_center_y - (float(viewport.height()) / 2.0))))

    def _fit_scale(self) -> float:
        if self._pixmap.isNull():
            return 1.0
        viewport = self._scroll.viewport().size()
        vw = max(1, int(viewport.width()) - self._fit_margin_px)
        vh = max(1, int(viewport.height()) - self._fit_margin_px)
        pw = max(1, int(self._pixmap.width()))
        ph = max(1, int(self._pixmap.height()))
        scale = min(float(vw) / float(pw), float(vh) / float(ph))
        if not self._allow_upscale_in_fit:
            scale = min(1.0, scale)
        return max(self._min_scale, min(scale, self._max_scale))

    @staticmethod
    def _is_fit_scale(scale: float, fit_scale: float) -> bool:
        tol = max(0.01, abs(fit_scale) * 0.01)
        return abs(scale - fit_scale) <= tol

    def _set_zoom_mode_internal(self, enabled: bool) -> None:
        new_mode = bool(enabled)
        changed = new_mode != self._zoom_mode
        self._zoom_mode = new_mode
        self._set_hint_text(self._zoom_hint if self._zoom_mode else self._fit_hint)
        self._sync_cursor()
        if changed and self._on_zoom_mode_changed is not None:
            self._on_zoom_mode_changed(self._zoom_mode)

    def _sync_cursor(self) -> None:
        if self._panning:
            self._label.setCursor(Qt.ClosedHandCursor)
            return
        if self._zoom_mode:
            self._label.setCursor(Qt.OpenHandCursor)
            return
        self._label.setCursor(Qt.ArrowCursor)

    def _on_wheel_zoom(self, steps: int, _pos: QPoint) -> None:
        self.zoom_steps(steps)

    def _on_drag_started(self, pos: QPoint) -> None:
        if not self._zoom_mode:
            self._panning = False
            self._sync_cursor()
            return
        self._panning = True
        self._pan_start = QPoint(pos)
        self._pan_h_start = int(self._scroll.horizontalScrollBar().value())
        self._pan_v_start = int(self._scroll.verticalScrollBar().value())
        self._sync_cursor()

    def _on_drag_moved(self, pos: QPoint) -> None:
        if not self._panning:
            return
        delta = pos - self._pan_start
        self._scroll.horizontalScrollBar().setValue(self._pan_h_start - int(delta.x()))
        self._scroll.verticalScrollBar().setValue(self._pan_v_start - int(delta.y()))

    def _on_drag_finished(self) -> None:
        self._panning = False
        self._sync_cursor()
