"""The picture mask of a mixed page, shown pulsing over the Picture Zones tab.

The overlay marks what the output keeps as a picture. Picture zones change it as
they will in the output, so the effect of a zone on the mask can be seen while
the zone is edited: layers that add to the mask paint over it and layers that
subtract from it cut it away, in the order the output generator applies them.
The overlay breathes between a faint and a strong tint to stay visible over
both pictures and text without hiding either.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPolygonF, QTransform

from scantailor.app.ui.theme.tokens import view_color
from scantailor.core.zones import PictureLayer

_FIRST_PHASE_DEG = 270  # The pulse starts at its faintest
_PHASE_STEP_DEG = 40
_STEP_INTERVAL_MS = 120
_PULSE_DEPTH = 0.35
_PULSE_FLOOR = 0.15
# Larger masks are drawn from a reduced copy, as a screen holds fewer pixels.
_MAX_SIDE = 2048

# The order the layers are applied in: subtract from the auto layer, add to the
# foreground, add to the background, add to the auto layer, subtract from all.
_PASSES: tuple[tuple[PictureLayer, bool], ...] = (
    (PictureLayer.ERASER1, True),
    (PictureLayer.FOREGROUND, False),
    (PictureLayer.BACKGROUND, False),
    (PictureLayer.PAINTER2, False),
    (PictureLayer.ERASER3, True),
)


@dataclass(frozen=True)
class MaskZone:
    """A picture zone as the mask sees it.

    Attributes:
        polygon: Outline in the pixels of the mask's image.
        layer: The layer the zone edits.
    """

    polygon: QPolygonF
    layer: PictureLayer


def opacity_at(phase_degrees: float) -> float:
    """Return how opaque the overlay is at a point of its pulse.

    Args:
        phase_degrees (float): Position in the pulse; 270 is the faintest.

    Returns:
        float: Opacity between 0.15 and 0.5.
    """
    pulse = 0.5 * (math.sin(math.radians(phase_degrees)) + 1.0)
    return _PULSE_DEPTH * pulse + _PULSE_FLOOR


class PictureMaskOverlay(QObject):
    """Draws the picture mask with its pulsing tint and edited by zones.

    The host repaints on ``changed`` and calls ``paint`` from its overlay. The
    pulse only runs while the overlay is ``running``.
    """

    def __init__(self, host: QObject, on_change: Callable[[], None]) -> None:
        """Create the overlay.

        Args:
            host (QObject): Owner of the timer.
            on_change (Callable[[], None]): Called when the overlay needs painting
                again.
        """
        super().__init__(host)
        self._on_change = on_change
        self._alpha: NDArray[np.uint8] | None = None
        self._color = _mask_color()
        self._base: QImage | None = None
        self._source_size = (0, 0)
        self._shown: QImage | None = None
        self._zones: Sequence[MaskZone] = ()
        self._phase = _FIRST_PHASE_DEG
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_STEP_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)
        self._running = False

    @property
    def phase(self) -> int:
        """Position in the pulse, in degrees."""
        return self._phase

    def has_mask(self) -> bool:
        """Return whether there is a picture mask to show."""
        return self._base is not None

    def set_mask(self, mask: NDArray[np.bool_] | None) -> None:
        """Give the overlay a mask, or take it away.

        Args:
            mask (NDArray[np.bool_] | None): True where the output keeps a
                picture, in the pixels of the image the overlay is drawn on.
        """
        if mask is None or not mask.any():
            self._alpha = None
            self._base = None
            self._shown = None
            self._timer.stop()
            self._on_change()
            return
        height, width = mask.shape
        self._source_size = (width, height)
        alpha = mask.astype(np.uint8) * 255
        scale = _MAX_SIDE / max(width, height)
        if scale < 1.0:
            alpha = np.asarray(
                cv2.resize(
                    alpha,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                ),
                dtype=np.uint8,
            )
        self._alpha = alpha
        self._color = _mask_color()
        self._base = _tinted(alpha, self._color)
        self._rebuild()
        self._restart()

    def set_zones(self, zones: Sequence[MaskZone], repaint: bool = True) -> None:
        """Change the zones whose layers edit the mask.

        Args:
            zones (Sequence[MaskZone]): Picture zones in image pixels.
            repaint (bool): Ask the host to paint again; False when the host is
                about to paint anyway.
        """
        self._zones = list(zones)
        if self._base is not None:
            self._rebuild()
            if repaint:
                self._on_change()

    def set_running(self, running: bool) -> None:
        """Start or stop the pulse; a stopped overlay is drawn at its faintest.

        Args:
            running (bool): Whether the overlay is on screen.
        """
        self._running = running
        self._restart()

    def _restart(self) -> None:
        if self._running and self._base is not None:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def _advance(self) -> None:
        self._phase = (self._phase + _PHASE_STEP_DEG) % 360
        self._on_change()
        self._restart()

    def _rebuild(self) -> None:
        """Compose the base mask with the zones' layers."""
        base = self._base
        assert base is not None
        shown = base.copy()
        width, height = self._source_size
        painter = QPainter(shown)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setTransform(
            QTransform.fromScale(shown.width() / width, shown.height() / height)
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._color)
        for layer, subtracts in _PASSES:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Clear
                if subtracts
                else QPainter.CompositionMode.CompositionMode_SourceOver
            )
            for zone in self._zones:
                if zone.layer is layer:
                    painter.drawPolygon(zone.polygon, Qt.FillRule.WindingFill)
        painter.end()
        self._shown = shown

    def paint(self, painter: QPainter, image_to_widget: QTransform) -> None:
        """Draw the mask over the image.

        Args:
            painter (QPainter): Painter on the host's viewport.
            image_to_widget (QTransform): Maps image pixels to viewport pixels.
        """
        color = _mask_color()
        if color != self._color and self._alpha is not None:
            self._color = color
            self._base = _tinted(self._alpha, color)
            self._rebuild()
        shown = self._shown
        if shown is None:
            return
        width, height = self._source_size
        painter.save()
        painter.setTransform(image_to_widget)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setOpacity(opacity_at(self._phase))
        painter.drawImage(QRectF(QPointF(0, 0), QPointF(width, height)), shown)
        painter.restore()


def _mask_color() -> QColor:
    color = view_color("zone_picture_border")
    color.setAlpha(255)
    return color


def _tinted(alpha: NDArray[np.uint8], color: QColor) -> QImage:
    """Return an image of a colour whose opacity is ``alpha``."""
    height, width = alpha.shape
    premultiplied = np.empty((height, width, 4), dtype=np.uint8)
    for channel, value in enumerate((color.red(), color.green(), color.blue(), 255)):
        premultiplied[..., channel] = (alpha.astype(np.uint16) * value // 255).astype(
            np.uint8
        )
    return QImage(
        premultiplied.data,
        width,
        height,
        width * 4,
        QImage.Format.Format_RGBA8888_Premultiplied,
    ).copy()
