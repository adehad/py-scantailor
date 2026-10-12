"""Slider that draws its own tick marks."""

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter, QPaintEvent, QPalette
from PySide6.QtWidgets import QSlider, QStyle, QStyleOptionSlider

_TICK_LENGTH = 4
_TICK_GAP = 3


class TickedSlider(QSlider):
    """Horizontal slider whose tick marks survive the application stylesheet.

    A stylesheet that styles the groove or the handle makes Qt skip the ticks,
    so they are painted here from the tick position and interval, in the
    palette's secondary text colour.
    """

    def _tick_room(self) -> int:
        if self.orientation() != Qt.Orientation.Horizontal:
            return 0
        position = self.tickPosition()
        if position == QSlider.TickPosition.NoTicks:
            return 0
        sides = 2 if position == QSlider.TickPosition.TicksBothSides else 1
        return sides * (_TICK_LENGTH + _TICK_GAP)

    def sizeHint(self) -> QSize:
        """Return the base hint grown by the room the ticks need.

        Returns:
            QSize: Preferred size.
        """
        hint = super().sizeHint()
        return QSize(hint.width(), hint.height() + self._tick_room())

    def minimumSizeHint(self) -> QSize:
        """Return the base minimum grown by the room the ticks need.

        Returns:
            QSize: Minimum size.
        """
        hint = super().minimumSizeHint()
        return QSize(hint.width(), hint.height() + self._tick_room())

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the slider, then its tick marks.

        Args:
            event (QPaintEvent): The paint event.
        """
        super().paintEvent(event)
        if not self._tick_room():
            return
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        style = self.style()
        groove = style.subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle = style.subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        span = max(groove.width() - handle.width(), 0)
        interval = self.tickInterval() or self.pageStep() or 1
        painter = QPainter(self)
        painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
        position = self.tickPosition()
        value = self.minimum()
        while value <= self.maximum():
            x = (
                groove.left()
                + handle.width() // 2
                + QStyle.sliderPositionFromValue(
                    self.minimum(),
                    self.maximum(),
                    value,
                    span,
                    self.invertedAppearance(),
                )
            )
            if position != QSlider.TickPosition.TicksAbove:
                top = handle.bottom() + _TICK_GAP
                painter.drawLine(x, top, x, top + _TICK_LENGTH - 1)
            if position in (
                QSlider.TickPosition.TicksAbove,
                QSlider.TickPosition.TicksBothSides,
            ):
                bottom = handle.top() - _TICK_GAP
                painter.drawLine(x, bottom - _TICK_LENGTH + 1, x, bottom)
            value += interval
        painter.end()
