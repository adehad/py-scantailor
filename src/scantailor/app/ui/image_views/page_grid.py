"""A millimetre grid drawn over a page to judge alignment by eye.

The Margins and Output views draw it on request. The lines run through the
centre of the page so a grid line falls at the same distance from each edge,
which is what a reader compares between left and right pages.
"""

import math

from PySide6.QtCore import QLineF, QRectF
from PySide6.QtGui import QPainter, QPen, QTransform

from scantailor.app.ui.theme.tokens import view_color

GRID_SPACING_MM = 10.0
# Closer lines than this, in screen pixels, turn a zoomed-out page into a grey
# wash; every other line is then skipped until they are far enough apart.
MIN_SCREEN_SPACING = 8.0


def grid_offsets(half_extent: float, spacing: float) -> list[float]:
    """Return the line positions from the centre that fit inside the page.

    Args:
        half_extent (float): Half the page's width or height.
        spacing (float): Distance between lines, in the same unit.

    Returns:
        list[float]: Offsets from the centre, ascending, including 0.
    """
    if spacing <= 0 or half_extent < 0:
        return []
    count = math.floor(half_extent / spacing + 1e-9)
    return [step * spacing for step in range(-count, count + 1)]


def _spacing(mm_per_pixel: float, screen_per_pixel: float) -> float:
    """Return the line spacing in page pixels, widened until it reads on screen."""
    spacing = GRID_SPACING_MM / mm_per_pixel
    while spacing * screen_per_pixel < MIN_SCREEN_SPACING:
        spacing *= 2
    return spacing


def paint_page_grid(
    painter: QPainter,
    area: QRectF,
    mm_per_pixel: tuple[float, float],
    virtual_to_widget: QTransform,
) -> None:
    """Draw grid lines every ``GRID_SPACING_MM`` across a page.

    Args:
        painter (QPainter): Painter on the view's viewport.
        area (QRectF): The page in virtual coordinates; lines stop at its edges.
        mm_per_pixel (tuple[float, float]): Millimetres per virtual pixel along
            x and y; nothing is drawn while either is unknown (zero or less).
        virtual_to_widget (QTransform): Maps virtual to viewport coordinates.
    """
    mm_x, mm_y = mm_per_pixel
    if area.isEmpty() or mm_x <= 0 or mm_y <= 0:
        return
    screen_x = math.hypot(virtual_to_widget.m11(), virtual_to_widget.m12())
    screen_y = math.hypot(virtual_to_widget.m21(), virtual_to_widget.m22())
    if screen_x <= 0 or screen_y <= 0:
        return
    centre = area.center()
    lines = [
        QLineF(centre.x() + dx, area.top(), centre.x() + dx, area.bottom())
        for dx in grid_offsets(area.width() / 2, _spacing(mm_x, screen_x))
    ] + [
        QLineF(area.left(), centre.y() + dy, area.right(), centre.y() + dy)
        for dy in grid_offsets(area.height() / 2, _spacing(mm_y, screen_y))
    ]
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.setPen(QPen(view_color("page_grid"), 1))
    painter.drawLines([virtual_to_widget.map(line) for line in lines])
    painter.restore()
