"""The degrees label under the Fix Orientation rotation indicator stays in view."""

import pytest
from PySide6 import QtGui
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.fix_orientation import FixOrientationImageView
from scantailor.core.models import OrthogonalRotation

VIEW_SIZE = 320
SMALL_POINT_SIZE = 9
LARGE_POINT_SIZE = 28
# Wide enough that "270" outgrows the fixed-size indicator above it.
HUGE_POINT_SIZE = 48


def _make_view(qtbot: QtBot, point_size: int) -> FixOrientationImageView:
    view = FixOrientationImageView()
    qtbot.addWidget(view)
    view.resize(VIEW_SIZE, VIEW_SIZE)
    view.set_rotation(OrthogonalRotation(degrees=270))
    view.show()
    qtbot.waitExposed(view)
    # Showing polishes the view, and an application style sheet left by an
    # earlier test can then replace a font set before it.
    font = view.viewport().font()
    font.setPointSize(point_size)
    view.viewport().setFont(font)
    assert view.viewport().font().pointSize() == point_size
    return view


def _label_pixel_rows(view: FixOrientationImageView) -> list[int]:
    """Return the rows below the indicator where the label left any ink.

    The view shows no image, so everything off the indicator is background.
    """
    image = view.viewport().grab().toImage()
    background = image.pixelColor(2, 2)
    indicator, label = view.indicator_rects()
    left = max(0, int(label.left()) - 2)
    right = min(image.width(), int(label.right()) + 3)
    rows = []
    for y in range(int(indicator.bottom()) + 1, image.height()):
        if any(image.pixelColor(x, y) != background for x in range(left, right)):
            rows.append(y)
    return rows


@pytest.mark.parametrize("point_size", [SMALL_POINT_SIZE, LARGE_POINT_SIZE])
def test_the_degrees_label_ends_above_the_bottom_edge_at_any_font_size(
    qtbot: QtBot, point_size: int
):
    view = _make_view(qtbot, point_size)

    rows = _label_pixel_rows(view)

    assert rows, "the label drew nothing"
    assert max(rows) < view.viewport().height() - 1


@pytest.mark.parametrize("point_size", [SMALL_POINT_SIZE, LARGE_POINT_SIZE])
def test_the_label_box_holds_the_full_line_height_of_the_font(
    qtbot: QtBot, point_size: int
):
    view = _make_view(qtbot, point_size)

    _indicator, label = view.indicator_rects()

    metrics = QtGui.QFontMetricsF(view.viewport().font())
    assert label.height() >= metrics.ascent() + metrics.descent()
    assert label.bottom() <= view.viewport().height() - view.INDICATOR_MARGIN


def test_a_label_wider_than_the_indicator_stays_inside_the_right_margin(
    qtbot: QtBot,
):
    view = _make_view(qtbot, HUGE_POINT_SIZE)

    indicator, label = view.indicator_rects()

    assert label.width() > indicator.width()
    assert label.right() <= view.viewport().width() - view.INDICATOR_MARGIN


def test_a_larger_font_grows_the_label_and_lifts_the_indicator(qtbot: QtBot):
    small_indicator, small_label = _make_view(qtbot, SMALL_POINT_SIZE).indicator_rects()
    large_indicator, large_label = _make_view(qtbot, LARGE_POINT_SIZE).indicator_rects()

    assert large_label.height() > small_label.height()
    assert large_indicator.bottom() < small_indicator.bottom()
    assert large_indicator.bottom() <= large_label.top()
