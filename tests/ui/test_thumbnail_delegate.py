"""ThumbnailDelegate painting, sampled from the pixels it draws."""

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app.ui.theme.palette import build_palette
from scantailor.app.ui.theme.tokens import DARK, LIGHT, Tokens
from scantailor.app.ui.thumbnails.model import IsCurrentRole
from scantailor.app.ui.thumbnails.widget import (
    _FRAME_MARGIN,
    _PADDING,
    ThumbnailDelegate,
    _caption_height,
)

THUMBNAIL = QtCore.QSize(60, 80)
TOLERANCE = 3


PICTURE_COLOUR = QtGui.QColor("#808080")


def _paint_cell(
    tokens: Tokens, *, selected: bool, current: bool, rendered: bool = True
) -> tuple[QtGui.QImage, QtGui.QPalette, QtCore.QRect]:
    """Paint one cell of a one-row model on the window background.

    A cell that is not ``rendered`` has no picture yet, as while the thumbnail
    is still being produced.
    """
    palette = build_palette(tokens)
    model = QtGui.QStandardItemModel()
    item = QtGui.QStandardItem("Page 12")
    if rendered:
        picture = QtGui.QImage(60, 80, QtGui.QImage.Format.Format_RGB32)
        picture.fill(PICTURE_COLOUR)
        item.setData(picture, QtCore.Qt.ItemDataRole.DecorationRole)
    item.setData(current, IsCurrentRole)
    model.appendRow(item)

    delegate = ThumbnailDelegate(THUMBNAIL)
    cell = QtCore.QRect(
        QtCore.QPoint(0, 0),
        delegate.sizeHint(QtWidgets.QStyleOptionViewItem(), model.index(0, 0)),
    )
    image = QtGui.QImage(cell.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(palette.color(QtGui.QPalette.ColorRole.Window))

    option = QtWidgets.QStyleOptionViewItem()
    option.rect = cell  # type: ignore[attr-defined]
    option.palette = palette  # type: ignore[attr-defined]
    if selected:
        option.state |= QtWidgets.QStyle.StateFlag.State_Selected  # type: ignore[attr-defined]
    painter = QtGui.QPainter(image)
    delegate.paint(painter, option, model.index(0, 0))
    painter.end()
    return image, palette, cell


def _blend(top: QtGui.QColor, base: QtGui.QColor) -> QtGui.QColor:
    alpha = top.alphaF()
    return QtGui.QColor.fromRgbF(
        top.redF() * alpha + base.redF() * (1 - alpha),
        top.greenF() * alpha + base.greenF() * (1 - alpha),
        top.blueF() * alpha + base.blueF() * (1 - alpha),
    )


def _distance(a: QtGui.QColor, b: QtGui.QColor) -> int:
    return max(
        abs(a.red() - b.red()), abs(a.green() - b.green()), abs(a.blue() - b.blue())
    )


@pytest.fixture(params=[LIGHT, DARK], ids=["light", "dark"])
def tokens(request: pytest.FixtureRequest) -> Tokens:
    return request.param


def test_selected_cell_fills_the_inset_frame_with_the_translucent_highlight(
    qapp: QtWidgets.QApplication, tokens: Tokens
):
    image, palette, _cell = _paint_cell(tokens, selected=True, current=False)
    window = palette.color(QtGui.QPalette.ColorRole.Window)
    expected = _blend(palette.color(QtGui.QPalette.ColorRole.Highlight), window)
    # One pixel left of the picture is inside the frame margin and outside the image.
    sample = image.pixelColor(_PADDING - 1, _PADDING + THUMBNAIL.height() // 2)

    assert _distance(sample, expected) <= TOLERANCE
    assert _distance(sample, window) > TOLERANCE


def test_unselected_cell_leaves_the_background_untouched(
    qapp: QtWidgets.QApplication, tokens: Tokens
):
    image, palette, _cell = _paint_cell(tokens, selected=False, current=False)
    window = palette.color(QtGui.QPalette.ColorRole.Window)

    sample = image.pixelColor(_PADDING - 1, _PADDING + THUMBNAIL.height() // 2)

    assert _distance(sample, window) <= TOLERANCE


def test_rendered_cell_draws_its_picture_inside_the_frame(
    qapp: QtWidgets.QApplication, tokens: Tokens
):
    image, palette, _cell = _paint_cell(tokens, selected=False, current=False)
    window = palette.color(QtGui.QPalette.ColorRole.Window)
    centre = image.pixelColor(
        _PADDING + THUMBNAIL.width() // 2, _PADDING + THUMBNAIL.height() // 2
    )
    corner = image.pixelColor(_PADDING + 2, _PADDING + 2)
    outside = image.pixelColor(_PADDING - 3, _PADDING + THUMBNAIL.height() // 2)

    assert _distance(centre, PICTURE_COLOUR) <= TOLERANCE
    assert _distance(corner, PICTURE_COLOUR) <= TOLERANCE
    assert _distance(outside, window) <= TOLERANCE


def test_cell_without_a_picture_shows_an_empty_outline_and_no_picture(
    qapp: QtWidgets.QApplication, tokens: Tokens
):
    image, palette, _cell = _paint_cell(
        tokens, selected=False, current=False, rendered=False
    )
    window = palette.color(QtGui.QPalette.ColorRole.Window)
    middle = _PADDING + THUMBNAIL.height() // 2
    centre = image.pixelColor(_PADDING + THUMBNAIL.width() // 2, middle)
    outline = [image.pixelColor(x, middle) for x in (_PADDING - 1, _PADDING)]

    assert _distance(centre, window) <= TOLERANCE
    assert max(_distance(c, window) for c in outline) > TOLERANCE


def test_cell_without_a_picture_still_marks_the_current_page(
    qapp: QtWidgets.QApplication, tokens: Tokens
):
    image, palette, _cell = _paint_cell(
        tokens, selected=True, current=True, rendered=False
    )
    accent = palette.color(QtGui.QPalette.ColorRole.Accent)
    middle = _PADDING + THUMBNAIL.height() // 2

    on_frame = image.pixelColor(_PADDING - _FRAME_MARGIN - 1, middle)

    assert _distance(on_frame, accent) <= TOLERANCE


def test_current_page_gets_an_accent_frame_around_the_selection_fill(
    qapp: QtWidgets.QApplication, tokens: Tokens
):
    image, palette, _cell = _paint_cell(tokens, selected=True, current=True)
    accent = palette.color(QtGui.QPalette.ColorRole.Accent)
    middle = _PADDING + THUMBNAIL.height() // 2

    on_frame = image.pixelColor(_PADDING - _FRAME_MARGIN - 1, middle)
    inside_frame = image.pixelColor(_PADDING - 1, middle)

    assert _distance(on_frame, accent) <= TOLERANCE
    assert _distance(inside_frame, accent) > TOLERANCE


def test_caption_is_drawn_in_the_text_colour_below_the_frame(
    qapp: QtWidgets.QApplication, tokens: Tokens
):
    image, palette, _cell = _paint_cell(tokens, selected=True, current=False)
    window = palette.color(QtGui.QPalette.ColorRole.Window)
    text = palette.color(QtGui.QPalette.ColorRole.Text)
    top = _PADDING + THUMBNAIL.height() + _FRAME_MARGIN
    pixels = [
        image.pixelColor(x, y)
        for y in range(top, top + _caption_height())
        for x in range(_PADDING, _PADDING + THUMBNAIL.width())
    ]
    inked = [c for c in pixels if _distance(c, window) > TOLERANCE]

    assert inked, "the caption drew nothing on the plain background"
    # Small anti-aliased text may never cover a whole pixel, so ask for one pixel
    # nine tenths of the way from the background to the text colour.
    assert min(_distance(c, text) for c in inked) <= _distance(window, text) / 10
    assert all(
        _distance(c, window) <= TOLERANCE
        or _distance(c, text) < _distance(window, text)
        for c in inked
    )
