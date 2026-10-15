"""Tests for the group box that folds from its whole header row."""

import pytest
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app.ui import theme
from scantailor.app.ui.theme import stylesheet
from scantailor.app.ui.theme.tokens import LIGHT
from scantailor.app.ui.widgets.collapsible_group_box import CollapsibleGroupBox


@pytest.fixture
def app(qapp: QtWidgets.QApplication, monkeypatch):
    """Application in the light look, restored afterwards."""
    monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
    style = qapp.style().name()
    palette = qapp.palette()
    font = qapp.font()
    theme.apply_theme(qapp, "light")
    yield qapp
    controller = qapp.findChild(QtCore.QObject, "scantailorThemeController")
    if controller is not None:
        shiboken6.delete(controller)
    qapp.setStyleSheet("")
    qapp.setStyle(style)
    qapp.setPalette(palette)
    qapp.setFont(font)


@pytest.fixture
def window(app, qtbot) -> QtWidgets.QWidget:
    """A top-level window kept alive for the length of the test."""
    widget = QtWidgets.QWidget()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def card(window, qtbot) -> tuple[CollapsibleGroupBox, QtWidgets.QLabel]:
    """A collapsible card with one label, shown at a fixed width."""
    outer = QtWidgets.QVBoxLayout(window)
    group = CollapsibleGroupBox("Binarization", window)
    group.setObjectName("cardGroup")
    body = QtWidgets.QLabel("Threshold", group)
    QtWidgets.QVBoxLayout(group).addWidget(body)
    outer.addWidget(group)
    outer.addStretch()
    window.resize(320, 300)
    window.show()
    qtbot.waitExposed(window)
    return group, body


def header_points(group: CollapsibleGroupBox) -> list[QtCore.QPoint]:
    header = group.header().geometry()
    y = header.center().y()
    return [
        QtCore.QPoint(header.left() + 2, y),
        QtCore.QPoint(header.center().x(), y),
        QtCore.QPoint(header.right() - 2, y),
    ]


def click(qtbot, group: CollapsibleGroupBox, point: QtCore.QPoint) -> None:
    target = group.childAt(point) or group
    qtbot.mouseClick(
        target, QtCore.Qt.MouseButton.LeftButton, pos=target.mapFrom(group, point)
    )


def test_a_click_anywhere_on_the_header_row_toggles_the_group(card, qtbot):
    group, _body = card
    for point in header_points(group):
        before = group.is_collapsed()
        click(qtbot, group, point)
        assert group.is_collapsed() is not before, point


def test_the_header_row_spans_the_card_and_is_at_least_a_24_px_target(card):
    group, _body = card
    header = group.header().geometry()
    assert header.left() == stylesheet.SPACE_XS
    assert header.right() == group.width() - 1 - stylesheet.SPACE_XS
    assert header.height() >= 24


def test_a_click_in_the_body_does_not_toggle_the_group(card, qtbot):
    group, body = card
    qtbot.mouseClick(body, QtCore.Qt.MouseButton.LeftButton)
    point = QtCore.QPoint(group.width() // 2, group.height() - 4)
    qtbot.mouseClick(group, QtCore.Qt.MouseButton.LeftButton, pos=point)
    assert not group.is_collapsed()


def test_the_chevron_sits_left_of_the_title(card):
    group, _body = card
    header = group.header()
    assert (
        header.toolButtonStyle() == QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    )
    assert header.layoutDirection() == QtCore.Qt.LayoutDirection.LeftToRight
    header.clearFocus()
    image = group.grab(header.geometry()).toImage()
    text = QtGui.QColor(LIGHT.text)

    def inked(x: int) -> bool:
        return any(
            image.pixelColor(x, y).lightness() < text.lightness() + 120
            for y in range(image.height())
        )

    first_ink = next(x for x in range(image.width()) if inked(x))
    chevron_end = stylesheet.SPACE_S + stylesheet.CHEVRON
    assert stylesheet.SPACE_S - 1 <= first_ink < chevron_end
    title_left = chevron_end + header.fontMetrics().horizontalAdvance(" ")
    assert any(inked(x) for x in range(title_left, title_left + 40))


def test_the_header_shows_a_pointing_hand(card):
    group, _body = card
    assert group.header().cursor().shape() == QtCore.Qt.CursorShape.PointingHandCursor
    assert group.cursor().shape() == QtCore.Qt.CursorShape.ArrowCursor


def test_space_on_the_focused_header_toggles_the_group(card, qtbot):
    group, _body = card
    header = group.header()
    assert header.focusPolicy() & QtCore.Qt.FocusPolicy.TabFocus
    header.setFocus(QtCore.Qt.FocusReason.TabFocusReason)
    qtbot.keyClick(header, QtCore.Qt.Key.Key_Space)
    assert group.is_collapsed()
    qtbot.keyClick(header, QtCore.Qt.Key.Key_Space)
    assert not group.is_collapsed()


def test_a_collapsed_group_shows_just_its_title(card, qtbot):
    group, body = card
    group.set_collapsed(True)
    qtbot.wait(10)
    assert not body.isVisible()
    assert group.header().isVisible()
    assert group.title() == "Binarization"
    expected = group.header().geometry().bottom() + 1 + stylesheet.SPACE_XS
    assert group.height() == expected
    group.set_collapsed(False)
    qtbot.wait(10)
    assert body.isVisible()
    assert group.height() > expected


def test_the_header_reports_its_state_to_assistive_technology(card):
    group, _body = card
    assert group.header().accessibleDescription() == "Expanded"
    group.set_collapsed(True)
    assert group.header().accessibleDescription() == "Collapsed"


def test_the_collapsed_state_is_remembered(card, qtbot):
    group, _body = card
    group.set_collapsed(True)
    again = CollapsibleGroupBox("Binarization", group.parentWidget())
    again.setObjectName("cardGroup")
    qtbot.addWidget(again)
    again.show()
    assert again.is_collapsed()


def test_title_is_the_header_text_and_the_frame_draws_none(card):
    group, _body = card
    group.setTitle("Despeckling")
    assert group.title() == "Despeckling"
    assert group.header().text() == "Despeckling"
    assert QtWidgets.QGroupBox.title(group) == ""


def test_raising_the_app_font_grows_the_header_row_and_title(card, app):
    group, body = card
    before = (
        group.header().height(),
        QtGui.QFontInfo(group.header().font()).pixelSize(),
        QtGui.QFontInfo(body.font()).pixelSize(),
    )
    font = app.font()
    font.setPointSizeF(font.pointSizeF() + 4)
    app.setFont(font)
    app.processEvents()
    after = (
        group.header().height(),
        QtGui.QFontInfo(group.header().font()).pixelSize(),
        QtGui.QFontInfo(body.font()).pixelSize(),
    )
    assert all(new > old for old, new in zip(before, after, strict=True))
