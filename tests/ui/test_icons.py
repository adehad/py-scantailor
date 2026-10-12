"""Tests for the palette-aware Lucide and custom icon set."""

import pathlib
import re

import pytest
import shiboken6
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

from scantailor.app.ui import icons, theme
from scantailor.app.ui.theme.legacy_icons import IconTinter
from scantailor.app.ui.theme.tokens import DARK, LIGHT

SRC_ROOT = pathlib.Path(icons.__file__).resolve().parents[2]
RESOURCES = SRC_ROOT / "resources"
SIZE = QtCore.QSize(24, 24)
Mode = QtGui.QIcon.Mode
State = QtGui.QIcon.State


@pytest.fixture
def palette_app(qapp: QtWidgets.QApplication):
    """Application palette with distinct, known colours, restored afterwards."""
    original = qapp.palette()
    palette = QtGui.QPalette(original)
    role = QtGui.QPalette.ColorRole
    palette.setColor(role.ButtonText, QtGui.QColor("#102030"))
    palette.setColor(
        QtGui.QPalette.ColorGroup.Disabled, role.ButtonText, QtGui.QColor("#a0a0a0")
    )
    palette.setColor(role.Accent, QtGui.QColor("#ff6600"))
    palette.setColor(role.HighlightedText, QtGui.QColor("#00aa44"))
    qapp.setPalette(palette)
    yield qapp
    qapp.setPalette(original)


def ink_color(pixmap: QtGui.QPixmap) -> QtGui.QColor:
    """Return the colour of the most opaque pixel."""
    image = pixmap.toImage()
    best = QtGui.QColor()
    best_alpha = -1
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            if pixel.alpha() > best_alpha:
                best, best_alpha = pixel, pixel.alpha()
    return best


def test_icon_is_not_null_and_renders_opaque_pixels(palette_app):
    pixmap = icons.icon("zoom-in").pixmap(SIZE)
    assert not icons.icon("zoom-in").isNull()
    assert not pixmap.isNull()
    assert ink_color(pixmap).alpha() == 255


def test_normal_and_disabled_pixmaps_differ_in_colour(palette_app):
    glyph = icons.icon("zoom-in")
    normal = ink_color(glyph.pixmap(SIZE, Mode.Normal))
    disabled = ink_color(glyph.pixmap(SIZE, Mode.Disabled))
    assert normal.name() == "#102030"
    assert disabled.name() == "#a0a0a0"
    assert normal != disabled


def test_active_mode_uses_the_accent_and_selected_the_highlighted_text(palette_app):
    glyph = icons.icon("zoom-in")
    assert ink_color(glyph.pixmap(SIZE, Mode.Active)).name() == "#ff6600"
    assert ink_color(glyph.pixmap(SIZE, Mode.Selected)).name() == "#00aa44"


def test_toggle_pair_draws_a_different_glyph_per_state(palette_app):
    glyph = icons.icon("eye-off", "eye")
    off = glyph.pixmap(SIZE, Mode.Normal, State.Off).toImage()
    on = glyph.pixmap(SIZE, Mode.Normal, State.On).toImage()
    assert off != on
    assert icons.icon("eye-off").pixmap(SIZE, Mode.Normal, State.On).toImage() == off


def test_toggle_pair_follows_a_checked_button(palette_app, qtbot):
    button = QtWidgets.QToolButton()
    qtbot.addWidget(button)
    button.setCheckable(True)
    button.setIcon(icons.icon("eye-off", "eye"))
    unchecked = button.icon().pixmap(SIZE, Mode.Normal, State.Off).toImage()
    checked = button.icon().pixmap(SIZE, Mode.Normal, State.On).toImage()
    assert unchecked != checked


def test_palette_change_recolours_the_same_icon(palette_app):
    glyph = icons.icon("zoom-in")
    before = ink_color(glyph.pixmap(SIZE)).name()
    changed = QtGui.QPalette(palette_app.palette())
    changed.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#eeeeee"))
    palette_app.setPalette(changed)
    assert before == "#102030"
    assert ink_color(glyph.pixmap(SIZE)).name() == "#eeeeee"


def test_pixmap_is_rendered_at_the_requested_device_pixel_ratio(palette_app):
    pixmap = icons.icon("zoom-in").pixmap(QtCore.QSize(16, 16), 2.0)
    assert pixmap.devicePixelRatio() == 2.0
    assert pixmap.size() == QtCore.QSize(32, 32)
    assert ink_color(pixmap).alpha() == 255


def test_icon_paints_directly_into_a_painter(palette_app):
    image = QtGui.QImage(SIZE, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    icons.icon("minus").paint(painter, QtCore.QRect(QtCore.QPoint(0, 0), SIZE))
    painter.end()
    assert image.pixelColor(12, 12).alpha() > 0


def test_unknown_glyph_name_raises_instead_of_returning_a_blank_icon(palette_app):
    with pytest.raises(ValueError, match="no-such-glyph"):
        icons.icon("no-such-glyph")
    with pytest.raises(ValueError, match="missing-on"):
        icons.icon("eye", "missing-on")


def test_icons_are_recognised_by_the_legacy_tinter(palette_app, qtbot):
    assert icons.is_palette_icon(icons.icon("minus"))
    assert not icons.is_palette_icon(QtGui.QIcon())
    button = QtWidgets.QToolButton()
    qtbot.addWidget(button)
    button.setIcon(icons.icon("minus"))
    tinter = IconTinter()
    tinter.eventFilter(button, QtCore.QEvent(QtCore.QEvent.Type.Polish))
    assert icons.is_palette_icon(button.icon())


def test_binder_gives_a_button_its_icon_from_the_name_property(palette_app, qtbot):
    binder = icons.IconBinder()
    button = QtWidgets.QToolButton()
    qtbot.addWidget(button)
    button.setProperty(icons.ICON_NAME_PROPERTY, "eye-off")
    button.setProperty(icons.ICON_NAME_ON_PROPERTY, "eye")
    assert button.icon().isNull()

    binder.eventFilter(button, QtCore.QEvent(QtCore.QEvent.Type.Polish))

    assert icons.is_palette_icon(button.icon())
    off = button.icon().pixmap(SIZE, Mode.Normal, State.Off).toImage()
    on = button.icon().pixmap(SIZE, Mode.Normal, State.On).toImage()
    assert off != on


def test_binder_ignores_buttons_without_the_property(palette_app, qtbot):
    binder = icons.IconBinder()
    button = QtWidgets.QToolButton()
    qtbot.addWidget(button)
    binder.eventFilter(button, QtCore.QEvent(QtCore.QEvent.Type.Polish))
    assert button.icon().isNull()


@pytest.fixture
def themed_app(qapp: QtWidgets.QApplication):
    """Application with the theme (and so the icon binder) installed."""
    original = qapp.palette()
    style = qapp.style().name()
    font = qapp.font()
    theme.apply_theme(qapp, "light")
    yield qapp
    controller = qapp.findChild(QtCore.QObject, "scantailorThemeController")
    if controller is not None:
        shiboken6.delete(controller)
    qapp.setStyleSheet("")
    qapp.setStyle(style)
    qapp.setPalette(original)
    qapp.setFont(font)


def test_main_window_toolbar_buttons_get_palette_icons_from_the_form(
    themed_app, main_window
):
    toggles = {"focusButton", "thumbColumnViewBtn", "sortingOrderBtn"}
    names = {
        "focusButton",
        "prevPageBtn",
        "nextPageBtn",
        "filterSelectedBtn",
        "magnifyThumbnailsBtn",
        "diminishThumbnailsBtn",
        "gotoPageBtn",
        "selectionModeBtn",
        "thumbColumnViewBtn",
        "sortingOrderBtn",
        "deviationHighlightingBtn",
    }
    for name in sorted(names):
        button = main_window.ui.findChild(QtWidgets.QToolButton, name)
        assert button is not None, name
        assert icons.is_palette_icon(button.icon()), name
        off = button.icon().pixmap(SIZE, Mode.Normal, State.Off).toImage()
        on = button.icon().pixmap(SIZE, Mode.Normal, State.On).toImage()
        assert (off != on) == (name in toggles), name


def test_icon_label_redraws_when_the_look_switches_scheme(qapp, qtbot, monkeypatch):
    monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
    original = (qapp.palette(), qapp.style().name(), qapp.font())
    try:
        theme.apply_theme(qapp, "light")
        label = icons.IconLabel("arrow-up", 24)
        qtbot.addWidget(label)
        label.show()
        assert ink_color(label.pixmap()).name() == LIGHT.text
        theme.apply_theme(qapp, "dark")
        assert ink_color(label.pixmap()).name() == DARK.text
        assert label.size() == SIZE
    finally:
        controller = qapp.findChild(QtCore.QObject, "scantailorThemeController")
        if controller is not None:
            shiboken6.delete(controller)
        qapp.setStyleSheet("")
        qapp.setStyle(original[1])
        qapp.setPalette(original[0])
        qapp.setFont(original[2])


def test_every_glyph_file_is_a_valid_svg_drawn_with_current_colour():
    files = sorted(
        [
            *(RESOURCES / "icons/lucide").glob("*.svg"),
            *(RESOURCES / "icons/custom").glob("*.svg"),
        ]
    )
    assert len(files) > 80
    for path in files:
        text = path.read_text("utf-8")
        assert "currentColor" in text, path.name
        assert QtSvg.QSvgRenderer(QtCore.QByteArray(text.encode())).isValid(), path.name


def test_custom_glyphs_share_the_lucide_grid_and_stroke():
    for path in sorted((RESOURCES / "icons/custom").glob("*.svg")):
        text = path.read_text("utf-8")
        assert 'viewBox="0 0 24 24"' in text, path.name
        assert 'stroke-width="2"' in text, path.name
        assert 'stroke-linecap="round"' in text, path.name
        assert 'stroke-linejoin="round"' in text, path.name


def test_glyph_files_and_licences_are_registered_as_resources():
    for resource in (
        ":/icons/lucide/chevron-down.svg",
        ":/icons/lucide/LICENSE",
        ":/icons/custom/align_top_left.svg",
        ":/fonts/inter/Inter-Regular.ttf",
        ":/fonts/inter/OFL.txt",
    ):
        assert QtCore.QFile.exists(resource), resource
    assert (RESOURCES / "fonts/noto-sans-kr/OFL.txt").is_file()


def test_every_glyph_named_in_the_sources_exists():
    call = re.compile(r"\bicon\(\s*\"([a-z0-9_-]+)\"(?:\s*,\s*\"([a-z0-9_-]+)\")?")
    prop = re.compile(
        r'<property name="iconName(?:On)?" stdset="0">\s*<string[^>]*>([^<]+)<'
    )
    names: set[str] = set()
    for path in SRC_ROOT.rglob("*.py"):
        if path.name == "resources_rc.py":
            continue
        source = path.read_text("utf-8")
        for match in call.finditer(source):
            names.update(group for group in match.groups() if group)
        names.update(re.findall(r'"(align_[a-z_]+)"', source))
    for path in SRC_ROOT.rglob("*.ui"):
        names.update(prop.findall(path.read_text("utf-8")))
    assert {"zoom-in", "play", "chevron-down", "align_top_left"} <= names
    for name in sorted(names):
        assert icons.svg_source(name), name
