"""Tests for the cross-platform light and dark look."""

import pathlib
import re

import pytest
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app.ui import theme
from scantailor.app.ui.theme import fonts
from scantailor.app.ui.theme.legacy_icons import tint_icon

SRC_ROOT = pathlib.Path(theme.__file__).resolve().parents[3]


@pytest.fixture
def app(qapp: QtWidgets.QApplication, monkeypatch):
    """Application restored to its original look; QSettings is isolated by conftest."""
    monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
    style = qapp.style().name()
    palette = qapp.palette()
    font = qapp.font()
    yield qapp
    controller = qapp.findChild(QtCore.QObject, "scantailorThemeController")
    if controller is not None:
        shiboken6.delete(controller)
    qapp.setStyleSheet("")
    qapp.setStyle(style)
    qapp.setPalette(palette)
    qapp.setFont(font)


def window_lightness(app: QtWidgets.QApplication) -> int:
    return app.palette().window().color().lightness()


class FakeSystem:
    """Stands in for the platform colour scheme, which offscreen cannot change."""

    def __init__(self, app: QtWidgets.QApplication, monkeypatch) -> None:
        self._app = app
        self.scheme = QtCore.Qt.ColorScheme.Light
        monkeypatch.setattr(theme, "system_color_scheme", lambda _app: self.scheme)

    def switch(self, scheme: QtCore.Qt.ColorScheme) -> None:
        """Change the scheme and notify as the platform would."""
        self.scheme = scheme
        self._app.styleHints().colorSchemeChanged.emit(scheme)


@pytest.fixture
def system(app, monkeypatch) -> FakeSystem:
    return FakeSystem(app, monkeypatch)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_forced_scheme_sets_stylesheet_and_returns_scheme(app, scheme):
    assert theme.apply_theme(app, scheme) == scheme
    assert app.styleSheet().strip()


def test_only_auto_light_and_dark_are_modes(app):
    assert theme.THEME_MODES == ("auto", "light", "dark")


def test_unknown_mode_falls_back_to_auto(app, system):
    system.scheme = QtCore.Qt.ColorScheme.Dark
    assert theme.apply_theme(app, "native") == "dark"


def test_auto_follows_the_operating_system_scheme(app, system):
    system.scheme = QtCore.Qt.ColorScheme.Dark
    assert theme.apply_theme(app) == "dark"
    dark = window_lightness(app)
    system.scheme = QtCore.Qt.ColorScheme.Light
    assert theme.apply_theme(app) == "light"
    assert window_lightness(app) > dark


def test_auto_updates_live_when_the_scheme_changes(app, system):
    theme.apply_theme(app)
    light = window_lightness(app)
    system.switch(QtCore.Qt.ColorScheme.Dark)
    dark = window_lightness(app)
    system.switch(QtCore.Qt.ColorScheme.Light)
    assert dark < light
    assert window_lightness(app) == light


def test_unknown_system_scheme_is_treated_as_light(app, system):
    system.scheme = QtCore.Qt.ColorScheme.Unknown
    assert theme.resolve_scheme(app, "auto") == "light"
    assert theme.resolve_scheme(app, "dark") == "dark"


def test_forced_mode_ignores_later_scheme_changes(app, system):
    theme.apply_theme(app, "light")
    light = window_lightness(app)
    system.switch(QtCore.Qt.ColorScheme.Dark)
    assert window_lightness(app) == light


def test_settings_force_a_scheme_against_the_system(app, system):
    system.scheme = QtCore.Qt.ColorScheme.Dark
    QtCore.QSettings().setValue(theme.SETTINGS_KEY, "light")
    assert theme.apply_theme(app) == "light"
    system.switch(QtCore.Qt.ColorScheme.Dark)
    assert app.palette().window().color().lightness() > 128


def test_environment_overrides_settings(app, monkeypatch):
    QtCore.QSettings().setValue(theme.SETTINGS_KEY, "light")
    monkeypatch.setenv(theme.THEME_ENV_VAR, "dark")
    assert theme.apply_theme(app) == "dark"


def test_setting_auto_follows_the_system(app, system):
    system.scheme = QtCore.Qt.ColorScheme.Dark
    QtCore.QSettings().setValue(theme.SETTINGS_KEY, "auto")
    assert theme.apply_theme(app) == "dark"


def test_unknown_environment_value_falls_back_to_auto(app, system, monkeypatch):
    system.scheme = QtCore.Qt.ColorScheme.Dark
    monkeypatch.setenv(theme.THEME_ENV_VAR, "bogus")
    assert theme.resolve_mode() == "auto"
    assert theme.apply_theme(app) == "dark"


def test_applying_twice_installs_one_controller(app):
    theme.apply_theme(app, "light")
    theme.apply_theme(app, "dark")
    controllers = app.findChildren(QtCore.QObject, "scantailorThemeController")
    assert len(controllers) == 1


def test_palette_accent_matches_the_scheme(app):
    theme.apply_theme(app, "light")
    light_accent = app.palette().color(QtGui.QPalette.ColorRole.Accent)
    theme.apply_theme(app, "dark")
    dark_accent = app.palette().color(QtGui.QPalette.ColorRole.Accent)
    assert light_accent.isValid()
    assert light_accent != dark_accent


def test_selection_highlight_is_translucent_so_delegates_do_not_paint_solid_blocks(
    app,
):
    theme.apply_theme(app, "light")
    highlight = app.palette().highlight().color()
    assert 0 < highlight.alpha() < 128


def test_stylesheet_glyph_files_exist_and_are_referenced(app):
    theme.apply_theme(app, "dark")
    sheet = app.styleSheet()
    for glyph in ("chevron_down", "chevron_up", "check"):
        assert f"{glyph}.svg" in sheet
    paths = re.findall(r"url\(([^)]+\.svg)\)", sheet)
    assert paths
    assert all(QtCore.QFile.exists(path) for path in paths)


def test_stylesheet_glyphs_are_painted_in_theme_colours_not_black(app):
    theme.apply_theme(app, "dark")
    path = re.findall(r"url\(([^)]*chevron_down\.svg)\)", app.styleSheet())[0]
    content = pathlib.Path(path).read_text("utf-8").lower()
    assert "currentcolor" not in content
    assert "#9aa3b5" in content


def test_stylesheet_has_no_lint_waiver_and_no_font_size_in_points(app):
    source = (pathlib.Path(theme.__file__).parent / "stylesheet.py").read_text("utf-8")
    assert "noqa" not in source
    theme.apply_theme(app, "light")
    assert "pt;" not in app.styleSheet()


def test_no_windows_backdrop_code_remains_anywhere_in_the_sources():
    pattern = re.compile(r"\bDWM|\bMica\b|dwmapi|backdrop|windll", re.IGNORECASE)
    offenders = [
        str(path.relative_to(SRC_ROOT))
        for path in SRC_ROOT.rglob("*.py")
        if path.name != "resources_rc.py" and pattern.search(path.read_text("utf-8"))
    ]
    assert offenders == []
    assert not (pathlib.Path(theme.__file__).parent / "backdrop.py").exists()


def test_thumbnail_current_frame_has_no_hard_coded_colour():
    source = (SRC_ROOT / "app/ui/thumbnails/widget.py").read_text("utf-8")
    assert "#e8a200" not in source.lower()
    assert "ColorRole.Accent" in source


def test_main_window_form_has_no_inline_tool_button_styles():
    form = (SRC_ROOT / "app/ui/MainWindow.ui").read_text("utf-8")
    assert "QToolButton:" not in form


def test_bundled_fonts_load_and_provide_both_families(app):
    families = fonts.load_fonts()
    assert "Inter" in families
    assert "Noto Sans KR" in families
    database_styles = QtGui.QFontDatabase.styles("Inter")
    assert {"Regular", "Medium", "SemiBold"} <= set(database_styles)


def test_app_font_is_inter_with_noto_for_hangul_and_pixel_sizes(app):
    theme.apply_theme(app, "light")
    font = app.font()
    assert font.families() == ["Inter", "Noto Sans KR"]
    assert QtGui.QFontInfo(font).family() == "Inter"
    assert font.pixelSize() == fonts.BASE_PX == 13
    assert (fonts.CAPTION_PX, fonts.HEADING_PX) == (11, 15)


def test_spin_boxes_use_tabular_figures(app, qtbot):
    theme.apply_theme(app, "light")
    spin = QtWidgets.QDoubleSpinBox()
    qtbot.addWidget(spin)
    spin.ensurePolished()
    assert spin.font().isFeatureSet(QtGui.QFont.Tag("tnum"))
    assert spin.font().featureValue(QtGui.QFont.Tag("tnum")) == 1


def test_line_edits_keep_proportional_figures(app, qtbot):
    theme.apply_theme(app, "light")
    edit = QtWidgets.QLineEdit()
    qtbot.addWidget(edit)
    edit.ensurePolished()
    assert not edit.font().isFeatureSet(QtGui.QFont.Tag("tnum"))


def test_tint_icon_recolours_opaque_pixels_and_keeps_transparency(app):
    source = QtGui.QPixmap(8, 8)
    source.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(source)
    painter.fillRect(0, 0, 4, 8, QtGui.QColor("black"))
    painter.end()

    tinted = (
        tint_icon(QtGui.QIcon(source), QtGui.QColor("#e8ecf4"), QtCore.QSize(8, 8))
        .pixmap(8, 8)
        .toImage()
    )

    assert tinted.pixelColor(1, 4).name() == "#e8ecf4"
    assert tinted.pixelColor(6, 4).alpha() == 0


def test_old_black_icons_follow_the_palette_when_the_scheme_changes(app, qtbot):
    theme.apply_theme(app, "light")
    source = QtGui.QPixmap(8, 8)
    source.fill(QtGui.QColor("black"))
    button = QtWidgets.QToolButton()
    qtbot.addWidget(button)
    button.setIconSize(QtCore.QSize(8, 8))
    button.setIcon(QtGui.QIcon(source))
    button.ensurePolished()
    light = button.icon().pixmap(8, 8).toImage().pixelColor(4, 4)

    theme.apply_theme(app, "dark")
    dark = button.icon().pixmap(8, 8).toImage().pixelColor(4, 4)

    assert dark.lightness() > light.lightness()
