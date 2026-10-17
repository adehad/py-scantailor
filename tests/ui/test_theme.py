"""Tests for the cross-platform light and dark look."""

import pathlib
import re

import pytest
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app.ui import theme
from scantailor.app.ui.theme import fonts, stylesheet
from scantailor.app.ui.theme.tokens import DARK, LIGHT

SRC_ROOT = pathlib.Path(theme.__file__).resolve().parents[3]


@pytest.fixture
def app(qapp: QtWidgets.QApplication, monkeypatch):
    """Application restored to its original look; QSettings is isolated by conftest."""
    monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
    for widget in qapp.topLevelWidgets():
        # A menu or combo box popup of a window an earlier test left behind is a
        # window too, and its wrapper can be gone by the time the loop reaches it.
        if shiboken6.isValid(widget):
            widget.hide()
            widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
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


def test_app_font_is_inter_with_noto_for_hangul_sized_from_the_system_font(app):
    theme.apply_theme(app, "light")
    font = app.font()
    system = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    assert font.families() == ["Inter", "Noto Sans KR"]
    assert QtGui.QFontInfo(font).family() == "Inter"
    assert font.pointSizeF() == pytest.approx(system.pointSizeF() * fonts.BODY_RATIO)


def test_small_text_is_never_below_the_system_font_size(app):
    theme.apply_theme(app, "light")
    system = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    assert fonts.small(app.font()).pointSizeF() >= system.pointSizeF() - 1e-6


def test_stylesheet_sets_no_body_font_size_and_scales_the_rest_from_the_app_font(app):
    theme.apply_theme(app, "light")
    body = fonts.pixel_size(app.font())
    sizes = {int(px) for px in re.findall(r"font-size: (\d+)px", app.styleSheet())}
    assert round(body) not in sizes
    assert sizes == {
        round(body * fonts.SMALL_RATIO),
        round(body * fonts.PANEL_HEADING_RATIO),
        round(body * fonts.DISPLAY_RATIO),
    }


def raise_app_font(app: QtWidgets.QApplication, points: float) -> None:
    font = app.font()
    font.setPointSizeF(font.pointSizeF() + points)
    app.setFont(font)
    app.processEvents()


def test_raising_the_app_font_grows_headings_body_and_small_text(app, qtbot):
    theme.apply_theme(app, "light")
    window = QtWidgets.QWidget()
    qtbot.addWidget(window)
    layout = QtWidgets.QVBoxLayout(window)
    body = QtWidgets.QLabel("Body")
    hint = QtWidgets.QLabel("Hint")
    hint.setProperty("textRole", "small")
    spin = QtWidgets.QSpinBox()
    dock = QtWidgets.QDockWidget("Filters")
    for widget in (body, hint, spin, dock):
        layout.addWidget(widget)
    window.show()
    qtbot.waitExposed(window)

    def sizes() -> list[int]:
        return [QtGui.QFontInfo(w.font()).pixelSize() for w in (body, hint, spin, dock)]

    before = sizes()
    raise_app_font(app, 4)
    after = sizes()

    assert all(new > old for old, new in zip(before, after, strict=True)), (
        before,
        after,
    )
    assert after[3] > after[0] > after[1]


def test_spin_boxes_use_tabular_figures(app, qtbot):
    theme.apply_theme(app, "light")
    spin = QtWidgets.QDoubleSpinBox()
    qtbot.addWidget(spin)
    spin.ensurePolished()
    assert spin.font().isFeatureSet(QtGui.QFont.Tag("tnum"))
    assert spin.font().featureValue(QtGui.QFont.Tag("tnum")) == 1


def test_spin_box_text_starts_inside_the_field_padding(app, qtbot):
    theme.apply_theme(app, "light")
    spin = QtWidgets.QDoubleSpinBox()
    qtbot.addWidget(spin)
    spin.resize(130, stylesheet.CONTROL_HEIGHT)
    spin.show()
    qtbot.waitExposed(spin)
    editor = spin.lineEdit()
    text_left = editor.x() + editor.textMargins().left()
    assert text_left == stylesheet.FIELD_BORDER + stylesheet.FIELD_PADDING_X


def test_line_edits_keep_proportional_figures(app, qtbot):
    theme.apply_theme(app, "light")
    edit = QtWidgets.QLineEdit()
    qtbot.addWidget(edit)
    edit.ensurePolished()
    assert not edit.font().isFeatureSet(QtGui.QFont.Tag("tnum"))


def test_semantic_roles_are_green_and_red_for_each_scheme():
    assert (LIGHT.positive, DARK.positive) == ("#008a1a", "#5ccf72")
    for tokens in (LIGHT, DARK):
        positive, danger = QtGui.QColor(tokens.positive), QtGui.QColor(tokens.danger)
        assert positive.green() > positive.red() and positive.green() > positive.blue()
        assert danger.red() > danger.green() and danger.red() > danger.blue()


def test_dark_tab_bar_draws_no_base_line_outside_the_pane_border(app):
    theme.apply_theme(app, "dark")
    sheet = app.styleSheet()

    assert "qproperty-drawBase: 0" in sheet
    assert f"border: 1px solid {DARK.panel_border}; border-radius" in sheet


def test_dark_dialog_tab_bar_shows_no_white_line_beside_the_tabs(app, qtbot):
    from scantailor.app.ui.dialogs.settings import SettingsDialog
    from scantailor.core.settings import ApplicationSettings

    theme.apply_theme(app, "dark")
    dialog = SettingsDialog(ApplicationSettings())
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait_exposed(dialog)
    tabs = dialog.findChild(QtWidgets.QTabWidget)
    assert tabs is not None

    image = dialog.grab().toImage()
    bar = tabs.tabBar()
    top_left = bar.mapTo(dialog, QtCore.QPoint(0, 0))
    last_tab = bar.tabRect(tabs.count() - 1)
    for y in range(top_left.y() - 2, top_left.y() + 2):
        for x in range(top_left.x() + last_tab.right() + 10, image.width() - 20):
            assert image.pixelColor(x, y).lightness() < 150, (x, y)


@pytest.mark.parametrize("scheme", theme.SCHEMES)
def test_deskew_arc_and_handles_are_drawn_in_the_theme_accent(app, qtbot, scheme):
    from scantailor.app.ui.image_views.deskew import DeskewImageView

    theme.apply_theme(app, scheme)
    view = DeskewImageView()
    qtbot.addWidget(view)
    view.resize(500, 400)
    view.show()
    qtbot.wait_exposed(view)
    accent = QtGui.QColor((DARK if scheme == "dark" else LIGHT).accent)
    size = view.viewport().size()
    image = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtCore.Qt.GlobalColor.transparent)

    painter = QtGui.QPainter(image)
    view._paint_rotation_arcs(painter)
    painter.end()

    opaque = {
        QtGui.QColor(image.pixel(x, y)).rgb()
        for y in range(image.height())
        for x in range(image.width())
        if QtGui.qAlpha(image.pixel(x, y)) == 255
    }
    assert view.palette().color(QtGui.QPalette.ColorRole.Accent) == accent
    assert accent.rgb() in opaque


def test_default_layouts_take_margins_and_gaps_from_the_spacing_scale(app, qtbot):
    theme.apply_theme(app, "light")
    dialog = QtWidgets.QDialog()
    qtbot.addWidget(dialog)
    dialog_layout = QtWidgets.QVBoxLayout(dialog)
    panel = QtWidgets.QWidget()
    panel_layout = QtWidgets.QVBoxLayout(panel)
    group = QtWidgets.QGroupBox("Group")
    group_layout = QtWidgets.QVBoxLayout(group)
    dialog_layout.addWidget(panel)
    panel_layout.addWidget(group)

    def margins(layout: QtWidgets.QLayout) -> set[int]:
        m = layout.contentsMargins()
        return {m.left(), m.top(), m.right(), m.bottom()}

    assert margins(dialog_layout) == {stylesheet.SPACE_L}
    assert margins(panel_layout) == {stylesheet.SPACE_S}
    assert margins(group_layout) == {0}
    assert dialog_layout.spacing() == stylesheet.SPACE_S


def test_an_untitled_group_reserves_no_title_band(app, qtbot):
    theme.apply_theme(app, "light")
    titled = QtWidgets.QGroupBox("Title")
    untitled = QtWidgets.QGroupBox()
    untitled.setProperty("untitled", True)
    for group in (titled, untitled):
        qtbot.addWidget(group)
        QtWidgets.QVBoxLayout(group).addWidget(QtWidgets.QLabel("Body"))
        group.ensurePolished()
    assert untitled.contentsRect().top() == stylesheet.SPACE_M + 1
    assert titled.contentsRect().top() > untitled.contentsRect().top() + 24


@pytest.mark.parametrize("scheme", theme.SCHEMES)
def test_dock_splitters_always_show_a_grip(app, qtbot, scheme):
    theme.apply_theme(app, scheme)
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)
    window.setCentralWidget(QtWidgets.QWidget())
    dock = QtWidgets.QDockWidget("Filters")
    dock.setWidget(QtWidgets.QWidget())
    window.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    window.resize(600, 400)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(50)

    gutter = QtCore.QRect(
        dock.geometry().right() + 1, 0, stylesheet.SPACE_S, window.height()
    )
    image = window.grab(gutter).toImage()
    background = image.pixelColor(0, 0).lightness()
    grip = [
        (x, y)
        for y in range(image.height())
        for x in range(image.width())
        if abs(image.pixelColor(x, y).lightness() - background) > 60
    ]
    assert grip
    assert all(abs(y - image.height() / 2) < 20 for _x, y in grip)


def test_scroll_bar_handles_are_visible_without_hovering(app, qtbot):
    theme.apply_theme(app, "light")
    area = QtWidgets.QScrollArea()
    qtbot.addWidget(area)
    content = QtWidgets.QWidget()
    content.setMinimumSize(100, 2000)
    area.setWidget(content)
    area.resize(200, 300)
    area.show()
    qtbot.waitExposed(area)
    bar = area.verticalScrollBar()
    image = bar.grab().toImage()
    background = image.pixelColor(0, 0)
    handle_columns = {
        x
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y) != background
    }
    assert len(handle_columns) >= 6
