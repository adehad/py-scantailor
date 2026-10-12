"""Overlay colour tokens of the image views and the dimmer disabled look."""

import dataclasses
import pathlib
import re

import pytest
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app.ui import image_views, theme
from scantailor.app.ui.theme.tokens import (
    DARK,
    LIGHT,
    ViewColors,
    qcolor,
    set_active_tokens,
    view_color,
)

IMAGE_VIEWS_DIR = pathlib.Path(image_views.__file__).resolve().parent

COLOR_LITERAL = re.compile(
    r"QColor\(|GlobalColor|Qt\.(?:white|black|red|green|blue|gray|yellow|cyan|magenta)\b"
)

# Lines that build a colour from data rather than naming one: a copy of a colour
# already resolved from the theme or palette, and the colour a user gave a zone.
ALLOWED_LITERALS = {
    ("deskew.py", "fill = QColor(accent)"),
    ("output.py", "fill = QColor(*zone.fill_color, self.FILL_PREVIEW_ALPHA)"),
    ("output.py", "fill = QColor(fill)"),
}


def stripped_literal_lines() -> list[tuple[str, str]]:
    found = []
    for path in sorted(IMAGE_VIEWS_DIR.glob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            code = line.split("#", 1)[0]
            if COLOR_LITERAL.search(code) or re.match(
                r"\s*from .*import .*QColor", code
            ):
                found.append((path.name, line.strip()))
    return found


class TestNoHardCodedColours:
    def test_image_views_name_colours_only_through_tokens(self) -> None:
        offenders = [
            entry
            for entry in stripped_literal_lines()
            if entry not in ALLOWED_LITERALS
            and not entry[1].startswith(("from ", "import "))
        ]

        assert offenders == []

    def test_allow_list_has_no_stale_entries(self) -> None:
        present = set(stripped_literal_lines())

        stale = {entry for entry in ALLOWED_LITERALS if entry not in present}

        assert stale == set()

    def test_no_class_level_colour_constants_remain(self) -> None:
        pattern = re.compile(r"^\s+[A-Z_]*COLORS?\b.*=", re.MULTILINE)

        offenders = [
            path.name
            for path in IMAGE_VIEWS_DIR.glob("*.py")
            if pattern.search(path.read_text(encoding="utf-8"))
        ]

        assert offenders == []


class TestViewTokens:
    @pytest.mark.parametrize("tokens", [LIGHT, DARK], ids=["light", "dark"])
    def test_every_overlay_role_is_a_valid_colour(self, tokens) -> None:
        for field in dataclasses.fields(ViewColors):
            color = qcolor(getattr(tokens.view, field.name))

            assert color.isValid(), field.name

    def test_each_scheme_defines_every_overlay_role(self) -> None:
        assert {f.name for f in dataclasses.fields(LIGHT.view)} == {
            f.name for f in dataclasses.fields(DARK.view)
        }
        assert len(dataclasses.fields(ViewColors)) > 0

    def test_roles_keep_their_hues(self) -> None:
        for tokens in (LIGHT, DARK):
            content = qcolor(tokens.view.content_border)
            page = qcolor(tokens.view.page_border)
            middle = qcolor(tokens.view.margin_middle_border)

            assert content.blue() > content.red()
            assert page.red() > page.blue()
            assert middle.green() > middle.red()

    @pytest.mark.parametrize("tokens", [LIGHT, DARK], ids=["light", "dark"])
    def test_active_tokens_drive_view_color(self, tokens) -> None:
        set_active_tokens(tokens)
        try:
            assert view_color("split_line") == qcolor(tokens.view.split_line)
        finally:
            set_active_tokens(LIGHT)

    def test_view_color_returns_an_independent_copy(self) -> None:
        first = view_color("split_line")
        first.setAlpha(1)

        assert view_color("split_line").alpha() != 1

    def test_schemes_choose_different_overlay_colours(self) -> None:
        assert LIGHT.view != DARK.view


@pytest.fixture
def themed(qapp: QtWidgets.QApplication, monkeypatch):
    """Apply a scheme and restore the application look afterwards."""
    monkeypatch.delenv(theme.THEME_ENV_VAR, raising=False)
    style = qapp.style().name()
    palette = qapp.palette()
    font = qapp.font()

    def apply(scheme: str) -> None:
        theme.apply_theme(qapp, scheme)

    yield apply
    controller = qapp.findChild(QtCore.QObject, "scantailorThemeController")
    if controller is not None:
        shiboken6.delete(controller)
    qapp.setStyleSheet("")
    qapp.setStyle(style)
    qapp.setPalette(palette)
    qapp.setFont(font)
    set_active_tokens(LIGHT)


# Skips the indicator, whose frame is drawn at any state.
CHECK_BOX_LABEL_LEFT = 44


def luminance(color: QtGui.QColor) -> float:
    return 0.2126 * color.redF() + 0.7152 * color.greenF() + 0.0722 * color.blueF()


def ink_contrast(host: QtWidgets.QDialog, widget: QtWidgets.QWidget) -> float:
    """Largest luminance gap between any pixel and the dominant colour."""
    image = host.grab().toImage()
    if isinstance(widget, QtWidgets.QCheckBox):
        left = CHECK_BOX_LABEL_LEFT
        image = image.copy(left, 0, image.width() - left, image.height())
    pixels = [
        QtGui.QColor(image.pixel(x, y))
        for y in range(image.height())
        for x in range(image.width())
    ]
    values = sorted(luminance(p) for p in pixels)
    background = values[len(values) // 2]
    return max(abs(v - background) for v in values)


def make_widgets() -> list[QtWidgets.QWidget]:
    button = QtWidgets.QPushButton("Apply")
    check = QtWidgets.QCheckBox("Enable OpenGL")
    edit = QtWidgets.QLineEdit("Deviation")
    combo = QtWidgets.QComboBox()
    combo.addItem("English")
    spin = QtWidgets.QSpinBox()
    spin.setValue(12)
    return [button, check, edit, combo, spin]


@pytest.mark.parametrize("scheme", ["light", "dark"])
class TestDisabledLook:
    def test_disabled_controls_have_visibly_lower_ink_contrast(
        self, themed, scheme: str
    ) -> None:
        themed(scheme)
        for widget in make_widgets():
            host = QtWidgets.QDialog()
            QtWidgets.QVBoxLayout(host).addWidget(widget)
            host.resize(220, 60)
            enabled = ink_contrast(host, widget)
            widget.setEnabled(False)
            disabled = ink_contrast(host, widget)

            assert disabled < 0.8 * enabled, (type(widget).__name__, enabled, disabled)
            host.deleteLater()

    def test_disabled_text_is_dimmer_than_secondary_text(
        self, themed, qapp: QtWidgets.QApplication, scheme: str
    ) -> None:
        themed(scheme)
        palette = qapp.palette()
        group = QtGui.QPalette.ColorGroup
        role = QtGui.QPalette.ColorRole
        window = palette.color(group.Active, role.Window)
        dim = qcolor((LIGHT if scheme == "light" else DARK).text_dim)

        for text_role in (role.WindowText, role.Text, role.ButtonText):
            disabled = palette.color(group.Disabled, text_role)

            assert abs(luminance(disabled) - luminance(window)) < abs(
                luminance(dim) - luminance(window)
            )

    def test_disabled_palette_fills_differ_from_enabled(
        self, themed, qapp: QtWidgets.QApplication, scheme: str
    ) -> None:
        themed(scheme)
        palette = qapp.palette()
        group = QtGui.QPalette.ColorGroup
        role = QtGui.QPalette.ColorRole

        for fill_role in (role.Base, role.Button, role.Highlight):
            assert palette.color(group.Disabled, fill_role) != palette.color(
                group.Active, fill_role
            )

    def test_applying_a_scheme_activates_its_overlay_tokens(
        self, themed, scheme: str
    ) -> None:
        themed(scheme)

        expected = (LIGHT if scheme == "light" else DARK).view.content_border

        assert view_color("content_border") == qcolor(expected)
