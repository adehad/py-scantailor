"""One cross-platform glass look, light or dark, with a bundled typeface and icons.

The look follows the operating system colour scheme and updates when it changes.
Type sizes follow the application font and update when it changes.
The QSettings key ``appearance/theme`` (``auto``, ``light`` or ``dark``) forces a
scheme, and the ``SCANTAILOR_THEME`` environment variable overrides both.
"""

import os

from loguru import logger
from PySide6 import QtCore, QtGui, QtWidgets

from scantailor.app.ui.icons import IconBinder, set_semantic_colors

from .assets import write_assets
from .docks import DockSizer
from .fonts import TabularFigures, apply_fonts
from .layout_style import LayoutStyle
from .palette import build_palette
from .spin_boxes import SpinBoxTextInset
from .stylesheet import build_stylesheet
from .tokens import DARK, LIGHT, set_active_tokens

__all__ = [
    "AUTO",
    "SCHEMES",
    "SETTINGS_KEY",
    "THEME_ENV_VAR",
    "THEME_MODES",
    "apply_theme",
    "resolve_mode",
    "resolve_scheme",
]

AUTO = "auto"
THEME_ENV_VAR = "SCANTAILOR_THEME"
SETTINGS_KEY = "appearance/theme"
_TOKENS = {"light": LIGHT, "dark": DARK}
SCHEMES = tuple(_TOKENS)
THEME_MODES = (AUTO, *SCHEMES)
_CONTROLLER_NAME = "scantailorThemeController"


def resolve_mode() -> str:
    """Pick the requested mode from SCANTAILOR_THEME, then QSettings.

    Returns:
        str: A name in THEME_MODES; unknown values fall back to "auto".
    """
    raw = os.environ.get(THEME_ENV_VAR) or str(
        QtCore.QSettings().value(SETTINGS_KEY, AUTO)
    )
    mode = raw.strip().lower()
    if mode not in THEME_MODES:
        logger.warning("Unknown theme {!r}; using {!r}", raw, AUTO)
        return AUTO
    return mode


def resolve_scheme(app: QtWidgets.QApplication, mode: str) -> str:
    """Turn a mode into the concrete scheme to draw.

    Args:
        app (QtWidgets.QApplication): The running application.
        mode (str): One of THEME_MODES.

    Returns:
        str: "light" or "dark"; "auto" follows the operating system and
            treats an unknown colour scheme as light.
    """
    if mode in _TOKENS:
        return mode
    return "dark" if system_color_scheme(app) == QtCore.Qt.ColorScheme.Dark else "light"


def system_color_scheme(app: QtWidgets.QApplication) -> QtCore.Qt.ColorScheme:
    """Report the operating system colour scheme.

    Args:
        app (QtWidgets.QApplication): The running application.

    Returns:
        QtCore.Qt.ColorScheme: Unknown when the platform does not say.
    """
    return app.styleHints().colorScheme()


class _ThemeController(QtCore.QObject):
    """Owns the application look and re-applies it when the OS scheme changes."""

    def __init__(self, app: QtWidgets.QApplication) -> None:
        super().__init__(app)
        self.setObjectName(_CONTROLLER_NAME)
        self._app = app
        self._requested: str | None = None
        self.scheme = "light"
        # Native styles ignore the palette and several QSS rules.
        app.setStyle(LayoutStyle(QtWidgets.QStyleFactory.create("Fusion")))
        apply_fonts(app)
        app.installEventFilter(IconBinder(self))
        app.installEventFilter(TabularFigures(self))
        app.installEventFilter(SpinBoxTextInset(self))
        app.installEventFilter(DockSizer(self))
        app.styleHints().colorSchemeChanged.connect(self._on_scheme_changed)
        app.installEventFilter(self)

    def apply(self, mode: str | None) -> str:
        """Draw the look for a mode; None re-reads the environment and settings."""
        self._requested = mode
        return self._draw()

    def _draw(self) -> str:
        mode = self._requested if self._requested is not None else resolve_mode()
        scheme = resolve_scheme(self._app, mode)
        tokens = _TOKENS[scheme]
        self.scheme = scheme
        set_active_tokens(tokens)
        set_semantic_colors(QtGui.QColor(tokens.positive), QtGui.QColor(tokens.danger))
        self._app.setPalette(build_palette(tokens))
        self._app.setStyleSheet(
            build_stylesheet(tokens, write_assets(tokens, scheme), self._app.font())
        )
        return scheme

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Rebuild the stylesheet sizes when the application font changes.

        Args:
            watched (QtCore.QObject): Object receiving the event.
            event (QtCore.QEvent): The event.

        Returns:
            bool: Always False; the event is never consumed.
        """
        if (
            watched is self._app
            and event.type() == QtCore.QEvent.Type.ApplicationFontChange
        ):
            self._draw()
        return False

    def _on_scheme_changed(self, _scheme: QtCore.Qt.ColorScheme) -> None:
        mode = self._requested if self._requested is not None else resolve_mode()
        if mode == AUTO:
            self._draw()


def apply_theme(app: QtWidgets.QApplication, mode: str | None = None) -> str:
    """Apply the look to the application and keep it in step with the OS.

    Safe to call again, for example after the setting changed.

    Args:
        app (QtWidgets.QApplication): The running application.
        mode (str | None): "auto", "light" or "dark"; None resolves it from the
            environment and settings, and is resolved again on every OS change.

    Returns:
        str: The scheme drawn now, "light" or "dark".
    """
    if mode is not None and mode not in THEME_MODES:
        logger.warning("Unknown theme {!r}; using {!r}", mode, AUTO)
        mode = AUTO
    controller = app.findChild(_ThemeController, _CONTROLLER_NAME)
    if controller is None:
        controller = _ThemeController(app)
    return controller.apply(mode)
