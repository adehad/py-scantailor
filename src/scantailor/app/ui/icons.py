"""Lucide and custom glyphs as QIcons drawn from the application palette.

Glyphs are SVG files in the Qt resources (``:/icons/lucide`` and ``:/icons/custom``)
that draw with ``currentColor``. The icon engine paints them on request, so every
size and device pixel ratio is rendered from the vector and a palette change is
picked up the next time a widget repaints.
"""

import functools

from loguru import logger
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

from scantailor.app._ui import resources_rc

assert resources_rc, "The import ensures the glyph files are registered with Qt."

ICON_NAME_PROPERTY = "iconName"
ICON_NAME_ON_PROPERTY = "iconNameOn"
ENGINE_KEY = "scantailor-svg"
_NAME_PREFIX = "scantailor:"
_RESOURCE_FOLDERS = (":/icons/lucide/", ":/icons/custom/")
_CURRENT_COLOR = b"currentColor"


@functools.cache
def svg_source(name: str) -> bytes:
    """Read a glyph from the resources.

    Args:
        name (str): File name without extension, for example ``"chevron-down"``.

    Returns:
        bytes: The SVG document.

    Raises:
        ValueError: If no glyph has that name.
    """
    for folder in _RESOURCE_FOLDERS:
        file = QtCore.QFile(f"{folder}{name}.svg")
        if file.open(QtCore.QIODevice.OpenModeFlag.ReadOnly):
            try:
                return bytes(file.readAll().data())
            finally:
                file.close()
    raise ValueError(f"Unknown icon {name!r}")


def tinted_svg(name: str, color: QtGui.QColor) -> bytes:
    """Return a glyph with ``currentColor`` replaced by an opaque colour.

    Args:
        name (str): Glyph name as for svg_source.
        color (QtGui.QColor): Colour to draw with; its alpha is ignored.

    Returns:
        bytes: The recoloured SVG document.
    """
    rgb = color.name(QtGui.QColor.NameFormat.HexRgb).encode()
    return svg_source(name).replace(_CURRENT_COLOR, rgb)


@functools.lru_cache(maxsize=512)
def _renderer(name: str, rgb: int) -> QtSvg.QSvgRenderer:
    color = QtGui.QColor.fromRgb(rgb)
    return QtSvg.QSvgRenderer(QtCore.QByteArray(tinted_svg(name, color)))


def mode_color(mode: QtGui.QIcon.Mode) -> QtGui.QColor:
    """Pick the glyph colour for an icon mode from the application palette.

    Args:
        mode (QtGui.QIcon.Mode): Normal is ButtonText, Disabled is the disabled
            ButtonText, Active (hover) is the accent and Selected is
            HighlightedText.

    Returns:
        QtGui.QColor: The colour, possibly translucent.
    """
    palette = QtGui.QGuiApplication.palette()
    role = QtGui.QPalette.ColorRole
    if mode == QtGui.QIcon.Mode.Disabled:
        return palette.color(QtGui.QPalette.ColorGroup.Disabled, role.ButtonText)
    if mode == QtGui.QIcon.Mode.Active:
        return palette.color(role.Accent)
    if mode == QtGui.QIcon.Mode.Selected:
        return palette.color(role.HighlightedText)
    return palette.color(role.ButtonText)


class SvgIconEngine(QtGui.QIconEngine):
    """Draws a glyph, or a pair of glyphs for the Off and On states of a toggle."""

    def __init__(self, name: str, name_on: str | None = None) -> None:
        """Create the engine.

        Args:
            name (str): Glyph for the Off state, and for On when name_on is None.
            name_on (str | None): Glyph for the On (checked) state.
        """
        super().__init__()
        svg_source(name)
        if name_on is not None:
            svg_source(name_on)
        self._name = name
        self._name_on = name_on

    def _glyph(self, state: QtGui.QIcon.State) -> str:
        if state == QtGui.QIcon.State.On and self._name_on is not None:
            return self._name_on
        return self._name

    def paint(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        mode: QtGui.QIcon.Mode,
        state: QtGui.QIcon.State,
    ) -> None:
        """Draw the glyph into a rectangle.

        Args:
            painter (QtGui.QPainter): Target painter.
            rect (QtCore.QRect): Where to draw, in painter coordinates.
            mode (QtGui.QIcon.Mode): Selects the colour.
            state (QtGui.QIcon.State): Selects the glyph of a toggle pair.
        """
        color = mode_color(mode)
        renderer = _renderer(self._glyph(state), color.rgb())
        painter.save()
        painter.setOpacity(painter.opacity() * color.alphaF())
        renderer.render(painter, QtCore.QRectF(rect))
        painter.restore()

    def pixmap(
        self, size: QtCore.QSize, mode: QtGui.QIcon.Mode, state: QtGui.QIcon.State
    ) -> QtGui.QPixmap:
        """Render at a device pixel ratio of 1.

        Args:
            size (QtCore.QSize): Pixmap size.
            mode (QtGui.QIcon.Mode): Selects the colour.
            state (QtGui.QIcon.State): Selects the glyph of a toggle pair.

        Returns:
            QtGui.QPixmap: The rendered glyph.
        """
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(
        self,
        size: QtCore.QSize,
        mode: QtGui.QIcon.Mode,
        state: QtGui.QIcon.State,
        scale: float,
    ) -> QtGui.QPixmap:
        """Render the vector at device resolution.

        Args:
            size (QtCore.QSize): Size in device independent pixels.
            mode (QtGui.QIcon.Mode): Selects the colour.
            state (QtGui.QIcon.State): Selects the glyph of a toggle pair.
            scale (float): Device pixel ratio.

        Returns:
            QtGui.QPixmap: A pixmap of size * scale device pixels.
        """
        device_size = QtCore.QSize(
            max(1, round(size.width() * scale)), max(1, round(size.height() * scale))
        )
        pixmap = QtGui.QPixmap(device_size)
        pixmap.setDevicePixelRatio(scale)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.paint(painter, QtCore.QRect(QtCore.QPoint(0, 0), size), mode, state)
        painter.end()
        return pixmap

    def clone(self) -> "SvgIconEngine":
        """Return a copy, as Qt requires for implicitly shared icons."""
        return SvgIconEngine(self._name, self._name_on)

    def key(self) -> str:
        """Identify the engine to Qt."""
        return ENGINE_KEY

    def iconName(self) -> str:
        """Return the glyph name, prefixed so theme code can recognise the icon."""
        return f"{_NAME_PREFIX}{self._name}"


def icon(name: str, name_on: str | None = None) -> QtGui.QIcon:
    """Create a palette-aware icon.

    Args:
        name (str): Lucide or custom glyph name, for example ``"zoom-in"``. For a
            toggle pair this is the Off glyph.
        name_on (str | None): Glyph shown while the button is checked.

    Returns:
        QtGui.QIcon: A non-null icon that follows the application palette.

    Raises:
        ValueError: If a glyph name is unknown.
    """
    return QtGui.QIcon(SvgIconEngine(name, name_on))


def is_palette_icon(candidate: QtGui.QIcon) -> bool:
    """Whether an icon was made by icon().

    Args:
        candidate (QtGui.QIcon): Icon to test.

    Returns:
        bool: True if it recolours itself from the palette.
    """
    return candidate.name().startswith(_NAME_PREFIX)


class IconLabel(QtWidgets.QLabel):
    """A label showing one glyph, redrawn when the palette or pixel ratio changes."""

    def __init__(
        self,
        name: str,
        size: int = 24,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Create the label.

        Args:
            name (str): Glyph name as for icon().
            size (int): Edge length of the glyph in device independent pixels.
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self._size = size
        self._name = name
        self.setFixedSize(size, size)
        self._refresh()

    def set_glyph(self, name: str) -> None:
        """Show another glyph.

        Args:
            name (str): Glyph name as for icon().
        """
        self._name = name
        self._refresh()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        """Redraw after a palette change.

        Args:
            event (QtCore.QEvent): The event.
        """
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.PaletteChange:
            self._refresh()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Redraw for the pixel ratio of the screen the label appears on.

        Args:
            event (QtGui.QShowEvent): The event.
        """
        super().showEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        size = QtCore.QSize(self._size, self._size)
        self.setPixmap(icon(self._name).pixmap(size, self.devicePixelRatioF()))


class IconBinder(QtCore.QObject):
    """Gives buttons their icon from the ``iconName`` dynamic property.

    A ``.ui`` file or a widget sets ``iconName`` (and ``iconNameOn`` for a toggle)
    and the icon is created when the button is first polished, so no code has to
    reach into the form.
    """

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Set the icon of a button on its first Polish event.

        Args:
            watched (QtCore.QObject): Object receiving the event.
            event (QtCore.QEvent): The event.

        Returns:
            bool: Always False; the event is never consumed.
        """
        if event.type() == QtCore.QEvent.Type.Polish and isinstance(
            watched, QtWidgets.QAbstractButton
        ):
            name = watched.property(ICON_NAME_PROPERTY)
            if name:
                name_on = watched.property(ICON_NAME_ON_PROPERTY) or None
                try:
                    watched.setIcon(icon(str(name), name_on and str(name_on)))
                except ValueError as error:
                    logger.warning("{}: {}", watched.objectName(), error)
                watched.setProperty(ICON_NAME_PROPERTY, None)
        return False
