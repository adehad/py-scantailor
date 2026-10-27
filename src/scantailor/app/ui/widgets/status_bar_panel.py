"""Status bar panel showing the page, its size, its resolution and the cursor."""

import math
from pathlib import PurePath

from PySide6 import QtCore, QtWidgets

from scantailor.app.ui.icons import icon
from scantailor.app.ui.theme.fonts import tabular
from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.app.ui.utils import coordinate_separator, format_number
from scantailor.core.models import Dpi, PageId, SubPage
from scantailor.core.settings import Units, ZoneCreationMode
from scantailor.core.units import convert_length, units_suffix

_MAX_FILE_NAME_LENGTH = 15
_KEPT_FILE_NAME_TAIL = 13
_KEPT_SPLIT_FILE_NAME_TAIL = 11
_ZONE_MODE_ICON_SIZE = 16
_ZONE_MODE_GLYPHS = {
    ZoneCreationMode.POLYGONAL: "zone-polygon",
    ZoneCreationMode.LASSO: "lasso",
    ZoneCreationMode.RECTANGULAR: "square-dashed",
}


def _round_half_up(value: float) -> float:
    return math.floor(value + 0.5)


def _short_file_name(page: PageId) -> str:
    """Return the image's file name, abbreviated to fit the panel."""
    name = PurePath(page.image_id.file_path).stem
    if len(name) > _MAX_FILE_NAME_LENGTH:
        name = "..." + name[-_KEPT_FILE_NAME_TAIL:]
    return name


class StatusBarPanel(QtWidgets.QWidget):
    """Zone creation mode, cursor position, size, source DPI, page number and name.

    Sizes and positions arrive in pixels and are shown in the unit of the
    ``UnitsProvider``, which the panel follows. The source DPI is the one the
    project holds for the page's image, so a page scanned at another resolution
    than the rest stands out.
    """

    def __init__(
        self,
        units_provider: UnitsProvider | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Build an empty panel.

        Args:
            units_provider (UnitsProvider | None): Source of the unit sizes and
                positions are shown in; the shared one when None.
            parent (QtWidgets.QWidget | None): Owner of the widget.
        """
        super().__init__(parent)
        self._units_provider = units_provider or UnitsProvider.get_instance()
        self._size_px: QtCore.QSizeF | None = None
        self._cursor_px: QtCore.QPointF | None = None
        self._dpi = 0.0
        self._zone_mode: ZoneCreationMode | None = None
        self._fields: list[tuple[QtWidgets.QFrame, QtWidgets.QLabel]] = []

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.zone_mode_label = self._add_field(
            layout,
            self.tr("Zone creation mode."),
            self.tr("The type of a new zone created."),
        )
        self.mouse_pos_label = self._add_field(
            layout,
            self.tr("Mouse position."),
            self.tr("Mouse position relative to page."),
        )
        self.phys_size_label = self._add_field(
            layout, self.tr("Image size."), self.tr("Image size in current units.")
        )
        self.source_dpi_label = self._add_field(
            layout,
            self.tr(
                "Resolution of the page's source image. When most pages have another "
                "resolution, theirs follows in brackets."
            ),
            self.tr("Resolution of the source image."),
        )
        self.page_no_label = self._add_field(
            layout,
            self.tr("Page number."),
            self.tr("Position of the selected page in current order."),
        )
        self.page_info_label = self._add_field(
            layout, self.tr("Page information."), self.tr("Page name and type.")
        )
        self.clear()
        self._units_provider.units_changed.connect(self._show_measurements)

    def _add_field(
        self, layout: QtWidgets.QHBoxLayout, tooltip: str, status_tip: str
    ) -> QtWidgets.QLabel:
        """Append a hidden label and the separator that precedes it."""
        separator = QtWidgets.QFrame(self)
        separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        label = QtWidgets.QLabel(self)
        label.setProperty("textRole", "small")
        label.setFont(tabular(label.font()))
        label.setToolTip(tooltip)
        label.setStatusTip(status_tip)
        layout.addWidget(separator)
        layout.addWidget(label)
        self._fields.append((separator, label))
        return label

    def clear(self) -> None:
        """Forget the page, the size and the cursor position.

        The zone creation mode stays; it belongs to the view, which reports it
        with ``set_zone_creation_mode``.
        """
        self._size_px = None
        self._cursor_px = None
        self._dpi = 0.0
        for _, label in self._fields:
            label.clear()
        self._show_zone_mode()

    def set_zone_creation_mode(self, mode: ZoneCreationMode | None) -> None:
        """Show the icon of the way new zones are drawn.

        Args:
            mode (ZoneCreationMode | None): The mode, or None to hide the field
                while the view draws no zones.
        """
        self._zone_mode = mode
        self._show_zone_mode()

    def _show_zone_mode(self) -> None:
        if self._zone_mode is None:
            self.zone_mode_label.clear()
        else:
            self.zone_mode_label.setPixmap(
                icon(_ZONE_MODE_GLYPHS[self._zone_mode]).pixmap(
                    _ZONE_MODE_ICON_SIZE, _ZONE_MODE_ICON_SIZE
                )
            )
        self._sync_visibility()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        """Redraw the zone mode icon in the new palette's ink."""
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.PaletteChange:
            self._show_zone_mode()

    def update_page(self, number: int, count: int, page: PageId) -> None:
        """Show which page is current.

        Args:
            number (int): One-based position of the page in the displayed order.
            count (int): Number of pages in the project.
            page (PageId): The current page.
        """
        self.page_no_label.setText(
            self.tr("p. %1 / %2").replace("%1", str(number)).replace("%2", str(count))
        )
        name = _short_file_name(page)
        if page.sub_page != SubPage.SINGLE_PAGE:
            marker = (
                self.tr(" [L]")
                if page.sub_page == SubPage.LEFT_PAGE
                else self.tr(" [R]")
            )
            name = name[-_KEPT_SPLIT_FILE_NAME_TAIL:] + marker
        self.page_info_label.setText(name)
        self._sync_visibility()

    def set_page_size(self, size: QtCore.QSizeF | None, dpi: float) -> None:
        """Show the size of the page in the view.

        Args:
            size (QtCore.QSizeF | None): Size in pixels at ``dpi``; None when the
                view shows no page.
            dpi (float): Resolution of the pixels; values of 1 or less mean it is
                unknown, which hides the size and the cursor position.
        """
        self._size_px = size
        self._dpi = dpi
        self._show_measurements()

    def set_source_dpi(self, dpi: Dpi | None, usual: Dpi | None = None) -> None:
        """Show the resolution of the current page's source image.

        Args:
            dpi (Dpi | None): Resolution the project holds for the image; None or
                an undefined one hides the field.
            usual (Dpi | None): Resolution of most pages, shown beside ``dpi``
                when the two differ; None when there is no such majority.
        """
        if dpi is None or dpi.is_null():
            self.source_dpi_label.clear()
        elif usual is None or usual.is_null() or usual == dpi:
            self.source_dpi_label.setText(self._dpi_text(dpi))
        else:
            self.source_dpi_label.setText(
                self.tr("%1 (most pages %2)")
                .replace("%1", self._dpi_text(dpi))
                .replace("%2", self._dpi_text(usual))
            )
        self._sync_visibility()

    def _dpi_text(self, dpi: Dpi) -> str:
        if dpi.horizontal == dpi.vertical:
            return self.tr("%1 dpi").replace("%1", str(dpi.horizontal))
        return (
            self.tr("%1 x %2 dpi")
            .replace("%1", str(dpi.horizontal))
            .replace("%2", str(dpi.vertical))
        )

    def set_cursor_position(self, position: QtCore.QPointF | None) -> None:
        """Show where the cursor is on the page.

        Args:
            position (QtCore.QPointF | None): Pixels from the page's top left
                corner, or None when the cursor is off the page.
        """
        self._cursor_px = position
        self._show_measurements()

    def _show_measurements(self) -> None:
        units = self._units_provider.units
        known_dpi = self._dpi > 1
        size, cursor = self._size_px, self._cursor_px
        if size is not None and known_dpi and not size.isEmpty():
            width = convert_length(size.width(), Units.PIXELS, units, self._dpi)
            height = convert_length(size.height(), Units.PIXELS, units, self._dpi)
            fine = units in (Units.CENTIMETERS, Units.INCHES)
            if fine:
                width, height = (_round_half_up(v * 10) / 10 for v in (width, height))
            else:
                width, height = _round_half_up(width), _round_half_up(height)
            numbers = f"{format_number(width)} x {format_number(height)}"
            self.phys_size_label.setText(f"{numbers} {units_suffix(units)}")
        else:
            self.phys_size_label.clear()
        if cursor is not None and known_dpi:
            x = convert_length(cursor.x(), Units.PIXELS, units, self._dpi)
            y = convert_length(cursor.y(), Units.PIXELS, units, self._dpi)
            if units in (Units.PIXELS, Units.MILLIMETERS):
                x, y = math.ceil(x), math.ceil(y)
            else:
                x, y = math.ceil(x * 10) / 10, math.ceil(y * 10) / 10
            self.mouse_pos_label.setText(
                format_number(x) + coordinate_separator() + format_number(y)
            )
        else:
            self.mouse_pos_label.clear()
        self._sync_visibility()

    def _sync_visibility(self) -> None:
        anything_before = False
        for separator, label in self._fields:
            shown = bool(label.text()) or not label.pixmap().isNull()
            label.setVisible(shown)
            separator.setVisible(shown and anything_before)
            anything_before = anything_before or shown
