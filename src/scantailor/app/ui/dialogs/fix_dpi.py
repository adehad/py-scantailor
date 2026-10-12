"""Fix DPI dialog for images whose resolution is undefined."""

from pathlib import Path
from typing import ClassVar

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator

from scantailor.app.ui import embed_ui, load_ui_widget
from scantailor.app.ui.utils import get_cwidget
from scantailor.core.models import Dpi, ImageId
from scantailor.core.project import DpiStatus, ImageInfo, ImageMetadata

_UI_FOLDER = Path(__file__).parent
_ROW_INDEX_ROLE = Qt.ItemDataRole.UserRole


_DPI_WARNINGS: dict[DpiStatus, str] = {
    DpiStatus.TOO_LARGE: "DPI is too large and most likely wrong.",
    DpiStatus.TOO_SMALL: (
        "DPI is too small. Even if it's correct, you are not going to get "
        "acceptable results with it."
    ),
    DpiStatus.TOO_SMALL_FOR_PIXEL_SIZE: (
        "DPI is too small for this pixel size. Such combination would probably "
        "lead to out of memory errors."
    ),
}
# Digits of the longest DPI the fields take.
_DPI_FIELD_MAX_LENGTH = 4


def _is_dpi_ok(info: ImageInfo, dpi: Dpi) -> bool:
    """Whether ``dpi`` is defined for the image; odd values only warrant a warning."""
    return _metadata_with(info, dpi).is_dpi_ok()


def _metadata_with(info: ImageInfo, dpi: Dpi) -> ImageMetadata:
    """Return the image's metadata as it would be with ``dpi`` applied."""
    return ImageMetadata(
        width=info.metadata.width, height=info.metadata.height, dpi=dpi
    )


def _worst_status(statuses: list[DpiStatus]) -> DpiStatus:
    """Pick the status to show for several images: the first one that is not OK."""
    return next((status for status in statuses if status != DpiStatus.OK), DpiStatus.OK)


def _dpi_text(dpi: Dpi) -> str:
    """Format a DPI for display, with a placeholder when it is undefined."""
    if dpi.is_null():
        return "Undefined"
    if dpi.horizontal == dpi.vertical:
        return str(dpi.horizontal)
    return f"{dpi.horizontal}x{dpi.vertical}"


def _file_label(info: ImageInfo) -> str:
    """Name shown for an image; multi-page files list the page number too."""
    name = info.id.file_path.name
    return f"{name} [{info.id.page}]" if info.id.page else name


class FixDpiDialog(QtWidgets.QDialog):
    """Dialog for correcting the DPI of images.

    The first tab lists images whose DPI is undefined, the second all images.
    Doubtful but defined values are accepted and flagged in the fields. OK
    stays disabled until no image needs fixing.
    """

    DPI_PRESETS: ClassVar[list[tuple[str, int, int]]] = [
        ("Custom", 0, 0),
        ("300", 300, 300),
        ("400", 400, 400),
        ("600", 600, 600),
        ("1200", 1200, 1200),
    ]

    def __init__(
        self,
        images: list[ImageInfo],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the dialog.

        Args:
            images (list[ImageInfo]): Images to list, in display order.
            parent (QtWidgets.QWidget | None): Parent widget.
        """
        super().__init__(parent)
        self._images = list(images)
        self._effective: list[Dpi] = [info.metadata.dpi for info in self._images]
        self._initial_text: tuple[str, str] = ("", "")

        self.ui = load_ui_widget(_UI_FOLDER / "FixDpiDialog.ui", self)
        embed_ui(self, self.ui)
        self._setup_widgets()
        self._populate_views()
        self._connect_signals()
        self._prefill_from_selection()

    def _setup_widgets(self) -> None:
        """Look up the form's widgets and configure them."""
        self._tab_widget = get_cwidget(self.ui, QtWidgets.QTabWidget, "tabWidget")
        self._undefined_view = get_cwidget(
            self.ui, QtWidgets.QTreeView, "undefinedDpiView"
        )
        self._all_pages_view = get_cwidget(self.ui, QtWidgets.QTreeView, "allPagesView")
        self._dpi_combo = get_cwidget(self.ui, QtWidgets.QComboBox, "dpiCombo")
        self._x_dpi = get_cwidget(self.ui, QtWidgets.QLineEdit, "xDpi")
        self._y_dpi = get_cwidget(self.ui, QtWidgets.QLineEdit, "yDpi")
        self._apply_btn = get_cwidget(self.ui, QtWidgets.QPushButton, "applyBtn")
        self._button_box = get_cwidget(self.ui, QtWidgets.QDialogButtonBox, "buttonBox")

        self._dpi_combo.clear()
        for name, _, _ in self.DPI_PRESETS:
            self._dpi_combo.addItem(name)
        for field in (self._x_dpi, self._y_dpi):
            field.setMaxLength(_DPI_FIELD_MAX_LENGTH)
            field.setValidator(QIntValidator(field))
        self._normal_palette = self._x_dpi.palette()
        self._error_palette = QtGui.QPalette(self._normal_palette)
        self._error_palette.setColor(
            QtGui.QPalette.ColorRole.Text, QtGui.QColor(Qt.GlobalColor.red)
        )

    def _populate_views(self) -> None:
        """Fill both tree views from the images."""
        self._undefined_model = QtGui.QStandardItemModel(self)
        self._all_model = QtGui.QStandardItemModel(self)
        for model in (self._undefined_model, self._all_model):
            model.setHorizontalHeaderLabels(["File", "DPI"])
        self._undefined_view.setModel(self._undefined_model)
        self._all_pages_view.setModel(self._all_model)

        for row, info in enumerate(self._images):
            self._all_model.appendRow(self._make_row(row, info))
            if not _is_dpi_ok(info, self._effective[row]):
                self._undefined_model.appendRow(self._make_row(row, info))

        for view in (self._undefined_view, self._all_pages_view):
            view.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
            )
            view.setRootIsDecorated(False)
            view.header().setStretchLastSection(True)

    def _make_row(self, index: int, info: ImageInfo) -> list[QtGui.QStandardItem]:
        """Build the model row of one image.

        Args:
            index (int): Position of the image in the dialog's image list.
            info (ImageInfo): The image.

        Returns:
            list[QtGui.QStandardItem]: Name and DPI cells.
        """
        name_item = QtGui.QStandardItem(_file_label(info))
        name_item.setData(index, _ROW_INDEX_ROLE)
        name_item.setToolTip(str(info.id.file_path))
        name_item.setEditable(False)
        dpi_item = QtGui.QStandardItem(_dpi_text(self._effective[index]))
        dpi_item.setEditable(False)
        return [name_item, dpi_item]

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        for view in (self._undefined_view, self._all_pages_view):
            view.selectionModel().selectionChanged.connect(self._prefill_from_selection)
        self._tab_widget.currentChanged.connect(self._prefill_from_selection)
        self._dpi_combo.currentIndexChanged.connect(self._on_dpi_preset_changed)
        self._x_dpi.textChanged.connect(self._on_dpi_text_changed)
        self._y_dpi.textChanged.connect(self._on_dpi_text_changed)
        self._apply_btn.clicked.connect(self._apply_dpi)

    def _current_view(self) -> QtWidgets.QTreeView:
        """Return the tree view of the visible tab."""
        if self._tab_widget.currentIndex() == 0:
            return self._undefined_view
        return self._all_pages_view

    def _selected_image_indexes(self) -> list[int]:
        """Return the image list positions selected in the visible tab."""
        view = self._current_view()
        model = view.model()
        return [
            model.data(model.index(index.row(), 0), _ROW_INDEX_ROLE)
            for index in view.selectionModel().selectedRows()
        ]

    def _entered_dpi(self) -> Dpi | None:
        """Return the DPI typed into the fields, or None if incomplete or negative."""
        try:
            horizontal, vertical = int(self._x_dpi.text()), int(self._y_dpi.text())
        except ValueError:
            return None
        if horizontal < 0 or vertical < 0:
            return None
        return Dpi(horizontal=horizontal, vertical=vertical)

    def _prefill_from_selection(self) -> None:
        """Show the DPI the selected images share; clear the fields if they differ."""
        selected = self._selected_image_indexes()
        shared = {
            (self._effective[i].horizontal, self._effective[i].vertical)
            for i in selected
        }
        horizontal, vertical = shared.pop() if len(shared) == 1 else (0, 0)
        if Dpi(horizontal=horizontal, vertical=vertical).is_null():
            self._initial_text = ("", "")
        else:
            self._initial_text = (str(horizontal), str(vertical))
        with QtCore.QSignalBlocker(self._x_dpi), QtCore.QSignalBlocker(self._y_dpi):
            self._x_dpi.setText(self._initial_text[0])
            self._y_dpi.setText(self._initial_text[1])
        self._on_dpi_text_changed()

    def _update_controls(self) -> None:
        """Enable the DPI controls and the buttons to match the current state."""
        selected = self._selected_image_indexes()
        for widget in (self._dpi_combo, self._x_dpi, self._y_dpi):
            widget.setEnabled(bool(selected))
        entered = self._entered_dpi()
        self._decorate_dpi_fields(selected, entered)
        unchanged = (self._x_dpi.text(), self._y_dpi.text()) == self._initial_text
        self._apply_btn.setEnabled(
            bool(selected)
            and not unchanged
            and entered is not None
            and all(_is_dpi_ok(self._images[i], entered) for i in selected)
        )
        ok_button = self._button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        if ok_button is not None:
            ok_button.setEnabled(not self.images_needing_fix())

    def _on_dpi_text_changed(self) -> None:
        """Keep the preset box in step with the typed values."""
        entered = self._entered_dpi()
        preset = 0
        if entered is not None:
            for row, (_, horizontal, vertical) in enumerate(self.DPI_PRESETS):
                if row and (horizontal, vertical) == (
                    entered.horizontal,
                    entered.vertical,
                ):
                    preset = row
        with QtCore.QSignalBlocker(self._dpi_combo):
            self._dpi_combo.setCurrentIndex(preset)
        self._update_controls()

    def _decorate_dpi_fields(self, selected: list[int], entered: Dpi | None) -> None:
        """Mark a field red and explain why when its value is doubtful.

        Args:
            selected (list[int]): Selected image positions.
            entered (Dpi | None): The typed DPI, or None while incomplete.
        """
        if entered is None or not selected:
            for field in (self._x_dpi, self._y_dpi):
                self._decorate(field, DpiStatus.UNDEFINED)
            return
        metadata = [_metadata_with(self._images[i], entered) for i in selected]
        self._decorate(
            self._x_dpi, _worst_status([m.horizontal_dpi_status() for m in metadata])
        )
        self._decorate(
            self._y_dpi, _worst_status([m.vertical_dpi_status() for m in metadata])
        )

    def _decorate(self, field: QtWidgets.QLineEdit, status: DpiStatus) -> None:
        """Colour ``field`` and set its tooltip for ``status``."""
        field.setPalette(
            self._normal_palette if status == DpiStatus.OK else self._error_palette
        )
        field.setToolTip(
            self.tr(_DPI_WARNINGS[status]) if status in _DPI_WARNINGS else ""
        )

    def _on_dpi_preset_changed(self, index: int) -> None:
        """Copy a preset's values into the DPI fields."""
        if not 0 <= index < len(self.DPI_PRESETS):
            return
        _, horizontal, vertical = self.DPI_PRESETS[index]
        if horizontal > 0:
            self._x_dpi.setText(str(horizontal))
            self._y_dpi.setText(str(vertical))

    def _apply_dpi(self) -> None:
        """Give the selected images the DPI typed into the fields."""
        entered = self._entered_dpi()
        selected = self._selected_image_indexes()
        if entered is None or not selected:
            return
        for index in selected:
            self._effective[index] = entered
        self._refresh_dpi_cells()
        self._update_controls()

    def _refresh_dpi_cells(self) -> None:
        """Show the current DPI of every row in both views."""
        for model in (self._undefined_model, self._all_model):
            for row in range(model.rowCount()):
                index = model.item(row, 0).data(_ROW_INDEX_ROLE)
                model.item(row, 1).setText(_dpi_text(self._effective[index]))

    def images_needing_fix(self) -> list[ImageId]:
        """Return the images whose current DPI is still undefined."""
        return [
            info.id
            for info, dpi in zip(self._images, self._effective, strict=True)
            if not _is_dpi_ok(info, dpi)
        ]

    def get_dpi_changes(self) -> dict[ImageId, Dpi]:
        """Return the DPI chosen for every image whose value changed.

        Returns:
            dict[ImageId, Dpi]: New DPI by image.
        """
        return {
            info.id: dpi
            for info, dpi in zip(self._images, self._effective, strict=True)
            if dpi != info.metadata.dpi
        }
