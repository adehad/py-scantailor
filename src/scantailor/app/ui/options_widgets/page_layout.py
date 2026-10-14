"""Options widget for Page Layout filter.

Provides UI controls for configuring margins and alignment.
"""

from PySide6 import QtCore
from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon, QIntValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.ui import icons
from scantailor.app.ui.icons import icon
from scantailor.app.ui.image_views.page_layout_state import PageLayoutState
from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.app.ui.widgets.collapsible_group_box import CollapsibleGroupBox
from scantailor.core.models import Dpi, Margins, PageId
from scantailor.core.project import DpiStatus, ImageMetadata
from scantailor.core.settings import Units
from scantailor.core.units import convert_length, spin_box_precision
from scantailor.filters.page_layout.alignment import (
    Alignment,
    HorizontalAlignment,
    VerticalAlignment,
)
from scantailor.filters.page_layout.filter import Filter

# Margins are stored in millimetres; this bounds what the spin boxes accept.
_MAX_MARGIN_MM = 9999.0
_OPPOSITE_SIDE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
# Digits of the longest DPI the source DPI fields take.
_DPI_FIELD_MAX_LENGTH = 4
_DPI_FIELD_WIDTH = 56
_SOURCE_DPI_PRESETS = (300, 400, 600, 1200)


class OptionsWidget(QWidget):
    """Options widget for the Page Layout filter.

    Provides controls for:
    - Margins (top, bottom, left, right) with optional auto-margins, shown and
      edited in the application's current unit
    - Alignment (9-point grid for manual, dropdown for mode selection)
    - Match size with other pages option

    Signals:
        margins_changed: Emitted when margin values change.
            Args: page_id (PageId), margins (Margins) in millimetres
        alignment_changed: Emitted when alignment changes.
            Args: page_id (PageId), alignment (Alignment)
        apply_margins_requested: Emitted when user clicks "Apply To..." for margins
        apply_alignment_requested: Emitted when user clicks "Apply To..." for alignment
        match_size_to_all_requested: Emitted after the current page is set to match
            size with other pages and the other pages should follow
        aggregate_size_changed: Emitted when the size the pages are matched to
            may have changed, so thumbnails and the shown page need redrawing
        fix_dpi_requested: Emitted when user clicks "Fix all..."
        source_dpi_changed: Emitted with horizontal and vertical DPI (int) when
            the user sets a valid source DPI for the current page's image
    """

    margins_changed = Signal(PageId, Margins)
    alignment_changed = Signal(PageId, Alignment)
    apply_margins_requested = Signal()
    apply_alignment_requested = Signal()
    match_size_to_all_requested = Signal()
    aggregate_size_changed = Signal()
    fix_dpi_requested = Signal()
    source_dpi_changed = Signal(int, int)

    def __init__(
        self,
        filter_: Filter,
        parent: QWidget | None = None,
        units_provider: UnitsProvider | None = None,
        state: PageLayoutState | None = None,
    ) -> None:
        """Initialize the options widget.

        Args:
            filter_: The Page Layout filter instance.
            parent: The parent widget.
            units_provider: Source of the unit margins are shown in; the
                application-wide one when None.
            state: Margin link and guide state to share with the image view; a
                private one when None.
        """
        super().__init__(parent)

        self._filter = filter_
        self._units_provider = units_provider or UnitsProvider.get_instance()
        self._margins_mm = Margins()
        self._dpi = (0.0, 0.0)
        self._source_pixel_size = (0, 0)
        self._source_dpi = Dpi()
        self._current_page_id: PageId | None = None
        self._updating = False  # Prevent signal loops
        self._state = state or PageLayoutState(self)

        self._setup_ui()
        self._connect_signals()
        self._apply_units()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Margins group
        self._margins_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate("page_layout::OptionsWidget", "Margins"),
            self,
        )
        self._margins_group.setObjectName("marginsGroup")
        margins_layout = QVBoxLayout(self._margins_group)

        # Auto margins checkbox row
        auto_layout = QHBoxLayout()
        auto_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self._auto_margins_check = QCheckBox(
            QtCore.QCoreApplication.translate(
                "page_layout::OptionsWidget", "Auto Margins"
            )
        )
        self._auto_margins_check.setObjectName("autoMargins")
        auto_layout.addWidget(self._auto_margins_check)
        auto_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        margins_layout.addLayout(auto_layout)

        # Margins spinboxes grid
        spinbox_layout = QHBoxLayout()
        spinbox_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        margins_grid = QGridLayout()

        # Top/Bottom spinboxes and link
        margins_grid.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate("page_layout::OptionsWidget", "Top")
            ),
            0,
            0,
        )
        self._top_spinbox = QDoubleSpinBox()
        self._top_spinbox.setObjectName("topMarginSpinBox")
        self._top_spinbox.setDecimals(1)
        self._top_spinbox.setRange(-9999.0, 9999.0)
        margins_grid.addWidget(self._top_spinbox, 0, 1)

        self._top_bottom_link = QToolButton()
        self._top_bottom_link.setObjectName("topBottomLink")
        self._top_bottom_link.setIcon(icon("link-2-off", "link-2"))
        self._top_bottom_link.setCheckable(True)
        self._top_bottom_link.setToolTip(
            QtCore.QCoreApplication.translate(
                "page_layout::OptionsWidget", "Link top and bottom margins"
            )
        )
        self._top_bottom_link.setMinimumSize(20, 36)
        self._top_bottom_link.setMaximumSize(20, 36)
        margins_grid.addWidget(self._top_bottom_link, 0, 2, 2, 1)

        margins_grid.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate(
                    "page_layout::OptionsWidget", "Bottom"
                )
            ),
            1,
            0,
        )
        self._bottom_spinbox = QDoubleSpinBox()
        self._bottom_spinbox.setObjectName("bottomMarginSpinBox")
        self._bottom_spinbox.setDecimals(1)
        self._bottom_spinbox.setRange(-9999.0, 9999.0)
        margins_grid.addWidget(self._bottom_spinbox, 1, 1)

        # Left/Right spinboxes and link
        margins_grid.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate("page_layout::OptionsWidget", "Left")
            ),
            2,
            0,
        )
        self._left_spinbox = QDoubleSpinBox()
        self._left_spinbox.setObjectName("leftMarginSpinBox")
        self._left_spinbox.setDecimals(1)
        self._left_spinbox.setRange(-9999.0, 9999.0)
        margins_grid.addWidget(self._left_spinbox, 2, 1)

        self._left_right_link = QToolButton()
        self._left_right_link.setObjectName("leftRightLink")
        self._left_right_link.setIcon(icon("link-2-off", "link-2"))
        self._left_right_link.setCheckable(True)
        self._left_right_link.setToolTip(
            QtCore.QCoreApplication.translate(
                "page_layout::OptionsWidget", "Link left and right margins"
            )
        )
        self._left_right_link.setMinimumSize(20, 36)
        self._left_right_link.setMaximumSize(20, 36)
        margins_grid.addWidget(self._left_right_link, 2, 2, 2, 1)

        margins_grid.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate("page_layout::OptionsWidget", "Right")
            ),
            3,
            0,
        )
        self._right_spinbox = QDoubleSpinBox()
        self._right_spinbox.setObjectName("rightMarginSpinBox")
        self._right_spinbox.setDecimals(1)
        self._right_spinbox.setRange(-9999.0, 9999.0)
        margins_grid.addWidget(self._right_spinbox, 3, 1)

        spinbox_layout.addLayout(margins_grid)
        spinbox_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        margins_layout.addLayout(spinbox_layout)

        margins_layout.addLayout(self._build_source_dpi_row())

        # Apply margins button
        apply_margins_layout = QHBoxLayout()
        apply_margins_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self._apply_margins_btn = QPushButton(
            QtCore.QCoreApplication.translate(
                "page_layout::OptionsWidget", "Apply To ..."
            )
        )
        self._apply_margins_btn.setObjectName("applyMarginsBtn")
        apply_margins_layout.addWidget(self._apply_margins_btn)
        self._fix_dpi_btn = QPushButton(self.tr("Fix all..."))
        self._fix_dpi_btn.setObjectName("fixDpiBtn")
        apply_margins_layout.addWidget(self._fix_dpi_btn)
        apply_margins_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        margins_layout.addLayout(apply_margins_layout)

        layout.addWidget(self._margins_group)

        # Alignment group
        self._alignment_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate(
                "page_layout::OptionsWidget", "Alignment"
            ),
            self,
        )
        self._alignment_group.setObjectName("alignmentGroup")
        alignment_layout = QVBoxLayout(self._alignment_group)

        # Match size checkbox
        match_layout = QHBoxLayout()
        match_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self._match_size_check = QCheckBox(
            QtCore.QCoreApplication.translate(
                "page_layout::OptionsWidget", "Match size with other pages"
            )
        )
        self._match_size_check.setObjectName("alignWithOthersCB")
        self._match_size_check.setChecked(True)
        match_layout.addWidget(self._match_size_check)
        match_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        alignment_layout.addLayout(match_layout)

        # Alignment mode dropdowns
        mode_layout = QHBoxLayout()
        mode_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        mode_grid = QGridLayout()
        mode_grid.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate(
                    "page_layout::OptionsWidget", "Horizontal mode:"
                )
            ),
            0,
            0,
        )
        self._h_mode_combo = QComboBox()
        self._h_mode_combo.setObjectName("hAlignmentModeCB")
        self._h_mode_combo.addItems(
            [
                QtCore.QCoreApplication.translate("page_layout::OptionsWidget", "Auto"),
                QtCore.QCoreApplication.translate(
                    "page_layout::OptionsWidget", "Manual"
                ),
                QtCore.QCoreApplication.translate(
                    "page_layout::OptionsWidget", "Original"
                ),
            ]
        )
        self._h_mode_combo.setCurrentIndex(1)  # Manual by default
        mode_grid.addWidget(self._h_mode_combo, 0, 1)

        mode_grid.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate(
                    "page_layout::OptionsWidget", "Vertical mode:"
                )
            ),
            1,
            0,
        )
        self._v_mode_combo = QComboBox()
        self._v_mode_combo.setObjectName("vAlignmentModeCB")
        self._v_mode_combo.addItems(
            [
                QtCore.QCoreApplication.translate("page_layout::OptionsWidget", "Auto"),
                QtCore.QCoreApplication.translate(
                    "page_layout::OptionsWidget", "Manual"
                ),
                QtCore.QCoreApplication.translate(
                    "page_layout::OptionsWidget", "Original"
                ),
            ]
        )
        self._v_mode_combo.setCurrentIndex(1)  # Manual by default
        mode_grid.addWidget(self._v_mode_combo, 1, 1)

        mode_layout.addLayout(mode_grid)
        mode_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        alignment_layout.addLayout(mode_layout)

        # 9-point alignment grid
        align_grid_layout = QHBoxLayout()
        align_grid_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        align_grid = QGridLayout()
        align_grid.setSpacing(16)

        self._align_button_group = QButtonGroup(self)

        # Create 3x3 alignment button grid
        align_buttons = [
            ("align-north-west", 0, 0, "alignTopLeftBtn"),
            ("arrow-up-to-line", 0, 1, "alignTopBtn"),
            ("align-north-east", 0, 2, "alignTopRightBtn"),
            ("arrow-left-to-line", 1, 0, "alignLeftBtn"),
            ("shrink", 1, 1, "alignCenterBtn"),
            ("arrow-right-to-line", 1, 2, "alignRightBtn"),
            ("align-south-west", 2, 0, "alignBottomLeftBtn"),
            ("arrow-down-to-line", 2, 1, "alignBottomBtn"),
            ("align-south-east", 2, 2, "alignBottomRightBtn"),
        ]

        self._align_buttons: dict[str, QToolButton] = {}
        for glyph, row, col, name in align_buttons:
            btn = QToolButton()
            btn.setObjectName(name)
            btn.setIcon(icon(glyph))
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setMinimumSize(32, 32)
            if name == "alignCenterBtn":
                btn.setChecked(True)
            align_grid.addWidget(btn, row, col)
            self._align_button_group.addButton(btn)
            self._align_buttons[name] = btn

        align_grid_layout.addLayout(align_grid)
        align_grid_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        alignment_layout.addLayout(align_grid_layout)

        self._freeze_aggregate_check = QCheckBox(
            self.tr("Lock aggregate size for matching")
        )
        self._freeze_aggregate_check.setObjectName("freezeAggregateHardSizeCb")
        self._freeze_aggregate_check.setToolTip(
            self.tr(
                "Keep the aggregate page size used for matching fixed while you "
                "switch pages, so the reference dimensions do not jump."
            )
        )
        alignment_layout.addWidget(self._freeze_aggregate_check)

        # Apply alignment button
        apply_align_layout = QHBoxLayout()
        apply_align_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self._apply_alignment_btn = QPushButton(
            QtCore.QCoreApplication.translate(
                "page_layout::OptionsWidget", "Apply To ..."
            )
        )
        self._apply_alignment_btn.setObjectName("applyAlignmentBtn")
        apply_align_layout.addWidget(self._apply_alignment_btn)
        self._match_size_to_all_btn = QPushButton(self.tr("Match size to all pages"))
        self._match_size_to_all_btn.setObjectName("matchSizeToAllBtn")
        self._match_size_to_all_btn.setToolTip(
            self.tr(
                'Apply "Match size with other pages" to every page so they all '
                "share the same output size."
            )
        )
        apply_align_layout.addWidget(self._match_size_to_all_btn)
        apply_align_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        alignment_layout.addLayout(apply_align_layout)

        layout.addWidget(self._alignment_group)

        # Vertical spacer
        layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

    def _build_source_dpi_row(self) -> QVBoxLayout:
        """Build the rows that show and edit the DPI of the current page's image.

        The preset and the two fields do not fit side by side in the options dock,
        so the fields sit on a second row.
        """
        column = QVBoxLayout()
        row = QHBoxLayout()
        self._source_dpi_label = QLabel(self.tr("Source DPI"))
        self._source_dpi_label.setObjectName("sourceDpiLabel")
        row.addWidget(self._source_dpi_label)

        self._source_dpi_combo = QComboBox()
        self._source_dpi_combo.setObjectName("sourceDpiCombo")
        self._source_dpi_combo.addItem(self.tr("Custom"), None)
        for dpi in _SOURCE_DPI_PRESETS:
            self._source_dpi_combo.addItem(f"{dpi} x {dpi}", QSize(dpi, dpi))
        row.addWidget(self._source_dpi_combo, 1)
        column.addLayout(row)

        fields = QHBoxLayout()
        fields.addStretch(1)
        self._source_x_dpi = QLineEdit()
        self._source_x_dpi.setObjectName("sourceXDpi")
        fields.addWidget(self._source_x_dpi)
        self._source_dpi_times_label = QLabel("\u00d7")
        fields.addWidget(self._source_dpi_times_label)
        self._source_y_dpi = QLineEdit()
        self._source_y_dpi.setObjectName("sourceYDpi")
        fields.addWidget(self._source_y_dpi)
        column.addLayout(fields)

        for field in (self._source_x_dpi, self._source_y_dpi):
            field.setMaxLength(_DPI_FIELD_MAX_LENGTH)
            field.setMaximumWidth(_DPI_FIELD_WIDTH)
            field.setValidator(QIntValidator(field))
        return column

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        # Margins
        self._auto_margins_check.toggled.connect(self._on_auto_margins_changed)
        for side, spin in self._spin_boxes().items():
            spin.valueChanged.connect(
                lambda _value, side=side: self._on_margin_edited(side)
            )
        self._top_bottom_link.toggled.connect(self._on_top_bottom_link_changed)
        self._left_right_link.toggled.connect(self._on_left_right_link_changed)
        self._state.links_changed.connect(self._show_links)
        self._show_links()
        self._apply_margins_btn.clicked.connect(self.apply_margins_requested.emit)
        self._fix_dpi_btn.clicked.connect(self.fix_dpi_requested.emit)
        self._source_dpi_combo.activated.connect(self._on_source_dpi_preset_chosen)
        self._source_x_dpi.editingFinished.connect(self._on_source_dpi_edited)
        self._source_y_dpi.editingFinished.connect(self._on_source_dpi_edited)
        self._units_provider.units_changed.connect(self._apply_units)

        # Alignment
        self._match_size_check.toggled.connect(self._on_match_size_changed)
        self._h_mode_combo.currentIndexChanged.connect(self._on_h_mode_changed)
        self._v_mode_combo.currentIndexChanged.connect(self._on_v_mode_changed)
        self._align_button_group.buttonClicked.connect(
            self._on_alignment_button_clicked
        )
        self._apply_alignment_btn.clicked.connect(self.apply_alignment_requested.emit)
        self._match_size_to_all_btn.clicked.connect(self._on_match_size_to_all)
        self._freeze_aggregate_check.clicked.connect(self._on_freeze_aggregate_toggled)

    @property
    def state(self) -> PageLayoutState:
        """The margin link and guide state shared with the image view."""
        return self._state

    def _show_links(self) -> None:
        """Make the link buttons show the shared state."""
        for button, linked in (
            (self._top_bottom_link, self._state.top_bottom_linked),
            (self._left_right_link, self._state.left_right_linked),
        ):
            button.blockSignals(True)
            button.setChecked(linked)
            button.blockSignals(False)

    def set_current_page(self, page_id: PageId) -> None:
        """Set the current page being edited.

        Args:
            page_id: The page ID to edit.
        """
        self._current_page_id = page_id
        self._update_ui()

    def set_page_dpi(self, horizontal: float, vertical: float) -> None:
        """Tell the widget the resolution of the current page.

        Only pixel units need it; margins are shown unchanged while it is unknown.

        Args:
            horizontal: Dots per inch along x, or 0 when unknown.
            vertical: Dots per inch along y, or 0 when unknown.
        """
        self._dpi = (horizontal, vertical)
        self._apply_units()

    def set_source_image(self, width: int, height: int, dpi: Dpi) -> None:
        """Show the pixel size and resolution of the current page's image.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            dpi: Resolution the project uses for the image; null when undefined.
        """
        self._source_pixel_size = (width, height)
        self._source_dpi = dpi
        self._show_source_dpi()

    def _source_metadata(self, dpi: Dpi) -> ImageMetadata:
        width, height = self._source_pixel_size
        return ImageMetadata(width=width, height=height, dpi=dpi)

    def _show_source_dpi(self) -> None:
        """Fill the source DPI fields from the stored resolution."""
        for field in (self._source_x_dpi, self._source_y_dpi):
            field.blockSignals(True)
        try:
            if self._source_dpi.is_null():
                self._source_x_dpi.clear()
                self._source_y_dpi.clear()
            else:
                self._source_x_dpi.setText(str(self._source_dpi.horizontal))
                self._source_y_dpi.setText(str(self._source_dpi.vertical))
        finally:
            for field in (self._source_x_dpi, self._source_y_dpi):
                field.blockSignals(False)
        self._refresh_source_dpi_feedback()

    def _typed_source_dpi(self) -> Dpi | None:
        """Return the DPI typed into the fields, or None while it is incomplete."""
        try:
            horizontal = int(self._source_x_dpi.text())
            vertical = int(self._source_y_dpi.text())
        except ValueError:
            return None
        if horizontal < 0 or vertical < 0:
            return None
        return Dpi(horizontal=horizontal, vertical=vertical)

    def _refresh_source_dpi_feedback(self) -> None:
        """Match the preset box to the typed values and flag doubtful ones."""
        typed = self._typed_source_dpi()
        preset = 0
        if typed is not None:
            for row in range(1, self._source_dpi_combo.count()):
                size = self._source_dpi_combo.itemData(row)
                if (typed.horizontal, typed.vertical) == (size.width(), size.height()):
                    preset = row
                    break
        self._source_dpi_combo.setCurrentIndex(preset)

        if typed is None:
            self._decorate_source_dpi_field(self._source_x_dpi, DpiStatus.OK)
            self._decorate_source_dpi_field(self._source_y_dpi, DpiStatus.OK)
            return
        metadata = self._source_metadata(typed)
        self._decorate_source_dpi_field(
            self._source_x_dpi, metadata.horizontal_dpi_status()
        )
        self._decorate_source_dpi_field(
            self._source_y_dpi, metadata.vertical_dpi_status()
        )

    def _decorate_source_dpi_field(self, field: QLineEdit, status: DpiStatus) -> None:
        """Colour a DPI field and explain why when its value is doubtful."""
        warnings = {
            DpiStatus.TOO_LARGE: self.tr("DPI is too large and most likely wrong."),
            DpiStatus.TOO_SMALL: self.tr(
                "DPI is too small. Even if it's correct, you are not going to get "
                "acceptable results with it."
            ),
            DpiStatus.TOO_SMALL_FOR_PIXEL_SIZE: self.tr(
                "An extremely low DPI value. That might correspond to a very large "
                "paper size for the pixel size in question."
            ),
        }
        warning = warnings.get(status)
        if warning is None:
            field.setStyleSheet("")
            field.setToolTip("")
            return
        # The application style sheet ignores a palette set on the field.
        danger = icons.mode_color(QIcon.Mode.Normal, "danger").name()
        field.setStyleSheet(f"QLineEdit {{ color: {danger}; }}")
        field.setToolTip(warning)

    def _on_source_dpi_preset_chosen(self, index: int) -> None:
        """Copy a preset's values into the fields and commit them."""
        size = self._source_dpi_combo.itemData(index)
        if size is None:
            return
        self._source_x_dpi.setText(str(size.width()))
        self._source_y_dpi.setText(str(size.height()))
        self._on_source_dpi_edited()

    def _on_source_dpi_edited(self) -> None:
        """Report a valid, changed source DPI; otherwise only update the feedback."""
        self._refresh_source_dpi_feedback()
        typed = self._typed_source_dpi()
        if typed is None or typed == self._source_dpi:
            return
        if not self._source_metadata(typed).is_dpi_ok():
            return
        self._source_dpi = typed
        self.set_page_dpi(typed.horizontal, typed.vertical)
        self.source_dpi_changed.emit(typed.horizontal, typed.vertical)

    def _on_match_size_to_all(self) -> None:
        """Turn on size matching for the current page and ask for it on all pages."""
        page_id = self._current_page_id
        if page_id is None:
            return
        params = self._filter.get_params(page_id)
        alignment = Alignment(
            vertical=params.alignment.vertical,
            horizontal=params.alignment.horizontal,
            is_null=False,
        )
        self._filter.set_params(page_id, params.with_alignment(alignment))
        self._updating = True
        try:
            self._match_size_check.setChecked(True)
        finally:
            self._updating = False
        self.alignment_changed.emit(page_id, alignment)
        self.match_size_to_all_requested.emit()
        self.aggregate_size_changed.emit()

    def _on_freeze_aggregate_toggled(self, checked: bool) -> None:
        """Hold or release the size that pages are matched to."""
        self._filter.settings.set_aggregate_hard_size_frozen(checked)
        self.aggregate_size_changed.emit()

    def _spin_boxes(self) -> dict[str, QDoubleSpinBox]:
        return {
            "top": self._top_spinbox,
            "bottom": self._bottom_spinbox,
            "left": self._left_spinbox,
            "right": self._right_spinbox,
        }

    def _dpi_for(self, side: str) -> float:
        return self._dpi[0] if side in ("left", "right") else self._dpi[1]

    def _apply_units(self) -> None:
        """Re-show the stored margins in the current unit."""
        units = self._units_provider.units
        decimals, step = spin_box_precision(units)
        was_updating, self._updating = self._updating, True
        try:
            for side, spin in self._spin_boxes().items():
                dpi = self._dpi_for(side)
                limit = convert_length(_MAX_MARGIN_MM, Units.MILLIMETERS, units, dpi)
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
                spin.setRange(-limit, limit)
                spin.setValue(
                    convert_length(
                        getattr(self._margins_mm, side), Units.MILLIMETERS, units, dpi
                    )
                )
        finally:
            self._updating = was_updating

    def _update_ui(self) -> None:
        """Update the UI to reflect current page settings."""
        if self._current_page_id is None:
            return

        self._updating = True
        try:
            params = self._filter.get_params(self._current_page_id)

            # Update margins
            self._auto_margins_check.setChecked(params.auto_margins)
            self._margins_mm = params.hard_margins_mm
            self._apply_units()
            self._release_unequal_links()

            # Enable/disable spinboxes based on auto margins
            spinboxes_enabled = not params.auto_margins
            self._top_spinbox.setEnabled(spinboxes_enabled)
            self._bottom_spinbox.setEnabled(spinboxes_enabled)
            self._left_spinbox.setEnabled(spinboxes_enabled)
            self._right_spinbox.setEnabled(spinboxes_enabled)

            # Update alignment
            alignment = params.alignment
            self._match_size_check.setChecked(alignment.aligned_with_others())
            self._freeze_aggregate_check.setChecked(
                self._filter.settings.is_aggregate_hard_size_frozen()
            )

            # Update mode combos
            h_mode_index = self._get_h_mode_index(alignment.horizontal)
            v_mode_index = self._get_v_mode_index(alignment.vertical)
            self._h_mode_combo.setCurrentIndex(h_mode_index)
            self._v_mode_combo.setCurrentIndex(v_mode_index)

            # Update alignment buttons
            self._update_alignment_buttons(alignment)
        finally:
            self._updating = False

    def _release_unequal_links(self) -> None:
        """Unlink a pair whose displayed values differ, as they cannot move together."""
        if self._left_spinbox.value() != self._right_spinbox.value():
            self._state.left_right_linked = False
        if self._top_spinbox.value() != self._bottom_spinbox.value():
            self._state.top_bottom_linked = False

    def _get_h_mode_index(self, mode: HorizontalAlignment) -> int:
        """Get combo box index for horizontal alignment mode."""
        if mode == HorizontalAlignment.HAUTO:
            return 0
        if mode == HorizontalAlignment.HORIGINAL:
            return 2
        return 1  # Manual

    def _get_v_mode_index(self, mode: VerticalAlignment) -> int:
        """Get combo box index for vertical alignment mode."""
        if mode == VerticalAlignment.VAUTO:
            return 0
        if mode == VerticalAlignment.VORIGINAL:
            return 2
        return 1  # Manual

    def _update_alignment_buttons(self, alignment: Alignment) -> None:
        """Update which alignment button is checked."""
        # Map alignment to button name
        h = alignment.horizontal
        v = alignment.vertical

        v_map = {
            VerticalAlignment.TOP: "Top",
            VerticalAlignment.VCENTER: "",
            VerticalAlignment.BOTTOM: "Bottom",
        }
        h_map = {
            HorizontalAlignment.LEFT: "Left",
            HorizontalAlignment.HCENTER: "",
            HorizontalAlignment.RIGHT: "Right",
        }

        v_part = v_map.get(v, "")
        h_part = h_map.get(h, "")

        if not v_part and not h_part:
            btn_name = "alignCenterBtn"
        elif not v_part:
            btn_name = f"align{h_part}Btn"
        elif not h_part:
            btn_name = f"align{v_part}Btn"
        else:
            btn_name = f"align{v_part}{h_part}Btn"

        if btn_name in self._align_buttons:
            self._align_buttons[btn_name].setChecked(True)

    def _on_auto_margins_changed(self, checked: bool) -> None:
        """Handle auto margins checkbox change."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_auto_margins(checked)
        self._filter.set_params(self._current_page_id, new_params)
        self._update_ui()

    def _emit_margins_changed(self) -> None:
        """Emit margins_changed signal with current values."""
        if self._current_page_id is None:
            return

        margins = self._margins_mm
        params = self._filter.get_params(self._current_page_id)
        new_params = params.with_margins(margins)
        self._filter.set_params(self._current_page_id, new_params)
        self.margins_changed.emit(self._current_page_id, margins)

    def _on_margin_edited(self, side: str) -> None:
        """Store an edited margin, and its opposite one when the two are linked.

        Margins the user did not touch keep their stored millimetre values, so
        the rounding of the displayed unit does not creep into them.
        """
        if self._current_page_id is None or self._updating:
            return

        vertical = side in ("top", "bottom")
        linked = (
            self._state.top_bottom_linked if vertical else self._state.left_right_linked
        )
        spin_boxes = self._spin_boxes()
        sides = [side, _OPPOSITE_SIDE[side]] if linked else [side]
        value = spin_boxes[side].value()
        units = self._units_provider.units
        updates: dict[str, float] = {}
        self._updating = True
        try:
            for edited in sides:
                spin_boxes[edited].setValue(value)
                updates[edited] = convert_length(
                    spin_boxes[edited].value(),
                    units,
                    Units.MILLIMETERS,
                    self._dpi_for(edited),
                )
        finally:
            self._updating = False
        self._margins_mm = self._margins_mm.model_copy(update=updates)
        self._emit_margins_changed()

    def _on_top_bottom_link_changed(self, checked: bool) -> None:
        """Handle top/bottom link button change."""
        self._state.top_bottom_linked = checked
        self._state.remember_top_bottom_link(checked)

    def _on_left_right_link_changed(self, checked: bool) -> None:
        """Handle left/right link button change."""
        self._state.left_right_linked = checked
        self._state.remember_left_right_link(checked)

    def _on_match_size_changed(self, checked: bool) -> None:
        """Handle match size checkbox change."""
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        new_alignment = Alignment(
            vertical=params.alignment.vertical,
            horizontal=params.alignment.horizontal,
            is_null=not checked,
        )
        new_params = params.with_alignment(new_alignment)
        self._filter.set_params(self._current_page_id, new_params)
        self.alignment_changed.emit(self._current_page_id, new_alignment)

    def _on_h_mode_changed(self, index: int) -> None:
        """Handle horizontal mode combo change."""
        if self._current_page_id is None or self._updating:
            return

        h_modes = [
            HorizontalAlignment.HAUTO,
            HorizontalAlignment.HCENTER,
            HorizontalAlignment.HORIGINAL,
        ]
        params = self._filter.get_params(self._current_page_id)
        new_alignment = Alignment(
            vertical=params.alignment.vertical,
            horizontal=h_modes[index],
            is_null=params.alignment.is_null,
        )
        new_params = params.with_alignment(new_alignment)
        self._filter.set_params(self._current_page_id, new_params)
        self.alignment_changed.emit(self._current_page_id, new_alignment)

    def _on_v_mode_changed(self, index: int) -> None:
        """Handle vertical mode combo change."""
        if self._current_page_id is None or self._updating:
            return

        v_modes = [
            VerticalAlignment.VAUTO,
            VerticalAlignment.VCENTER,
            VerticalAlignment.VORIGINAL,
        ]
        params = self._filter.get_params(self._current_page_id)
        new_alignment = Alignment(
            vertical=v_modes[index],
            horizontal=params.alignment.horizontal,
            is_null=params.alignment.is_null,
        )
        new_params = params.with_alignment(new_alignment)
        self._filter.set_params(self._current_page_id, new_params)
        self.alignment_changed.emit(self._current_page_id, new_alignment)

    def _on_alignment_button_clicked(self, button: QToolButton) -> None:
        """Handle alignment button click."""
        if self._current_page_id is None or self._updating:
            return

        # Parse button name to get alignment
        name = button.objectName()
        v_align = VerticalAlignment.VCENTER
        h_align = HorizontalAlignment.HCENTER

        if "Top" in name:
            v_align = VerticalAlignment.TOP
        elif "Bottom" in name:
            v_align = VerticalAlignment.BOTTOM

        if "Left" in name:
            h_align = HorizontalAlignment.LEFT
        elif "Right" in name:
            h_align = HorizontalAlignment.RIGHT

        params = self._filter.get_params(self._current_page_id)
        new_alignment = Alignment(
            vertical=v_align,
            horizontal=h_align,
            is_null=params.alignment.is_null,
        )
        new_params = params.with_alignment(new_alignment)
        self._filter.set_params(self._current_page_id, new_params)

        # Update mode combos to Manual since we manually selected
        self._updating = True
        self._h_mode_combo.setCurrentIndex(1)  # Manual
        self._v_mode_combo.setCurrentIndex(1)  # Manual
        self._updating = False

        self.alignment_changed.emit(self._current_page_id, new_alignment)
