"""Options widget for Select Content filter.

Provides UI controls for configuring content and page box detection.
"""

from PySide6 import QtCore
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.app.ui.widgets.collapsible_group_box import CollapsibleGroupBox
from scantailor.core.models import PageId
from scantailor.core.settings import Units
from scantailor.core.units import convert_length, spin_box_precision
from scantailor.filters.select_content.content_box import PageBox
from scantailor.filters.select_content.detection import (
    ContentDetectionMode,
    PageDetectionMode,
)
from scantailor.filters.select_content.filter import Filter


class OptionsWidget(QWidget):
    """Options widget for the Select Content filter.

    Provides controls for:
    - Page box detection mode (disabled/auto/manual) with optional fine-tuning
    - Size of a manual page box (width/height) in the current units
    - Content box detection mode (disabled/auto/manual)
    - Apply to multiple pages button

    Signals:
        page_mode_changed: Emitted when page detection mode changes.
            Args: page_id (PageId), mode (PageDetectionMode)
        content_mode_changed: Emitted when content detection mode changes.
            Args: page_id (PageId), mode (ContentDetectionMode)
        fine_tune_changed: Emitted when fine-tune corners checkbox changes.
            Args: page_id (PageId), enabled (bool)
        page_box_changed: Emitted when the width or height fields resized the
            page box. Args: page_id (PageId)
        apply_to_requested: Emitted when user clicks "Apply to..."
    """

    page_mode_changed = Signal(PageId, PageDetectionMode)
    content_mode_changed = Signal(PageId, ContentDetectionMode)
    fine_tune_changed = Signal(PageId, bool)
    page_box_changed = Signal(PageId)
    apply_to_requested = Signal()

    def __init__(
        self,
        filter_: Filter,
        parent: QWidget | None = None,
        units_provider: UnitsProvider | None = None,
    ) -> None:
        """Initialize the options widget.

        Args:
            filter_ (Filter): The Select Content filter instance.
            parent (QWidget | None): The parent widget.
            units_provider (UnitsProvider | None): Source of the unit the page
                size is shown in; the shared provider when None.
        """
        super().__init__(parent)

        self._filter = filter_
        self._units_provider = units_provider or UnitsProvider.get_instance()
        self._current_page_id: PageId | None = None
        self._dpi = (0.0, 0.0)
        self._updating = False  # Prevent signal loops

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Outer groupbox (no title, just groups content)
        outer_group = QGroupBox(self)
        outer_group.setProperty("untitled", True)
        outer_layout = QVBoxLayout(outer_group)

        # Page Box group
        self._page_box_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget", "Page Box"
            ),
            outer_group,
        )
        self._page_box_group.setObjectName("pageBoxGroup")
        page_layout = QGridLayout(self._page_box_group)

        # Disable button (row 0, spans both columns)
        self._page_disable_btn = QPushButton(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget", "Disable"
            )
        )
        self._page_disable_btn.setObjectName("pageDetectDisableBtn")
        self._page_disable_btn.setCheckable(True)
        self._page_disable_btn.setChecked(True)
        self._page_disable_btn.setAutoExclusive(True)
        self._page_disable_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget",
                "Use the whole image as the page. Pick this when the scan is "
                "already cropped to the edges of the paper.",
            )
        )
        page_layout.addWidget(self._page_disable_btn, 0, 0, 1, 2)

        # Auto/Manual buttons (row 1)
        self._page_auto_btn = QPushButton(
            QtCore.QCoreApplication.translate("select_content::OptionsWidget", "Auto")
        )
        self._page_auto_btn.setObjectName("pageDetectAutoBtn")
        self._page_auto_btn.setCheckable(True)
        self._page_auto_btn.setAutoExclusive(True)
        self._page_auto_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget",
                "Find the edges of the paper from long straight lines near the "
                "image border. Use it when the scanner lid or background shows "
                "around the page. If no edges are found, the whole image is used.",
            )
        )
        page_layout.addWidget(self._page_auto_btn, 1, 0)

        self._page_manual_btn = QPushButton(
            QtCore.QCoreApplication.translate("select_content::OptionsWidget", "Manual")
        )
        self._page_manual_btn.setObjectName("pageDetectManualBtn")
        self._page_manual_btn.setCheckable(True)
        self._page_manual_btn.setAutoExclusive(True)
        self._page_manual_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget",
                "Keep the page box you draw on the image. Pick this when "
                "automatic detection misses the edges of the paper.",
            )
        )
        page_layout.addWidget(self._page_manual_btn, 1, 1)

        # Options subgroup (row 2)
        self._page_options_group = QGroupBox(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget", "Options"
            ),
            self._page_box_group,
        )
        self._page_options_group.setObjectName("pageDetectOptions")
        options_layout = QVBoxLayout(self._page_options_group)

        # Fine tune checkbox
        self._fine_tune_check = QCheckBox(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget", "Fine Tune Page Corners"
            )
        )
        self._fine_tune_check.setObjectName("fineTuneBtn")
        self._fine_tune_check.setToolTip(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget",
                "Move the detected page corners inward while they sit on dark "
                "scanner background. This helps when the paper lies slightly "
                "crooked on the scanner glass.",
            )
        )
        options_layout.addWidget(self._fine_tune_check)

        self._dimensions_widget = QWidget(self._page_options_group)
        self._dimensions_widget.setObjectName("dimensionsWidget")
        dimensions_layout = QHBoxLayout(self._dimensions_widget)
        dimensions_layout.setContentsMargins(0, 0, 0, 0)
        dimensions_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        dims_grid = QGridLayout()
        dims_grid.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate(
                    "select_content::OptionsWidget", "Width"
                )
            ),
            0,
            0,
        )
        dims_grid.addWidget(
            QLabel(
                QtCore.QCoreApplication.translate(
                    "select_content::OptionsWidget", "Height"
                )
            ),
            1,
            0,
        )

        self._width_spinbox = QDoubleSpinBox()
        self._width_spinbox.setObjectName("widthSpinBox")
        self._width_spinbox.setRange(0.1, 99999.0)
        self._width_spinbox.setKeyboardTracking(False)
        self._width_spinbox.setToolTip(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget",
                "Width of the page box in the units chosen in the "
                "settings. Type a value to resize the box from its top left "
                "corner, or drag its edges in the page view.",
            )
        )
        dims_grid.addWidget(self._width_spinbox, 0, 1)

        self._height_spinbox = QDoubleSpinBox()
        self._height_spinbox.setObjectName("heightSpinBox")
        self._height_spinbox.setRange(0.1, 99999.0)
        self._height_spinbox.setKeyboardTracking(False)
        self._height_spinbox.setToolTip(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget",
                "Height of the page box in the units chosen in the "
                "settings. Type a value to resize the box from its top left "
                "corner, or drag its edges in the page view.",
            )
        )
        dims_grid.addWidget(self._height_spinbox, 1, 1)

        dimensions_layout.addLayout(dims_grid)
        dimensions_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        options_layout.addWidget(self._dimensions_widget)

        page_layout.addWidget(self._page_options_group, 2, 0, 1, 2)
        outer_layout.addWidget(self._page_box_group)

        # Content Box group
        self._content_box_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget", "Content Box"
            ),
            outer_group,
        )
        self._content_box_group.setObjectName("contentBoxGroup")
        content_layout = QGridLayout(self._content_box_group)

        # Disable button (row 0, spans both columns)
        self._content_disable_btn = QPushButton(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget", "Disable"
            )
        )
        self._content_disable_btn.setObjectName("contentDetectDisableBtn")
        self._content_disable_btn.setCheckable(True)
        self._content_disable_btn.setAutoExclusive(True)
        self._content_disable_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget",
                "Treat the whole page as content, so nothing is trimmed. Pick "
                "this for covers, photos or pages printed to the edge.",
            )
        )
        content_layout.addWidget(self._content_disable_btn, 0, 0, 1, 2)

        # Auto/Manual buttons (row 1)
        self._content_auto_btn = QPushButton(
            QtCore.QCoreApplication.translate("select_content::OptionsWidget", "Auto")
        )
        self._content_auto_btn.setObjectName("contentDetectAutoBtn")
        self._content_auto_btn.setCheckable(True)
        self._content_auto_btn.setChecked(True)
        self._content_auto_btn.setAutoExclusive(True)
        self._content_auto_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget",
                "Draw the box around the text and pictures found on the page, "
                "leaving out dark shadows along the scan edges. Check the result, "
                "since stray marks or page numbers can widen the box.",
            )
        )
        content_layout.addWidget(self._content_auto_btn, 1, 0)

        self._content_manual_btn = QPushButton(
            QtCore.QCoreApplication.translate("select_content::OptionsWidget", "Manual")
        )
        self._content_manual_btn.setObjectName("contentDetectManualBtn")
        self._content_manual_btn.setCheckable(True)
        self._content_manual_btn.setAutoExclusive(True)
        self._content_manual_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget",
                "Keep the content box you draw on the image. Drag its edges in "
                "the page view; batch processing leaves it where you put it.",
            )
        )
        content_layout.addWidget(self._content_manual_btn, 1, 1)

        outer_layout.addWidget(self._content_box_group)

        # Apply to button
        apply_layout = QHBoxLayout()
        apply_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._apply_btn = QPushButton(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget", "Apply to ..."
            )
        )
        self._apply_btn.setObjectName("applyToBtn")
        self._apply_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "select_content::OptionsWidget",
                "Copy the page box and content box settings of this page to "
                "other pages you choose.",
            )
        )
        apply_layout.addWidget(self._apply_btn)

        apply_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        outer_layout.addLayout(apply_layout)

        layout.addWidget(outer_group)

        # Vertical spacer
        layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        # Page detection mode buttons
        self._page_disable_btn.clicked.connect(
            lambda: self._on_page_mode_changed(PageDetectionMode.DISABLED)
        )
        self._page_auto_btn.clicked.connect(
            lambda: self._on_page_mode_changed(PageDetectionMode.AUTO)
        )
        self._page_manual_btn.clicked.connect(
            lambda: self._on_page_mode_changed(PageDetectionMode.MANUAL)
        )

        # Content detection mode buttons
        self._content_disable_btn.clicked.connect(
            lambda: self._on_content_mode_changed(ContentDetectionMode.DISABLED)
        )
        self._content_auto_btn.clicked.connect(
            lambda: self._on_content_mode_changed(ContentDetectionMode.AUTO)
        )
        self._content_manual_btn.clicked.connect(
            lambda: self._on_content_mode_changed(ContentDetectionMode.MANUAL)
        )

        # Fine tune checkbox
        self._fine_tune_check.toggled.connect(self._on_fine_tune_changed)

        self._width_spinbox.valueChanged.connect(self._on_dimensions_changed)
        self._height_spinbox.valueChanged.connect(self._on_dimensions_changed)
        self._units_provider.units_changed.connect(self._show_page_size)

        # Apply button
        self._apply_btn.clicked.connect(self.apply_to_requested.emit)

    def set_current_page(self, page_id: PageId) -> None:
        """Set the current page being edited.

        Args:
            page_id: The page ID to edit.
        """
        self._current_page_id = page_id
        self._update_ui()

    def _update_ui(self) -> None:
        """Update the UI to reflect current page settings."""
        if self._current_page_id is None:
            return

        self._updating = True
        try:
            params = self._filter.get_params(self._current_page_id)

            # Update page detection mode buttons
            self._page_disable_btn.setChecked(params.is_page_disabled())
            self._page_auto_btn.setChecked(params.is_page_auto())
            self._page_manual_btn.setChecked(params.is_page_manual())

            self._page_options_group.setVisible(not params.is_page_disabled())
            self._fine_tune_check.setVisible(params.is_page_auto())
            self._dimensions_widget.setVisible(params.is_page_manual())

            # Update fine tune checkbox
            self._fine_tune_check.setChecked(params.fine_tune_corners)

            # Update content detection mode buttons
            self._content_disable_btn.setChecked(params.is_content_disabled())
            self._content_auto_btn.setChecked(params.is_content_auto())
            self._content_manual_btn.setChecked(params.is_content_manual())
        finally:
            self._updating = False
        self._show_page_size()

    def set_page_dpi(self, horizontal: float, vertical: float) -> None:
        """Tell the widget the resolution of the current page.

        The page box is held in pixels, so every unit but pixels needs it; the
        size is shown unconverted while it is unknown.

        Args:
            horizontal (float): Dots per inch along x, or 0 when unknown.
            vertical (float): Dots per inch along y, or 0 when unknown.
        """
        self._dpi = (horizontal, vertical)
        self._show_page_size()

    def _show_page_size(self) -> None:
        """Show the current page box size in the current unit."""
        if self._current_page_id is None:
            return
        page_box = self._filter.get_params(self._current_page_id).page_box
        units = self._units_provider.units
        decimals, step = spin_box_precision(units)
        horizontal, vertical = self._dpi
        was_updating, self._updating = self._updating, True
        try:
            for spin, pixels, dpi in (
                (self._width_spinbox, page_box.width, horizontal),
                (self._height_spinbox, page_box.height, vertical),
            ):
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
                spin.setValue(convert_length(pixels, Units.PIXELS, units, dpi))
        finally:
            self._updating = was_updating

    def _on_dimensions_changed(self) -> None:
        """Resize the page box to the typed size, keeping its top left corner.

        As when the box is dragged, the page box still grows to hold the
        content box.
        """
        if self._current_page_id is None or self._updating:
            return
        params = self._filter.get_params(self._current_page_id)
        units = self._units_provider.units
        horizontal, vertical = self._dpi
        resized = PageBox(
            x=params.page_box.x,
            y=params.page_box.y,
            width=convert_length(
                self._width_spinbox.value(), units, Units.PIXELS, horizontal
            ),
            height=convert_length(
                self._height_spinbox.value(), units, Units.PIXELS, vertical
            ),
        )
        if params.content_box.is_valid():
            united = resized.to_content_box().union(params.content_box)
            resized = PageBox(
                x=united.x, y=united.y, width=united.width, height=united.height
            )
        self._filter.set_manual_page(self._current_page_id, resized)
        self._show_page_size()
        self.page_box_changed.emit(self._current_page_id)

    def _on_page_mode_changed(self, mode: PageDetectionMode) -> None:
        """Handle page detection mode button clicks.

        Args:
            mode: The selected page detection mode.
        """
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        if params.page_detection_mode != mode:
            if mode == PageDetectionMode.DISABLED:
                new_params = params.with_disabled_page()
            elif mode == PageDetectionMode.AUTO:
                new_params = params.with_auto_page()
            else:
                new_params = params.model_copy(
                    update={"page_detection_mode": PageDetectionMode.MANUAL}
                )
            self._filter.set_params(self._current_page_id, new_params)
            self._update_ui()
            self.page_mode_changed.emit(self._current_page_id, mode)

    def _on_content_mode_changed(self, mode: ContentDetectionMode) -> None:
        """Handle content detection mode button clicks.

        Args:
            mode: The selected content detection mode.
        """
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        if params.content_detection_mode != mode:
            if mode == ContentDetectionMode.DISABLED:
                new_params = params.with_disabled_content()
            elif mode == ContentDetectionMode.AUTO:
                new_params = params.with_auto_content()
            else:
                new_params = params.model_copy(
                    update={"content_detection_mode": ContentDetectionMode.MANUAL}
                )
            self._filter.set_params(self._current_page_id, new_params)
            self._update_ui()
            self.content_mode_changed.emit(self._current_page_id, mode)

    def _on_fine_tune_changed(self, checked: bool) -> None:
        """Handle fine tune checkbox toggle.

        Args:
            checked: Whether fine tuning is enabled.
        """
        if self._current_page_id is None or self._updating:
            return

        params = self._filter.get_params(self._current_page_id)
        if params.fine_tune_corners != checked:
            new_params = params.with_fine_tune(checked)
            self._filter.set_params(self._current_page_id, new_params)
            self.fine_tune_changed.emit(self._current_page_id, checked)
