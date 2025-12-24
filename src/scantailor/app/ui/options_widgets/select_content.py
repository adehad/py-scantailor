"""Options widget for Select Content filter.

Provides UI controls for configuring content and page box detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

from scantailor.app.ui.widgets import CollapsibleGroupBox
from scantailor.core import PageId
from scantailor.filters.select_content.detection import (
    ContentDetectionMode,
    PageDetectionMode,
)

if TYPE_CHECKING:
    from scantailor.filters.select_content.filter import Filter


class OptionsWidget(QWidget):
    """Options widget for the Select Content filter.

    Provides controls for:
    - Page box detection mode (disabled/auto/manual) with optional fine-tuning
    - Page dimensions (width/height)
    - Content box detection mode (disabled/auto/manual)
    - Apply to multiple pages button

    Signals:
        page_mode_changed: Emitted when page detection mode changes.
            Args: page_id (PageId), mode (PageDetectionMode)
        content_mode_changed: Emitted when content detection mode changes.
            Args: page_id (PageId), mode (ContentDetectionMode)
        fine_tune_changed: Emitted when fine-tune corners checkbox changes.
            Args: page_id (PageId), enabled (bool)
        apply_to_requested: Emitted when user clicks "Apply to..."
    """

    page_mode_changed = Signal(PageId, PageDetectionMode)
    content_mode_changed = Signal(PageId, ContentDetectionMode)
    fine_tune_changed = Signal(PageId, bool)
    apply_to_requested = Signal()

    def __init__(
        self,
        filter_: Filter,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the options widget.

        Args:
            filter_: The Select Content filter instance.
            parent: The parent widget.
        """
        super().__init__(parent)

        self._filter = filter_
        self._current_page_id: PageId | None = None
        self._updating = False  # Prevent signal loops

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Outer groupbox (no title, just groups content)
        outer_group = QGroupBox(self)
        outer_layout = QVBoxLayout(outer_group)

        # Page Box group
        self._page_box_group = CollapsibleGroupBox("Page Box", outer_group)
        self._page_box_group.setObjectName("pageBoxGroup")
        page_layout = QGridLayout(self._page_box_group)

        # Disable button (row 0, spans both columns)
        self._page_disable_btn = QPushButton("Disable")
        self._page_disable_btn.setObjectName("pageDetectDisableBtn")
        self._page_disable_btn.setCheckable(True)
        self._page_disable_btn.setChecked(True)
        self._page_disable_btn.setAutoExclusive(True)
        self._page_disable_btn.setToolTip("Use full image as page")
        page_layout.addWidget(self._page_disable_btn, 0, 0, 1, 2)

        # Auto/Manual buttons (row 1)
        self._page_auto_btn = QPushButton("Auto")
        self._page_auto_btn.setObjectName("pageDetectAutoBtn")
        self._page_auto_btn.setCheckable(True)
        self._page_auto_btn.setAutoExclusive(True)
        self._page_auto_btn.setToolTip("Automatically detect page boundaries")
        page_layout.addWidget(self._page_auto_btn, 1, 0)

        self._page_manual_btn = QPushButton("Manual")
        self._page_manual_btn.setObjectName("pageDetectManualBtn")
        self._page_manual_btn.setCheckable(True)
        self._page_manual_btn.setAutoExclusive(True)
        self._page_manual_btn.setToolTip("Manually specify page boundaries")
        page_layout.addWidget(self._page_manual_btn, 1, 1)

        # Options subgroup (row 2)
        self._page_options_group = QGroupBox("Options", self._page_box_group)
        self._page_options_group.setObjectName("pageDetectOptions")
        options_layout = QVBoxLayout(self._page_options_group)

        # Fine tune checkbox
        self._fine_tune_check = QCheckBox("Fine Tune Page Corners")
        self._fine_tune_check.setObjectName("fineTuneBtn")
        self._fine_tune_check.setToolTip(
            "Shift with corners while they are in black."
        )
        options_layout.addWidget(self._fine_tune_check)

        # Dimensions widget
        dimensions_layout = QHBoxLayout()
        dimensions_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        dims_grid = QGridLayout()
        dims_grid.addWidget(QLabel("Width"), 0, 0)
        dims_grid.addWidget(QLabel("Height"), 1, 0)

        self._width_spinbox = QDoubleSpinBox()
        self._width_spinbox.setObjectName("widthSpinBox")
        self._width_spinbox.setDecimals(1)
        self._width_spinbox.setRange(0.1, 99999.0)
        dims_grid.addWidget(self._width_spinbox, 0, 1)

        self._height_spinbox = QDoubleSpinBox()
        self._height_spinbox.setObjectName("heightSpinBox")
        self._height_spinbox.setDecimals(1)
        self._height_spinbox.setRange(0.1, 99999.0)
        dims_grid.addWidget(self._height_spinbox, 1, 1)

        dimensions_layout.addLayout(dims_grid)
        dimensions_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        options_layout.addLayout(dimensions_layout)

        page_layout.addWidget(self._page_options_group, 2, 0, 1, 2)
        outer_layout.addWidget(self._page_box_group)

        # Content Box group
        self._content_box_group = CollapsibleGroupBox("Content Box", outer_group)
        self._content_box_group.setObjectName("contentBoxGroup")
        content_layout = QGridLayout(self._content_box_group)

        # Disable button (row 0, spans both columns)
        self._content_disable_btn = QPushButton("Disable")
        self._content_disable_btn.setObjectName("contentDetectDisableBtn")
        self._content_disable_btn.setCheckable(True)
        self._content_disable_btn.setAutoExclusive(True)
        self._content_disable_btn.setToolTip("Treat entire page as content")
        content_layout.addWidget(self._content_disable_btn, 0, 0, 1, 2)

        # Auto/Manual buttons (row 1)
        self._content_auto_btn = QPushButton("Auto")
        self._content_auto_btn.setObjectName("contentDetectAutoBtn")
        self._content_auto_btn.setCheckable(True)
        self._content_auto_btn.setChecked(True)
        self._content_auto_btn.setAutoExclusive(True)
        self._content_auto_btn.setToolTip("Automatically detect content boundaries")
        content_layout.addWidget(self._content_auto_btn, 1, 0)

        self._content_manual_btn = QPushButton("Manual")
        self._content_manual_btn.setObjectName("contentDetectManualBtn")
        self._content_manual_btn.setCheckable(True)
        self._content_manual_btn.setAutoExclusive(True)
        self._content_manual_btn.setToolTip("Manually specify content boundaries")
        content_layout.addWidget(self._content_manual_btn, 1, 1)

        outer_layout.addWidget(self._content_box_group)

        # Apply to button
        apply_layout = QHBoxLayout()
        apply_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._apply_btn = QPushButton("Apply to ...")
        self._apply_btn.setObjectName("applyToBtn")
        self._apply_btn.setToolTip("Apply settings to other pages")
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

            # Show/hide page options based on mode
            show_page_options = not params.is_page_disabled()
            self._page_options_group.setVisible(show_page_options)

            # Update fine tune checkbox
            self._fine_tune_check.setChecked(params.fine_tune_corners)

            # Update content detection mode buttons
            self._content_disable_btn.setChecked(params.is_content_disabled())
            self._content_auto_btn.setChecked(params.is_content_auto())
            self._content_manual_btn.setChecked(params.is_content_manual())

            # Update dimension spinboxes if we have size info
            if params.content_size_mm.width > 0:
                self._width_spinbox.setValue(params.content_size_mm.width)
            if params.content_size_mm.height > 0:
                self._height_spinbox.setValue(params.content_size_mm.height)
        finally:
            self._updating = False

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
