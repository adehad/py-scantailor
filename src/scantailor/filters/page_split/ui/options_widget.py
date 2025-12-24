"""Options widget for Page Split filter.

Provides UI controls for selecting page layout type and split mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from scantailor.app.widgets import CollapsibleGroupBox
from scantailor.core import PageId
from scantailor.filters.page_split.layout_type import LayoutType

if TYPE_CHECKING:
    from scantailor.filters.page_split.filter import Filter


class OptionsWidget(QWidget):
    """Options widget for the Page Split filter.

    Provides buttons to select the page layout type (single, page+offcut, two pages)
    and whether the split line should be auto-detected or manually placed.

    Signals:
        layout_type_changed: Emitted when layout type changes.
            Args: page_id (PageId), layout_type (LayoutType)
        split_mode_changed: Emitted when split mode (auto/manual) changes.
            Args: page_id (PageId), is_manual (bool)
        change_requested: Emitted when user clicks "Change..."
    """

    layout_type_changed = Signal(PageId, LayoutType)
    split_mode_changed = Signal(PageId, bool)
    change_requested = Signal()

    def __init__(
        self,
        filter_: Filter,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the options widget.

        Args:
            filter_: The Page Split filter instance.
            parent: The parent widget.
        """
        super().__init__(parent)

        self._filter = filter_
        self._current_page_id: PageId | None = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Page Layout group box
        self._layout_group = CollapsibleGroupBox("Page Layout", self)
        self._layout_group.setObjectName("pageLayoutGroup")
        layout_group_layout = QVBoxLayout(self._layout_group)

        # Layout type buttons row
        buttons_layout = QHBoxLayout()
        buttons_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        # Create button group for exclusive selection
        self._layout_button_group = QButtonGroup(self)

        self._single_page_btn = QToolButton()
        self._single_page_btn.setObjectName("singlePageUncutBtn")
        self._single_page_btn.setCheckable(True)
        self._single_page_btn.setAutoExclusive(True)
        self._single_page_btn.setText("☐")  # Single page icon placeholder
        self._single_page_btn.setToolTip("Single page (no cutting)")
        self._single_page_btn.setMinimumSize(40, 40)
        self._layout_button_group.addButton(self._single_page_btn, 0)
        buttons_layout.addWidget(self._single_page_btn)

        self._page_offcut_btn = QToolButton()
        self._page_offcut_btn.setObjectName("pagePlusOffcutBtn")
        self._page_offcut_btn.setCheckable(True)
        self._page_offcut_btn.setAutoExclusive(True)
        self._page_offcut_btn.setText("☐|")  # Page + offcut icon placeholder
        self._page_offcut_btn.setToolTip("Page with offcut to remove")
        self._page_offcut_btn.setMinimumSize(40, 40)
        self._layout_button_group.addButton(self._page_offcut_btn, 1)
        buttons_layout.addWidget(self._page_offcut_btn)

        self._two_pages_btn = QToolButton()
        self._two_pages_btn.setObjectName("twoPagesBtn")
        self._two_pages_btn.setCheckable(True)
        self._two_pages_btn.setAutoExclusive(True)
        self._two_pages_btn.setText("☐|☐")  # Two pages icon placeholder
        self._two_pages_btn.setToolTip("Two pages (split in middle)")
        self._two_pages_btn.setMinimumSize(40, 40)
        self._layout_button_group.addButton(self._two_pages_btn, 2)
        buttons_layout.addWidget(self._two_pages_btn)

        buttons_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        layout_group_layout.addLayout(buttons_layout)

        # Scope label (shows current setting)
        self._scope_label = QLabel("?")
        self._scope_label.setObjectName("scopeLabel")
        self._scope_label.setAlignment(
            self._scope_label.alignment()
        )  # Keep default center
        layout_group_layout.addWidget(self._scope_label)

        # Change button row
        change_layout = QHBoxLayout()
        change_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._change_btn = QPushButton("Change ...")
        self._change_btn.setObjectName("changeBtn")
        self._change_btn.setToolTip("Change layout type for multiple pages")
        change_layout.addWidget(self._change_btn)

        change_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        layout_group_layout.addLayout(change_layout)

        layout.addWidget(self._layout_group)

        # Split Line group box
        self._split_group = CollapsibleGroupBox("Split Line", self)
        self._split_group.setObjectName("splitLineGroup")
        split_group_layout = QVBoxLayout(self._split_group)

        # Auto/Manual buttons
        split_mode_layout = QHBoxLayout()

        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setObjectName("autoBtn")
        self._auto_btn.setCheckable(True)
        self._auto_btn.setChecked(True)
        self._auto_btn.setAutoExclusive(True)
        self._auto_btn.setToolTip("Automatically detect split line")
        split_mode_layout.addWidget(self._auto_btn)

        self._manual_btn = QPushButton("Manual")
        self._manual_btn.setObjectName("manualBtn")
        self._manual_btn.setCheckable(True)
        self._manual_btn.setAutoExclusive(True)
        self._manual_btn.setToolTip("Manually position split line")
        split_mode_layout.addWidget(self._manual_btn)

        split_group_layout.addLayout(split_mode_layout)
        layout.addWidget(self._split_group)

        # Vertical spacer
        layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self._single_page_btn.clicked.connect(
            lambda: self._on_layout_type_changed(LayoutType.SINGLE_PAGE_UNCUT)
        )
        self._page_offcut_btn.clicked.connect(
            lambda: self._on_layout_type_changed(LayoutType.PAGE_PLUS_OFFCUT)
        )
        self._two_pages_btn.clicked.connect(
            lambda: self._on_layout_type_changed(LayoutType.TWO_PAGES)
        )
        self._change_btn.clicked.connect(self.change_requested.emit)
        self._auto_btn.clicked.connect(lambda: self._on_split_mode_changed(False))
        self._manual_btn.clicked.connect(lambda: self._on_split_mode_changed(True))

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
            self._scope_label.setText("?")
            return

        params = self._filter.get_params(self._current_page_id)
        layout_type = params.layout_type

        # Update layout type buttons
        if layout_type == LayoutType.SINGLE_PAGE_UNCUT:
            self._single_page_btn.setChecked(True)
            self._scope_label.setText("Single page")
        elif layout_type == LayoutType.PAGE_PLUS_OFFCUT:
            self._page_offcut_btn.setChecked(True)
            self._scope_label.setText("Page + offcut")
        elif layout_type == LayoutType.TWO_PAGES:
            self._two_pages_btn.setChecked(True)
            self._scope_label.setText("Two pages")
        else:  # AUTO
            self._scope_label.setText("Auto-detect")

        # Update split mode buttons
        is_manual = params.is_manual()
        self._auto_btn.setChecked(not is_manual)
        self._manual_btn.setChecked(is_manual)

        # Show/hide split line group based on layout type
        show_split = layout_type in (LayoutType.PAGE_PLUS_OFFCUT, LayoutType.TWO_PAGES)
        self._split_group.setVisible(show_split)

    def _on_layout_type_changed(self, layout_type: LayoutType) -> None:
        """Handle layout type button clicks.

        Args:
            layout_type: The selected layout type.
        """
        if self._current_page_id is None:
            return

        params = self._filter.get_params(self._current_page_id)
        if params.layout_type != layout_type:
            # Update params with new layout type
            new_params = params.with_layout_type(layout_type)
            self._filter.set_params(self._current_page_id, new_params)
            self._update_ui()
            self.layout_type_changed.emit(self._current_page_id, layout_type)

    def _on_split_mode_changed(self, is_manual: bool) -> None:
        """Handle split mode button clicks.

        Args:
            is_manual: True for manual mode, False for auto.
        """
        if self._current_page_id is None:
            return

        params = self._filter.get_params(self._current_page_id)
        if params.is_manual() != is_manual:
            new_params = params.with_manual_mode(is_manual)
            self._filter.set_params(self._current_page_id, new_params)
            self.split_mode_changed.emit(self._current_page_id, is_manual)
