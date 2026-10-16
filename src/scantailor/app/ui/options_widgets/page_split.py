"""Options widget for Page Split filter.

Provides UI controls for selecting page layout type and split mode.
"""

from PySide6 import QtCore
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

from scantailor.app.ui.icons import icon
from scantailor.app.ui.widgets.collapsible_group_box import CollapsibleGroupBox
from scantailor.core.models import ImageId, PageId
from scantailor.filters.page_split.filter import Filter
from scantailor.filters.page_split.layout_type import LayoutType


class OptionsWidget(QWidget):
    """Options widget for the Page Split filter.

    Provides buttons to select the page layout type (single, page+offcut, two pages)
    and whether the split line should be auto-detected or manually placed.

    Until the page has a stored layout the Auto and Manual buttons are disabled
    and no layout type is shown for an automatic request. :meth:`refresh` shows
    the layout once processing has stored it.

    Signals:
        layout_type_changed: Emitted when layout type changes.
            Args: page_id (PageId), layout_type (LayoutType)
        split_mode_changed: Emitted when split mode (auto/manual) changes.
            Args: page_id (PageId), is_manual (bool)
        page_count_requested: Emitted when the chosen layout type fixes how many
            logical pages the image has.
            Args: image_id (ImageId), count (int)
        change_requested: Emitted when user clicks "Change..."
    """

    layout_type_changed = Signal(PageId, LayoutType)
    split_mode_changed = Signal(PageId, bool)
    page_count_requested = Signal(ImageId, int)
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
        # Pressing the button that is already checked changes nothing.
        self._shown_type: LayoutType | None = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._layout_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate(
                "page_split::OptionsWidget", "Page Layout"
            ),
            self,
        )
        self._layout_group.setObjectName("pageLayoutGroup")
        layout_group_layout = QVBoxLayout(self._layout_group)

        buttons_layout = QHBoxLayout()
        buttons_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

        self._layout_button_group = QButtonGroup(self)
        self._single_page_btn = self._layout_button(
            "singlePageUncutBtn", "split-single-uncut", 0
        )
        self._page_offcut_btn = self._layout_button(
            "pagePlusOffcutBtn", "split-right-page-plus-offcut", 1
        )
        self._two_pages_btn = self._layout_button("twoPagesBtn", "book-open", 2)
        self._single_page_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "page_split::OptionsWidget",
                "One page that fills the scan. Nothing is cut off.",
            )
        )
        self._page_offcut_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "page_split::OptionsWidget",
                "One page with unwanted material beside it, such as the edge of "
                "the facing page. A cut line removes the extra part.",
            )
        )
        self._two_pages_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "page_split::OptionsWidget",
                "An open book with two pages on one scan. The scan is split at "
                "the line between them and each half becomes its own page.",
            )
        )
        for button in self._layout_button_group.buttons():
            buttons_layout.addWidget(button)

        buttons_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        layout_group_layout.addLayout(buttons_layout)

        self._scope_label = QLabel("?")
        self._scope_label.setObjectName("scopeLabel")
        layout_group_layout.addWidget(self._scope_label)

        change_layout = QHBoxLayout()
        change_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        self._change_btn = QPushButton(
            QtCore.QCoreApplication.translate("page_split::OptionsWidget", "Change ...")
        )
        self._change_btn.setObjectName("changeBtn")
        self._change_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "page_split::OptionsWidget",
                "Choose whether the layout is detected automatically or fixed by "
                "you, and which pages the choice applies to.",
            )
        )
        change_layout.addWidget(self._change_btn)
        change_layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )
        layout_group_layout.addLayout(change_layout)

        layout.addWidget(self._layout_group)

        self._split_group = CollapsibleGroupBox(
            QtCore.QCoreApplication.translate(
                "page_split::OptionsWidget", "Split Line"
            ),
            self,
        )
        self._split_group.setObjectName("splitLineGroup")
        split_group_layout = QVBoxLayout(self._split_group)

        split_mode_layout = QHBoxLayout()
        self._mode_button_group = QButtonGroup(self)

        self._auto_btn = QPushButton(
            QtCore.QCoreApplication.translate("page_split::OptionsWidget", "Auto")
        )
        self._auto_btn.setObjectName("autoBtn")
        self._auto_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "page_split::OptionsWidget",
                "Let the program find the gap between the pages and place the "
                "split line there each time the page is processed.",
            )
        )
        self._auto_btn.setCheckable(True)
        self._mode_button_group.addButton(self._auto_btn)
        split_mode_layout.addWidget(self._auto_btn)

        self._manual_btn = QPushButton(
            QtCore.QCoreApplication.translate("page_split::OptionsWidget", "Manual")
        )
        self._manual_btn.setObjectName("manualBtn")
        self._manual_btn.setToolTip(
            QtCore.QCoreApplication.translate(
                "page_split::OptionsWidget",
                "Keep the split line where you dragged it on the image. Pick this "
                "when detection puts the line in the wrong place.",
            )
        )
        self._manual_btn.setCheckable(True)
        self._mode_button_group.addButton(self._manual_btn)
        split_mode_layout.addWidget(self._manual_btn)

        split_group_layout.addLayout(split_mode_layout)
        layout.addWidget(self._split_group)

        layout.addItem(
            QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

    def _layout_button(self, name: str, glyph: str, button_id: int) -> QToolButton:
        button = QToolButton()
        button.setObjectName(name)
        button.setCheckable(True)
        button.setIcon(icon(glyph))
        button.setMinimumSize(40, 40)
        self._layout_button_group.addButton(button, button_id)
        return button

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
        self._show_requested_type()
        self.refresh()

    def refresh(self) -> None:
        """Show the page's stored layout, as it is after processing."""
        if self._current_page_id is None:
            return
        params = self._filter.get_params(self._current_page_id)
        layout = params.page_layout
        if layout is None:
            return

        self._change_btn.setEnabled(True)
        self._auto_btn.setEnabled(True)
        self._manual_btn.setEnabled(True)
        (self._manual_btn if params.is_manual() else self._auto_btn).setChecked(True)

        layout_type = layout.to_layout_type()
        self._show_layout_type(layout_type)
        self._split_group.setVisible(layout_type != LayoutType.SINGLE_PAGE_UNCUT)
        if params.layout_type == LayoutType.AUTO_LAYOUT_TYPE:
            self._scope_label.setText(
                QtCore.QCoreApplication.translate(
                    "page_split::OptionsWidget", "Auto detected"
                )
            )

    def _show_requested_type(self) -> None:
        """Show the requested layout type while nothing is detected yet."""
        if self._current_page_id is None:
            return
        requested = self._filter.get_layout_type(self._current_page_id)
        self._show_layout_type(
            None if requested == LayoutType.AUTO_LAYOUT_TYPE else requested
        )
        self._split_group.setVisible(requested != LayoutType.SINGLE_PAGE_UNCUT)
        if requested == LayoutType.AUTO_LAYOUT_TYPE:
            self._change_btn.setEnabled(False)
            self._scope_label.setText("?")
        else:
            self._change_btn.setEnabled(True)
            self._scope_label.setText(
                QtCore.QCoreApplication.translate(
                    "page_split::OptionsWidget", "Set manually"
                )
            )
        self._uncheck(self._mode_button_group)
        self._auto_btn.setEnabled(False)
        self._manual_btn.setEnabled(False)

    def _show_layout_type(self, layout_type: LayoutType | None) -> None:
        """Check the button of a layout type; None checks none."""
        self._shown_type = layout_type
        buttons = {
            LayoutType.SINGLE_PAGE_UNCUT: self._single_page_btn,
            LayoutType.PAGE_PLUS_OFFCUT: self._page_offcut_btn,
            LayoutType.TWO_PAGES: self._two_pages_btn,
        }
        if layout_type is None:
            self._uncheck(self._layout_button_group)
        else:
            buttons[layout_type].setChecked(True)

    @staticmethod
    def _uncheck(group: QButtonGroup) -> None:
        """Uncheck every button of an exclusive group."""
        group.setExclusive(False)
        for button in group.buttons():
            button.setChecked(False)
        group.setExclusive(True)

    def _on_layout_type_changed(self, layout_type: LayoutType) -> None:
        """Handle layout type button clicks.

        A single uncut page, or two pages with a split line placed by hand, are
        switched in place so the line survives. Every other change makes the
        next run detect a layout of the new type.

        Args:
            layout_type: The selected layout type.
        """
        page = self._current_page_id
        if page is None or layout_type == self._shown_type:
            return
        self._shown_type = layout_type

        self._split_group.setVisible(layout_type != LayoutType.SINGLE_PAGE_UNCUT)
        self._scope_label.setText(
            QtCore.QCoreApplication.translate(
                "page_split::OptionsWidget", "Set manually"
            )
        )

        params = self._filter.get_params(page)
        detects_again = layout_type == LayoutType.PAGE_PLUS_OFFCUT or (
            layout_type != LayoutType.SINGLE_PAGE_UNCUT and params.is_auto()
        )
        if detects_again:
            self._filter.set_layout_type(page, layout_type)
        else:
            self._filter.retype_layout(page, layout_type)

        count = layout_type.sub_page_count()
        if count is not None:
            self.page_count_requested.emit(page.image_id, count)
        self.layout_type_changed.emit(page, layout_type)

    def _on_split_mode_changed(self, is_manual: bool) -> None:
        """Handle split mode button clicks.

        Args:
            is_manual: True for manual mode, False for auto.
        """
        if self._current_page_id is None:
            return

        params = self._filter.get_params(self._current_page_id)
        if params.is_manual() == is_manual:
            return
        if is_manual:
            self._filter.set_params(
                self._current_page_id, params.with_manual_mode(True)
            )
        else:
            self._filter.clear_detection(self._current_page_id)
        self.split_mode_changed.emit(self._current_page_id, is_manual)
