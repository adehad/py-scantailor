"""Split Mode dialog for Page Split filter.

Allows users to select the split mode (auto/manual) and apply scope
(this page, all pages, selected pages, etc.) for page splitting.
"""

from enum import Enum, auto
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Signal

from scantailor.app.ui import load_ui_widget
from scantailor.app.ui.utils import get_cwidget
from scantailor.core.models import PageId
from scantailor.filters.page_split.layout_type import LayoutType

_UI_FOLDER = Path(__file__).parent


class ApplyScope(Enum):
    """Scope for applying the split mode change."""

    THIS_PAGE = auto()
    ALL_PAGES = auto()
    THIS_PAGE_AND_FOLLOWING = auto()
    THIS_PAGE_EVERY_OTHER = auto()
    EVERY_OTHER = auto()
    SELECTED_PAGES = auto()
    EVERY_OTHER_SELECTED = auto()


class SplitModeDialog(QtWidgets.QDialog):
    """Dialog for selecting page split mode and apply scope.

    Allows the user to choose between automatic and manual split detection,
    optionally apply cut, and select which pages to apply the settings to.

    Signals:
        settings_accepted: Emitted when OK is clicked with selected settings.
            Args: (page_ids: set[PageId], layout_type: LayoutType, apply_cut: bool)
    """

    settings_accepted = Signal(set, object, bool)  # page_ids, layout_type, apply_cut

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        current_page: PageId | None = None,
        all_pages: list[PageId] | None = None,
        selected_pages: set[PageId] | None = None,
        layout_type: LayoutType | None = None,
        auto_detected_layout_type: LayoutType | None = None,
    ) -> None:
        """Initialize the Split Mode dialog.

        Args:
            parent: Parent widget.
            current_page: The current page being edited.
            all_pages: List of all pages in the project.
            selected_pages: Currently selected pages in the thumbnail list.
            layout_type: Current layout type setting.
            auto_detected_layout_type: Auto-detected layout type for current page.
        """
        super().__init__(parent)

        self._current_page = current_page
        self._all_pages = all_pages or []
        self._selected_pages = selected_pages or set()
        self._layout_type = layout_type
        self._auto_detected_layout_type = auto_detected_layout_type

        self._load_ui()
        self._connect_signals()
        self._update_ui_state()

    def _load_ui(self) -> None:
        """Load the UI from the .ui file."""
        self._ui = load_ui_widget(_UI_FOLDER / "SplitModeDialog.ui", self)

        # Set up the dialog layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._ui)

        self.setWindowTitle("Split Pages")
        self.setModal(True)

        # Get references to widgets
        self._layout_type_label = get_cwidget(
            self._ui, QtWidgets.QLabel, "layoutTypeLabel"
        )

        # Mode radio buttons
        self._mode_auto = get_cwidget(self._ui, QtWidgets.QRadioButton, "modeAuto")
        self._mode_manual = get_cwidget(self._ui, QtWidgets.QRadioButton, "modeManual")

        # Options
        self._apply_cut_option = get_cwidget(
            self._ui, QtWidgets.QCheckBox, "applyCutOption"
        )

        # Scope radio buttons
        self._this_page_rb = get_cwidget(self._ui, QtWidgets.QRadioButton, "thisPageRB")
        self._all_pages_rb = get_cwidget(self._ui, QtWidgets.QRadioButton, "allPagesRB")
        self._this_page_and_followers_rb = get_cwidget(
            self._ui, QtWidgets.QRadioButton, "thisPageAndFollowersRB"
        )
        self._this_every_other_rb = get_cwidget(
            self._ui, QtWidgets.QRadioButton, "thisEveryOtherRB"
        )
        self._every_other_rb = get_cwidget(
            self._ui, QtWidgets.QRadioButton, "everyOtherRB"
        )
        self._selected_pages_rb = get_cwidget(
            self._ui, QtWidgets.QRadioButton, "selectedPagesRB"
        )
        self._every_other_selected_rb = get_cwidget(
            self._ui, QtWidgets.QRadioButton, "everyOtherSelectedRB"
        )

        # Hints
        self._selected_pages_hint = get_cwidget(
            self._ui, QtWidgets.QLabel, "selectedPagesHint"
        )
        self._every_other_selected_hint = get_cwidget(
            self._ui, QtWidgets.QLabel, "everyOtherSelectedHint"
        )

        # Button box
        self._button_box = get_cwidget(
            self._ui, QtWidgets.QDialogButtonBox, "buttonBox"
        )

    def _connect_signals(self) -> None:
        """Connect widget signals to handlers."""
        # Mode selection
        self._mode_auto.toggled.connect(self._on_mode_changed)
        self._mode_manual.toggled.connect(self._on_mode_changed)

        # Button box
        self._button_box.accepted.connect(self._on_submit)
        self._button_box.rejected.connect(self.reject)

    def _update_ui_state(self) -> None:
        """Update UI based on current state."""
        # Disable selected pages options if no pages are selected
        has_selection = len(self._selected_pages) > 0
        self._selected_pages_rb.setEnabled(has_selection)
        self._every_other_selected_rb.setEnabled(has_selection)
        self._selected_pages_hint.setEnabled(has_selection)
        self._every_other_selected_hint.setEnabled(has_selection)

        # Update mode selection based on current layout type
        if self._layout_type is not None:
            # Import here to avoid circular imports
            from scantailor.filters.page_split.layout_type import LayoutType

            if self._layout_type == LayoutType.AUTO_LAYOUT_TYPE:
                self._mode_auto.setChecked(True)
            else:
                self._mode_manual.setChecked(True)

        self._update_layout_icon()

    def _on_mode_changed(self) -> None:
        """Handle mode selection change."""
        self._update_layout_icon()

    def _update_layout_icon(self) -> None:
        """Update the layout type icon based on selection."""
        # In a full implementation, this would update the icon based on
        # the selected mode and auto-detected layout type
        pass

    def _get_selected_scope(self) -> ApplyScope:
        """Get the selected apply scope.

        Returns:
            The selected ApplyScope.
        """
        if self._this_page_rb.isChecked():
            return ApplyScope.THIS_PAGE
        elif self._all_pages_rb.isChecked():
            return ApplyScope.ALL_PAGES
        elif self._this_page_and_followers_rb.isChecked():
            return ApplyScope.THIS_PAGE_AND_FOLLOWING
        elif self._this_every_other_rb.isChecked():
            return ApplyScope.THIS_PAGE_EVERY_OTHER
        elif self._every_other_rb.isChecked():
            return ApplyScope.EVERY_OTHER
        elif self._selected_pages_rb.isChecked():
            return ApplyScope.SELECTED_PAGES
        elif self._every_other_selected_rb.isChecked():
            return ApplyScope.EVERY_OTHER_SELECTED
        else:
            return ApplyScope.THIS_PAGE

    def _get_affected_pages(self) -> set[PageId]:
        """Get the set of pages affected by the current scope selection.

        Returns:
            Set of PageId objects that will be affected.
        """
        if not self._current_page or not self._all_pages:
            return set()

        scope = self._get_selected_scope()

        if scope == ApplyScope.THIS_PAGE:
            return {self._current_page}

        elif scope == ApplyScope.ALL_PAGES:
            return set(self._all_pages)

        elif scope == ApplyScope.THIS_PAGE_AND_FOLLOWING:
            try:
                idx = self._all_pages.index(self._current_page)
                return set(self._all_pages[idx:])
            except ValueError:
                return {self._current_page}

        elif scope == ApplyScope.THIS_PAGE_EVERY_OTHER:
            try:
                idx = self._all_pages.index(self._current_page)
                return {self._all_pages[i] for i in range(idx, len(self._all_pages), 2)}
            except ValueError:
                return {self._current_page}

        elif scope == ApplyScope.EVERY_OTHER:
            return {self._all_pages[i] for i in range(0, len(self._all_pages), 2)}

        elif scope == ApplyScope.SELECTED_PAGES:
            return self._selected_pages.copy()

        elif scope == ApplyScope.EVERY_OTHER_SELECTED:
            # Get selected pages in order, take every other one
            selected_in_order = [
                p for p in self._all_pages if p in self._selected_pages
            ]
            # Include current page and every other from there
            result = set()
            include_next = True
            for page in selected_in_order:
                if page == self._current_page:
                    include_next = True
                if include_next:
                    result.add(page)
                include_next = not include_next
            return result

        return {self._current_page}

    def _get_combined_layout_type(self) -> LayoutType:
        """Get the combined layout type based on mode selection.

        Returns:
            The selected LayoutType.
        """
        from scantailor.filters.page_split.layout_type import LayoutType

        if self._mode_auto.isChecked():
            return LayoutType.AUTO_LAYOUT_TYPE
        else:
            # Manual mode - return the current or auto-detected type
            if self._layout_type and self._layout_type != LayoutType.AUTO_LAYOUT_TYPE:
                return self._layout_type
            elif self._auto_detected_layout_type:
                return self._auto_detected_layout_type
            else:
                return LayoutType.SINGLE_PAGE_UNCUT

    def _on_submit(self) -> None:
        """Handle OK button click."""
        pages = self._get_affected_pages()
        layout_type = self._get_combined_layout_type()
        apply_cut = self._apply_cut_option.isChecked()

        self.settings_accepted.emit(pages, layout_type, apply_cut)
        self.accept()

    @property
    def apply_cut(self) -> bool:
        """Get whether the apply cut option is checked."""
        return self._apply_cut_option.isChecked()

    @property
    def is_auto_mode(self) -> bool:
        """Get whether auto mode is selected."""
        return self._mode_auto.isChecked()
