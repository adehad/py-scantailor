"""Split Mode dialog for Page Split filter.

Allows users to select the split mode (auto/manual) and apply scope
(this page, all pages, selected pages, etc.) for page splitting.
"""

from collections.abc import Collection
from pathlib import Path

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal

from scantailor.app.ui import embed_ui, load_ui_widget
from scantailor.app.ui.icons import icon
from scantailor.app.ui.utils import get_cwidget
from scantailor.core.models import PageId
from scantailor.core.propagation import ApplyScope, select_pages
from scantailor.filters.page_split.layout_type import LayoutType

_UI_FOLDER = Path(__file__).parent
_LAYOUT_GLYPHS = {
    LayoutType.SINGLE_PAGE_UNCUT: "split-single-uncut",
    LayoutType.PAGE_PLUS_OFFCUT: "split-right-page-plus-offcut",
    LayoutType.TWO_PAGES: "book-open",
}


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
        selected_pages: Collection[PageId] | None = None,
        layout_type: LayoutType | None = None,
        auto_detected_layout_type: LayoutType | None = None,
    ) -> None:
        """Initialize the Split Mode dialog.

        Args:
            parent: Parent widget.
            current_page: The current page being edited.
            all_pages: List of all pages in the project.
            selected_pages: Currently selected pages in the thumbnail list;
                kept in the order of ``all_pages``.
            layout_type: Current layout type setting.
            auto_detected_layout_type: Auto-detected layout type for current page.
        """
        super().__init__(parent)

        self._current_page = current_page
        self._all_pages = all_pages or []
        selected = set(selected_pages or ())
        self._selected_pages = [page for page in self._all_pages if page in selected]
        self._layout_type = layout_type
        self._auto_detected_layout_type = auto_detected_layout_type

        self._load_ui()
        self._connect_signals()
        self._update_ui_state()

    def _load_ui(self) -> None:
        """Load the UI from the .ui file."""
        self._ui = load_ui_widget(_UI_FOLDER / "SplitModeDialog.ui", self)

        embed_ui(self, self._ui)
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
        has_selection = len(self._selected_pages) > 1
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

        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        """Handle mode selection change."""
        manual = self._mode_manual.isChecked()
        self._apply_cut_option.setEnabled(manual)
        if not manual:
            self._apply_cut_option.setChecked(False)
        self._update_layout_icon()

    def _update_layout_icon(self) -> None:
        """Update the layout type icon based on selection."""
        glyph = _LAYOUT_GLYPHS.get(self._get_combined_layout_type(), "wand-sparkles")
        size = QtCore.QSize(48, 48)
        self._layout_type_label.setPixmap(
            icon(glyph).pixmap(size, self._layout_type_label.devicePixelRatioF())
        )

    def _get_selected_scope(self) -> ApplyScope:
        """Get the selected apply scope.

        "Every other selected page" starts from the current page when it is
        selected, as the dialog's hint says. The C++ application starts from the
        first selected page instead.

        Returns:
            The selected ApplyScope.
        """
        if self._this_page_rb.isChecked():
            return ApplyScope.THIS_PAGE
        elif self._all_pages_rb.isChecked():
            return ApplyScope.ALL_PAGES
        elif self._this_page_and_followers_rb.isChecked():
            return ApplyScope.THIS_PAGE_AND_FOLLOWERS
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

        Scope resolution is ``select_pages``, including its deliberate
        deviation from C++ for "every other selected page".

        Returns:
            Set of PageId objects that will be affected.
        """
        if not self._current_page or not self._all_pages:
            return set()
        return set(
            select_pages(
                self._get_selected_scope(),
                self._current_page,
                self._all_pages,
                self._selected_pages,
            )
        )

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
