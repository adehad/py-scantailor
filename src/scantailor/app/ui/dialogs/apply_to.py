"""Dialog that picks the pages a stage's settings are copied to."""

from collections.abc import Mapping, Sequence

from PySide6 import QtWidgets

from scantailor.core.models import PageId
from scantailor.core.propagation import ApplyScope, select_pages

_SCOPE_BUTTONS: tuple[tuple[ApplyScope, str, str], ...] = (
    (ApplyScope.THIS_PAGE, "thisPageRB", "This page only (already applied)"),
    (ApplyScope.ALL_PAGES, "allPagesRB", "All pages"),
    (
        ApplyScope.THIS_PAGE_AND_FOLLOWERS,
        "thisPageAndFollowersRB",
        "This page and the following ones",
    ),
    (
        ApplyScope.THIS_PAGE_EVERY_OTHER,
        "thisEveryOtherRB",
        "This page and the following every other page",
    ),
    (ApplyScope.EVERY_OTHER, "everyOtherRB", "Every other page"),
    (ApplyScope.SELECTED_PAGES, "selectedPagesRB", "Selected pages"),
    (
        ApplyScope.EVERY_OTHER_SELECTED,
        "everyOtherSelectedRB",
        "Every other selected page",
    ),
)
_SELECTION_SCOPES = (ApplyScope.SELECTED_PAGES, ApplyScope.EVERY_OTHER_SELECTED)


class ApplyToDialog(QtWidgets.QDialog):
    """Asks which pages receive the current page's settings.

    The selection scopes need more than one selected page and stay disabled
    otherwise. After the dialog is accepted, ``pages()`` lists the targets.
    """

    def __init__(
        self,
        current_page: PageId,
        all_pages: Sequence[PageId],
        selected_pages: Sequence[PageId] = (),
        parent: QtWidgets.QWidget | None = None,
        title: str = "Apply to",
        options: Mapping[str, str] | None = None,
    ) -> None:
        """Build the dialog.

        Args:
            current_page (PageId): The page whose settings are copied.
            all_pages (Sequence[PageId]): Every page in display order.
            selected_pages (Sequence[PageId]): Selected pages in display order.
            parent (QtWidgets.QWidget | None): Owner of the dialog.
            title (str): Window title.
            options (Mapping[str, str] | None): Extra check boxes, as option key
                to label; all start checked.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self._current_page = current_page
        self._all_pages = list(all_pages)
        self._selected_pages = list(selected_pages)

        layout = QtWidgets.QVBoxLayout(self)
        group = QtWidgets.QGroupBox(self.tr("Apply to"), self)
        group_layout = QtWidgets.QVBoxLayout(group)
        self._buttons: dict[ApplyScope, QtWidgets.QRadioButton] = {}
        for scope, name, label in _SCOPE_BUTTONS:
            button = QtWidgets.QRadioButton(self.tr(label), group)
            button.setObjectName(name)
            group_layout.addWidget(button)
            self._buttons[scope] = button
            if scope is ApplyScope.SELECTED_PAGES:
                group_layout.addWidget(
                    self._hint(
                        self.tr(
                            "Use Ctrl+Click / Shift+Click to select multiple pages."
                        )
                    )
                )
            if scope is ApplyScope.EVERY_OTHER_SELECTED:
                group_layout.addWidget(
                    self._hint(self.tr("The current page will be included."))
                )
        self._buttons[ApplyScope.THIS_PAGE].setChecked(True)
        if len(self._selected_pages) <= 1:
            for scope in _SELECTION_SCOPES:
                self._buttons[scope].setEnabled(False)
        layout.addWidget(group)

        self._options: dict[str, QtWidgets.QCheckBox] = {}
        for key, label in (options or {}).items():
            box = QtWidgets.QCheckBox(label, self)
            box.setChecked(True)
            layout.addWidget(box)
            self._options[key] = box

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _hint(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setContentsMargins(24, 0, 0, 0)
        font = label.font()
        font.setPointSize(max(font.pointSize() - 2, 6))
        label.setFont(font)
        return label

    @property
    def scope(self) -> ApplyScope:
        """The chosen scope."""
        for scope, button in self._buttons.items():
            if button.isChecked():
                return scope
        return ApplyScope.THIS_PAGE

    @scope.setter
    def scope(self, scope: ApplyScope) -> None:
        self._buttons[scope].setChecked(True)

    def pages(self) -> list[PageId]:
        """Return the pages the chosen scope covers, in display order."""
        return select_pages(
            self.scope, self._current_page, self._all_pages, self._selected_pages
        )

    def option_checked(self, key: str) -> bool:
        """Return whether an extra option is checked.

        Args:
            key (str): The option key given at construction.

        Returns:
            bool: The check box state.
        """
        return self._options[key].isChecked()
