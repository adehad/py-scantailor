"""Dialog that picks the pages a stage's settings are copied to."""

from collections.abc import Mapping, Sequence
from typing import cast

from PySide6 import QtCore, QtWidgets

from scantailor.app.ui.icons import icon
from scantailor.core.models import PageId
from scantailor.core.propagation import ApplyScope, select_pages
from scantailor.filters.output.params import ApplyGroup

# Labels are translated with self.tr() in ApplyToDialog, so their context is that class.
_SCOPE_BUTTONS: tuple[tuple[ApplyScope, str, str], ...] = (
    (
        ApplyScope.THIS_PAGE,
        "thisPageRB",
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyToDialog", "This page only (already applied)"
            ),
        ),
    ),
    (
        ApplyScope.ALL_PAGES,
        "allPagesRB",
        cast(str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "All pages")),
    ),
    (
        ApplyScope.THIS_PAGE_AND_FOLLOWERS,
        "thisPageAndFollowersRB",
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyToDialog", "This page and the following ones"
            ),
        ),
    ),
    (
        ApplyScope.THIS_PAGE_EVERY_OTHER,
        "thisEveryOtherRB",
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyToDialog", "This page and the following every other page"
            ),
        ),
    ),
    (
        ApplyScope.EVERY_OTHER,
        "everyOtherRB",
        cast(str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "Every other page")),
    ),
    (
        ApplyScope.SELECTED_PAGES,
        "selectedPagesRB",
        cast(str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "Selected pages")),
    ),
    (
        ApplyScope.EVERY_OTHER_SELECTED,
        "everyOtherSelectedRB",
        cast(
            str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "Every other selected page")
        ),
    ),
)
# The main window translates these titles in its own context.
OUTPUT_GROUP_TITLES: dict[ApplyGroup, str] = {
    ApplyGroup.DPI: cast(
        str, QtCore.QT_TRANSLATE_NOOP("MainWindowManager", "Apply Output Resolution")
    ),
    ApplyGroup.COLOR_MODE: cast(
        str, QtCore.QT_TRANSLATE_NOOP("MainWindowManager", "Apply Color Settings")
    ),
    ApplyGroup.SPLITTING: cast(
        str, QtCore.QT_TRANSLATE_NOOP("MainWindowManager", "Apply Splitting Settings")
    ),
    ApplyGroup.BINARIZATION: cast(
        str,
        QtCore.QT_TRANSLATE_NOOP("MainWindowManager", "Apply Binarization Settings"),
    ),
    ApplyGroup.DESPECKLE: cast(
        str, QtCore.QT_TRANSLATE_NOOP("MainWindowManager", "Apply Despeckling Level")
    ),
    ApplyGroup.DEWARPING: cast(
        str, QtCore.QT_TRANSLATE_NOOP("MainWindowManager", "Apply Dewarping Mode")
    ),
    ApplyGroup.DEPTH_PERCEPTION: cast(
        str, QtCore.QT_TRANSLATE_NOOP("MainWindowManager", "Apply Depth Perception")
    ),
}
DESKEW_TITLE = cast(str, QtCore.QT_TRANSLATE_NOOP("MainWindowManager", "Apply Deskew"))
# Which parts of the Deskew settings are copied, as option key to label.
DESKEW_OPTIONS: dict[str, str] = {
    "deskew": cast(
        str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "Deskew angle and mode")
    ),
    "oblique": cast(
        str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "Oblique angle and mode")
    ),
}
MARGINS_TITLE = "Apply Margins"
MARGINS_OPTIONS_TITLE = "Apply margins"
# Which sides of the margins are copied, as option key to label, in the order shown.
MARGIN_SIDE_OPTIONS: dict[str, str] = {
    "left": cast(str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "Left")),
    "right": cast(str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "Right")),
    "top": cast(str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "Top")),
    "bottom": cast(str, QtCore.QT_TRANSLATE_NOOP("ApplyToDialog", "Bottom")),
}
# Each side is marked with the edge the margin lies along.
MARGIN_SIDE_ICONS: dict[str, str] = {
    "left": "arrow-left-to-line",
    "right": "arrow-right-to-line",
    "top": "arrow-up-to-line",
    "bottom": "arrow-down-to-line",
}
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
        option_icons: Mapping[str, str] | None = None,
        options_title: str | None = None,
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
            option_icons (Mapping[str, str] | None): Glyph name shown beside the
                check box of an option key.
            options_title (str | None): When given, the extra check boxes sit side
                by side in a group box with this title instead of one per line.
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
        option_parent: QtWidgets.QWidget = self
        option_layout: QtWidgets.QBoxLayout = layout
        if options and options_title is not None:
            options_group = QtWidgets.QGroupBox(self.tr(options_title), self)
            options_group.setObjectName("optionsGroup")
            option_layout = QtWidgets.QHBoxLayout(options_group)
            layout.addWidget(options_group)
            option_parent = options_group
        for key, label in (options or {}).items():
            box = QtWidgets.QCheckBox(self.tr(label), option_parent)
            box.setObjectName(f"{key}Option")
            box.setChecked(True)
            glyph = (option_icons or {}).get(key)
            if glyph is not None:
                box.setIcon(icon(glyph))
            option_layout.addWidget(box)
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
        label.setProperty("textRole", "small")
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

    def checked_options(self) -> set[str]:
        """Return the keys of the extra options that are checked."""
        return {key for key, box in self._options.items() if box.isChecked()}

    def option_checked(self, key: str) -> bool:
        """Return whether an extra option is checked.

        Args:
            key (str): The option key given at construction.

        Returns:
            bool: The check box state.
        """
        return self._options[key].isChecked()


class ApplyDeskewDialog(ApplyToDialog):
    """Apply dialog of the Deskew stage."""

    # lupdate only reads literal QT_TRANSLATE_NOOP calls, so the labels this
    # dialog shows are listed here under its own context.
    translatable_labels: tuple[str, ...] = (
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyDeskewDialog", "This page only (already applied)"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyDeskewDialog", "All pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyDeskewDialog", "This page and the following ones"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyDeskewDialog", "This page and the following every other page"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyDeskewDialog", "Every other page"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyDeskewDialog", "Selected pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyDeskewDialog", "Every other selected page"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyDeskewDialog", "Apply to"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyDeskewDialog",
                "Use Ctrl+Click / Shift+Click to select multiple pages.",
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyDeskewDialog", "The current page will be included."
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyDeskewDialog", "Deskew angle and mode"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyDeskewDialog", "Oblique angle and mode"),
        ),
    )


class ApplyFixOrientationDialog(ApplyToDialog):
    """Apply dialog of the Fix Orientation stage."""

    # lupdate only reads literal QT_TRANSLATE_NOOP calls, so the labels this
    # dialog shows are listed here under its own context.
    translatable_labels: tuple[str, ...] = (
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyFixOrientationDialog", "This page only (already applied)"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyFixOrientationDialog", "All pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyFixOrientationDialog", "This page and the following ones"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyFixOrientationDialog",
                "This page and the following every other page",
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyFixOrientationDialog", "Every other page"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyFixOrientationDialog", "Selected pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyFixOrientationDialog", "Every other selected page"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyFixOrientationDialog", "Apply to"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyFixOrientationDialog",
                "Use Ctrl+Click / Shift+Click to select multiple pages.",
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyFixOrientationDialog", "The current page will be included."
            ),
        ),
    )


class ApplyPageLayoutDialog(ApplyToDialog):
    """Apply dialog of the Page Layout stage, for margins and alignment."""

    # lupdate only reads literal QT_TRANSLATE_NOOP calls, so the labels this
    # dialog shows are listed here under its own context.
    translatable_labels: tuple[str, ...] = (
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyPageLayoutDialog", "This page only (already applied)"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyPageLayoutDialog", "All pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyPageLayoutDialog", "This page and the following ones"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyPageLayoutDialog", "This page and the following every other page"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyPageLayoutDialog", "Every other page"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyPageLayoutDialog", "Selected pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyPageLayoutDialog", "Every other selected page"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyPageLayoutDialog", "Apply to"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyPageLayoutDialog",
                "Use Ctrl+Click / Shift+Click to select multiple pages.",
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyPageLayoutDialog", "The current page will be included."
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyPageLayoutDialog", "Left"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyPageLayoutDialog", "Right"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyPageLayoutDialog", "Top"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyPageLayoutDialog", "Bottom"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyPageLayoutDialog", "Apply margins"),
        ),
    )


class ApplySelectContentDialog(ApplyToDialog):
    """Apply dialog of the Select Content stage."""

    # lupdate only reads literal QT_TRANSLATE_NOOP calls, so the labels this
    # dialog shows are listed here under its own context.
    translatable_labels: tuple[str, ...] = (
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplySelectContentDialog", "This page only (already applied)"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplySelectContentDialog", "All pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplySelectContentDialog", "This page and the following ones"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplySelectContentDialog",
                "This page and the following every other page",
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplySelectContentDialog", "Every other page"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplySelectContentDialog", "Selected pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplySelectContentDialog", "Every other selected page"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplySelectContentDialog", "Apply to"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplySelectContentDialog",
                "Use Ctrl+Click / Shift+Click to select multiple pages.",
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplySelectContentDialog", "The current page will be included."
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplySelectContentDialog", "Apply content box"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplySelectContentDialog", "Apply page box"),
        ),
    )


class ApplyColorsDialog(ApplyToDialog):
    """Apply dialog of the Output stage for the colour and other option groups."""

    # lupdate only reads literal QT_TRANSLATE_NOOP calls, so the labels this
    # dialog shows are listed here under its own context.
    translatable_labels: tuple[str, ...] = (
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyColorsDialog", "This page only (already applied)"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyColorsDialog", "All pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyColorsDialog", "This page and the following ones"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyColorsDialog", "This page and the following every other page"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyColorsDialog", "Every other page"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyColorsDialog", "Selected pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyColorsDialog", "Every other selected page"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyColorsDialog", "Apply to"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyColorsDialog",
                "Use Ctrl+Click / Shift+Click to select multiple pages.",
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyColorsDialog", "The current page will be included."
            ),
        ),
    )


class ApplyChangeDpiDialog(ApplyToDialog):
    """Apply dialog of the Output resolution group."""

    # lupdate only reads literal QT_TRANSLATE_NOOP calls, so the labels this
    # dialog shows are listed here under its own context.
    translatable_labels: tuple[str, ...] = (
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDpiDialog", "This page only (already applied)"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyChangeDpiDialog", "All pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDpiDialog", "This page and the following ones"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDpiDialog", "This page and the following every other page"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyChangeDpiDialog", "Every other page"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyChangeDpiDialog", "Selected pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDpiDialog", "Every other selected page"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyChangeDpiDialog", "Apply to"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDpiDialog",
                "Use Ctrl+Click / Shift+Click to select multiple pages.",
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDpiDialog", "The current page will be included."
            ),
        ),
    )


class ApplyChangeDewarpingDialog(ApplyToDialog):
    """Apply dialog of the Output dewarping group."""

    # lupdate only reads literal QT_TRANSLATE_NOOP calls, so the labels this
    # dialog shows are listed here under its own context.
    translatable_labels: tuple[str, ...] = (
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDewarpingDialog", "This page only (already applied)"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyChangeDewarpingDialog", "All pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDewarpingDialog", "This page and the following ones"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDewarpingDialog",
                "This page and the following every other page",
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyChangeDewarpingDialog", "Every other page"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyChangeDewarpingDialog", "Selected pages"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDewarpingDialog", "Every other selected page"
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP("ApplyChangeDewarpingDialog", "Apply to"),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDewarpingDialog",
                "Use Ctrl+Click / Shift+Click to select multiple pages.",
            ),
        ),
        cast(
            str,
            QtCore.QT_TRANSLATE_NOOP(
                "ApplyChangeDewarpingDialog", "The current page will be included."
            ),
        ),
    )
