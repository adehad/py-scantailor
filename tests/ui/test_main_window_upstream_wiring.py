"""The main window routes the new Margins and Fix Orientation controls."""

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.apply_to import ApplyToDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.models import Dpi, Margins
from scantailor.core.propagation import ApplyScope
from scantailor.core.stage_sequence import FilterStage
from scantailor.filters.fix_orientation.image_trim import ImageTrim
from tests.ui.conftest import settle


@pytest.fixture
def settled_window(
    qtbot: QtBot, project_window: MainWindowManager
) -> MainWindowManager:
    settle(project_window)
    return project_window


def _switch_stage(window: MainWindowManager, stage: FilterStage) -> None:
    getattr(window, f"actionSwitchFilter{int(stage) + 1}").trigger()


def _answer_apply_dialog(checked_sides: dict[str, bool] | None = None) -> list[str]:
    """Accept the next apply dialog for every page; return its option titles."""
    seen: list[str] = []

    def poll() -> None:
        modal = QtWidgets.QApplication.activeModalWidget()
        if not isinstance(modal, ApplyToDialog):
            QtCore.QTimer.singleShot(10, poll)
            return
        seen.extend(box.text() for box in modal._options.values())
        modal.scope = ApplyScope.ALL_PAGES
        for key, checked in (checked_sides or {}).items():
            modal._options[key].setChecked(checked)
        modal.accept()

    QtCore.QTimer.singleShot(10, poll)
    return seen


def _find[T: QtWidgets.QWidget](
    window: MainWindowManager, kind: type[T], name: str
) -> T:
    widget = window.filterOptions.findChild(kind, name)
    assert widget is not None
    return widget


class TestMargins:
    def test_only_the_checked_sides_are_copied(self, settled_window: MainWindowManager):
        window = settled_window
        _switch_stage(window, FilterStage.PAGE_LAYOUT)
        pages = window.pages()
        current = window.current_page
        assert current is not None
        layout = window.stages.page_layout_filter.settings
        layout.set_margins(current, Margins(left=11, right=12, top=13, bottom=14))
        before = layout.get_params(pages[2]).hard_margins_mm

        sides = _answer_apply_dialog(
            {"left": True, "right": False, "top": True, "bottom": False}
        )
        _find(window, QtWidgets.QPushButton, "applyMarginsBtn").click()

        copied = layout.get_params(pages[2]).hard_margins_mm
        assert sides == ["Left", "Right", "Top", "Bottom"]
        assert (copied.left, copied.top) == (11, 13)
        assert (copied.right, copied.bottom) == (before.right, before.bottom)

    def test_choosing_no_side_changes_nothing(self, settled_window: MainWindowManager):
        window = settled_window
        _switch_stage(window, FilterStage.PAGE_LAYOUT)
        current = window.current_page
        assert current is not None
        layout = window.stages.page_layout_filter.settings
        layout.set_margins(current, Margins.uniform(9.0))
        window.modified = False

        _answer_apply_dialog(
            {"left": False, "right": False, "top": False, "bottom": False}
        )

        assert not window.show_apply_dialog(FilterStage.PAGE_LAYOUT, part="margins")
        assert not window.modified

    def test_match_size_to_all_pages_turns_matching_on_everywhere(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.PAGE_LAYOUT)
        layout = window.stages.page_layout_filter.settings
        for page in window.pages():
            params = layout.get_params(page)
            layout.set_params(
                page,
                params.with_alignment(
                    params.alignment.model_copy(update={"is_null": True})
                ),
            )

        _find(window, QtWidgets.QPushButton, "matchSizeToAllBtn").click()

        assert all(not layout.get_params(p).alignment.is_null for p in window.pages())
        assert window.modified

    def test_typing_a_source_dpi_stores_it_for_the_image(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.PAGE_LAYOUT)
        window.modified = False
        x_field = _find(window, QtWidgets.QLineEdit, "sourceXDpi")
        y_field = _find(window, QtWidgets.QLineEdit, "sourceYDpi")

        x_field.setText("400")
        y_field.setText("400")
        y_field.editingFinished.emit()

        info = window._current_image_info()
        assert info is not None
        assert info.metadata.dpi == Dpi.uniform(400)
        assert window.modified

    def test_the_source_dpi_fields_show_the_current_image(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.PAGE_LAYOUT)
        info = window._current_image_info()
        assert info is not None

        shown = _find(window, QtWidgets.QLineEdit, "sourceXDpi").text()

        assert shown == str(info.metadata.dpi.horizontal)

    def test_fix_all_opens_the_fix_dpi_dialog(self, settled_window: MainWindowManager):
        window = settled_window
        _switch_stage(window, FilterStage.PAGE_LAYOUT)
        opened: list[int] = []
        window.fixDpiDialogRequested = lambda: opened.append(1)  # type: ignore[method-assign]
        _switch_stage(window, FilterStage.SELECT_CONTENT)
        _switch_stage(window, FilterStage.PAGE_LAYOUT)

        _find(window, QtWidgets.QPushButton, "fixDpiBtn").click()

        assert opened == [1]


class TestTrim:
    def test_enabling_the_trim_in_the_panel_stores_it_and_trims_the_view(
        self, settled_window: MainWindowManager, qtbot: QtBot
    ):
        window = settled_window
        _switch_stage(window, FilterStage.FIX_ORIENTATION)
        settle(window)
        current = window.current_page
        assert current is not None

        _find(window, QtWidgets.QCheckBox, "trimEnabledCheck").setChecked(True)
        _find(window, QtWidgets.QSpinBox, "trimLeftSpin").setValue(10)

        stored = window.stages.fix_orientation_filter.get_trim(current.image_id)
        assert stored.enabled
        assert stored.left == 10
        assert window.modified

    def test_apply_to_copies_the_trim_with_the_rotation(
        self, settled_window: MainWindowManager
    ):
        window = settled_window
        _switch_stage(window, FilterStage.FIX_ORIENTATION)
        current = window.current_page
        assert current is not None
        pages = window.pages()
        orientation = window.stages.fix_orientation_filter
        orientation.set_trim(
            current.image_id, ImageTrim(enabled=True, left=5, top=6, right=7, bottom=8)
        )

        _answer_apply_dialog()
        window.show_apply_dialog(FilterStage.FIX_ORIENTATION)

        assert orientation.get_trim(pages[2].image_id) == ImageTrim(
            enabled=True, left=5, top=6, right=7, bottom=8
        )
