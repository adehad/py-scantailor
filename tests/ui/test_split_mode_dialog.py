"""Apply scopes and options of the Split Mode dialog, as in the C++ dialog."""

from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.split_mode import SplitModeDialog
from scantailor.core.models import ImageId, PageId
from scantailor.filters.page_split.layout_type import LayoutType

PAGES = [
    PageId(image_id=ImageId(file_path=Path(f"p{index}.png"))) for index in range(6)
]


def _accepted(dialog: SplitModeDialog) -> set[PageId]:
    chosen: list[set[PageId]] = []
    dialog.settings_accepted.connect(lambda pages, _type, _cut: chosen.append(pages))
    dialog._on_submit()
    return chosen[0]


def _dialog(
    qtbot: QtBot,
    current: int,
    selected: list[int] | None = None,
    layout_type: LayoutType = LayoutType.AUTO_LAYOUT_TYPE,
) -> SplitModeDialog:
    dialog = SplitModeDialog(
        current_page=PAGES[current],
        all_pages=PAGES,
        selected_pages={PAGES[i] for i in selected or []},
        layout_type=layout_type,
    )
    qtbot.addWidget(dialog)
    return dialog


@pytest.mark.parametrize(
    ("current", "expected"),
    [(1, [1, 3, 5]), (2, [0, 2, 4])],
    ids=["odd-current", "even-current"],
)
def test_every_other_page_shares_the_parity_of_the_current_page(
    qtbot: QtBot, current: int, expected: list[int]
):
    dialog = _dialog(qtbot, current)
    dialog._every_other_rb.setChecked(True)

    assert _accepted(dialog) == {PAGES[i] for i in expected}


@pytest.mark.parametrize(
    ("current", "expected"),
    [(1, [1, 3, 5]), (2, [2, 4])],
    ids=["odd-current", "even-current"],
)
def test_this_page_every_other_starts_at_the_current_page(
    qtbot: QtBot, current: int, expected: list[int]
):
    dialog = _dialog(qtbot, current)
    dialog._this_every_other_rb.setChecked(True)

    assert _accepted(dialog) == {PAGES[i] for i in expected}


def test_followers_include_the_current_page_and_all_later_ones(qtbot: QtBot):
    dialog = _dialog(qtbot, 3)
    dialog._this_page_and_followers_rb.setChecked(True)

    assert _accepted(dialog) == set(PAGES[3:])


@pytest.mark.parametrize(
    ("current", "expected"),
    [(1, [1, 4]), (2, [2, 5])],
    ids=["first-selected-current", "second-selected-current"],
)
def test_every_other_selected_page_is_taken_in_page_order_including_the_current(
    qtbot: QtBot, current: int, expected: list[int]
):
    dialog = _dialog(qtbot, current, selected=[4, 1, 2, 5])
    dialog._every_other_selected_rb.setChecked(True)

    assert _accepted(dialog) == {PAGES[i] for i in expected}


def test_selected_pages_scope_returns_the_selection(qtbot: QtBot):
    dialog = _dialog(qtbot, 0, selected=[2, 3])
    dialog._selected_pages_rb.setChecked(True)

    assert _accepted(dialog) == {PAGES[2], PAGES[3]}


@pytest.mark.parametrize("selected", [[], [2]], ids=["none", "one"])
def test_selected_scopes_are_disabled_with_at_most_one_selected_page(
    qtbot: QtBot, selected: list[int]
):
    dialog = _dialog(qtbot, 0, selected=selected)

    assert not dialog._selected_pages_rb.isEnabled()
    assert not dialog._every_other_selected_rb.isEnabled()
    assert dialog._all_pages_rb.isEnabled()


def test_selected_scopes_are_enabled_with_two_selected_pages(qtbot: QtBot):
    dialog = _dialog(qtbot, 0, selected=[2, 3])

    assert dialog._selected_pages_rb.isEnabled()
    assert dialog._every_other_selected_rb.isEnabled()


def test_apply_cut_is_disabled_and_cleared_in_auto_mode(qtbot: QtBot):
    dialog = _dialog(qtbot, 0, layout_type=LayoutType.TWO_PAGES)
    assert dialog._apply_cut_option.isEnabled()
    dialog._apply_cut_option.setChecked(True)

    dialog._mode_auto.setChecked(True)

    assert not dialog._apply_cut_option.isEnabled()
    assert not dialog._apply_cut_option.isChecked()
    assert not dialog.apply_cut


def test_apply_cut_starts_disabled_when_the_page_is_in_auto_mode(qtbot: QtBot):
    dialog = _dialog(qtbot, 0, layout_type=LayoutType.AUTO_LAYOUT_TYPE)

    assert not dialog._apply_cut_option.isEnabled()
    assert not dialog._apply_cut_option.isChecked()
