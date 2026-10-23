"""Thumbnail list context menu and the Remove pages dialog's output choice."""

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QCheckBox, QDialogButtonBox, QLabel, QWidget
from pytestqt.qtbot import QtBot

from scantailor.app.ui import icons
from scantailor.app.ui.dialogs.remove_pages import RemovePagesDialog
from scantailor.app.ui.thumbnails import ThumbnailListWidget
from scantailor.core.models import ImageId, PageId
from scantailor.core.project import ImageMetadata, Project
from tests.ui.hosting import show_in_host


def render(page: PageId, stage: str) -> QImage:
    image = QImage(20, 30, QImage.Format.Format_RGB32)
    image.fill(QColor("gray"))
    return image


@pytest.fixture
def project() -> Project:
    project = Project()
    for number in (1, 2, 3):
        project.add_image(
            ImageId(file_path=Path(f"page_{number}.png")),
            ImageMetadata(width=100, height=100),
        )
    return project


@pytest.fixture
def widget(qtbot: QtBot, host: QWidget, project: Project) -> ThumbnailListWidget:
    thumbnails = ThumbnailListWidget(render, project, stage="page_split", parent=host)
    show_in_host(qtbot, thumbnails, 240, 700)
    return thumbnails


def test_page_menu_lists_insert_before_after_and_remove(
    widget: ThumbnailListWidget,
) -> None:
    menu = widget.page_menu(widget.pages()[0])

    actions = [a for a in menu.actions() if not a.isSeparator()]
    assert [a.text() for a in actions] == [
        "Insert before ...",
        "Insert after ...",
        "Remove from project ...",
    ]
    assert any(a.isSeparator() for a in menu.actions())
    assert all(icons.is_palette_icon(a.icon()) for a in actions)
    assert [a.icon().name() for a in actions] == [
        icons.icon("insert-before").name(),
        icons.icon("insert-after").name(),
        icons.icon("trash-2", tint="danger").name(),
    ]


def test_empty_area_menu_offers_insert_here(widget: ThumbnailListWidget) -> None:
    menu = widget.empty_area_menu()

    (action,) = menu.actions()
    assert action.text() == "Insert here ..."
    assert icons.is_palette_icon(action.icon())


def test_insert_actions_emit_the_clicked_page(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    page = widget.pages()[1]
    menu = widget.page_menu(page)
    before, after, _ = [a for a in menu.actions() if not a.isSeparator()]

    with qtbot.waitSignal(widget.insert_before_requested) as before_signal:
        before.trigger()
    with qtbot.waitSignal(widget.insert_after_requested) as after_signal:
        after.trigger()

    assert before_signal.args == [page]
    assert after_signal.args == [page]


def test_remove_action_emits_all_selected_pages(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    pages = widget.pages()
    widget.view.selectAll()
    menu = widget.page_menu(pages[0])
    remove = [a for a in menu.actions() if not a.isSeparator()][2]

    with qtbot.waitSignal(widget.remove_pages_requested) as removed:
        remove.trigger()

    assert removed.args == [pages]


def test_insert_here_action_emits_its_signal(
    qtbot: QtBot, widget: ThumbnailListWidget
) -> None:
    (action,) = widget.empty_area_menu().actions()

    with qtbot.waitSignal(widget.insert_here_requested):
        action.trigger()


def test_right_click_on_an_unselected_page_makes_it_current_before_the_menu(
    widget: ThumbnailListWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[list[str]] = []
    monkeypatch.setattr(
        ThumbnailListWidget,
        "_show_menu",
        lambda self, menu, *_: shown.append([a.text() for a in menu.actions()]),
    )
    target = widget.pages()[2]

    widget.view.customContextMenuRequested.emit(widget.item_rect(target).center())

    assert widget.current_page() == target
    assert shown[0][0] == "Insert before ..."


def test_right_click_on_a_selected_page_keeps_the_selection(
    widget: ThumbnailListWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ThumbnailListWidget, "_show_menu", lambda *_: None)
    pages = widget.pages()
    widget.view.selectAll()

    widget.view.customContextMenuRequested.emit(widget.item_rect(pages[1]).center())

    assert widget.selected_pages() == pages


def test_right_click_past_the_last_page_offers_insert_here(
    widget: ThumbnailListWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[list[str]] = []
    monkeypatch.setattr(
        ThumbnailListWidget,
        "_show_menu",
        lambda self, menu, *_: shown.append([a.text() for a in menu.actions()]),
    )
    below_everything = QPoint(5, widget.view.viewport().height() - 2)

    widget.view.customContextMenuRequested.emit(below_everything)

    assert shown == [["Insert here ..."]]


def test_no_menu_without_a_project(
    qtbot: QtBot, host: QWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    thumbnails = ThumbnailListWidget(render, parent=host)
    shown: list[object] = []
    monkeypatch.setattr(
        ThumbnailListWidget,
        "_show_menu",
        lambda *_: shown.append(1),
    )

    thumbnails.view.customContextMenuRequested.emit(QPoint(3, 3))

    assert shown == []


def test_disabled_context_menu_shows_nothing(
    widget: ThumbnailListWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[object] = []
    monkeypatch.setattr(
        ThumbnailListWidget,
        "_show_menu",
        lambda *_: shown.append(1),
    )
    widget.set_context_menu_enabled(False)

    widget.view.customContextMenuRequested.emit(
        widget.item_rect(widget.pages()[0]).center()
    )

    assert shown == []


def _driven_dialog(qtbot: QtBot) -> RemovePagesDialog:
    dialog = RemovePagesDialog(page_count=2)
    qtbot.addWidget(dialog)
    return dialog


def test_dialog_offers_output_removal_ticked_by_default(qtbot: QtBot) -> None:
    dialog = _driven_dialog(qtbot)

    check = dialog.findChild(QCheckBox, "eraseOutputFiles")
    assert check is not None
    assert check.text() == "Remove output files too"
    assert dialog.erase_output_files


def test_dialog_note_follows_the_output_choice(qtbot: QtBot) -> None:
    dialog = _driven_dialog(qtbot)
    note = dialog.findChild(QLabel, "multiPageWarning")
    check = dialog.findChild(QCheckBox, "eraseOutputFiles")
    assert note is not None and check is not None
    assert note.text() == (
        "Corresponding output files will be deleted, while input files will remain."
    )

    check.setChecked(False)

    assert not dialog.erase_output_files
    assert "output files will remain" in note.text()


def test_dialog_confirm_button_is_labelled_remove(qtbot: QtBot) -> None:
    dialog = _driven_dialog(qtbot)
    buttons = dialog.findChild(QDialogButtonBox, "buttonBox")
    assert buttons is not None

    assert buttons.button(QDialogButtonBox.StandardButton.Ok).text() == "Remove"


@pytest.mark.parametrize(
    ("ticked", "button", "expected"),
    [
        (True, QDialogButtonBox.StandardButton.Ok, True),
        (False, QDialogButtonBox.StandardButton.Ok, False),
        (True, QDialogButtonBox.StandardButton.Cancel, None),
    ],
)
def test_ask_removal_reports_the_choice_or_none_when_cancelled(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    ticked: bool,
    button: QDialogButtonBox.StandardButton,
    expected: bool | None,
) -> None:
    def fake_exec(dialog: RemovePagesDialog) -> int:
        check = dialog.findChild(QCheckBox, "eraseOutputFiles")
        buttons = dialog.findChild(QDialogButtonBox, "buttonBox")
        assert check is not None and buttons is not None
        check.setChecked(ticked)
        buttons.button(button).click()
        return dialog.result()

    monkeypatch.setattr(RemovePagesDialog, "exec", fake_exec)

    assert RemovePagesDialog.ask_removal(2) is expected
    assert RemovePagesDialog.confirm_removal(2) is (expected is not None)
