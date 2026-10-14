"""The New Project files dialog lists PDFs, extracts their pages and takes drops."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pdf_helpers import PAGE_WORDS, write_text_pdf
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs import project_files
from scantailor.app.ui.dialogs.pdf_import import ExistingPages
from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.pdf_worker import ImportedPdf
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi
from scantailor.core.pdf_source import PdfSource

if TYPE_CHECKING:
    from conftest import FileDialogStub

LEFT = QtCore.Qt.MouseButton.LeftButton


def _names(model: QtCore.QAbstractItemModel) -> list[str]:
    return [model.index(row, 0).data() for row in range(model.rowCount())]


def _select(view: QtWidgets.QListView, rows: list[int]) -> None:
    view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
    for row in rows:
        view.selectionModel().select(
            view.model().index(row, 0),
            QtCore.QItemSelectionModel.SelectionFlag.Select,
        )


def _write_image(path: Path, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_image(path, np.zeros((40, 30), dtype=bool), Dpi.uniform(dpi))
    return path


def _fake_extract(
    calls: list[Path] | None = None, *, cancel: bool = False, pages: int = 2
) -> Callable[[Path, QtWidgets.QWidget], ImportedPdf | None]:
    """Stand in for rendering: write page images and return them as a render would."""

    def extract(pdf_path: Path, _owner: QtWidgets.QWidget) -> ImportedPdf | None:
        if calls is not None:
            calls.append(pdf_path)
        if cancel:
            return None
        folder = pdf_path.with_name(f"{pdf_path.stem}_pages")
        images = [
            _write_image(folder / f"page_{index + 1:04d}.png") for index in range(pages)
        ]
        source = PdfSource(
            pdf_path=pdf_path,
            dpi=300,
            pages_dir=folder,
            pages={image.name: index for index, image in enumerate(images)},
        )
        return ImportedPdf(source, images)

    return extract


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    scans = tmp_path / "scans"
    scans.mkdir()
    return scans


def _dialog(
    qtbot: QtBot,
    input_dir: Path,
    extract: Callable[[Path, QtWidgets.QWidget], ImportedPdf | None] | None = None,
    project_files: list[str] | None = None,
) -> ProjectFilesDialog:
    dialog = ProjectFilesDialog(
        input_dir=input_dir, project_files=project_files, extract_pdf=extract
    )
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog


def test_pdfs_are_listed_above_the_images_with_their_page_counts(
    qtbot: QtBot, folder: Path
):
    write_text_pdf(folder / "issue.pdf")
    _write_image(folder / "cover.png")

    dialog = _dialog(qtbot, folder)

    assert _names(dialog._off_project_model) == [
        f"issue.pdf ({len(PAGE_WORDS)} pages)",
        "cover.png",
    ]


def test_a_pdf_whose_pages_are_already_extracted_says_so(qtbot: QtBot, folder: Path):
    write_text_pdf(folder / "issue.pdf")
    _write_image(folder / "issue_pages" / "page_0001.png")

    dialog = _dialog(qtbot, folder)

    assert _names(dialog._off_project_model) == [
        f"issue.pdf ({len(PAGE_WORDS)} pages, extracted)"
    ]


def test_an_unreadable_pdf_is_not_listed(qtbot: QtBot, folder: Path):
    (folder / "broken.pdf").write_bytes(b"not a pdf")

    dialog = _dialog(qtbot, folder)

    assert _names(dialog._off_project_model) == []


def test_adding_a_pdf_extracts_it_and_moves_its_pages_into_the_project(
    qtbot: QtBot, folder: Path
):
    write_text_pdf(folder / "issue.pdf")
    calls: list[Path] = []
    dialog = _dialog(qtbot, folder, _fake_extract(calls))

    _select(dialog._off_project_list, [0])
    dialog._add_to_project_btn.click()

    assert calls == [folder / "issue.pdf"]
    assert _names(dialog._in_project_model) == ["page_0001.png", "page_0002.png"]
    assert _names(dialog._off_project_model) == [
        f"issue.pdf ({len(PAGE_WORDS)} pages, extracted)"
    ]


def test_a_cancelled_extraction_leaves_both_lists_as_they_were(
    qtbot: QtBot, folder: Path
):
    write_text_pdf(folder / "issue.pdf")
    dialog = _dialog(qtbot, folder, _fake_extract(cancel=True))
    before = _names(dialog._off_project_model)

    _select(dialog._off_project_list, [0])
    dialog._add_to_project_btn.click()

    assert _names(dialog._off_project_model) == before
    assert _names(dialog._in_project_model) == []
    assert dialog.get_imported_pdfs() == []


def test_a_mixed_selection_adds_the_images_at_once_and_extracts_the_pdf(
    qtbot: QtBot, folder: Path
):
    write_text_pdf(folder / "issue.pdf")
    _write_image(folder / "cover.png")
    calls: list[Path] = []
    dialog = _dialog(qtbot, folder, _fake_extract(calls))

    _select(dialog._off_project_list, [0, 1])
    dialog._add_to_project_btn.click()

    assert calls == [folder / "issue.pdf"]
    assert _names(dialog._in_project_model) == [
        "cover.png",
        "page_0001.png",
        "page_0002.png",
    ]


def test_a_pdf_that_is_not_extracted_stops_the_rest_of_the_selection(
    qtbot: QtBot, folder: Path
):
    write_text_pdf(folder / "first.pdf")
    write_text_pdf(folder / "second.pdf")
    calls: list[Path] = []
    dialog = _dialog(qtbot, folder, _fake_extract(calls, cancel=True))

    _select(dialog._off_project_list, [0, 1])
    dialog._add_to_project_btn.click()

    assert calls == [folder / "first.pdf"]


def test_pdfs_of_another_folder_are_added_with_add_pdf(
    qtbot: QtBot,
    folder: Path,
    tmp_path: Path,
    file_dialogs: "FileDialogStub",
):
    other = write_text_pdf(tmp_path / "elsewhere.pdf")
    dialog = _dialog(qtbot, folder, _fake_extract())
    file_dialogs.open_file = str(other)

    dialog._add_pdf_btn.click()

    assert _names(dialog._off_project_model) == [
        f"elsewhere.pdf ({len(PAGE_WORDS)} pages)"
    ]


def test_extracted_pages_come_back_with_their_source_and_follow_removals(
    qtbot: QtBot, folder: Path
):
    write_text_pdf(folder / "issue.pdf")
    dialog = _dialog(qtbot, folder, _fake_extract())
    _select(dialog._off_project_list, [0])
    dialog._add_to_project_btn.click()

    _select(dialog._in_project_list, [1])
    dialog._remove_from_project_btn.click()

    [imported] = dialog.get_imported_pdfs()
    assert imported.source.pdf_path == folder / "issue.pdf"
    assert imported.source.pages == {"page_0001.png": 0}
    assert imported.images == [folder / "issue_pages" / "page_0001.png"]


def test_dropping_a_folder_makes_it_the_input_folder(
    qtbot: QtBot, folder: Path, tmp_path: Path
):
    other = tmp_path / "other"
    _write_image(other / "loose.png")
    dialog = _dialog(qtbot, folder)

    _drop(dialog, [other])

    assert dialog.get_input_directory() == other
    assert _names(dialog._off_project_model) == ["loose.png"]


def test_dropping_a_pdf_extracts_it_and_dropping_an_image_adds_it(
    qtbot: QtBot, folder: Path, tmp_path: Path
):
    dropped_pdf = write_text_pdf(tmp_path / "dropped.pdf")
    loose = _write_image(tmp_path / "loose.png")
    calls: list[Path] = []
    dialog = _dialog(qtbot, folder, _fake_extract(calls))

    _drop(dialog, [dropped_pdf, loose])

    assert calls == [dropped_pdf]
    assert _names(dialog._in_project_model) == [
        "loose.png",
        "page_0001.png",
        "page_0002.png",
    ]


def _drop(dialog: ProjectFilesDialog, paths: list[Path]) -> None:
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(path)) for path in paths])
    event = QtGui.QDropEvent(
        QtCore.QPointF(1, 1),
        QtCore.Qt.DropAction.CopyAction,
        mime,
        LEFT,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    dialog.dropEvent(event)


def test_the_default_extraction_asks_then_renders_the_pages_beside_the_pdf(
    qtbot: QtBot,
    folder: Path,
    respond_to_modal,
):
    write_text_pdf(folder / "issue.pdf")
    dialog = _dialog(qtbot, folder)
    respond_to_modal(accept=True)
    _select(dialog._off_project_list, [0])

    dialog._add_to_project_btn.click()

    pages = folder / "issue_pages"
    assert _names(dialog._in_project_model) == ["page_0001.png", "page_0002.png"]
    assert (pages / "page_0002.png").is_file()
    [imported] = dialog.get_imported_pdfs()
    assert imported.source.dpi == 300
    assert imported.source.pages_dir == pages


@pytest.fixture
def prompted_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[ExistingPages], None]:
    """Make the default extraction use an existing folder, with one answer given."""

    def choose(answer: ExistingPages) -> None:
        monkeypatch.setattr(project_files, "ask_existing_pages", lambda *_args: answer)

    return choose


def test_existing_pages_can_be_used_without_rendering_again(
    qtbot: QtBot,
    folder: Path,
    respond_to_modal,
    prompted_extract: Callable[[ExistingPages], None],
    monkeypatch: pytest.MonkeyPatch,
):
    write_text_pdf(folder / "issue.pdf")
    _write_image(folder / "issue_pages" / "page_0001.png", dpi=200)
    _write_image(folder / "issue_pages" / "page_0002.png", dpi=200)
    prompted_extract(ExistingPages.USE)
    monkeypatch.setattr(
        project_files,
        "import_pdf_pages",
        lambda *_a, **_k: pytest.fail("the existing pages must not be rendered"),
    )
    dialog = _dialog(qtbot, folder)
    respond_to_modal(accept=True)
    _select(dialog._off_project_list, [0])

    dialog._add_to_project_btn.click()

    assert _names(dialog._in_project_model) == ["page_0001.png", "page_0002.png"]
    [imported] = dialog.get_imported_pdfs()
    assert imported.source.dpi == 200


def test_rendering_again_replaces_the_existing_pages(
    qtbot: QtBot,
    folder: Path,
    respond_to_modal,
    prompted_extract: Callable[[ExistingPages], None],
    monkeypatch: pytest.MonkeyPatch,
):
    write_text_pdf(folder / "issue.pdf")
    _write_image(folder / "issue_pages" / "page_0001.png")
    prompted_extract(ExistingPages.RENDER_AGAIN)
    replacing: list[bool] = []
    real = project_files.import_pdf_pages

    def recording(pdf_path, out_dir, dpi, report, cancel, **kwargs):
        replacing.append(kwargs["clear_existing"])
        return real(pdf_path, out_dir, dpi, report, cancel, **kwargs)

    monkeypatch.setattr(project_files, "import_pdf_pages", recording)
    dialog = _dialog(qtbot, folder)
    respond_to_modal(accept=True)
    _select(dialog._off_project_list, [0])

    dialog._add_to_project_btn.click()

    assert replacing == [True]
    assert _names(dialog._in_project_model) == ["page_0001.png", "page_0002.png"]


def test_cancelling_the_existing_pages_question_adds_nothing(
    qtbot: QtBot,
    folder: Path,
    respond_to_modal,
    prompted_extract: Callable[[ExistingPages], None],
):
    write_text_pdf(folder / "issue.pdf")
    _write_image(folder / "issue_pages" / "page_0001.png")
    prompted_extract(ExistingPages.CANCEL)
    dialog = _dialog(qtbot, folder)
    respond_to_modal(accept=True)
    _select(dialog._off_project_list, [0])

    dialog._add_to_project_btn.click()

    assert _names(dialog._in_project_model) == []
    assert dialog.get_imported_pdfs() == []
