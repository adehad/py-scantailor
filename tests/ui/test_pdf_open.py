"""File > Open PDF extracts a PDF's pages and ends in New Project with them added."""

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pdf_helpers import (
    IMPORT_DPI,
    PAGE_WORDS,
    accept_project_files_dialog,
    pdf_window,  # noqa: F401
    source_pdf,  # noqa: F401
)
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import pdf_worker
from scantailor.app.ui.dialogs import project_files
from scantailor.app.ui.dialogs.pdf_import import (
    DEFAULT_IMPORT_DPI,
    ExistingPages,
    ExtractOptions,
    PdfExtractDialog,
)
from scantailor.app.ui.dialogs.pdf_progress import PdfProgressDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.pdf_worker import PdfSource
from scantailor.core.pdf_import import page_count
from scantailor.core.pdf_pool import PdfCancelled
from scantailor.core.pdf_source import SOURCE_FILE_NAME

if TYPE_CHECKING:
    from conftest import FileDialogStub


def test_extract_prompt_offers_all_pages_at_300_dpi_by_default(
    source_pdf: Path,  # noqa: F811
    qtbot: QtBot,
):
    dialog = PdfExtractDialog(source_pdf, 2, source_pdf.parent / "book_pages")
    qtbot.addWidget(dialog)

    assert dialog.options() == ExtractOptions(1, None, DEFAULT_IMPORT_DPI)


def test_a_page_range_that_starts_after_it_ends_disables_extract(
    source_pdf: Path,  # noqa: F811
    qtbot: QtBot,
):
    dialog = PdfExtractDialog(source_pdf, 2, source_pdf.parent / "book_pages")
    qtbot.addWidget(dialog)
    dialog._page_range.setChecked(True)
    dialog._first.setValue(2)
    dialog._last.setValue(1)

    ok = dialog._buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled()

    dialog._last.setValue(2)
    assert ok.isEnabled()


def test_options_carry_the_range_and_the_resolution(
    source_pdf: Path,  # noqa: F811
    qtbot: QtBot,
):
    dialog = PdfExtractDialog(source_pdf, 2, source_pdf.parent / "book_pages")
    qtbot.addWidget(dialog)
    dialog._page_range.setChecked(True)
    dialog._first.setValue(2)
    dialog._last.setValue(2)
    dialog._dpi.setValue(150)

    assert dialog.options() == ExtractOptions(2, 2, 150)


def test_open_pdf_action_needs_no_project(main_window: MainWindowManager):
    assert main_window.actionOpenPdf.isEnabled()
    assert not main_window.actionExportPdf.isEnabled()


def test_opening_a_two_page_pdf_ends_in_a_project_of_its_two_pages(
    pdf_window: MainWindowManager,  # noqa: F811
    source_pdf: Path,  # noqa: F811
):
    project = pdf_window.project
    assert project is not None

    assert len(pdf_window.pages()) == len(PAGE_WORDS)
    assert pdf_window.current_page == pdf_window.pages()[0]
    for info in project.images:
        assert info.metadata.dpi.horizontal == IMPORT_DPI
        assert info.id.file_path.parent == source_pdf.parent / "book_pages"
        assert info.id.file_path.is_file()


def test_output_folder_defaults_to_a_folder_next_to_the_pdf(
    pdf_window: MainWindowManager,  # noqa: F811
    source_pdf: Path,  # noqa: F811
):
    assert pdf_window.project is not None
    assert (
        pdf_window.project.output_directory == source_pdf.parent / "book_pages" / "out"
    )
    assert pdf_window.project.output_directory.is_dir()


def test_project_remembers_source_pdf_and_page_mapping(
    pdf_window: MainWindowManager,  # noqa: F811
    source_pdf: Path,  # noqa: F811
):
    project = pdf_window.project
    assert project is not None
    assert len(project.pdf_sources) == 1
    source = project.pdf_sources[0]

    assert source.pdf_path == source_pdf
    assert source.dpi == IMPORT_DPI
    assert source.pages_dir == source_pdf.parent / "book_pages"
    assert [source.page_index(info.id.file_path) for info in project.images] == [0, 1]
    assert not (source_pdf.parent / "book_pages" / SOURCE_FILE_NAME).exists()
    assert pdf_window.actionExportPdf.isEnabled()


def test_output_dpi_defaults_to_the_import_dpi(pdf_window: MainWindowManager):  # noqa: F811
    default = pdf_window.stages.output_filter.settings.get_default_dpi()

    assert (default.horizontal, default.vertical) == (IMPORT_DPI, IMPORT_DPI)


def test_saved_project_still_knows_its_source_pdf_when_reopened(
    pdf_window: MainWindowManager,  # noqa: F811
    file_dialogs: FileDialogStub,
    tmp_path: Path,
):
    file_dialogs.save_file = str(tmp_path / "book.ScanTailor")
    assert pdf_window.save_project_as()
    pdf_window.closeProject()
    assert pdf_window.pdf_sources == []

    assert pdf_window.open_project_file(tmp_path / "book.ScanTailor")

    assert len(pdf_window.pdf_sources) == 1
    assert pdf_window.pdf_sources[0].dpi == IMPORT_DPI


def test_project_saved_with_a_sidecar_mapping_still_finds_its_source_pdf(
    pdf_window: MainWindowManager,  # noqa: F811
    file_dialogs: FileDialogStub,
    tmp_path: Path,
):
    source = pdf_window.pdf_sources[0]
    project_file = tmp_path / "book.ScanTailor"
    file_dialogs.save_file = str(project_file)
    assert pdf_window.save_project_as()
    pdf_window.closeProject()
    data = json.loads(project_file.read_text(encoding="utf-8"))
    del data["pdf_sources"]
    project_file.write_text(json.dumps(data), encoding="utf-8")
    pages_dir = source.pdf_path.parent / "book_pages"
    (pages_dir / SOURCE_FILE_NAME).write_text(
        source.model_dump_json(exclude={"pages_dir"}), encoding="utf-8"
    )

    assert pdf_window.open_project_file(project_file)

    assert pdf_window.pdf_sources == [source]


def test_cancelling_the_extract_prompt_creates_no_project(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal,
    source_pdf: Path,  # noqa: F811
):
    file_dialogs.open_file = str(source_pdf)
    respond_to_modal(accept=False)

    main_window.actionOpenPdf.trigger()

    assert main_window.project is None
    assert not (source_pdf.parent / "book_pages").exists()


def test_cancelling_the_file_dialog_does_nothing(
    main_window: MainWindowManager, file_dialogs: FileDialogStub
):
    file_dialogs.open_file = ""

    main_window.actionOpenPdf.trigger()

    assert main_window.project is None


def test_cancelling_a_running_extraction_discards_the_pages(
    qtbot: QtBot,
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal,
    source_pdf: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
):
    started = threading.Event()
    release = threading.Event()

    def slow_import(
        pdf_path: Path,
        out_dir: Path,
        dpi: int,
        *,
        cancel: threading.Event,
        **_kwargs: object,
    ) -> list[Path]:
        started.set()
        assert release.wait(20)
        if cancel.is_set():
            raise PdfCancelled
        return []

    def cancel_once_running() -> None:
        if not started.is_set():
            QtCore.QTimer.singleShot(10, cancel_once_running)
            return
        progress = main_window.ui.findChild(PdfProgressDialog)
        assert progress is not None
        progress.reject()
        release.set()

    monkeypatch.setattr("scantailor.app.ui.pdf_worker.import_pdf", slow_import)
    file_dialogs.open_file = str(source_pdf)
    respond_to_modal(accept=True)
    QtCore.QTimer.singleShot(10, cancel_once_running)

    main_window.actionOpenPdf.trigger()

    assert main_window.project is None
    assert not (source_pdf.parent / "book_pages" / "page_0001.png").exists()
    assert not (source_pdf.parent / "book_pages" / SOURCE_FILE_NAME).exists()


def test_unreadable_pdf_reports_the_error_and_creates_no_project(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    warnings: list[str] = []
    monkeypatch.setattr(
        "scantailor.app.ui.dialogs.project_files.QtWidgets.QMessageBox.warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    file_dialogs.open_file = str(broken)

    main_window.actionOpenPdf.trigger()

    assert main_window.project is None
    assert len(warnings) == 1
    assert "not a readable PDF" in warnings[0]


def test_reopening_a_pdf_can_use_the_pages_it_already_has(
    pdf_window: MainWindowManager,  # noqa: F811
    file_dialogs: FileDialogStub,
    respond_to_modal,
    source_pdf: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
):
    pages_before = [p.name for p in (source_pdf.parent / "book_pages").glob("page_*")]
    pdf_window.modified = False
    pdf_window.closeProject()
    monkeypatch.setattr(
        project_files,
        "ask_existing_pages",
        lambda *_args: ExistingPages.USE,
    )
    monkeypatch.setattr(
        project_files,
        "import_pdf_pages",
        lambda *_args, **_kwargs: pytest.fail("the pages must not be rendered again"),
    )
    file_dialogs.open_file = str(source_pdf)
    respond_to_modal(accept=True)
    answered = accept_project_files_dialog()

    pdf_window.actionOpenPdf.trigger()

    assert answered == [True]
    assert sorted(
        info.id.file_path.name for info in pdf_window.project.images
    ) == sorted(pages_before)
    assert pdf_window.pdf_sources[0].dpi == IMPORT_DPI


def test_rendering_again_replaces_the_pages_and_leaves_other_files(
    pdf_window: MainWindowManager,  # noqa: F811
    file_dialogs: FileDialogStub,
    respond_to_modal,
    source_pdf: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
):
    folder = source_pdf.parent / "book_pages"
    stale = folder / "page_0009.png"
    stale.write_bytes(b"stale")
    notes = folder / "notes.txt"
    notes.write_text("keep", encoding="utf-8")
    pdf_window.modified = False
    pdf_window.closeProject()
    monkeypatch.setattr(
        project_files,
        "ask_existing_pages",
        lambda *_args: ExistingPages.RENDER_AGAIN,
    )
    file_dialogs.open_file = str(source_pdf)
    respond_to_modal(accept=True)
    answered = accept_project_files_dialog()

    pdf_window.actionOpenPdf.trigger()

    assert answered == [True]
    assert not stale.exists()
    assert notes.read_text(encoding="utf-8") == "keep"
    assert page_count(source_pdf) == len(PAGE_WORDS)
    assert (folder / "page_0002.png").is_file()


def test_the_extraction_uses_the_resolution_chosen_in_the_prompt(
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal,
    source_pdf: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
):
    rendered_at: list[int] = []
    real_import = pdf_worker.import_pdf_pages

    def recording_import(pdf_path, out_dir, dpi, report, cancel, **kwargs):
        rendered_at.append(dpi)
        return real_import(pdf_path, out_dir, dpi, report, cancel, **kwargs)

    monkeypatch.setattr(project_files, "import_pdf_pages", recording_import)
    file_dialogs.open_file = str(source_pdf)
    respond_to_modal(accept=True)
    answered = accept_project_files_dialog()

    main_window.actionOpenPdf.trigger()

    assert answered == [True]
    assert rendered_at == [DEFAULT_IMPORT_DPI]


def test_source_lookup_ignores_folders_without_a_mapping(tmp_path: Path):
    assert PdfSource.find([tmp_path / "a.png"]) == []


def test_source_lookup_survives_a_damaged_mapping_file(tmp_path: Path):
    (tmp_path / SOURCE_FILE_NAME).write_text("{", encoding="utf-8")

    assert PdfSource.find([tmp_path / "a.png"]) == []


def test_extraction_of_a_long_enough_pdf_renders_on_a_process_pool(
    source_pdf: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pool_sizes: list[int] = []
    real_spawn_pool = pdf_worker.spawn_pool

    def recording_spawn_pool(workers: int):
        pool_sizes.append(workers)
        return real_spawn_pool(min(workers, 2))

    monkeypatch.setattr(pdf_worker, "_POOL_MIN_PAGES", 2)
    monkeypatch.setattr(pdf_worker, "spawn_pool", recording_spawn_pool)

    result = pdf_worker.import_pdf_pages(
        source_pdf, tmp_path / "pages", 100, lambda *_: None, threading.Event()
    )

    assert len(pool_sizes) == 1
    assert [image.name for image in result.images] == ["page_0001.png", "page_0002.png"]
