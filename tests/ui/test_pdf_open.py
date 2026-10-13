"""File > Open PDF turns the pages of a PDF into a project."""

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pdf_helpers import IMPORT_DPI, PAGE_WORDS, pdf_window, source_pdf  # noqa: F401
from pytestqt.qtbot import QtBot

from scantailor.app.ui import pdf_worker
from scantailor.app.ui.dialogs.pdf_import import PdfImportDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.pdf_worker import SOURCE_FILE_NAME, PdfSource
from scantailor.core.pdf_pool import PdfCancelled

if TYPE_CHECKING:
    from conftest import FileDialogStub


def test_import_dialog_offers_300_dpi_by_default(source_pdf: Path, qtbot: QtBot):  # noqa: F811
    dialog = PdfImportDialog(source_pdf)
    qtbot.addWidget(dialog)

    assert dialog.dpi() == 300


def test_open_pdf_action_needs_no_project(main_window: MainWindowManager):
    assert main_window.actionOpenPdf.isEnabled()
    assert not main_window.actionExportPdf.isEnabled()


def test_opening_a_two_page_pdf_creates_a_two_page_project(
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
    source = pdf_window.pdf_source
    assert project is not None
    assert source is not None

    assert source.pdf_path == source_pdf
    assert source.dpi == IMPORT_DPI
    assert [source.page_index(info.id.file_path) for info in project.images] == [0, 1]
    assert (source_pdf.parent / "book_pages" / SOURCE_FILE_NAME).is_file()
    assert pdf_window.actionExportPdf.isEnabled()


def test_output_dpi_defaults_to_the_import_dpi(pdf_window: MainWindowManager):  # noqa: F811
    default = pdf_window.stages.output_filter.settings.get_default_dpi()

    assert (default.horizontal, default.vertical) == (IMPORT_DPI, IMPORT_DPI)


def test_saved_project_still_knows_its_source_pdf_when_reopened(
    pdf_window: MainWindowManager,  # noqa: F811
    file_dialogs: "FileDialogStub",
    tmp_path: Path,
):
    file_dialogs.save_file = str(tmp_path / "book.ScanTailor")
    assert pdf_window.save_project_as()
    pdf_window.closeProject()
    assert pdf_window.pdf_source is None

    assert pdf_window.open_project_file(tmp_path / "book.ScanTailor")

    assert pdf_window.pdf_source is not None
    assert pdf_window.pdf_source.dpi == IMPORT_DPI


def test_cancelling_the_dpi_dialog_creates_no_project(
    qtbot: QtBot,
    main_window: MainWindowManager,
    file_dialogs: "FileDialogStub",
    respond_to_modal,
    source_pdf: Path,  # noqa: F811
):
    file_dialogs.open_file = str(source_pdf)
    respond_to_modal(accept=False)

    main_window.actionOpenPdf.trigger()

    assert main_window.project is None
    assert not main_window.pdf_job.is_running
    assert not (source_pdf.parent / "book_pages").exists()


def test_cancelling_the_file_dialog_does_nothing(
    main_window: MainWindowManager, file_dialogs: "FileDialogStub"
):
    file_dialogs.open_file = ""

    main_window.actionOpenPdf.trigger()

    assert main_window.project is None
    assert not main_window.pdf_job.is_running


def test_cancelling_a_running_import_discards_the_pages(
    qtbot: QtBot,
    main_window: MainWindowManager,
    file_dialogs: "FileDialogStub",
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

    monkeypatch.setattr("scantailor.app.ui.pdf_worker.import_pdf", slow_import)
    file_dialogs.open_file = str(source_pdf)
    respond_to_modal(accept=True)
    main_window.actionOpenPdf.trigger()
    assert started.wait(20)
    progress = main_window._pdf_progress
    assert progress is not None

    with qtbot.waitSignal(main_window.pdf_job.cancelled, timeout=20000):
        progress.reject()
        release.set()

    assert main_window.project is None
    assert not (source_pdf.parent / "book_pages" / "page_0001.png").exists()
    assert not (source_pdf.parent / "book_pages" / SOURCE_FILE_NAME).exists()
    assert main_window._pdf_progress is None


def test_unreadable_pdf_reports_the_error_and_creates_no_project(
    qtbot: QtBot,
    main_window: MainWindowManager,
    file_dialogs: "FileDialogStub",
    respond_to_modal,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    warnings: list[str] = []
    monkeypatch.setattr(
        "scantailor.app.ui.MainWindow.QMessageBox.warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    file_dialogs.open_file = str(broken)
    respond_to_modal(accept=True)

    with qtbot.waitSignal(main_window.pdf_job.failed, timeout=20000):
        main_window.actionOpenPdf.trigger()

    assert main_window.project is None
    assert len(warnings) == 1
    assert "not a readable PDF" in warnings[0]


def test_source_lookup_ignores_folders_without_a_mapping(tmp_path: Path):
    assert PdfSource.find([tmp_path / "a.png"]) is None


def test_source_lookup_survives_a_damaged_mapping_file(tmp_path: Path):
    (tmp_path / SOURCE_FILE_NAME).write_text("{", encoding="utf-8")

    assert PdfSource.find([tmp_path / "a.png"]) is None


def test_import_of_a_long_enough_pdf_renders_on_a_process_pool(
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
