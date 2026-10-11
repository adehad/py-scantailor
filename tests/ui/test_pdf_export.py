"""File > Export PDF writes the processed pages of the project to one PDF."""

import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from pdf_helpers import (  # noqa: F401
    PAGE_WORDS,
    image_filters,
    page_count,
    page_text,
    pdf_window,
    source_pdf,
    write_processed_outputs,
)
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.pdf_export import PdfExportDialog
from scantailor.app.ui.MainWindow import MainWindowManager


class _Dialogs:
    """Export dialog factory that records its arguments and adjusts the dialog."""

    def __init__(self, codec_index: int = 0, carry_text: bool = True) -> None:
        self.codec_index = codec_index
        self.carry_text = carry_text
        self.created: list[tuple[Path, bool]] = []

    def __call__(
        self, default_path: Path, can_carry_text: bool, parent: QtWidgets.QWidget
    ) -> PdfExportDialog:
        self.created.append((default_path, can_carry_text))
        dialog = PdfExportDialog(
            default_path, can_carry_text=can_carry_text, parent=parent
        )
        codec = dialog.findChild(QtWidgets.QComboBox)
        assert codec is not None
        codec.setCurrentIndex(self.codec_index)
        carry = dialog.findChild(QtWidgets.QCheckBox)
        assert carry is not None
        if carry.isEnabled():
            carry.setChecked(self.carry_text)
        QtWidgets.QApplication.processEvents()
        return dialog


@pytest.fixture
def respond(respond_to_modal: Callable[..., None]) -> Callable[..., None]:
    return respond_to_modal


def _export(
    qtbot: QtBot,
    window: MainWindowManager,
    respond: Callable[..., None],
    dialogs: _Dialogs,
) -> None:
    window.pdf_export_dialog_factory = dialogs
    respond(accept=True)
    with qtbot.waitSignal(window.pdf_job.finished, timeout=60000):
        window.actionExportPdf.trigger()


def test_export_writes_one_jbig2_page_per_project_page(
    qtbot: QtBot,
    pdf_window: MainWindowManager,  # noqa: F811
    respond: Callable[..., None],
    source_pdf: Path,  # noqa: F811
):
    write_processed_outputs(pdf_window)
    out = source_pdf.parent / "book_processed.pdf"

    _export(qtbot, pdf_window, respond, _Dialogs())

    assert page_count(out) == len(PAGE_WORDS)
    assert image_filters(out) == ["/JBIG2Decode"] * len(PAGE_WORDS)
    assert pdf_window.last_pdf_export is not None
    assert pdf_window.last_pdf_export.pages == len(PAGE_WORDS)
    assert not out.with_name(out.name + ".part").exists()


def test_export_carries_the_text_layer_of_a_pdf_project(
    qtbot: QtBot,
    pdf_window: MainWindowManager,  # noqa: F811
    respond: Callable[..., None],
    source_pdf: Path,  # noqa: F811
):
    write_processed_outputs(pdf_window)

    _export(qtbot, pdf_window, respond, _Dialogs())

    out = source_pdf.parent / "book_processed.pdf"
    assert pdf_window.last_pdf_export is not None
    assert pdf_window.last_pdf_export.text_pages == len(PAGE_WORDS)
    for index, word in enumerate(PAGE_WORDS):
        assert word in page_text(out, index)


def test_export_without_the_text_layer_option_leaves_the_pages_textless(
    qtbot: QtBot,
    pdf_window: MainWindowManager,  # noqa: F811
    respond: Callable[..., None],
    source_pdf: Path,  # noqa: F811
):
    write_processed_outputs(pdf_window)

    _export(qtbot, pdf_window, respond, _Dialogs(carry_text=False))

    assert pdf_window.last_pdf_export is not None
    assert pdf_window.last_pdf_export.text_pages == 0
    assert page_text(source_pdf.parent / "book_processed.pdf", 0).strip() == ""


def test_g4_codec_writes_ccitt_pages(
    qtbot: QtBot,
    pdf_window: MainWindowManager,  # noqa: F811
    respond: Callable[..., None],
    source_pdf: Path,  # noqa: F811
):
    write_processed_outputs(pdf_window)

    _export(qtbot, pdf_window, respond, _Dialogs(codec_index=1))

    out = source_pdf.parent / "book_processed.pdf"
    assert image_filters(out) == ["/CCITTFaxDecode"] * len(PAGE_WORDS)


def test_text_option_is_offered_only_for_projects_made_from_a_pdf(
    qtbot: QtBot,
    pdf_window: MainWindowManager,  # noqa: F811
    respond: Callable[..., None],
):
    pdf_dialogs, plain_dialogs = _Dialogs(), _Dialogs()
    write_processed_outputs(pdf_window)
    pdf_window.pdf_export_dialog_factory = pdf_dialogs
    respond(accept=False)
    pdf_window.actionExportPdf.trigger()
    assert pdf_dialogs.created[0][1] is True

    # Pages of plain image files have no source PDF.
    pdf_window.pdf_source = None
    pdf_window.pdf_export_dialog_factory = plain_dialogs
    respond(accept=False)
    pdf_window.actionExportPdf.trigger()

    assert plain_dialogs.created[0][1] is False


def test_export_is_refused_while_pages_are_unprocessed(
    pdf_window: MainWindowManager,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
):
    asked: list[tuple[int, int]] = []
    pdf_window.run_batch_prompt = lambda unprocessed, total: (
        asked.append((unprocessed, total)) or False
    )
    dialogs = _Dialogs()
    pdf_window.pdf_export_dialog_factory = dialogs

    pdf_window.actionExportPdf.trigger()

    assert asked == [(2, 2)]
    assert dialogs.created == []
    assert not pdf_window.pdf_job.is_running


def test_only_pages_without_output_are_offered_for_batch_processing(
    pdf_window: MainWindowManager,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
):
    write_processed_outputs(pdf_window)
    first = pdf_window.pages()[0]
    names_dir = pdf_window.project.output_directory  # type: ignore[union-attr]
    next(names_dir.glob("page_0001*.tif")).unlink()
    started: list[list] = []
    monkeypatch.setattr(pdf_window, "_start_batch", started.append)
    pdf_window.run_batch_prompt = lambda _unprocessed, _total: True

    pdf_window.actionExportPdf.trigger()

    assert started == [[first]]


def test_declining_the_batch_offer_starts_nothing(
    pdf_window: MainWindowManager,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
):
    started: list[list] = []
    monkeypatch.setattr(pdf_window, "_start_batch", started.append)
    pdf_window.run_batch_prompt = lambda _unprocessed, _total: False

    pdf_window.actionExportPdf.trigger()

    assert started == []


def test_cancelling_an_export_leaves_no_file_behind(
    qtbot: QtBot,
    pdf_window: MainWindowManager,  # noqa: F811
    respond: Callable[..., None],
    source_pdf: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
):
    write_processed_outputs(pdf_window)
    reading = threading.Event()
    release = threading.Event()

    def blocked_text_layer(jobs: list[object], **_kwargs: object) -> list[None]:
        reading.set()
        assert release.wait(20)
        return [None] * len(jobs)

    monkeypatch.setattr(
        "scantailor.app.ui.pdf_worker.build_text_layers", blocked_text_layer
    )
    pdf_window.pdf_export_dialog_factory = _Dialogs()
    respond(accept=True)
    pdf_window.actionExportPdf.trigger()
    assert reading.wait(20)
    progress = pdf_window._pdf_progress
    assert progress is not None
    qtbot.waitUntil(
        lambda: progress.status_text.startswith("Reading text"), timeout=5000
    )

    with qtbot.waitSignal(pdf_window.pdf_job.cancelled, timeout=20000):
        progress.reject()
        release.set()

    out = source_pdf.parent / "book_processed.pdf"
    assert not out.exists()
    assert not out.with_name(out.name + ".part").exists()
    assert pdf_window.last_pdf_export is None
    assert pdf_window._pdf_progress is None


def test_cancelled_export_keeps_an_earlier_file_at_the_destination(
    qtbot: QtBot,
    pdf_window: MainWindowManager,  # noqa: F811
    respond: Callable[..., None],
    source_pdf: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
):
    write_processed_outputs(pdf_window)
    out = source_pdf.parent / "book_processed.pdf"
    out.write_bytes(b"earlier")
    writing = threading.Event()
    release = threading.Event()

    def slow_export(pages: object, path: Path, **_kwargs: object) -> None:
        Path(path).write_bytes(b"partial")
        writing.set()
        assert release.wait(20)

    monkeypatch.setattr("scantailor.app.ui.pdf_worker.export_pdf", slow_export)
    pdf_window.pdf_export_dialog_factory = _Dialogs()
    respond(accept=True)
    pdf_window.actionExportPdf.trigger()
    assert writing.wait(20)
    progress = pdf_window._pdf_progress
    assert progress is not None

    with qtbot.waitSignal(pdf_window.pdf_job.cancelled, timeout=20000):
        progress.reject()
        release.set()

    assert out.read_bytes() == b"earlier"
    assert not out.with_name(out.name + ".part").exists()


def test_export_failure_is_reported_to_the_user(
    qtbot: QtBot,
    pdf_window: MainWindowManager,  # noqa: F811
    respond: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
):
    write_processed_outputs(pdf_window)
    warnings: list[str] = []
    monkeypatch.setattr(
        "scantailor.app.ui.MainWindow.QMessageBox.warning",
        lambda _parent, _title, text: warnings.append(text),
    )

    def failing_export(*_args: object, **_kwargs: object) -> None:
        raise ValueError("disk full")

    monkeypatch.setattr("scantailor.app.ui.pdf_worker.export_pdf", failing_export)
    pdf_window.pdf_export_dialog_factory = _Dialogs()
    respond(accept=True)

    with qtbot.waitSignal(pdf_window.pdf_job.failed, timeout=20000):
        pdf_window.actionExportPdf.trigger()

    assert warnings == ["ValueError: disk full"]


def test_dialog_disables_text_option_without_a_source_pdf(qtbot: QtBot, tmp_path: Path):
    dialog = PdfExportDialog(tmp_path / "x.pdf", can_carry_text=False)
    qtbot.addWidget(dialog)

    options = dialog.options()
    carry = dialog.findChild(QtWidgets.QCheckBox)

    assert carry is not None
    assert not carry.isEnabled()
    assert not options.carry_text
    assert options.bilevel_codec == "jbig2"
    assert options.jpeg_quality == 85


def test_dialog_adds_the_pdf_suffix_to_the_chosen_path(qtbot: QtBot, tmp_path: Path):
    dialog = PdfExportDialog(tmp_path / "result", can_carry_text=True)
    qtbot.addWidget(dialog)

    assert dialog.options().out_path == tmp_path / "result.pdf"


@pytest.mark.slow
def test_pdf_round_trip_through_a_real_batch_keeps_pages_and_text(
    qtbot: QtBot,
    pdf_window: MainWindowManager,  # noqa: F811
    respond: Callable[..., None],
    source_pdf: Path,  # noqa: F811
):
    pdf_window.goFirstPage()
    with qtbot.waitSignal(pdf_window.batch_runner.finished, timeout=240000):
        pdf_window.startBatchProcessing()
    assert pdf_window.last_batch_result is not None
    assert len(pdf_window.last_batch_result.written) == len(PAGE_WORDS)

    _export(qtbot, pdf_window, respond, _Dialogs())

    out = source_pdf.parent / "book_processed.pdf"
    assert page_count(out) == len(PAGE_WORDS)
    assert image_filters(out) == ["/JBIG2Decode"] * len(PAGE_WORDS)
    for index, word in enumerate(PAGE_WORDS):
        assert word in page_text(out, index)


def test_export_refuses_to_overwrite_its_own_source_pdf(
    tmp_path: Path,
    source_pdf: Path,  # noqa: F811
):
    from scantailor.app.ui.pdf_worker import (
        PagePlan,
        PdfExportRequest,
        PdfSource,
        export_pdf_pages,
    )

    image = tmp_path / "out.png"
    image.write_bytes(b"")
    before = source_pdf.read_bytes()
    request = PdfExportRequest(
        [PagePlan(image, (300.0, 300.0))],
        source_pdf,
        source=PdfSource(pdf_path=source_pdf, dpi=300),
    )

    with pytest.raises(ValueError, match="source PDF"):
        export_pdf_pages(request, lambda *_: None, threading.Event())

    assert source_pdf.read_bytes() == before
