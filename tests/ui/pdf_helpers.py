"""Synthetic PDFs and processed outputs shared by the PDF UI tests."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pikepdf
import pypdfium2 as pdfium
import pytest
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.batch import RenderInfo, output_file_names
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi
from scantailor.core.transformation import ImageTransformation

if TYPE_CHECKING:
    from conftest import FileDialogStub

PAGE_WORDS = ["alpha", "bravo"]
IMPORT_DPI = 300
# 300x400 pt pages rendered at 300 dpi.
RENDERED_SIZE = (1250, 1667)


def write_text_pdf(path: Path) -> Path:
    """Write a PDF with one word of live text per page."""
    pdf = pikepdf.Pdf.new()
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Helvetica,
        )
    )
    for word in PAGE_WORDS:
        page = pdf.add_blank_page(page_size=(300, 400))
        page.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
        page.Contents = pdf.make_stream(
            f"BT /F1 32 Tf 60 200 Td ({word}) Tj ET".encode()
        )
    pdf.save(path)
    return path


def write_processed_outputs(window: MainWindowManager) -> None:
    """Write a bilevel output file for every page and record how it was made.

    Stands in for a batch run: the files carry a black bar, and the render info
    maps source pixels straight to output pixels.
    """
    project = window.project
    assert project is not None
    pages = project.get_pages()
    names = output_file_names(
        pages, project.layout_direction, project.output_file_label_map()
    )
    width, height = RENDERED_SIZE
    black = np.zeros((height, width), dtype=bool)
    black[100:140, 100:900] = True
    project.output_directory.mkdir(parents=True, exist_ok=True)
    for page in pages:
        write_image(
            project.output_directory / names[page], black, Dpi.uniform(IMPORT_DPI)
        )
        transformation = ImageTransformation.from_size(
            width, height, Dpi.uniform(IMPORT_DPI)
        )
        window._render_info[page] = RenderInfo(
            transformation, Dpi.uniform(IMPORT_DPI), False
        )


def image_filters(pdf_path: Path) -> list[str]:
    """Return the filter of the first image on every page of a PDF."""
    with pikepdf.open(pdf_path) as pdf:
        filters = []
        for page in pdf.pages:
            images = [
                xobject
                for xobject in page.Resources.XObject.values()
                if xobject.get("/Subtype") == "/Image"
            ]
            filters.append(str(images[0].Filter))
        return filters


def page_text(pdf_path: Path, index: int) -> str:
    """Return the text pdfium extracts from one page."""
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        return document[index].get_textpage().get_text_range()
    finally:
        document.close()


def page_count(pdf_path: Path) -> int:
    """Return the number of pages of a PDF."""
    with pikepdf.open(pdf_path) as pdf:
        return len(pdf.pages)


@pytest.fixture
def source_pdf(tmp_path: Path) -> Path:
    """A two-page PDF with live text, in its own folder."""
    folder = tmp_path / "pdfs"
    folder.mkdir()
    return write_text_pdf(folder / "book.pdf")


def accept_project_files_dialog(timeout_ms: int = 30000) -> list[bool]:
    """Accept the New Project dialog once Open PDF shows it, after the extraction.

    The extraction progress window is modal too, so a generic responder could
    answer that instead; this one waits for the files dialog specifically.

    Args:
        timeout_ms (int): How long to wait for the dialog to appear.

    Returns:
        list[bool]: Holds one entry, True once the dialog was accepted.
    """
    from PySide6 import QtCore, QtWidgets

    from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog

    answered: list[bool] = []
    deadline = QtCore.QDeadlineTimer(timeout_ms)

    def poll() -> None:
        modal = QtWidgets.QApplication.activeModalWidget()
        if isinstance(modal, ProjectFilesDialog):
            modal.accept()
            answered.append(True)
        elif not deadline.hasExpired() and not answered:
            QtCore.QTimer.singleShot(10, poll)

    QtCore.QTimer.singleShot(10, poll)
    return answered


@pytest.fixture
def pdf_window(
    qtbot: QtBot,
    main_window: MainWindowManager,
    file_dialogs: FileDialogStub,
    respond_to_modal: Callable[..., None],
    source_pdf: Path,
) -> MainWindowManager:
    """Main window with the two pages of ``source_pdf`` opened as a project."""
    file_dialogs.open_file = str(source_pdf)
    respond_to_modal(accept=True)
    answered = accept_project_files_dialog()
    main_window.actionOpenPdf.trigger()
    assert answered == [True]
    qtbot.waitUntil(lambda: main_window.project is not None, timeout=30000)
    return main_window
