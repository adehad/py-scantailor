"""The main window hands stored output records to a batch and keeps the new ones."""

from pathlib import Path
from typing import Any

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import MainWindow
from scantailor.app.ui.dialogs.batch_progress import BatchProgressDialog
from scantailor.core.batch import BatchResult
from scantailor.core.models import ImageId, PageId
from scantailor.core.project import FileStamp, PageOutputRecord
from scantailor.core.stage_sequence import FilterStage


def _record(page: Any, digest: str) -> PageOutputRecord:
    return PageOutputRecord(
        page=page,
        params_hash=digest,
        source=FileStamp(size=1, mtime_ns=2),
        files={"page.tif": FileStamp(size=3, mtime_ns=4)},
    )


@pytest.fixture
def started(
    project_window: MainWindow.MainWindowManager, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    """Record the batch runner's start calls instead of running a batch."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        project_window.batch_runner,
        "start",
        lambda *_args, **kwargs: calls.append(kwargs),
    )
    return calls


def test_batch_from_output_receives_the_projects_stored_records(
    project_window: MainWindow.MainWindowManager, started: list[dict[str, Any]]
):
    project = project_window.project
    assert project is not None
    page = project.get_pages()[0]
    project.set_output_records({page: _record(page, "stored")})
    project_window._select_stage_row(int(FilterStage.OUTPUT))

    project_window.startBatchProcessing()

    assert started[0]["output_records"] == {page: _record(page, "stored")}


def test_finished_batch_stores_its_records_in_the_project(
    project_window: MainWindow.MainWindowManager,
):
    project = project_window.project
    assert project is not None
    first, second = project.get_pages()[:2]
    project.set_output_records(
        {first: _record(first, "old"), second: _record(second, "kept")}
    )

    project_window._on_batch_finished(
        BatchResult(output_records={first: _record(first, "new")})
    )

    assert project.output_record_map() == {
        first: _record(first, "new"),
        second: _record(second, "kept"),
    }


def _summary(dialog: BatchProgressDialog) -> str:
    label = dialog.findChild(QtWidgets.QLabel)
    assert label is not None
    return label.text()


def test_progress_summary_counts_kept_pages_apart_from_written_ones(qtbot: QtBot):
    dialog = BatchProgressDialog(3, renders=True)
    qtbot.addWidget(dialog)
    kept, written, failed = (
        PageId(image_id=ImageId(file_path=Path(name)))
        for name in ("a.png", "b.png", "c.png")
    )

    dialog.finish(
        BatchResult(
            written={kept: Path("a.tif"), written: Path("b.tif")},
            reused=[kept],
            failed={failed: "unreadable"},
        )
    )

    assert _summary(dialog) == "Done: 1 written, 1 kept, 1 failed"


def test_progress_summary_leaves_kept_out_when_every_page_rendered(qtbot: QtBot):
    dialog = BatchProgressDialog(1, renders=True)
    qtbot.addWidget(dialog)
    page = PageId(image_id=ImageId(file_path=Path("a.png")))

    dialog.finish(BatchResult(written={page: Path("a.tif")}))

    assert _summary(dialog) == "Done: 1 written, 0 failed"
