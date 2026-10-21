"""A batch runs up to the selected stage; only a batch from Output renders."""

from pathlib import Path
from typing import Any

import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import MainWindow
from scantailor.app.ui.dialogs.batch_progress import BatchProgressDialog
from scantailor.core.batch import BatchResult, PageDetected, PageFailed
from scantailor.core.models import ImageId, PageId
from scantailor.core.stage_sequence import FilterStage


def _page(name: str) -> PageId:
    return PageId(image_id=ImageId(file_path=Path(name)))


def _status(dialog: BatchProgressDialog) -> str:
    label = dialog.findChild(QtWidgets.QLabel)
    assert label is not None
    return label.text()


def _bar(dialog: BatchProgressDialog) -> int:
    bar = dialog.findChild(QtWidgets.QProgressBar)
    assert bar is not None
    return bar.value()


@pytest.fixture
def started(
    project_window: MainWindow.MainWindowManager, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    """Record the batch runner's start calls instead of running a batch."""
    calls: list[dict[str, Any]] = []

    def fake_start(*_args: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(project_window.batch_runner, "start", fake_start)
    return calls


class TestDetectionOnlyProgressWindow:
    def test_it_never_switches_to_rendering(self, qtbot: QtBot):
        dialog = BatchProgressDialog(2, renders=False)
        qtbot.addWidget(dialog)

        dialog.add_detected(PageDetected(_page("a.png")))
        dialog.add_detected(PageDetected(_page("b.png")))

        assert dialog.phase == "detecting"
        assert _status(dialog) == "Detecting 2/2"

    def test_the_bar_counts_detected_and_failed_pages(self, qtbot: QtBot):
        dialog = BatchProgressDialog(3, renders=False)
        qtbot.addWidget(dialog)

        dialog.add_detected(PageDetected(_page("a.png")))
        dialog.add_result(PageFailed(_page("b.png"), "unreadable"))

        assert _bar(dialog) == 2

    def test_the_summary_counts_processed_pages(self, qtbot: QtBot):
        dialog = BatchProgressDialog(2, renders=False)
        qtbot.addWidget(dialog)
        result = BatchResult(
            detected=[_page("a.png")], failed={_page("b.png"): "unreadable"}
        )

        dialog.finish(result)

        assert _status(dialog) == "Done: 1 processed, 1 failed"


class TestPlayButtonStage:
    @pytest.mark.parametrize(
        "stage", [FilterStage.DESKEW, FilterStage.SELECT_CONTENT, FilterStage.OUTPUT]
    )
    def test_batch_runs_up_to_the_selected_stage(
        self,
        project_window: MainWindow.MainWindowManager,
        started: list[dict[str, Any]],
        stage: FilterStage,
    ):
        project_window._select_stage_row(int(stage))

        project_window.startBatchProcessing()

        assert [call["end_stage"] for call in started] == [stage]

    @pytest.mark.usefixtures("started")
    def test_a_batch_before_output_opens_a_detection_only_window(
        self, project_window: MainWindow.MainWindowManager
    ):
        project_window._select_stage_row(int(FilterStage.PAGE_LAYOUT))

        project_window.startBatchProcessing()
        dialog = project_window.batch_dialog
        assert dialog is not None
        pages = project_window.pages()
        for page in pages:
            dialog.add_detected(PageDetected(page))

        assert dialog.phase == "detecting"
        assert _status(dialog) == f"Detecting {len(pages)}/{len(pages)}"

    def test_export_pdf_processes_missing_pages_through_output(
        self,
        project_window: MainWindow.MainWindowManager,
        started: list[dict[str, Any]],
    ):
        project_window._select_stage_row(int(FilterStage.SELECT_CONTENT))
        project_window.run_batch_prompt = lambda _unprocessed, _total: True

        project_window.exportPdf()

        assert [call["end_stage"] for call in started] == [FilterStage.OUTPUT]
