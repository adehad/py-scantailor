"""The batch progress dialog and the processing indicator."""

from pathlib import Path

from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.batch_progress import BatchProgressDialog
from scantailor.app.ui.widgets.processing_indicator import ProcessingIndicator
from scantailor.core.batch import BatchResult, PageDone, PageFailed
from scantailor.core.models import ImageId, PageId


def _page(name: str) -> PageId:
    return PageId(image_id=ImageId(file_path=Path(f"/scans/{name}.png")))


def test_each_reported_page_adds_a_line_and_advances_the_bar(qtbot: QtBot):
    dialog = BatchProgressDialog(total=2)
    qtbot.addWidget(dialog)
    first, second = _page("a"), _page("b")

    dialog.add_result(PageDone(first, Path("/out/a.tif")))
    dialog.add_result(PageFailed(second, "bad file"))

    lines = dialog.result_lines()
    assert lines[0] == "a.png: a.tif"
    assert "b.png" in lines[1]
    assert "bad file" in lines[1]
    assert dialog.reported == 2
    bar = dialog.findChild(QtWidgets.QProgressBar)
    assert bar is not None
    assert bar.value() == 2


def test_finish_summarises_the_outcome_and_turns_stop_into_close(qtbot: QtBot):
    dialog = BatchProgressDialog(total=3)
    qtbot.addWidget(dialog)
    result = BatchResult(
        written={_page("a"): Path("/out/a.tif")},
        failed={_page("b"): "bad"},
        cancelled=[_page("c")],
    )

    dialog.finish(result)

    labels = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]
    assert any("1 written" in text and "1 failed" in text for text in labels)
    assert any("1 not started" in text for text in labels)
    button = dialog.findChild(QtWidgets.QPushButton)
    assert button is not None
    assert button.text() == "Close"
    assert not dialog.is_running


def test_stop_button_requests_a_stop_once_and_waits_for_the_result(qtbot: QtBot):
    dialog = BatchProgressDialog(total=3)
    qtbot.addWidget(dialog)
    button = dialog.findChild(QtWidgets.QPushButton)
    assert button is not None

    with qtbot.waitSignal(dialog.stop_requested, timeout=1000):
        button.click()

    assert not button.isEnabled()
    assert dialog.is_running


def test_closing_a_running_dialog_asks_for_a_stop_instead_of_closing(qtbot: QtBot):
    dialog = BatchProgressDialog(total=1)
    qtbot.addWidget(dialog)
    dialog.show()

    with qtbot.waitSignal(dialog.stop_requested, timeout=1000):
        dialog.reject()

    assert dialog.isVisible()


def test_finished_dialog_closes(qtbot: QtBot):
    dialog = BatchProgressDialog(total=1)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.finish(BatchResult())

    dialog.reject()

    assert not dialog.isVisible()


def test_indicator_is_hidden_until_activated(qtbot: QtBot):
    indicator = ProcessingIndicator()
    qtbot.addWidget(indicator)

    assert not indicator.is_active
    assert indicator.isHidden()

    indicator.set_active(True)
    assert indicator.is_active
    assert not indicator.isHidden()

    indicator.set_active(False)
    assert not indicator.is_active
    assert indicator.isHidden()
