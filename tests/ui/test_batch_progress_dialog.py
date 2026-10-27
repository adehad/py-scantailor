"""The batch progress window shows the detection phase, then the rendering phase."""

from pathlib import Path

from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.dialogs.batch_progress import BatchProgressDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.batch import PageDetected, PageDone, PageFailed
from scantailor.core.models import ImageId, PageId


def _page(name: str) -> PageId:
    return PageId(image_id=ImageId(file_path=Path(name)))


def _status(dialog: BatchProgressDialog) -> str:
    label = dialog.findChild(QtWidgets.QLabel)
    assert label is not None
    return label.text()


def test_dialog_starts_in_the_detecting_phase(qtbot: QtBot):
    dialog = BatchProgressDialog(3)
    qtbot.addWidget(dialog)

    assert dialog.phase == "detecting"
    assert _status(dialog) == "Detecting 0/3"


def test_a_new_total_from_page_split_keeps_the_detecting_phase_until_all_are_measured(
    qtbot: QtBot,
):
    dialog = BatchProgressDialog(1)
    qtbot.addWidget(dialog)

    dialog.set_total(2)
    dialog.add_detected(PageDetected(_page("spread.png")))

    assert dialog.phase == "detecting"
    assert _status(dialog) == "Detecting 1/2"


def test_a_page_that_failed_before_page_split_listed_the_pages_stays_in_the_total(
    qtbot: QtBot,
):
    dialog = BatchProgressDialog(2)
    qtbot.addWidget(dialog)
    dialog.add_result(PageFailed(_page("unreadable.png"), "unreadable"))

    dialog.set_total(2)

    bar = dialog.findChild(QtWidgets.QProgressBar)
    assert bar is not None
    assert bar.maximum() == 3
    assert _status(dialog) == "Detecting 0/3"


def test_each_detected_page_advances_the_detecting_count(qtbot: QtBot):
    dialog = BatchProgressDialog(3)
    qtbot.addWidget(dialog)

    dialog.add_detected(PageDetected(_page("a.png")))
    dialog.add_detected(PageDetected(_page("b.png")))

    assert _status(dialog) == "Detecting 2/3"
    assert dialog.phase == "detecting"


def test_detecting_every_page_switches_to_rendering(qtbot: QtBot):
    dialog = BatchProgressDialog(2)
    qtbot.addWidget(dialog)

    dialog.add_detected(PageDetected(_page("a.png")))
    dialog.add_detected(PageDetected(_page("b.png")))

    assert dialog.phase == "rendering"
    assert _status(dialog) == "Rendering 0/2"


def test_rendered_pages_advance_the_rendering_count(qtbot: QtBot):
    dialog = BatchProgressDialog(2)
    qtbot.addWidget(dialog)
    for name in ("a.png", "b.png"):
        dialog.add_detected(PageDetected(_page(name)))

    dialog.add_result(PageDone(_page("a.png"), Path("a.tif")))

    assert _status(dialog) == "Rendering 1/2"
    assert dialog.reported == 1


def test_a_page_that_failed_detection_does_not_end_the_detecting_phase(
    qtbot: QtBot,
):
    dialog = BatchProgressDialog(3)
    qtbot.addWidget(dialog)
    dialog.add_detected(PageDetected(_page("a.png")))

    dialog.add_result(PageFailed(_page("b.png"), "unreadable"))

    assert dialog.phase == "detecting"
    assert _status(dialog) == "Detecting 1/3"
    assert dialog.reported == 1


def test_first_finished_page_proves_detection_is_over(qtbot: QtBot):
    dialog = BatchProgressDialog(3)
    qtbot.addWidget(dialog)
    dialog.add_detected(PageDetected(_page("a.png")))
    dialog.add_detected(PageDetected(_page("b.png")))

    dialog.add_result(PageDone(_page("a.png"), Path("a.tif")))

    assert dialog.phase == "rendering"


def test_runner_detected_signal_reaches_the_open_dialog(
    qtbot: QtBot, main_window: MainWindowManager
):
    dialog = BatchProgressDialog(2, main_window.ui)
    qtbot.addWidget(dialog)
    main_window.batch_dialog = dialog

    main_window.batch_runner.detected.emit(PageDetected(_page("a.png")))

    assert _status(dialog) == "Detecting 1/2"
    main_window.batch_runner.detected.emit(PageDetected(_page("b.png")))
    assert _status(dialog) == "Rendering 0/2"
