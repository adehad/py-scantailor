import pathlib

from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot


def test_page_images_are_written_as_600x800_pngs_with_dark_blobs(page_images):
    assert len(page_images.paths) == 3
    for path in page_images.paths:
        image = QtGui.QImage(str(path))
        assert (image.width(), image.height()) == (600, 800)
        assert image.pixelColor(0, 0).lightness() == 255
        assert image.pixelColor(*page_images.ink_point).lightness() < 100


def test_isolated_settings_write_under_tmp_path(
    isolated_settings: pathlib.Path, tmp_path: pathlib.Path
):
    settings = QtCore.QSettings()
    settings.setValue("probe", 1)
    settings.sync()

    assert tmp_path in pathlib.Path(settings.fileName()).parents


def test_respond_to_modal_accepts_blocking_dialog(qtbot: QtBot, respond_to_modal):
    dialog = QtWidgets.QDialog()
    qtbot.addWidget(dialog)
    respond_to_modal(accept=True)

    assert dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted


def test_respond_to_modal_reports_a_response_that_met_no_modal(
    qtbot: QtBot, respond_to_modal
):
    respond_to_modal(accept=True, timeout_ms=50)

    qtbot.waitUntil(lambda: respond_to_modal.unanswered == 1, timeout=2000)

    assert respond_to_modal.finish() == 1
    respond_to_modal.forget_unanswered()


def test_respond_to_modal_reports_a_response_still_waiting_at_the_end(
    respond_to_modal,
):
    respond_to_modal(accept=True, timeout_ms=60_000)

    assert respond_to_modal.finish() == 1
    respond_to_modal.forget_unanswered()


def test_respond_to_modal_counts_nothing_when_the_dialog_was_answered(
    qtbot: QtBot, respond_to_modal
):
    dialog = QtWidgets.QDialog()
    qtbot.addWidget(dialog)
    respond_to_modal(accept=True)

    dialog.exec()

    assert respond_to_modal.finish() == 0


def test_respond_to_modal_rejects_blocking_dialog(qtbot: QtBot, respond_to_modal):
    dialog = QtWidgets.QDialog()
    qtbot.addWidget(dialog)
    respond_to_modal(accept=False)

    assert dialog.exec() == QtWidgets.QDialog.DialogCode.Rejected
