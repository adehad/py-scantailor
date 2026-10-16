"""How a PdfJob disposes of its worker once the job's thread has ended."""

import threading

from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from scantailor.app.ui.pdf_worker import PdfJob


def _record_destroying_thread(job: PdfJob) -> list[threading.Thread]:
    destroyed_on: list[threading.Thread] = []
    assert job._worker is not None
    job._worker.destroyed.connect(
        lambda: destroyed_on.append(threading.current_thread()),
        QtCore.Qt.ConnectionType.DirectConnection,
    )
    return destroyed_on


def test_the_worker_is_destroyed_on_the_gui_thread_when_the_job_finishes(
    qtbot: QtBot,
):
    """A worker destroyed on its own ending thread races Python for the object."""
    job = PdfJob()

    with qtbot.waitSignal(job.finished, timeout=10_000) as finished:
        job.start(lambda _report, _cancel: "result")
        destroyed_on = _record_destroying_thread(job)

    assert finished.args == ["result"]
    assert destroyed_on == [threading.main_thread()]
    assert not job.is_running


def test_waiting_for_the_job_also_destroys_the_worker_on_the_gui_thread(
    qtbot: QtBot,
):
    job = PdfJob()
    job.start(lambda _report, _cancel: None)
    destroyed_on = _record_destroying_thread(job)

    assert job.wait(10_000)

    assert destroyed_on == [threading.main_thread()]
