"""The batch window's system load slider, and what a completed batch does."""

from typing import Any

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui import MainWindow
from scantailor.app.ui.dialogs.batch_progress import BatchProgressDialog
from scantailor.app.ui.panels import system_load
from scantailor.app.ui.panels.system_load import SystemLoadWidget
from scantailor.core.batch import BatchResult
from scantailor.core.stage_sequence import FilterStage


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


@pytest.fixture
def announcements(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record taskbar alerts and beeps instead of making them."""
    made: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QApplication, "alert", lambda *_args: made.append("alert")
    )
    monkeypatch.setattr(QtWidgets.QApplication, "beep", lambda: made.append("beep"))
    return made


@pytest.fixture
def eight_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the system load slider a range of 1 to 8 on any machine."""
    monkeypatch.setattr(system_load.os, "cpu_count", lambda: 8)


def _set_active_window(
    monkeypatch: pytest.MonkeyPatch, window: QtWidgets.QWidget | None
) -> None:
    monkeypatch.setattr(QtWidgets.QApplication, "activeWindow", lambda: window)


def _start(window: MainWindow.MainWindowManager) -> BatchProgressDialog:
    window._select_stage_row(int(FilterStage.SELECT_CONTENT))
    window.startBatchProcessing()
    dialog = window.batch_dialog
    assert dialog is not None
    return dialog


def _slider(dialog: BatchProgressDialog) -> QtWidgets.QSlider:
    slider = dialog.findChild(QtWidgets.QSlider, "slider")
    assert slider is not None
    return slider


@pytest.mark.usefixtures("eight_cores")
class TestSystemLoadInTheBatchWindow:
    def test_the_window_holds_the_slider_and_an_unchecked_beep_checkbox(
        self, qtbot: QtBot
    ):
        dialog = BatchProgressDialog(2)
        qtbot.addWidget(dialog)

        assert dialog.findChild(SystemLoadWidget) is not None
        beep = dialog.findChild(QtWidgets.QCheckBox, "beepWhenFinished")
        assert beep is not None
        assert not beep.isChecked()

    def test_the_slider_starts_at_the_saved_page_count(self, qtbot: QtBot):
        QtCore.QSettings().setValue(system_load._SETTINGS_KEY, 3)

        dialog = BatchProgressDialog(2)
        qtbot.addWidget(dialog)

        assert dialog.load == 3

    def test_moving_the_slider_is_announced_and_saved_for_the_next_batch(
        self, qtbot: QtBot
    ):
        dialog = BatchProgressDialog(2)
        qtbot.addWidget(dialog)

        with qtbot.waitSignal(dialog.load_changed) as blocker:
            _slider(dialog).setValue(2)

        assert blocker.args == [2]
        assert QtCore.QSettings().value(system_load._SETTINGS_KEY, type=int) == 2
        later = BatchProgressDialog(2)
        qtbot.addWidget(later)
        assert later.load == 2


@pytest.mark.usefixtures("eight_cores")
class TestWorkerLimit:
    def test_a_batch_starts_with_the_saved_page_count_on_a_full_size_pool(
        self,
        project_window: MainWindow.MainWindowManager,
        started: list[dict[str, Any]],
    ):
        QtCore.QSettings().setValue(system_load._SETTINGS_KEY, 2)

        _start(project_window)

        assert project_window.batch_runner.worker_limit == 2
        assert len(started) == 1
        assert started[0].get("max_workers") is None

    @pytest.mark.usefixtures("started")
    def test_moving_the_slider_during_a_batch_changes_the_running_limit(
        self, project_window: MainWindow.MainWindowManager
    ):
        dialog = _start(project_window)

        _slider(dialog).setValue(5)

        assert project_window.batch_runner.worker_limit == 5


@pytest.mark.usefixtures("started")
class TestCompletedBatch:
    def test_it_flashes_the_taskbar_when_no_window_of_the_app_is_active(
        self,
        project_window: MainWindow.MainWindowManager,
        announcements: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ):
        _start(project_window)
        _set_active_window(monkeypatch, None)

        project_window._on_batch_finished(BatchResult())

        assert announcements == ["alert"]

    def test_it_does_not_flash_the_taskbar_while_the_user_is_in_the_app(
        self,
        project_window: MainWindow.MainWindowManager,
        announcements: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ):
        _start(project_window)
        _set_active_window(monkeypatch, project_window.ui)

        project_window._on_batch_finished(BatchResult())

        assert announcements == []

    def test_it_beeps_when_beep_when_finished_is_checked(
        self,
        project_window: MainWindow.MainWindowManager,
        announcements: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ):
        dialog = _start(project_window)
        _set_active_window(monkeypatch, project_window.ui)
        dialog.beep_when_finished = True

        project_window._on_batch_finished(BatchResult())

        assert announcements == ["beep"]

    @pytest.mark.usefixtures("announcements")
    def test_the_beep_choice_carries_over_to_the_next_batch(
        self, project_window: MainWindow.MainWindowManager
    ):
        _start(project_window).beep_when_finished = True
        project_window._on_batch_finished(BatchResult())

        assert _start(project_window).beep_when_finished

    @pytest.mark.usefixtures("announcements")
    def test_it_returns_to_the_first_page(
        self, project_window: MainWindow.MainWindowManager
    ):
        pages = project_window.thumbnails.pages()
        project_window.goLastPage()
        assert project_window.current_page == pages[-1]
        _start(project_window)

        project_window._on_batch_finished(BatchResult())

        assert project_window.current_page == pages[0]

    def test_a_stopped_batch_stays_on_the_page_and_stays_silent(
        self,
        project_window: MainWindow.MainWindowManager,
        announcements: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ):
        pages = project_window.thumbnails.pages()
        project_window.goLastPage()
        _start(project_window).beep_when_finished = True
        _set_active_window(monkeypatch, None)

        project_window._on_batch_finished(BatchResult(cancelled=[pages[-1]]))

        assert announcements == []
        assert project_window.current_page == pages[-1]
