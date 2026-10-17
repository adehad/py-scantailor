"""Shared harness for automated UI pathway tests (pytest-qt, offscreen)."""

import dataclasses
import gc
import os
import pathlib
from collections.abc import Callable, Iterator

# Must be set before the first QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import shiboken6
from loguru import logger
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app._ui import resources_rc
from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.garbage_collection import (
    GuiThreadCollector,
    collect_garbage_on_the_gui_thread,
)
from scantailor.app.ui.MainWindow import MainWindowManager, SaveChoice
from scantailor.main import setup_qcore

PAGE_WIDTH = 600
PAGE_HEIGHT = 800
# Every text line starts at this margin and is at least this wide and tall.
FIRST_LINE_TOP = 80
LINE_LEFT = 60
LINE_PITCH = 40
BLOB_HEIGHT = 14
MIN_WORD_WIDTH = 20
# 300 dpi; the default Qt writes (about 100 dpi) would open the Fix DPI dialog.
SCAN_DOTS_PER_METER = 11811


@pytest.fixture(scope="session", autouse=True)
def _conf_qt():
    """Session-wide Qt core and resource initialisation."""
    setup_qcore()
    resources_rc.qInitResources()
    yield
    # Explicitly release resources to keep valgrind / sanitizers happy
    if hasattr(resources_rc, "qCleanupResources"):
        resources_rc.qCleanupResources()


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """Redirect QSettings to a per-test directory so user settings are never touched."""
    settings_dir = tmp_path / "qsettings"
    settings_dir.mkdir()
    previous_format = QtCore.QSettings.defaultFormat()
    previous_org = QtCore.QCoreApplication.organizationName()
    previous_app = QtCore.QCoreApplication.applicationName()

    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    for scope in (
        QtCore.QSettings.Scope.UserScope,
        QtCore.QSettings.Scope.SystemScope,
    ):
        QtCore.QSettings.setPath(
            QtCore.QSettings.Format.IniFormat, scope, str(settings_dir)
        )
    QtCore.QCoreApplication.setOrganizationName(f"scantailor-test-{tmp_path.name}")
    QtCore.QCoreApplication.setApplicationName("scantailor-test")
    yield settings_dir

    QtCore.QSettings.setDefaultFormat(previous_format)
    QtCore.QCoreApplication.setOrganizationName(previous_org)
    QtCore.QCoreApplication.setApplicationName(previous_app)


@pytest.fixture(autouse=True)
def garbage_is_collected_on_the_gui_thread(
    qapp: QtWidgets.QApplication,
) -> Iterator[GuiThreadCollector]:
    """Collect cyclic garbage on the GUI thread only, as the application does.

    The collection at teardown frees what the test left while Qt is idle, and
    automatic collection comes back for the tests outside this package that run
    on the same worker.
    """
    collector = collect_garbage_on_the_gui_thread(qapp)
    yield collector
    gc.collect()
    collector.stop()


@pytest.fixture(autouse=True)
def no_window_outlives_its_test() -> Iterator[None]:
    """Hide windows a test left showing and let their deferred deletes run.

    QTest.mouseMove moves the platform cursor, and the offscreen platform hands the
    move to the first exposed top-level window, in creation order, that contains
    the point. A window left over from an earlier test on the same worker therefore
    swallows the hover meant for the widget under test.
    """
    yield
    for widget in QtWidgets.QApplication.topLevelWidgets():
        if widget.isVisible():
            widget.hide()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.fixture(autouse=True)
def no_translation_outlives_its_test(
    qapp: QtWidgets.QApplication,
) -> Iterator[None]:
    """Uninstall translators a test installed, such as by accepting a new language.

    A translator left on the shared application would show later tests on the same
    worker translated text where they look for English.
    """
    yield
    for translator in qapp.findChildren(QtCore.QTranslator):
        qapp.removeTranslator(translator)
        translator.setParent(None)


@pytest.fixture(autouse=True)
def cursor_starts_in_the_screen_corner(qapp: QtWidgets.QApplication) -> None:
    """Park the platform cursor where no test points it before each test.

    QTest.mouseMove sets the platform cursor, and Qt drops a set to the point the
    cursor is already at, so no move event reaches the widget. The cursor keeps its
    place across the tests of a worker: a test that moves to the point an earlier
    test left it at would never see its hover.

    Args:
        qapp (QtWidgets.QApplication): Needed so a screen exists.
    """
    screen = qapp.primaryScreen()
    QtGui.QCursor.setPos(screen, screen.geometry().bottomRight())


@pytest.fixture
def main_window(
    qtbot: QtBot, isolated_settings: pathlib.Path
) -> Iterator[MainWindowManager]:
    """Construct, show and activate the main window; close it on teardown."""
    manager = MainWindowManager()

    def discard_changes(_widget: QtWidgets.QWidget) -> None:
        # Closing a modified project would otherwise block on the save prompt.
        manager.modified = False

    qtbot.addWidget(manager.ui, before_close_func=discard_changes)
    manager.ui.show()
    qtbot.wait_active(manager.ui)
    yield manager
    if shiboken6.isValid(manager.ui):
        discard_changes(manager.ui)
        # Closing a parent leaves its visible child dialogs (batch progress,
        # about) showing.
        for dialog in manager.ui.findChildren(QtWidgets.QDialog):
            dialog.close()
        manager.ui.close()


@pytest.fixture
def wait_for_thumbnails() -> Callable[[MainWindowManager], None]:
    """Return a function that waits until the thumbnail workers are done.

    Workers store auto-detected parameters for pages other than the current
    one, so a test that reads or counts parameters or renders must wait for
    them first. Their results are delivered through the event loop, which the
    function also drains.
    """

    def wait(window: MainWindowManager, timeout_ms: int = 60_000) -> None:
        assert window.thumbnails.model.wait_for_idle(timeout_ms), (
            "thumbnail workers still busy"
        )
        QtWidgets.QApplication.processEvents()

    return wait


@dataclasses.dataclass
class PageImages:
    """Synthetic scanned pages written to disk."""

    directory: pathlib.Path
    paths: list[pathlib.Path]
    ink_point: tuple[int, int]
    """(x, y) of a pixel that is dark on every page."""


def _draw_text_like_blobs(rng: np.random.Generator) -> np.ndarray:
    page = np.full((PAGE_HEIGHT, PAGE_WIDTH, 3), 255, dtype=np.uint8)
    for line_top in range(FIRST_LINE_TOP, PAGE_HEIGHT - 80, LINE_PITCH):
        x = LINE_LEFT
        while x < PAGE_WIDTH - 60:
            word_width = int(rng.integers(MIN_WORD_WIDTH, 80))
            page[
                line_top : line_top + BLOB_HEIGHT,
                x : min(x + word_width, PAGE_WIDTH - 60),
            ] = 30
            x += word_width + 12
    return page


@pytest.fixture
def page_images(tmp_path: pathlib.Path) -> PageImages:
    """Write three 600x800 PNG pages with text-like dark blobs into tmp_path."""
    directory = tmp_path / "scans"
    directory.mkdir()
    paths = []
    for index in range(3):
        rng = np.random.default_rng(seed=index)
        pixels = np.ascontiguousarray(_draw_text_like_blobs(rng))
        image = QtGui.QImage(
            pixels.data,
            PAGE_WIDTH,
            PAGE_HEIGHT,
            3 * PAGE_WIDTH,
            QtGui.QImage.Format.Format_RGB888,
        )
        path = directory / f"page_{index + 1:03d}.png"
        image.setDotsPerMeterX(SCAN_DOTS_PER_METER)
        image.setDotsPerMeterY(SCAN_DOTS_PER_METER)
        assert image.save(str(path)), f"could not write {path}"
        paths.append(path)
    ink_point = (
        LINE_LEFT + MIN_WORD_WIDTH // 2,
        FIRST_LINE_TOP + BLOB_HEIGHT // 2,
    )
    return PageImages(directory=directory, paths=paths, ink_point=ink_point)


@dataclasses.dataclass
class FileDialogStub:
    """Programmable replacement for static QFileDialog calls."""

    open_file: str = ""
    save_file: str = ""
    directory: str = ""
    calls: list[tuple[str, tuple, dict]] = dataclasses.field(default_factory=list)


@pytest.fixture
def file_dialogs(monkeypatch: pytest.MonkeyPatch) -> FileDialogStub:
    """Stub static QFileDialog getters so no native/modal dialog opens."""
    stub = FileDialogStub()

    def get_open_file_name(*args, **kwargs):
        stub.calls.append(("getOpenFileName", args, kwargs))
        return stub.open_file, ""

    def get_save_file_name(*args, **kwargs):
        stub.calls.append(("getSaveFileName", args, kwargs))
        return stub.save_file, ""

    def get_existing_directory(*args, **kwargs):
        stub.calls.append(("getExistingDirectory", args, kwargs))
        return stub.directory

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", get_open_file_name)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", get_save_file_name)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getExistingDirectory", get_existing_directory
    )
    return stub


class ModalResponder:
    """Answers modal dialogs from the event loop and fails when none appears.

    The callback runs while the modal dialog is blocking in exec(), so it works
    for dialogs and QMessageBox instances alike. A response that finds no modal
    within its timeout, or is still waiting when the test ends, is recorded in
    ``unanswered`` and fails the test at teardown.
    """

    def __init__(self) -> None:
        self.unanswered = 0
        self._pending = 0
        self._cancelled = False

    def __call__(self, accept: bool = True, timeout_ms: int = 5000) -> None:
        """Schedule a response to the next modal dialog.

        Args:
            accept: Accept the dialog; otherwise reject it.
            timeout_ms: How long to wait for a modal dialog to appear.
        """
        deadline = QtCore.QDeadlineTimer(timeout_ms)
        self._pending += 1

        def poll() -> None:
            if self._cancelled:
                return
            modal = QtWidgets.QApplication.activeModalWidget()
            if isinstance(modal, QtWidgets.QDialog):
                self._pending -= 1
                if accept:
                    modal.accept()
                else:
                    modal.reject()
            elif deadline.hasExpired():
                self._pending -= 1
                self.unanswered += 1
            else:
                QtCore.QTimer.singleShot(10, poll)

        QtCore.QTimer.singleShot(10, poll)

    def finish(self) -> int:
        """Stop waiting and return how many responses never met a modal dialog."""
        self._cancelled = True
        self.unanswered += self._pending
        self._pending = 0
        return self.unanswered

    def forget_unanswered(self) -> None:
        """Accept that scheduled responses found no modal, so teardown passes."""
        self.unanswered = 0


@pytest.fixture
def respond_to_modal(qtbot: QtBot) -> Iterator[ModalResponder]:
    """Return a function that schedules a response to the next modal dialog."""
    responder = ModalResponder()
    yield responder
    unanswered = responder.finish()
    assert unanswered == 0, (
        f"{unanswered} scheduled modal response(s) never met a modal dialog"
    )


@pytest.fixture
def log_messages() -> Iterator[list[str]]:
    """Capture loguru messages emitted during the test."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="DEBUG"
    )
    yield messages
    logger.remove(handler_id)


@dataclasses.dataclass
class DialogFactory:
    """Project files dialog factory that records the dialogs it created."""

    input_dir: pathlib.Path
    names: list[str]
    output_dir: pathlib.Path | None = None
    created: list[ProjectFilesDialog] = dataclasses.field(default_factory=list)

    def __call__(self, parent: QtWidgets.QWidget) -> ProjectFilesDialog:
        dialog = ProjectFilesDialog(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            project_files=self.names,
            parent=parent,
        )
        self.created.append(dialog)
        return dialog


@pytest.fixture
def dialog_factory(page_images: PageImages, tmp_path: pathlib.Path) -> DialogFactory:
    """Factory choosing all synthetic page images, with an output dir in tmp_path."""
    return DialogFactory(
        input_dir=page_images.directory,
        names=[path.name for path in page_images.paths],
        output_dir=tmp_path / "out",
    )


@pytest.fixture
def project_window(
    main_window: MainWindowManager,
    dialog_factory: DialogFactory,
    respond_to_modal: Callable[..., None],
) -> MainWindowManager:
    """Main window with a new, unsaved project holding the three synthetic pages."""
    main_window.project_files_dialog_factory = dialog_factory
    respond_to_modal(accept=True)
    main_window.actionNewProject.trigger()
    assert main_window.project is not None
    # Windows will not move or delete a scan a render thread still holds open.
    main_window.renderer.wait_for_done(10_000)
    main_window.thumbnails.model.wait_for_idle(10_000)
    return main_window


@dataclasses.dataclass
class SavePromptScript:
    """Scripted answers for the unsaved-changes prompt."""

    answers: list[SaveChoice] = dataclasses.field(default_factory=list)
    prompts: int = 0

    def __call__(self) -> SaveChoice:
        self.prompts += 1
        return self.answers.pop(0)


@pytest.fixture
def save_prompt(main_window: MainWindowManager) -> SavePromptScript:
    """Replace the unsaved-changes prompt with a scripted one."""
    script = SavePromptScript()
    main_window.save_prompt = script
    return script
