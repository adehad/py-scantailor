"""Shared harness for automated UI pathway tests (pytest-qt, offscreen)."""

import dataclasses
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
from scantailor.app.ui.MainWindow import MainWindowManager, SaveChoice
from scantailor.main import setup_qcore

PAGE_WIDTH = 600
PAGE_HEIGHT = 800


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
        manager.ui.close()


@dataclasses.dataclass
class PageImages:
    """Synthetic scanned pages written to disk."""

    directory: pathlib.Path
    paths: list[pathlib.Path]


def _draw_text_like_blobs(rng: np.random.Generator) -> np.ndarray:
    page = np.full((PAGE_HEIGHT, PAGE_WIDTH, 3), 255, dtype=np.uint8)
    for line_top in range(80, PAGE_HEIGHT - 80, 40):
        x = 60
        while x < PAGE_WIDTH - 60:
            word_width = int(rng.integers(20, 80))
            page[line_top : line_top + 14, x : min(x + word_width, PAGE_WIDTH - 60)] = (
                30
            )
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
        assert image.save(str(path)), f"could not write {path}"
        paths.append(path)
    return PageImages(directory=directory, paths=paths)


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


@pytest.fixture
def respond_to_modal(qtbot: QtBot) -> Callable[..., None]:
    """Return a function that schedules a response to the next modal dialog.

    The callback runs from the event loop while the modal dialog is blocking
    in exec(), so it works for dialogs and QMessageBox instances alike.
    """

    def schedule(accept: bool = True, timeout_ms: int = 5000) -> None:
        deadline = QtCore.QDeadlineTimer(timeout_ms)

        def poll() -> None:
            modal = QtWidgets.QApplication.activeModalWidget()
            if isinstance(modal, QtWidgets.QDialog):
                if accept:
                    modal.accept()
                else:
                    modal.reject()
            elif not deadline.hasExpired():
                QtCore.QTimer.singleShot(10, poll)

        QtCore.QTimer.singleShot(10, poll)

    return schedule


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
