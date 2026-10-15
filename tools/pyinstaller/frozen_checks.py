"""Checks that a frozen ScanTailor folder works, run by the release workflow.

app.spec builds this as scantailor-checks only when SCANTAILOR_BUILD_CHECKS=1 and
the workflow deletes it before archiving, so it never ships.

Usage:
    scantailor-checks imports MODULES_FILE
    scantailor-checks gui PAGES_DIR WORK_DIR

``imports`` imports every module in MODULES_FILE (used_modules.txt) whose
top-level package the build contains, and lists the ones that fail. scantailor
has no dynamic imports, so PyInstaller finds every module of it the app can
reach; one of its modules missing from the build is listed as imported only by
the tests, not as a failure.

``gui`` drives the real window on Qt's offscreen platform: it creates a project
from the images in PAGES_DIR, shows every stage, runs the batch, opens the
settings, default parameters and about dialogs, switches to Korean through the
settings dialog and exports a PDF. Any warning the app logs fails the run.

The script imports only the stdlib and scantailor; Qt and loguru come through
scantailor's own modules, so the checks cannot add a module to the bundle that
the app does not already import (tools/bundle_report.py confirms it).
"""

import argparse
import importlib
import importlib.util
import multiprocessing
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.settings import ApplicationSettings
from scantailor.main import QtCore, QtWidgets, apply_theme, setup_qcore
from scantailor.translations import logger

_STEP_TIMEOUT_S: Final = 900.0
_DIALOG_TIMEOUT_S: Final = 60.0
_CHECK_LANGUAGE: Final = "ko"
_FIRST_PARTY: Final = "scantailor"
_WINDOW_SIZE: Final = (1280, 900)


class CheckFailure(Exception):
    """A step of the GUI check did not reach the expected state."""


@dataclass
class ImportReport:
    """Outcome of the import check.

    Attributes:
        failures: One line per module that failed to import.
        absent: Top-level packages skipped because the build lacks them.
        tests_only: scantailor modules the build lacks, which no entry point
            imports.
    """

    failures: list[str] = field(default_factory=list)
    absent: set[str] = field(default_factory=set)
    tests_only: list[str] = field(default_factory=list)


def check_imports(names: Iterable[str]) -> ImportReport:
    """Import each module whose top-level package is present.

    Args:
        names (Iterable[str]): Dotted module names.

    Returns:
        ImportReport: What failed, what was skipped and why.
    """
    report = ImportReport()
    for name in names:
        package = name.partition(".")[0]
        if package in report.absent:
            continue
        if importlib.util.find_spec(package) is None:
            report.absent.add(package)
            continue
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            if package == _FIRST_PARTY and _is_missing_module_of(error, name):
                report.tests_only.append(name)
            else:
                report.failures.append(f"{name}: ModuleNotFoundError: {error}")
        except Exception as error:
            report.failures.append(f"{name}: {type(error).__name__}: {error}")
    return report


def _is_missing_module_of(error: ModuleNotFoundError, name: str) -> bool:
    """Tell whether ``name`` or one of its parent packages is what is missing."""
    return error.name is not None and (
        name == error.name or name.startswith(f"{error.name}.")
    )


def read_module_list(path: Path) -> list[str]:
    """Read a module list, one name per line, skipping comments."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _isolate_settings(work_dir: Path) -> None:
    """Keep ApplicationSettings and QSettings inside the scratch folder."""
    ApplicationSettings.set_config_path(work_dir / "settings.json")
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    for scope in (QtCore.QSettings.Scope.UserScope, QtCore.QSettings.Scope.SystemScope):
        QtCore.QSettings.setPath(
            QtCore.QSettings.Format.IniFormat, scope, str(work_dir / "qsettings")
        )
    QtCore.QCoreApplication.setOrganizationName("scantailor-checks")
    QtCore.QCoreApplication.setApplicationName("scantailor-checks")


class _Driver:
    """Pumps the event loop and answers modal dialogs for the GUI check."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def pump(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.app.processEvents()
            time.sleep(0.02)

    def wait_until(self, done: Callable[[], bool], what: str) -> None:
        end = time.monotonic() + _STEP_TIMEOUT_S
        while not done():
            if time.monotonic() > end:
                raise CheckFailure(f"timed out waiting for {what}")
            self.pump(0.1)

    def through_modal(
        self, what: str, trigger: Callable[[], object], answer: Callable[[Any], None]
    ) -> None:
        """Run ``trigger``, which blocks in a dialog, and ``answer`` that dialog.

        Raises:
            CheckFailure: No dialog appeared.
        """
        answered: list[str] = []
        deadline = time.monotonic() + _DIALOG_TIMEOUT_S

        def poll() -> None:
            dialog = self.app.activeModalWidget()
            if dialog is None:
                if time.monotonic() < deadline:
                    QtCore.QTimer.singleShot(100, poll)
                return
            answered.append(type(dialog).__name__)
            answer(dialog)

        QtCore.QTimer.singleShot(100, poll)
        trigger()
        if not answered:
            raise CheckFailure(f"{what} showed no dialog")
        print(f"  {what}: {answered[0]}")


def _accept(dialog: Any) -> None:
    dialog.accept()


def _reject(dialog: Any) -> None:
    dialog.reject()


def _choose_language(dialog: Any) -> None:
    """Pick the check language in the settings dialog and press OK."""
    box = dialog.findChild(QtWidgets.QComboBox, "languageBox")
    index = box.findData(_CHECK_LANGUAGE) if box is not None else -1
    if index < 0:
        dialog.reject()
        raise CheckFailure(f"the settings dialog offers no {_CHECK_LANGUAGE}")
    box.setCurrentIndex(index)
    buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
    buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()


def _drive_window(driver: _Driver, pages_dir: Path, work_dir: Path) -> None:
    """Take the main window through the steps the module docstring lists."""
    names = sorted(path.name for path in pages_dir.iterdir() if path.is_file())
    output_dir = work_dir / "out"
    manager = MainWindowManager(
        project_files_dialog_factory=lambda parent: ProjectFilesDialog(
            input_dir=pages_dir,
            output_dir=output_dir,
            project_files=names,
            parent=parent,
        )
    )
    manager.ui.resize(*_WINDOW_SIZE)
    manager.ui.show()
    driver.pump(0.5)
    try:
        driver.through_modal("new project", manager.actionNewProject.trigger, _accept)
        driver.wait_until(lambda: len(manager.pages()) == len(names), "the project")

        for stage in range(1, 7):
            getattr(manager, f"switchFilter{stage}")()
            driver.pump(0.3)
            driver.wait_until(lambda: not manager.render_pending, f"stage {stage}")
            print(f"  stage {stage}: rendered")

        manager.startBatchProcessing()
        driver.pump(0.3)
        driver.wait_until(lambda: not manager.batch_runner.is_running, "the batch")
        if manager.batch_dialog is not None:
            manager.batch_dialog.close()
        print("  batch: finished")

        driver.through_modal("settings", manager.openSettingsDialog, _reject)
        driver.through_modal(
            "default parameters", manager.openDefaultParamsDialog, _reject
        )
        manager.showAboutDialog()
        driver.pump(0.5)
        about = [
            w for w in driver.app.topLevelWidgets() if type(w).__name__ == "AboutDialog"
        ]
        if not about:
            raise CheckFailure("the about dialog did not open")
        for widget in about:
            widget.close()
        print("  about: AboutDialog")

        driver.through_modal("language", manager.openSettingsDialog, _choose_language)
        driver.pump(0.5)
        translators = driver.app.findChildren(
            QtCore.QTranslator, "scantailor-translator"
        )
        if len(translators) != 2:
            raise CheckFailure(
                f"switching to {_CHECK_LANGUAGE} installed {len(translators)} "
                "translators, expected Qt's and ScanTailor's"
            )

        driver.through_modal("export PDF", manager.exportPdf, _accept)
        driver.wait_until(lambda: not manager.pdf_job.is_running, "the PDF export")
        driver.pump(0.3)
        result = manager.last_pdf_export
        expected = len(manager.pages())
        if result is None or result.pages != expected or not result.out_path.is_file():
            raise CheckFailure(
                f"the PDF export wrote {result}, expected {expected} pages"
            )
        print(f"  export PDF: {result.pages} pages")
    finally:
        manager.modified = False
        manager.ui.close()
        driver.pump(0.2)


def run_gui_check(pages_dir: Path, work_dir: Path) -> list[str]:
    """Drive the window offscreen through a whole project.

    Args:
        pages_dir (Path): Folder of page images to build the project from.
        work_dir (Path): Empty scratch folder for the output, settings and PDF.

    Returns:
        list[str]: Problems found; empty when the run passed.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    problems: list[str] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    _isolate_settings(work_dir)

    shown: set[str] = set()

    def on_qt_message(kind: Any, _context: Any, message: str) -> None:
        if kind in (QtCore.QtMsgType.QtCriticalMsg, QtCore.QtMsgType.QtFatalMsg):
            problems.append(f"Qt: {message}")
        if message not in shown:
            shown.add(message)
            print(f"  Qt {kind.name}: {message}")

    def on_exception(kind: type[BaseException], error: BaseException, tb: Any) -> None:
        problems.append("".join(traceback.format_exception(kind, error, tb)))

    sink = logger.add(
        lambda message: problems.append(str(message).strip()),
        level="WARNING",
        format="{level}: {name}: {message}",
    )
    previous_hook, sys.excepthook = sys.excepthook, on_exception
    QtCore.qInstallMessageHandler(on_qt_message)
    try:
        setup_qcore()
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        apply_theme(app)
        _drive_window(_Driver(app), pages_dir, work_dir)
    except CheckFailure as error:
        problems.append(str(error))
    finally:
        QtCore.qInstallMessageHandler(None)
        sys.excepthook = previous_hook
        logger.remove(sink)
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    checks = parser.add_subparsers(dest="check", required=True)
    imports = checks.add_parser("imports", help="import every listed module")
    imports.add_argument("modules", type=Path)
    gui = checks.add_parser("gui", help="drive the window through a project")
    gui.add_argument("pages", type=Path)
    gui.add_argument("work", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one check and report it.

    Args:
        argv (list[str] | None): Command line without the program name.

    Returns:
        int: 0 when the check passed, 1 otherwise.
    """
    args = parse_args(argv)
    if args.check == "imports":
        names = read_module_list(args.modules)
        report = check_imports(names)
        if report.absent:
            print(f"not bundled, skipped: {', '.join(sorted(report.absent))}")
        for name in report.tests_only:
            print(f"imported only by the tests: {name}")
        for failure in report.failures:
            print(f"cannot import {failure}")
        print(f"{len(report.failures)} of {len(names)} listed modules failed to import")
        return 1 if report.failures else 0
    print("GUI check:")
    problems = run_gui_check(args.pages, args.work)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print("GUI check " + ("FAILED" if problems else "passed"))
    return 1 if problems else 0


if __name__ == "__main__":
    # The batch spawns workers by relaunching this executable; freeze_support
    # turns that relaunch into a worker before the arguments are read.
    multiprocessing.freeze_support()
    sys.stdout.reconfigure(line_buffering=True)
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # PySide's module teardown at interpreter exit runs the app's event filters
    # after their widgets are gone, which can lose output; the result is final.
    os._exit(code)
