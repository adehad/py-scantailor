"""Capture screenshots of the real application for the user manual.

Drives the application offscreen on a few synthetic pages and writes one PNG per
screen and colour scheme, so the manual cannot drift from the interface.

Usage:
    python tools/docs_screenshots.py [--out DIR] [--scheme light|dark|both]
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "_generated" / "screenshots"
SCHEMES = ("light", "dark")

STAGE_SHOTS = (
    "stage1_fix_orientation",
    "stage2_split_pages",
    "stage3_deskew",
    "stage4_select_content",
    "stage5_margins",
    "stage6_output",
)
SHOT_NAMES = (
    "welcome",
    "new_project",
    *STAGE_SHOTS,
    "settings",
    "default_parameters",
    "batch_progress",
    "export_pdf",
)

_PAGE_COUNT = 4
_PAGE_SIZE = (1275, 1650)
_PAGE_DPI = 300
_WINDOW_SIZE = (1440, 1000)
_RENDER_TIMEOUT_S = 90.0
_BATCH_TIMEOUT_S = 300.0

_LOREM = (
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "sphinx of black quartz judge my vow",
    "how vexingly quick daft zebras jump",
    "waltz bad nymph for quick jigs vex",
)


def sample_page_names() -> list[str]:
    """Return the file names of the generated sample pages."""
    return [f"page_{number:02d}.png" for number in range(1, _PAGE_COUNT + 1)]


def generate_sample_pages(directory: Path) -> list[Path]:
    """Write synthetic scanned pages: skewed text on paper inside a dark scanner bed.

    The pages are random but seeded, so every run produces identical files.

    Args:
        directory (Path): Folder to write into; created when missing.

    Returns:
        list[Path]: The written PNG files, 300 DPI, 8-bit RGB.
    """
    import cv2
    import numpy as np
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    width, height = _PAGE_SIZE
    paths = []
    for index, name in enumerate(sample_page_names()):
        rng = np.random.default_rng(index)
        paper = np.full((height, width, 3), 238, np.uint8)
        paper_box = (110, 150, width - 110, height - 150)
        left, top, right, bottom = paper_box
        row = top + 60
        cv2.putText(
            paper,
            f"Chapter {index + 1}",
            (left + 60, row + 40),
            cv2.FONT_HERSHEY_DUPLEX,
            2.0,
            (30, 30, 30),
            4,
            cv2.LINE_AA,
        )
        row += 130
        line = 0
        while row < bottom - 90:
            if index % 2 and 420 < row - top < 760:
                row += 340
                cv2.rectangle(
                    paper,
                    (left + 120, row - 320),
                    (right - 120, row - 40),
                    (90, 90, 90),
                    -1,
                )
                continue
            text = _LOREM[(line + index) % len(_LOREM)]
            cv2.putText(
                paper,
                text,
                (left + 60, row),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.05,
                (25, 25, 25),
                2,
                cv2.LINE_AA,
            )
            row += 56
            line += 1
        page = np.full((height, width, 3), 40, np.uint8)
        page[top:bottom, left:right] = paper[top:bottom, left:right]
        angle = (-1.8, 1.2, -0.7, 2.1)[index % 4]
        rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        page = cv2.warpAffine(page, rotation, (width, height), borderValue=(40, 40, 40))
        noise = rng.normal(0, 5, page.shape)
        page = np.clip(page.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        path = directory / name
        Image.fromarray(page).save(path, dpi=(_PAGE_DPI, _PAGE_DPI))
        paths.append(path)
    return paths


def _capture(scheme: str, out_dir: Path) -> None:
    """Drive the application in this process and save every screenshot.

    Args:
        scheme (str): ``light`` or ``dark``.
        out_dir (Path): Folder that receives the PNGs.
    """
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["SCANTAILOR_THEME"] = scheme
    from PySide6 import QtCore, QtWidgets

    settings_dir = tempfile.mkdtemp(prefix="st-docs-settings-")
    work_dir = Path(tempfile.mkdtemp(prefix="st-docs-work-"))
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        settings_dir,
    )
    QtCore.QCoreApplication.setOrganizationName("scantailor-docs")
    QtCore.QCoreApplication.setApplicationName("scantailor-docs")

    from scantailor import main as st_main

    st_main.setup_qcore()
    app = QtWidgets.QApplication(sys.argv[:1])
    from scantailor.app._ui import resources_rc
    from scantailor.app.ui import MainWindow as mw
    from scantailor.app.ui.theme import apply_theme

    apply_theme(app)
    resources_rc.qInitResources()

    shots = out_dir / scheme
    shots.mkdir(parents=True, exist_ok=True)
    sample_dir = work_dir / "sample"
    generate_sample_pages(sample_dir)

    def pump(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(0.02)

    def pump_until(done: Callable[[], bool], timeout: float, what: str) -> None:
        end = time.monotonic() + timeout
        while not done():
            if time.monotonic() > end:
                raise TimeoutError(f"timed out waiting for {what}")
            pump(0.1)

    def save(widget: QtWidgets.QWidget, name: str) -> None:
        widget.grab().save(str(shots / f"{name}.png"))

    def in_modal(name: str, trigger: Callable[[], object]) -> None:
        """Run ``trigger`` (which blocks in a dialog), grab that dialog, close it."""

        def shoot() -> None:
            dialog = app.activeModalWidget()
            if dialog is None:
                QtCore.QTimer.singleShot(100, shoot)
                return
            pump(0.5)
            save(dialog, name)
            dialog.reject()

        QtCore.QTimer.singleShot(300, shoot)
        trigger()

    manager = mw.MainWindowManager()
    manager.settings.setValue(mw._LAST_INPUT_DIR_KEY, str(sample_dir))
    manager.ui.resize(*_WINDOW_SIZE)
    manager.ui.show()
    pump(1.0)
    save(manager.ui, "welcome")

    def fill_new_project() -> None:
        dialog = app.activeModalWidget()
        if dialog is None:
            QtCore.QTimer.singleShot(100, fill_new_project)
            return
        model = dialog._off_project_list.model()
        selection = dialog._off_project_list.selectionModel()
        listed = model.stringList()
        for row, name in enumerate(listed):
            if name in sample_page_names():
                selection.select(model.index(row), selection.SelectionFlag.Select)
        dialog._add_selected_to_project()
        dialog._output_dir_line.setText(str(work_dir / "out"))
        pump(0.3)
        save(dialog, "new_project")
        dialog.accept()

    (work_dir / "out").mkdir()
    QtCore.QTimer.singleShot(300, fill_new_project)
    manager.actionNewProject.trigger()
    pump_until(lambda: len(manager.pages()) == _PAGE_COUNT, 30.0, "the project")

    def show_stage(index: int) -> None:
        getattr(manager, f"switchFilter{index}")()
        pump(0.5)
        pump_until(
            lambda: not manager._render_pending, _RENDER_TIMEOUT_S, "the page render"
        )
        pump(1.5)
        save(manager.ui, STAGE_SHOTS[index - 1])

    manager.current_page = manager.pages()[0]
    for index in range(1, 6):
        show_stage(index)

    manager.startBatchProcessing()
    pump_until(
        lambda: not manager.batch_runner.is_running, _BATCH_TIMEOUT_S, "the batch"
    )
    pump(1.0)
    if manager.batch_dialog is not None:
        save(manager.batch_dialog, "batch_progress")
        manager.batch_dialog.close()
    show_stage(6)

    in_modal("settings", manager.actionSettings.trigger)
    in_modal("default_parameters", manager.actionDefaults.trigger)
    in_modal("export_pdf", manager.exportPdf)

    manager.modified = False
    manager.ui.close()
    pump(0.2)


def _run_both(out_dir: Path) -> int:
    """Capture both schemes in parallel child processes.

    Args:
        out_dir (Path): Folder that receives the ``light`` and ``dark`` folders.

    Returns:
        int: Zero when every child succeeded.
    """
    children = [
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--scheme",
                scheme,
                "--out",
                str(out_dir),
            ]
        )
        for scheme in SCHEMES
    ]
    return max(child.wait() for child in children)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv (list[str] | None): Command line without the program name.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scheme", choices=(*SCHEMES, "both"), default="both")
    args = parser.parse_args(argv)
    if args.scheme == "both":
        return _run_both(args.out)
    _capture(args.scheme, args.out)
    return 0


# Batch workers are spawned processes that re-import this module.
if __name__ == "__main__":
    sys.exit(main())
