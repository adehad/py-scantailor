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

# Read when loguru is first imported, in this process and in the spawned batch
# workers that inherit the environment.
os.environ.setdefault("LOGURU_LEVEL", "WARNING")

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
    "stage1_fix_orientation_trim",
    "stage3_deskew_oblique",
    "stage5_margins_apply",
    "stage2_split_pages_tilted",
    "stage2_split_pages_removed_half",
    "stage5_margins_guides",
    "thumbnails_incomplete",
    "stage6_tab_picture_mask",
    "status_tip",
    "stage6_output_options",
    "stage6_output_guides",
    "stage6_output_edge_methods",
    "stage6_output_picture_shape",
    "stage6_dewarping_metrics",
    "stage6_tab_picture_zones",
    "stage6_tab_fill_zones",
    "stage6_tab_dewarping",
    "stage6_tab_despeckling",
    "stage6_zone_menu",
    "stage6_zone_properties",
    "thumbnails_sorting",
    "tools_menu",
    "settings",
    "default_parameters",
    "batch_progress",
    "export_pdf",
)

_PAGE_COUNT = 4
_PAGE_SIZE = (1275, 1650)
_PAGE_DPI = 300
_WINDOW_SIZE = (1440, 1100)
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
    from PySide6 import QtCore, QtTest, QtWidgets

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
    from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
    from scantailor.app.ui.image_views.output import ImageViewTab, OutputImageView
    from scantailor.app.ui.theme import apply_theme
    from scantailor.app.ui.widgets.collapsible_group_box import CollapsibleGroupBox
    from scantailor.core.models import SubPage
    from scantailor.core.stage_sequence import FilterStage
    from scantailor.core.zones import (
        FillZoneProperties,
        PictureLayer,
        PictureZoneProperties,
    )
    from scantailor.filters.output.params import FillZoneSpec, PictureZoneSpec
    from scantailor.filters.page_layout.guide import Guide
    from scantailor.filters.page_split.page_layout import PageLayout, PageLayoutType

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

    def in_modal(
        name: str, trigger: Callable[[], object], *, accept: bool = False
    ) -> None:
        """Run ``trigger`` (which blocks in a dialog), grab that dialog, close it."""

        def shoot() -> None:
            dialog = app.activeModalWidget()
            if dialog is None:
                QtCore.QTimer.singleShot(100, shoot)
                return
            pump(0.5)
            save(dialog, name)
            if accept:
                dialog.accept()
            else:
                dialog.reject()

        QtCore.QTimer.singleShot(300, shoot)
        trigger()

    def save_sorting_panel(name: str) -> None:
        """Save the thumbnail column with its sort controls, as part of the window.

        A child grabbed on its own loses the window's background, so the window is
        grabbed and cropped.
        """
        widget = manager.pauseSortingBtn.parentWidget()
        while widget is not None and widget.height() < _WINDOW_SIZE[1] // 2:
            widget = widget.parentWidget()
        assert widget is not None
        origin = widget.mapTo(manager.ui, QtCore.QPoint(0, 0))
        manager.ui.grab(QtCore.QRect(origin, widget.size())).save(
            str(shots / f"{name}.png")
        )

    (work_dir / "out").mkdir()

    def project_files_dialog(parent: QtWidgets.QWidget) -> ProjectFilesDialog:
        return ProjectFilesDialog(
            input_dir=sample_dir,
            output_dir=work_dir / "out",
            project_files=sample_page_names(),
            parent=parent,
        )

    manager = mw.MainWindowManager(project_files_dialog_factory=project_files_dialog)
    manager.ui.resize(*_WINDOW_SIZE)
    manager.ui.show()
    pump(1.0)
    save(manager.ui, "welcome")

    in_modal("new_project", manager.actionNewProject.trigger, accept=True)
    pump_until(lambda: len(manager.pages()) == _PAGE_COUNT, 30.0, "the project")

    def visit_stage(index: int, settle_s: float = 1.5) -> None:
        getattr(manager, f"switchFilter{index}")()
        pump(0.5)
        pump_until(
            lambda: not manager.render_pending,
            _RENDER_TIMEOUT_S,
            "the page render",
        )
        pump(settle_s)

    def show_stage(index: int, name: str | None = None, settle_s: float = 1.5) -> None:
        visit_stage(index, settle_s)
        save(manager.ui, name or STAGE_SHOTS[index - 1])

    def finish_render() -> None:
        pump(0.5)
        pump_until(
            lambda: not manager.render_pending,
            _RENDER_TIMEOUT_S,
            "the page render",
        )
        pump(1.5)

    def options_child[T: QtWidgets.QWidget](kind: type[T], name: str) -> T:
        found = manager.filterOptions.findChild(kind, name)
        assert found is not None, name
        return found

    def capture_trim() -> None:
        enable = options_child(QtWidgets.QCheckBox, "trimEnabledCheck")
        enable.setChecked(True)
        for name, pixels in (
            ("trimLeftSpin", 90),
            ("trimTopSpin", 100),
            ("trimRightSpin", 90),
            ("trimBottomSpin", 100),
        ):
            spin = options_child(QtWidgets.QSpinBox, name)
            spin.setValue(pixels)
        finish_render()
        save(manager.ui, "stage1_fix_orientation_trim")
        enable.setChecked(False)
        finish_render()

    def output_view() -> OutputImageView:
        view = manager.imageFrameLayout.currentWidget()
        assert isinstance(view, OutputImageView)
        return view

    def add_sample_zones() -> None:
        """Give the current page a picture zone and a fill zone, and render it.

        Leaving the Output stage and coming back renders the page again with the
        stored zones.
        """
        page = manager.current_page
        output_filter = manager.stages.output_filter
        picture = PictureZoneSpec(
            points=((300.0, 420.0), (900.0, 420.0), (900.0, 780.0), (300.0, 780.0)),
            properties=PictureZoneProperties(layer=PictureLayer.PAINTER2),
        )
        fill = FillZoneSpec(
            points=((300.0, 900.0), (900.0, 900.0), (900.0, 1100.0), (300.0, 1100.0)),
            properties=FillZoneProperties(color=(255, 240, 200)),
        )
        output_filter.set_params(
            page,
            output_filter.get_params(page)
            .with_picture_zones((picture,))
            .with_fill_zones((fill,)),
        )
        visit_stage(5)
        visit_stage(6)

    def capture_split_and_guides() -> None:
        """Tilt a split line, drop one half of an image, and place two guides."""
        first = manager.pages()[0]
        split_filter = manager.stages.page_split_filter
        layout = split_filter.get_page_layout(first)
        assert layout is not None
        x, y, width, height = layout.outline
        line = ((x + width * 0.45, y), (x + width * 0.55, y + height))
        tilted = PageLayout(
            outline=layout.outline,
            layout_type=PageLayoutType.TWO_PAGES,
            cutter1=line,
            cutter2=line,
        )
        split_filter.set_manual_layout(first, tilted)
        manager.current_page = first
        visit_stage(3)
        show_stage(2, "stage2_split_pages_tilted")

        project = manager.project
        last = manager.pages()[-1]
        position = next(
            n for n, info in enumerate(project.images) if info.id == last.image_id
        )
        project.images[position] = project.images[position].model_copy(
            update={"num_sub_pages": 1, "left_half_removed": True}
        )
        manager._refresh_page_list(last)
        manager.current_page = manager.pages()[-1]
        split_filter.set_manual_layout(manager.current_page, tilted)
        visit_stage(3)
        show_stage(2, "stage2_split_pages_removed_half")
        manager._on_unremove_page_requested(manager.current_page)
        assert manager.current_page.sub_page in (SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE)

        layout_settings = manager.stages.page_layout_filter.settings
        layout_settings.set_guides(
            [
                Guide(position_mm=-30.0, is_horizontal=False),
                Guide(position_mm=40.0, is_horizontal=True),
            ]
        )
        layout_settings.set_middle_rect_shown(True)
        manager.current_page = manager.pages()[1]
        visit_stage(4)
        show_stage(5, "stage5_margins_guides")

    manager.current_page = manager.pages()[0]
    for index in range(1, 6):
        show_stage(index)
        if index == 1:
            capture_trim()
        if index == 3:
            save_sorting_panel("thumbnails_incomplete")
        if index == 5:
            in_modal(
                "stage5_margins_apply",
                lambda: manager.show_apply_dialog(FilterStage.PAGE_LAYOUT, "margins"),
            )

    manager.startBatchProcessing()
    pump_until(
        lambda: not manager.batch_runner.is_running, _BATCH_TIMEOUT_S, "the batch"
    )
    pump(1.0)
    if manager.batch_dialog is not None:
        save(manager.batch_dialog, "batch_progress")
        manager.batch_dialog.close()
    show_stage(6)

    method = options_child(QtWidgets.QComboBox, "methodCombo")
    fill_outside = options_child(QtWidgets.QCheckBox, "fillOutsidePageBoxCB")
    previous_method = method.currentIndex()
    method.setCurrentIndex(method.findText("Window"))
    fill_outside.setChecked(True)
    # Collapsed so that the Options and Binarization panels fit in one screenshot.
    folded = [
        options_child(CollapsibleGroupBox, name) for name in ("dpiGroup", "modeGroup")
    ]
    for group in folded:
        group.set_collapsed(True)
    finish_render()
    save(manager.ui, "stage6_output_options")
    for group in folded:
        group.set_collapsed(False)
    fill_outside.setChecked(False)
    method.setCurrentIndex(previous_method)
    finish_render()

    output_view().set_guides_shown(True)
    pump(0.5)
    save(manager.ui, "stage6_output_guides")
    output_view().set_guides_shown(False)

    def reveal(widget: QtWidgets.QWidget) -> None:
        """Scroll the options panel until ``widget`` shows."""
        area = widget.parentWidget()
        while area is not None and not isinstance(area, QtWidgets.QScrollArea):
            area = area.parentWidget()
        if area is not None:
            area.ensureWidgetVisible(widget)
        pump(0.3)

    method.setCurrentIndex(method.findText("EdgePlus"))
    finish_render()
    reveal(options_child(QtWidgets.QWidget, "binarizationGroup"))
    save(manager.ui, "stage6_output_edge_methods")
    method.setCurrentIndex(previous_method)
    finish_render()

    mode_button = options_child(QtWidgets.QAbstractButton, "mixedBtn")
    mode_button.click()
    finish_render()
    reveal(options_child(QtWidgets.QWidget, "pictureShapeOptions"))
    save(manager.ui, "stage6_output_picture_shape")
    options_child(QtWidgets.QAbstractButton, "bwBtn").click()
    finish_render()

    options_child(QtWidgets.QAbstractButton, "dewarpAutoBtn").click()
    finish_render()
    reveal(options_child(QtWidgets.QWidget, "dewarpingGroup"))
    save(manager.ui, "stage6_dewarping_metrics")
    output_view().set_current_tab(ImageViewTab.DEWARPING)
    pump(3.0)
    save(manager.ui, "stage6_tab_dewarping")
    output_view().set_current_tab(ImageViewTab.OUTPUT)
    options_child(QtWidgets.QAbstractButton, "dewarpOffBtn").click()
    finish_render()

    add_sample_zones()
    view = output_view()
    for tab_name, shot in (
        ("PICTURE_ZONES", "stage6_tab_picture_zones"),
        ("FILL_ZONES", "stage6_tab_fill_zones"),
        ("DESPECKLING", "stage6_tab_despeckling"),
    ):
        view.set_current_tab(ImageViewTab[tab_name])
        pump(3.0)
        save(manager.ui, shot)

    view.set_current_tab(ImageViewTab.PICTURE_ZONES)
    pump(0.5)
    zone_centre = view.zones()[0].polygon.boundingRect().center()
    zone_pos = view.virtual_to_widget_point(zone_centre)
    menu = view.build_context_menu(zone_pos)
    menu.popup(view.viewport().mapToGlobal(zone_pos.toPoint()))
    pump(0.5)
    save(menu, "stage6_zone_menu")
    menu.hide()
    in_modal("stage6_zone_properties", lambda: view.edit_zone_properties(0))

    QtTest.QTest.mouseMove(view.viewport(), zone_pos.toPoint())
    pump(0.5)
    save(manager.ui, "status_tip")

    options_child(QtWidgets.QAbstractButton, "mixedBtn").click()
    finish_render()
    view = output_view()
    view.set_current_tab(ImageViewTab.PICTURE_ZONES)
    pump(1.0)
    save(manager.ui, "stage6_tab_picture_mask")
    options_child(QtWidgets.QAbstractButton, "bwBtn").click()
    finish_render()
    output_view().set_current_tab(ImageViewTab.OUTPUT)

    manager.sortOptions.setCurrentIndex(1)
    manager.pauseSortingBtn.setChecked(True)
    pump(0.5)
    save_sorting_panel("thumbnails_sorting")
    manager.pauseSortingBtn.setChecked(False)
    manager.sortOptions.setCurrentIndex(0)

    tools = manager.ui.findChild(QtWidgets.QMenu, "menuDebug")
    assert tools is not None
    tools.popup(manager.ui.mapToGlobal(QtCore.QPoint(60, 30)))
    pump(0.5)
    save(tools, "tools_menu")
    tools.hide()

    manager.stages.deskew_filter.set_manual_oblique(manager.current_page, 2.0)
    show_stage(3, "stage3_deskew_oblique", settle_s=6.0)
    manager.stages.deskew_filter.set_manual_oblique(manager.current_page, 0.0)

    capture_split_and_guides()

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
