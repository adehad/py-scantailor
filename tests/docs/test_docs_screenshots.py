"""Whole-window screenshots of the application for the user manual.

Each test drives the real main window on the sample book from ``tools/sample_book.py``
and saves the window, or one dialog or menu, under ``SCANTAILOR_DOCS_SCREENSHOTS``
(default ``docs/_generated/screenshots``). Every shot is taken on the page that shows
its feature. The light and dark schemes are separate tests so that they can run on
two workers.

Run with ``pytest -m docs_screenshot``, or ``tools/docs_screenshots.py``.
"""

import importlib.util
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from scantailor.app.ui.dialogs.project_files import ProjectFilesDialog
from scantailor.app.ui.image_views.output import ImageViewTab, OutputImageView
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.theme import apply_theme
from scantailor.app.ui.widgets.collapsible_group_box import CollapsibleGroupBox
from scantailor.core.models import PageId, SubPage
from scantailor.core.stage_sequence import FilterStage
from scantailor.core.zones import (
    FillZoneProperties,
    PictureLayer,
    PictureZoneProperties,
)
from scantailor.filters.output.params import FillZoneSpec, PictureZoneSpec
from scantailor.filters.page_layout.guide import Guide
from scantailor.filters.page_split.page_layout import PageLayout, PageLayoutType

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "tools"
_WINDOW_SIZE = (1440, 1100)
_RENDER_TIMEOUT_S = 90.0
_BATCH_TIMEOUT_S = 600.0


def _load_tool(name: str) -> ModuleType:
    """Import a script from ``tools/`` by path, as the docs build does."""
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# File names of the sample book's pages (see tools/sample_book.py).
(
    SINGLE,
    SPREAD_LIGHT,
    SPREAD_DARK,
    OFFCUT,
    SKEWED,
    OBLIQUE,
    CURVED,
    PICTURES,
    SPECKS,
    SIDEWAYS,
) = _load_tool("sample_book").SAMPLE_PAGE_NAMES


def page_named(manager: MainWindowManager, name: str) -> PageId:
    """Return the first page of the project whose source file has this name.

    Spreads split into two pages, so project positions do not match the sample order.
    """
    for page in manager.pages():
        if page.image_id.file_path.stem == name:
            return page
    raise AssertionError(f"{name} is not in the project")


def _shot_names() -> tuple[str, ...]:
    return _load_tool("docs_screenshots").SHOT_NAMES


def _output_dir() -> Path:
    configured = os.environ.get("SCANTAILOR_DOCS_SCREENSHOTS")
    if configured:
        return Path(configured)
    return REPO_ROOT / "docs" / "_generated" / "screenshots"


@pytest.mark.docs_screenshot
# Each scheme drives the window through the whole manual, about 4 minutes.
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "scheme",
    [
        pytest.param("light", marks=pytest.mark.xdist_group("docs-light")),
        pytest.param("dark", marks=pytest.mark.xdist_group("docs-dark")),
    ],
)
def test_manual_screenshots_capture_every_named_shot(
    scheme: str,
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the window through the manual's stages and save every named PNG.

    Args:
        scheme (str): ``light`` or ``dark``; selects the theme and the subfolder.
        qapp (QtWidgets.QApplication): The application, from pytest-qt.
        monkeypatch (pytest.MonkeyPatch): Sets the theme for this test only.
    """
    monkeypatch.setenv("SCANTAILOR_THEME", scheme)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    apply_theme(qapp)
    shots = _output_dir() / scheme
    shots.mkdir(parents=True, exist_ok=True)
    # Paths show in several dialogs; a short folder keeps them readable.
    work_dir = Path(tempfile.mkdtemp(prefix="st-docs-work-"))
    (work_dir / "out").mkdir()
    book = _load_tool("sample_book").write_sample_book(work_dir / "sample")
    sample_dir = book.pages[0].parent
    names = [path.name for path in book.pages]

    app = qapp

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
        name: str,
        trigger: Callable[[], object],
        *,
        accept: bool = False,
        prepare: Callable[[QtWidgets.QWidget], None] | None = None,
    ) -> None:
        """Run ``trigger``, which blocks in a dialog, grab that dialog and close it.

        ``prepare`` gets the dialog first, to pick a tab or a row for the shot.
        """

        def shoot() -> None:
            dialog = app.activeModalWidget()
            if dialog is None:
                QtCore.QTimer.singleShot(100, shoot)
                return
            if prepare is not None:
                prepare(dialog)
            pump(0.5)
            save(dialog, name)
            if accept:
                dialog.accept()
            else:
                dialog.reject()

        QtCore.QTimer.singleShot(300, shoot)
        trigger()

    def save_sorting_panel(name: str) -> None:
        """Save the thumbnail column with its sort controls, cropped from the window.

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

    def project_files_dialog(parent: QtWidgets.QWidget) -> ProjectFilesDialog:
        return ProjectFilesDialog(
            input_dir=sample_dir,
            output_dir=work_dir / "out",
            project_files=names,
            parent=parent,
        )

    def finish_render() -> None:
        pump(0.5)
        pump_until(
            lambda: not manager.render_pending, _RENDER_TIMEOUT_S, "the page render"
        )
        pump(1.5)

    def visit_stage(index: int, settle_s: float = 1.5) -> None:
        getattr(manager, f"switchFilter{index}")()
        pump(0.5)
        pump_until(
            lambda: not manager.render_pending, _RENDER_TIMEOUT_S, "the page render"
        )
        pump(settle_s)

    def show(
        stage: int,
        page: str,
        name: str | None = None,
        settle_s: float = 1.5,
    ) -> None:
        """Make ``page`` current, show ``stage`` and save the window if named."""
        manager.current_page = page_named(manager, page)
        visit_stage(stage, settle_s)
        if name is not None:
            save(manager.ui, name)

    def options_child[T: QtWidgets.QWidget](kind: type[T], name: str) -> T:
        found = manager.filterOptions.findChild(kind, name)
        assert found is not None, name
        return found

    def save_part(widget: QtWidgets.QWidget, name: str, margin: int = 8) -> None:
        """Save the part of the window around ``widget``, cropped from the window."""
        origin = widget.mapTo(manager.ui, QtCore.QPoint(0, 0))
        area = QtCore.QRect(origin, widget.size()).adjusted(
            -margin, -margin, margin, margin
        )
        manager.ui.grab(area.intersected(manager.ui.rect())).save(
            str(shots / f"{name}.png")
        )

    def output_view() -> OutputImageView:
        view = manager.imageFrameLayout.currentWidget()
        assert isinstance(view, OutputImageView)
        return view

    def reveal(widget: QtWidgets.QWidget) -> None:
        """Scroll the options panel until ``widget`` shows."""
        area = widget.parentWidget()
        while area is not None and not isinstance(area, QtWidgets.QScrollArea):
            area = area.parentWidget()
        if area is not None:
            area.ensureWidgetVisible(widget)
        pump(0.3)

    manager = MainWindowManager(project_files_dialog_factory=project_files_dialog)
    manager.ui.resize(*_WINDOW_SIZE)
    manager.ui.show()
    pump(1.0)
    save(manager.ui, "welcome")

    in_modal("new_project", manager.actionNewProject.trigger, accept=True)
    pump_until(lambda: len(manager.pages()) == len(names), 60.0, "the project")
    pdf_folder = work_dir / "scans"
    pdf_folder.mkdir()
    for path in (book.pdf, *book.pages[:2]):
        shutil.copy(path, pdf_folder)
    capture_open_pdf(
        app,
        ProjectFilesDialog(
            input_dir=pdf_folder, output_dir=work_dir / "out", parent=manager.ui
        ),
        lambda image: image.save(str(shots / "open_pdf.png")),
        pump,
    )

    show(1, SIDEWAYS, "stage1_fix_orientation")
    manager.current_page = page_named(manager, OFFCUT)
    visit_stage(1)
    capture_trim(
        options_child,
        finish_render,
        lambda: save(manager.ui, "stage1_fix_orientation_trim"),
    )

    show(2, SPREAD_LIGHT, "stage2_split_pages")
    show(3, SKEWED, "stage3_deskew")
    save_sorting_panel("thumbnails_incomplete")
    show(4, OFFCUT, "stage4_select_content")
    save_part(manager.filterList, "stage_play_button")
    capture_page_box_auto(
        options_child,
        finish_render,
        lambda: save(manager.ui, "stage4_select_content_page_box_auto"),
    )
    show(5, SINGLE, "stage5_margins")
    in_modal(
        "stage5_margins_apply",
        lambda: manager.show_apply_dialog(FilterStage.PAGE_LAYOUT, "margins"),
    )

    manager._select_stage_row(int(FilterStage.OUTPUT))
    manager.startBatchProcessing()
    pump_until(
        lambda: not manager.batch_runner.is_running, _BATCH_TIMEOUT_S, "the batch"
    )
    pump(1.0)
    if manager.batch_dialog is not None:
        save(manager.batch_dialog, "batch_progress")
        manager.batch_dialog.close()

    show(6, SINGLE, "stage6_output")
    method = options_child(QtWidgets.QComboBox, "methodCombo")
    fill_outside = options_child(QtWidgets.QCheckBox, "fillOutsidePageBoxCB")
    previous_method = method.currentIndex()
    method.setCurrentIndex(method.findText("Window"))
    fill_outside.setChecked(True)
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

    method.setCurrentIndex(method.findText("EdgePlus"))
    finish_render()
    reveal(options_child(QtWidgets.QWidget, "binarizationGroup"))
    save(manager.ui, "stage6_output_edge_methods")
    method.setCurrentIndex(previous_method)
    finish_render()

    manager.current_page = page_named(manager, PICTURES)
    visit_stage(6)
    options_child(QtWidgets.QAbstractButton, "mixedBtn").click()
    finish_render()
    reveal(options_child(QtWidgets.QWidget, "pictureShapeOptions"))
    save(manager.ui, "stage6_output_picture_shape")
    options_child(QtWidgets.QAbstractButton, "bwBtn").click()
    finish_render()

    manager.current_page = page_named(manager, CURVED)
    visit_stage(6)
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

    manager.current_page = page_named(manager, PICTURES)
    add_sample_zones(manager)
    visit_stage(6)
    view = output_view()
    for tab_name, shot in (
        ("PICTURE_ZONES", "stage6_tab_picture_zones"),
        ("FILL_ZONES", "stage6_tab_fill_zones"),
    ):
        view.set_current_tab(ImageViewTab[tab_name])
        pump(3.0)
        save(manager.ui, shot)

    manager.current_page = page_named(manager, SPECKS)
    visit_stage(6)
    output_view().set_current_tab(ImageViewTab.DESPECKLING)
    pump(3.0)
    save(manager.ui, "stage6_tab_despeckling")
    despeckle = options_child(QtWidgets.QSlider, "despeckleSlider")
    previous_level = despeckle.value()
    despeckle.setValue(despeckle.maximum())
    reveal(options_child(QtWidgets.QWidget, "despeckleGroup"))
    # The despeckling tab delays the render until the slider has rested.
    pump(2.0)
    finish_render()
    save(manager.ui, "stage6_output_despeckle_levels")
    despeckle.setValue(previous_level)
    finish_render()
    output_view().set_current_tab(ImageViewTab.OUTPUT)

    manager.current_page = page_named(manager, PICTURES)
    visit_stage(6)
    view = output_view()
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

    page_menu = manager.thumbnails.page_menu(manager.current_page)
    page_menu.popup(manager.ui.mapToGlobal(QtCore.QPoint(120, 240)))
    pump(0.5)
    save(page_menu, "thumbnail_menu")
    page_menu.hide()
    page_menu.deleteLater()
    in_modal("remove_pages", lambda: manager.remove_pages([manager.current_page]))

    show(3, OBLIQUE)
    options_child(QtWidgets.QAbstractButton, "obliqueAutoBtn").click()
    finish_render()
    pump(4.0)
    oblique = options_child(QtWidgets.QDoubleSpinBox, "obliqueSpinBox").value()
    assert abs(oblique) > 1.0, f"the oblique page was not detected: {oblique}"
    save(manager.ui, "stage3_deskew_oblique")

    capture_split_and_guides(manager, show, visit_stage, shots, save, pump)

    in_modal("settings", manager.actionSettings.trigger)
    in_modal("default_parameters", manager.actionDefaults.trigger)
    in_modal("export_pdf", manager.exportPdf)
    in_modal("fix_dpi", manager.actionFixDpi.trigger, prepare=show_all_pages_tab)
    in_modal("relinking", manager.actionRelinking.trigger, prepare=select_first_row)

    manager.modified = False
    manager.ui.close()
    pump(0.2)
    shutil.rmtree(work_dir, ignore_errors=True)

    _assert_complete(shots)


def select_first_row(dialog: QtWidgets.QWidget) -> None:
    """Select the first row of the dialog's list, so its details show.

    Args:
        dialog (QtWidgets.QWidget): A dialog with one item view.
    """
    view = dialog.findChild(QtWidgets.QAbstractItemView)
    assert view is not None
    view.setCurrentIndex(view.model().index(0, 0))


def show_all_pages_tab(dialog: QtWidgets.QWidget) -> None:
    """Turn the Fix DPI dialog to All Pages, which lists the sample's pages.

    The sample pages all carry a resolution, so the Need Fixing tab is empty.

    Args:
        dialog (QtWidgets.QWidget): The Fix DPI dialog.
    """
    tabs = dialog.findChild(QtWidgets.QTabWidget, "tabWidget")
    view = dialog.findChild(QtWidgets.QTreeView, "allPagesView")
    assert tabs is not None and view is not None
    tabs.setCurrentIndex(1)
    view.setCurrentIndex(view.model().index(0, 0))
    view.resizeColumnToContents(0)
    dialog.resize(max(dialog.width(), 460), max(dialog.height(), 420))


def capture_open_pdf(
    app: QtWidgets.QApplication,
    dialog: ProjectFilesDialog,
    save_image: Callable[[QtGui.QPixmap], object],
    pump: Callable[[float], None],
) -> None:
    """Pick the sample PDF in the New Project dialog and save its Extract prompt.

    The prompt is painted over the dialog that opened it, as the user sees them.
    It is cancelled, so nothing is rendered.

    Args:
        app (QtWidgets.QApplication): The application, to find the prompt.
        dialog (ProjectFilesDialog): A New Project dialog on the sample folder.
        save_image (Callable[[QtGui.QPixmap], object]): Writes the shot.
        pump (Callable[[float], None]): Runs the event loop for some seconds.
    """
    dialog.show()
    pump(0.5)
    files = dialog.findChild(QtWidgets.QListView, "offProjectList")
    assert files is not None
    files.setCurrentIndex(files.model().index(0, 0))
    assert str(files.currentIndex().data()).startswith("sample_book.pdf")

    def shoot() -> None:
        prompt = app.activeModalWidget()
        if prompt is None or prompt is dialog:
            QtCore.QTimer.singleShot(100, shoot)
            return
        pump(0.5)
        base = dialog.grab()
        overlay = prompt.grab()
        # Below the PDF's row, so the list stays readable under the prompt.
        row = files.visualRect(files.model().index(0, 0))
        top = files.viewport().mapTo(dialog, row.bottomLeft()).y() + 2 * row.height()
        corner = QtCore.QPoint(max(0, (base.width() - overlay.width()) // 2), top)
        shot = QtGui.QPixmap(
            base.size().expandedTo(
                QtCore.QSize(
                    corner.x() + overlay.width(), corner.y() + overlay.height()
                )
            )
        )
        shot.fill(dialog.palette().color(QtGui.QPalette.ColorRole.Window))
        painter = QtGui.QPainter(shot)
        painter.drawPixmap(0, 0, base)
        painter.drawPixmap(corner, overlay)
        painter.setPen(dialog.palette().color(QtGui.QPalette.ColorRole.Mid))
        painter.drawRect(QtCore.QRect(corner, overlay.size()).adjusted(0, 0, -1, -1))
        painter.end()
        save_image(shot)
        prompt.reject()

    QtCore.QTimer.singleShot(300, shoot)
    add = dialog.findChild(QtWidgets.QToolButton, "addToProjectBtn")
    assert add is not None
    add.click()
    dialog.reject()
    dialog.deleteLater()


def capture_page_box_auto(options_child, finish_render, save_auto) -> None:
    """Set the page box to Auto and save the window, then restore its mode."""
    modes = [
        options_child(QtWidgets.QAbstractButton, name)
        for name in ("pageDetectDisableBtn", "pageDetectAutoBtn", "pageDetectManualBtn")
    ]
    previous = next(button for button in modes if button.isChecked())
    modes[1].click()
    finish_render()
    save_auto()
    previous.click()
    finish_render()


def capture_trim(options_child, finish_render, save_trim) -> None:
    """Turn on Fix Orientation's trim and save the window, then turn it off again."""
    enable = options_child(QtWidgets.QCheckBox, "trimEnabledCheck")
    enable.setChecked(True)
    for name, pixels in (
        ("trimLeftSpin", 90),
        ("trimTopSpin", 100),
        ("trimRightSpin", 90),
        ("trimBottomSpin", 100),
    ):
        options_child(QtWidgets.QSpinBox, name).setValue(pixels)
    finish_render()
    save_trim()
    enable.setChecked(False)
    finish_render()


def add_sample_zones(manager: MainWindowManager) -> None:
    """Give the current page a picture zone and a fill zone."""
    page = manager.current_page
    output_filter = manager.stages.output_filter
    picture = PictureZoneSpec(
        points=((140.0, 480.0), (740.0, 480.0), (740.0, 670.0), (140.0, 670.0)),
        properties=PictureZoneProperties(layer=PictureLayer.PAINTER2),
    )
    fill = FillZoneSpec(
        points=((620.0, 1080.0), (820.0, 1080.0), (820.0, 1170.0), (620.0, 1170.0)),
        properties=FillZoneProperties(color=(255, 240, 200)),
    )
    output_filter.set_params(
        page,
        output_filter.get_params(page)
        .with_picture_zones((picture,))
        .with_fill_zones((fill,)),
    )


def capture_split_and_guides(manager, show, visit_stage, shots, save, pump) -> None:
    """Tilt a split line, drop one half of a spread, and place two guides."""
    spread = page_named(manager, SPREAD_LIGHT)
    split_filter = manager.stages.page_split_filter
    layout = split_filter.get_page_layout(spread)
    assert layout is not None
    x, y, width, height = layout.outline
    line = ((x + width * 0.45, y), (x + width * 0.55, y + height))
    tilted = PageLayout(
        outline=layout.outline,
        layout_type=PageLayoutType.TWO_PAGES,
        cutter1=line,
        cutter2=line,
    )
    split_filter.set_manual_layout(spread, tilted)
    show(2, SPREAD_LIGHT, "stage2_split_pages_tilted")

    dark = page_named(manager, SPREAD_DARK)
    project = manager.project
    position = next(
        n for n, info in enumerate(project.images) if info.id == dark.image_id
    )
    project.images[position] = project.images[position].model_copy(
        update={"num_sub_pages": 1, "left_half_removed": True}
    )
    manager._refresh_page_list(dark)
    manager.current_page = page_named(manager, SPREAD_DARK)
    split_filter.set_manual_layout(manager.current_page, tilted)
    visit_stage(2)
    save(manager.ui, "stage2_split_pages_removed_half")
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
    show(5, SINGLE, "stage5_margins_guides")


def _assert_complete(shots: Path) -> None:
    """Check that every named shot was written and has real content."""
    for name in _shot_names():
        path = shots / f"{name}.png"
        assert path.is_file(), f"missing {path.name}"
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"))
        assert pixels.std() > 5, f"{path.name} is blank"
