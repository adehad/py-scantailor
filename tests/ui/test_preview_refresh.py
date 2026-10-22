"""The main window's image view repaints as soon as a new render reaches it."""

import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.base import ImageViewBase
from scantailor.app.ui.image_views.output import ImageViewTab, OutputImageView
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.stage_sequence import FilterStage
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

RENDER_TIMEOUT_MS = 60_000
PAINT_TIMEOUT_MS = 2_000
CONTENT_BOX = ContentBox(x=100, y=200, width=300, height=400)


class PaintCounter(QtCore.QObject):
    """Counts the paint events a widget receives."""

    def __init__(self, widget: QtWidgets.QWidget) -> None:
        """Start counting the paints of a widget.

        Args:
            widget (QtWidgets.QWidget): The widget to watch.
        """
        super().__init__(widget)
        self.paints = 0
        widget.installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Count a paint and let the widget handle the event.

        Args:
            watched (QtCore.QObject): The watched widget.
            event (QtCore.QEvent): The event about to reach it.

        Returns:
            bool: Always False, so the event is delivered.
        """
        if event.type() == QtCore.QEvent.Type.Paint:
            self.paints += 1
        return False


def _settle_every_page(window: MainWindowManager) -> None:
    """Fix each page's geometry so that every stage, Output included, may render."""
    stages = window.stages
    for page in window.pages():
        stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
        stages.deskew_filter.set_manual_angle(page, 0.0)
        stages.select_content_filter.set_manual_content(page, CONTENT_BOX)


def _open_stage(
    qtbot: QtBot, window: MainWindowManager, stage: FilterStage
) -> ImageViewBase:
    getattr(window, f"actionSwitchFilter{int(stage) + 1}").trigger()
    view = window._panels.view
    qtbot.waitUntil(
        lambda: view.image is not None and not window.render_pending,
        timeout=RENDER_TIMEOUT_MS,
    )
    # Let the first paint and the full-quality redraw pass before counting.
    qtbot.wait(ImageViewBase.HQ_TRANSFORM_DELAY_MS * 2)
    return view


@pytest.fixture
def settled_window(project_window: MainWindowManager) -> MainWindowManager:
    """Project window whose three pages can be shown at every stage."""
    _settle_every_page(project_window)
    return project_window


@pytest.mark.parametrize("stage", list(FilterStage), ids=lambda stage: stage.name)
def test_selecting_another_page_repaints_the_view_with_its_render(
    qtbot: QtBot, settled_window: MainWindowManager, stage: FilterStage
):
    window = settled_window
    view = _open_stage(qtbot, window, stage)
    shown_before = view.image
    counter = PaintCounter(view.viewport())
    other_page = window.pages()[1]
    assert other_page != window.current_page

    with qtbot.waitSignal(window.renderer.render_done, timeout=RENDER_TIMEOUT_MS):
        window.thumbnails.set_current_page(other_page)
    qtbot.waitUntil(lambda: view.image is not shown_before, timeout=RENDER_TIMEOUT_MS)

    assert window._panels.view is view
    assert view.image.constBits() != shown_before.constBits()
    qtbot.waitUntil(lambda: counter.paints > 0, timeout=PAINT_TIMEOUT_MS)


def test_turning_the_page_repaints_the_fix_orientation_view(
    qtbot: QtBot, settled_window: MainWindowManager
):
    view = _open_stage(qtbot, settled_window, FilterStage.FIX_ORIENTATION)
    counter = PaintCounter(view.viewport())

    settled_window._panels.options.findChild(
        QtWidgets.QToolButton, "rotateRightBtn"
    ).click()

    assert view.rotation.degrees == 90
    qtbot.waitUntil(lambda: counter.paints > 0, timeout=PAINT_TIMEOUT_MS)


@pytest.mark.parametrize(
    "tab", [ImageViewTab.FILL_ZONES, ImageViewTab.PICTURE_ZONES, ImageViewTab.OUTPUT]
)
def test_switching_an_output_tab_repaints_the_view(
    qtbot: QtBot, settled_window: MainWindowManager, tab: ImageViewTab
):
    # Outside Mixed mode the Picture Zones tab shows a message instead of the page.
    output = settled_window.stages.output_filter
    for page in settled_window.pages():
        output.set_params(
            page, output.get_params(page).with_color_mode(ColorMode.MIXED)
        )
    view = _open_stage(qtbot, settled_window, FilterStage.OUTPUT)
    assert isinstance(view, OutputImageView)
    if view.current_tab() == tab:
        view.set_current_tab(ImageViewTab.FILL_ZONES)
        qtbot.wait(ImageViewBase.HQ_TRANSFORM_DELAY_MS * 2)
    counter = PaintCounter(view.viewport())

    view.set_current_tab(tab)

    qtbot.waitUntil(lambda: counter.paints > 0, timeout=PAINT_TIMEOUT_MS)
