"""The Picture Zones tab says zones apply only in Mixed mode, and edits nothing then."""

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.output import (
    ImageViewTab,
    OutputImageView,
    ViewZone,
    ZoneKind,
)
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.models import Dpi
from scantailor.core.stage_sequence import FilterStage
from scantailor.core.transformation import ImageTransformation
from scantailor.filters.output.params import ColorMode
from scantailor.filters.page_layout.params import ContentRect, ContentSize

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800
LEFT = QtCore.Qt.MouseButton.LeftButton
MESSAGE = "Picture zones are only available in Mixed mode."
RENDER_TIMEOUT_MS = 60_000


def _triangle() -> ViewZone:
    return ViewZone(
        ZoneKind.PICTURE,
        QtGui.QPolygonF(
            [
                QtCore.QPointF(50, 400),
                QtCore.QPointF(150, 400),
                QtCore.QPointF(100, 500),
            ]
        ),
    )


@pytest.fixture
def view(qtbot: QtBot) -> OutputImageView:
    """An Output view of a white page with the Picture Zones tab open."""
    widget = OutputImageView()
    qtbot.addWidget(widget)
    widget.resize(QtCore.QSize(400, 500))
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_image(
        np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8),
        transformation=ImageTransformation.from_size(
            IMAGE_WIDTH, IMAGE_HEIGHT, Dpi.uniform(300)
        ),
    )
    widget.set_tabs(list(ImageViewTab), ImageViewTab.PICTURE_ZONES)
    return widget


def _message(view: OutputImageView) -> QtWidgets.QLabel:
    label = view.findChild(QtWidgets.QLabel, "outputTabMessage")
    assert label is not None
    return label


def _click(qtbot: QtBot, view: OutputImageView, x: int, y: int) -> None:
    qtbot.mouseClick(view.viewport(), LEFT, pos=QtCore.QPoint(x, y))


class TestOutputView:
    def test_a_click_on_the_tab_with_the_message_starts_no_zone(
        self, qtbot: QtBot, view: OutputImageView
    ) -> None:
        view.set_tab_message(ImageViewTab.PICTURE_ZONES, MESSAGE)

        _click(qtbot, view, 100, 100)
        _click(qtbot, view, 200, 100)
        _click(qtbot, view, 200, 200)
        qtbot.keyClick(view, QtCore.Qt.Key.Key_Return)

        assert view.zones() == []
        assert _message(view).isVisible()
        assert _message(view).text() == MESSAGE

    def test_delete_on_the_tab_with_the_message_keeps_the_zone(
        self, qtbot: QtBot, view: OutputImageView
    ) -> None:
        view.set_zones([_triangle()])
        view.set_tab_message(ImageViewTab.PICTURE_ZONES, MESSAGE)

        qtbot.keyClick(view, QtCore.Qt.Key.Key_Delete)

        assert len(view.zones()) == 1

    def test_the_message_drops_a_zone_being_drawn(
        self, qtbot: QtBot, view: OutputImageView
    ) -> None:
        _click(qtbot, view, 100, 100)
        _click(qtbot, view, 200, 100)

        view.set_tab_message(ImageViewTab.PICTURE_ZONES, MESSAGE)
        view.set_tab_message(ImageViewTab.PICTURE_ZONES, None)
        _click(qtbot, view, 200, 200)
        qtbot.keyClick(view, QtCore.Qt.Key.Key_Return)

        assert view.zones() == []

    def test_without_the_message_the_tab_draws_zones_again(
        self, qtbot: QtBot, view: OutputImageView
    ) -> None:
        view.set_tab_message(ImageViewTab.PICTURE_ZONES, MESSAGE)
        view.set_tab_message(ImageViewTab.PICTURE_ZONES, None)

        _click(qtbot, view, 100, 100)
        _click(qtbot, view, 200, 100)
        _click(qtbot, view, 200, 200)
        qtbot.keyClick(view, QtCore.Qt.Key.Key_Return)

        assert [zone.kind for zone in view.zones()] == [ZoneKind.PICTURE]
        assert not _message(view).isVisible()


@pytest.mark.slow
def test_the_tab_shows_the_message_outside_mixed_mode_only(
    qtbot: QtBot, project_window: MainWindowManager
) -> None:
    getattr(
        project_window, f"actionSwitchFilter{int(FilterStage.OUTPUT) + 1}"
    ).trigger()
    layout = project_window.stages.page_layout_filter.settings
    for page in project_window.pages():
        layout.update_content_size_and_get_aggregate(
            page,
            ContentRect(width=600, height=800),
            ContentRect(width=500, height=700),
            ContentSize(width_mm=100, height_mm=150),
        )
    with qtbot.waitSignal(
        project_window.renderer.output_extras_found, timeout=RENDER_TIMEOUT_MS
    ):
        project_window._request_render()
    panels = project_window._panels
    assert panels is not None
    view = panels.view
    assert isinstance(view, OutputImageView)
    qtbot.waitUntil(lambda: view.image is not None, timeout=RENDER_TIMEOUT_MS)
    view.set_current_tab(ImageViewTab.PICTURE_ZONES)
    page = project_window.current_page
    assert page is not None
    output = project_window.stages.output_filter
    assert output.get_params(page).color_mode != ColorMode.MIXED
    assert _message(view).isVisibleTo(view)
    assert _message(view).text() == MESSAGE

    mixed = panels.options.findChild(QtWidgets.QPushButton, "mixedBtn")
    assert mixed is not None
    mixed.click()

    qtbot.waitUntil(
        lambda: not _message(view).isVisibleTo(view), timeout=RENDER_TIMEOUT_MS
    )
