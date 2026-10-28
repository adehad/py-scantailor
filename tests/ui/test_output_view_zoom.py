"""The Output preview keeps its zoom and scroll position when it is rebuilt."""

import pytest
from pytestqt.qtbot import QtBot

from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.core.models import Dpi
from scantailor.filters.output.color_mode import ColorMode
from tests.ui.conftest import RENDER_TIMEOUT_MS, flush_events, settle
from tests.ui.test_image_views_interaction import _open_output_view


def _rerender_after(qtbot: QtBot, window: MainWindowManager, edit) -> None:
    with qtbot.waitSignal(window.renderer.render_done, timeout=RENDER_TIMEOUT_MS):
        edit()
    settle(window)


@pytest.mark.slow
def test_zoom_survives_a_rerender_that_keeps_the_image_size(
    qtbot: QtBot, project_window: MainWindowManager
):
    view = _open_output_view(qtbot, project_window)
    view.set_zoom(2.0)
    zoomed = view.zoom_level
    options = project_window._panels.options

    _rerender_after(
        qtbot,
        project_window,
        lambda: options._edit(lambda p: p.with_color_mode(ColorMode.COLOR_GRAYSCALE)),
    )

    assert view.zoom_level == pytest.approx(zoomed)


@pytest.mark.slow
def test_the_zoomed_part_of_the_page_survives_another_output_resolution(
    qtbot: QtBot, project_window: MainWindowManager
):
    view = _open_output_view(qtbot, project_window)
    view.set_zoom(2.0)
    shown_width = view.zoom_level * view.virtual_display_area.width()
    options = project_window._panels.options
    current = project_window.current_page
    other = Dpi.uniform(
        300
        if project_window.stages.output_filter.get_params(current).output_dpi
        != Dpi.uniform(300)
        else 200
    )

    _rerender_after(
        qtbot,
        project_window,
        lambda: options._edit(lambda p: p.with_output_dpi(other)),
    )

    assert view.zoom_level * view.virtual_display_area.width() == pytest.approx(
        shown_width
    )


@pytest.mark.slow
def test_scroll_position_survives_a_rerender_that_keeps_the_image_size(
    qtbot: QtBot, project_window: MainWindowManager
):
    view = _open_output_view(qtbot, project_window)
    view.set_zoom(3.0)
    flush_events()
    bar = view.verticalScrollBar()
    bar.setValue(bar.minimum() + (bar.maximum() - bar.minimum()) // 3)
    before = view.verticalScrollBar().value()
    assert before > 0
    options = project_window._panels.options

    _rerender_after(
        qtbot,
        project_window,
        lambda: options._edit(lambda p: p.with_color_mode(ColorMode.COLOR_GRAYSCALE)),
    )

    assert view.verticalScrollBar().value() == pytest.approx(before, abs=2)
