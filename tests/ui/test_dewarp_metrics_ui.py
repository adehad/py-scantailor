"""The Output stage shows what auto dewarping found and takes the minimum span."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.output import ImageViewTab, OutputImageView
from scantailor.app.ui.MainWindow import MainWindowManager
from scantailor.app.ui.options_widgets.output import OptionsWidget
from scantailor.app.ui.stage_panels import StagePanels
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import OutputExtras
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation
from scantailor.dewarping.auto_model import DewarpMetrics, ModelOutcome
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.output.filter import Filter
from scantailor.filters.output.params import Params
from scantailor.filters.page_layout.params import ContentRect, ContentSize
from tests.ui.conftest import RENDER_TIMEOUT_MS

PAGE = PageId(
    image_id=ImageId(file_path=Path("page.tif")), sub_page=SubPage.SINGLE_PAGE
)
OTHER_PAGE = PageId(
    image_id=ImageId(file_path=Path("other.tif")), sub_page=SubPage.SINGLE_PAGE
)
REFUSED = DewarpMetrics(
    lines_traced=33,
    lines_kept=2,
    edges_traced=0,
    outcome=ModelOutcome.SPAN_TOO_SMALL,
    min_span_percent=40.0,
    span_percent=7.0,
    top_from_edge=False,
    bottom_from_edge=False,
)
BUILT = DewarpMetrics(
    lines_traced=20,
    lines_kept=18,
    edges_traced=2,
    outcome=ModelOutcome.BUILT,
    min_span_percent=40.0,
    span_percent=104.0,
    top_from_edge=True,
    bottom_from_edge=True,
)


def _auto(min_span: int = 40) -> Params:
    return Params(
        dewarping=DewarpingOptions(mode=DewarpingMode.AUTO),
        dewarp_min_span_percent=min_span,
    )


class _Options:
    def __init__(self, qtbot: QtBot, params: Params) -> None:
        self.filter = Filter()
        self.filter.set_params(PAGE, params)
        self.widget = OptionsWidget(self.filter)
        qtbot.addWidget(self.widget)
        self.widget.set_current_page(PAGE)
        self.changed: list[PageId] = []
        self.widget.params_changed.connect(self.changed.append)

    @property
    def params(self) -> Params:
        return self.filter.get_params(PAGE)

    def find[T: QtWidgets.QWidget](self, kind: type[T], name: str) -> T:
        found = self.widget.findChild(kind, name)
        assert found is not None, name
        return found

    def shown(self, name: str) -> bool:
        return self.find(QtWidgets.QWidget, name).isVisibleTo(self.widget)


class TestOptionsPanel:
    def test_the_minimum_span_shows_the_stored_value_in_auto_mode_only(
        self, qtbot: QtBot
    ) -> None:
        auto = _Options(qtbot, _auto(min_span=35))
        off = _Options(qtbot, Params())

        assert auto.shown("dewarpMinSpanRow")
        assert auto.find(QtWidgets.QSpinBox, "dewarpMinSpanSB").value() == 35
        assert not off.shown("dewarpMinSpanRow")

    def test_the_spin_box_stores_the_minimum_span_after_the_delay(
        self, qtbot: QtBot
    ) -> None:
        options = _Options(qtbot, _auto())

        options.find(QtWidgets.QSpinBox, "dewarpMinSpanSB").setValue(60)

        assert options.params.dewarp_min_span_percent == 60
        assert options.changed == []
        options.widget.flush_pending_reload()
        assert options.changed == [PAGE]

    def test_switching_to_auto_reveals_the_minimum_span(self, qtbot: QtBot) -> None:
        options = _Options(qtbot, Params())

        options.find(QtWidgets.QPushButton, "dewarpAutoBtn").click()

        assert options.shown("dewarpMinSpanRow")

    def test_the_metrics_show_in_plain_words_once_given(self, qtbot: QtBot) -> None:
        options = _Options(qtbot, _auto())
        assert not options.shown("dewarpMetricsLabel")

        options.widget.set_dewarp_metrics(REFUSED)

        label = options.find(QtWidgets.QLabel, "dewarpMetricsLabel")
        assert options.shown("dewarpMetricsLabel")
        assert label.text() == REFUSED.describe()
        assert "Below the 40% minimum" in label.text()

    def test_the_metrics_hide_when_dewarping_is_not_automatic(
        self, qtbot: QtBot
    ) -> None:
        options = _Options(qtbot, _auto())
        options.widget.set_dewarp_metrics(BUILT)

        options.find(QtWidgets.QPushButton, "dewarpOffBtn").click()

        assert not options.shown("dewarpMetricsLabel")

    def test_taking_the_metrics_away_hides_them(self, qtbot: QtBot) -> None:
        options = _Options(qtbot, _auto())
        options.widget.set_dewarp_metrics(BUILT)

        options.widget.set_dewarp_metrics(None)

        assert not options.shown("dewarpMetricsLabel")

    def test_another_page_starts_without_the_metrics_of_the_last(
        self, qtbot: QtBot
    ) -> None:
        options = _Options(qtbot, _auto())
        options.filter.set_params(OTHER_PAGE, _auto())
        options.widget.set_dewarp_metrics(BUILT)

        options.widget.set_current_page(OTHER_PAGE)

        assert not options.shown("dewarpMetricsLabel")

    def test_the_same_page_keeps_them(self, qtbot: QtBot) -> None:
        options = _Options(qtbot, _auto())
        options.widget.set_dewarp_metrics(BUILT)

        options.widget.set_current_page(PAGE)

        assert options.shown("dewarpMetricsLabel")


@pytest.fixture
def view(qtbot: QtBot) -> OutputImageView:
    widget = OutputImageView()
    qtbot.addWidget(widget)
    widget.resize(QtCore.QSize(400, 500))
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_image(
        np.full((800, 600), 255, dtype=np.uint8),
        transformation=ImageTransformation.from_size(600, 800, Dpi.uniform(300)),
    )
    widget.set_tabs(list(ImageViewTab))
    return widget


class TestDewarpingTabNote:
    def test_the_note_shows_on_the_dewarping_tab_only(
        self, view: OutputImageView
    ) -> None:
        view.set_dewarp_note(REFUSED.describe())
        note = view.findChild(QtWidgets.QLabel, "dewarpMetricsNote")
        assert note is not None

        assert not note.isVisible()
        view.set_current_tab(ImageViewTab.DEWARPING)
        assert note.isVisible()
        assert view.dewarp_note() == REFUSED.describe()
        view.set_current_tab(ImageViewTab.OUTPUT)
        assert not note.isVisible()

    def test_the_note_lies_along_the_bottom_of_the_view(
        self, view: OutputImageView
    ) -> None:
        view.set_current_tab(ImageViewTab.DEWARPING)
        view.set_dewarp_note(BUILT.describe())
        note = view.findChild(QtWidgets.QLabel, "dewarpMetricsNote")
        assert note is not None

        assert note.geometry().bottom() == view.height() - 1
        assert note.width() == view.width()

    def test_the_note_follows_a_resize(self, view: OutputImageView) -> None:
        view.set_current_tab(ImageViewTab.DEWARPING)
        view.set_dewarp_note(BUILT.describe())
        note = view.findChild(QtWidgets.QLabel, "dewarpMetricsNote")
        assert note is not None

        view.resize(QtCore.QSize(300, 420))

        assert note.geometry().bottom() == view.height() - 1

    def test_an_empty_note_shows_nothing(self, view: OutputImageView) -> None:
        view.set_current_tab(ImageViewTab.DEWARPING)
        view.set_dewarp_note(BUILT.describe())

        view.set_dewarp_note(None)

        note = view.findChild(QtWidgets.QLabel, "dewarpMetricsNote")
        assert note is not None
        assert not note.isVisible()
        assert view.dewarp_note() == ""


class TestStagePanels:
    @pytest.fixture
    def panels(
        self, make_stage_panels: Callable[[FilterStage, StageSequence], StagePanels]
    ) -> StagePanels:
        sequence = StageSequence()
        sequence.output_filter.set_params(PAGE, _auto())
        panels = make_stage_panels(FilterStage.OUTPUT, sequence)
        panels.set_page(PAGE)
        return panels

    def test_the_extras_of_a_render_feed_the_panel_and_the_tab(
        self, panels: StagePanels
    ) -> None:
        page = np.full((100, 80), 200, dtype=np.uint8)

        panels.set_output_extras(OutputExtras(page, None, 300, dewarp_metrics=BUILT))

        options = panels.options
        assert isinstance(options, OptionsWidget)
        label = options.findChild(QtWidgets.QLabel, "dewarpMetricsLabel")
        assert label is not None
        assert label.text() == BUILT.describe()
        view = panels.view
        assert isinstance(view, OutputImageView)
        assert view.dewarp_note() == BUILT.describe()

    def test_extras_without_metrics_clear_both(self, panels: StagePanels) -> None:
        page = np.full((100, 80), 200, dtype=np.uint8)
        panels.set_output_extras(OutputExtras(page, None, 300, dewarp_metrics=BUILT))

        panels.set_output_extras(OutputExtras(page, None, 300))

        view = panels.view
        assert isinstance(view, OutputImageView)
        assert view.dewarp_note() == ""
        label = panels.options.findChild(QtWidgets.QLabel, "dewarpMetricsLabel")
        assert label is not None
        assert not label.isVisibleTo(panels.options)

    def test_leaving_the_page_clears_the_note(self, panels: StagePanels) -> None:
        page = np.full((100, 80), 200, dtype=np.uint8)
        panels.set_output_extras(OutputExtras(page, None, 300, dewarp_metrics=BUILT))

        panels.set_page(None)

        view = panels.view
        assert isinstance(view, OutputImageView)
        assert view.dewarp_note() == ""


@pytest.mark.slow
def test_the_note_shows_on_the_dewarping_tab_after_an_auto_render(
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
    project_window.renderer.wait_for_done()
    options = project_window._panels.options
    assert isinstance(options, OptionsWidget)
    auto_button = options.findChild(QtWidgets.QPushButton, "dewarpAutoBtn")
    assert auto_button is not None

    with qtbot.waitSignal(
        project_window.renderer.output_extras_found, timeout=RENDER_TIMEOUT_MS
    ):
        auto_button.click()
    view = project_window._panels.view
    assert isinstance(view, OutputImageView)
    qtbot.waitUntil(lambda: view.dewarp_note() != "", timeout=RENDER_TIMEOUT_MS)
    view.set_current_tab(ImageViewTab.DEWARPING)

    note = view.findChild(QtWidgets.QLabel, "dewarpMetricsNote")
    assert note is not None
    assert note.isVisible()
    assert note.geometry().bottom() == view.height() - 1
    assert "lines" in note.text()
