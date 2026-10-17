"""The Select Content width and height fields show and resize the page box."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QWidget
from pytestqt.qtbot import QtBot

from scantailor.app.ui.image_views.select_content import SelectContentImageView
from scantailor.app.ui.options_widgets import select_content
from scantailor.app.ui.stage_panels import StagePanels
from scantailor.app.ui.units_provider import UnitsProvider
from scantailor.core.models import ImageId, PageId, SubPage
from scantailor.core.settings import Units
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.select_content.content_box import ContentBox, PageBox
from scantailor.filters.select_content.detection import PageDetectionMode

IMAGE = ImageId(file_path=Path("scan.png"))
PAGE = PageId(image_id=IMAGE, sub_page=SubPage.SINGLE_PAGE)
DPI = 254.0  # 10 pixels per millimetre
MANUAL_PAGE = PageBox(x=20.0, y=30.0, width=1000.0, height=1500.0)


def _spin(widget: QWidget, name: str) -> QDoubleSpinBox:
    spin = widget.findChild(QDoubleSpinBox, name)
    assert spin is not None
    return spin


@pytest.fixture
def provider() -> UnitsProvider:
    return UnitsProvider(Units.MILLIMETERS)


@pytest.fixture
def stages() -> StageSequence:
    sequence = StageSequence()
    sequence.select_content_filter.set_manual_page(PAGE, MANUAL_PAGE)
    return sequence


@pytest.fixture
def widget(
    qtbot: QtBot, stages: StageSequence, provider: UnitsProvider
) -> select_content.OptionsWidget:
    options = select_content.OptionsWidget(
        stages.select_content_filter, units_provider=provider
    )
    qtbot.addWidget(options)
    options.set_page_dpi(DPI, DPI)
    options.set_current_page(PAGE)
    return options


class TestShownSize:
    def test_the_fields_show_the_page_box_size_in_millimetres(
        self, widget: select_content.OptionsWidget
    ) -> None:
        assert _spin(widget, "widthSpinBox").value() == pytest.approx(100.0)
        assert _spin(widget, "heightSpinBox").value() == pytest.approx(150.0)

    def test_switching_to_pixels_shows_the_stored_pixels(
        self, widget: select_content.OptionsWidget, provider: UnitsProvider
    ) -> None:
        provider.set_units(Units.PIXELS)

        assert _spin(widget, "widthSpinBox").value() == pytest.approx(1000.0)
        assert _spin(widget, "heightSpinBox").value() == pytest.approx(1500.0)

    def test_switching_units_leaves_the_page_box_alone(
        self,
        widget: select_content.OptionsWidget,
        provider: UnitsProvider,
        stages: StageSequence,
    ) -> None:
        provider.set_units(Units.INCHES)

        assert stages.select_content_filter.get_params(PAGE).page_box == MANUAL_PAGE


class TestEditing:
    def test_typing_a_width_resizes_the_page_box_from_its_top_left_corner(
        self, widget: select_content.OptionsWidget, stages: StageSequence
    ) -> None:
        _spin(widget, "widthSpinBox").setValue(80.0)

        page_box = stages.select_content_filter.get_params(PAGE).page_box
        assert (page_box.x, page_box.y) == (MANUAL_PAGE.x, MANUAL_PAGE.y)
        assert page_box.width == pytest.approx(800.0)
        assert page_box.height == pytest.approx(MANUAL_PAGE.height)

    def test_typing_a_height_in_inches_is_converted_to_pixels(
        self,
        widget: select_content.OptionsWidget,
        provider: UnitsProvider,
        stages: StageSequence,
    ) -> None:
        provider.set_units(Units.INCHES)

        _spin(widget, "heightSpinBox").setValue(5.0)

        page_box = stages.select_content_filter.get_params(PAGE).page_box
        assert page_box.height == pytest.approx(5.0 * DPI)

    def test_the_edit_is_announced_for_the_page(
        self, qtbot: QtBot, widget: select_content.OptionsWidget
    ) -> None:
        with qtbot.waitSignal(widget.page_box_changed) as blocker:
            _spin(widget, "widthSpinBox").setValue(90.0)

        assert blocker.args == [PAGE]

    def test_the_page_box_still_grows_to_hold_the_content_box(
        self, widget: select_content.OptionsWidget, stages: StageSequence
    ) -> None:
        filter_ = stages.select_content_filter
        filter_.set_manual_content(
            PAGE, ContentBox(x=100.0, y=100.0, width=700.0, height=900.0)
        )

        _spin(widget, "widthSpinBox").setValue(10.0)

        page_box = filter_.get_params(PAGE).page_box
        assert page_box.right == pytest.approx(800.0)
        assert _spin(widget, "widthSpinBox").value() == pytest.approx(78.0)

    def test_showing_another_page_does_not_write_a_page_box(
        self, widget: select_content.OptionsWidget, stages: StageSequence
    ) -> None:
        other = PageId(image_id=ImageId(file_path=Path("b.png")))

        widget.set_current_page(other)

        assert stages.select_content_filter.get_params(other).page_box.is_empty()


class TestVisibility:
    @pytest.mark.parametrize(
        ("mode", "fields", "fine_tune"),
        [
            (PageDetectionMode.MANUAL, True, False),
            (PageDetectionMode.AUTO, False, True),
        ],
    )
    def test_the_size_shows_for_a_manual_box_and_fine_tuning_for_an_automatic_one(
        self,
        widget: select_content.OptionsWidget,
        stages: StageSequence,
        mode: PageDetectionMode,
        fields: bool,
        fine_tune: bool,
    ) -> None:
        filter_ = stages.select_content_filter
        params = filter_.get_params(PAGE)
        filter_.set_params(
            PAGE, params.model_copy(update={"page_detection_mode": mode})
        )

        widget.set_current_page(PAGE)

        dimensions = widget.findChild(QWidget, "dimensionsWidget")
        fine_tune_check = widget.findChild(QCheckBox, "fineTuneBtn")
        assert dimensions is not None
        assert fine_tune_check is not None
        assert dimensions.isVisibleTo(widget) is fields
        assert fine_tune_check.isVisibleTo(widget) is fine_tune


@pytest.fixture
def shared_units() -> Iterator[UnitsProvider]:
    UnitsProvider.reset_instance()
    shared = UnitsProvider.get_instance()
    shared.set_units(Units.PIXELS)
    yield shared
    UnitsProvider.reset_instance()


@pytest.fixture
def panels(qapp: object, shared_units: UnitsProvider) -> Iterator[StagePanels]:
    sequence = StageSequence()
    sequence.select_content_filter.set_manual_page(PAGE, MANUAL_PAGE)
    panels = StagePanels(FilterStage.SELECT_CONTENT, sequence)
    panels._page = PAGE
    panels.options.set_current_page(PAGE)  # type: ignore[attr-defined]
    panels._adopt_source(IMAGE, np.full((2000, 1500), 255, dtype=np.uint8), DPI)
    panels._stage_image = np.zeros((2000, 1500), dtype=np.uint8)
    panels._refresh_geometry()
    yield panels
    panels.view.viewport().removeEventFilter(panels)


class TestPanelSync:
    def test_dragging_the_page_box_updates_the_fields(
        self, panels: StagePanels
    ) -> None:
        view = panels.view
        assert isinstance(view, SelectContentImageView)

        view.manual_page_rect_set.emit(QtCore.QRectF(20.0, 30.0, 640.0, 480.0))

        assert _spin(panels.options, "widthSpinBox").value() == pytest.approx(640.0)
        assert _spin(panels.options, "heightSpinBox").value() == pytest.approx(480.0)

    def test_typing_a_size_moves_the_box_shown_in_the_view(
        self, panels: StagePanels
    ) -> None:
        view = panels.view
        assert isinstance(view, SelectContentImageView)

        _spin(panels.options, "heightSpinBox").setValue(700.0)

        assert view.page_rect == QtCore.QRectF(20.0, 30.0, 1000.0, 700.0)
