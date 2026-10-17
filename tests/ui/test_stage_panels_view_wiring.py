"""What the stage panels hand to the views: halves, content masks, tab pictures."""

import gc
from collections.abc import Iterator
from pathlib import Path

import cv2
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
from scantailor.app.ui.image_views.page_layout import PageLayoutImageView
from scantailor.app.ui.image_views.page_split import PageSplitImageView
from scantailor.app.ui.image_views.select_content import SelectContentImageView
from scantailor.app.ui.stage_panels import StagePanels, _fill_zone_mapper, _Source
from scantailor.core.content_mask import ContentMask
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import OutputExtras
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.transformation import ImageTransformation
from scantailor.core.zones import FillZoneProperties
from scantailor.filters.output.dewarping_options import DewarpingMode
from scantailor.filters.output.params import DistortionModelSpec, FillZoneSpec
from scantailor.filters.output.params import Params as OutputParams
from scantailor.filters.page_layout.guide import Guide as StoredGuide
from scantailor.filters.page_split.page_layout import PageLayout

pytestmark = pytest.mark.usefixtures("qapp")

IMAGE = ImageId(file_path=Path("scan.png"))
PAGE = PageId(image_id=IMAGE, sub_page=SubPage.SINGLE_PAGE)
HEIGHT = 80
WIDTH = 60
MODEL = DistortionModelSpec(
    top=((5.0, 8.0), (30.0, 6.0), (55.0, 8.0)),
    bottom=((5.0, 70.0), (30.0, 74.0), (55.0, 70.0)),
)


_made: list[StagePanels] = []


def _panels(stage: FilterStage) -> StagePanels:
    panels = StagePanels(stage, StageSequence())
    _made.append(panels)
    panels._page = PAGE
    panels.options.set_current_page(PAGE)  # type: ignore[attr-defined]
    return panels


@pytest.fixture(autouse=True)
def _release_panels() -> Iterator[None]:
    """Stop the panels watching their viewports before they are collected."""
    yield
    while _made:
        panels = _made.pop()
        panels.view.viewport().removeEventFilter(panels)


def _source() -> _Source:
    pixels = np.full((HEIGHT * 10, WIDTH * 10), 255, dtype=np.uint8)
    pixels[300:340, 100:500] = 20
    return _Source(IMAGE, pixels, 300.0)


class TestOutputGridEdits:
    @pytest.fixture
    def panels(self) -> StagePanels:
        panels = _panels(FilterStage.OUTPUT)
        panels._output_transformation = ImageTransformation.from_size(
            WIDTH, HEIGHT, Dpi.uniform(300)
        )
        return panels

    def _mode(self, panels: StagePanels) -> DewarpingMode:
        return panels.sequence.output_filter.get_params(PAGE).dewarping.mode

    def test_editing_the_grid_while_dewarping_is_off_turns_it_to_manual(self, panels):
        assert self._mode(panels) is DewarpingMode.OFF

        panels._on_distortion_model_edited(MODEL)

        assert self._mode(panels) is DewarpingMode.MANUAL

    def test_the_edited_model_is_stored_with_the_mode(self, panels):
        panels._on_distortion_model_edited(MODEL)

        stored = panels.sequence.output_filter.get_params(PAGE).distortion_model
        assert np.array(stored.top) == pytest.approx(np.array(MODEL.top))

    @pytest.mark.parametrize("mode", [DewarpingMode.AUTO, DewarpingMode.MARGINAL])
    def test_automatic_and_marginal_dewarping_turn_to_manual_too(self, panels, mode):
        filter_ = panels.sequence.output_filter
        params = filter_.get_params(PAGE)
        filter_.set_params(
            PAGE, params.with_dewarping(params.dewarping.with_mode(mode))
        )

        panels._on_distortion_model_edited(MODEL)

        assert self._mode(panels) is DewarpingMode.MANUAL

    def test_the_change_is_reported_once(self, panels):
        reported: list[object] = []
        panels.settings_changed.connect(lambda: reported.append(1))
        options = panels.options
        options.params_changed.connect(lambda *_: None)  # type: ignore[attr-defined]

        panels._on_distortion_model_edited(MODEL)

        assert reported == [1]

    def test_the_dewarping_buttons_show_the_new_mode(self, panels):
        panels._on_distortion_model_edited(MODEL)

        assert panels.options._dewarp_manual_btn.isChecked()  # type: ignore[attr-defined]


class TestFillZonesTabPicture:
    @staticmethod
    def _feed(panels: StagePanels, dewarping: DewarpingMode) -> OutputImageView:
        filter_ = panels.sequence.output_filter
        params = filter_.get_params(PAGE)
        filter_.set_params(
            PAGE, params.with_dewarping(params.dewarping.with_mode(dewarping))
        )
        view = panels.view
        assert isinstance(view, OutputImageView)
        output = np.full((HEIGHT, WIDTH), 200, dtype=np.uint8)
        page = np.full((HEIGHT, WIDTH), 100, dtype=np.uint8)
        panels._stage_image = output
        view.set_image(output)
        panels.set_output_extras(OutputExtras(page, None, 300))
        xform = ImageTransformation.from_size(WIDTH, HEIGHT, Dpi.uniform(300))
        panels._feed_output_tabs(view, filter_.get_params(PAGE), xform, np.ones(2))
        return view

    @pytest.mark.parametrize(
        "mode", [DewarpingMode.OFF, DewarpingMode.AUTO, DewarpingMode.MANUAL]
    )
    def test_the_fill_zone_tab_shows_the_output_whether_or_not_it_is_dewarped(
        self, mode
    ):
        panels = _panels(FilterStage.OUTPUT)

        view = self._feed(panels, mode)

        assert ImageViewTab.FILL_ZONES not in view._tab_images
        view.set_current_tab(ImageViewTab.FILL_ZONES)
        assert view.image is not None
        assert view.image.pixelColor(5, 5).red() == 200

    def test_a_new_page_drops_the_fill_zone_picture(self):
        panels = _panels(FilterStage.OUTPUT)
        view = self._feed(panels, DewarpingMode.AUTO)

        panels._clear_overlays()

        assert ImageViewTab.FILL_ZONES not in view._tab_images


class TestFillZonesOnDewarpedOutput:
    """Zones are stored on the page and drawn on the output, as upstream does."""

    PAGE_POLYGON = ((20.0, 30.0), (40.0, 30.0), (40.0, 50.0), (20.0, 50.0))

    @staticmethod
    def _params(mode: DewarpingMode, model: DistortionModelSpec = MODEL):
        params = OutputParams().with_distortion_model(model)
        return params.with_dewarping(params.dewarping.with_mode(mode))

    @staticmethod
    def _xform() -> ImageTransformation:
        return ImageTransformation.from_size(WIDTH, HEIGHT, Dpi.uniform(300))

    def _mapper(self, mode: DewarpingMode = DewarpingMode.MANUAL):
        output = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        return _fill_zone_mapper(self._params(mode), self._xform(), np.ones(2), output)

    @pytest.mark.parametrize(
        ("mode", "model"),
        [
            (DewarpingMode.OFF, MODEL),
            (DewarpingMode.MANUAL, DistortionModelSpec()),
        ],
    )
    def test_there_is_no_mapper_without_dewarping_or_a_model(self, mode, model):
        output = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

        mapper = _fill_zone_mapper(
            self._params(mode, model), self._xform(), np.ones(2), output
        )

        assert mapper is None

    def test_a_dewarped_page_gets_a_mapper(self):
        assert self._mapper() is not None

    def test_a_fill_zone_is_drawn_where_the_dewarp_puts_the_page_polygon(self):
        mapper = self._mapper()
        assert mapper is not None
        spec = FillZoneSpec(
            points=self.PAGE_POLYGON, properties=FillZoneProperties(color=(0, 0, 0))
        )

        zone = StagePanels._view_zone(
            spec, ZoneKind.FILL, self._xform(), np.ones(2), mapper
        )

        shown = np.array([[p.x(), p.y()] for p in zone.polygon.toList()])
        assert shown == pytest.approx(
            mapper.to_dewarped(np.array(self.PAGE_POLYGON)), abs=1e-6
        )
        assert not shown == pytest.approx(np.array(self.PAGE_POLYGON), abs=0.5)

    def test_a_fill_zone_without_a_mapper_stays_on_the_page(self):
        spec = FillZoneSpec(
            points=self.PAGE_POLYGON, properties=FillZoneProperties(color=(0, 0, 0))
        )

        zone = StagePanels._view_zone(spec, ZoneKind.FILL, self._xform(), np.ones(2))

        shown = np.array([[p.x(), p.y()] for p in zone.polygon.toList()])
        assert shown == pytest.approx(np.array(self.PAGE_POLYGON), abs=1e-6)

    def test_a_zone_edited_on_the_output_is_stored_on_the_page(self):
        panels = _panels(FilterStage.OUTPUT)
        mapper = self._mapper()
        assert mapper is not None
        panels._output_transformation = self._xform()
        panels._output_scale = np.ones(2)
        panels._fill_zone_mapper = mapper
        view = panels.view
        assert isinstance(view, OutputImageView)
        shown = mapper.to_dewarped(np.array(self.PAGE_POLYGON))
        view.set_zones(
            [
                ViewZone(
                    ZoneKind.FILL,
                    QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in shown]),
                    (0, 0, 0),
                    FillZoneProperties(color=(0, 0, 0)),
                )
            ]
        )

        panels._on_zones_changed()

        (stored,) = panels.sequence.output_filter.get_params(PAGE).fill_zones
        assert np.array(stored.points) == pytest.approx(
            np.array(self.PAGE_POLYGON), abs=0.1
        )


class TestPageSplit:
    def test_removed_halves_are_looked_up_for_the_shown_image(self):
        panels = _panels(FilterStage.PAGE_SPLIT)
        panels._source = _source()
        panels._stage_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        asked: list[ImageId] = []

        def removed(image: ImageId) -> tuple[bool, bool]:
            asked.append(image)
            return True, False

        panels.set_removed_pages_provider(removed)
        panels.sequence.page_split_filter.set_params(
            PAGE,
            panels.sequence.page_split_filter.get_params(PAGE).with_manual_layout(
                PageLayout.two_pages(WIDTH * 10, HEIGHT * 10, 300)
            ),
        )
        panels._refresh_geometry()

        view = panels.view
        assert isinstance(view, PageSplitImageView)
        assert asked == [IMAGE]
        assert view._removed == {
            SubPage.LEFT_PAGE: True,
            SubPage.RIGHT_PAGE: False,
        }

    def test_asking_for_a_removed_page_back_names_the_page(self, qtbot: QtBot):
        panels = _panels(FilterStage.PAGE_SPLIT)

        with qtbot.waitSignal(panels.unremove_page_requested) as asked:
            panels.view.unremove_requested.emit(SubPage.LEFT_PAGE)  # type: ignore[attr-defined]

        assert asked.args == [PageId(image_id=IMAGE, sub_page=SubPage.LEFT_PAGE)]

    def test_without_a_provider_no_half_is_shown_removed(self):
        panels = _panels(FilterStage.PAGE_SPLIT)
        panels._source = _source()
        panels._stage_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

        panels._refresh_geometry()

        assert panels.view._removed == {  # type: ignore[attr-defined]
            SubPage.LEFT_PAGE: False,
            SubPage.RIGHT_PAGE: False,
        }


class TestContentMask:
    @pytest.mark.parametrize(
        "stage", [FilterStage.SELECT_CONTENT, FilterStage.PAGE_LAYOUT]
    )
    def test_the_view_gets_a_mask_provider_that_works_on_demand(self, stage):
        panels = _panels(stage)
        panels._source = _source()
        panels._stage_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

        panels._refresh_geometry()

        view = panels.view
        assert isinstance(view, SelectContentImageView | PageLayoutImageView)
        provider = view._mask_provider
        assert provider is not None
        mask = provider()
        assert isinstance(mask, ContentMask)
        assert not mask.is_null()
        assert mask.find_content_in_area(60, 140, 100, 60) is not None

    def test_dropped_panels_free_their_view_without_the_cycle_collector(self):
        panels = StagePanels(FilterStage.SELECT_CONTENT, StageSequence())
        panels._page = PAGE
        panels.options.set_current_page(PAGE)  # type: ignore[attr-defined]
        panels._source = _source()
        panels._stage_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        panels._refresh_geometry()
        destroyed: list[bool] = []
        panels.view.destroyed.connect(lambda: destroyed.append(True))
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            del panels
            assert destroyed
        finally:
            if was_enabled:
                gc.enable()

    def test_the_mask_is_computed_once_for_an_unchanged_page(self, monkeypatch):
        from scantailor.app.ui import stage_panels

        calls: list[int] = []
        original = stage_panels.compute_content_mask

        def counting(pixels, xform):
            calls.append(1)
            return original(pixels, xform)

        monkeypatch.setattr(stage_panels, "compute_content_mask", counting)
        panels = _panels(FilterStage.SELECT_CONTENT)
        panels._source = _source()
        panels._stage_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        panels._refresh_geometry()
        provider = panels.view._mask_provider  # type: ignore[attr-defined]

        first, second = provider(), provider()
        panels._refresh_geometry()
        third = panels.view._mask_provider()  # type: ignore[attr-defined]

        assert first is second is third
        assert calls == [1]

    def test_no_mask_without_the_source_image(self):
        panels = _panels(FilterStage.SELECT_CONTENT)
        panels._source = _source()
        panels._stage_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        panels._refresh_geometry()
        provider = panels.view._mask_provider  # type: ignore[attr-defined]
        panels._source = None

        assert provider() is None

    def test_the_page_layout_view_follows_the_alignment_of_the_page(self):
        panels = _panels(FilterStage.PAGE_LAYOUT)
        panels._source = _source()
        panels._stage_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        filter_ = panels.sequence.page_layout_filter
        params = filter_.get_params(PAGE)
        aligned = params.alignment

        panels._refresh_geometry()
        assert panels.view._aligned == aligned.aligned_with_others()  # type: ignore[attr-defined]

        from scantailor.filters.page_layout.alignment import Alignment

        filter_.set_params(
            PAGE,
            params.with_alignment(
                Alignment(
                    vertical=aligned.vertical,
                    horizontal=aligned.horizontal,
                    is_null=True,
                )
            ),
        )
        panels._refresh_geometry()

        assert not panels.view._aligned  # type: ignore[attr-defined]

    def test_the_link_state_is_shared_between_options_and_view(self):
        panels = _panels(FilterStage.PAGE_LAYOUT)

        assert panels.view.state is panels.options.state  # type: ignore[attr-defined]


class TestPageLayoutGuides:
    def test_a_guide_added_in_the_view_is_stored_in_the_settings(self):
        panels = _panels(FilterStage.PAGE_LAYOUT)

        panels.view.state.add_guide(is_horizontal=True, position_mm=3.5)  # type: ignore[attr-defined]

        stored = panels.sequence.page_layout_filter.settings.get_guides()
        assert [(g.position_mm, g.is_horizontal) for g in stored] == [(3.5, True)]

    def test_removing_the_guides_clears_the_settings(self):
        panels = _panels(FilterStage.PAGE_LAYOUT)
        state = panels.view.state  # type: ignore[attr-defined]
        state.add_guide(is_horizontal=False, position_mm=1.0)

        state.clear_guides()

        assert panels.sequence.page_layout_filter.settings.get_guides() == ()

    def test_stored_guides_are_shown_by_new_panels(self):
        sequence = StageSequence()
        sequence.page_layout_filter.settings.set_guides(
            [StoredGuide(position_mm=-2.0, is_horizontal=False)]
        )

        panels = StagePanels(FilterStage.PAGE_LAYOUT, sequence)
        _made.append(panels)

        guides = panels.view.state.guides  # type: ignore[attr-defined]
        assert [(g.position_mm, g.is_horizontal) for g in guides] == [(-2.0, False)]
        assert panels.options.state.guides == guides  # type: ignore[attr-defined]

    def test_hiding_the_hard_margins_rectangle_is_stored(self):
        panels = _panels(FilterStage.PAGE_LAYOUT)

        panels.view.state.middle_rect_shown = False  # type: ignore[attr-defined]

        assert not panels.sequence.page_layout_filter.settings.is_middle_rect_shown()

    def test_a_stored_choice_to_hide_the_rectangle_is_taken_up(self):
        sequence = StageSequence()
        sequence.page_layout_filter.settings.set_middle_rect_shown(False)

        panels = StagePanels(FilterStage.PAGE_LAYOUT, sequence)
        _made.append(panels)

        assert not panels.view.state.middle_rect_shown  # type: ignore[attr-defined]


def test_panels_are_plain_qt_objects():
    panels = _panels(FilterStage.PAGE_SPLIT)

    assert isinstance(panels, QtCore.QObject)


class TestDewarpingGridWaitsForItsTab:
    """Laying out the grid is slow on a large page, so only an open tab gets it."""

    @staticmethod
    def _show_output(panels: StagePanels) -> list[DistortionModelSpec]:
        """Show a rendered page with a stored model; return what the grid is fed."""
        filter_ = panels.sequence.output_filter
        filter_.set_params(PAGE, filter_.get_params(PAGE).with_distortion_model(MODEL))
        view = panels.view
        assert isinstance(view, OutputImageView)
        fed: list[DistortionModelSpec] = []
        feed = view.set_distortion_model

        def record(model: DistortionModelSpec, depth: float) -> None:
            fed.append(model)
            feed(model, depth)

        view.set_distortion_model = record  # type: ignore[method-assign]
        output = np.full((HEIGHT, WIDTH), 200, dtype=np.uint8)
        panels._stage_image = output
        view.set_image(output)
        panels.set_output_extras(OutputExtras(output, None, 300))
        xform = ImageTransformation.from_size(WIDTH, HEIGHT, Dpi.uniform(300))
        panels._output_transformation = xform
        panels._feed_output_tabs(view, filter_.get_params(PAGE), xform, np.ones(2))
        return fed

    def test_the_grid_is_not_laid_out_while_its_tab_is_closed(self):
        panels = _panels(FilterStage.OUTPUT)

        fed = self._show_output(panels)

        assert fed == []

    def test_opening_the_dewarping_tab_lays_out_the_stored_model(self):
        panels = _panels(FilterStage.OUTPUT)
        fed = self._show_output(panels)
        view = panels.view
        assert isinstance(view, OutputImageView)

        view.set_current_tab(ImageViewTab.DEWARPING)

        assert len(fed) == 1
        assert np.array(fed[0].top) == pytest.approx(np.array(MODEL.top))

    def test_opening_the_tab_again_does_not_lay_out_the_same_model_again(self):
        panels = _panels(FilterStage.OUTPUT)
        fed = self._show_output(panels)
        view = panels.view
        assert isinstance(view, OutputImageView)

        view.set_current_tab(ImageViewTab.DEWARPING)
        view.set_current_tab(ImageViewTab.OUTPUT)
        view.set_current_tab(ImageViewTab.DEWARPING)

        assert len(fed) == 1


def test_a_decode_that_ends_after_the_widgets_are_deleted_is_dropped(
    qtbot: QtBot, tmp_path: Path
):
    scan = tmp_path / "scan.png"
    assert cv2.imwrite(str(scan), np.full((400, 600), 200, dtype=np.uint8))
    panels = StagePanels(FilterStage.FIX_ORIENTATION, StageSequence())
    panels.set_page(PageId(image_id=ImageId(file_path=scan)))
    panels.options.deleteLater()
    panels.view.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    panels._source_pool.waitForDone()

    QtWidgets.QApplication.processEvents()

    assert panels._source is None
