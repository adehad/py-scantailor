"""Automatic and marginal dewarping keep their model and freeze on a DPI change."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.batch import run_batch
from scantailor.core.image_io import write_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.output.params import ApplyGroup, DistortionModelSpec, Params
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox
from tests.dewarping.synthetic import (
    BACKGROUND,
    PAGE_DPI,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    curved_text_page,
    row_profile_variance,
)

_CONTENT = (100, 100, PAGE_WIDTH - 200, PAGE_HEIGHT - 200)
_EDGES = DistortionModelSpec(
    top=((100.0, 200.0), (600.0, 230.0), (1100.0, 200.0)),
    bottom=((100.0, 1400.0), (600.0, 1430.0), (1100.0, 1400.0)),
)


def _page_id(name: str = "text") -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path(f"{name}.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


def _sequence(page_id: PageId, mode: DewarpingMode, **changes) -> StageSequence:
    sequence = StageSequence()
    x, y, width, height = _CONTENT
    sequence.page_split_filter.set_layout_type(page_id, LayoutType.SINGLE_PAGE_UNCUT)
    sequence.deskew_filter.set_manual_angle(page_id, 0.0)
    sequence.select_content_filter.set_manual_content(
        page_id, ContentBox(x=x, y=y, width=width, height=height)
    )
    params = sequence.output_filter.get_params(page_id).model_copy(
        update={
            "color_mode": ColorMode.COLOR_GRAYSCALE,
            "output_dpi": Dpi.uniform(PAGE_DPI),
            "dewarping": DewarpingOptions(mode=mode, post_deskew=False),
            **changes,
        }
    )
    sequence.output_filter.set_params(page_id, params)
    return sequence


def _run(image: np.ndarray, page_id: PageId, sequence: StageSequence):
    return process_page(image, page_id, sequence, PipelineOptions(dpi=float(PAGE_DPI)))


class TestPipelineReportsTheModel:
    def test_auto_mode_reports_the_edges_it_found(self) -> None:
        page_id = _page_id()

        result = _run(
            curved_text_page(), page_id, _sequence(page_id, DewarpingMode.AUTO)
        )

        assert result.distortion_model is not None
        assert result.distortion_model.is_defined()

    def test_marginal_mode_reports_the_edges_it_found(self) -> None:
        page_id = _page_id()
        sequence = _sequence(page_id, DewarpingMode.MARGINAL)
        paper = np.full((PAGE_HEIGHT, PAGE_WIDTH), 220, dtype=np.uint8)

        result = _run(paper, page_id, sequence)

        assert result.metadata["OUTPUT"]["was_dewarped"] is True
        assert result.distortion_model is not None
        assert result.distortion_model.is_defined()

    @pytest.mark.parametrize("mode", [DewarpingMode.OFF, DewarpingMode.MANUAL])
    def test_modes_that_build_nothing_report_no_model(
        self, mode: DewarpingMode
    ) -> None:
        page_id = _page_id()
        sequence = _sequence(page_id, mode, distortion_model=_EDGES)

        result = _run(curved_text_page(), page_id, sequence)

        assert result.distortion_model is None

    def test_a_page_without_text_reports_no_model(self) -> None:
        page_id = _page_id()
        blank = np.full((PAGE_HEIGHT, PAGE_WIDTH), BACKGROUND, dtype=np.uint8)

        result = _run(blank, page_id, _sequence(page_id, DewarpingMode.AUTO))

        assert result.distortion_model is None

    def test_the_reported_model_as_manual_dewarps_like_auto(self) -> None:
        """Edges are in source coordinates, so they work again in manual mode."""
        page_id = _page_id()
        warped = curved_text_page()
        auto_sequence = _sequence(page_id, DewarpingMode.AUTO)
        auto = _run(warped, page_id, auto_sequence)
        assert auto.distortion_model is not None
        manual_sequence = _sequence(
            page_id, DewarpingMode.MANUAL, distortion_model=auto.distortion_model
        )

        manual = _run(warped, page_id, manual_sequence)

        assert manual.metadata["OUTPUT"]["was_dewarped"] is True
        assert manual.image.shape == auto.image.shape
        columns = (150, 1050)
        assert row_profile_variance(manual.image, columns) == pytest.approx(
            row_profile_variance(auto.image, columns), rel=0.1
        )


class TestStoringTheModel:
    @pytest.mark.parametrize("mode", [DewarpingMode.AUTO, DewarpingMode.MARGINAL])
    def test_generated_modes_keep_the_model_in_the_page_params(
        self, mode: DewarpingMode
    ) -> None:
        page_id = _page_id()
        sequence = _sequence(page_id, mode)

        changed = sequence.output_filter.store_distortion_model(page_id, _EDGES)

        assert changed
        assert sequence.output_filter.get_params(page_id).distortion_model == _EDGES

    @pytest.mark.parametrize("mode", [DewarpingMode.OFF, DewarpingMode.MANUAL])
    def test_a_page_no_longer_in_a_generated_mode_keeps_its_own_model(
        self, mode: DewarpingMode
    ) -> None:
        page_id = _page_id()
        mine = DistortionModelSpec(
            top=((0.0, 0.0), (1.0, 0.0)), bottom=((0.0, 9.0), (1.0, 9.0))
        )
        sequence = _sequence(page_id, mode, distortion_model=mine)

        changed = sequence.output_filter.store_distortion_model(page_id, _EDGES)

        assert not changed
        assert sequence.output_filter.get_params(page_id).distortion_model == mine

    def test_storing_the_same_model_again_changes_nothing(self) -> None:
        page_id = _page_id()
        sequence = _sequence(page_id, DewarpingMode.AUTO, distortion_model=_EDGES)

        assert not sequence.output_filter.store_distortion_model(page_id, _EDGES)


class TestOutputDpiFreezesAutoDewarping:
    @staticmethod
    def _params(mode: DewarpingMode, model: DistortionModelSpec) -> Params:
        return Params(
            output_dpi=Dpi.uniform(300),
            dewarping=DewarpingOptions(mode=mode),
            distortion_model=model,
        )

    def test_auto_with_a_stored_model_becomes_manual_and_keeps_the_model(self) -> None:
        params = self._params(DewarpingMode.AUTO, _EDGES)

        changed = params.with_output_dpi(Dpi.uniform(600))

        assert changed.output_dpi == Dpi.uniform(600)
        assert changed.dewarping.mode == DewarpingMode.MANUAL
        assert changed.distortion_model == _EDGES

    def test_auto_without_a_model_stays_auto(self) -> None:
        params = self._params(DewarpingMode.AUTO, DistortionModelSpec())

        assert params.with_output_dpi(Dpi.uniform(600)).dewarping.mode == (
            DewarpingMode.AUTO
        )

    def test_the_same_dpi_does_not_freeze(self) -> None:
        params = self._params(DewarpingMode.AUTO, _EDGES)

        assert params.with_output_dpi(Dpi.uniform(300)).dewarping.mode == (
            DewarpingMode.AUTO
        )

    @pytest.mark.parametrize(
        "mode", [DewarpingMode.OFF, DewarpingMode.MARGINAL, DewarpingMode.MANUAL]
    )
    def test_other_modes_are_not_touched(self, mode: DewarpingMode) -> None:
        params = self._params(mode, _EDGES)

        assert params.with_output_dpi(Dpi.uniform(600)).dewarping.mode == mode

    def test_applying_the_resolution_to_another_page_freezes_it_too(self) -> None:
        target = self._params(DewarpingMode.AUTO, _EDGES)
        source = Params(output_dpi=Dpi.uniform(600))

        merged = target.with_group_from(source, ApplyGroup.DPI)

        assert merged.output_dpi == Dpi.uniform(600)
        assert merged.dewarping.mode == DewarpingMode.MANUAL


@pytest.mark.slow
def test_batch_stores_the_model_of_an_auto_page(tmp_path: Path) -> None:
    source = tmp_path / "in" / "text.tif"
    source.parent.mkdir()
    write_image(source, curved_text_page(), Dpi.uniform(PAGE_DPI))
    page_id = PageId(image_id=ImageId(file_path=source), sub_page=SubPage.SINGLE_PAGE)
    sequence = _sequence(page_id, DewarpingMode.AUTO)

    result = run_batch(sequence, [page_id], tmp_path / "out", max_workers=1)

    assert page_id in result.written
    stored = sequence.output_filter.get_params(page_id)
    assert stored.dewarping.mode == DewarpingMode.AUTO
    assert stored.distortion_model.is_defined()
