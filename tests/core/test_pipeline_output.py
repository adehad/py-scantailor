"""Output stage in the pipeline: output DPI, zones from Params and dewarping."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.image_io import load_image, write_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.core.zones import PictureLayer, PictureZoneProperties
from scantailor.filters.output import generator
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.dewarping_options import DewarpingMode, DewarpingOptions
from scantailor.filters.output.params import (
    DistortionModelSpec,
    FillZoneSpec,
    PictureZoneSpec,
)
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox
from tests.dewarping.synthetic import (
    PAGE_DPI,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    curved_text_page,
    row_profile_variance,
    straight_text_page,
)

SOURCE_DPI = 300.0
_BLOCK = (60, 80, 120, 100)


def _page_id(name: str = "page") -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path(f"{name}.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


def _sequence_for(page_id: PageId, content: tuple[int, int, int, int]) -> StageSequence:
    """Stages with a fixed uncut page, no skew and the given content box."""
    sequence = StageSequence()
    x, y, w, h = content
    sequence.page_split_filter.set_layout_type(page_id, LayoutType.SINGLE_PAGE_UNCUT)
    sequence.deskew_filter.set_manual_angle(page_id, 0.0)
    sequence.select_content_filter.set_manual_content(
        page_id, ContentBox(x=x, y=y, width=w, height=h)
    )
    return sequence


def _set_output(sequence: StageSequence, page_id: PageId, **changes) -> None:
    params = sequence.output_filter.get_params(page_id).model_copy(update=changes)
    sequence.output_filter.set_params(page_id, params)


def _block_page() -> np.ndarray:
    page = np.full((300, 400), 220, dtype=np.uint8)
    x, y, w, h = _BLOCK
    page[y : y + h, x : x + w] = 30
    return page


class TestOutputDpi:
    """The output is resampled once to the output DPI."""

    def test_600_dpi_from_a_300_dpi_source_doubles_the_size(self) -> None:
        """The page layout size in millimetres is kept, so pixels double."""
        page_id = _page_id()
        sequence = _sequence_for(page_id, _BLOCK)
        _set_output(
            sequence,
            page_id,
            color_mode=ColorMode.COLOR_GRAYSCALE,
            output_dpi=Dpi.uniform(300),
        )
        at_source = process_page(
            _block_page(), page_id, sequence, PipelineOptions(dpi=SOURCE_DPI)
        )
        _set_output(sequence, page_id, output_dpi=Dpi.uniform(600))

        doubled = process_page(
            _block_page(), page_id, sequence, PipelineOptions(dpi=SOURCE_DPI)
        )

        assert doubled.image.shape[0] == pytest.approx(
            2 * at_source.image.shape[0], abs=2
        )
        assert doubled.image.shape[1] == pytest.approx(
            2 * at_source.image.shape[1], abs=2
        )

    def test_result_reports_the_output_dpi_and_it_is_written_to_the_file(
        self, tmp_path: Path
    ) -> None:
        """A 300 dpi source written after output at 600 dpi reads back as 600 dpi."""
        page_id = _page_id()
        sequence = _sequence_for(page_id, _BLOCK)
        _set_output(
            sequence,
            page_id,
            color_mode=ColorMode.COLOR_GRAYSCALE,
            output_dpi=Dpi.uniform(600),
        )

        result = process_page(
            _block_page(), page_id, sequence, PipelineOptions(dpi=SOURCE_DPI)
        )
        assert result.dpi == Dpi.uniform(600)
        path = tmp_path / "out.tif"
        assert result.dpi is not None
        write_image(path, result.image, result.dpi)

        _loaded, metadata = load_image(ImageId(file_path=path))
        assert metadata.dpi == Dpi.uniform(600)

    def test_result_has_no_dpi_before_the_output_stage(self) -> None:
        """Stopping early leaves the resolution unset."""
        page_id = _page_id()
        sequence = _sequence_for(page_id, _BLOCK)

        result = process_page(
            _block_page(),
            page_id,
            sequence,
            PipelineOptions(dpi=SOURCE_DPI, end_stage=FilterStage.PAGE_LAYOUT),
        )

        assert result.dpi is None


class TestZonesFromParams:
    """Zones in the output Params reach generate_output in the output's pixels."""

    def test_picture_and_fill_zones_and_page_outline_are_passed_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source-coordinate zones arrive moved by the page margins."""
        calls: list[dict] = []
        real = generator.generate_output

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(generator, "generate_output", spy)
        page_id = _page_id()
        sequence = _sequence_for(page_id, _BLOCK)
        square = ((100.0, 100.0), (140.0, 100.0), (140.0, 140.0), (100.0, 140.0))
        _set_output(
            sequence,
            page_id,
            color_mode=ColorMode.MIXED,
            output_dpi=Dpi.uniform(SOURCE_DPI),
            picture_zones=(
                PictureZoneSpec(
                    points=square,
                    properties=PictureZoneProperties(layer=PictureLayer.PAINTER2),
                ),
            ),
            fill_zones=(FillZoneSpec(points=square),),
        )

        process_page(_block_page(), page_id, sequence, PipelineOptions(dpi=SOURCE_DPI))

        (call,) = calls
        (picture_zone,) = call["picture_zones"]
        (fill_zone,) = call["fill_zones"]
        assert picture_zone.properties.layer == PictureLayer.PAINTER2
        offsets = picture_zone.spline.points - np.array(square)
        assert np.ptp(offsets, axis=0) == pytest.approx((0.0, 0.0), abs=1e-6)
        np.testing.assert_allclose(fill_zone.spline.points, picture_zone.spline.points)
        outline = np.array(call["page_outline"])
        assert outline.shape == (4, 2)
        assert np.ptp(outline, axis=0) == pytest.approx((400.0, 300.0), abs=1.0)

    def test_without_zones_nothing_is_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pages with no zones hand over empty lists."""
        calls: list[dict] = []
        real = generator.generate_output

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(generator, "generate_output", spy)
        page_id = _page_id()
        sequence = _sequence_for(page_id, _BLOCK)

        process_page(_block_page(), page_id, sequence, PipelineOptions(dpi=SOURCE_DPI))

        assert calls[0]["picture_zones"] == []
        assert calls[0]["fill_zones"] == []


def _page_sequence_for_text(
    mode: DewarpingMode, **changes
) -> tuple[StageSequence, PageId]:
    page_id = _page_id("text")
    sequence = _sequence_for(
        page_id, (TEXT_LEFT_PX, 100, PAGE_WIDTH - 2 * TEXT_LEFT_PX, PAGE_HEIGHT - 200)
    )
    _set_output(
        sequence,
        page_id,
        color_mode=ColorMode.COLOR_GRAYSCALE,
        output_dpi=Dpi.uniform(PAGE_DPI),
        dewarping=DewarpingOptions(mode=mode, post_deskew=False),
        **changes,
    )
    return sequence, page_id


TEXT_LEFT_PX = 100


def _run_text(image: np.ndarray, sequence: StageSequence, page_id: PageId):
    return process_page(image, page_id, sequence, PipelineOptions(dpi=float(PAGE_DPI)))


class TestDewarpingInThePipeline:
    """Dewarping runs on the output view, in the transformed coordinates."""

    def test_auto_mode_straightens_a_curved_page(self) -> None:
        """Rows hold the ink of whole lines afterwards, so the row profile sharpens."""
        warped = curved_text_page()
        sequence, page_id = _page_sequence_for_text(DewarpingMode.OFF)
        flat_off = _run_text(warped, sequence, page_id)
        sequence, page_id = _page_sequence_for_text(DewarpingMode.AUTO)

        dewarped = _run_text(warped, sequence, page_id)

        assert dewarped.metadata["OUTPUT"]["was_dewarped"] is True
        columns = (150, 1050)
        assert row_profile_variance(
            dewarped.image, columns
        ) > 2.0 * row_profile_variance(flat_off.image, columns)

    def test_dewarping_off_leaves_geometry_alone(self) -> None:
        """With dewarping off the output is the page view, pixel for pixel."""
        page = straight_text_page()
        sequence, page_id = _page_sequence_for_text(DewarpingMode.OFF)
        off = _run_text(page, sequence, page_id)
        geometry_only = process_page(
            page,
            page_id,
            sequence,
            PipelineOptions(dpi=float(PAGE_DPI), end_stage=FilterStage.PAGE_LAYOUT),
        )

        assert off.metadata["OUTPUT"]["was_dewarped"] is False
        assert off.image.shape == geometry_only.image.shape
        inner = (slice(300, 1000), slice(300, 800))
        np.testing.assert_array_equal(off.image[inner], geometry_only.image[inner])

    def test_manual_model_in_source_coordinates_straightens_the_page(self) -> None:
        """A model given as the page's top and bottom lines dewarps like auto mode."""
        warped = curved_text_page()
        xs = np.linspace(TEXT_LEFT_PX, PAGE_WIDTH - TEXT_LEFT_PX, 9)
        sag = 60.0 * np.sin(np.pi * xs / PAGE_WIDTH)
        model = DistortionModelSpec(
            top=tuple((float(x), 200.0 + float(s)) for x, s in zip(xs, sag)),
            bottom=tuple((float(x), 1400.0 + float(s)) for x, s in zip(xs, sag)),
        )
        sequence, page_id = _page_sequence_for_text(
            DewarpingMode.MANUAL, distortion_model=model
        )
        manual = _run_text(warped, sequence, page_id)
        sequence, page_id = _page_sequence_for_text(DewarpingMode.OFF)
        flat_off = _run_text(warped, sequence, page_id)

        assert manual.metadata["OUTPUT"]["was_dewarped"] is True
        columns = (150, 1050)
        assert row_profile_variance(manual.image, columns) > 1.5 * row_profile_variance(
            flat_off.image, columns
        )

    def test_manual_mode_without_a_model_does_nothing(self) -> None:
        """No model means no dewarping."""
        sequence, page_id = _page_sequence_for_text(DewarpingMode.MANUAL)

        result = _run_text(straight_text_page(), sequence, page_id)

        assert result.metadata["OUTPUT"]["was_dewarped"] is False

    def test_auto_mode_on_a_page_without_text_does_nothing(self) -> None:
        """A blank page yields no model, so the page is left as it is."""
        blank = np.full((PAGE_HEIGHT, PAGE_WIDTH), 235, dtype=np.uint8)
        sequence, page_id = _page_sequence_for_text(DewarpingMode.AUTO)

        result = _run_text(blank, sequence, page_id)

        assert result.metadata["OUTPUT"]["was_dewarped"] is False

    def test_marginal_mode_runs_and_keeps_the_page_size(self) -> None:
        """Marginal dewarping hands generate_output a dewarper for the spine strip."""
        sequence, page_id = _page_sequence_for_text(DewarpingMode.OFF)
        undewarped = _run_text(curved_text_page(), sequence, page_id)
        sequence, page_id = _page_sequence_for_text(DewarpingMode.MARGINAL)

        result = _run_text(curved_text_page(), sequence, page_id)

        assert result.metadata["OUTPUT"]["was_dewarped"] is True
        assert result.image.shape == undewarped.image.shape
