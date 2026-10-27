"""Tests for batch geometry, stored-parameter replay and the output hand-off."""

from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, Margins, PageId, SubPage
from scantailor.core.pipeline import PipelineOptions, process_batch, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.output import generator
from scantailor.filters.output.color_options import ColorCommonOptions
from scantailor.filters.output.settings import Settings as OutputSettings
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

# 254 dpi makes one millimetre exactly ten pixels.
DPI = 254.0
PX_PER_MM = 10
BACKGROUND = 200
INK = 40
RGB_INK = (10, 120, 220)

# Default page layout margins: 10 mm left/right, 5 mm top/bottom.
DEFAULT_SIDE_PX = 10 * PX_PER_MM
DEFAULT_TOP_PX = 5 * PX_PER_MM


@pytest.fixture(autouse=True)
def _output_at_source_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep output pixels one to one with the source so sizes stay in millimetres."""
    monkeypatch.setattr(OutputSettings, "DEFAULT_DPI", int(DPI))


def _page_id(name: str) -> PageId:
    return PageId(
        image_id=ImageId(file_path=Path(f"{name}.tiff")),
        sub_page=SubPage.SINGLE_PAGE,
    )


def _configure(
    sequence: StageSequence,
    page_id: PageId,
    content: tuple[int, int, int, int],
    angle: float = 0.0,
) -> None:
    """Set every detecting stage to a manual value."""
    x, y, w, h = content
    sequence.page_split_filter.set_layout_type(page_id, LayoutType.SINGLE_PAGE_UNCUT)
    sequence.deskew_filter.set_manual_angle(page_id, angle)
    sequence.select_content_filter.set_manual_content(
        page_id, ContentBox(x=x, y=y, width=w, height=h)
    )


def _page(
    sequence: StageSequence,
    name: str,
    content: tuple[int, int, int, int],
    angle: float = 0.0,
) -> tuple[np.ndarray, PageId]:
    """A 500 x 500 page with an ink block at content (x, y, w, h), all stages manual."""
    page_id = _page_id(name)
    x, y, w, h = content
    image = np.full((500, 500), BACKGROUND, dtype=np.uint8)
    image[y : y + h, x : x + w] = INK
    _configure(sequence, page_id, content, angle)
    return image, page_id


def _rgb_page(sequence: StageSequence) -> tuple[np.ndarray, PageId]:
    page_id = _page_id("rgb")
    image = np.full((500, 500, 3), 255, dtype=np.uint8)
    image[100:180, 150:270] = RGB_INK
    _configure(sequence, page_id, (150, 100, 120, 80))
    return image, page_id


class TestBatchAggregateSize:
    """Every page of a batch is sized against the same aggregate."""

    def test_pages_with_different_content_get_identical_sizes_from_the_first(
        self,
    ) -> None:
        """The first page already has the soft margins the last page implies."""
        sequence = StageSequence()
        pages = [
            _page(sequence, "small", (50, 50, 100, 80)),
            _page(sequence, "wide", (50, 50, 150, 120)),
            _page(sequence, "tall", (50, 50, 60, 200)),
        ]

        results = process_batch(pages, sequence, PipelineOptions(dpi=DPI))

        # Widest content 150 px + 2 * 100 px, tallest 200 px + 2 * 50 px.
        for result in results:
            assert result.image.shape[0] == pytest.approx(300, abs=1)
            assert result.image.shape[1] == pytest.approx(350, abs=1)

    def test_a_page_without_content_comes_out_blank_at_the_batch_size(self) -> None:
        """No content leaves the aggregate alone and the page white, as upstream."""
        sequence = StageSequence()
        text = _page(sequence, "text", (150, 100, 200, 300))
        empty_id = _page_id("empty")
        sequence.page_split_filter.set_layout_type(
            empty_id, LayoutType.SINGLE_PAGE_UNCUT
        )
        sequence.deskew_filter.set_manual_angle(empty_id, 0.0)
        # Only the edge of the book, which Select Content takes for shadow.
        empty = np.full((500, 500), BACKGROUND, dtype=np.uint8)
        empty[:, :12] = INK

        results = process_batch(
            [text, (empty, empty_id)], sequence, PipelineOptions(dpi=DPI)
        )

        # Content 200 x 300 px + 2 * 100 px wide, + 2 * 50 px high.
        assert [result.image.shape for result in results] == [(400, 400)] * 2
        assert (results[1].image == 255).all()

    def test_first_page_of_a_rerun_matches_the_batch(self) -> None:
        """Processing a page alone afterwards reproduces its batch output."""
        sequence = StageSequence()
        pages = [
            _page(sequence, "small", (50, 50, 100, 80)),
            _page(sequence, "tall", (50, 50, 60, 200)),
        ]
        options = PipelineOptions(dpi=DPI)

        batch = process_batch(pages, sequence, options)
        again = process_page(*pages[0], sequence, options)

        np.testing.assert_array_equal(again.image, batch[0].image)

    def test_batch_ending_before_page_layout_leaves_layout_settings_alone(
        self,
    ) -> None:
        """No pre-pass runs when the layout stage is out of range."""
        sequence = StageSequence()
        pages = [_page(sequence, "a", (50, 50, 100, 80))]

        process_batch(
            pages, sequence, PipelineOptions(end_stage=FilterStage.DESKEW, dpi=DPI)
        )

        assert not sequence.page_layout_filter.settings.is_params_set(pages[0][1])


class TestStartStageReplaysStoredGeometry:
    """Stages before start_stage still shape the image from stored parameters."""

    def test_output_only_run_crops_and_deskews_a_skewed_page(self) -> None:
        """Starting at OUTPUT gives the same pixels as running every stage."""
        sequence = StageSequence()
        image, page_id = _page(sequence, "skewed", (150, 150, 120, 100), angle=3.0)
        sequence.page_layout_filter.set_margins(page_id, Margins.uniform(2.0))

        full = process_page(image, page_id, sequence, PipelineOptions(dpi=DPI))
        replay = process_page(
            image,
            page_id,
            sequence,
            PipelineOptions(start_stage=FilterStage.OUTPUT, dpi=DPI),
        )

        assert replay.image.shape == (100 + 40, 120 + 40)
        np.testing.assert_array_equal(replay.image, full.image)
        assert replay.stages_completed == [FilterStage.OUTPUT]

    def test_replay_does_not_detect_missing_parameters(self) -> None:
        """Without stored deskew parameters the replayed page is not rotated."""
        sequence = StageSequence()
        page_id = _page_id("unset")
        image = np.full((200, 300), BACKGROUND, dtype=np.uint8)
        sequence.page_layout_filter.set_margins(page_id, Margins())
        sequence.select_content_filter.set_manual_content(
            page_id, ContentBox(x=0, y=0, width=300, height=200)
        )

        result = process_page(
            image,
            page_id,
            sequence,
            PipelineOptions(
                start_stage=FilterStage.PAGE_LAYOUT,
                end_stage=FilterStage.PAGE_LAYOUT,
                dpi=DPI,
            ),
        )

        assert not sequence.deskew_filter.settings.is_params_set(page_id)
        assert result.image.shape[:2] == image.shape


class TestDefaultMargins:
    """Default margins are the C++ ones: 10 mm left/right, 5 mm top/bottom."""

    def test_borders_around_the_content_at_a_known_dpi(self) -> None:
        """At 254 dpi the content sits 100 px from the sides and 50 px from the top."""
        sequence = StageSequence()
        image, page_id = _page(sequence, "page", (150, 100, 120, 80))

        result = process_page(
            image,
            page_id,
            sequence,
            PipelineOptions(end_stage=FilterStage.PAGE_LAYOUT, dpi=DPI),
        )

        out = result.image
        assert out.shape == (80 + 2 * DEFAULT_TOP_PX, 120 + 2 * DEFAULT_SIDE_PX)
        ink = np.argwhere(out == INK)
        assert tuple(ink.min(axis=0)) == (DEFAULT_TOP_PX, DEFAULT_SIDE_PX)
        assert tuple(ink.max(axis=0) + 1) == (
            DEFAULT_TOP_PX + 80,
            DEFAULT_SIDE_PX + 120,
        )


class TestRgbInput:
    """Colour pages keep their channels through the geometry stages."""

    def test_rgb_page_keeps_three_channels_and_colours(self) -> None:
        """The geometry stages resample all channels with one transformation."""
        sequence = StageSequence()
        image, page_id = _rgb_page(sequence)

        result = process_page(
            image,
            page_id,
            sequence,
            PipelineOptions(end_stage=FilterStage.PAGE_LAYOUT, dpi=DPI),
        )

        assert result.image.shape == (80 + 100, 120 + 200, 3)
        patch = result.image[DEFAULT_TOP_PX : DEFAULT_TOP_PX + 80]
        patch = patch[:, DEFAULT_SIDE_PX : DEFAULT_SIDE_PX + 120]
        assert np.all(patch == RGB_INK)

    def test_rgb_page_runs_through_output(self) -> None:
        """The full pipeline accepts a colour input."""
        sequence = StageSequence()
        image, page_id = _rgb_page(sequence)

        result = process_page(image, page_id, sequence, PipelineOptions(dpi=DPI))

        assert result.image.shape[:2] == (80 + 100, 120 + 200)


class TestOutputHandOff:
    """The output stage receives the colour options and the content polygon."""

    def test_content_polygon_and_colour_options_reach_generate_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The polygon is the content box in output-image coordinates."""
        calls: list[dict] = []
        real = generator.generate_output

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(generator, "generate_output", spy)
        sequence = StageSequence()
        image, page_id = _page(sequence, "page", (150, 100, 120, 80))
        custom = ColorCommonOptions(fill_margins=False, normalize_illumination=True)
        stored = sequence.output_filter.get_params(page_id)
        sequence.output_filter.set_params(page_id, stored.with_color_common(custom))

        process_page(image, page_id, sequence, PipelineOptions(dpi=DPI))

        assert len(calls) == 1
        polygon = np.array(calls[0]["content_polygon"])
        left, top = DEFAULT_SIDE_PX, DEFAULT_TOP_PX
        expected = [
            (left, top),
            (left + 120, top),
            (left + 120, top + 80),
            (left, top + 80),
        ]
        np.testing.assert_allclose(polygon, expected, atol=1.0)
        assert calls[0]["color_options"] == custom

    def test_content_polygon_follows_deskew(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rotated page still maps the content box into the output frame."""
        recorded: list[list[tuple[float, float]]] = []
        real = generator.generate_output

        def spy(*args, **kwargs):
            recorded.append(kwargs["content_polygon"])
            return real(*args, **kwargs)

        monkeypatch.setattr(generator, "generate_output", spy)
        sequence = StageSequence()
        image, page_id = _page(sequence, "skew", (150, 100, 120, 80), angle=2.0)

        result = process_page(image, page_id, sequence, PipelineOptions(dpi=DPI))

        polygon = np.array(recorded[0])
        # The box is given in the deskewed view, so it stays an upright
        # 120 x 80 rectangle sitting on the default margins.
        np.testing.assert_allclose(
            polygon.max(axis=0) - polygon.min(axis=0), (120, 80), atol=1.0
        )
        np.testing.assert_allclose(
            polygon.min(axis=0), (DEFAULT_SIDE_PX, DEFAULT_TOP_PX), atol=1.0
        )
        height, width = result.image.shape[:2]
        assert polygon[:, 0].min() >= 0
        assert polygon[:, 1].min() >= 0
        assert polygon[:, 0].max() <= width
        assert polygon[:, 1].max() <= height


class TestAutoMarginsWithEmptyContent:
    """Auto margins are measured against the content box as detected."""

    def test_empty_content_box_margins_are_not_centred_on_the_page(
        self,
    ) -> None:
        """An empty box at the page origin gets all spare space after it."""
        from scantailor.filters.page_layout.params import Params

        sequence = StageSequence()
        image, page_id = _page(sequence, "empty", (100, 100, 0, 0))
        sequence.page_layout_filter.settings.set_params(
            page_id, Params(auto_margins=True)
        )

        result = process_page(
            image,
            page_id,
            sequence,
            PipelineOptions(end_stage=FilterStage.PAGE_LAYOUT, dpi=DPI),
        )

        margins = result.metadata["PAGE_LAYOUT"]["margins"]
        assert margins["left"] == pytest.approx(0.0, abs=0.05)
        assert margins["right"] == pytest.approx(50.0, abs=0.05)
