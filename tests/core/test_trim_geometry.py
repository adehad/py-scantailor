"""The manual trim of Fix Orientation shapes the geometry of every later stage.

Upstream applies the trim as the pre-crop area of the transformation built by
Fix Orientation (``fix_orientation/Task.cpp``); Page Split then narrows it to
the intersection with the page outline (``page_split/Task.cpp``,
``intersectPreCropWithPageOutline``). The expected values below follow from
that: crop to the trimmed rectangle, then to the page, with the later stages
working in the cropped frame.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from scantailor.core.models import ImageId, Margins, OrthogonalRotation, PageId, SubPage
from scantailor.core.page_geometry import build_transformation
from scantailor.core.pipeline import PipelineOptions, process_page
from scantailor.core.stage_sequence import FilterStage, StageSequence
from scantailor.filters.deskew.dependencies import Dependencies as DeskewDependencies
from scantailor.filters.fix_orientation.image_trim import ImageTrim
from scantailor.filters.output.settings import Settings as OutputSettings
from scantailor.filters.select_content.content_box import ContentBox
from scantailor.imageproc.geometry import rotate_orthogonal

# 254 dpi makes one millimetre exactly ten pixels.
DPI = 254.0
PX_PER_MM = 10

WIDTH = 400
HEIGHT = 300
GUTTER_X = 200
LEFT_BACKGROUND = 200
RIGHT_BACKGROUND = 120
INK = 10
# (x, y, width, height) of the ink block on the right page, in source pixels.
INK_BLOCK = (GUTTER_X + 40, 60, 110, 140)

TRIM = ImageTrim(enabled=True, left=20, top=10, right=30, bottom=10)
# What the trim leaves of the source: x in [20, 370], y in [10, 290].
TRIMMED_X = (20, 370)
TRIMMED_Y = (10, 290)

IMAGE_ID = ImageId(file_path=Path("spread.tiff"))


@pytest.fixture(autouse=True)
def _output_at_source_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OutputSettings, "DEFAULT_DPI", int(DPI))


def _page(sub_page: SubPage) -> PageId:
    return PageId(image_id=IMAGE_ID, sub_page=sub_page)


def _spread() -> np.ndarray:
    image = np.full((HEIGHT, WIDTH), LEFT_BACKGROUND, dtype=np.uint8)
    image[:, GUTTER_X:] = RIGHT_BACKGROUND
    x, y, w, h = INK_BLOCK
    image[y : y + h, x : x + w] = INK
    return image


def _sequence(trim: ImageTrim, split_x: int = GUTTER_X) -> StageSequence:
    sequence = StageSequence()
    sequence.fix_orientation_filter.set_trim(IMAGE_ID, trim)
    for sub_page in (SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE):
        sequence.page_split_filter.set_manual_split(
            _page(sub_page), split_x, WIDTH, HEIGHT
        )
    return sequence


def _run(
    sequence: StageSequence,
    sub_page: SubPage,
    end: FilterStage,
    image: np.ndarray | None = None,
):
    options = PipelineOptions(end_stage=end, dpi=DPI)
    return process_page(
        _spread() if image is None else image, _page(sub_page), sequence, options
    )


def _bounds(area: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(area[:, 0].min()),
        float(area[:, 1].min()),
        float(area[:, 0].max()),
        float(area[:, 1].max()),
    )


class TestFixOrientationStage:
    def test_without_a_trim_the_whole_image_is_kept(self):
        result = _run(
            _sequence(ImageTrim()), SubPage.RIGHT_PAGE, FilterStage.FIX_ORIENTATION
        )

        np.testing.assert_array_equal(result.image, _spread())

    def test_the_trim_cuts_its_margins_from_every_edge(self):
        result = _run(_sequence(TRIM), SubPage.RIGHT_PAGE, FilterStage.FIX_ORIENTATION)

        np.testing.assert_array_equal(result.image, _spread()[10:290, 20:370])
        assert result.metadata["FIX_ORIENTATION"]["trim_area"] is not None

    def test_a_disabled_trim_with_margins_cuts_nothing(self):
        trim = ImageTrim(enabled=False, left=50, top=50, right=50, bottom=50)

        result = _run(_sequence(trim), SubPage.RIGHT_PAGE, FilterStage.FIX_ORIENTATION)

        np.testing.assert_array_equal(result.image, _spread())

    def test_the_trim_is_cut_from_the_unrotated_image_then_turned_with_it(self):
        """Margins name edges of the stored image, so a quarter turn moves them."""
        source = np.arange(300 * 400, dtype=np.uint32).reshape(400, 300) % 251
        source = source.astype(np.uint8)
        trim = ImageTrim(enabled=True, left=10, top=20, right=30, bottom=40)
        sequence = StageSequence()
        sequence.fix_orientation_filter.set_rotation(
            IMAGE_ID, OrthogonalRotation(degrees=90)
        )
        sequence.fix_orientation_filter.set_trim(IMAGE_ID, trim)

        result = _run(
            sequence, SubPage.SINGLE_PAGE, FilterStage.FIX_ORIENTATION, source
        )

        # Cutting the stored image first and turning the remainder must agree.
        expected = rotate_orthogonal(np.ascontiguousarray(source[20:360, 10:270]), 90)
        np.testing.assert_array_equal(result.image, expected)
        assert _bounds(result.transformation.pre_crop_area) == (40, 10, 380, 270)


class TestPageSplitStage:
    def test_untrimmed_pages_are_cropped_to_their_side_of_the_split(self):
        right = _run(_sequence(ImageTrim()), SubPage.RIGHT_PAGE, FilterStage.PAGE_SPLIT)

        np.testing.assert_array_equal(right.image, _spread()[:, GUTTER_X:])

    def test_the_right_page_is_the_intersection_of_outline_and_trim(self):
        # Page outline x in [200, 400] and trim x in [20, 370] meet in [200, 370].
        right = _run(_sequence(TRIM), SubPage.RIGHT_PAGE, FilterStage.PAGE_SPLIT)

        assert _bounds(right.transformation.pre_crop_area) == (200, 10, 370, 290)
        np.testing.assert_array_equal(right.image, _spread()[10:290, 200:370])

    def test_the_left_page_is_the_intersection_of_outline_and_trim(self):
        # Page outline x in [0, 200] and trim x in [20, 370] meet in [20, 200].
        left = _run(_sequence(TRIM), SubPage.LEFT_PAGE, FilterStage.PAGE_SPLIT)

        assert _bounds(left.transformation.pre_crop_area) == (20, 10, 200, 290)
        np.testing.assert_array_equal(left.image, _spread()[10:290, 20:200])

    def test_a_page_wholly_inside_the_trim_is_not_narrowed_further(self):
        sequence = _sequence(ImageTrim(enabled=True, left=5, top=5, right=5, bottom=5))

        left = _run(sequence, SubPage.LEFT_PAGE, FilterStage.PAGE_SPLIT)

        assert _bounds(left.transformation.pre_crop_area) == (5, 5, 200, 295)

    def test_a_page_outside_the_trim_falls_back_to_the_page_outline(self):
        """Upstream keeps the page outline when it does not meet the trim."""
        trim = ImageTrim(enabled=True, left=250, top=0, right=0, bottom=0)
        sequence = _sequence(trim)

        left = _run(sequence, SubPage.LEFT_PAGE, FilterStage.PAGE_SPLIT)

        assert _bounds(left.transformation.pre_crop_area) == (0, 0, 200, 300)

    def test_the_trim_is_recorded_in_the_split_dependencies(self):
        """Detection runs on the trimmed region, so the trim keys the layout."""
        page = _page(SubPage.RIGHT_PAGE)
        recorded = []
        for trim in (ImageTrim(), TRIM):
            sequence = StageSequence()
            sequence.fix_orientation_filter.set_trim(IMAGE_ID, trim)
            _run(sequence, SubPage.RIGHT_PAGE, FilterStage.PAGE_SPLIT)
            recorded.append(sequence.page_split_filter.get_params(page).dependencies)

        assert recorded[0].trim is None
        assert recorded[1].trim == (20, 10, 350, 280)
        assert recorded[0].image_size == recorded[1].image_size == (WIDTH, HEIGHT)


class TestDeskewStage:
    ANGLE = 2.0

    @staticmethod
    def _rotated_extent(width: float, height: float, degrees: float):
        radians = math.radians(abs(degrees))
        return (
            width * math.cos(radians) + height * math.sin(radians),
            width * math.sin(radians) + height * math.cos(radians),
        )

    @pytest.mark.parametrize(
        ("trim", "crop_size"),
        [(ImageTrim(), (200, 300)), (TRIM, (170, 280))],
        ids=["untrimmed", "trimmed"],
    )
    def test_the_page_is_rotated_about_the_centre_of_the_cropped_region(
        self, trim: ImageTrim, crop_size: tuple[int, int]
    ):
        sequence = _sequence(trim)
        page = _page(SubPage.RIGHT_PAGE)
        sequence.deskew_filter.set_manual_angle(page, self.ANGLE)

        result = _run(sequence, SubPage.RIGHT_PAGE, FilterStage.DESKEW)

        expected = self._rotated_extent(*crop_size, self.ANGLE)
        rect = result.transformation.resulting_rect
        assert (rect.width, rect.height) == pytest.approx(expected, abs=0.01)

    def test_detected_skew_records_the_trimmed_page_outline_as_its_dependency(self):
        page = _page(SubPage.RIGHT_PAGE)
        dependencies = {}
        for name, trim in (("plain", ImageTrim()), ("trimmed", TRIM)):
            sequence = _sequence(trim)
            _run(sequence, SubPage.RIGHT_PAGE, FilterStage.DESKEW)
            dependencies[name] = sequence.deskew_filter.get_params(page).dependencies

        assert _bounds(np.array(dependencies["plain"].page_outline)) == (
            200,
            0,
            400,
            300,
        )
        assert _bounds(np.array(dependencies["trimmed"].page_outline)) == (
            200,
            10,
            370,
            290,
        )
        assert not dependencies["plain"].matches(dependencies["trimmed"])

    def test_a_skew_detected_before_the_trim_is_detected_again_after_it(self):
        page = _page(SubPage.RIGHT_PAGE)
        sequence = _sequence(ImageTrim())
        _run(sequence, SubPage.RIGHT_PAGE, FilterStage.DESKEW)
        stored = sequence.deskew_filter.get_params(page)
        sequence.fix_orientation_filter.set_trim(IMAGE_ID, TRIM)
        trimmed = _run(sequence, SubPage.RIGHT_PAGE, FilterStage.FIX_ORIENTATION)
        trimmed_outline = DeskewDependencies.of(
            _run(
                sequence, SubPage.RIGHT_PAGE, FilterStage.PAGE_SPLIT
            ).transformation.pre_crop_area,
            0,
        )

        assert trimmed.image.shape == (280, 350)
        assert sequence.deskew_filter.needs_detection(page, trimmed_outline)
        assert not stored.dependencies.matches(trimmed_outline)


class TestSelectContentStage:
    @staticmethod
    def _content_centre(result) -> tuple[float, float]:
        box = result.metadata["SELECT_CONTENT"]["content_box"]
        return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    def test_the_content_box_is_measured_in_the_untrimmed_frame_without_a_trim(self):
        result = _run(
            _sequence(ImageTrim()), SubPage.RIGHT_PAGE, FilterStage.SELECT_CONTENT
        )

        # The block starts 40 px into the page and 60 px down.
        assert self._content_centre(result) == pytest.approx((95, 130), abs=2)

    def test_the_content_box_is_measured_in_the_frame_of_the_trimmed_page(self):
        result = _run(_sequence(TRIM), SubPage.RIGHT_PAGE, FilterStage.SELECT_CONTENT)

        # The page now starts at (200, 10), so the same block sits 10 px higher.
        assert self._content_centre(result) == pytest.approx((95, 120), abs=2)
        page_box = result.metadata["SELECT_CONTENT"]["page_box"]
        assert (page_box["width"], page_box["height"]) == (170, 280)

    def test_a_stored_content_box_is_used_in_the_trimmed_frame_as_it_is(self):
        sequence = _sequence(TRIM)
        page = _page(SubPage.RIGHT_PAGE)
        sequence.select_content_filter.set_manual_content(
            page, ContentBox(x=40, y=50, width=110, height=140)
        )

        result = _run(sequence, SubPage.RIGHT_PAGE, FilterStage.SELECT_CONTENT)

        assert result.metadata["SELECT_CONTENT"]["content_box"] == {
            "x": 40,
            "y": 50,
            "width": 110,
            "height": 140,
        }


class TestPageLayoutStage:
    @staticmethod
    def _configured(trim: ImageTrim, content: ContentBox) -> StageSequence:
        sequence = _sequence(trim)
        page = _page(SubPage.RIGHT_PAGE)
        sequence.deskew_filter.set_manual_angle(page, 0.0)
        sequence.select_content_filter.set_manual_content(page, content)
        return sequence

    def test_auto_margins_make_the_page_the_size_of_the_trimmed_crop(self):
        sequence = self._configured(TRIM, ContentBox(x=40, y=50, width=110, height=140))
        page = _page(SubPage.RIGHT_PAGE)
        layout = sequence.page_layout_filter
        layout.set_params(page, layout.get_params(page).with_auto_margins())

        result = _run(sequence, SubPage.RIGHT_PAGE, FilterStage.PAGE_LAYOUT)

        np.testing.assert_array_equal(result.image, _spread()[10:290, 200:370])

    def test_hard_margins_surround_the_content_found_in_the_trimmed_frame(self):
        content = ContentBox(x=40, y=50, width=110, height=140)
        sequence = self._configured(TRIM, content)
        page = _page(SubPage.RIGHT_PAGE)
        sequence.page_layout_filter.set_margins(
            page, Margins(left=2, right=3, top=1, bottom=4)
        )

        result = _run(sequence, SubPage.RIGHT_PAGE, FilterStage.PAGE_LAYOUT)

        image = result.image
        left, top = 2 * PX_PER_MM, 1 * PX_PER_MM
        assert image.shape == (140 + 50, 110 + 50)
        assert np.all(image[top : top + 140, left : left + 110] == INK)
        assert np.count_nonzero(image == INK) == 110 * 140

    def test_margins_reaching_past_the_trim_edge_are_blank_not_the_trimmed_pixels(self):
        """The same margins give the same page size; the cut-off strip is white."""
        plain = self._configured(
            ImageTrim(), ContentBox(x=40, y=60, width=110, height=140)
        )
        trimmed = self._configured(TRIM, ContentBox(x=40, y=50, width=110, height=140))
        page = _page(SubPage.RIGHT_PAGE)
        margins = Margins(left=2, right=3, top=1, bottom=4)
        plain.page_layout_filter.set_margins(page, margins)
        trimmed.page_layout_filter.set_margins(page, margins)

        plain_image = _run(plain, SubPage.RIGHT_PAGE, FilterStage.PAGE_LAYOUT).image
        trimmed_image = _run(trimmed, SubPage.RIGHT_PAGE, FilterStage.PAGE_LAYOUT).image

        assert trimmed_image.shape == plain_image.shape
        # The trimmed page ends at x = 170 of the page frame, which is column
        # 170 - 40 + 20 = 150 of the output; the 3 mm margin would reach 180.
        np.testing.assert_array_equal(trimmed_image[:, :150], plain_image[:, :150])
        assert np.all(trimmed_image[:, 150:] == 255)
        assert np.all(plain_image[:, 150:] == RIGHT_BACKGROUND)


class TestOutputStage:
    def test_the_output_has_the_size_of_the_trimmed_page(self):
        sequence = _sequence(TRIM)
        page = _page(SubPage.RIGHT_PAGE)
        sequence.deskew_filter.set_manual_angle(page, 0.0)
        sequence.select_content_filter.set_manual_content(
            page, ContentBox(x=40, y=50, width=110, height=140)
        )
        layout = sequence.page_layout_filter
        layout.set_params(page, layout.get_params(page).with_auto_margins())

        result = _run(sequence, SubPage.RIGHT_PAGE, FilterStage.OUTPUT)

        assert result.image.shape[:2] == (280, 170)


class TestRebuiltTransformations:
    """``build_transformation`` must reproduce what each pipeline stage ends with."""

    @pytest.mark.parametrize("sub_page", [SubPage.LEFT_PAGE, SubPage.RIGHT_PAGE])
    @pytest.mark.parametrize(
        "through",
        [
            FilterStage.FIX_ORIENTATION,
            FilterStage.PAGE_SPLIT,
            FilterStage.DESKEW,
            FilterStage.PAGE_LAYOUT,
        ],
    )
    def test_the_rebuilt_transformation_matches_the_pipeline(
        self, sub_page: SubPage, through: FilterStage
    ):
        sequence = _sequence(TRIM)
        page = _page(sub_page)
        sequence.deskew_filter.set_manual_angle(page, 1.5)
        sequence.select_content_filter.set_manual_content(
            page, ContentBox(x=30, y=40, width=100, height=150)
        )
        end = FilterStage.PAGE_LAYOUT if through == FilterStage.PAGE_LAYOUT else through

        ran = _run(sequence, sub_page, end).transformation
        rebuilt = build_transformation(sequence, page, (WIDTH, HEIGHT), DPI, through)

        np.testing.assert_allclose(rebuilt.transform, ran.transform, atol=1e-9)
        assert rebuilt.resulting_rect.width == pytest.approx(ran.resulting_rect.width)
        assert rebuilt.resulting_rect.height == pytest.approx(ran.resulting_rect.height)

    def test_the_first_stage_is_cropped_to_the_trim(self):
        xform = build_transformation(
            _sequence(TRIM),
            _page(SubPage.RIGHT_PAGE),
            (WIDTH, HEIGHT),
            DPI,
            FilterStage.FIX_ORIENTATION,
        )

        assert _bounds(xform.pre_crop_area) == (20, 10, 370, 290)
        assert xform.transform_point(20, 10) == pytest.approx((0, 0))
        assert (xform.resulting_rect.width, xform.resulting_rect.height) == (350, 280)

    def test_a_quarter_turn_moves_the_trim_with_the_pixels(self):
        sequence = StageSequence()
        sequence.fix_orientation_filter.set_rotation(
            IMAGE_ID, OrthogonalRotation(degrees=90)
        )
        sequence.fix_orientation_filter.set_trim(
            IMAGE_ID, ImageTrim(enabled=True, left=10, top=20, right=30, bottom=40)
        )

        xform = build_transformation(
            sequence,
            _page(SubPage.SINGLE_PAGE),
            (400, 300),
            DPI,
            FilterStage.FIX_ORIENTATION,
        )

        assert _bounds(xform.pre_crop_area) == (40, 10, 380, 270)

    def test_the_trim_does_not_move_the_rebuilt_page_outside_its_split(self):
        xform = build_transformation(
            _sequence(TRIM),
            _page(SubPage.LEFT_PAGE),
            (WIDTH, HEIGHT),
            DPI,
            FilterStage.PAGE_SPLIT,
        )

        assert _bounds(xform.pre_crop_area) == (20, 10, 200, 290)
