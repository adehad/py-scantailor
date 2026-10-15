"""The automatic distortion model of a page, from its text lines and page edges."""

import cv2
import numpy as np
import pytest

from scantailor.dewarping.auto_model import (
    build_auto_distortion_model,
    straighten_outer_border,
    trivial_distortion_model,
)
from scantailor.dewarping.distortion_model import Curve, DistortionModel
from scantailor.dewarping.page_dewarper import dewarper_from_model
from tests.dewarping.book_pages import BentPage, bent_book_page

DPI = 200
CONTENT = (160.0, 200.0, 890.0, 1090.0)
DEPTH = 2.0


def _unevenness(page: BentPage, model: DistortionModel) -> np.ndarray:
    """Peak-to-peak height, in flat page pixels, of lines dewarped with the model."""
    dewarper = dewarper_from_model(model, DEPTH)
    assert dewarper is not None
    spread = []
    for baseline in page.baselines[::3]:
        xs = np.linspace(page.margin, page.flat_size[0] - page.margin, 25)
        flat = np.column_stack([xs, np.full_like(xs, baseline - 6.0)])
        dewarped = dewarper.map_points_to_dewarped_space(page.flat_to_warped(flat))
        spread.append(np.ptp(dewarped[:, 1]) * page.flat_size[1])
    return np.array(spread)


@pytest.fixture(scope="module")
def flat_page() -> BentPage:
    return bent_book_page(bow=0.0, squeeze=0.0)


class TestBentPages:
    @pytest.mark.parametrize(
        ("bow", "squeeze"), [(45.0, -25.0), (70.0, 10.0), (-40.0, -60.0)]
    )
    def test_the_model_straightens_the_text_lines_of_a_bent_page(
        self, bow: float, squeeze: float
    ) -> None:
        page = bent_book_page(bow, squeeze)

        model = build_auto_distortion_model(page.image, DPI, CONTENT)

        assert model is not None
        assert _unevenness(page, model).max() < 1.5

    def test_the_curves_of_the_model_follow_the_page_edges(self) -> None:
        page = bent_book_page(45.0, -25.0)

        model = build_auto_distortion_model(page.image, DPI, CONTENT)

        assert model is not None
        assert model.top_curve is not None and model.bottom_curve is not None
        top = model.top_curve.polyline
        # The edges and the outermost text lines are the model's candidates; the
        # top curve is the highest of them and has the page's bulge.
        assert top[:, 1].mean() < 400
        sag = top[len(top) // 2, 1] - 0.5 * (top[0, 1] + top[-1, 1])
        assert sag == pytest.approx(45.0, abs=12.0)

    def test_the_model_is_valid_and_runs_left_to_right(self) -> None:
        page = bent_book_page()

        model = build_auto_distortion_model(page.image, DPI, CONTENT)

        assert model is not None and model.is_valid()
        assert model.top_curve is not None
        assert model.top_curve.polyline[0, 0] < model.top_curve.polyline[-1, 0]

    def test_the_result_is_repeatable(self) -> None:
        page = bent_book_page()

        first = build_auto_distortion_model(page.image, DPI, CONTENT)
        second = build_auto_distortion_model(page.image, DPI, CONTENT)

        assert first is not None and second is not None
        assert first.top_curve is not None and second.top_curve is not None
        np.testing.assert_array_equal(
            first.top_curve.polyline, second.top_curve.polyline
        )


class TestFlatPages:
    """C++ never rejects a flat page by name; its curves just come out straight."""

    def test_the_model_of_a_flat_page_is_flat(self, flat_page: BentPage) -> None:
        model = build_auto_distortion_model(flat_page.image, DPI, CONTENT)

        assert model is not None
        assert model.top_curve is not None and model.bottom_curve is not None
        for curve in (model.top_curve, model.bottom_curve):
            polyline = curve.polyline
            assert np.ptp(polyline[:, 1]) < 1.0

    def test_dewarping_a_flat_page_with_its_model_moves_no_text_line(
        self, flat_page: BentPage
    ) -> None:
        model = build_auto_distortion_model(flat_page.image, DPI, CONTENT)

        assert model is not None
        assert _unevenness(flat_page, model).max() < 0.5

    def test_a_flat_page_is_not_sheared(self, flat_page: BentPage) -> None:
        model = build_auto_distortion_model(flat_page.image, DPI, CONTENT)

        assert model is not None
        assert model.top_curve is not None and model.bottom_curve is not None
        top = model.top_curve.polyline
        bottom = model.bottom_curve.polyline
        assert top[0, 1] == pytest.approx(top[-1, 1], abs=1.0)
        assert bottom[0, 1] == pytest.approx(bottom[-1, 1], abs=1.0)


class TestPagesWithoutAModel:
    def test_a_blank_page_gives_none(self) -> None:
        blank = np.full((1600, 1200), 235, dtype=np.uint8)

        assert build_auto_distortion_model(blank, DPI, CONTENT) is None

    def test_a_page_with_one_line_of_text_gives_none(self) -> None:
        page = np.full((1600, 1200), 235, dtype=np.uint8)
        cv2.putText(
            page,
            "just one line of text",
            (200, 600),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            20,
            2,
            cv2.LINE_AA,
        )

        assert build_auto_distortion_model(page, DPI, CONTENT) is None

    def test_noise_gives_none(self) -> None:
        noise = np.clip(
            np.random.default_rng(0).normal(180, 40, (1600, 1200)), 0, 255
        ).astype(np.uint8)

        assert build_auto_distortion_model(noise, DPI, CONTENT) is None


class TestOtherEdgeImage:
    def test_a_model_expressed_in_the_pixels_of_another_image(self) -> None:
        page = bent_book_page(45.0, -25.0)
        large = cv2.resize(
            page.image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
        )
        to_large = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

        plain = build_auto_distortion_model(page.image, DPI, CONTENT)
        scaled = build_auto_distortion_model(
            page.image, DPI, CONTENT, edge_gray=large, to_edge_image=to_large
        )

        assert plain is not None and scaled is not None
        assert plain.top_curve is not None and scaled.top_curve is not None
        np.testing.assert_allclose(
            scaled.top_curve.polyline[[0, -1]],
            2.0 * plain.top_curve.polyline[[0, -1]],
            atol=6.0,
        )

    def test_the_other_image_and_its_matrix_go_together(self) -> None:
        page = bent_book_page()

        with pytest.raises(ValueError, match="go together"):
            build_auto_distortion_model(page.image, DPI, CONTENT, edge_gray=page.image)
        with pytest.raises(ValueError, match="go together"):
            build_auto_distortion_model(
                page.image, DPI, CONTENT, to_edge_image=np.eye(3)
            )


def _model(
    top: list[tuple[float, float]], bottom: list[tuple[float, float]]
) -> DistortionModel:
    return DistortionModel(
        top_curve=Curve.from_polyline(np.array(top)),
        bottom_curve=Curve.from_polyline(np.array(bottom)),
    )


class TestStraighteningTheOuterBorder:
    """A border leaning over 2.75 degrees is made vertical by extending a curve."""

    def test_a_left_border_leaning_inwards_extends_the_bottom_curve(self) -> None:
        # The top-left end is further left (outer) than the bottom-left end.
        model = _model(
            [(100.0, 100.0), (900.0, 100.0)], [(300.0, 1000.0), (900.0, 1000.0)]
        )

        fixed = straighten_outer_border(model, spine_on_right=True)

        assert fixed.top_curve is model.top_curve
        assert fixed.bottom_curve is not None
        bottom = fixed.bottom_curve.polyline
        assert tuple(bottom[0]) == (100.0, 1000.0)
        assert len(bottom) == 3

    def test_a_left_border_leaning_outwards_extends_the_top_curve(self) -> None:
        """The C++ code replaces the bottom curve with the extended top one here."""
        model = _model(
            [(300.0, 100.0), (900.0, 100.0)], [(100.0, 1000.0), (900.0, 1000.0)]
        )

        fixed = straighten_outer_border(model, spine_on_right=True)

        assert fixed.bottom_curve is model.bottom_curve
        assert fixed.top_curve is not None
        top = fixed.top_curve.polyline
        assert tuple(top[0]) == (100.0, 100.0)
        assert len(top) == 3
        assert fixed.is_valid()

    def test_a_right_border_leaning_inwards_extends_the_bottom_curve(self) -> None:
        model = _model(
            [(100.0, 100.0), (900.0, 100.0)], [(100.0, 1000.0), (700.0, 1000.0)]
        )

        fixed = straighten_outer_border(model, spine_on_right=False)

        assert fixed.top_curve is model.top_curve
        assert fixed.bottom_curve is not None
        assert tuple(fixed.bottom_curve.polyline[-1]) == (900.0, 1000.0)

    def test_a_right_border_leaning_outwards_extends_the_top_curve(self) -> None:
        model = _model(
            [(100.0, 100.0), (700.0, 100.0)], [(100.0, 1000.0), (900.0, 1000.0)]
        )

        fixed = straighten_outer_border(model, spine_on_right=False)

        assert fixed.bottom_curve is model.bottom_curve
        assert fixed.top_curve is not None
        assert tuple(fixed.top_curve.polyline[-1]) == (900.0, 100.0)

    def test_a_border_leaning_at_most_2_75_degrees_is_left_alone(self) -> None:
        # atan(40 / 900) is 2.54 degrees.
        model = _model(
            [(100.0, 100.0), (900.0, 100.0)], [(140.0, 1000.0), (900.0, 1000.0)]
        )

        assert straighten_outer_border(model, spine_on_right=True) is model

    def test_only_the_outer_border_is_considered(self) -> None:
        # The right border leans strongly, but a single page's outer border is the
        # left one.
        model = _model(
            [(100.0, 100.0), (900.0, 100.0)], [(100.0, 1000.0), (600.0, 1000.0)]
        )

        assert straighten_outer_border(model, spine_on_right=True) is model

    def test_the_extension_point_is_in_single_precision(self) -> None:
        model = _model(
            [(100.123456789012, 100.0), (900.0, 100.0)],
            [(300.0, 1000.0), (900.0, 1000.0)],
        )

        fixed = straighten_outer_border(model, spine_on_right=True)

        assert fixed.bottom_curve is not None
        assert fixed.bottom_curve.polyline[0, 0] == float(np.float32(100.123456789012))


class TestTrivialModel:
    def test_it_is_the_content_rectangle(self) -> None:
        model = trivial_distortion_model((100.0, 200.0, 300.0, 400.0))

        assert model.is_valid()
        assert model.top_curve is not None and model.bottom_curve is not None
        np.testing.assert_array_equal(
            model.top_curve.polyline, [[100, 200], [400, 200]]
        )
        np.testing.assert_array_equal(
            model.bottom_curve.polyline, [[100, 600], [400, 600]]
        )

    def test_an_empty_rectangle_gives_a_one_pixel_square_around_its_corner(
        self,
    ) -> None:
        model = trivial_distortion_model((10.0, 20.0, 0.0, 0.0))

        assert model.top_curve is not None and model.bottom_curve is not None
        np.testing.assert_array_equal(
            model.top_curve.polyline, [[9.5, 19.5], [10.5, 19.5]]
        )
