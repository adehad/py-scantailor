"""Tests for page layout geometry helpers."""

import numpy as np
import pytest

from scantailor.core.models import Dpi, Margins
from scantailor.core.transformation import ImageTransformation, Rect
from scantailor.filters.page_layout.alignment import (
    Alignment,
    HorizontalAlignment,
    VerticalAlignment,
)
from scantailor.filters.page_layout.page_rect import (
    adapt_content_rect,
    calc_margins_mm,
    calc_page_polygon,
    calc_soft_margins_mm,
    rect_size_mm,
)
from scantailor.filters.page_layout.params import ContentRect, ContentSize, Params

# 254 dpi: ten pixels per millimetre.
DPI = Dpi.uniform(254)


@pytest.fixture
def xform() -> ImageTransformation:
    """Identity transformation over a 200 x 300 px (20 x 30 mm) image."""
    return ImageTransformation.from_size(200, 300, DPI)


class TestRectSizeMm:
    """Rectangle sizes are measured on the original image."""

    def test_axis_aligned_rect(self, xform: ImageTransformation) -> None:
        """Ten pixels per millimetre."""
        size = rect_size_mm(xform, Rect(x=10, y=20, width=50, height=100))

        assert size == pytest.approx((5.0, 10.0))

    def test_rotation_does_not_change_physical_size(self) -> None:
        """A rectangle in the deskewed view keeps its millimetre size."""
        rotated = ImageTransformation.from_size(200, 300, DPI)
        rotated.set_post_rotation(10.0)

        size = rect_size_mm(rotated, Rect(x=30, y=30, width=50, height=100))

        assert size == pytest.approx((5.0, 10.0))


class TestAdaptContentRect:
    """Empty content is replaced by a point-like rectangle at the centre."""

    def test_non_empty_rect_is_returned_unchanged(
        self, xform: ImageTransformation
    ) -> None:
        """Real content is left alone."""
        rect = Rect(x=1, y=2, width=3, height=4)

        assert adapt_content_rect(xform, rect) is rect

    def test_empty_rect_becomes_centred_stand_in(
        self, xform: ImageTransformation
    ) -> None:
        """The stand-in is centred in the view and has area."""
        adapted = adapt_content_rect(xform, Rect())

        assert not adapted.is_empty()
        assert adapted.center() == pytest.approx((100, 150))


class TestCalcMarginsMm:
    """Auto margins reproduce the page size around the content."""

    def test_margins_follow_pixel_distances(self, xform: ImageTransformation) -> None:
        """Left/right split the spare width in proportion to the pixel gaps."""
        page = Rect(x=0, y=0, width=200, height=300)
        content = Rect(x=50, y=30, width=100, height=200)

        margins = calc_margins_mm(xform, page, content)

        assert margins.left == pytest.approx(5.0)
        assert margins.right == pytest.approx(5.0)
        assert margins.top == pytest.approx(3.0)
        assert margins.bottom == pytest.approx(7.0)

    def test_content_filling_the_page_gives_zero_margins(
        self, xform: ImageTransformation
    ) -> None:
        """No spare room, no margins."""
        page = Rect(x=0, y=0, width=200, height=300)

        margins = calc_margins_mm(xform, page, page)

        assert margins == Margins()


class TestCalcSoftMarginsMm:
    """The shortfall against the aggregate size is placed by alignment."""

    @staticmethod
    def _soft(alignment: Alignment) -> Margins:
        content = Rect(x=50, y=50, width=100, height=100)
        page = Rect(x=0, y=0, width=200, height=200)
        return calc_soft_margins_mm(
            hard_size_mm=(10.0, 10.0),
            aggregate_size_mm=(14.0, 18.0),
            alignment=alignment,
            content_rect=content,
            content_size_mm=(10.0, 10.0),
            page_rect=page,
        )

    def test_centered_splits_evenly(self) -> None:
        """Both axes put half of the shortfall on each side."""
        margins = self._soft(Alignment.centered())

        assert (margins.left, margins.right) == pytest.approx((2.0, 2.0))
        assert (margins.top, margins.bottom) == pytest.approx((4.0, 4.0))

    def test_top_left_puts_everything_after_the_content(self) -> None:
        """Left and top alignment pad the right and bottom."""
        margins = self._soft(Alignment.top_left())

        assert (margins.left, margins.right) == pytest.approx((0.0, 4.0))
        assert (margins.top, margins.bottom) == pytest.approx((0.0, 8.0))

    def test_right_bottom_puts_everything_before_the_content(self) -> None:
        """Right and bottom alignment pad the left and top."""
        alignment = Alignment(
            horizontal=HorizontalAlignment.RIGHT, vertical=VerticalAlignment.BOTTOM
        )

        margins = self._soft(alignment)

        assert (margins.left, margins.right) == pytest.approx((4.0, 0.0))
        assert (margins.top, margins.bottom) == pytest.approx((8.0, 0.0))

    def test_independent_page_gets_no_soft_margins(self) -> None:
        """A null alignment opts the page out."""
        assert self._soft(Alignment.independent()) == Margins()

    def test_auto_alignment_snaps_to_the_nearer_side(self) -> None:
        """Content sitting at the left of the page is aligned left."""
        alignment = Alignment(
            horizontal=HorizontalAlignment.HAUTO, vertical=VerticalAlignment.VAUTO
        )

        margins = calc_soft_margins_mm(
            hard_size_mm=(10.0, 10.0),
            aggregate_size_mm=(20.0, 10.0),
            alignment=alignment,
            content_rect=Rect(x=0, y=0, width=100, height=100),
            content_size_mm=(10.0, 10.0),
            page_rect=Rect(x=0, y=0, width=300, height=100),
        )

        assert margins.left == pytest.approx(0.0)
        assert margins.right == pytest.approx(10.0)

    def test_no_shortfall_gives_zero_margins(self) -> None:
        """A page already at the aggregate size is not padded."""
        margins = calc_soft_margins_mm(
            hard_size_mm=(10.0, 10.0),
            aggregate_size_mm=(10.0, 10.0),
            alignment=Alignment.centered(),
            content_rect=Rect(x=0, y=0, width=100, height=100),
            content_size_mm=(10.0, 10.0),
            page_rect=Rect(x=0, y=0, width=100, height=100),
        )

        assert margins == Margins()


class TestCalcPagePolygon:
    """The page polygon is content plus hard and soft margins."""

    def test_polygon_covers_content_and_margins(
        self, xform: ImageTransformation
    ) -> None:
        """Hard margins extend each edge by their millimetre size in pixels."""
        content = Rect(x=50, y=60, width=100, height=120)
        params = Params(
            hard_margins_mm=Margins(left=2, right=3, top=1, bottom=4),
            content_size_mm=ContentSize(width_mm=10, height_mm=12),
            auto_margins=False,
        )

        polygon = calc_page_polygon(xform, content, params, (15.0, 17.0))

        assert polygon.min(axis=0) == pytest.approx((30, 50))
        assert polygon.max(axis=0) == pytest.approx((180, 220))

    def test_polygon_origin_is_on_the_pixel_grid(
        self, xform: ImageTransformation
    ) -> None:
        """Fractional margins still give whole-pixel top-left corner."""
        content = Rect(x=50, y=60, width=100, height=120)
        params = Params(
            hard_margins_mm=Margins(left=1.23, top=0.57),
            content_size_mm=ContentSize(width_mm=10, height_mm=12),
            content_rect=ContentRect(x=50, y=60, width=100, height=120),
            auto_margins=False,
        )

        polygon = calc_page_polygon(xform, content, params, (0.0, 0.0))

        origin = polygon.min(axis=0)
        assert np.allclose(origin, np.round(origin))
