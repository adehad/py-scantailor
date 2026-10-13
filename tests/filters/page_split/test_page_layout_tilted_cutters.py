"""Page outlines follow tilted cutter lines instead of their mid x."""

import numpy as np

from scantailor.filters.page_split.page_layout import PageLayout, PageLayoutType

OUTLINE = (0.0, 0.0, 100.0, 100.0)


def test_two_pages_split_along_a_tilted_cutter() -> None:
    line = ((40.0, 0.0), (60.0, 100.0))
    layout = PageLayout(
        outline=OUTLINE,
        layout_type=PageLayoutType.TWO_PAGES,
        cutter1=line,
        cutter2=line,
    )

    left = layout.get_left_page_outline()
    right = layout.get_right_page_outline()

    assert left is not None and right is not None
    np.testing.assert_allclose(left, [[0, 0], [40, 0], [60, 100], [0, 100]])
    np.testing.assert_allclose(right, [[40, 0], [100, 0], [100, 100], [60, 100]])


def test_a_page_between_two_tilted_cutters_is_a_quadrilateral() -> None:
    layout = PageLayout(
        outline=OUTLINE,
        layout_type=PageLayoutType.SINGLE_PAGE_CUT,
        cutter1=((10.0, 0.0), (20.0, 100.0)),
        cutter2=((80.0, 0.0), (70.0, 100.0)),
    )

    outline = layout.get_single_page_outline()

    assert outline is not None
    np.testing.assert_allclose(outline, [[10, 0], [80, 0], [70, 100], [20, 100]])


def test_cutters_given_in_the_wrong_order_still_bound_the_page() -> None:
    layout = PageLayout(
        outline=OUTLINE,
        layout_type=PageLayoutType.SINGLE_PAGE_CUT,
        cutter1=((80.0, 0.0), (70.0, 100.0)),
        cutter2=((10.0, 0.0), (20.0, 100.0)),
    )

    outline = layout.get_single_page_outline()

    assert outline is not None
    np.testing.assert_allclose(outline, [[10, 0], [80, 0], [70, 100], [20, 100]])


def test_a_cutter_that_extends_past_the_outline_is_extended_to_its_edges() -> None:
    line = ((50.0, 25.0), (50.0, 75.0))
    layout = PageLayout(
        outline=OUTLINE,
        layout_type=PageLayoutType.TWO_PAGES,
        cutter1=line,
        cutter2=line,
    )

    left = layout.get_left_page_outline()

    assert left is not None
    np.testing.assert_allclose(left, [[0, 0], [50, 0], [50, 100], [0, 100]])
