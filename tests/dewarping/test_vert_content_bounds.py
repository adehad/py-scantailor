"""Left and right content bounds of a cleaned binary page.

Reference numbers marked "C++" come from compiling the upstream sources of
``SequentialColumnProcessor`` with g++ 12. The RANSAC seeds nearest the edge are
ranked by their real distance, where the C++ code ranks them by a boolean, so
the reference build ranks them the same way.
"""

import numpy as np
import pytest

from scantailor.dewarping.vert_content_bounds import detect_vert_content_bounds


def _block(width: int, height: int, left: int, right: int, top: int, bottom: int):
    page = np.zeros((height, width), dtype=bool)
    page[top:bottom, left:right] = True
    return page


def _x_at(line, y: float) -> float:
    (x1, y1), (x2, y2) = line
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)


class TestBoundsOfSimpleContent:
    def test_a_rectangle_is_bounded_one_pixel_outside_each_side(self) -> None:
        page = _block(300, 200, 50, 250, 30, 170)

        left, right = detect_vert_content_bounds(page)

        assert _x_at(left, 0) == pytest.approx(49.0)
        assert _x_at(left, 200) == pytest.approx(49.0)
        assert _x_at(right, 0) == pytest.approx(250.0)
        assert _x_at(right, 200) == pytest.approx(250.0)

    def test_bounds_run_from_the_top_to_the_bottom_of_the_image(self) -> None:
        page = _block(300, 200, 50, 250, 30, 170)

        for line in detect_vert_content_bounds(page):
            assert line[0][1] == pytest.approx(0.0)
            assert line[1][1] == pytest.approx(200.0)

    def test_a_page_without_ink_has_vertical_bounds_next_to_the_left_edge(self) -> None:
        left, right = detect_vert_content_bounds(np.zeros((100, 150), dtype=bool))

        assert _x_at(left, 50) == pytest.approx(-1.0)
        assert _x_at(right, 50) == pytest.approx(1.0)

    def test_a_skewed_block_gives_skewed_bounds(self) -> None:
        page = np.zeros((300, 400), dtype=bool)
        for y in range(20, 280):
            shift = int(0.05 * y)
            page[y, 60 + shift : 340 + shift] = True

        left, right = detect_vert_content_bounds(page)

        for line in (left, right):
            slope = (line[1][0] - line[0][0]) / (line[1][1] - line[0][1])
            assert slope == pytest.approx(0.05, abs=0.01)


class TestAgainstCpp:
    def _page(self) -> np.ndarray:
        page = np.zeros((300, 360), dtype=bool)
        page[8:11, 10:340] = True
        for line_index in range(28):
            y = 20 + 9 * line_index
            x0 = int(40 + 0.03 * y)
            x1 = int(300 + 0.02 * y)
            page[y : y + 3, x0:x1] = True
            page[y + 4 : y + 6, x0 + 6 : x1 - 30] = True
        return page

    def test_the_bounds_match_the_cpp_ones(self) -> None:
        left, right = detect_vert_content_bounds(self._page())

        np.testing.assert_allclose(
            np.array(left).reshape(-1), [7.54902, 0.0, 51.078431, 300.0], atol=1e-5
        )
        np.testing.assert_allclose(
            np.array(right).reshape(-1), [341.372549, 0.0, 300.196078, 300.0], atol=1e-5
        )

    def test_repeating_the_detection_gives_the_same_bounds(self) -> None:
        page = self._page()

        assert detect_vert_content_bounds(page) == detect_vert_content_bounds(page)
