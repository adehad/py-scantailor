"""Snake refinement of traced text lines.

Reference numbers marked "C++" come from compiling the upstream
``TextLineRefiner`` snake code with g++ 12 and running it on the same gradients.
The C++ code works in single precision, so positions agree to a few thousandths.
"""

import numpy as np
import pytest

from scantailor.dewarping.text_line_refiner import (
    _evolve_snake,
    _evolve_snakes,
    _make_snake,
    refine_text_lines,
)
from tests.dewarping.kernel_reference import formula_grid, recorded


def _band_page(
    top: int, bottom: int, width: int = 400, height: int = 220
) -> np.ndarray:
    page = np.full((height, width), 235, dtype=np.uint8)
    page[top:bottom, 20 : width - 20] = 25
    return page


class TestSnakeCreation:
    def test_nodes_are_spread_evenly_with_about_twenty_pixels_between_them(
        self,
    ) -> None:
        polyline = np.column_stack([np.linspace(0, 200, 11), np.full(11, 50.0)])

        cx, cy, rib = _make_snake(polyline)

        # One node per 20 pixels of length, the last one at the end of the line.
        assert len(cx) == 10
        np.testing.assert_allclose(np.diff(cx), 200 / 9)
        np.testing.assert_allclose(cy, 50.0)
        np.testing.assert_allclose(rib, 4.0)

    def test_both_ends_of_the_polyline_are_nodes(self) -> None:
        polyline = np.array([[10.0, 5.0], [75.0, 9.0], [133.0, 12.0]])

        cx, cy, _ = _make_snake(polyline)

        assert (cx[0], cy[0]) == (10.0, 5.0)
        assert (cx[-1], cy[-1]) == (133.0, 12.0)

    def test_a_polyline_shorter_than_two_spacings_becomes_one_node(self) -> None:
        cx, cy, _ = _make_snake(np.array([[10.0, 5.0], [40.0, 5.0]]))

        assert len(cx) == 1
        assert (cx[0], cy[0]) == (10.0, 5.0)


class TestRefinement:
    def test_a_line_moves_to_the_middle_of_the_band_of_ink_it_follows(self) -> None:
        page = _band_page(100, 120)
        xs = np.linspace(40, 360, 17)
        polyline = np.column_stack([xs, np.full_like(xs, 104.0)])

        (refined,) = refine_text_lines(page, [polyline], (0.0, 1.0))

        assert np.abs(refined[:, 1] - 110.0).max() < 3.0

    def test_every_line_comes_back_with_its_nodes(self) -> None:
        page = _band_page(100, 120)
        xs = np.linspace(40, 360, 17)
        lines = [np.column_stack([xs, np.full_like(xs, y)]) for y in (104.0, 108.0)]

        refined = refine_text_lines(page, lines, (0.0, 1.0))

        assert len(refined) == 2
        assert all(len(line) == 16 for line in refined)

    def test_no_lines_give_no_lines(self) -> None:
        assert refine_text_lines(_band_page(100, 120), [], (0.0, 1.0)) == []

    def test_a_short_line_comes_back_as_one_node(self) -> None:
        page = _band_page(100, 120)

        (refined,) = refine_text_lines(
            page, [np.array([[100.0, 104.0], [120.0, 104.0]])], (0.0, 1.0)
        )

        assert refined.shape == (1, 2)


class TestAgainstCpp:
    def _gradients(self) -> tuple[np.ndarray, np.ndarray]:
        height, width = 140, 300
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        centre = height / 2 + 8 * np.sin(np.pi * xx / width)
        base = -np.exp(-((yy - (centre - 8)) ** 2) / (2 * 3.0**2)) + np.exp(
            -((yy - (centre + 8)) ** 2) / (2 * 3.0**2)
        )
        coarse = (0.3 * base + 0.002 * np.sin(xx / 5) * np.cos(yy / 3)).astype(
            np.float32
        )
        fine = (0.3 * base + 0.002 * np.cos(xx / 4) * np.sin(yy / 3)).astype(np.float32)
        return coarse, fine

    def test_evolved_snake_matches_the_cpp_one(self) -> None:
        coarse, fine = self._gradients()
        xs = np.linspace(15, 285, 20)
        polyline = np.column_stack([xs, 68 + 5 * np.sin(np.pi * xs / 300)])
        cx, cy, rib = _make_snake(polyline)

        remaining = _evolve_snake(cx, cy, rib, coarse, 0.0, 1.0, 100, True)
        _evolve_snake(cx, cy, rib, fine, 0.0, 1.0, remaining, False)

        np.testing.assert_allclose(
            cy,
            [
                71.022,
                73.021,
                74.95,
                76.001,
                77.002,
                77.993,
                78.009,
                77.993,
                77.001,
                76.001,
                74.943,
                73.021,
                71.023,
            ],
            atol=0.05,
        )
        np.testing.assert_allclose(rib, 8.009, atol=0.01)

    def test_the_snake_stops_early_once_it_has_converged_on_the_coarse_gradient(
        self,
    ) -> None:
        coarse, _ = self._gradients()
        xs = np.linspace(15, 285, 20)
        polyline = np.column_stack([xs, 68 + 5 * np.sin(np.pi * xs / 300)])
        cx, cy, rib = _make_snake(polyline)

        remaining = _evolve_snake(cx, cy, rib, coarse, 0.0, 1.0, 100, True)

        assert remaining > 0

    def test_a_snake_without_iterations_is_left_alone(self) -> None:
        coarse, _ = self._gradients()
        xs = np.linspace(15, 285, 20)
        cx, cy, rib = _make_snake(np.column_stack([xs, np.full_like(xs, 68.0)]))
        before = cy.copy()

        assert _evolve_snake(cx, cy, rib, coarse, 0.0, 1.0, 0, True) == 0
        np.testing.assert_array_equal(cy, before)


class TestRecordedOutputs:
    """Snakes evolve on formula gradients exactly as recorded."""

    @pytest.mark.parametrize(
        ("snake", "stage"), [(snake, stage) for snake in range(5) for stage in (0, 1)]
    )
    def test_a_snake_evolves_as_recorded(self, snake: int, stage: int) -> None:
        remaining, stop = [(100, True), (37, False)][stage]
        cx, cy, rib = recorded(f"evolve{snake}_{stage}_in")

        left = _evolve_snake(
            cx, cy, rib, formula_grid(140, 300, 23, 17), 0.0, 1.0, remaining, stop
        )

        assert left == recorded(f"evolve{snake}_{stage}_left")
        np.testing.assert_array_equal(
            np.stack([cx, cy, rib]), recorded(f"evolve{snake}_{stage}_out")
        )

    def test_snakes_of_different_lengths_evolve_together_as_they_do_alone(
        self,
    ) -> None:
        snakes = [tuple(recorded(f"batch{n}_in")) for n in range(6)]

        left = _evolve_snakes(
            snakes, formula_grid(140, 300, 23, 17), (0.0, 1.0), [100] * 6, True
        )
        _evolve_snakes(snakes, formula_grid(140, 300, 13, 11), (0.0, 1.0), left, False)

        for n, (cx, cy, rib) in enumerate(snakes):
            np.testing.assert_array_equal(
                np.stack([cx, cy, rib]), recorded(f"batch{n}_out")
            )
