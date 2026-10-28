"""The cylindrical surface dewarper on surfaces whose dewarped coordinates are known.

Straight directrices make a flat page, whose dewarped coordinates are those of
the homography that takes the unit square to the four corners. Curved ones make
a cylinder: each directrix is a line of constant y, and x follows the arc length.
"""

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from scantailor.dewarping.dewarper import (
    CylindricalSurfaceDewarper,
    _coupled_polylines,
    _four_point_homography,
    _invert,
)

# Mapping a point solves small linear systems and intersects polylines in
# floating point; anything above a millionth of the page is a real error.
_EXACT = 1e-6

UNIT_CORNERS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)


def _straight(start: ArrayLike, end: ArrayLike) -> NDArray[np.float64]:
    """Eleven points evenly spaced along a straight directrix.

    Args:
        start (ArrayLike): (x, y) of the first point.
        end (ArrayLike): (x, y) of the last point.

    Returns:
        NDArray[np.float64]: (11, 2) points from ``start`` to ``end``.
    """
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    return first + np.linspace(0.0, 1.0, 11)[:, None] * (last - first)


def _directrices() -> tuple[np.ndarray, np.ndarray]:
    """Top and bottom directrices of an asymmetric, slightly tilted curved page.

    Returns:
        tuple[np.ndarray, np.ndarray]: (41, 2) points of the
        top and of the bottom directrix, left to right.
    """
    t = np.arange(41)
    top = np.column_stack(
        [100 + 25 * t, 300 + 45 * np.sin(np.pi * t / 40) + 0.03 * 25 * t]
    )
    bottom = np.column_stack(
        [90 + 26 * t, 1300 - 30 * np.sin(np.pi * t / 40) + 0.01 * 25 * t]
    )
    return top, bottom


class TestFlatPage:
    def test_a_parallelogram_maps_by_its_affine_coordinates(self) -> None:
        """A point at origin + u * (top edge) + v * (left edge) maps to (u, v)."""
        origin = np.array([100.0, 50.0])
        across = np.array([400.0, 40.0])
        down = np.array([30.0, 600.0])
        top = _straight(origin, origin + across)
        bottom = _straight(origin + down, origin + down + across)
        uv = np.array(
            [*UNIT_CORNERS, [0.5, 0.5], [0.25, 0.75], [0.9, 0.1]], dtype=float
        )
        points = origin + uv[:, :1] * across + uv[:, 1:] * down

        dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom, 2.0)

        np.testing.assert_allclose(
            dewarper.map_points_to_dewarped_space(points), uv, atol=_EXACT
        )
        assert dewarper.directrix_arc_length == pytest.approx(1.0, abs=_EXACT)

    def test_a_trapezoid_maps_by_perspective_not_by_proportion(self) -> None:
        """A page seen in perspective: the near bottom edge looks wider.

        The diagonals cross at the page centre, (300, 322.2) in the image, which
        is well above the trapezoid's half height of 350.
        """
        top = _straight([100, 100], [500, 100])
        bottom = _straight([50, 600], [550, 600])
        points = np.array(
            [[100, 100], [500, 100], [50, 600], [550, 600], [300, 2900 / 9]]
        )

        dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom, 2.0)

        np.testing.assert_allclose(
            dewarper.map_points_to_dewarped_space(points),
            [*UNIT_CORNERS, [0.5, 0.5]],
            atol=_EXACT,
        )


def _symmetric_cylinder() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A page bulging toward the camera, mirror symmetric about x = 600.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: (41, 2) points of the
        top and of the bottom directrix, left to right.
    """
    t = np.arange(41)
    bulge = 60 * np.sin(np.pi * t / 40)
    top = np.column_stack([100 + 25 * t, 300 - bulge])
    bottom = np.column_stack([100 + 25 * t, 1300 + bulge])
    return top, bottom


class TestCylinder:
    def test_each_directrix_is_a_line_of_constant_dewarped_y(self) -> None:
        top, bottom = _directrices()

        dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom, 2.0)

        on_top = dewarper.map_points_to_dewarped_space(top)
        on_bottom = dewarper.map_points_to_dewarped_space(bottom)
        np.testing.assert_allclose(on_top[:, 1], 0.0, atol=_EXACT)
        np.testing.assert_allclose(on_bottom[:, 1], 1.0, atol=_EXACT)
        for mapped in (on_top, on_bottom):
            assert mapped[0, 0] == pytest.approx(0.0, abs=_EXACT)
            assert mapped[-1, 0] == pytest.approx(1.0, abs=_EXACT)
            assert np.all(np.diff(mapped[:, 0]) > 0)

    def test_a_symmetric_cylinder_maps_symmetrically(self) -> None:
        top, bottom = _symmetric_cylinder()
        points = np.array([[600, 800], [600, 240], [600, 1360], [350, 800], [850, 800]])

        dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom, 2.0)

        mapped = dewarper.map_points_to_dewarped_space(points)
        np.testing.assert_allclose(
            mapped[:3], [[0.5, 0.5], [0.5, 0.0], [0.5, 1.0]], atol=_EXACT
        )
        assert mapped[3, 0] + mapped[4, 0] == pytest.approx(1.0, abs=_EXACT)
        assert mapped[3, 1] == pytest.approx(0.5, abs=_EXACT)

    def test_the_inclined_sides_of_a_cylinder_hold_more_of_the_page(self) -> None:
        """The page turns away from the camera toward its sides, so a quarter of
        the image width from the left edge covers more than a quarter of the
        page, and the curved directrix is longer than its chord.
        """
        top, bottom = _symmetric_cylinder()

        dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom, 2.0)

        quarter = dewarper.map_to_dewarped_space(np.array([350.0, 800.0]))
        assert quarter[0] > 0.25 + 0.01
        assert dewarper.directrix_arc_length > 1.0 + 0.01

    def test_dewarped_points_map_back_to_where_they_came_from(self) -> None:
        top, bottom = _directrices()
        dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom, 2.0)
        inside = np.array(
            [[500, 360], [300, 800], [600, 950], [900, 1100]], dtype=float
        )

        for point, dewarped in zip(
            inside, dewarper.map_points_to_dewarped_space(inside), strict=True
        ):
            np.testing.assert_allclose(
                dewarper.map_to_warped_space(dewarped), point, atol=1e-5
            )


class TestSteps:
    def test_the_four_corners_determine_a_homography_mapping_them_exactly(self) -> None:
        plane = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        image = np.array([[10, 20], [310, 40], [5, 420], [330, 460]], dtype=float)

        homography = _four_point_homography(plane, image)

        mapped = np.column_stack([plane, np.ones(4)]) @ homography.matrix.T
        np.testing.assert_allclose(mapped[:, :2] / mapped[:, 2:], image, atol=1e-9)

    def test_collinear_corners_do_not_determine_a_homography(self) -> None:
        plane = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        image = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)

        with pytest.raises(np.linalg.LinAlgError):
            _four_point_homography(plane, image)

    def test_inverting_gives_the_inverse_mapping(self) -> None:
        plane = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        image = np.array([[10, 20], [310, 40], [5, 420], [330, 460]], dtype=float)
        homography = _four_point_homography(plane, image)

        product = _invert(homography).matrix @ homography.matrix

        np.testing.assert_allclose(product, np.eye(3), atol=1e-9)

    def test_the_coupled_walk_covers_the_surface_from_the_first_corner_to_the_last(
        self,
    ) -> None:
        top, bottom = _directrices()
        corners_plane = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        corners_image = np.array([top[0], top[-1], bottom[0], bottom[-1]])
        pln2img = _four_point_homography(corners_plane, corners_image)
        img2pln = _invert(pln2img)

        steps = list(_coupled_polylines(top, bottom, pln2img, img2pln))

        # A vertex within a pixel of the other curve's point is not walked twice.
        assert len(top) <= len(steps) <= len(top) + len(bottom)
        plane_x = [step[2] for step in steps]
        assert plane_x[0] == pytest.approx(0.0, abs=1e-9)
        assert plane_x[-1] == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.diff(plane_x) > -1e-9)
