"""The cylindrical surface dewarper against the upstream C++ one.

The expected numbers come from compiling upstream's ``CylindricalSurfaceDewarper``
(with its arc length mapper and polyline intersector) with g++ 12 and mapping
the same points. The distortion model builder scores its candidate curves with
this mapping, so the two have to agree.
"""

import numpy as np
import pytest

from scantailor.dewarping.dewarper import (
    CylindricalSurfaceDewarper,
    _coupled_polylines,
    _four_point_homography,
    _invert,
)

POINTS = np.array(
    [
        [120, 320],
        [500, 360],
        [950, 420],
        [300, 800],
        [600, 950],
        [900, 1100],
        [200, 1250],
        [700, 1280],
        [1050, 1310],
        [500, 250],
        [640, 1400],
        [80, 700],
    ],
    dtype=float,
)
EXPECTED = np.array(
    [
        [0.020679, 0.017299],
        [0.400663, 0.005881],
        [0.838795, 0.080761],
        [0.205663, 0.501059],
        [0.489369, 0.653919],
        [0.771761, 0.804624],
        [0.108243, 0.959167],
        [0.577436, 1.003161],
        [0.918208, 1.00795],
        [0.401317, -0.119032],
        [0.521391, 1.130259],
        [-0.016118, 0.409887],
    ]
)


def _directrices() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(41)
    top = np.column_stack(
        [100 + 25 * t, 300 + 45 * np.sin(np.pi * t / 40) + 0.03 * 25 * t]
    )
    bottom = np.column_stack(
        [90 + 26 * t, 1300 - 30 * np.sin(np.pi * t / 40) + 0.01 * 25 * t]
    )
    return top, bottom


class TestAgainstCpp:
    def test_points_map_to_the_same_dewarped_coordinates(self) -> None:
        top, bottom = _directrices()

        dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom, 2.0)

        np.testing.assert_allclose(
            dewarper.map_points_to_dewarped_space(POINTS), EXPECTED, atol=2e-6
        )

    def test_the_arc_length_of_the_directrix_is_the_same(self) -> None:
        top, bottom = _directrices()

        dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom, 2.0)

        assert dewarper.directrix_arc_length == pytest.approx(
            1.06208583730044, abs=1e-9
        )

    def test_dewarped_points_map_back_to_where_they_came_from(self) -> None:
        top, bottom = _directrices()
        dewarper = CylindricalSurfaceDewarper.from_directrices(top, bottom, 2.0)
        inside = POINTS[[1, 3, 4, 5]]

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
