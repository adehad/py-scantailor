"""Points mapped through the dewarp land where the generated output shows them."""

import numpy as np
import pytest

from scantailor.dewarping.page_dewarper import dewarper_from_polylines
from scantailor.dewarping.point_mapper import DewarpingPointMapper
from scantailor.dewarping.raster_dewarper import (
    compute_dewarp_maps,
    output_model_domain,
    remap_with_maps,
)
from scantailor.filters.output.generator import _apply_post_deskew

HEIGHT, WIDTH = 240, 320
DEPTH = 2.0
BLACK = 0
WHITE = 255


def _edges() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    xs = np.linspace(0.0, WIDTH, 9)
    sag = 14.0 * np.sin(np.pi * xs / WIDTH)
    top = [(float(x), float(20.0 + s)) for x, s in zip(xs, sag)]
    bottom = [(float(x), float(HEIGHT - 20.0 + s)) for x, s in zip(xs, sag)]
    return top, bottom


def _page_with_dot(centre: tuple[int, int]) -> np.ndarray:
    page = np.full((HEIGHT, WIDTH), WHITE, dtype=np.uint8)
    x, y = centre
    page[y - 2 : y + 3, x - 2 : x + 3] = BLACK
    return page


def _dewarped(page: np.ndarray, angle: float) -> np.ndarray:
    top, bottom = _edges()
    dewarper = dewarper_from_polylines(top, bottom, DEPTH)
    assert dewarper is not None
    size = (WIDTH, HEIGHT)
    maps = compute_dewarp_maps(dewarper, size, output_model_domain(dewarper, size))
    out = remap_with_maps(page, maps, WHITE)
    return _apply_post_deskew(out, angle, WHITE) if angle else out


def _dot_centre(image: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(image < 128)
    assert xs.size
    return np.array([xs.mean() + 0.5, ys.mean() + 0.5])


def _mapper(angle: float) -> DewarpingPointMapper:
    top, bottom = _edges()
    return DewarpingPointMapper(top, bottom, DEPTH, (WIDTH, HEIGHT), angle)


@pytest.mark.parametrize("angle", [0.0, 3.0, -2.5])
@pytest.mark.parametrize("dot", [(80, 90), (240, 150), (160, 60)])
def test_a_page_point_lands_on_the_pixel_the_output_shows_it_at(angle, dot):
    page = _page_with_dot(dot)
    shown = _dot_centre(_dewarped(page, angle))

    mapped = _mapper(angle).to_dewarped(np.array([[dot[0] + 0.5, dot[1] + 0.5]]))[0]

    assert mapped == pytest.approx(shown, abs=1.5)


@pytest.mark.parametrize("angle", [0.0, 4.0])
def test_mapping_to_the_output_and_back_returns_the_page_point(angle):
    points = np.array([[60.0, 70.0], [250.0, 160.0], [160.0, 120.0]])
    mapper = _mapper(angle)

    back = mapper.to_warped(mapper.to_dewarped(points))

    assert back == pytest.approx(points, abs=0.05)


def test_an_empty_set_of_points_maps_to_an_empty_set():
    assert _mapper(0.0).to_warped(np.empty((0, 2))).shape == (0, 2)


def test_edges_that_make_no_surface_are_refused():
    with pytest.raises(ValueError):
        DewarpingPointMapper([(0.0, 0.0)], [(0.0, 10.0)], DEPTH, (WIDTH, HEIGHT))
