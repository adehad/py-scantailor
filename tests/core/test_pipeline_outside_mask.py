"""Tests for the mask of view pixels that have no source pixel."""

import cv2
import numpy as np

from scantailor.core.pipeline import _PageState
from scantailor.core.transformation import render_transformed


def test_unrotated_view_has_no_outside_pixels() -> None:
    """A view identical to the source is covered everywhere."""
    state = _PageState(np.full((120, 200), 128, dtype=np.uint8), dpi=300)

    mask = state.outside_source_mask()

    assert mask.shape == (120, 200)
    assert not mask.any()


def test_rotated_view_marks_the_exposed_corners_only() -> None:
    """Deskew rotation exposes corners; the centre stays covered."""
    state = _PageState(np.full((300, 400), 128, dtype=np.uint8), dpi=300)
    state.xform.set_post_rotation(10.0)
    view = state.view()

    mask = state.outside_source_mask()

    assert mask.shape == view.shape[:2]
    assert mask[0, 0]
    assert not mask[mask.shape[0] // 2, mask.shape[1] // 2]


def test_the_mask_differs_from_resampling_a_blank_source_only_along_its_edges() -> None:
    state = _PageState(np.full((240, 180, 3), 90, dtype=np.uint8), dpi=300)
    state.crop_to(np.array([(10.0, 5.0), (170.0, 20.0), (160.0, 230.0), (0.0, 220.0)]))
    state.xform.set_post_rotation(7.5)
    state.xform.set_post_oblique(3.0)
    resampled = render_transformed(np.zeros((240, 180), np.uint8), state.xform) > 127

    mask = state.outside_source_mask()

    edges = cv2.morphologyEx(
        mask.view(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    assert mask.any() and not mask.all()
    assert not (mask != resampled)[~edges].any()


def test_colour_source_gives_a_two_dimensional_mask() -> None:
    """Colour sources collapse to one boolean plane."""
    state = _PageState(np.full((100, 100, 3), 90, dtype=np.uint8), dpi=300)
    state.xform.set_post_rotation(5.0)

    mask = state.outside_source_mask()

    assert mask.ndim == 2
    assert mask.dtype == np.bool_


def test_a_mask_at_a_lower_resolution_has_the_shrunk_view_size() -> None:
    state = _PageState(np.full((1200, 900), 90, dtype=np.uint8), dpi=600)
    state.xform.set_post_rotation(4.0)

    mask = state.outside_source_mask(150)

    width, height = state.size_at(150)
    assert mask.shape == (height, width)
    assert mask[0, 0]
    assert not mask[height // 2, width // 2]
