"""Tests for the mask of view pixels that have no source pixel."""

import numpy as np

from scantailor.core.pipeline import _outside_source_mask, _PageState
from scantailor.core.transformation import render_transformed


def test_unrotated_view_has_no_outside_pixels() -> None:
    """A view identical to the source is covered everywhere."""
    state = _PageState(np.full((120, 200), 128, dtype=np.uint8), dpi=300)

    mask = _outside_source_mask(state)

    assert mask.shape == (120, 200)
    assert not mask.any()


def test_rotated_view_marks_the_exposed_corners_only() -> None:
    """Deskew rotation exposes corners; the centre stays covered."""
    state = _PageState(np.full((300, 400), 128, dtype=np.uint8), dpi=300)
    state.xform.set_post_rotation(10.0)
    view = state.view()

    mask = _outside_source_mask(state)

    assert mask.shape == view.shape[:2]
    assert mask[0, 0]
    assert not mask[mask.shape[0] // 2, mask.shape[1] // 2]


def test_colour_source_marks_what_any_of_its_channels_would() -> None:
    """The mask of a colour page matches resampling every channel and merging."""
    state = _PageState(np.full((240, 180, 3), 90, dtype=np.uint8), dpi=300)
    state.crop_to(np.array([(10.0, 5.0), (170.0, 20.0), (160.0, 230.0), (0.0, 220.0)]))
    state.xform.set_post_rotation(7.5)
    state.xform.set_post_oblique(3.0)
    every_channel = render_transformed(np.zeros_like(state.source), state.xform)

    mask = _outside_source_mask(state)

    np.testing.assert_array_equal(mask, np.any(every_channel > 127, axis=2))
    assert mask.any() and not mask.all()


def test_colour_source_gives_a_two_dimensional_mask() -> None:
    """Colour sources collapse to one boolean plane."""
    state = _PageState(np.full((100, 100, 3), 90, dtype=np.uint8), dpi=300)
    state.xform.set_post_rotation(5.0)

    mask = _outside_source_mask(state)

    assert mask.ndim == 2
    assert mask.dtype == np.bool_
