"""Tests for greyscale morphological reconstruction."""

import numpy as np
import pytest

from scantailor.imageproc.reconstruction import reconstruction

_NEIGHBOURS = {
    4: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    8: [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)],
}


def _geodesic_dilation_until_stable(seed, mask, connectivity):
    """Reference: one-pixel geodesic dilations repeated until nothing changes."""
    lowest = -np.inf if seed.dtype.kind == "f" else np.iinfo(seed.dtype).min
    marker = seed.copy()
    while True:
        padded = np.pad(marker, 1, constant_values=lowest)
        grown = marker.copy()
        height, width = marker.shape
        for dy, dx in _NEIGHBOURS[connectivity]:
            shifted = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            np.maximum(grown, shifted, out=grown)
        grown = np.minimum(grown, mask)
        if np.array_equal(grown, marker):
            return marker
        marker = grown


def _expected(seed, mask, method, connectivity):
    if method == "dilation":
        return _geodesic_dilation_until_stable(seed, mask, connectivity)
    flip = (lambda image: -image) if seed.dtype.kind == "f" else np.invert
    return flip(_geodesic_dilation_until_stable(flip(seed), flip(mask), connectivity))


def _random_pair(rng, shape, method, dtype=np.uint8):
    info = np.iinfo(dtype)
    mask = rng.integers(info.min, info.max, shape, endpoint=True, dtype=dtype)
    noise = rng.integers(info.min, info.max, shape, endpoint=True, dtype=dtype)
    seed = np.minimum(noise, mask) if method == "dilation" else np.maximum(noise, mask)
    return seed, mask


@pytest.mark.parametrize("method", ["dilation", "erosion"])
@pytest.mark.parametrize("connectivity", [4, 8])
class TestMatchesRepeatedGeodesicDilation:
    def test_random_images_give_the_same_result_as_the_reference(
        self, method, connectivity
    ):
        rng = np.random.default_rng(1)
        for shape in [(1, 1), (1, 17), (13, 1), (2, 2), (31, 47), (64, 64)]:
            seed, mask = _random_pair(rng, shape, method)
            np.testing.assert_array_equal(
                reconstruction(seed, mask, method, connectivity),
                _expected(seed, mask, method, connectivity),
            )

    def test_smooth_random_images_with_long_propagation_paths(
        self, method, connectivity
    ):
        rng = np.random.default_rng(2)
        mask = np.asarray(
            np.cumsum(rng.integers(-3, 4, (80, 80)), axis=1) % 256, dtype=np.uint8
        )
        seed = np.where(
            rng.random(mask.shape) < 0.01, mask, 0 if method == "dilation" else 255
        )
        seed = seed.astype(np.uint8)
        np.testing.assert_array_equal(
            reconstruction(seed, mask, method, connectivity),
            _expected(seed, mask, method, connectivity),
        )

    def test_single_seed_pixel_floods_its_connected_plateau(self, method, connectivity):
        mask = np.full((20, 20), 200 if method == "dilation" else 50, dtype=np.uint8)
        mask[10, :] = 0 if method == "dilation" else 255
        outside = 0 if method == "dilation" else 255
        seed = np.full_like(mask, outside)
        seed[3, 4] = mask[3, 4]
        result = reconstruction(seed, mask, method, connectivity)
        np.testing.assert_array_equal(
            result, _expected(seed, mask, method, connectivity)
        )
        assert result[3, 4] == mask[3, 4]
        assert result[9, 19] == mask[9, 19]
        assert result[11, 0] == outside

    def test_flat_images_are_returned_unchanged(self, method, connectivity):
        flat = np.full((9, 11), 77, dtype=np.uint8)
        np.testing.assert_array_equal(
            reconstruction(flat, flat, method, connectivity), flat
        )

    def test_uint8_extremes(self, method, connectivity):
        rng = np.random.default_rng(3)
        mask = rng.choice(np.array([0, 255], dtype=np.uint8), (40, 40))
        floor = 0 if method == "dilation" else 255
        seed = np.where(rng.random(mask.shape) < 0.05, mask, floor).astype(np.uint8)
        np.testing.assert_array_equal(
            reconstruction(seed, mask, method, connectivity),
            _expected(seed, mask, method, connectivity),
        )

    @pytest.mark.parametrize(
        "dtype", [np.uint16, np.int16, np.float32, np.int8, np.uint32, np.int64]
    )
    def test_other_dtypes(self, method, connectivity, dtype):
        rng = np.random.default_rng(4)
        if dtype is np.float32:
            mask = rng.normal(size=(30, 30)).astype(dtype)
            noise = rng.normal(size=(30, 30)).astype(dtype)
            seed = (
                np.minimum(noise, mask)
                if method == "dilation"
                else np.maximum(noise, mask)
            )
        else:
            seed, mask = _random_pair(rng, (30, 30), method, dtype)
        result = reconstruction(seed, mask, method, connectivity)
        assert result.dtype == dtype
        np.testing.assert_array_equal(
            result, _expected(seed, mask, method, connectivity)
        )

    def test_inputs_are_not_modified(self, method, connectivity):
        rng = np.random.default_rng(5)
        seed, mask = _random_pair(rng, (25, 25), method)
        seed_before, mask_before = seed.copy(), mask.copy()
        reconstruction(seed, mask, method, connectivity)
        np.testing.assert_array_equal(seed, seed_before)
        np.testing.assert_array_equal(mask, mask_before)

    def test_non_contiguous_views_are_accepted(self, method, connectivity):
        rng = np.random.default_rng(6)
        seed, mask = _random_pair(rng, (40, 40), method)
        np.testing.assert_array_equal(
            reconstruction(seed[::2, 1::2], mask[::2, 1::2], method, connectivity),
            _expected(
                seed[::2, 1::2].copy(), mask[::2, 1::2].copy(), method, connectivity
            ),
        )


def test_connectivity_four_does_not_cross_diagonals():
    mask = np.array([[9, 0], [0, 9]], dtype=np.uint8)
    seed = np.array([[9, 0], [0, 0]], dtype=np.uint8)
    assert reconstruction(seed, mask, "dilation", 4)[1, 1] == 0
    assert reconstruction(seed, mask, "dilation", 8)[1, 1] == 9


def test_default_method_is_dilation_with_eight_connectivity():
    mask = np.array([[9, 0], [0, 9]], dtype=np.uint8)
    seed = np.array([[9, 0], [0, 0]], dtype=np.uint8)
    assert reconstruction(seed, mask)[1, 1] == 9


def test_empty_image_gives_empty_result():
    empty = np.zeros((0, 5), dtype=np.uint8)
    assert reconstruction(empty, empty).shape == (0, 5)


class TestRejectsBadArguments:
    def test_seed_above_mask_for_dilation(self):
        with pytest.raises(ValueError, match="exceed"):
            reconstruction(np.full((2, 2), 5, np.uint8), np.zeros((2, 2), np.uint8))

    def test_seed_below_mask_for_erosion(self):
        with pytest.raises(ValueError, match="below"):
            reconstruction(
                np.zeros((2, 2), np.uint8), np.full((2, 2), 5, np.uint8), "erosion"
            )

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            reconstruction(np.zeros((2, 2), np.uint8), np.zeros((2, 3), np.uint8))

    def test_dtype_mismatch(self):
        with pytest.raises(ValueError, match="dtype"):
            reconstruction(np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint16))

    def test_three_dimensional_input(self):
        with pytest.raises(ValueError, match="2-D"):
            reconstruction(np.zeros((2, 2, 2), np.uint8), np.zeros((2, 2, 2), np.uint8))

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="method"):
            reconstruction(
                np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8), "open"
            )  # type: ignore[arg-type]

    def test_unsupported_connectivity(self):
        with pytest.raises(ValueError, match="connectivity"):
            reconstruction(
                np.zeros((2, 2), np.uint8), np.zeros((2, 2), np.uint8), "dilation", 6
            )  # type: ignore[arg-type]


def _serpentine_corridor(side: int) -> np.ndarray:
    """A one-pixel corridor that turns back at every other row end."""
    corridor = np.zeros((side, side), dtype=bool)
    corridor[::2, :] = True
    for row in range(1, side, 2):
        corridor[row, -1 if row % 4 == 1 else 0] = True
    return corridor


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("transposed", [False, True])
def test_winding_corridor_is_followed_to_its_end(connectivity, transposed):
    corridor = _serpentine_corridor(41)
    if transposed:
        corridor = np.ascontiguousarray(corridor.T)
    mask = np.where(corridor, 200, 0).astype(np.uint8)
    seed = np.zeros_like(mask)
    seed[0, 0] = 200
    result = reconstruction(seed, mask, "dilation", connectivity)
    np.testing.assert_array_equal(
        result, _geodesic_dilation_until_stable(seed, mask, connectivity)
    )
    np.testing.assert_array_equal(result == 200, corridor)


def _blocky_pair(rng, shape, block):
    """Mask of random plateaus and a seed with a few raised pixels."""
    coarse = rng.integers(0, 256, (shape[0] // block + 1, shape[1] // block + 1))
    mask = np.kron(coarse, np.ones((block, block), dtype=np.int64))
    mask = mask[: shape[0], : shape[1]].astype(np.uint8)
    seed = np.where(rng.random(shape) < 0.002, mask, 0).astype(np.uint8)
    return seed, mask


@pytest.mark.parametrize("connectivity", [4, 8])
def test_plateau_images_wider_or_taller_than_square_match_the_reference(connectivity):
    rng = np.random.default_rng(7)
    for shape in [(90, 230), (230, 90)]:
        seed, mask = _blocky_pair(rng, shape, 5)
        np.testing.assert_array_equal(
            reconstruction(seed, mask, "dilation", connectivity),
            _geodesic_dilation_until_stable(seed, mask, connectivity),
        )


def test_frame_seed_fills_dark_holes_as_in_picture_detection():
    rng = np.random.default_rng(8)
    rough = rng.integers(0, 256, (70, 110)).astype(np.uint8)
    picture_like = np.asarray(np.maximum(rough, np.roll(rough, 1, axis=1)))
    frame = np.full_like(picture_like, 255)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = 0
    seed = np.maximum(frame, picture_like)
    np.testing.assert_array_equal(
        reconstruction(seed, picture_like, "erosion", 8),
        _expected(seed, picture_like, "erosion", 8),
    )
