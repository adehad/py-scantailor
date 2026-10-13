"""Gray level counts are exact whatever the size or memory layout of the image."""

import numpy as np
import pytest

from scantailor.imageproc.filters import gray_histogram

_LEVELS = 256


def _expected(image: np.ndarray) -> np.ndarray:
    return np.bincount(image.ravel(), minlength=_LEVELS).astype(np.int64)


class TestGrayHistogram:
    """Counts equal ``numpy.bincount`` where float32 counting would round."""

    def test_counts_past_the_float32_limit_stay_exact(self) -> None:
        """A page of one level has more pixels than float32 holds exactly."""
        side = 5001
        image = np.full((side, side), 235, dtype=np.uint8)
        image[0, 0] = 7

        counts = gray_histogram(image)

        assert counts[235] == side * side - 1
        assert counts[7] == 1
        assert counts.sum() == image.size

    @pytest.mark.parametrize("shape", [(1, 1), (1, 300), (300, 1), (37, 91)])
    def test_matches_bincount_on_small_and_thin_images(
        self, shape: tuple[int, int]
    ) -> None:
        image = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)

        assert np.array_equal(gray_histogram(image), _expected(image))

    def test_a_strided_view_is_counted_like_its_copy(self) -> None:
        image = np.random.default_rng(2).integers(0, 256, (50, 60, 3), dtype=np.uint8)
        channel = image[..., 1]

        assert not channel.flags.c_contiguous
        assert np.array_equal(gray_histogram(channel), _expected(channel.copy()))

    def test_an_empty_image_has_no_counts(self) -> None:
        counts = gray_histogram(np.empty((0, 5), dtype=np.uint8))

        assert counts.shape == (_LEVELS,)
        assert not counts.any()

    def test_other_integer_types_are_counted_too(self) -> None:
        image = np.array([[0, 1, 1], [255, 255, 255]], dtype=np.int32)

        counts = gray_histogram(image)  # type: ignore[arg-type]

        assert counts[0] == 1
        assert counts[1] == 2
        assert counts[255] == 3
