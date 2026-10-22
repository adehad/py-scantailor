"""``load_image`` scales 32-bit float TIFF samples to 8 bits."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from scantailor.core.image_io import load_image
from scantailor.core.models import ImageId


def _float_tiff(tmp_path: Path, samples: NDArray[np.float32]) -> ImageId:
    path = tmp_path / "float.tif"
    Image.fromarray(samples.astype(np.float32), mode="F").save(path)
    with Image.open(path) as image:
        assert image.mode == "F"
    return ImageId(file_path=path)


def _ramp(low: float, high: float) -> NDArray[np.float32]:
    return np.tile(np.linspace(low, high, 256, dtype=np.float32), (8, 1))


def test_samples_from_0_to_1_map_to_black_through_white(tmp_path: Path) -> None:
    loaded, _ = load_image(_float_tiff(tmp_path, _ramp(0.0, 1.0)))

    assert loaded.dtype == np.uint8
    assert loaded.shape == (8, 256)
    assert np.array_equal(loaded[0], np.arange(256))


def test_a_page_within_0_to_1_keeps_its_tone_rather_than_stretching(
    tmp_path: Path,
) -> None:
    loaded, _ = load_image(_float_tiff(tmp_path, np.full((4, 4), 0.5, np.float32)))

    assert np.all(loaded == 128)


def test_samples_outside_0_to_1_are_stretched_from_lowest_to_highest(
    tmp_path: Path,
) -> None:
    loaded, _ = load_image(_float_tiff(tmp_path, _ramp(-100.0, 155.0)))

    assert loaded[0, 0] == 0
    assert loaded[0, -1] == 255
    assert np.all(np.diff(loaded[0].astype(int)) >= 0)


def test_nan_counts_as_the_lowest_sample_and_infinity_as_the_highest(
    tmp_path: Path,
) -> None:
    samples = np.array([[np.nan, 0.25, np.inf, -np.inf, 0.75]], dtype=np.float32)

    loaded, _ = load_image(_float_tiff(tmp_path, samples))

    assert loaded.tolist() == [[0, 64, 255, 0, 191]]


def test_a_flat_page_outside_0_to_1_loads_without_dividing_by_zero(
    tmp_path: Path,
) -> None:
    loaded, _ = load_image(_float_tiff(tmp_path, np.full((4, 4), 40.0, np.float32)))

    assert loaded.shape == (4, 4)
    assert loaded.dtype == np.uint8
    assert np.all(loaded == 0)


def test_a_page_with_no_finite_sample_loads_black(tmp_path: Path) -> None:
    samples = np.full((4, 4), np.nan, dtype=np.float32)

    loaded, _ = load_image(_float_tiff(tmp_path, samples))

    assert np.all(loaded == 0)


def test_samples_near_the_float32_limits_stretch_without_overflow(
    tmp_path: Path,
) -> None:
    samples = np.array([[-3.0e38, 0.0, 3.0e38]], dtype=np.float32)

    loaded, _ = load_image(_float_tiff(tmp_path, samples))

    assert loaded.tolist() == [[0, 128, 255]]
