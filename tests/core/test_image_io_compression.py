"""TIFF compression choices in ``write_image``."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from scantailor.core.image_io import tiff_compressions_in_use, write_image
from scantailor.core.settings import (
    ApplicationSettings,
    TiffCompression,
    TiffCompressions,
)

COMPRESSION_TAG = 259


def _tag(path: Path) -> int:
    with Image.open(path) as image:
        return image.tag_v2[COMPRESSION_TAG]


def _bilevel() -> NDArray[np.bool_]:
    mask = np.zeros((32, 48), dtype=bool)
    mask[8:24, 10:30] = True
    return mask


def _grey() -> NDArray[np.uint8]:
    return np.tile(np.arange(48, dtype=np.uint8), (32, 1))


def _colour() -> NDArray[np.uint8]:
    return np.dstack([_grey(), _grey()[:, ::-1], _grey()]).astype(np.uint8)


class TestChoiceFollowsImageKind:
    @pytest.mark.parametrize(
        "method",
        [
            TiffCompression.NONE,
            TiffCompression.LZW,
            TiffCompression.DEFLATE,
            TiffCompression.CCITT_FAX4,
        ],
    )
    def test_bilevel_image_uses_the_black_and_white_method(
        self, tmp_path: Path, method: TiffCompression
    ) -> None:
        path = tmp_path / "bw.tif"

        write_image(path, _bilevel(), compressions=TiffCompressions(bw=method))

        assert _tag(path) == method.value

    @pytest.mark.parametrize(
        "method",
        [
            TiffCompression.NONE,
            TiffCompression.LZW,
            TiffCompression.DEFLATE,
            TiffCompression.JPEG,
        ],
    )
    @pytest.mark.parametrize("pixels", [_grey, _colour], ids=["grey", "colour"])
    def test_grey_and_colour_images_use_the_colour_method(
        self,
        tmp_path: Path,
        method: TiffCompression,
        pixels: Callable[[], NDArray[np.uint8]],
    ) -> None:
        path = tmp_path / "c.tif"

        write_image(path, pixels(), compressions=TiffCompressions(color=method))

        assert _tag(path) == method.value

    def test_the_other_kind_keeps_its_own_method(self, tmp_path: Path) -> None:
        path = tmp_path / "bw.tif"
        choice = TiffCompressions(
            bw=TiffCompression.DEFLATE, color=TiffCompression.NONE
        )

        write_image(path, _bilevel(), compressions=choice)

        assert _tag(path) == TiffCompression.DEFLATE.value

    def test_default_choice_is_g4_for_bilevel_and_lzw_for_colour(
        self, tmp_path: Path
    ) -> None:
        write_image(tmp_path / "a.tif", _bilevel())
        write_image(tmp_path / "b.tif", _colour())

        assert _tag(tmp_path / "a.tif") == TiffCompression.CCITT_FAX4.value
        assert _tag(tmp_path / "b.tif") == TiffCompression.LZW.value

    def test_lossless_choices_keep_the_pixels(self, tmp_path: Path) -> None:
        path = tmp_path / "g.tif"

        write_image(
            path, _grey(), compressions=TiffCompressions(color=TiffCompression.DEFLATE)
        )

        with Image.open(path) as image:
            assert np.array_equal(np.asarray(image), _grey())

    def test_png_ignores_the_choice(self, tmp_path: Path) -> None:
        path = tmp_path / "g.png"

        write_image(
            path, _grey(), compressions=TiffCompressions(color=TiffCompression.JPEG)
        )

        with Image.open(path) as image:
            assert np.array_equal(np.asarray(image), _grey())


class TestChoiceInEffect:
    def test_block_sets_the_default_and_restores_it(self, tmp_path: Path) -> None:
        with tiff_compressions_in_use(TiffCompressions(bw=TiffCompression.NONE)):
            write_image(tmp_path / "inside.tif", _bilevel())
        write_image(tmp_path / "after.tif", _bilevel())

        assert _tag(tmp_path / "inside.tif") == TiffCompression.NONE.value
        assert _tag(tmp_path / "after.tif") == TiffCompression.CCITT_FAX4.value

    def test_explicit_argument_wins_over_the_block(self, tmp_path: Path) -> None:
        with tiff_compressions_in_use(TiffCompressions(bw=TiffCompression.NONE)):
            write_image(
                tmp_path / "x.tif",
                _bilevel(),
                compressions=TiffCompressions(bw=TiffCompression.LZW),
            )

        assert _tag(tmp_path / "x.tif") == TiffCompression.LZW.value


class TestChoiceValidation:
    def test_jpeg_cannot_be_the_black_and_white_method(self) -> None:
        with pytest.raises(ValueError, match="bilevel"):
            TiffCompressions(bw=TiffCompression.JPEG)

    def test_g4_cannot_be_the_colour_method(self) -> None:
        with pytest.raises(ValueError, match="colour"):
            TiffCompressions(color=TiffCompression.CCITT_FAX4)

    def test_settings_reject_a_method_the_image_kind_cannot_use(self) -> None:
        with pytest.raises(ValueError, match="bilevel"):
            ApplicationSettings(tiff_bw_compression=TiffCompression.JPEG)
        with pytest.raises(ValueError, match="grey or colour"):
            ApplicationSettings(tiff_color_compression=TiffCompression.CCITT_FAX4)

    def test_settings_expose_the_choice_as_plain_data(self) -> None:
        settings = ApplicationSettings(
            tiff_bw_compression=TiffCompression.DEFLATE,
            tiff_color_compression=TiffCompression.JPEG,
        )

        assert settings.tiff_compressions == TiffCompressions(
            bw=TiffCompression.DEFLATE, color=TiffCompression.JPEG
        )
