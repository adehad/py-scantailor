"""Every encoded image must decode to exactly its input in MuPDF and in poppler.

Standalone ``.jb2`` files are checked with MuPDF only; poppler reads JBIG2
solely inside PDF.
"""

from pathlib import Path

import numpy as np
import pymupdf
import pytest
from pdf_decoders import EmbeddedImage, build_pdf, decode_with_both, decode_with_pymupdf
from sample_images import checkerboard, fuzz_images, noise, scanned_page, text_page

from jbig2 import encode_file, encode_pdf_stream


def _blank_rows_page() -> np.ndarray:
    page = np.zeros((900, 333), dtype=bool)
    page[100:110, 20:300] = True
    page[400:401, :] = True
    page[600:620:2, 50:60] = True
    return page


CASES: dict[str, np.ndarray] = {
    "all_white": np.zeros((37, 50), dtype=bool),
    "all_black": np.ones((37, 50), dtype=bool),
    "one_pixel_white": np.zeros((1, 1), dtype=bool),
    "one_pixel_black": np.ones((1, 1), dtype=bool),
    "single_row": noise(1, 257, 0.5, seed=1),
    "single_column": noise(301, 1, 0.5, seed=2),
    "width_not_multiple_of_8": noise(40, 13, 0.3, seed=3),
    "width_not_multiple_of_32": noise(40, 97, 0.3, seed=4),
    "width_one_past_32": noise(17, 33, 0.5, seed=5),
    "checkerboard_1px": checkerboard(64, 71, 1),
    "checkerboard_3px": checkerboard(50, 50, 3),
    "checkerboard_8px": checkerboard(99, 130, 8),
    "noise_0.01": noise(200, 211, 0.01, seed=6),
    "noise_0.5": noise(200, 211, 0.5, seed=7),
    "noise_0.99": noise(200, 211, 0.99, seed=8),
    "large_blank_rows": _blank_rows_page(),
    "uint8_nonzero_is_black": (noise(30, 45, 0.5, seed=9) * np.uint8(77)).astype(np.uint8),
}


@pytest.mark.parametrize("tpgdon", [True, False], ids=["tpgdon", "no_tpgdon"])
def test_both_decoders_reproduce_every_case_exactly(tmp_path: Path, tpgdon: bool) -> None:
    names = list(CASES)
    streams = [
        EmbeddedImage(
            encode_pdf_stream(CASES[n], tpgdon=tpgdon), CASES[n].shape[1], CASES[n].shape[0]
        )
        for n in names
    ]
    decoded = decode_with_both(streams, tmp_path)
    for decoder, images in decoded.items():
        for name, image in zip(names, images, strict=True):
            np.testing.assert_array_equal(image, CASES[name] != 0, err_msg=f"{decoder}: {name}")


@pytest.mark.parametrize(
    ("name", "make_image"),
    [("text_page_2100x3000", text_page), ("scanned_page", scanned_page)],
)
def test_both_decoders_reproduce_full_pages_exactly(tmp_path: Path, name: str, make_image) -> None:
    image = make_image()
    stream = encode_pdf_stream(image, dpi=(300, 300))
    decoded = decode_with_both([EmbeddedImage(stream, image.shape[1], image.shape[0])], tmp_path)
    for decoder, images in decoded.items():
        np.testing.assert_array_equal(images[0], image, err_msg=f"{decoder}: {name}")


@pytest.mark.parametrize("seed", [2024, 7])
def test_both_decoders_reproduce_200_fuzzed_images(tmp_path: Path, seed: int) -> None:
    images = fuzz_images(200, seed=seed)
    streams = [
        EmbeddedImage(encode_pdf_stream(img, tpgdon=bool(i % 2)), img.shape[1], img.shape[0])
        for i, img in enumerate(images)
    ]
    decoded = decode_with_both(streams, tmp_path)
    for decoder, results in decoded.items():
        for i, (result, expected) in enumerate(zip(results, images, strict=True)):
            np.testing.assert_array_equal(
                result, expected, err_msg=f"{decoder}: image {i} seed {seed}"
            )


def test_dpi_does_not_change_decoded_pixels() -> None:
    image = noise(20, 30, 0.4, seed=10)
    pdf = build_pdf(
        [
            EmbeddedImage(encode_pdf_stream(image, dpi=dpi), 30, 20)
            for dpi in (None, (300, 300), (200, 600))
        ]
    )
    for decoded in decode_with_pymupdf(pdf):
        np.testing.assert_array_equal(decoded, image)


@pytest.mark.parametrize("tpgdon", [True, False], ids=["tpgdon", "no_tpgdon"])
def test_mupdf_reproduces_standalone_jb2_files(tmp_path: Path, tpgdon: bool) -> None:
    for name, image in CASES.items():
        path = tmp_path / f"{name}.jb2"
        path.write_bytes(encode_file(image, dpi=(300, 300), tpgdon=tpgdon))
        pix = pymupdf.Pixmap(str(path))
        rows = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        decoded = rows[:, : pix.width * pix.n : pix.n] < 128
        np.testing.assert_array_equal(decoded, image != 0, err_msg=name)
