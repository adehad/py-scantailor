"""Makes a missing poppler visible as a skip instead of silently dropping it."""

import shutil

import numpy as np
import pytest
from pdf_decoders import (
    POPPLER_MISSING_REASON,
    EmbeddedImage,
    build_pdf,
    decode_with_both,
    decode_with_poppler,
)

from jbig2_numba import encode_pdf_stream


def test_poppler_decodes_the_encoder_output(tmp_path) -> None:
    if shutil.which("pdfimages") is None:
        pytest.skip(POPPLER_MISSING_REASON)
    image = np.zeros((20, 30), dtype=bool)
    image[5:10, 5:20] = True
    stream = EmbeddedImage(encode_pdf_stream(image), 30, 20)

    decoded = decode_with_both([stream], tmp_path)

    assert np.array_equal(decoded["poppler"][0], image)


def test_decode_with_poppler_skips_when_pdfimages_is_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    pdf = build_pdf([EmbeddedImage(encode_pdf_stream(np.zeros((4, 4), bool)), 4, 4)])

    with pytest.raises(pytest.skip.Exception, match="pdfimages"):
        decode_with_poppler(pdf, tmp_path)


def test_decode_with_both_still_decodes_with_mupdf_when_poppler_is_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    image = np.zeros((8, 8), dtype=bool)
    image[2:4, 2:6] = True

    decoded = decode_with_both([EmbeddedImage(encode_pdf_stream(image), 8, 8)], tmp_path)

    assert list(decoded) == ["mupdf"]
    assert np.array_equal(decoded["mupdf"][0], image)
