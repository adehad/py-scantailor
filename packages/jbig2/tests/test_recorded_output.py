"""Output must stay byte for byte what the encoder produced when the digests were recorded.

The digests in ``fixtures/recorded_digests.json`` come from the first release
of this encoder, whose output the PDF round trips verified. Inputs are built
from NumPy random generators only, so they are the same on every platform.
"""

import hashlib
import json
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from sample_images import checkerboard, fuzz_images, noise

from jbig2 import encode_generic
from jbig2.generic_region import _BAND_PIXELS
from jbig2.mq_coder import MQEncoder, encode_decisions

_DIGESTS = json.loads((Path(__file__).parent / "fixtures" / "recorded_digests.json").read_text())


def _repeated_rows(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.random((height // 7 + 1, width)) < 0.2
    return np.repeat(rows, 7, axis=0)[:height]


IMAGES: dict[str, Callable[[], np.ndarray]] = {
    "noise_5pct_301x517": lambda: noise(301, 517, 0.05, seed=1),
    "noise_50pct_97x203": lambda: noise(97, 203, 0.5, seed=2),
    "noise_95pct_64x64": lambda: noise(64, 64, 0.95, seed=3),
    "checkerboard_1px_120x77": lambda: checkerboard(120, 77, 1),
    "checkerboard_9px_200x333": lambda: checkerboard(200, 333, 9),
    "repeated_rows_400x250": lambda: _repeated_rows(400, 250, seed=4),
    "sparse_specks_600x800": lambda: noise(600, 800, 0.002, seed=5),
    # Taller than one band of rows, so the coder state crosses band edges.
    "sparse_specks_three_bands": lambda: noise(3 * _BAND_PIXELS // 450 + 17, 450, 0.002, seed=6),
    "single_row_70001": lambda: noise(1, 70001, 0.3, seed=7),
    "single_column_5000": lambda: noise(5000, 1, 0.3, seed=8),
    "all_white_50x60": lambda: np.zeros((50, 60), dtype=bool),
    "all_black_50x60": lambda: np.ones((50, 60), dtype=bool),
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decision_streams() -> list[tuple[np.ndarray, np.ndarray, int]]:
    rng = np.random.default_rng(12)
    streams = []
    for _ in range(50):
        count = int(rng.integers(0, 4000))
        context_count = int(rng.integers(1, 30))
        contexts = rng.integers(0, context_count, count)
        if rng.random() < 0.5:
            contexts = np.sort(contexts)
        bits = (rng.random(count) < rng.random()).astype(np.uint8)
        streams.append((contexts, bits, context_count))
    return streams


@pytest.mark.parametrize("tpgdon", [True, False], ids=["tpgdon", "no_tpgdon"])
@pytest.mark.parametrize("name", sorted(IMAGES))
def test_generic_region_matches_recorded_digest(name: str, tpgdon: bool) -> None:
    data = encode_generic(IMAGES[name](), tpgdon=tpgdon)
    assert _digest(data) == _DIGESTS[f"{name}|tpgdon={tpgdon}"]


def test_fuzzed_images_match_recorded_digests() -> None:
    digests = [_digest(encode_generic(image)) for image in fuzz_images(200, seed=11)]
    assert digests == _DIGESTS["fuzz_200_seed_11"]


def test_decision_streams_match_recorded_digests() -> None:
    digests = [_digest(encode_decisions(*stream)) for stream in _decision_streams()]
    assert digests == _DIGESTS["decision_streams_seed_12"]


def test_feeding_decisions_in_pieces_gives_the_same_bytes() -> None:
    for contexts, bits, context_count in _decision_streams()[:10]:
        whole = encode_decisions(contexts, bits, context_count)
        encoder = MQEncoder(context_count)
        cuts = [0, len(bits) // 3, len(bits) // 3 + 1, len(bits)]
        for start, stop in pairwise(cuts):
            encoder.encode(contexts[start:stop], bits[start:stop])
        assert encoder.finish() == whole
