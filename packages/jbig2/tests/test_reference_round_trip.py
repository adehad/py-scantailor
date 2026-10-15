"""MQ coder and generic region coding against the pure-Python T.88 decoders."""

import numpy as np
import pytest
from reference_decoder import MQDecoder, decode_generic
from sample_images import checkerboard, fuzz_images, noise

from jbig2 import encode_generic
from jbig2.mq_coder import encode_decisions


@pytest.mark.parametrize("seed", range(5))
def test_mq_decisions_round_trip_across_skewed_contexts(seed: int) -> None:
    rng = np.random.default_rng(seed)
    probabilities = np.array([0.0, 0.001, 0.2, 0.5, 0.9, 1.0])
    contexts = rng.integers(0, probabilities.size, 20_000)
    bits = (rng.random(contexts.size) < probabilities[contexts]).astype(np.uint8)

    data = encode_decisions(contexts, bits, probabilities.size)

    decoder = MQDecoder(data, probabilities.size)
    decoded = [decoder.decode(int(cx)) for cx in contexts]
    np.testing.assert_array_equal(decoded, bits)


def test_mq_output_ends_with_marker_and_never_contains_one_early() -> None:
    rng = np.random.default_rng(3)
    data = encode_decisions(np.zeros(50_000, dtype=np.int64), rng.integers(0, 2, 50_000), 1)

    assert data.endswith(b"\xff\xac")
    body = np.frombuffer(data[:-2], dtype=np.uint8)
    after_ff = body[1:][body[:-1] == 0xFF]
    assert (after_ff <= 0x8F).all(), "0xFF followed by a byte above 0x8F is a marker"


def test_mq_long_runs_of_one_symbol_compress_to_a_few_bytes() -> None:
    data = encode_decisions(np.zeros(100_000, dtype=np.int64), np.zeros(100_000, dtype=np.uint8), 1)

    assert len(data) < 20


def test_mq_rejects_bad_decisions() -> None:
    with pytest.raises(ValueError, match="same length"):
        encode_decisions(np.zeros(3), np.zeros(2), 1)
    with pytest.raises(ValueError, match="context out of range"):
        encode_decisions(np.array([2]), np.array([0]), 2)
    with pytest.raises(ValueError, match="0 or 1"):
        encode_decisions(np.array([0]), np.array([2]), 1)


@pytest.mark.parametrize("tpgdon", [True, False])
@pytest.mark.parametrize(
    "image",
    [
        np.zeros((1, 1), dtype=bool),
        np.ones((1, 1), dtype=bool),
        np.ones((5, 9), dtype=bool),
        checkerboard(12, 17, 1),
        noise(1, 70, 0.5, seed=0),
        noise(70, 1, 0.5, seed=1),
        noise(33, 41, 0.05, seed=2),
    ],
    ids=["white_1x1", "black_1x1", "black", "checker", "row", "column", "sparse"],
)
def test_generic_region_round_trips(image: np.ndarray, tpgdon: bool) -> None:
    data = encode_generic(image, tpgdon=tpgdon)

    decoded = decode_generic(data, image.shape[1], image.shape[0], tpgdon=tpgdon)

    np.testing.assert_array_equal(decoded, image)


def test_generic_region_round_trips_fuzzed_images() -> None:
    for i, image in enumerate(fuzz_images(60, seed=99)):
        tpgdon = bool(i % 2)
        decoded = decode_generic(
            encode_generic(image, tpgdon=tpgdon), image.shape[1], image.shape[0], tpgdon=tpgdon
        )
        np.testing.assert_array_equal(decoded, image, err_msg=f"image {i}")


def test_typical_prediction_shrinks_pages_of_repeated_rows() -> None:
    image = np.zeros((2000, 800), dtype=bool)
    image[1000:1004, 100:700] = True

    assert len(encode_generic(image, tpgdon=True)) < len(encode_generic(image, tpgdon=False))
