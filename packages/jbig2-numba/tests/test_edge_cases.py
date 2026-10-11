"""Adversarial inputs: extreme shapes, memory layouts, dtypes, byte stuffing, threads, cache."""

import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from pdf_decoders import EmbeddedImage, decode_with_both
from reference_decoder import MQDecoder, decode_generic
from sample_images import noise

from jbig2_numba import encode_file, encode_generic, encode_pdf_stream
from jbig2_numba.mq_coder import encode_decisions

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"


def _assert_both_decoders(images: list[np.ndarray], workdir: Path, *, tpgdon: bool = True) -> None:
    streams = [
        EmbeddedImage(encode_pdf_stream(img, tpgdon=tpgdon), img.shape[1], img.shape[0])
        for img in images
    ]
    for decoder, results in decode_with_both(streams, workdir).items():
        for i, (result, image) in enumerate(zip(results, images, strict=True)):
            np.testing.assert_array_equal(result, image != 0, err_msg=f"{decoder}: image {i}")


@pytest.mark.parametrize("tpgdon", [True, False], ids=["tpgdon", "no_tpgdon"])
def test_very_wide_and_tall_images_round_trip(tmp_path: Path, tpgdon: bool) -> None:
    images = [
        noise(1, 65535, 0.3, seed=1),
        noise(3, 65535, 0.5, seed=2),
        noise(2, 65536, 0.2, seed=3),
        noise(70000, 1, 0.5, seed=4),
        np.ones((1, 70001), dtype=bool),
    ]
    _assert_both_decoders(images, tmp_path, tpgdon=tpgdon)


def _tpgdon_pattern_images() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    width = 61
    alternating = np.zeros((40, width), dtype=bool)
    alternating[1::2] = rng.random(width) < 0.5
    runs = []
    while sum(len(r) for r in runs) < 200:
        row = rng.random(width) < 0.4
        runs.append(np.repeat(row[None], rng.integers(1, 9), axis=0))
    first_row_black = np.zeros((10, width), dtype=bool)
    first_row_black[0] = True
    skip_then_not = np.zeros((300, width), dtype=bool)
    skip_then_not[250:] = rng.random((50, width)) < 0.5
    noise_then_blank = np.vstack([rng.random((20, width)) < 0.5, np.zeros((50, width), bool)])
    return {
        "alternating": alternating,
        "irregular_runs": np.concatenate(runs),
        "first_row_black_rest_white": first_row_black,
        "skip_300_then_noise": skip_then_not,
        "noise_then_blank": noise_then_blank,
        "every_row_identical_black": np.ones((64, width), dtype=bool),
    }


@pytest.mark.parametrize("tpgdon", [True, False], ids=["tpgdon", "no_tpgdon"])
def test_typical_prediction_patterns_round_trip(tmp_path: Path, tpgdon: bool) -> None:
    _assert_both_decoders(list(_tpgdon_pattern_images().values()), tmp_path, tpgdon=tpgdon)


def test_typical_prediction_patterns_match_reference_decoder() -> None:
    for name, image in _tpgdon_pattern_images().items():
        for tpgdon in (True, False):
            data = encode_generic(image, tpgdon=tpgdon)
            decoded = decode_generic(data, image.shape[1], image.shape[0], tpgdon=tpgdon)
            np.testing.assert_array_equal(decoded, image, err_msg=f"{name} tpgdon={tpgdon}")


def test_blank_page_with_typical_prediction_is_tiny() -> None:
    assert len(encode_generic(np.zeros((3000, 2100), dtype=bool), tpgdon=True)) < 64


@pytest.mark.parametrize("value", [2, 3, 127, 128, 254, 255])
def test_uint8_values_other_than_zero_and_one_are_black(tmp_path: Path, value: int) -> None:
    mask = noise(25, 37, 0.5, seed=value)
    image = mask.astype(np.uint8) * np.uint8(value)
    assert encode_pdf_stream(image) == encode_pdf_stream(mask)
    _assert_both_decoders([image], tmp_path)


def test_uint8_with_many_distinct_nonzero_values_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(40, 50), dtype=np.uint8)
    image[rng.random(image.shape) < 0.5] = 0
    _assert_both_decoders([image], tmp_path)


def _read_only(array: np.ndarray) -> np.ndarray:
    copy = array.copy()
    copy.flags.writeable = False
    return copy


def _layouts() -> dict[str, np.ndarray]:
    base = noise(41, 67, 0.4, seed=21)
    big = noise(82, 134, 0.4, seed=22)
    return {
        "c_contiguous": base.copy(),
        "fortran": np.asfortranarray(base),
        "transposed_view": noise(67, 41, 0.4, seed=23).T,
        "row_stride_2": big[::2, :67],
        "column_stride_2": big[:41, ::2],
        "both_strides": big[::2, ::2],
        "negative_row_stride": base[::-1],
        "negative_column_stride": base[:, ::-1],
        "negative_both": base[::-1, ::-1],
        "uint8_fortran": np.asfortranarray(base.astype(np.uint8) * np.uint8(9)),
        "uint8_strided": (big.astype(np.uint8) * np.uint8(255))[::2, ::2],
        "read_only": _read_only(base),
        "single_row_view": big[5:6, :],
        "column_view": big[:, 7:8],
    }


@pytest.mark.parametrize("name", list(_layouts()))
def test_memory_layout_does_not_change_output(name: str) -> None:
    view = _layouts()[name]
    contiguous = np.ascontiguousarray(view)
    for encode in (encode_generic, encode_pdf_stream, encode_file):
        assert encode(view) == encode(contiguous), f"{encode.__name__}: {name}"


def test_memory_layouts_decode_correctly(tmp_path: Path) -> None:
    _assert_both_decoders(list(_layouts().values()), tmp_path)


def test_views_and_read_only_inputs_are_not_modified() -> None:
    for name, view in _layouts().items():
        before = view.copy()
        encode_file(view)
        np.testing.assert_array_equal(view, before, err_msg=name)


def test_bool_and_equivalent_uint8_give_identical_streams() -> None:
    mask = noise(30, 31, 0.5, seed=31)
    assert encode_file(mask) == encode_file(mask.astype(np.uint8))


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint16, np.float64])
def test_other_dtypes_are_rejected(dtype) -> None:
    with pytest.raises(TypeError):
        encode_generic(np.zeros((3, 3), dtype=dtype))


def test_ndarray_subclasses_are_encoded_like_plain_arrays() -> None:
    mask = noise(10, 11, 0.5, seed=32)
    assert encode_generic(np.ma.masked_array(mask)) == encode_generic(mask)


@pytest.mark.parametrize("bad", [(0, 300), (300, 0), (-1, -1)])
def test_non_positive_dpi_is_rejected(bad: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="dpi"):
        encode_pdf_stream(noise(4, 4, 0.5, seed=33), dpi=bad)


def test_extreme_dpi_values_are_accepted() -> None:
    image = noise(4, 4, 0.5, seed=34)
    assert encode_pdf_stream(image, dpi=(1, 1))
    assert encode_pdf_stream(image, dpi=(100000, 100000))


def _ff_pairs(data: bytes) -> int:
    return sum(1 for a, _ in pairwise(data) if a == 0xFF)


def test_mq_coder_stuffed_bytes_match_reference_decoder() -> None:
    """Search skewed and uniform streams so 0xFF bytes and carries occur."""
    stuffed_total = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        count = 4000
        contexts = rng.integers(0, 1 + seed % 5, size=count)
        probability = (0.5, 0.9, 0.02, 0.98, 0.5)[seed % 5]
        bits = (rng.random(count) < probability).astype(np.uint8)
        data = encode_decisions(contexts, bits, 8)
        assert data.endswith(b"\xff\xac")
        body = data[:-2]
        assert all(b2 <= 0x8F for b1, b2 in pairwise(body) if b1 == 0xFF)
        stuffed_total += _ff_pairs(body)
        decoder = MQDecoder(data, 8)
        decoded = [decoder.decode(int(cx)) for cx in contexts]
        assert decoded == bits.tolist(), f"seed {seed}"
    assert stuffed_total > 0, "search never produced a 0xFF byte"


def test_images_that_emit_ff_bytes_round_trip(tmp_path: Path) -> None:
    candidates = []
    for seed in range(400):
        rng = np.random.default_rng(1000 + seed)
        density = float(rng.choice([0.5, 0.45, 0.55, 0.3]))
        candidates.append(noise(int(rng.integers(8, 40)), int(rng.integers(8, 90)), density, seed))
    with_ff = [img for img in candidates if _ff_pairs(encode_generic(img)[:-2]) > 0]
    assert len(with_ff) >= 10, "too few images with 0xFF in the body"
    _assert_both_decoders(with_ff[:80], tmp_path)


def test_no_early_marker_in_noise_streams() -> None:
    for seed in range(60):
        image = noise(60, 80, 0.5, seed=seed)
        data = encode_generic(image)
        assert b"\xff\xac" not in data[:-2]
        np.testing.assert_array_equal(decode_generic(data, 80, 60, tpgdon=True), image)


def test_incompressible_data_stays_inside_output_buffer(tmp_path: Path) -> None:
    rng = np.random.default_rng(77)
    images = [
        rng.random((64, 256)) < 0.5,
        np.indices((64, 256)).sum(axis=0) % 2 == 0,
        (rng.random((64, 256)) < 0.5) ^ (np.arange(256) % 3 == 0),
    ]
    _assert_both_decoders(images, tmp_path)


def test_concurrent_encoding_matches_serial_output() -> None:
    images = [noise(60 + i, 70 + 3 * i, 0.1 + 0.05 * (i % 10), seed=i) for i in range(64)]
    serial = [encode_file(img, dpi=(300, 300)) for img in images]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in range(3):
            assert list(pool.map(lambda img: encode_file(img, dpi=(300, 300)), images)) == serial


def test_repeated_calls_are_stateless() -> None:
    a = noise(50, 50, 0.5, seed=41)
    b = noise(50, 50, 0.2, seed=42)
    first_a = encode_generic(a)
    encode_generic(b)
    assert encode_generic(a) == first_a


_CACHE_PROBE = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    from jbig2_numba import encode_generic, generic_region
    data = encode_generic(np.random.default_rng(0).random((50, 60)) < 0.3)
    stats = generic_region._encode_template0.stats
    print(json.dumps({
        "misses": sum(stats.cache_misses.values()),
        "hits": sum(stats.cache_hits.values()),
        "digest": data.hex(),
    }))
    """
)


def test_numba_cache_is_reused_from_a_fresh_process(tmp_path: Path) -> None:
    cache_dir = tmp_path / "numba_cache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}
    env.pop("NUMBA_DISABLE_JIT", None)
    runs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-I", "-c", _CACHE_PROBE, str(PACKAGE_SRC)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        runs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert runs[0]["misses"] >= 1
    assert runs[1]["misses"] == 0, "second process recompiled instead of loading the cache"
    assert runs[1]["hits"] >= 1
    assert runs[0]["digest"] == runs[1]["digest"]
    assert any(cache_dir.rglob("*.nbi"))
