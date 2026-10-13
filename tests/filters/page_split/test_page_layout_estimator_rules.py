"""Folding-line rules and the 150 dpi clean-up, against the C++ transcription.

The generated scans of the parity matrix reach only some of these branches, so
the rules are driven here with random lines and random binary pages.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from scantailor.filters.page_split import page_layout_estimator as estimator
from scantailor.filters.page_split.layout_type import LayoutType

from . import cpp_reference as cpp

UPSTREAM_TYPES = {
    LayoutType.AUTO_LAYOUT_TYPE: cpp.AUTO,
    LayoutType.PAGE_PLUS_OFFCUT: cpp.PLUS_OFFCUT,
    LayoutType.TWO_PAGES: cpp.TWO_PAGES,
}


def _mid_x(line) -> float:
    return 0.5 * (line[0][0] + line[1][0])


def _random_lines(seed: int, width: int, height: int):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(0, 5))
    lines = []
    for _ in range(count):
        x = float(rng.uniform(0, width))
        tilt = float(rng.uniform(-6, 6))
        lines.append(((x - tilt, 0.0), (x + tilt, float(height))))
    return lines


@pytest.mark.parametrize("seed", range(150))
@pytest.mark.parametrize("requested", list(UPSTREAM_TYPES), ids=lambda t: t.value)
@pytest.mark.parametrize("shape", [(500, 800), (800, 500)], ids=["wide", "tall"])
def test_folding_line_rules_match_upstream(
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    requested: LayoutType,
    shape: tuple[int, int],
) -> None:
    height, width = shape
    lines = _random_lines(seed, width, height)
    monkeypatch.setattr(estimator, "find_lines", lambda *_args: list(lines))
    monkeypatch.setattr(cpp, "vert_line_finder", lambda *_args: list(lines))
    gray = np.zeros(shape, dtype=np.uint8)

    got = estimator._try_cut_at_folding_line(requested, gray, 300.0)
    expected = cpp.try_cut_at_folding_line(UPSTREAM_TYPES[requested], gray, 300.0)

    if expected is None:
        assert got is None
        return
    kind, line1, line2 = expected
    assert got is not None
    assert got.layout_type.value == kind
    if kind == cpp.UNCUT:
        return
    assert _mid_x(got.lines[0]) == pytest.approx(_mid_x(line1))
    if kind == cpp.CUT:
        assert _mid_x(got.lines[1]) == pytest.approx(_mid_x(line2))


def _random_black_page(seed: int) -> NDArray[np.bool_]:
    """Rectangles of mixed sizes, thin rules and wide or tall shadow bars."""
    rng = np.random.default_rng(seed)
    img = np.zeros((660, 840), dtype=np.bool_)
    for _ in range(int(rng.integers(5, 40))):
        w = int(rng.integers(1, 40))
        h = int(rng.integers(1, 40))
        x = int(rng.integers(0, 840 - w))
        y = int(rng.integers(0, 660 - h))
        img[y : y + h, x : x + w] = True
    for _ in range(int(rng.integers(0, 4))):
        x = int(rng.integers(0, 800))
        y = int(rng.integers(0, 580))
        img[y : y + int(rng.integers(30, 320)), x : x + int(rng.integers(1, 5))] = True
    if seed % 3 == 0:
        edge = int(rng.integers(0, 3))
        img[:, edge : edge + int(rng.integers(10, 20))] = True
    if seed % 4 == 1:
        # Around the 200 x 14 limit of a shadow, once halved to 150 dpi.
        x, y = int(rng.integers(0, 400)), int(rng.integers(0, 600))
        img[y : y + 32, x : x + int(rng.integers(300, 440))] = True
    if seed % 4 == 2:
        x, y = int(rng.integers(0, 800)), int(rng.integers(0, 60))
        img[y : y + int(rng.integers(540, 640)), x : x + 32] = True
    if seed % 5 == 0:
        top = int(rng.integers(0, 20))
        img[top : top + int(rng.integers(10, 18)), :] = True
    return img


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(120))
def test_garbage_and_shadow_removal_matches_upstream(seed: int) -> None:
    page = _random_black_page(seed)

    got = estimator._remove_garbage_and_halve(page)

    np.testing.assert_array_equal(got, cpp.remove_garbage_and_2x_downscale(page))


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(120))
@pytest.mark.parametrize("pages", [1, 2])
@pytest.mark.parametrize("requested", list(UPSTREAM_TYPES), ids=lambda t: t.value)
def test_content_spans_from_components_match_upstream(
    seed: int, pages: int, requested: LayoutType
) -> None:
    img = _random_black_page(seed + 1000)
    left = estimator._has_left_offcut(img)
    right = estimator._has_right_offcut(img)
    assert left == cpp.check_for_left_offcut(img)
    assert right == cpp.check_for_right_offcut(img)

    got = estimator._cut_at_whitespace_deskewed_150(requested, pages, img, left, right)
    kind, line1, line2 = cpp.cut_at_whitespace_deskewed_150(
        UPSTREAM_TYPES[requested], pages, img, left, right
    )

    assert got.layout_type.value == kind
    if line1 is not None:
        assert _mid_x(got.lines[0]) == pytest.approx(_mid_x(line1))
    if kind == cpp.CUT:
        assert _mid_x(got.lines[1]) == pytest.approx(_mid_x(line2))
