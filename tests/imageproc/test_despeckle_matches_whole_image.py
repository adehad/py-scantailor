"""Despeckling in bands matches evaluating the whole image at once."""

import time

import cv2
import numpy as np
import pytest

from scantailor.imageproc import despeckle as module
from scantailor.imageproc.despeckle import despeckle

_SCALE = module._VERTICAL_SCALE


def pairs_over_whole_image(nodes, sources, fixed=None):
    stretched = np.repeat(np.where(sources, 0, 255).astype(np.uint8), _SCALE, 0)
    _, label_image = cv2.distanceTransformWithLabels(
        stretched, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    cells = np.asarray(label_image, dtype=np.int32)
    source_y, source_x = np.nonzero(sources)
    stretched_y = (source_y * _SCALE)[:, None] + np.arange(_SCALE)
    cell_ids = cells[stretched_y, source_x[:, None]].reshape(-1)
    table_x = np.zeros(cell_ids.max() + 1, dtype=np.int32)
    table_y = np.zeros_like(table_x)
    table_node = np.full_like(table_x, -1)
    table_x[cell_ids] = np.repeat(source_x, _SCALE)
    table_y[cell_ids] = np.repeat(source_y, _SCALE)
    table_node[cell_ids] = np.repeat(nodes[source_y, source_x], _SCALE)

    cell_node = table_node[cells]
    if fixed is not None:
        fixed_stretched = np.repeat(fixed, _SCALE, 0)
        cell_node[fixed_stretched] = np.repeat(nodes, _SCALE, 0)[fixed_stretched]

    def source_points(rows, cols):
        cell = cells[rows, cols]
        x = table_x[cell].astype(np.int64)
        y = table_y[cell].astype(np.int64)
        if fixed is not None:
            own = fixed[rows // _SCALE, cols]
            x = np.where(own, cols, x)
            y = np.where(own, rows // _SCALE, y)
        return x, y

    parts_a, parts_b, parts_d = [], [], []
    comparisons = (
        (cell_node[::_SCALE, :-1], cell_node[::_SCALE, 1:], (_SCALE, 1), (0, 1)),
        (
            cell_node[:-_SCALE:_SCALE],
            cell_node[_SCALE::_SCALE],
            (_SCALE, 1),
            (_SCALE, 0),
        ),
    )
    for first, second, step, offset in comparisons:
        i, j = np.nonzero(first != second)
        row_a, col_a = i * step[0], j * step[1]
        row_b, col_b = row_a + offset[0], col_a + offset[1]
        x_a, y_a = source_points(row_a, col_a)
        x_b, y_b = source_points(row_b, col_b)
        parts_a.append(first[i, j].astype(np.int64))
        parts_b.append(second[i, j].astype(np.int64))
        parts_d.append((x_a - x_b) ** 2 + (y_a - y_b) ** 2)
    node_a = np.concatenate(parts_a)
    node_b = np.concatenate(parts_b)
    sq_dist = np.concatenate(parts_d)
    valid = (node_a >= 0) & (node_b >= 0)
    return module._min_per_pair(
        np.minimum(node_a, node_b)[valid],
        np.maximum(node_a, node_b)[valid],
        sq_dist[valid],
        int(nodes.max()) + 1,
    )


def despeckle_over_whole_image(binary, dpi, level):
    if level <= 0:
        return binary.copy()
    thresholds = module._Thresholds(level, dpi if dpi > 0 else 300)
    ink = binary == 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.view(np.uint8), connectivity=8
    )
    if count == 1:
        return binary.copy()
    width = stats[:, cv2.CC_STAT_WIDTH]
    height = stats[:, cv2.CC_STAT_HEIGHT]
    is_big = (width >= thresholds.big_object_size) | (
        height >= thresholds.big_object_size
    )
    is_big[0] = False
    small_labels = np.flatnonzero(~is_big)[1:]
    node_of_label = np.zeros(count, dtype=np.int32)
    node_of_label[small_labels] = np.arange(1, small_labels.size + 1, dtype=np.int32)
    node_of_label[0] = -1
    if not is_big.any():
        keep = np.zeros(small_labels.size + 1, dtype=bool)
        return module._remove_unkept(binary, node_of_label, labels, keep)
    nodes = node_of_label[labels]
    node_pixels = np.empty(small_labels.size + 1, dtype=np.float64)
    node_pixels[0] = binary.size
    node_pixels[1:] = stats[small_labels, cv2.CC_STAT_AREA]
    lo, hi, sq_dist = pairs_over_whole_image(nodes, ink)
    near_small, near_big = module._anchor_tags(
        lo, hi, sq_dist, node_pixels, thresholds, node_pixels.size
    )
    only_small = near_small & ~near_big
    only_small[0] = False
    if only_small.any():
        second = pairs_over_whole_image(
            nodes, ink & only_small[nodes], fixed=ink & ~only_small[nodes]
        )
        lo, hi, sq_dist = module._min_per_pair(
            np.concatenate([lo, second[0]]),
            np.concatenate([hi, second[1]]),
            np.concatenate([sq_dist, second[2]]),
            node_pixels.size,
        )
    keep = module._reachable_from_big(lo, hi, sq_dist, node_pixels, thresholds)
    return module._remove_unkept(binary, node_of_label, labels, keep)


def _text_like_page(seed: int, dpi: int, height: int, width: int) -> np.ndarray:
    """Words, specks and small blobs scattered over a page, some near its edges."""
    rng = np.random.default_rng(seed)
    page = np.full((height, width), 255, dtype=np.uint8)
    for _ in range(int(rng.integers(5, 60))):
        top = int(rng.integers(0, height - 40))
        left = int(rng.integers(0, width - 80))
        scale = float(rng.uniform(0.5, 1.5)) * dpi / 300
        cv2.putText(
            page, "wordx", (left, top + 30), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 1
        )
    specks = int(rng.integers(0, 400))
    page[rng.integers(0, height, specks), rng.integers(0, width, specks)] = 0
    for _ in range(int(rng.integers(0, 40))):
        top = int(rng.integers(0, height - 6))
        left = int(rng.integers(0, width - 6))
        size = int(rng.integers(1, 5))
        page[top : top + size, left : left + size] = 0
    return page


@pytest.fixture(params=[1, 4, 8], ids=lambda n: f"{n}-threads")
def thread_count(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> int:
    monkeypatch.setattr(cv2, "getNumThreads", lambda: request.param)
    return request.param


class TestMatchesWholeImageEvaluation:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("dpi", [150, 300, 600])
    @pytest.mark.parametrize("level", [1.0, 2.0, 3.0])
    def test_scattered_page_gives_identical_pixels(
        self, thread_count: int, seed: int, dpi: int, level: float
    ) -> None:
        page = _text_like_page(seed, dpi, 1100 + 37 * seed, 600 + 53 * seed)

        banded = despeckle(page, dpi, level)

        assert np.array_equal(banded, despeckle_over_whole_image(page, dpi, level))

    def test_small_cluster_in_a_large_empty_page_gives_identical_pixels(
        self, thread_count: int
    ) -> None:
        page = np.full((1500, 900), 255, dtype=np.uint8)
        page[700:740, 400:520] = 0
        page[690:694, 410:414] = 0
        page[760:762, 600:602] = 0
        page[20:22, 20:22] = 0

        assert np.array_equal(
            despeckle(page, 300, 2.0), despeckle_over_whole_image(page, 300, 2.0)
        )

    def test_ink_touching_every_page_edge_gives_identical_pixels(
        self, thread_count: int
    ) -> None:
        page = _text_like_page(11, 300, 900, 700)
        page[0, :40] = 0
        page[-1, 200:260] = 0
        page[300:340, 0] = 0
        page[500:520, -1] = 0

        assert np.array_equal(
            despeckle(page, 300, 2.0), despeckle_over_whole_image(page, 300, 2.0)
        )

    def test_specks_that_only_a_second_look_attaches_give_identical_pixels(
        self, thread_count: int
    ) -> None:
        page = np.full((1200, 400), 255, dtype=np.uint8)
        page[100:600, 200:204] = 0
        for top in range(120, 1100, 9):
            page[top : top + 2, 230 + (top % 7) : 232 + (top % 7)] = 0

        assert np.array_equal(
            despeckle(page, 300, 2.0), despeckle_over_whole_image(page, 300, 2.0)
        )


class TestSourcesByLabel:
    """The border search reads each cell's source pixel off its label number."""

    def test_opencv_numbers_zero_pixels_in_row_major_order(self) -> None:
        rng = np.random.default_rng(0)
        image = np.where(rng.random((97, 131)) < 0.3, 0, 255).astype(np.uint8)

        _, labels = cv2.distanceTransformWithLabels(
            image, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
        )

        zeros = image == 0
        assert np.array_equal(labels[zeros], np.arange(1, zeros.sum() + 1))

    def test_every_label_leads_back_to_its_pixel_and_node(self) -> None:
        rng = np.random.default_rng(1)
        nodes = rng.integers(0, 40, (61, 83)).astype(np.int32)
        nodes[rng.random(nodes.shape) < 0.7] = -1
        nodes[:, 0] = -1  # a column and a row without any source
        nodes[5] = -1
        sources = nodes >= 0
        stretched = np.repeat(np.where(sources, 0, 255).astype(np.uint8), _SCALE, 0)
        _, labels = cv2.distanceTransformWithLabels(
            stretched, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
        )

        flat, node = module._sources_by_label(nodes, sources, _SCALE)

        rows, cols = np.nonzero(stretched == 0)
        source_labels = labels[rows, cols]
        assert np.array_equal(flat[source_labels], (rows // _SCALE) * 83 + cols)
        assert np.array_equal(node[source_labels], nodes[rows // _SCALE, cols])
        assert flat.size == node.size == source_labels.size + 1


@pytest.mark.slow
class TestSpeed:
    def test_banded_evaluation_is_faster_than_the_whole_image(self) -> None:
        page = _text_like_page(3, 300, 3000, 2100)
        for top in range(200, 2800, 80):
            for left in range(150, 1950, 40):
                page[top : top + 24, left : left + 12] = 0
        despeckle(page, 300, 2.0)  # compile

        timings = {}
        for name, run in (
            ("banded", lambda: despeckle(page, 300, 2.0)),
            ("whole", lambda: despeckle_over_whole_image(page, 300, 2.0)),
        ):
            best = float("inf")
            for _ in range(2):
                start = time.perf_counter()
                run()
                best = min(best, time.perf_counter() - start)
            timings[name] = best

        assert timings["banded"] < timings["whole"]
