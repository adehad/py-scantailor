"""Tests for the batch runner: naming, process pool behaviour and end-to-end output."""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest
from PIL import Image

from scantailor.core import batch
from scantailor.core.batch import (
    BatchResult,
    PageDone,
    PageFailed,
    output_file_names,
    run_batch,
    run_pool,
)
from scantailor.core.image_io import load_image, write_image
from scantailor.core.models import Dpi, ImageId, PageId, SubPage
from scantailor.core.stage_sequence import StageSequence
from scantailor.filters.page_split.layout_type import LayoutType
from scantailor.filters.select_content.content_box import ContentBox

SOURCE_DPI = 300
OUTPUT_DPI = 600


def _page(path: str, page: int = 0, sub: SubPage = SubPage.SINGLE_PAGE) -> PageId:
    return PageId(image_id=ImageId(file_path=Path(path), page=page), sub_page=sub)


class TestOutputFileNames:
    """File naming follows OutputFileNameGenerator.cpp."""

    def test_plain_page_uses_base_name_with_tif_suffix(self):
        page = _page("/scans/cover.png")

        assert output_file_names([page]) == {page: "cover.tif"}

    def test_only_the_last_suffix_is_dropped(self):
        page = _page("/scans/book.v2.jpg")

        assert output_file_names([page])[page] == "book.v2.tif"

    def test_same_file_name_in_other_directories_is_labelled_in_order(self):
        first, second, third = (
            _page("/a/scan.tif"),
            _page("/b/scan.tif"),
            _page("/c/scan.tif"),
        )

        names = output_file_names([first, second, third])

        assert [names[p] for p in (first, second, third)] == [
            "scan.tif",
            "scan(1).tif",
            "scan(2).tif",
        ]

    def test_two_pages_of_one_file_share_a_label(self):
        left = _page("/a/scan.tif", sub=SubPage.LEFT_PAGE)
        right = _page("/a/scan.tif", sub=SubPage.RIGHT_PAGE)
        other = _page("/b/scan.tif")

        names = output_file_names([left, right, other])

        assert names[left] == "scan_1L.tif"
        assert names[right] == "scan_2R.tif"
        assert names[other] == "scan(1).tif"

    def test_same_stem_with_different_extension_is_not_a_collision(self):
        png, tif = _page("/a/scan.png"), _page("/b/scan.tif")

        names = output_file_names([png, tif])

        assert names[png] == names[tif] == "scan.tif"

    def test_multi_page_file_gets_zero_padded_page_number(self):
        page = _page("/a/book.tif", page=7)

        assert output_file_names([page])[page] == "book_page0007.tif"

    def test_right_to_left_numbers_the_right_half_first(self):
        left = _page("/a/s.tif", sub=SubPage.LEFT_PAGE)
        right = _page("/a/s.tif", sub=SubPage.RIGHT_PAGE)

        names = output_file_names([left, right], layout_direction="RTL")

        assert names[left] == "s_2L.tif"
        assert names[right] == "s_1R.tif"

    def test_multi_page_label_and_split_suffixes_combine_in_cpp_order(self):
        first = _page("/a/b.tif", page=2, sub=SubPage.LEFT_PAGE)
        second = _page("/c/b.tif", page=2, sub=SubPage.RIGHT_PAGE)

        names = output_file_names([first, second])

        assert names[first] == "b_page0002_1L.tif"
        assert names[second] == "b(1)_page0002_2R.tif"


def _thread_counts(_: int) -> tuple[int, str | None]:
    import os

    return cv2.getNumThreads(), os.environ.get("NUMBA_NUM_THREADS")


def _slow_square(value: int) -> int:
    time.sleep(0.4)
    return value * value


def _fail_on_three(value: int) -> int:
    if value == 3:
        raise ValueError("three is not allowed")
    return value


def _worker_marker(value: int) -> int:
    return value + _WORKER_STATE["offset"]


_WORKER_STATE = {"offset": 0}


def _set_offset(offset: int) -> None:
    _WORKER_STATE["offset"] = offset


@pytest.mark.slow
class TestWorkerThreads:
    def test_initializer_limits_opencv_and_numba_to_one_thread(self):
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
            initializer=batch.configure_worker_threads,
        ) as executor:
            cv_threads, numba_threads = executor.submit(_thread_counts, 0).result()

        assert cv_threads == 1
        assert numba_threads == "1"

    def test_run_pool_applies_thread_limits_and_extra_initializer(self):
        seen: dict[int, int] = {}

        def record(job, result, error):
            assert error is None
            seen[job] = result

        run_pool(
            _worker_marker,
            [1, 2],
            record,
            max_workers=1,
            initializer=_set_offset,
            initargs=(10,),
        )

        assert seen == {1: 11, 2: 12}


@pytest.mark.slow
class TestRunPool:
    def test_failing_job_is_reported_and_others_complete(self):
        outcomes: dict[int, tuple[int | None, BaseException | None]] = {}

        def record(job, result, error):
            outcomes[job] = (result, error)

        unscheduled = run_pool(_fail_on_three, [1, 2, 3, 4, 5], record, max_workers=2)

        assert unscheduled == []
        assert {job: result for job, (result, _) in outcomes.items() if result} == {
            1: 1,
            2: 2,
            4: 4,
            5: 5,
        }
        result, error = outcomes[3]
        assert result is None
        assert isinstance(error, ValueError)
        assert "three is not allowed" in str(error)

    def test_cancel_after_first_result_stops_scheduling(self):
        cancel = Event()
        finished: list[int] = []

        def record(job, result, error):
            assert error is None
            finished.append(job)
            cancel.set()

        unscheduled = run_pool(
            _slow_square, list(range(8)), record, max_workers=2, cancel=cancel
        )

        # One job may already have been in flight beside the first one.
        assert len(finished) <= 2
        assert len(finished) + len(unscheduled) == 8
        assert unscheduled == list(range(len(finished), 8))

    def test_already_cancelled_runs_nothing(self):
        cancel = Event()
        cancel.set()

        def never(job, result, error):
            raise AssertionError("no job should run")

        unscheduled = run_pool(_slow_square, [1, 2, 3], never, cancel=cancel)

        assert unscheduled == [1, 2, 3]


def _make_source(path: Path, shade: int) -> None:
    """A 300 dpi grey page with a dark block on a light background."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((400, 300), 220, dtype=np.uint8)
    pixels[100:300, 80:220] = shade
    write_image(path, pixels, Dpi.uniform(SOURCE_DPI))


def _configure_manual(stages: StageSequence, page: PageId) -> None:
    stages.page_split_filter.set_layout_type(page, LayoutType.SINGLE_PAGE_UNCUT)
    stages.deskew_filter.set_manual_angle(page, 0.0)
    stages.select_content_filter.set_manual_content(
        page, ContentBox(x=80, y=100, width=140, height=200)
    )


@pytest.fixture
def sources(tmp_path: Path) -> list[PageId]:
    paths = [
        tmp_path / "in" / "a.tif",
        tmp_path / "in" / "other" / "a.tif",
        tmp_path / "in" / "b.tif",
        tmp_path / "in" / "c.tif",
    ]
    for shade, path in zip((30, 50, 70, 90), paths, strict=True):
        _make_source(path, shade)
    return [_page(str(path)) for path in paths]


@pytest.fixture
def stages(sources: list[PageId]) -> StageSequence:
    sequence = StageSequence()
    for page in sources:
        _configure_manual(sequence, page)
    return sequence


@pytest.mark.slow
class TestRunBatch:
    def test_four_pages_are_written_with_cpp_names_and_output_dpi(
        self, tmp_path: Path, sources: list[PageId], stages: StageSequence
    ):
        out = tmp_path / "out"
        events = []

        result = run_batch(
            stages, sources, out, max_workers=2, on_progress=events.append
        )

        assert sorted(p.name for p in out.iterdir()) == [
            "a(1).tif",
            "a.tif",
            "b.tif",
            "c.tif",
        ]
        assert result.failed == {}
        assert result.cancelled == []
        assert sorted(e.page for e in events if isinstance(e, PageDone)) == sorted(
            sources
        )
        for path in result.written.values():
            with Image.open(path) as image:
                assert image.info["dpi"] == pytest.approx(
                    (OUTPUT_DPI, OUTPUT_DPI), abs=1
                )
                assert image.size[0] > 0 and image.size[1] > 0

    def test_outputs_share_one_page_size(
        self, tmp_path: Path, sources: list[PageId], stages: StageSequence
    ):
        result = run_batch(stages, sources, tmp_path / "out", max_workers=2)

        sizes = {
            load_image(ImageId(file_path=p))[0].shape for p in result.written.values()
        }
        assert len(sizes) == 1

    def test_missing_source_is_reported_and_others_complete(
        self, tmp_path: Path, sources: list[PageId], stages: StageSequence
    ):
        missing = _page(str(tmp_path / "in" / "missing.tif"))
        _configure_manual(stages, missing)
        events = []

        result = run_batch(
            stages,
            [*sources[:2], missing, *sources[2:]],
            tmp_path / "out",
            max_workers=2,
            on_progress=events.append,
        )

        assert set(result.failed) == {missing}
        assert "missing.tif" in result.failed[missing]
        assert set(result.written) == set(sources)
        failures = [e for e in events if isinstance(e, PageFailed)]
        assert [f.page for f in failures] == [missing]

    def test_cancelled_before_start_writes_nothing(
        self, tmp_path: Path, sources: list[PageId], stages: StageSequence
    ):
        cancel = Event()
        cancel.set()

        result = run_batch(stages, sources, tmp_path / "out", cancel=cancel)

        assert result == BatchResult(cancelled=sources)
        assert not (tmp_path / "out").exists()

    def test_cancel_from_progress_callback_skips_remaining_pages(
        self, tmp_path: Path, sources: list[PageId], stages: StageSequence
    ):
        cancel = Event()

        result = run_batch(
            stages,
            sources,
            tmp_path / "out",
            max_workers=1,
            on_progress=lambda event: (
                cancel.set() if isinstance(event, PageDone) else None
            ),
            cancel=cancel,
        )

        assert len(result.written) == 1
        assert len(result.cancelled) == len(sources) - 1
        assert set(result.written) | set(result.cancelled) == set(sources)


class TestAsWritable:
    def test_binary_output_becomes_ink_is_true_mask(self):
        image = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        mask = batch._as_writable(image, is_binary=True)

        assert mask.dtype == np.bool_
        assert mask.tolist() == [[True, False], [False, True]]

    def test_alpha_channel_is_dropped_from_colour_output(self):
        image = np.zeros((2, 2, 4), dtype=np.uint8)

        assert batch._as_writable(image, is_binary=False).shape == (2, 2, 3)

    def test_grey_output_is_passed_through_unchanged(self):
        image = np.full((2, 2), 7, dtype=np.uint8)

        assert batch._as_writable(image, is_binary=False) is image
