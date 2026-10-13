"""Tests for the thread-safe memory of results computed from large arrays."""

import threading

import numpy as np
import pytest

from scantailor.core.result_memo import ResultMemo, array_digest


class TestResultMemo:
    def test_equal_keys_are_computed_once(self) -> None:
        memo = ResultMemo[int](capacity=4)
        calls: list[str] = []

        def compute() -> int:
            calls.append("run")
            return 7

        first = memo.get_or_compute("a", compute)
        second = memo.get_or_compute("a", compute)

        assert (first, second) == (7, 7)
        assert calls == ["run"]

    def test_different_keys_are_computed_separately(self) -> None:
        memo = ResultMemo[str](capacity=4)

        assert memo.get_or_compute("a", lambda: "first") == "first"
        assert memo.get_or_compute("b", lambda: "second") == "second"

    def test_least_recently_used_result_is_dropped_beyond_capacity(self) -> None:
        memo = ResultMemo[int](capacity=2)
        memo.get_or_compute("a", lambda: 1)
        memo.get_or_compute("b", lambda: 2)
        memo.get_or_compute("a", lambda: 99)  # touch a, so b is the oldest
        memo.get_or_compute("c", lambda: 3)

        assert memo.peek("a") == 1
        assert memo.peek("b") is None
        assert memo.peek("c") == 3

    def test_failed_computation_is_not_remembered(self) -> None:
        memo = ResultMemo[int](capacity=2)

        def explode() -> int:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            memo.get_or_compute("a", explode)

        assert memo.peek("a") is None
        assert memo.get_or_compute("a", lambda: 5) == 5

    def test_peek_does_not_compute(self) -> None:
        memo = ResultMemo[int](capacity=2)

        assert memo.peek("missing") is None

    def test_clear_forgets_everything(self) -> None:
        memo = ResultMemo[int](capacity=2)
        memo.get_or_compute("a", lambda: 1)

        memo.clear()

        assert memo.peek("a") is None

    def test_threads_asking_for_one_key_share_a_single_computation(self) -> None:
        memo = ResultMemo[int](capacity=2)
        started = threading.Event()
        release = threading.Event()
        calls: list[str] = []
        results: list[int] = []

        def slow() -> int:
            calls.append("run")
            started.set()
            assert release.wait(timeout=10)
            return 42

        def ask() -> None:
            results.append(memo.get_or_compute("page", slow))

        threads = [threading.Thread(target=ask) for _ in range(4)]
        threads[0].start()
        assert started.wait(timeout=10)
        for thread in threads[1:]:
            thread.start()
        release.set()
        for thread in threads:
            thread.join(timeout=10)

        assert calls == ["run"]
        assert results == [42, 42, 42, 42]


class TestArrayDigest:
    def test_equal_arrays_have_the_same_digest(self) -> None:
        a = np.arange(12, dtype=np.uint8).reshape(3, 4)

        assert array_digest(a) == array_digest(a.copy())

    def test_one_changed_value_changes_the_digest(self) -> None:
        a = np.zeros((4, 4), dtype=np.uint8)
        b = a.copy()
        b[2, 3] = 1

        assert array_digest(a) != array_digest(b)

    def test_same_bytes_in_another_shape_have_another_digest(self) -> None:
        a = np.zeros((2, 8), dtype=np.uint8)

        assert array_digest(a) != array_digest(a.reshape(4, 4))

    def test_same_bytes_of_another_type_have_another_digest(self) -> None:
        a = np.zeros(8, dtype=np.uint8)

        assert array_digest(a) != array_digest(a.view(np.int8))

    def test_a_non_contiguous_view_matches_its_contiguous_copy(self) -> None:
        a = np.arange(40, dtype=np.uint8).reshape(5, 8)
        view = a[:, ::2]

        assert array_digest(view) == array_digest(np.ascontiguousarray(view))
