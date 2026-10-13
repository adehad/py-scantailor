"""Bounded memory of results computed from large arrays, safe across threads."""

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable

import numpy as np
from numpy.typing import NDArray

__all__ = ["ResultMemo", "array_digest"]

_DIGEST_BYTES = 16


def array_digest(array: NDArray[np.generic]) -> bytes:
    """Return a digest that tells arrays of equal shape, type and values apart.

    Args:
        array (NDArray[np.generic]): Any array; not modified.

    Returns:
        bytes: Digest of the shape, dtype and contents.
    """
    hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    hasher.update(f"{array.shape}{array.dtype.str}".encode())
    hasher.update(np.ascontiguousarray(array))
    return hasher.digest()


class ResultMemo[Result]:
    """Least recently used results of a pure function, keyed by its inputs.

    Threads asking for a key that another thread is computing wait for that
    result instead of computing it again.

    Args:
        capacity (int): Most results kept.
    """

    def __init__(self, capacity: int) -> None:
        """Create an empty memo; arguments are described on the class."""
        self._capacity = capacity
        self._lock = threading.Lock()
        self._results: OrderedDict[Hashable, Result] = OrderedDict()
        self._computing: dict[Hashable, threading.Lock] = {}

    def get_or_compute(self, key: Hashable, compute: Callable[[], Result]) -> Result:
        """Return the remembered result for ``key``, computing it first if needed.

        Args:
            key (Hashable): Everything ``compute`` depends on.
            compute (Callable[[], Result]): Produces the result; called without
                any lock held by this memo other than ``key``'s own.

        Returns:
            Result: The result for ``key``.
        """
        with self._lock:
            if key in self._results:
                self._results.move_to_end(key)
                return self._results[key]
            gate = self._computing.setdefault(key, threading.Lock())
        with gate:
            with self._lock:
                if key in self._results:
                    self._results.move_to_end(key)
                    return self._results[key]
            try:
                result = compute()
            except BaseException:
                with self._lock:
                    self._computing.pop(key, None)
                raise
            with self._lock:
                self._results[key] = result
                self._computing.pop(key, None)
                while len(self._results) > self._capacity:
                    self._results.popitem(last=False)
            return result

    def peek(self, key: Hashable) -> Result | None:
        """Return the remembered result for ``key`` without computing anything.

        Args:
            key (Hashable): Everything the result depends on.

        Returns:
            Result | None: The result, or None when it is not remembered.
        """
        with self._lock:
            if key not in self._results:
                return None
            self._results.move_to_end(key)
            return self._results[key]

    def clear(self) -> None:
        """Forget every remembered result."""
        with self._lock:
            self._results.clear()
