"""Fixtures for the image processing tests."""

import itertools
from collections.abc import Callable

import pytest

from scantailor.imageproc import row_bands


@pytest.fixture
def band_height(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Return a setter that makes every banded kernel cut bands of that many rows.

    A height of 0 or less gives one band over the whole image, which is the
    kernel evaluated without banding.
    """

    def cut_at(height: int) -> None:
        def fixed(rows: int, halo: int, bytes_per_row: int) -> list[row_bands.RowBand]:
            step = height if height > 0 else max(rows, 1)
            edges = [*range(0, rows, step), rows] if rows else [0, 0]
            return [
                row_bands.RowBand(
                    first, stop, max(0, first - halo), min(rows, stop + halo)
                )
                for first, stop in itertools.pairwise(edges)
            ]

        monkeypatch.setattr(row_bands, "row_bands", fixed)

    return cut_at
