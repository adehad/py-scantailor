"""Fixtures for the command-line tests: generated sample pages, never owner scans."""

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scantailor.cli.create import create_project

SAMPLE_DPI = 300
PAGE_WIDTH = 900
PAGE_HEIGHT = 1200


def write_sample_page(path: Path, shift: int, dpi: tuple[int, int] | None) -> Path:
    """Draw a white page with dark text-like bars and save it as PNG.

    Args:
        path (pathlib.Path): Destination PNG file.
        shift (int): Vertical offset in pixels, so that two pages differ.
        dpi (tuple[int, int] | None): Resolution stored in the file, or None to
            store none.

    Returns:
        pathlib.Path: The file that was written.
    """
    canvas = np.full((PAGE_HEIGHT, PAGE_WIDTH), 255, dtype=np.uint8)
    for row in range(120 + shift, PAGE_HEIGHT - 100, 60):
        canvas[row : row + 22, 110 : PAGE_WIDTH - 110] = 20
    image = Image.fromarray(canvas)
    if dpi is None:
        image.save(path)
    else:
        image.save(path, dpi=dpi)
    return path


@pytest.fixture
def page_writer() -> Callable[[Path, int, tuple[int, int] | None], Path]:
    """The sample page writer, for tests that need pages of their own."""
    return write_sample_page


@pytest.fixture
def sample_scans(tmp_path: Path) -> Iterator[Path]:
    """Two generated 300 dpi pages in a scans folder under tmp_path."""
    folder = tmp_path / "scans"
    folder.mkdir()
    write_sample_page(folder / "page_01.png", 0, (SAMPLE_DPI, SAMPLE_DPI))
    write_sample_page(folder / "page_02.png", 30, (SAMPLE_DPI, SAMPLE_DPI))
    yield folder


@pytest.fixture
def project_path(tmp_path: Path, sample_scans: Path) -> Path:
    """A project file created from the sample scans, with no batch run yet."""
    path = tmp_path / "proj" / "project.json"
    create_project(
        sample_scans,
        path,
        dpi=None,
        layout_direction="LTR",
        output_directory=None,
        pages_directory=None,
        force=False,
    )
    return path
