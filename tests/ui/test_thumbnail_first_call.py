"""Thumbnail workers that make the first use of PySide types together do not fail."""

import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scantailor.app.ui.page_renderer import warm_up_qt_image_types

_SRC = Path(__file__).resolve().parents[2] / "src"
_WORKERS = 48
_ROUNDS = 16

# The interpreter is fresh each time because the types are only made once.
_STRESS = textwrap.dedent(
    f"""
    import sys, threading
    from pathlib import Path
    import numpy as np
    from PySide6 import QtCore
    from scantailor.app.ui.page_renderer import render_thumbnail
    from scantailor.core.image_io import write_image
    from scantailor.core.models import Dpi, ImageId, PageId
    from scantailor.core.stage_sequence import FilterStage, StageSequence

    folder = Path(sys.argv[1])
    pages = []
    for index in range({_WORKERS}):
        path = folder / f"page{{index}}.png"
        pixels = np.full((60 + index, 40, 3), 200, dtype=np.uint8)
        write_image(path, pixels, Dpi.uniform(300))
        pages.append(PageId(image_id=ImageId(file_path=path)))
    barrier = threading.Barrier(len(pages))
    errors = []

    def work(page):
        barrier.wait()
        try:
            image = render_thumbnail(
                page, StageSequence(), FilterStage.FIX_ORIENTATION, QtCore.QSize(20, 20)
            )
            assert image.width() > 0
        except BaseException as error:
            errors.append(repr(error))

    threads = [threading.Thread(target=work, args=(page,)) for page in pages]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    print("ERRORS", errors)
    """
)


def _stress_once(folder: Path) -> str:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _STRESS, str(folder)],
        env={**os.environ, "PYTHONPATH": str(_SRC)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    lines = [
        line for line in completed.stdout.splitlines() if line.startswith("ERRORS")
    ]
    return lines[-1] if lines else f"no result: {completed.stderr[-500:]}"


@pytest.mark.slow
def test_first_thumbnails_started_together_all_render(tmp_path: Path) -> None:
    """Workers racing on PySide's lazy types would fail with a missing attribute."""
    folders = []
    for index in range(_ROUNDS):
        folder = tmp_path / f"round{index}"
        folder.mkdir()
        folders.append(folder)

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(_stress_once, folders))

    assert outcomes == ["ERRORS []"] * _ROUNDS


def test_warming_up_the_image_types_can_be_repeated() -> None:
    """The warm-up runs on import and again when a renderer wants to be sure."""
    warm_up_qt_image_types()
    warm_up_qt_image_types()
