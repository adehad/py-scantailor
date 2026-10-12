"""Tests of the script that captures the manual's screenshots."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "docs_screenshots.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("docs_screenshots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sample_pages_are_300_dpi_and_identical_between_runs(tmp_path: Path) -> None:
    """Generated pages carry a resolution and are reproducible byte for byte."""
    script = _load_script()

    first = script.generate_sample_pages(tmp_path / "a")
    second = script.generate_sample_pages(tmp_path / "b")

    assert [p.name for p in first] == script.sample_page_names()
    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()
        with Image.open(left) as image:
            assert tuple(round(v) for v in image.info["dpi"]) == (300, 300)


@pytest.mark.slow
def test_script_writes_every_expected_screenshot_non_blank(tmp_path: Path) -> None:
    """Both colour schemes get every named PNG, each with real content."""
    script = _load_script()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    for scheme in script.SCHEMES:
        for name in script.SHOT_NAMES:
            path = tmp_path / scheme / f"{name}.png"
            assert path.is_file(), f"missing {scheme}/{name}.png"
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("L"))
            assert pixels.std() > 5, f"{scheme}/{name}.png is blank"
