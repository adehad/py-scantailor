"""The frozen checks run unfrozen: the import check and the offscreen GUI run."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
_CHECKS = _TOOLS / "pyinstaller" / "frozen_checks.py"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checks() -> ModuleType:
    """Load tools/pyinstaller/frozen_checks.py by path."""
    return _load("frozen_checks", _CHECKS)


def test_import_check_passes_modules_that_import(checks: ModuleType) -> None:
    report = checks.check_imports(["json", "scantailor.core.settings"])
    assert report == checks.ImportReport()


def test_import_check_fails_a_missing_module_of_a_third_party_package(
    checks: ModuleType,
) -> None:
    report = checks.check_imports(["numpy.no_such_module"])
    assert len(report.failures) == 1
    assert report.failures[0].startswith("numpy.no_such_module: ModuleNotFoundError")


def test_import_check_lists_missing_scantailor_modules_as_tests_only(
    checks: ModuleType,
) -> None:
    report = checks.check_imports(["scantailor.no_such", "scantailor.no_such.sub"])
    assert report.failures == []
    assert report.tests_only == ["scantailor.no_such", "scantailor.no_such.sub"]


def test_import_check_skips_packages_the_build_does_not_contain(
    checks: ModuleType,
) -> None:
    report = checks.check_imports(["no_such_package", "no_such_package.sub", "json"])
    assert report.failures == []
    assert report.absent == {"no_such_package"}


def test_module_list_skips_comments_and_blank_lines(
    checks: ModuleType, tmp_path: Path
) -> None:
    listing = tmp_path / "modules.txt"
    listing.write_text("# header\n\nnumpy\n  scipy.ndimage \n")
    assert checks.read_module_list(listing) == ["numpy", "scipy.ndimage"]


def test_used_modules_list_names_scantailor_and_its_dependencies(
    checks: ModuleType,
) -> None:
    names = checks.read_module_list(_TOOLS / "pyinstaller" / "used_modules.txt")
    assert "scantailor.main" in names
    assert "numpy" in names
    assert not [name for name in names if name.startswith(("pytest", "_pytest"))]


@pytest.mark.slow
def test_gui_check_drives_a_whole_project_without_problems(tmp_path: Path) -> None:
    """Run the GUI check as the frozen build does, in its own process."""
    sample_book = _load("sample_book", _TOOLS / "sample_book.py")
    pages = tmp_path / "pages"
    pages.mkdir()
    for name in sample_book.SAMPLE_PAGE_NAMES[:3]:
        page = sample_book.render_sample_page(name)
        image = sample_book.Image.fromarray(page.image)
        image.save(pages / f"{name}.png", dpi=(sample_book.SAMPLE_DPI,) * 2)
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    completed = subprocess.run(
        [sys.executable, str(_CHECKS), "gui", str(pages), str(tmp_path / "work")],
        capture_output=True,
        text=True,
        env=environment,
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "export PDF: " in completed.stdout
