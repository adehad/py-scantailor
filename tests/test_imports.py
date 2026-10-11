"""Test that all modules can be imported without circular import errors.

This catches circular imports that only manifest at runtime (not caught by
type checkers). Each subpackage is imported in a subprocess to ensure a
clean import state.
"""

import os
import pkgutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

import scantailor


def _find_all_modules(package) -> list[str]:
    """Recursively find all module names in a package."""
    modules = [package.__name__]
    if not hasattr(package, "__path__"):
        return modules
    for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
        modules.append(info.name)
    return modules


ALL_MODULES = _find_all_modules(scantailor)

# The module fixture starts one interpreter per module; one xdist worker must own
# them all, or every worker spawns the full set and the imports time out.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("imports")]


def _import_in_clean_interpreter(module_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture(scope="module")
def import_results() -> dict[str, subprocess.CompletedProcess[str]]:
    """Import every module in its own interpreter, concurrently."""
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = pool.map(_import_in_clean_interpreter, ALL_MODULES)
        return dict(zip(ALL_MODULES, results, strict=True))


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_import(
    module_name: str, import_results: dict[str, subprocess.CompletedProcess[str]]
):
    """Each module should import without errors in a clean interpreter."""
    result = import_results[module_name]
    assert result.returncode == 0, f"Failed to import {module_name}:\n{result.stderr}"
