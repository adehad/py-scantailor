"""Test that all modules can be imported without circular import errors.

This catches circular imports that only manifest at runtime (not caught by
type checkers). Each subpackage is imported in a subprocess to ensure a
clean import state.
"""

import pkgutil
import subprocess
import sys

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


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_import(module_name: str):
    """Each module should import without errors in a clean interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Failed to import {module_name}:\n{result.stderr}"
