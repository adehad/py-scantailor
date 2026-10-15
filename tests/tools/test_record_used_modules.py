"""Filtering the recorded imports in tools/pyinstaller/record_used_modules.py."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def record() -> ModuleType:
    """Load tools/pyinstaller/record_used_modules.py by path."""
    return _load(
        "record_used_modules", _TOOLS / "pyinstaller" / "record_used_modules.py"
    )


def test_filter_keeps_scantailor_and_runtime_packages_only(record: ModuleType) -> None:
    loaded = {
        "scantailor",
        "scantailor.main",
        "numpy.linalg",
        "pytest",
        "json",
        "PySide6.QtWidgets",
        "PySide6.QtTest",
    }
    kept = record.filter_modules(loaded, {"scantailor", "numpy", "PySide6"})
    assert kept == [
        "PySide6.QtWidgets",
        "numpy.linalg",
        "scantailor",
        "scantailor.main",
    ]


def test_distribution_names_normalise_like_pep_503(record: ModuleType) -> None:
    assert record.normalise("PySide6_Essentials") == "pyside6-essentials"
    assert record.normalise("typing.extensions") == "typing-extensions"


def test_runtime_closure_follows_requirements_but_not_extras(
    record: ModuleType,
) -> None:
    runtime = record.runtime_distributions()
    assert {"py-scantailor", "numpy", "pyside6", "llvmlite"} <= runtime
    assert "pytest" not in runtime
