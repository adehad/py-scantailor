"""The runtime-only build environment of tools/build_exe.py."""

import importlib.util
import sys
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
def build_exe() -> ModuleType:
    """Load tools/build_exe.py by path."""
    return _load("build_exe", _TOOLS / "build_exe.py")


def test_build_environment_leaves_out_every_group_but_the_installer(
    build_exe: ModuleType, tmp_path: Path
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[dependency-groups]\ninstaller = ["PyInstaller"]\ntest = ["pytest"]\n'
        'docs = ["Sphinx"]\n'
    )
    assert build_exe.other_groups(pyproject) == ["docs", "test"]


def test_build_environment_interpreter_follows_the_platform_layout(
    build_exe: ModuleType,
) -> None:
    python = build_exe.venv_python(Path("venv"))
    expected = "Scripts" if sys.platform == "win32" else "bin"
    assert python.parent == Path("venv") / expected
