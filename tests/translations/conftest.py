"""Fixtures that compile the bundled .ts sources into a temporary .qm folder."""

import importlib.util
import pathlib
from collections.abc import Iterator
from types import ModuleType

import pytest
from PySide6 import QtCore, QtWidgets

from scantailor.translations import TRANSLATIONS_DIR
from scantailor.translations.catalogues import compile_all, find_lrelease

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def build_script() -> ModuleType:
    """Import tools/build_translations.py without putting tools/ on sys.path.

    Returns:
        ModuleType: The loaded build script module.
    """
    spec = importlib.util.spec_from_file_location(
        "build_translations", _REPO_ROOT / "tools" / "build_translations.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def compiled_qm_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Compile every bundled .ts file once per session.

    The .qm files are gitignored build output, so tests compile their own copy
    rather than depend on a prior build step.

    Returns:
        pathlib.Path: Folder holding the compiled scantailor_*.qm files.
    """
    out_dir = tmp_path_factory.mktemp("qm")
    compile_all(TRANSLATIONS_DIR, out_dir, find_lrelease())
    return out_dir


@pytest.fixture
def removed_translators(qapp: QtWidgets.QApplication) -> Iterator[None]:
    """Uninstall every translator a test loaded into the shared application.

    Translators are parented to the application, so they are found as children.
    """
    yield
    for translator in qapp.findChildren(QtCore.QTranslator):
        qapp.removeTranslator(translator)
