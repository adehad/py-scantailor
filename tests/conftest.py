"""Fixtures shared by every test package.

Setting ``SCANTAILOR_RECORD_IMPORTS=<file>`` writes the name of every module the
session loaded to ``<file>``, one per line, merged across xdist workers.
``tools/pyinstaller/record_used_modules.py`` turns that into the list the frozen
build is checked against.
"""

import os
import pathlib
import sys
from collections.abc import Iterator

import pytest

from scantailor.core.settings import ApplicationSettings

RECORD_IMPORTS_ENV = "SCANTAILOR_RECORD_IMPORTS"


@pytest.fixture(autouse=True)
def isolated_application_settings(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """Point ApplicationSettings at a per-test file, never the user's settings.json."""
    path = tmp_path / "application_settings.json"
    ApplicationSettings.set_config_path(path)
    yield path
    ApplicationSettings.reset_instance()


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Write the loaded module names when SCANTAILOR_RECORD_IMPORTS is set.

    Each xdist worker writes its own part file before it reports back, so the
    controller, which finishes last, folds the parts into the target.

    Args:
        session (pytest.Session): The finishing session.
    """
    configured = os.environ.get(RECORD_IMPORTS_ENV)
    if not configured:
        return
    target = pathlib.Path(configured)
    loaded = set(sys.modules)
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is not None:
        part = target.with_name(f"{target.name}.{worker}")
        part.write_text("\n".join(sorted(loaded)) + "\n")
        return
    for part in target.parent.glob(f"{target.name}.gw*"):
        loaded.update(part.read_text().split())
        part.unlink()
    target.write_text("\n".join(sorted(loaded)) + "\n")
