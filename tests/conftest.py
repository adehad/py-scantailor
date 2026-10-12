"""Fixtures shared by every test package."""

import pathlib
from collections.abc import Iterator

import pytest

from scantailor.core.settings import ApplicationSettings


@pytest.fixture(autouse=True)
def isolated_application_settings(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """Point ApplicationSettings at a per-test file, never the user's settings.json."""
    path = tmp_path / "application_settings.json"
    ApplicationSettings.set_config_path(path)
    yield path
    ApplicationSettings.reset_instance()
