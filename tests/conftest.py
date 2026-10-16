"""Fixtures shared by every test package."""

import faulthandler
import pathlib
from collections.abc import Iterator

import pytest
from _pytest.faulthandler import fault_handler_stderr_fd_key

from scantailor.core.settings import ApplicationSettings


@pytest.fixture(autouse=True)
def isolated_application_settings(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """Point ApplicationSettings at a per-test file, never the user's settings.json."""
    path = tmp_path / "application_settings.json"
    ApplicationSettings.set_config_path(path)
    yield path
    ApplicationSettings.reset_instance()


def pytest_timeout_set_timer(item: pytest.Item, settings: object) -> bool:
    """Dump every thread's stack and exit when a test overruns its timeout.

    pytest-timeout prints its own dump through the terminal writer, which an xdist
    worker never shows. faulthandler writes to the stderr pytest kept aside for
    crash dumps, which reaches the log, and exits from C even while Qt blocks.

    Args:
        item (pytest.Item): The test about to run.
        settings (object): pytest-timeout's settings; ``timeout`` is in seconds.

    Returns:
        bool: True, so pytest-timeout does not start its own timer as well.
    """
    faulthandler.dump_traceback_later(
        settings.timeout,  # type: ignore[attr-defined]
        exit=True,
        file=item.config.stash[fault_handler_stderr_fd_key],
    )
    return True


def pytest_timeout_cancel_timer(item: pytest.Item) -> bool:
    """Stop the dump that pytest_timeout_set_timer armed.

    Args:
        item (pytest.Item): The test that has finished.

    Returns:
        bool: True, so pytest-timeout does not look for a timer of its own.
    """
    faulthandler.cancel_dump_traceback_later()
    return True
