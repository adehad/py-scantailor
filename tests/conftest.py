"""Fixtures shared by every test package."""

import faulthandler
import pathlib
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from _pytest.faulthandler import fault_handler_stderr_fd_key
from loguru import logger

from scantailor.core import diagnostics
from scantailor.core.settings import ApplicationSettings


def quiet_loguru() -> None:
    """Replace every loguru handler with one stderr sink at WARNING.

    loguru's import-time sink writes DEBUG with local variables to stderr,
    which buries test and script output.
    """
    logger.remove()
    logger.add(sys.stderr, level="WARNING", diagnose=False)


@pytest.fixture(autouse=True, scope="session")
def quiet_stderr_logging() -> None:
    """Keep loguru's stderr output to warnings and errors for the whole run."""
    quiet_loguru()


@pytest.fixture(autouse=True)
def no_process_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> list[dict[str, Any]]:
    """Stand in for configure_logging and record how it was called.

    The real one writes to the user's log folder and replaces the process hooks
    pytest and pytest-qt rely on. tests/core/test_diagnostics.py imports the
    real function by name before this patch, so it still tests it.

    Returns:
        list[dict[str, Any]]: One entry per call: the kind and the options.
    """
    calls: list[dict[str, Any]] = []

    def fake_configure_logging(
        kind: diagnostics.SessionKind, **options: Any
    ) -> diagnostics.DiagnosticsSession:
        calls.append({"kind": kind, **options})
        log_file = options.get("log_file")
        if log_file is None and options.get("write_to_log_folder", True):
            log_file = tmp_path / diagnostics.session_log_name(kind)
        return diagnostics.DiagnosticsSession(log_file=log_file)

    monkeypatch.setattr(diagnostics, "configure_logging", fake_configure_logging)
    return calls


@pytest.fixture
def log_messages() -> Iterator[list[str]]:
    """Capture loguru messages emitted during the test."""
    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="DEBUG"
    )
    yield messages
    logger.remove(handler_id)


@pytest.fixture
def log_records() -> Iterator[list[dict[str, Any]]]:
    """Capture loguru records emitted during the test, from any thread.

    Each entry has the record's ``level`` name, ``message`` and ``extra``, so a
    test can check the page and stage context a line was logged under.
    """
    records: list[dict[str, Any]] = []

    def keep(message: Any) -> None:
        record = message.record
        records.append(
            {
                "level": record["level"].name,
                "message": record["message"],
                "extra": dict(record["extra"]),
            }
        )

    handler_id = logger.add(keep, level="DEBUG")
    yield records
    logger.remove(handler_id)


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
