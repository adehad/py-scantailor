"""Fixtures shared by every test package."""

import faulthandler
import importlib.util
import pathlib
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import cv2
import pytest
from _pytest.faulthandler import fault_handler_stderr_fd_key
from loguru import logger
from PySide6 import QtCore

from scantailor.core import diagnostics
from scantailor.core.settings import ApplicationSettings


def quiet_loguru() -> None:
    """Replace every loguru handler with one stderr sink at WARNING.

    loguru's import-time sink writes DEBUG with local variables to stderr,
    which buries test and script output.
    """
    logger.remove()
    logger.add(sys.stderr, level="WARNING", diagnose=False)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --qt-locale, which runs the suite as a user of that locale would.

    Args:
        parser (pytest.Parser): pytest's command line parser.
    """
    parser.addoption(
        "--qt-locale",
        default=None,
        help=(
            "Qt locale name, such as sv_SE, to make QLocale's default for the "
            "whole run. Use a decimal-comma locale to catch tests that compare "
            "formatted numbers."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Make the --qt-locale locale Qt's default before any test or widget runs.

    Args:
        config (pytest.Config): The run's configuration.
    """
    name = config.getoption("--qt-locale")
    if name:
        QtCore.QLocale.setDefault(QtCore.QLocale(name))


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """List the timings benchmark tests recorded with ``record_property``.

    Only properties named ``benchmark_*`` are listed. They travel in the test
    reports, so they reach the summary from xdist workers too.

    Args:
        terminalreporter (pytest.TerminalReporter): pytest's reporter.
    """
    lines = [
        f"{report.nodeid}: {name} = {value:.3f}"
        for reports in terminalreporter.stats.values()
        for report in reports
        if getattr(report, "when", None) == "call"
        for name, value in getattr(report, "user_properties", ())
        if name.startswith("benchmark_")
    ]
    if lines:
        terminalreporter.section("benchmarks")
        for line in lines:
            terminalreporter.write_line(line)


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


@pytest.fixture(scope="session")
def sample_book_tool() -> ModuleType:
    """``tools/sample_book.py``, imported by path since ``tools/`` is not a package."""
    path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "sample_book.py"
    spec = importlib.util.spec_from_file_location("sample_book", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def sample_book(
    sample_book_tool: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> Any:
    """The sample book's pages and PDF, written once per test session.

    Tests must treat the files as read-only, since every test that asks for this
    fixture shares them. Copy a page before handing it to anything that writes
    beside its input.

    Returns:
        Any: The tool's ``SampleBook``: ``pages`` in sample order and ``pdf``.
    """
    return sample_book_tool.write_sample_book(tmp_path_factory.mktemp("sample_book"))


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


@pytest.fixture
def one_opencv_thread() -> Iterator[None]:
    """Run OpenCV on one thread, as batch workers do, then restore the count.

    Banded kernels hold one band per thread, so memory budgets are stated for
    one thread.
    """
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)
