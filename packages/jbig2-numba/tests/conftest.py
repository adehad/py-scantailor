"""Collect measurements from tests and show them in the terminal summary."""

import pytest

_MEASUREMENTS: list[str] = []


@pytest.fixture
def report_measurement():
    """Record a line for the end-of-run summary.

    Returns:
        Callable[[str], None]: Appends one line.
    """
    return _MEASUREMENTS.append


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Print the recorded measurements.

    Args:
        terminalreporter (pytest.TerminalReporter): pytest's reporter.
    """
    if _MEASUREMENTS:
        terminalreporter.section("jbig2-numba measurements")
        for line in _MEASUREMENTS:
            terminalreporter.write_line(line)
