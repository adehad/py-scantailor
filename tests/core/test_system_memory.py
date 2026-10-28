"""Physical and resident memory readings on Windows and POSIX, and their failures."""

import ctypes
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scantailor.core import system_memory
from scantailor.core.system_memory import process_rss_bytes, total_ram_bytes

_GIB = 1 << 30


def _structure(by_reference: Any) -> Any:
    """Return the ctypes structure that ``ctypes.byref`` wrapped."""
    return by_reference._obj


def _kernel32(
    total_physical: int | None = None, working_set: int | None = None
) -> SimpleNamespace:
    """Stand in for kernel32, failing each call whose value is None, as Win32 does."""

    def global_memory_status_ex(status: Any) -> int:
        if total_physical is None:
            return 0
        _structure(status).total_physical = total_physical
        return 1

    def get_process_memory_info(process: Any, counters: Any, size: int) -> int:
        if working_set is None:
            return 0
        _structure(counters).WorkingSetSize = working_set
        return 1

    def get_current_process() -> int:
        return 1

    return SimpleNamespace(
        GlobalMemoryStatusEx=global_memory_status_ex,
        K32GetProcessMemoryInfo=get_process_memory_info,
        GetCurrentProcess=get_current_process,
    )


@pytest.fixture
def on_windows(monkeypatch: pytest.MonkeyPatch) -> Callable[[SimpleNamespace], None]:
    """Return a setter that runs the module as on Windows with the given kernel32."""

    def install(kernel32: SimpleNamespace) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False
        )

    return install


@pytest.fixture
def on_posix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run the module as on Linux with 4 KiB pages; return the statm file it reads."""
    sizes = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 2 * _GIB // 4096}
    statm = tmp_path / "statm"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "sysconf", sizes.__getitem__, raising=False)
    monkeypatch.setattr(system_memory, "_STATM", statm)
    return statm


class TestOnWindows:
    def test_total_ram_is_the_physical_memory_kernel32_reports(self, on_windows):
        on_windows(_kernel32(total_physical=16 * _GIB))

        assert total_ram_bytes() == 16 * _GIB

    def test_total_ram_is_unknown_when_kernel32_fails(self, on_windows):
        on_windows(_kernel32())

        assert total_ram_bytes() is None

    def test_resident_memory_is_the_working_set_kernel32_reports(self, on_windows):
        on_windows(_kernel32(working_set=300 << 20))

        assert process_rss_bytes() == 300 << 20

    def test_resident_memory_is_unknown_when_kernel32_fails(self, on_windows):
        on_windows(_kernel32())

        assert process_rss_bytes() is None


class TestOnPosix:
    def test_total_ram_is_the_page_size_times_the_physical_pages(self, on_posix):
        assert total_ram_bytes() == 2 * _GIB

    def test_total_ram_is_unknown_when_sysconf_lacks_the_name(
        self, on_posix, monkeypatch
    ):
        def unknown(name: str) -> int:
            raise ValueError(name)

        monkeypatch.setattr(os, "sysconf", unknown, raising=False)

        assert total_ram_bytes() is None

    def test_resident_memory_is_the_resident_pages_of_statm(self, on_posix):
        on_posix.write_text("90000 2500 600 10 0 4000 0\n", encoding="ascii")

        assert process_rss_bytes() == 2500 * 4096

    @pytest.mark.parametrize("statm", [None, "", "90000 lots"])
    def test_resident_memory_is_unknown_without_a_readable_statm(self, on_posix, statm):
        if statm is not None:
            on_posix.write_text(statm, encoding="ascii")

        assert process_rss_bytes() is None


def test_this_machine_reports_more_ram_than_this_process_holds():
    total, resident = total_ram_bytes(), process_rss_bytes()

    assert total is not None and resident is not None
    assert total > resident > 0
