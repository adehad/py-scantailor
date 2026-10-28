"""Physical memory of the machine and resident memory of this process.

Standard library only: ctypes on Windows, sysconf and procfs elsewhere. Each
function returns None when the platform does not say, so callers fall back to
not limiting anything.
"""

import ctypes
import os
import sys
from pathlib import Path

_STATM = Path("/proc/self/statm")


class _MemoryStatusEx(ctypes.Structure):
    # MEMORYSTATUSEX; ctypes matches fields by order, not name.
    _fields_ = (
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    )


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    )


def total_ram_bytes() -> int | None:
    """Physical memory installed, in bytes.

    Returns:
        int | None: Total physical memory, or None when it cannot be read.
    """
    if sys.platform == "win32":
        status = _MemoryStatusEx()
        status.length = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.total_physical)
    else:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except AttributeError, OSError, ValueError:
            return None


def process_rss_bytes() -> int | None:
    """Resident memory of the current process, in bytes.

    Returns:
        int | None: The working set on Windows, the resident set from procfs
            elsewhere, or None when neither can be read.
    """
    if sys.platform == "win32":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        if not kernel32.K32GetProcessMemoryInfo(
            ctypes.c_void_p(kernel32.GetCurrentProcess()),
            ctypes.byref(counters),
            counters.cb,
        ):
            return None
        return int(counters.WorkingSetSize)
    else:
        try:
            resident_pages = int(_STATM.read_text().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except OSError, ValueError, IndexError:
            return None
