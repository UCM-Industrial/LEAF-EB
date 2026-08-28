"""Small cross-platform helpers for LEAF memory management."""

from __future__ import annotations

import ctypes
import gc
import os
import sys


MIB = 1024 ** 2
GIB = 1024 ** 3
DEFAULT_WORKER_MEMORY_MB = 700
MIN_MEMORY_RESERVE_MB = 768
MEMORY_RESERVE_FRACTION = 0.15


def release_unused_memory() -> None:
    """Collect Python garbage and return free native heap pages when safe.

    NumPy and pandas allocate large temporary buffers outside Python's object
    graph.  On glibc-based Linux systems, ``malloc_trim`` can return those
    released pages to the operating system between LEAF pipeline stages.
    Other platforms simply perform normal garbage collection.
    """

    gc.collect()
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim")
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        return


def available_memory_bytes() -> int | None:
    """Return currently available physical memory when it can be detected."""

    if os.name == "nt":
        return _windows_available_memory()

    if sys.platform.startswith("linux"):
        system_available = _linux_system_available_memory()
        cgroup_available = _linux_cgroup_available_memory()
        candidates = [
            value for value in (system_available, cgroup_available)
            if value is not None and value > 0]
        if candidates:
            return min(candidates)

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return None
    return None



def _linux_system_available_memory() -> int | None:
    """Return Linux MemAvailable from procfs when present."""

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _linux_cgroup_available_memory() -> int | None:
    """Return remaining memory inside a Linux cgroup limit, if any."""

    pairs = [
        ("/sys/fs/cgroup/memory.max",
         "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"),]
    for limit_path, usage_path in pairs:
        try:
            limit_text = open(
                limit_path, "r", encoding="utf-8").read().strip()
            if limit_text.lower() == "max":
                continue
            limit = int(limit_text)
            usage = int(open(
                usage_path, "r", encoding="utf-8").read().strip())
        except (OSError, ValueError):
            continue
        # cgroup v1 may expose a huge sentinel when memory is unlimited.
        if limit <= 0 or limit >= (1 << 60):
            continue
        return max(0, limit - usage)
    return None

def _windows_available_memory() -> int | None:
    """Return available physical memory using GlobalMemoryStatusEx."""

    class MemoryStatus(ctypes.Structure):
        """Windows MEMORYSTATUSEX structure used by GlobalMemoryStatusEx."""

        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),]

    try:
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    except (AttributeError, OSError):
        return None


def memory_safe_worker_count(
        requested: int,
        task_count: int,
        cpu_limit: int,
        memory_per_worker_mb: int = DEFAULT_WORKER_MEMORY_MB,
) -> tuple[int, dict[str, float | int | None]]:
    """Return a worker count bounded by CPU, tasks, and available RAM.

    ``requested`` remains the user's upper bound.  The memory guard uses a
    conservative per-worker estimate and keeps part of currently available
    RAM free for the operating system and the LEAF coordinator process.
    """

    requested = max(1, int(requested))
    task_count = max(1, int(task_count))
    cpu_limit = max(1, int(cpu_limit))
    base_limit = min(requested, task_count, cpu_limit)
    available = available_memory_bytes()

    details: dict[str, float | int | None] = {
        "requested": requested,
        "task_count": task_count,
        "cpu_limit": cpu_limit,
        "available_gib": None,
        "memory_limit": None,
        "memory_per_worker_mb": int(memory_per_worker_mb),}

    if available is None or available <= 0:
        return base_limit, details

    reserve = max(
        MIN_MEMORY_RESERVE_MB * MIB,
        int(available * MEMORY_RESERVE_FRACTION),)
    usable = max(0, available - reserve)
    worker_bytes = max(1, int(memory_per_worker_mb)) * MIB
    memory_limit = max(1, int(usable // worker_bytes))
    details["available_gib"] = available / GIB
    details["memory_limit"] = memory_limit
    return min(base_limit, memory_limit), details
