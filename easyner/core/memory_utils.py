"""Memory monitoring utilities for cross-platform resource management.

This module provides unified memory monitoring functions that work across
different operating systems with appropriate fallbacks when preferred
methods are unavailable.
"""

import importlib.util
import logging
import sys
import warnings
from typing import Optional, Union

logger = logging.getLogger(__name__)


class MemoryUsageDetectionError(Exception):
    """Available memory detection fails."""

    pass


HAS_PSUTIL = importlib.util.find_spec("psutil") is not None

if not HAS_PSUTIL:
    warnings.warn(
        "psutil not available - using simplified memory monitoring",
        stacklevel=2,
    )

# Memory thresholds for application behavior
DEFAULT_MEMORY_HIGH_THRESHOLD = 60  # Begin memory-saving measures
DEFAULT_MEMORY_CRITICAL_THRESHOLD = 75  # Take immediate action to reduce memory


def get_memory_usage() -> float:  # noqa: C901
    """Get current memory usage as a percentage of total system memory.

    Uses multiple fallback methods for cross-platform compatibility:
    1. psutil (if available) - most accurate, works on all platforms
    2. resource module (Unix-only) - per-process metrics
    3. /proc/meminfo (Linux-only) - system-wide metrics
    4. vm_stat (macOS-only) - system-wide metrics

    Returns:
        float: Memory usage as a percentage (0-100)

    Raises:
        AvailableMemoryDetectionError: If all detection methods fail

    """
    # Track attempted methods for better error reporting
    attempted_methods = []
    errors = []

    # Try psutil first (most accurate, cross-platform)
    if HAS_PSUTIL:
        attempted_methods.append("psutil")
        try:
            import psutil

            return psutil.virtual_memory().percent
        except Exception as e:
            errors.append(f"psutil: {e}")
            logger.debug(f"psutil memory detection failed: {e}")

    # Try resource module for Unix process-specific metrics
    attempted_methods.append("resource")
    try:
        import resource

        rusage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # macOS reports in bytes, Linux in KB
        if sys.platform == "darwin":
            rusage_bytes = rusage_kb
        else:
            rusage_bytes = rusage_kb * 1024

        # Get total memory through platform-specific means
        total_bytes = get_total_memory_bytes()

        if total_bytes:
            return (rusage_bytes / total_bytes) * 100
    except Exception as e:
        errors.append(f"resource: {e}")
        logger.debug(f"resource-based memory detection failed: {e}")

    # Try platform-specific approaches
    if sys.platform.startswith("linux"):
        attempted_methods.append("linux_proc")
        try:
            return _get_memory_usage_linux()
        except MemoryUsageDetectionError as e:
            errors.append(f"linux: {e}")

    elif sys.platform.startswith("darwin"):
        attempted_methods.append("darwin_vm_stat")
        try:
            return _get_memory_usage_darwin()
        except MemoryUsageDetectionError as e:
            errors.append(f"darwin: {e}")

    elif sys.platform == "win32":
        attempted_methods.append("windows_api")
        try:
            return _get_memory_usage_windows()
        except MemoryUsageDetectionError as e:
            errors.append(f"windows: {e}")

    # All methods failed
    error_msg = f"All memory detection methods failed ({', '.join(attempted_methods)})"
    if errors:
        error_details = "; ".join(errors)
        error_msg += f": {error_details}"

    logger.error(error_msg)
    raise MemoryUsageDetectionError(error_msg)


def _get_memory_usage_linux() -> float:
    """Get memory usage on Linux using /proc/meminfo.

    Returns:
        float: Memory usage as a percentage (0-100)

    """
    msg = (
        "Failed to parse /proc/meminfo - "
        "ensure /proc/meminfo is available and readable"
    )

    try:
        with open("/proc/meminfo") as f:
            meminfo = f.readlines()

        mem_total = None
        mem_available = None

        for line in meminfo:
            if "MemTotal" in line:
                mem_total = int(line.split()[1])
            elif "MemAvailable" in line:
                mem_available = int(line.split()[1])

        if mem_total and mem_available:
            return 100 - (mem_available / mem_total * 100)

        raise MemoryUsageDetectionError(
            msg,
        )
    except Exception as e:
        logger.debug(msg)
        raise MemoryUsageDetectionError(msg) from e


def _get_memory_usage_darwin() -> float:
    msg = "macOS vm_stat method failed - ensure vm_stat is available"
    try:
        import subprocess

        vm_stat = subprocess.check_output(["vm_stat"], universal_newlines=True)

        # Parse vm_stat output for page counts
        lines = vm_stat.split("\n")

        # Extract page counts from lines like "Pages free:  12345."
        pages = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":")
                # Extract numeric value, remove dot at end
                try:
                    pages[key.strip()] = int(value.strip().rstrip("."))
                except ValueError:
                    continue

        # Free memory = free pages + inactive pages
        free_pages = pages.get("Pages free", 0) + pages.get("Pages inactive", 0)
        total_pages = (
            free_pages
            + pages.get("Pages active", 0)
            + pages.get("Pages wired down", 0)
            + pages.get("Pages occupied by compressor", 0)
        )

        if total_pages > 0:
            return 100 - ((free_pages / total_pages) * 100)
        raise MemoryUsageDetectionError(msg)
    except Exception as e:
        raise MemoryUsageDetectionError(msg) from e


def _get_memory_usage_windows() -> float:
    msg = "Windows memory detection failed"
    try:
        # Windows-specific fallback using ctypes
        import ctypes

        if not hasattr(ctypes, "windll"):
            msg = "ctypes.windll is not available on this platform"
            raise ImportError(msg)
        # Windows-specific fallback using ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore
        # "windll" is not a known attribute of module "ctypes"
        #  is expected on non windows machines
        c_status = ctypes.c_ulong()
        c_status.value = 0
        kernel32.GlobalMemoryStatusEx(ctypes.byref(c_status))
        return c_status.value
    except ImportError as e:
        msg = msg + " - ctypes not available"
        raise MemoryUsageDetectionError(msg) from e
    except Exception as e:
        logger.debug(f"Windows memory detection failed: {e}")
        raise MemoryUsageDetectionError(msg) from e


def get_total_memory_bytes() -> Optional[int]:  # noqa: C901
    """Get total system memory in bytes using platform-appropriate methods.

    Returns:
        Optional[int]: Total memory in bytes or None if detection fails

    """
    # Try psutil first
    if HAS_PSUTIL:
        try:
            import psutil

            return psutil.virtual_memory().total
        except Exception:
            pass

    # Try platform-specific approaches
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        # Value is in KB
                        return int(line.split()[1]) * 1024

        elif sys.platform == "darwin":
            # Try sysctl on macOS
            import subprocess

            output = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                universal_newlines=True,
            )
            return int(output.strip())

        elif sys.platform == "win32":
            # Windows approach using ctypes
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            memory_status = MEMORYSTATUSEX()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
            return memory_status.ullTotalPhys

    except Exception as e:
        logger.debug(f"Error detecting total memory: {e}")

    return None


def get_memory_status(
    high_threshold: int = DEFAULT_MEMORY_HIGH_THRESHOLD,
    critical_threshold: int = DEFAULT_MEMORY_CRITICAL_THRESHOLD,
) -> int:
    """Get memory status code based on current usage and thresholds.

    Args:
        high_threshold: Threshold for high memory status (default: 60%)
        critical_threshold: Threshold for critical memory status (default: 75%)

    Returns:
        int: Memory status code (0=OK, 1=HIGH, 2=CRITICAL)

    Raises:
        AvailableMemoryDetectionError: If memory detection fails

    """
    try:
        memory_percent = get_memory_usage()

        if memory_percent >= critical_threshold:
            return 2  # CRITICAL
        elif memory_percent >= high_threshold:
            return 1  # HIGH
        else:
            return 0  # OK
    except MemoryUsageDetectionError as e:
        msg = f"Cannot determine memory usage. Failed to get available memory: {e}"
        raise MemoryUsageDetectionError(
            msg,
        ) from e


def get_memory_info() -> dict[str, Union[float, int]]:
    """Get detailed memory information for diagnostic purposes.

    Returns:
        Dict with memory metrics including:
        - percent: Memory usage percentage
        - total_mb: Total system memory in MB
        - used_mb: Used memory in MB
        - available_mb: Available memory in MB

    """
    result = {
        "percent": get_memory_usage(),
        "total_mb": None,
        "used_mb": None,
        "available_mb": None,
    }

    # Try to get detailed memory info
    if HAS_PSUTIL:
        try:
            import psutil

            mem = psutil.virtual_memory()
            result["total_mb"] = mem.total / (1024 * 1024)
            result["used_mb"] = mem.used / (1024 * 1024)
            result["available_mb"] = mem.available / (1024 * 1024)
            return result
        except Exception:
            pass

    # Fallback - try to get total memory at least
    total_bytes = get_total_memory_bytes()
    if total_bytes:
        result["total_mb"] = total_bytes / (1024 * 1024)
        # Estimate used based on percentage
        result["used_mb"] = result["total_mb"] * (result["percent"] / 100)
        result["available_mb"] = result["total_mb"] - result["used_mb"]

    return result
