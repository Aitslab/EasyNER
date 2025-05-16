"""Tests for memory utility functions."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Import the module first so we can reference it directly
import easyner.core.memory_utils as memory_utils
from easyner.core.memory_utils import (
    DEFAULT_MEMORY_CRITICAL_THRESHOLD,
    DEFAULT_MEMORY_HIGH_THRESHOLD,
    get_memory_info,
    get_memory_status,
    get_memory_usage,
    get_total_memory_bytes,
)


class TestMemoryUtils:
    """Test memory utility functions."""

    def test_get_memory_usage_returns_valid_percentage(self) -> None:
        """Test that memory usage returns a valid percentage."""
        memory_usage = get_memory_usage()
        assert isinstance(memory_usage, float)
        assert 0 <= memory_usage <= 100

    def test_get_total_memory_bytes_returns_positive_value_or_none(self) -> None:
        """Test that total memory is a positive value or None."""
        total_memory = get_total_memory_bytes()
        assert total_memory is None or (
            isinstance(total_memory, int) and total_memory > 0
        )

    def test_get_memory_status_levels(self) -> None:
        """Test memory status levels based on different percentages."""
        # Use patch with the module reference instead of string
        with patch.object(memory_utils, "get_memory_usage") as mock_usage:
            # Test normal memory status
            mock_usage.return_value = DEFAULT_MEMORY_HIGH_THRESHOLD - 1
            assert get_memory_status() == 0

            # Test high memory status
            mock_usage.return_value = DEFAULT_MEMORY_HIGH_THRESHOLD + 1
            assert get_memory_status() == 1

            # Test critical memory status
            mock_usage.return_value = DEFAULT_MEMORY_CRITICAL_THRESHOLD + 1
            assert get_memory_status() == 2

    def test_get_memory_info_structure(self) -> None:
        """Test that get_memory_info returns expected structure."""
        memory_info = get_memory_info()
        assert "percent" in memory_info
        assert isinstance(memory_info["percent"], float)

        # These could be None if detection fails
        for key in ["total_mb", "used_mb", "available_mb"]:
            assert key in memory_info
            assert memory_info[key] is None or isinstance(
                memory_info[key],
                (int, float),
            )


class TestPlatformSpecificFallbacks:
    """Test platform-specific fallback methods."""

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="Linux-specific test",
    )
    def test_linux_proc_meminfo_fallback(self) -> None:
        """Test Linux-specific /proc/meminfo fallback."""
        # Mock psutil to be unavailable
        with (
            patch.object(memory_utils, "HAS_PSUTIL", False),
            patch.dict("sys.modules", {"resource": None}),
        ):
            # Ensure /proc/meminfo exists before testing
            if os.path.exists("/proc/meminfo"):
                memory_usage = get_memory_usage()
                assert 0 <= memory_usage <= 100

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific test")
    def test_macos_vm_stat_fallback(self) -> None:
        """Test macOS-specific vm_stat fallback."""
        # Only run on macOS with vm_stat command
        if sys.platform == "darwin":
            # Mock psutil and resource to be unavailable
            with (
                patch.object(memory_utils, "HAS_PSUTIL", False),
                patch.dict("sys.modules", {"resource": None}),
            ):
                try:
                    import subprocess

                    subprocess.check_output(["vm_stat"], universal_newlines=True)
                    # If vm_stat is available, test should work
                    memory_usage = get_memory_usage()
                    assert 0 <= memory_usage <= 100
                except (FileNotFoundError, subprocess.SubprocessError):
                    # vm_stat not available, skip test
                    pytest.skip("vm_stat command not available")

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_windows_fallback(self) -> None:
        """Test Windows-specific memory detection."""
        # Mock psutil to be unavailable
        with patch.object(memory_utils, "HAS_PSUTIL", False):
            memory_usage = get_memory_usage()
            assert 0 <= memory_usage <= 100


class TestPsutilPriority:
    """Test that psutil is used when available."""

    def test_psutil_prioritized_when_available(self) -> None:
        """Test that psutil is used when available."""
        # Create a mock psutil module
        mock_psutil: MagicMock = MagicMock()
        mock_memory: MagicMock = MagicMock()
        mock_memory.percent = 42.0
        mock_psutil.virtual_memory.return_value = mock_memory

        # Patch the import system
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            # And ensure HAS_PSUTIL is True
            with patch.object(memory_utils, "HAS_PSUTIL", True):
                # Call the function
                result = get_memory_usage()

                # Verify results
                assert result == 42.0
                mock_psutil.virtual_memory.assert_called_once()


class TestMemoryFallbacks:
    """Test behavior when memory detection methods fail."""

    def test_fallback_chain(self) -> None:
        """Test the fallback chain when primary methods fail."""
        with (
            patch.object(memory_utils, "HAS_PSUTIL", False),
            patch.dict("sys.modules", {"resource": None}),
            patch.object(sys, "platform", "unknown_platform"),
        ):
            # All methods should fail, returning the default value
            with pytest.raises(memory_utils.MemoryUsageDetectionError):
                get_memory_usage()


if __name__ == "__main__":
    pytest.main()
