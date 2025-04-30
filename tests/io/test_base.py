"""Unit tests for the IOHandler base class in the EasyNer library.

This module contains:
- A MockHandler class for testing purposes.
- Test cases for IOHandler's methods, including initialization,
directory creation, and file existence checks.
"""

import os
import tempfile

import pytest

from easyner.io.handlers.base import IOHandler


# Create a concrete implementation of IOHandler for testing
class MockHandler(IOHandler):
    """Mock handler for testing purposes."""

    def read(self, file_path, **kwargs):  # noqa: ANN001, ANN003, D102
        return {"test": "data"}

    def write(self, data, file_path, **kwargs) -> None:  # noqa: D102
        with open(file_path, "w", encoding=self.encoding) as f:
            f.write("test")


class TestIOHandlerBase:
    """Test suite for the IOHandler base class.

    This class contains unit tests for various methods of the IOHandler base class,
    including initialization, directory creation, and file existence checks.
    """

    def _test_init_default_encoding(self) -> None:
        handler = MockHandler()
        assert handler.encoding == IOHandler.DEFAULT_ENCODING

    def _test_init_custom_encoding(self) -> None:
        custom_encoding = "latin-1"
        handler = MockHandler(encoding=custom_encoding)
        assert handler.encoding == custom_encoding

    def _test_ensure_dir_exists(self) -> None:
        handler = MockHandler()

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_dir = os.path.join(tmp_dir, "test_dir")
            test_file = os.path.join(test_dir, "test_file.json")

            # Directory should not exist initially
            assert not os.path.exists(test_dir)

            # Call ensure_dir_exists
            handler.ensure_dir_exists(test_file)

            # Directory should exist now
            assert os.path.exists(test_dir)

            # Calling again should not raise an error
            handler.ensure_dir_exists(test_file)

    def _test_check_file_exists(self) -> None:
        handler = MockHandler()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a test file
            test_file = os.path.join(tmp_dir, "test_file.json")
            with open(test_file, "w") as f:
                f.write("test")

            # Should not raise for existing file
            handler.check_file_exists(test_file)

            # Should raise for non-existing file
            non_existent_file = os.path.join(tmp_dir, "non_existent.json")
            with pytest.raises(FileNotFoundError):
                handler.check_file_exists(non_existent_file)
