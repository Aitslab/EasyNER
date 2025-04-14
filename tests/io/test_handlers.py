"""Unit tests for the EasyNer I/O handlers.

This module contains test cases for the JsonHandler and ParquetHandler classes,
which handle reading and writing JSON and Parquet files, respectively.
"""

import json
import os
import tempfile

import pytest

from easyner.io.handlers import JsonHandler, ParquetHandler


class TestJsonHandler:
    """Test cases for the JsonHandler class.

    This class contains unit tests for verifying the functionality of the JsonHandler,
    including reading and writing JSON files under various conditions.
    """

    def _test_extension_constant(self) -> None:
        assert JsonHandler.EXTENSION == "json"

    def _test_read_valid_json(self) -> None:
        handler = JsonHandler()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid test JSON file
            test_data = {"key1": "value1", "key2": {"nested": "value2"}}
            test_file = os.path.join(tmp_dir, "test_file.json")

            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

                # Read the file back
            result = handler.read(test_file)
            assert result == test_data

    def _test_read_invalid_json(self) -> None:
        handler = JsonHandler()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create an invalid JSON file
            test_file = os.path.join(tmp_dir, "invalid.json")

            with open(test_file, "w", encoding="utf-8") as f:
                f.write('{"key": "value", invalid')

            # Should raise ValueError for invalid JSON
            with pytest.raises(ValueError, match="Error decoding JSON file"):
                handler.read(test_file)

    def _test_read_nonexistent_file(self) -> None:
        handler = JsonHandler()

        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent_file = os.path.join(tmp_dir, "non_existent.json")

            # Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError):
                handler.read(non_existent_file)

    def _test_write_json_no_indent(self) -> None:
        handler = JsonHandler()

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_data = {"key1": "value1", "key2": [1, 2, 3]}
            test_file = os.path.join(tmp_dir, "output.json")

            # Write data to file
            handler.write(test_data, test_file)

            # Read the file back and verify contents
            with open(test_file, encoding="utf-8") as f:
                result = json.load(f)

            assert result == test_data

    def _test_write_json_with_indent(self) -> None:
        handler = JsonHandler()

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_data = {"key1": "value1", "key2": [1, 2, 3]}
            test_file = os.path.join(tmp_dir, "output_pretty.json")

            # Write data to file with indentation
            handler.write(test_data, test_file, indent=2)

            # Read the file back and verify contents
            with open(test_file, encoding="utf-8") as f:
                result = json.load(f)

            assert result == test_data

            # Also check if it's pretty-printed (indented)
            with open(test_file, encoding="utf-8") as f:
                content = f.read()

            # Pretty-printed JSON should span multiple lines
            assert content.count("\n") > 0


class TestParquetHandler:
    """Test cases for the ParquetHandler class.

    This class contains unit tests for verifying the functionality
    of the ParquetHandler, including reading and writing Parquet
    files under various conditions.
    """

    def _test_extension_constant(self) -> None:
        assert ParquetHandler.EXTENSION == "parquet"

    @pytest.mark.xfail(reason="Parquet reading not implemented yet")
    def _test_read_not_implemented(self) -> None:
        handler = ParquetHandler()

        with pytest.raises(NotImplementedError):
            handler.read("dummy.parquet")

    @pytest.mark.xfail(reason="Parquet writing not implemented yet")
    def _test_write_not_implemented(self) -> None:
        handler = ParquetHandler()

        with pytest.raises(NotImplementedError):
            handler.write({"test": "data"}, "dummy.parquet")
