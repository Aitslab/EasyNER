import pytest
from easyner.io.factory import get_io_handler
from easyner.io.handlers import JsonHandler, ParquetHandler


class TestIOFactory:

    def test_get_json_handler(self):
        # Get JSON handler with default encoding
        handler = get_io_handler("json")
        assert isinstance(handler, JsonHandler)
        assert handler.encoding == JsonHandler.DEFAULT_ENCODING

        # Get JSON handler with custom encoding
        custom_encoding = "latin-1"
        handler = get_io_handler("json", encoding=custom_encoding)
        assert isinstance(handler, JsonHandler)
        assert handler.encoding == custom_encoding

        # Case insensitivity
        handler = get_io_handler("JSON")
        assert isinstance(handler, JsonHandler)

    def test_get_parquet_handler(self):
        # Get Parquet handler
        handler = get_io_handler("parquet")
        assert isinstance(handler, ParquetHandler)

        # Case insensitivity
        handler = get_io_handler("PARQUET")
        assert isinstance(handler, ParquetHandler)

    def test_unsupported_format(self):
        # Test with an unsupported format
        with pytest.raises(ValueError, match="Unsupported file format"):
            get_io_handler("unsupported_format")

    def test_empty_format(self):
        # Test with an empty string
        with pytest.raises(ValueError, match="Unsupported file format"):
            get_io_handler("")
