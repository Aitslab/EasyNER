"""Factory module for IO handlers.

This module provides a factory function to retrieve the appropriate IOHandler
subclass instance based on the specified file format. Supported formats include
JSON and Parquet, with the ability to add more handlers as needed.
"""

from typing import Optional


from .handlers import IOHandler, JsonHandler, ParquetHandler, SQLiteHandler

# Import other handlers as they are created

_handlers = {
    "json": JsonHandler,
    "parquet": ParquetHandler,
    "sqlite": SQLiteHandler,
    # Add other format mappings here
}


def get_io_handler(file_format: str, encoding: Optional[str] = None) -> IOHandler:
    """Factory function to get an instance of the appropriate IOHandler.

    Args:
        file_format: The desired format ('json', 'parquet', etc.).
        encoding: Optional encoding for the handler.

    Returns:
        An instance of the corresponding IOHandler subclass.

    Raises:
        ValueError: If the requested format is not supported.

    """
    format_lower = file_format.lower()
    handler_class = _handlers.get(format_lower)

    if handler_class:
        return handler_class(encoding=encoding)
    else:
        msg = (
            f"Unsupported file format: '{file_format}'. "
            f"Supported formats: {list(_handlers.keys())}"
        )
        raise ValueError(
            msg,
        )
