"""Base class for file I/O operations.

It defines an abstract base class `IOHandler` with methods for
reading and writing files, as well as common helper methods for
ensuring directories exist and checking file existence.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class IOHandler(ABC):
    """Abstract Base Class for file I/O operations."""

    DEFAULT_ENCODING = "utf-8"

    def __init__(self, encoding: Optional[str] = None) -> None:
        """Initialize the IOHandler with an optional encoding.

        Args:
            encoding (Optional[str]): The encoding to use for file operations.
            Defaults to 'utf-8'.

        """
        self.encoding = encoding or self.DEFAULT_ENCODING

    @abstractmethod
    def read(self, file_path: str, **kwargs) -> Any:  # noqa: ANN003, ANN401
        """Read data from the specified file path."""
        pass

    @abstractmethod
    def write(
        self,
        data: Any,  # noqa: ANN401
        file_path: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:  # noqa: ANN003, ANN401
        """Write data to the specified file path."""
        pass

    # --- Common Helper Methods (can be part of the base class) ---
    def ensure_dir_exists(self, file_path: str) -> None:
        """Ensure the directory for the given file path exists."""
        dir_path = os.path.dirname(file_path)
        if dir_path:  # Only try to create if there's actually a directory part
            os.makedirs(dir_path, exist_ok=True)

    def check_file_exists(self, file_path: str) -> None:
        """Check if a file exists, raising FileNotFoundError if not."""
        if not Path(file_path).is_file():
            msg = f"Input file not found: {file_path}"
            raise FileNotFoundError(msg)
