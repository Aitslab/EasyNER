"""IO handlers for reading and writing data in various formats."""

import concurrent.futures
import json
import logging
import os
from typing import Any, NoReturn

from .base import IOHandler

# import pyarrow.parquet as pq # Import when implementing
# import duckdb # Import when implementing


logger = logging.getLogger(__name__)


class JsonHandler(IOHandler):
    """Handles reading and writing JSON files."""

    EXTENSION = "json"

    def read(
        self,
        file_path: str,
        timeout: int = 180,
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:
        """Read data from a JSON file
        If file > 10MB → Try mmap + orjson → Success? Return result : Continue
        ↓
        Try direct orjson → Success? Return result : Continue
        ↓
        Use ThreadPoolExecutor to call _load_json with timeout
        ↓
        _load_json uses standard json.load.
        """
        self.check_file_exists(file_path)

        # Get file size for logging
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        # Use memory mapping for large files (>10MB)
        if file_size_mb > 10:
            try:
                import mmap

                import orjson

                with open(file_path, "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    try:
                        view = memoryview(mm)
                        return orjson.loads(view)
                    finally:
                        del view
            except Exception as e:
                logger.warning(
                    f"Memory mapping failed, falling back to standard method: {e}",
                )

        # Existing orjson implementation for smaller files or if memory mapping fails
        try:
            import orjson

            try:
                with open(file_path, "rb") as f:
                    return orjson.loads(f.read())
            except Exception as e:
                logger.warning(
                    f"orjson failed, falling back to standard json: {e}",
                )
        except ImportError:
            pass

        # Use ThreadPoolExecutor for timeout protection
        # This is safe in multiprocessing as each process has its
        # own Python interpreter and thread pool, so there's no
        # interference between processes
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._load_json, file_path)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as e:
                error_msg = (
                    "JSON parsing timed out after"
                    f"{timeout}s for file: {file_path} ({file_size_mb:.2f}MB)"
                )
                logger.error(error_msg)
                raise TimeoutError(error_msg) from e
            except Exception as e:
                logger.error(f"Error reading JSON file {file_path}: {e}")
                raise

    def _load_json(self, file_path: str) -> Any:
        """Helper method to load JSON within executor with error handling."""  # noqa: D401
        # Check for empty file
        if os.path.getsize(file_path) == 0:
            msg = f"Empty file detected: no data in {file_path}"
            raise ValueError(msg)

        with open(file_path, encoding=self.encoding) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                # Wrap JSON decode errors with more descriptive messages
                msg = f"Error decoding JSON file in {file_path}: {str(e)}"
                raise ValueError(
                    msg,
                ) from e

    from typing import Optional

    def write(
        self,
        data,
        file_path: str,
        indent: Optional[int] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Write data to a JSON file with orjson optimization when available."""
        self.ensure_dir_exists(file_path)

        try:
            # First try orjson if available (much faster)
            try:
                import orjson

                # Configure orjson options based on parameters
                options = 0
                if indent is not None:
                    options |= orjson.OPT_INDENT_2

                # orjson returns bytes, not string
                json_bytes = orjson.dumps(data, option=options)

                # Write bytes directly to file
                with open(file_path, "wb") as f:
                    f.write(json_bytes)

                logger.debug(f"Successfully wrote data to {file_path} using orjson")
                return

            except ImportError:
                # orjson not available, use standard json
                pass
            except Exception as e:
                logger.warning(
                    f"orjson writing failed, falling back to standard json: {e}",
                )
                # Fall back to standard json
                pass

            # Standard json as fallback
            with open(file_path, "w", encoding=self.encoding) as f:
                json.dump(data, f, indent=indent)

            logger.debug(f"Successfully wrote data to {file_path} using standard json")

        except Exception as e:
            error_msg = f"Error writing JSON file {file_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
