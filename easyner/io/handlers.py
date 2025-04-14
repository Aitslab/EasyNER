"""IO handlers for reading and writing data in various formats."""

import concurrent.futures
import json
import logging
import os
from typing import Any, NoReturn

# import pyarrow.parquet as pq # Import when implementing
# import duckdb # Import when implementing
from .base import IOHandler

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
        """Read data from a JSON file with timeout protection."""
        self.check_file_exists(file_path)

        # Get file size for logging
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        # First try orjson if available (much faster)
        try:
            import orjson

            try:
                with open(file_path, "rb") as f:
                    return orjson.loads(f.read())
            except Exception as e:
                logger.warning(f"orjson failed, falling back to standard json: {e}")
                # Fall back to standard json with timeout
        except ImportError:
            pass  # orjson not available, use standard json

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


class ParquetHandler(IOHandler):
    """Handles reading and writing Parquet files using DuckDB."""

    # Requires duckdb installation

    EXTENSION = "parquet"

    def read(self, file_path: str, **kwargs: Any) -> NoReturn:  # noqa: ANN401
        """Read data from a Parquet file."""
        # Raise NotImplementedError before file existence check for testing purposes
        msg = "Parquet reading not yet implemented."
        raise NotImplementedError(msg)
        # When implementing, uncomment the following:
        # self.check_file_exists(file_path)
        # --- Placeholder for Parquet reading using DuckDB ---
        # try:
        #     import duckdb
        #     conn = duckdb.connect(database=':memory:')
        #     # Read parquet file into a DuckDB table
        #     result = conn.execute(f"SELECT * FROM read_parquet('{file_path}')").fetchall()  # noqa: E501
        #     # Convert to dictionary structure expected by the pipeline
        #     data = self._convert_duckdb_result_to_dict(result)
        #     conn.close()
        #     return data
        # except ImportError:
        #     raise RuntimeError("DuckDB not installed. Install with 'pip install duckdb'")  # noqa: E501
        # except Exception as e:
        #     raise RuntimeError(f"Error reading Parquet file {file_path}: {e}") from e

    def write(self, data, file_path: str, **kwargs: Any) -> NoReturn:  # noqa: ANN401
        """Write data to a Parquet file."""
        # Raise NotImplementedError before directory creation for testing purposes
        msg = "Parquet writing not yet implemented."
        raise NotImplementedError(msg)
        # When implementing, uncomment the following:
        # self.ensure_dir_exists(file_path)
        # --- Placeholder for Parquet writing using DuckDB ---
        # try:
        #     import duckdb
        #     import pandas as pd
        #     conn = duckdb.connect(database=':memory:')
        #     # Convert nested dict to a flattened format suitable for DataFrame
        #     flattened_data = self._flatten_dict_for_parquet(data)
        #     df = pd.DataFrame(flattened_data)
        #     # Register DataFrame as a table in DuckDB
        #     conn.register('temp_table', df)
        #     # Write the table to a Parquet file
        #     conn.execute(f"COPY temp_table TO '{file_path}' (FORMAT PARQUET)")
        #     conn.close()
        # except ImportError:
        #     raise RuntimeError("Required packages not installed. Install with 'pip install duckdb pandas'")  # noqa: E501
        # except Exception as e:
        #     raise RuntimeError(f"Error writing Parquet file {file_path}: {e}") from e

    # --- Helper methods for data structure conversion (Needed for Parquet) ---
    # def _convert_duckdb_result_to_dict(self, result):
    #     """Converts a DuckDB query result to the nested article dictionary structure."""  # noqa: E501
    #     # Implementation depends on the exact schema used in the Parquet file
    #     pass

    # def _flatten_dict_for_parquet(self, data):
    #     """Flattens the nested article dictionary for storage in Parquet."""
    #     # Implementation depends on the desired Parquet schema
    #     pass
