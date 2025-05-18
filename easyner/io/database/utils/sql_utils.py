"""SQL file utility functions for database operations."""

import hashlib
from pathlib import Path
from typing import Optional, Union


def read_sql_file(file_path: Union[str, Path]) -> str:
    """Read an SQL file and return its contents.

    This utility function can be used to read any SQL file content,
    not just schema files.

    Args:
        file_path: Path to the SQL file to read

    Returns:
        The contents of the SQL file as a string

    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an issue reading the file

    """
    try:
        path = Path(file_path) if isinstance(file_path, str) else file_path
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg) from e
    except Exception as e:
        msg = f"Error reading SQL file: {file_path}"
        raise OSError(msg) from e


def get_file_hash(
    file_path: Union[str, Path],
    algorithm: str = "sha256",
    buffer_size: int = 65536,
) -> str:
    """Calculate a hash of a file's contents.

    Args:
        file_path: Path to the file to hash
        algorithm: Hash algorithm to use (default: sha256)
        buffer_size: Buffer size for reading file chunks (default: 64KB)

    Returns:
        The hexadecimal digest of the file hash

    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an issue reading the file

    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    hash_obj = hashlib.new(algorithm)

    with open(path, "rb") as f:
        while chunk := f.read(buffer_size):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()
