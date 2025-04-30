import logging
import os
import re
from typing import List, Optional


def get_batch_number(filename):
    # Use get_batch_number for sorting file lists when you want graceful fallback
    # Extract batch numbers using regex pattern for "batch-X.json" format
    pattern = re.compile(r"batch-(\d+)\.json$")

    match = pattern.search(os.path.basename(filename))
    if match:
        return int(match.group(1))
    # Fall back to lexicographical for non-matching files
    return os.path.basename(filename)


def get_batch_file_index(batch_file: str) -> int:
    """
    Extract the batch index from a filename.
    Use case when you need strict validation and errors for malformed filenames

    Parameters:
    -----------
    batch_file: str
        Path to the batch file

    Returns:
    --------
    int: The extracted batch index

    Raises:
    -------
    ValueError: If the filename doesn't contain a numeric index
    """
    filename = os.path.basename(batch_file)

    # Get the name part (before any extensions)
    name_part = filename.split(".")[0]

    # Find the last series of digits at the end of the name part
    match = re.search(r"(\d+)$", name_part)

    if match:
        # If match is preceded by numeric characters anywhere in the name raise warning about ambigous batch filename
        if re.search(r"\d", name_part[: -len(match.group(1))]):
            print(
                f"Warning: Ambiguous batch filename '{filename}'. "
                "Batch number should be at the end of the filename."
            )

        # Match is not at the end of the name part
        if match.start() != len(name_part) - len(match.group(1)):
            raise ValueError(
                f"Batch filename '{filename}' contains non-numeric characters after the batch number."
            )
        return int(match.group(1))

    print(f"Error extracting index from {batch_file}")
    raise ValueError(
        "Batch filenames must contain a pure numeric index before the extension"
    )


def filter_batch_files(
    file_list,
    start: Optional[int] = None,
    end: Optional[int] = None,
    exclude_batches: Optional[List[int]] = None,
):
    """
    Filter files based on index range.

    Parameters:
    -----------
    list_files: list
        List of file paths to filter
    start: int
        Starting index (inclusive)
    end: int
        Ending index (inclusive)

    Returns:
    --------
    list: Filtered list of file paths
    """
    if not file_list:
        raise ValueError(
            "The file list is empty. Cannot filter an empty list."
        )
    if start is not None and end is not None:
        if start > end:
            raise ValueError("Start index cannot be greater than end index.")
    if exclude_batches == []:
        logging.warning(
            "Empty exclude_batches list provided. No batches will be excluded."
        )
        exclude_batches = None  # Reset to None to avoid confusion

    filtered_list_files = []
    for file in file_list:
        try:
            batch_idx = extract_batch_index(file)
            if exclude_batches and batch_idx in exclude_batches:
                continue

            if start is not None:
                if batch_idx < start:
                    continue
            if end is not None:
                if batch_idx > end:
                    continue

            filtered_list_files.append(file)
        except (ValueError, IndexError):
            raise ValueError(
                f"Batch filenames must contain numeric indices when filtering: {file}"
            )

    return filtered_list_files


def _remove_all_files_from_dir(dir_path: str):
    """
    Keep the directory but clear its contents
    Example usage: Clear old results from the output directory.

    Parameters:
    -----------
    dir_path: str
        Path to the output directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return

    try:
        files_to_remove = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
        ]

        if files_to_remove:
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                except (PermissionError, OSError) as e:
                    print(f"Warning: Could not remove {file_path}: {e}")

            print(f"Cleared {len(files_to_remove)} files from {dir_path}")
        else:
            print(f"No files to clear in {dir_path}")

    except OSError as e:
        print(f"Error while clearing files from {dir_path}: {e}")
        raise
