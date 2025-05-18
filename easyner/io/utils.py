import logging
import os
import re
from typing import List, Optional
from glob import glob


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
        # If match is preceded by numeric characters anywhere in the name raise
        # warning about ambigous batch filename
        if re.search(r"\d", name_part[: -len(match.group(1))]):
            logging.warning(
                f"Ambiguous batch filename '{filename}'. "
                "Batch number should be at the end of the filename."
            )

        # Match is not at the end of the name part
        if match.start() != len(name_part) - len(match.group(1)):
            raise ValueError(
                f"Batch filename '{filename}' contains non-numeric "
                "characters after the batch number."
            )
        return int(match.group(1))

    logging.error(f"Error extracting index from {batch_file}")
    raise ValueError(
        "Batch filenames must contain a pure numeric index before"
        " the extension"
    )


def filter_batch_files(
    file_list: List[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
    exclude_batches: Optional[List[int]] = None,
) -> List[str]:
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
    exclude_batches: Optional[List[int]]
        List of batch indices to exclude

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

    filtered_list_files: List[str] = []
    for file in file_list:
        try:
            batch_idx = get_batch_file_index(file)
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
                f"Batch filenames must contain numeric indices "
                f"when filtering: {file}"
            )

    # Add logging here to report filtering results
    logging.info(
        f"Applied batch filtering: start={start}, end={end}. "
        f"{len(filtered_list_files)} files remain out of {len(file_list)}."
    )
    if exclude_batches:
        logging.info(
            f"Excluded {len(exclude_batches)} specific batch indices."
        )

    return filtered_list_files


def get_batch_indices(output_path: str) -> List[int]:
    """
    Get list of batch indices from files in the specified directory.

    Parameters:
    -----------
    output_path: str
        Path to the directory containing batch files

    Returns:
    --------
    List[int]: List of processed batch indices
    """
    if not os.path.isdir(output_path):
        logging.warning(f"Directory does not exist: {output_path}")
        return []

    output_files = glob(os.path.join(output_path, "*.json"))
    batch_indices = [
        get_batch_file_index(os.path.basename(f))
        for f in output_files
        if os.path.isfile(f)
    ]
    return batch_indices


def check_for_duplicate_batch_indices(file_list: List[str]) -> None:
    """
    Check for duplicate batch indices in file list and raise error if found.

    Parameters:
    -----------
    file_list: List[str]
        List of files to check

    Raises:
    -------
    ValueError: If duplicate batch indices are found
    """
    if not file_list:
        return

    # Get all batch indices
    batch_indices = [get_batch_file_index(f) for f in file_list]

    # Find duplicates
    seen = set()
    duplicates = {}

    for i, idx in enumerate(batch_indices):
        if idx in seen:
            if idx not in duplicates:
                duplicates[idx] = []
            duplicates[idx].append(file_list[i])
        else:
            seen.add(idx)

    # Raise error if duplicates found
    if duplicates:
        duplicate_info = "\n".join(
            [f"Index {idx}: {files}" for idx, files in duplicates.items()]
        )
        raise ValueError(
            f"Duplicate batch indices found in input files:\n{duplicate_info}"
        )


def _remove_all_files_from_dir(dir_path: str) -> None:
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


def safe_batch_file_index_sort(file_list: List[str]) -> List[str]:
    """
    Sort files by batch index with fallback to lexicographical sorting.

    Parameters:
    -----------
    file_list: List[str]
        List of files to sort

    Returns:
    --------
    List[str]: Sorted list of files
    """
    if not file_list:
        return []

    # First try to extract batch indices for all files
    try:
        return sorted(file_list, key=get_batch_file_index)
    except ValueError:
        # Fall back to lexicographical sorting if batch indices can't be
        # extracted
        logging.warning(
            "Couldn't sort by batch index. Using lexicographical sort."
        )
        return sorted(file_list)
