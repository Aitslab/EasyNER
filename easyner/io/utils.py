import os
import re


def get_batch_number(filename):
    # Use get_batch_number for sorting file lists when you want graceful fallback
    # Extract batch numbers using regex pattern for "batch-X.json" format
    pattern = re.compile(r"batch-(\d+)\.json$")

    match = pattern.search(os.path.basename(filename))
    if match:
        return int(match.group(1))
    # Fall back to lexicographical for non-matching files
    return os.path.basename(filename)


def extract_batch_index(batch_file: str) -> int:
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
    regex = re.compile(r"\d+")
    try:
        return int(regex.findall(os.path.basename(batch_file))[-1])
    except (IndexError, ValueError) as e:
        print(f"Error extracting index from {batch_file}")
        raise ValueError(f"Batch filenames must contain numeric indices: {e}")


def filter_files(list_files, start, end):
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
    filtered_list_files = []
    for file in list_files:
        file_idx = int(
            os.path.splitext(os.path.basename(file))[0].split("-")[-1]
        )
        if file_idx >= start and file_idx <= end:
            filtered_list_files.append(file)

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
