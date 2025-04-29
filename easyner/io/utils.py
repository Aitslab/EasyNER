import os
import re


def extract_batch_index(batch_file: str) -> int:
    """
    Extract the batch index from a filename.

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
