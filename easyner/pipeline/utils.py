import os
import re
import logging

logger = logging.getLogger(__name__)


def get_batch_index_from_filename(filename: str) -> int:
    """
    Extracts the trailing number (batch index) from a filename.
    Assumes a pattern like 'prefix-123.ext' or 'prefix_123.ext'.
    Only extracts and returns integer values.

    Args:
        filename: The input filename
        (e.g., 'data-001.json', '/path/to/result-123.parquet').

    Returns:
        The extracted batch index as an integer.

    Raises:
        ValueError: If no numeric index can be reliably extracted.
    """
    basename = os.path.basename(filename)
    # Regex to find sequences of digits.
    matches = re.findall(r"\d+", basename)

    if not matches:
        # No numeric parts found at all
        logger.error(
            f"Could not find any numeric part in filename: {basename}"
        )
        raise ValueError(
            f"Could not find batch index number in filename: {basename}"
        )

    # Assume the last sequence of digits found is the batch index
    index_str = matches[-1]

    try:
        return int(index_str)
    except ValueError:
        # This path should be very unlikely if re.findall(r'\d+') succeeded,
        # but handle defensively. It means the found digits couldn't form an
        # int.
        logger.error(
            f"Could not convert extracted numeric part '{index_str}' to int in"
            f" filename: {basename}"
        )
        # Use a message indicating conversion failure of a numeric part
        raise ValueError(
            f"Could not convert extracted number '{index_str}' to int in"
            f" filename: {basename}"
        )


def construct_output_path(
    output_dir: str,
    file_prefix: str,
    input_filename: str,
    output_extension: str,
) -> str:
    """
    Constructs an output file path based on the batch index from an input
    filename.

    Args:
        output_dir: The directory for the output file.
        file_prefix: The desired prefix for the output filename (e.g.,
            'ner_results').
        input_filename: The input filename from which to extract the batch
            index.
        output_extension: The desired extension for the output file (e.g.,
            'json', 'parquet').

    Returns:
        The fully constructed output file path.
    """
    try:
        batch_index = get_batch_index_from_filename(
            input_filename
        )  # Expects int or ValueError
        # Ensure extension doesn't start with a dot if passed that way
        output_extension = output_extension.lstrip(".")
        output_basename = f"{file_prefix}-{batch_index}.{output_extension}"
        # Ensure the output directory exists
        abs_output_dir = os.path.abspath(output_dir)
        try:
            os.makedirs(abs_output_dir, exist_ok=True)
        except PermissionError as pe:
            logger.error(
                f"Permission denied creating directory {abs_output_dir}: {pe}"
            )
            raise  # Re-raise the permission error
        except Exception as e:
            logger.error(f"Error creating directory {abs_output_dir}: {e}")
            raise  # Re-raise other directory creation errors

        return os.path.join(
            abs_output_dir, output_basename
        )  # Return absolute path
    except ValueError as e:  # Catch errors from get_batch_index_from_filename
        logger.error(
            "Failed to construct output path based on input "
            f"'{input_filename}': {e}"
        )
        raise  # Re-raise the exception after logging
    except Exception as e:
        logger.error(
            "Unexpected error constructing output path for "
            f"'{input_filename}': {e}"
        )
        raise
