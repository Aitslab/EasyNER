import gc
import logging
import time
from abc import ABC, abstractmethod
from glob import glob
from multiprocessing import current_process

from easyner.io import get_io_handler
from easyner.pipeline.utils import get_batch_index_from_filename

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.loaders")


class DataLoaderBase(ABC):
    """Abstract base class for data loaders in the EasyNER pipeline.

    Methods
    -------
    load_data() -> dict | list
        Abstract method to load data; must be implemented by subclasses.

    """

    @abstractmethod
    def load_data(self) -> dict | list:
        pass


class StandardLoader(DataLoaderBase):
    def __init__(self, input_path, io_format="json"):
        self.input_path = input_path
        self.io_format = io_format
        logger.debug(
            f"Initialized StandardLoader with input path: {input_path}, "
            f"format: {io_format}",
        )

    def load_data(self) -> dict:
        logger.info(f"Loading data from {self.input_path}")
        try:
            # Use IO handler to read data
            io_handler = get_io_handler(self.io_format)
            data = io_handler.read(self.input_path)
            logger.info(
                f"Successfully loaded {len(data)} articles from " f"{self.input_path}",
            )
            return data
        except Exception as e:
            logger.error(
                f"Error loading data from {self.input_path}: {e}",
                exc_info=True,
            )
            raise


class PubMedLoader(DataLoaderBase):
    """Loader for pre-batched PubMed files.

    Supporting file selection by index range and batch loading.

    Attributes
    ----------
    input_folder : str
        Path to the folder containing PubMed batch files.
    limit : str or list[int]
        "ALL" to load all files, or a list [start, end] to select a range.
    key : str
        Key pattern to match files in the folder.
    io_format : str
        Format of the files (e.g., "json").

    Methods
    -------
    load_data() -> list
        Loads file paths of PubMed batches based on the configured limit.
    load_batch(file_path)
        Loads a single batch file with memory optimization.
    get_batch_index(input_file)
        Extracts the batch index from a filename.

    """

    def __init__(
        self,
        input_folder: str,
        limit: str | list[int] = "ALL",
        key: str = "n",
        io_format: str = "json",
    ) -> None:
        self.input_folder = input_folder
        self.limit = limit
        self.key = key
        self.io_format = io_format
        logger.debug(
            f"Initialized PubMedLoader with input folder: {input_folder}, "
            f"limit: {limit}, key: {key}, format: {io_format}",
        )

    def load_data(self) -> list:
        """Load pre-batched PubMed files based on configured limits."""
        logger.info(f"Loading PubMed data from {self.input_folder}")
        io_handler = get_io_handler(self.io_format)
        pattern = f"{self.input_folder}/*{self.key}*.{io_handler.EXTENSION}"
        all_files = sorted(glob(pattern))

        if not all_files:
            logger.warning(
                f"No PubMed files found matching pattern: {pattern}",
            )
            return []

        logger.info(f"Found {len(all_files)} PubMed files")

        if self.limit == "ALL":
            logger.info("Processing all PubMed files")
            return all_files
        elif isinstance(self.limit, list) and len(self.limit) == 2:
            start, end = self.limit
            selected_files = all_files[start:end]
            logger.info(
                f"Processing PubMed files from index {start} to {end} "
                f"({len(selected_files)} files)",
            )
            return selected_files
        else:
            logger.warning(
                f"Invalid limit format: {self.limit}, defaulting to ALL",
            )
            return all_files

    def load_batch(self, file_path):
        """Load a single batch file with memory optimization."""
        process_id = current_process().name
        logger.debug(f"[{process_id}] Loading batch file: {file_path}")
        start_time = time.time()

        try:
            # Use IO handler to read data
            io_handler = get_io_handler(self.io_format)
            data = io_handler.read(file_path)

            elapsed = time.time() - start_time
            logger.debug(
                f"[{process_id}] Successfully loaded {len(data)} articles "
                f"from {file_path} "
                f"in {elapsed:.2f}s",
            )

            # Explicitly trigger garbage collection
            gc.collect()

            return data

        except Exception as e:
            logger.error(
                f"[{process_id}] Error loading batch file {file_path}: {e}",
                exc_info=True,
            )
            raise

    def get_batch_index(self, input_file: str) -> int:
        """Extract batch index from filename using the pipeline utility function."""
        try:
            # Use the centralized utility function
            return get_batch_index_from_filename(input_file)
        except ValueError as e:  # Catch specific error from utility
            logger.error(
                f"Error extracting batch index from {input_file}: {e}",
                exc_info=False,
            )  # Log less verbosely
            # Decide on fallback behavior - returning 0 might still cause
            # issues. Consider raising the error or returning None and
            # handling it upstream. For now, keeping the previous behavior:
            return 0
        except Exception as e:
            logger.error(
                f"Unexpected error extracting batch index from {input_file}: " f"{e}",
                exc_info=True,
            )
            return 0
