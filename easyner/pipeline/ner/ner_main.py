# coding=utf-8

import logging
import os
import torch
from glob import glob
from typing import List, Dict, Any, Tuple
from easyner.pipeline.ner.factory import NERProcessorFactory
from easyner.io.utils import (
    get_batch_file_index,
    filter_batch_files,
    get_batch_indices,
    check_for_duplicate_batch_indices,
)


def _find_files(input_dir: str) -> List[str]:
    """
    Find input files based on configuration.

    Returns:
    --------
    List[str]: List of input files
    """
    if not os.path.isdir(input_dir):
        logging.warning(f"Input directory does not exist: {input_dir}")
        return []

    input_file_list = glob(os.path.join(input_dir, "*.json"))

    if not input_file_list:
        logging.warning(f"No input files found in directory: {input_dir}")
        return []

    return input_file_list


class NERPipeline:
    """Main class for the NER pipeline that handles processing workflow."""

    def __init__(self, config: Dict[str, Any], cpu_limit: int = 1):
        """
        Initialize the NER pipeline.

        Parameters:
        -----------
        config: Dict[str, Any]
            Configuration for NER processing
        cpu_limit: int
            Maximum number of CPUs to use for multiprocessing

        Raises:
        -------
        KeyError: If required config keys are missing
        ValueError: If paths are invalid
        """
        # Validate required configuration
        required_keys = ["input_path", "output_path"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Required config key missing: {key}")

        # Store and validate input path
        self.input_path = config["input_path"]
        if not os.path.exists(self.input_path):
            raise ValueError(f"Input path does not exist: {self.input_path}")
        if not os.path.isdir(self.input_path):
            raise ValueError(
                f"Input path is not a directory: {self.input_path}"
            )

        # Store and validate output path
        self.output_path = config["output_path"]
        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise ValueError(
                    f"Cannot create output directory: {self.output_path}. {str(e)}"
                )
        elif not os.path.isdir(self.output_path):
            raise ValueError(
                f"Output path exists but is not a directory: {self.output_path}"
            )

        # Extract frequently used configuration values as class attributes
        self.batch_start_index = config.get("batch_start_index")
        self.batch_end_index = config.get("batch_end_index")
        self.batch_span_start, self.batch_span_end = (
            self._get_batch_span_from_env_or_config()
        )

        self.reprocess = config.get("reprocess", False)
        self.clear_old_results = config.get("clear_old_results", False)
        self.input_file_list = self._get_filtered_input_files(
            reprocess=self.reprocess
        )
        self.cpu_limit = cpu_limit

        # Store original config for any values not extracted as attributes
        # (This can be gradually phased out as you identify more attributes)
        self._config = config

        # Create the appropriate processor using the factory
        self.processor = NERProcessorFactory.create_processor(config)

    def _get_batch_span_from_env_or_config(self):
        """
        Get batch span from environment variables or configuration.

        Returns:
        --------
        batch_span_start: int, batch_span_end: int
            Start and end indices for batch processing
        """
        batch_span_start = os.environ.get("NER_ARTICLE_START")
        if batch_span_start is not None:
            try:
                batch_span_start = int(batch_span_start)
            except ValueError:
                logging.warning(
                    f"Invalid NER_ARTICLE_START value: {batch_span_start}"
                )
                batch_span_start = self.batch_start_index
        else:
            batch_span_start = self.batch_start_index

        batch_span_end = os.environ.get("NER_ARTICLE_END")
        if batch_span_end is not None:
            try:
                batch_span_end = int(batch_span_end)
            except ValueError:
                logging.warning(
                    f"Invalid NER_ARTICLE_END value: {batch_span_end}"
                )
                batch_span_end = self.batch_end_index
        else:
            batch_span_end = self.batch_end_index

        return batch_span_start, batch_span_end

    def _get_filtered_input_files(self, reprocess: bool = False) -> List[str]:
        """
        Find and sort input files based on configuration.
        Can be overridden by environment variables NER_ARTICLE_START and NER_ARTICLE_END.

        Returns:
        --------
        List[str]: Sorted list of input files to process
        """

        try:
            input_files = _find_files(self.input_path)

            if not input_files:
                raise ValueError(
                    f"No input files found in directory: {self.input_path}"
                )

            input_files = sorted(input_files, key=get_batch_file_index)

            check_for_duplicate_batch_indices(input_files)

            processed_batches = (
                None if reprocess else get_batch_indices(self.output_path)
            )

            filtered_files = filter_batch_files(
                input_files,
                start=self.batch_span_start,
                end=self.batch_span_end,
                exclude_batches=processed_batches,
            )

            return filtered_files
        except Exception as e:
            logging.error(
                f"Unexpected error while processing input files: {e}"
            )
            raise

    def run(self) -> None:
        """
        Main entry point for the NER pipeline.
        - Sets up output directory
        - Discovers and filters files
        - Delegates processing to the appropriate processor
        """
        print("----Starting NER pipeline----")

        # Set up output directory
        if self.clear_old_results:
            from easyner.io.utils import _remove_all_files_from_dir

            _remove_all_files_from_dir(self.output_path)
        else:
            os.makedirs(self.output_path, exist_ok=True)

        # Let the processor handle the dataset in the most appropriate way
        device = torch.device(0 if torch.cuda.is_available() else "cpu")
        self.processor.process_dataset(self.input_file_list, device=device)

        print("----NER pipeline processing complete----")


# For backward compatibility
def run_ner_module(ner_config: Dict[str, Any], cpu_limit: int) -> None:
    """
    Legacy entry point for the NER pipeline.

    Parameters:
    -----------
    ner_config: Dict[str, Any]
        Configuration for NER processing
    cpu_limit: int
        Maximum number of CPUs to use for multiprocessing
    """
    pipeline = NERPipeline(ner_config, cpu_limit)
    pipeline.run()


if __name__ == "__main__":
    pass
