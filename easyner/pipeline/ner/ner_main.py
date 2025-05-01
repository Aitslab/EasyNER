# coding=utf-8

import os
import torch
from glob import glob
from typing import List, Dict, Any
from easyner.pipeline.ner.factory import NERProcessorFactory
from easyner.io.utils import get_batch_file_index


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
        """
        self.config = config
        self.config["cpu_limit"] = cpu_limit

        # Create the appropriate processor using the factory
        self.processor = NERProcessorFactory.create_processor(self.config)

    def _get_input_files_sorted(self, reprocess: bool = False) -> List[str]:
        """
        Find and sort input files based on configuration.
        Can be overridden by envionment variables
          - NER_ARTICLE_START
          - NER_ARTICLE_END

        Defaults to idempotent behavior and checks for matching batch numbers in output path. Input files with matching batch numbers are ignored.

        Returns:
        --------
        List[str]: Sorted list of input files to process
        """
        input_file_list = glob(f'{self.config["input_path"]}*.json')
        if not input_file_list:
            return []

        try:
            input_file_list = sorted(input_file_list, key=get_batch_file_index)
        except ValueError:
            # Fall back to lexicographical sorting if batch numbers can't be extracted
            print(
                "Warning: Could not extract batch numbers from all filenames. Falling back to lexicographical sorting."
            )
            input_file_list = sorted(input_file_list)

        if not reprocess:  # Check for existing processed files
            from easyner.io.utils import filter_batch_files

            output_files = glob(f'{self.config["output_path"]}*.json')
            processed_batches = [
                get_batch_file_index(os.path.basename(f))
                for f in output_files
                if os.path.isfile(f)
            ]
            input_file_list = filter_batch_files(
                input_file_list,
                start=None,
                end=None,
                exclude_batches=processed_batches,
            )

        # Check environment variables first, then config
        env_start = os.environ.get("NER_ARTICLE_START")
        env_end = os.environ.get("NER_ARTICLE_END")

        if env_start and env_end:
            # Environment variables override config
            from easyner.io.utils import filter_batch_files

            start = int(env_start)
            end = int(env_end)
            input_file_list = filter_batch_files(input_file_list, start, end)
            print(
                f"Processing articles in range {start} to {end} (from environment variables)"
            )
        elif "article_limit" in self.config and isinstance(
            self.config["article_limit"], list
        ):
            # Fall back to config if no environment variables
            from easyner.io.utils import filter_batch_files

            start = self.config["article_limit"][0]
            end = self.config["article_limit"][1]
            input_file_list = filter_batch_files(input_file_list, start, end)
            print(
                f"Processing articles in range {start} to {end} (from config)"
            )

        return input_file_list

    def run(self) -> None:
        """
        Main entry point for the NER pipeline.
        - Sets up output directory
        - Discovers and filters files
        - Delegates processing to the appropriate processor
        """
        print("----Starting NER pipeline----")

        # Set up output directory
        if self.config.get("clear_old_results", False):
            from easyner.io.utils import _remove_all_files_from_dir

            _remove_all_files_from_dir(self.config["output_path"])
        else:
            os.makedirs(self.config["output_path"], exist_ok=True)

        # Get sorted input files
        reprocess = self.config.get("reprocess", False)
        input_file_list = self._get_input_files_sorted(reprocess=reprocess)

        # Skip processing if no input files are found
        if not input_file_list:
            print("No input files found. Skipping processing.")
            return

        # Let the processor handle the dataset in the most appropriate way
        device = torch.device(0 if torch.cuda.is_available() else "cpu")
        self.processor.process_dataset(input_file_list, device=device)

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
