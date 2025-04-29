# coding=utf-8

import os
import torch
from glob import glob
from typing import List, Dict, Any
from easyner.pipeline.ner.factory import NERProcessorFactory


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

    def _get_input_files_sorted(self) -> List[str]:
        """
        Find and sort input files based on configuration.

        Returns:
        --------
        List[str]: Sorted list of input files to process
        """
        input_file_list = sorted(
            glob(f'{self.config["input_path"]}*.json'),
            key=lambda x: int(
                os.path.splitext(os.path.basename(x))[0].split("-")[-1]
            ),
        )

        # Apply file range filtering if configured
        if "article_limit" in self.config and isinstance(
            self.config["article_limit"], list
        ):
            from easyner.io.utils import filter_files

            start = self.config["article_limit"][0]
            end = self.config["article_limit"][1]

            input_file_list = filter_files(input_file_list, start, end)
            print(f"Processing articles in range {start} to {end}")

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
        if self.config.get("clear_old_results", True):
            from easyner.io.utils import _remove_all_files_from_dir

            _remove_all_files_from_dir(self.config["output_path"])
        else:
            os.makedirs(self.config["output_path"], exist_ok=True)

        # Get sorted input files
        input_file_list = self._get_input_files_sorted()

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
