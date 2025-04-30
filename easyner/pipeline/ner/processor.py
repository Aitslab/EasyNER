from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from easyner import util
from easyner.io.factory import get_io_handler
from easyner.io.utils import get_batch_file_index


class NERProcessor(ABC):
    """Abstract base class for NER processors."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the NER processor.

        Parameters
        ----------
        config: Dict[str, Any]
            Configuration settings for the processor

        """
        self.config = config
        self.output_template = "{output_path}/{output_file_prefix}-{batch_index}.json"

    @abstractmethod
    def process_dataset(
        self,
        input_files: list[str],
        device: Any = None,
    ) -> None:
        """Process the entire dataset of files with named entity recognition.

        Parameters
        ----------
        input_files: List[str]
            List of input file paths to process
        device: Any, optional
            Device to use for processing

        """
        pass

    def _build_output_filepath(self, batch_index: int) -> str:
        """Generate the output file path based on the configuration and batch index.

        Parameters
        ----------
        batch_index: int
            The batch index to include in the filename

        Returns
        -------
        str: The formatted output file path

        """
        return self.output_template.format(
            output_path=self.config["output_path"],
            output_file_prefix=self.config["output_file_prefix"],
            batch_index=batch_index,
        )

    def _read_batch_file(self, batch_file: str) -> tuple[list[dict], int]:
        """Read a batch file and extract its index.

        Parameters
        ----------
        batch_file: str
            Path to the batch file to read

        Returns
        -------
        Tuple[List[Dict], int]: Articles and batch index

        """
        articles = get_io_handler("json").read(batch_file)
        batch_index = get_batch_file_index(batch_file)
        return articles, batch_index

    def _save_processed_articles(
        self,
        articles: list[dict],
        batch_index: int,
    ) -> None:
        """Save processed articles to the appropriate output file.

        Parameters
        ----------
        articles: List[Dict]
            Processed articles to save
        batch_index: int
            Batch index for the output filename

        """
        output_file = self._build_output_filepath(batch_index)
        util.append_to_json_file(output_file, articles)
