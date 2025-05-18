import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import pubmed_parser as pp
from tqdm import tqdm

from easyner.pipeline.pubmed.utils import _resolve_path


class BasePubMedLoader(ABC):
    """Abstract base class for PubMed XML loaders."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        baseline: str,
        file_start: Optional[int] = None,
        file_end: Optional[int] = None,
    ) -> None:
        """Initialize the base PubMed loader.

        Args:
            input_path: Directory containing input XML files
            output_path: Directory where processed files will be written
            k: Baseline identifier used in filename parsing
            file_start: Optional start index for file range processing
            file_end: Optional end index for file range processing

        """
        # Resolve paths against project root if they're relative
        self.input_path = _resolve_path(input_path)
        self.output_path = _resolve_path(output_path)
        # Ensure k is a string
        self.baseline = str(baseline)
        self.file_start = file_start
        self.file_end = file_end

        # Validate file range if both are provided
        if self.file_start is not None and self.file_end is not None:
            if self.file_start > self.file_end:
                msg = f"file_start ({self.file_start}) cannot be greater than file_end ({self.file_end})"
                raise ValueError(msg)

        os.makedirs(self.output_path, exist_ok=True)

    def _get_input_files(self, input_path: str) -> list[str]:
        """Get input files using path objects for reliable path handling.

        Args:
            input_path: Directory containing the input files

        Returns:
            List of input file paths sorted by file number

        """
        # k is used for keyword to split the filename obtained from pubmed.
        # It's different for each annual baseline
        input_path_obj = Path(input_path)

        # Use Path's glob method which handles path separators correctly
        input_files = sorted(
            [str(p) for p in input_path_obj.glob("*.gz")],
            key=lambda x: int(
                os.path.splitext(os.path.basename(x))[0].split(self.baseline + "n")[-1][
                    :-4
                ],
            ),
        )

        # Filter files by range if specified
        if input_files and (self.file_start is not None or self.file_end is not None):
            filtered_files = []
            for file_path in input_files:
                try:
                    file_num = int(
                        os.path.splitext(os.path.basename(file_path))[0].split(
                            self.baseline + "n",
                        )[-1][:-4],
                    )

                    # Apply file_start filter if specified
                    if self.file_start is not None and file_num < self.file_start:
                        continue

                    # Apply file_end filter if specified
                    if self.file_end is not None and file_num > self.file_end:
                        continue

                    filtered_files.append(file_path)
                except (ValueError, IndexError):
                    # Skip files that don't match expected naming pattern
                    continue

            input_files = filtered_files
            print(
                f"After applying range filters (start={self.file_start}, end={self.file_end}): {len(input_files)} files",
            )

        # Add debug output for the number of files found
        print(f"Found {len(input_files)} XML files in {input_path}")
        if len(input_files) == 0:
            print(
                f"WARNING: No XML files found in {input_path} matching pattern '*.gz'",
            )
            print("Make sure the path exists and contains gzipped XML files.")
        return input_files

    @abstractmethod
    def _process_article_data(self, data: list[dict[str, Any]]) -> Any:
        """Process article data extracted from XML.

        Args:
            data: List of article data dictionaries from pubmed_parser

        Returns:
            Processed article data in implementation-specific format

        """
        pass

    @abstractmethod
    def _write_output(self, data: Any, input_file: str) -> None:
        """Write processed article data to output.

        Args:
            data: The processed article data
            input_file: Original input file path

        """
        pass

    def _load_xml(self, input_file: str) -> list[dict[str, Any]]:
        """Load XML file and parse using pubmed_parser.

        Args:
            input_file: Path to the input XML file

        Returns:
            List of article data dictionaries

        """
        return pp.parse_medline_xml(input_file, year_info_only=False)

    def run_loader(self) -> None:
        """Run the loader of PubMed files."""
        print(f"Starting to load PubMed files from {self.input_path}")
        input_files_list = self._get_input_files(self.input_path)

        # Add more debug information about the files being processed
        if len(input_files_list) > 0:
            print(f"Processing {len(input_files_list)} XML files")
            print(f"First file: {os.path.basename(input_files_list[0])}")
            print(f"Last file: {os.path.basename(input_files_list[-1])}")
        else:
            print("No files to process. Please check the input path and file pattern.")
            return

        # Use tqdm with the list directly for proper progress tracking
        for input_file in tqdm(input_files_list, desc="Processing files"):
            self.current_input_file = input_file
            data = self._load_xml(input_file)
            processed_data = self._process_article_data(data)
            self._write_output(processed_data, input_file)
