from abc import ABC, abstractmethod
import json
import os
from glob import glob
import logging

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.loaders")


class DataLoaderBase(ABC):
    @abstractmethod
    def load_data(self):
        pass


class StandardLoader(DataLoaderBase):
    def __init__(self, input_path):
        self.input_path = input_path
        logger.debug(f"Initialized StandardLoader with input path: {input_path}")

    def load_data(self):
        logger.info(f"Loading data from {self.input_path}")
        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            logger.info(f"Successfully loaded {len(data)} articles from {self.input_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {self.input_path}: {e}", exc_info=True)
            raise


class PubMedLoader(DataLoaderBase):
    def __init__(self, input_folder, limit="ALL", key="n"):
        self.input_folder = input_folder
        self.limit = limit
        self.key = key
        logger.debug(
            f"Initialized PubMedLoader with input folder: {input_folder}, limit: {limit}, key: {key}"
        )

    def load_data(self):
        """Load pre-batched PubMed files based on configured limits"""
        logger.info(f"Loading PubMed data from {self.input_folder}")
        pattern = f"{self.input_folder}/*{self.key}*.json"
        all_files = sorted(glob(pattern))

        if not all_files:
            logger.warning(f"No PubMed files found matching pattern: {pattern}")
            return []

        logger.info(f"Found {len(all_files)} PubMed files")

        if self.limit == "ALL":
            logger.info("Processing all PubMed files")
            return all_files
        elif isinstance(self.limit, list) and len(self.limit) == 2:
            start, end = self.limit
            selected_files = all_files[start:end]
            logger.info(
                f"Processing PubMed files from index {start} to {end} ({len(selected_files)} files)"
            )
            return selected_files
        else:
            logger.warning(f"Invalid limit format: {self.limit}, defaulting to ALL")
            return all_files

    def load_batch(self, file_path):
        """Load a single batch file"""
        logger.debug(f"Loading batch file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug(f"Successfully loaded {len(data)} articles from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading batch file {file_path}: {e}", exc_info=True)
            raise

    def get_batch_index(self, input_file):
        """Extract batch index from filename"""
        try:
            index = int(os.path.splitext(os.path.basename(input_file))[0].split(self.key)[-1])
            return index
        except Exception as e:
            logger.error(f"Error extracting batch index from {input_file}: {e}", exc_info=True)
            return 0
