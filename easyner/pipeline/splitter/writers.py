from abc import ABC, abstractmethod
import json
import os
import logging

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.writers")


class OutputWriterBase(ABC):
    @abstractmethod
    def write(self, articles, batch_idx, tokenizer_name):
        """
        Write processed articles to output

        Args:
            articles: Dictionary of processed articles
            batch_idx: Index of the current batch
            tokenizer_name: Name of the tokenizer used
        """
        pass


class JSONWriter(OutputWriterBase):
    def __init__(self, output_folder, output_file_prefix):
        self.output_folder = output_folder
        self.output_file_prefix = output_file_prefix
        os.makedirs(output_folder, exist_ok=True)
        logger.debug(
            f"Initialized JSONWriter with output folder: {output_folder}, prefix: {output_file_prefix}"
        )

    def write(self, articles, batch_idx, tokenizer_name):
        """Write articles to a JSON file"""
        output_file = f"{self.output_folder}/{self.output_file_prefix}_{tokenizer_name}-split-{batch_idx}.json"
        logger.debug(f"Writing {len(articles)} articles to {output_file}")

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(articles, indent=2, ensure_ascii=False))
            logger.debug(f"Successfully wrote {len(articles)} articles to {output_file}")
        except Exception as e:
            logger.error(f"Error writing to {output_file}: {e}", exc_info=True)
            raise

        # Return batch index and number of articles processed
        return batch_idx, len(articles)
