from abc import ABC, abstractmethod
import os
import logging
import gc  # Add garbage collection module

from easyner.io import get_io_handler

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
    def __init__(self, output_folder, output_file_prefix, io_format="json"):
        self.output_folder = output_folder
        self.output_file_prefix = output_file_prefix
        self.io_format = io_format
        os.makedirs(output_folder, exist_ok=True)
        logger.debug(
            f"Initialized JSONWriter with output folder: {output_folder}, prefix: {output_file_prefix}, format: {io_format}"
        )

    def write(self, articles, batch_idx, tokenizer_name):
        """Write articles to a file using the IO module"""
        # Get the appropriate IO handler
        io_handler = get_io_handler(self.io_format)

        # Construct output filename
        output_file = f"{self.output_folder}/{self.output_file_prefix}_{tokenizer_name}-split-{batch_idx}.{io_handler.EXTENSION}"
        logger.debug(f"Writing {len(articles)} articles to {output_file}")
        article_count = len(articles)

        try:
            # Use IO handler to write data
            if self.io_format == "json":
                # For JSON, we can specify indentation for readability
                io_handler.write(articles, output_file, indent=2)
            else:
                # For other formats like Parquet, use default options
                io_handler.write(articles, output_file)

            logger.debug(
                f"Successfully wrote {article_count} articles to {output_file} using {self.io_format} format"
            )

        except Exception as e:
            logger.error(f"Error writing to {output_file}: {e}", exc_info=True)
            raise

        # Clear memory before returning
        del articles
        gc.collect()

        return batch_idx, article_count
