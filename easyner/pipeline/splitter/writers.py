from abc import ABC, abstractmethod
import json
import os
import logging
import gc  # Add garbage collection module

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
        """Write articles to a JSON file with improved memory efficiency"""
        output_file = f"{self.output_folder}/{self.output_file_prefix}_{tokenizer_name}-split-{batch_idx}.json"
        logger.debug(f"Writing {len(articles)} articles to {output_file}")
        article_count = len(articles)

        try:
            # Use orjson for better performance if available
            try:
                import orjson

                with open(output_file, "wb") as f:
                    # Serialize directly to file to avoid double memory usage
                    f.write(orjson.dumps(articles, option=orjson.OPT_INDENT_2))

                logger.debug(
                    f"Successfully wrote {article_count} articles to {output_file} using orjson"
                )
            except ImportError:
                # Stream writing with regular json to reduce peak memory usage
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("{\n")
                    for i, (article_id, article_data) in enumerate(articles.items()):
                        json_str = json.dumps(article_data, ensure_ascii=False, indent=2)
                        if i > 0:
                            f.write(",\n")
                        f.write(f'  "{article_id}": {json_str}')
                    f.write("\n}")

                logger.debug(f"Successfully wrote {article_count} articles to {output_file}")

        except Exception as e:
            logger.error(f"Error writing to {output_file}: {e}", exc_info=True)
            raise

        # Clear memory before returning
        del articles
        gc.collect()

        return batch_idx, article_count
