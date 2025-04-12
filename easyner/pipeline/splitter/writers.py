from abc import ABC, abstractmethod
import json
import os


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

    def write(self, articles, batch_idx, tokenizer_name):
        """Write articles to a JSON file"""
        output_file = f"{self.output_folder}/{self.output_file_prefix}_{tokenizer_name}-split-{batch_idx}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(articles, indent=2, ensure_ascii=False))
        # Return batch index and number of articles processed
        return batch_idx, len(articles)
