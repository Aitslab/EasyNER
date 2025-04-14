import pytest
import os
import tempfile
import json
import shutil
from pathlib import Path

from easyner.io import get_io_handler
from easyner.pipeline.splitter.writers import JSONWriter
from easyner.pipeline.splitter.loaders import StandardLoader, PubMedLoader


class TestSplitterIOIntegration:
    """Integration tests for splitter package with IO module."""

    @pytest.fixture
    def sample_article_data(self):
        """Generate sample article data for testing."""
        return {
            "12345": {
                "title": "Sample Article Title",
                "text": "This is a sample article text. It contains multiple sentences. Testing is important.",
            },
            "67890": {
                "title": "Another Sample Article",
                "abstract": "This is a sample abstract. It has sentences too.",
            },
        }

    @pytest.fixture
    def sample_processed_data(self):
        """Generate sample processed data (with sentences split)."""
        return {
            "12345": {
                "title": "Sample Article Title",
                "sentences": [
                    {"text": "This is a sample article text."},
                    {"text": "It contains multiple sentences."},
                    {"text": "Testing is important."},
                ],
            },
            "67890": {
                "title": "Another Sample Article",
                "sentences": [
                    {"text": "This is a sample abstract."},
                    {"text": "It has sentences too."},
                ],
            },
        }

    def test_json_writer_with_io_module(self, sample_processed_data):
        """Test that JSONWriter correctly uses the IO module to write data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize JSONWriter with json io_format
            writer = JSONWriter(
                output_folder=tmp_dir, output_file_prefix="test_split", io_format="json"
            )

            # Write processed data
            batch_idx = 123
            tokenizer_name = "TestTokenizer"
            writer.write(sample_processed_data, batch_idx, tokenizer_name)

            # Check that file was created with correct name
            output_file = os.path.join(
                tmp_dir, f"test_split_{tokenizer_name}-split-{batch_idx}.json"
            )
            assert os.path.exists(output_file)

            # Read back the data using IO module
            io_handler = get_io_handler("json")
            read_data = io_handler.read(output_file)

            # Verify data integrity
            assert read_data == sample_processed_data
            assert "12345" in read_data
            assert "67890" in read_data
            assert len(read_data["12345"]["sentences"]) == 3
            assert len(read_data["67890"]["sentences"]) == 2

    def test_standard_loader_with_io_module(self, sample_article_data):
        """Test that StandardLoader correctly uses the IO module to read data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a sample input file
            input_file = os.path.join(tmp_dir, "sample_input.json")
            with open(input_file, "w") as f:
                json.dump(sample_article_data, f)

            # Initialize StandardLoader
            loader = StandardLoader(input_path=input_file, io_format="json")

            # Load the data
            loaded_data = loader.load_data()

            # Verify data integrity
            assert loaded_data == sample_article_data
            assert "12345" in loaded_data
            assert "67890" in loaded_data
            assert loaded_data["12345"]["title"] == "Sample Article Title"
            assert loaded_data["67890"]["title"] == "Another Sample Article"

    def test_pubmed_loader_batch_index_extraction(self):
        """Test that PubMedLoader correctly extracts batch indices using the IO module."""
        # Initialize PubMedLoader
        loader = PubMedLoader(input_folder="dummy_folder", io_format="json")

        # Test batch index extraction
        assert loader.get_batch_index("/path/to/pubmed-123.json") == 123
        assert loader.get_batch_index("pubmed-n456.json") == 456
        assert loader.get_batch_index("/data/pubmed-batch-789.json") == 789
