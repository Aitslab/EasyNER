import pytest
import json
import os
import tempfile
import shutil
from glob import glob

from easyner.pipeline.splitter.splitter_runner import run_splitter
from easyner.pipeline.splitter.writers import JSONWriter
from easyner.pipeline.splitter.tokenizers import NLTKTokenizer, SpacyTokenizer


@pytest.fixture
def temp_input_file():
    """Create a temporary input file with test data"""
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "test_input.json")

    # Create test data with a few sample articles
    test_data = {
        "article1": {"text": "This is sentence one. This is sentence two."},
        "article2": {"text": "Another article with multiple sentences. Second sentence here."},
    }

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)

    yield input_path

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_splitter_output_format(temp_input_file, temp_output_dir):
    """Test that splitter outputs files with proper format"""
    # Set up configuration
    config = {
        "input_path": temp_input_file,
        "output_folder": temp_output_dir,
        "output_file_prefix": "test",
        "tokenizer": "nltk",
        "batch_size": 2,
        "CPU_LIMIT": 1,
    }

    # Run the splitter
    run_splitter(config)

    # Check that output files exist
    output_files = glob(os.path.join(temp_output_dir, "test_nltk-split-*.json"))
    assert len(output_files) > 0, "No output files were created"

    # Check the format of each output file
    for output_file in output_files:
        with open(output_file, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        # Verify it's a dictionary
        assert isinstance(output_data, dict), "Output is not a dictionary"

        # Check that we have article entries
        assert len(output_data) > 0, "Output file is empty"

        # Check the structure of articles
        for article_id, article_data in output_data.items():
            assert "sentences" in article_data, f"Article {article_id} has no 'sentences' field"
            assert isinstance(
                article_data["sentences"], list
            ), f"Article {article_id} 'sentences' is not a list"

            # Check that we have sentences
            assert len(article_data["sentences"]) > 0, f"Article {article_id} has no sentences"

            # Check sentence format
            for sentence in article_data["sentences"]:
                assert isinstance(
                    sentence, str
                ), f"Sentence in article {article_id} is not a string"


def test_spacy_splitter_output_format(temp_input_file, temp_output_dir):
    """Test that splitter outputs files with proper format using Spacy tokenizer"""
    # Set up configuration
    config = {
        "input_path": temp_input_file,
        "output_folder": temp_output_dir,
        "output_file_prefix": "test",
        "tokenizer": "spacy",
        "model": "en_core_web_sm",  # Specify the spaCy model
        "batch_size": 2,
        "CPU_LIMIT": 1,
    }

    # Run the splitter
    run_splitter(config)

    # Check that output files exist
    output_files = glob(os.path.join(temp_output_dir, "test_spacy-split-*.json"))
    assert len(output_files) > 0, "No output files were created"

    # Check the format of each output file
    for output_file in output_files:
        with open(output_file, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        # Verify it's a dictionary
        assert isinstance(output_data, dict), "Output is not a dictionary"

        # Check that we have article entries
        assert len(output_data) > 0, "Output file is empty"

        # Check the structure of articles
        for article_id, article_data in output_data.items():
            assert "sentences" in article_data, f"Article {article_id} has no 'sentences' field"
            assert isinstance(
                article_data["sentences"], list
            ), f"Article {article_id} 'sentences' is not a list"

            # Check that we have sentences
            assert len(article_data["sentences"]) > 0, f"Article {article_id} has no sentences"

            # Check sentence format
            for sentence in article_data["sentences"]:
                assert isinstance(
                    sentence, str
                ), f"Sentence in article {article_id} is not a string"


def test_direct_writer_format():
    """Test the JSONWriter directly to ensure correct format"""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a writer
        writer = JSONWriter(output_folder=temp_dir, output_file_prefix="direct_test")

        # Create test data
        test_articles = {"article1": {"sentences": ["Test sentence 1.", "Test sentence 2."]}}

        # Write the data
        batch_idx, num_articles = writer.write(test_articles, 1, "test_tokenizer")

        # Verify file exists
        output_file = os.path.join(temp_dir, "direct_test_test_tokenizer-split-1.json")
        assert os.path.exists(output_file), "Output file was not created"

        # Check file content
        with open(output_file, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        # Verify structure
        assert output_data == test_articles, "Output data does not match input data"
        assert batch_idx == 1, "Batch index was not returned correctly"
        assert num_articles == 1, "Number of articles was not returned correctly"

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
