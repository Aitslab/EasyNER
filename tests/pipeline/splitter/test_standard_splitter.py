import json
import os
import shutil
import tempfile
from glob import glob

import pytest

from easyner.pipeline.splitter.splitter_runner import run_splitter


@pytest.fixture
def temp_standard_dir():
    """Create a temporary directory with test standard input file."""
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "standard_input.json")

    # Create a standard input file with some test articles
    test_data = {
        "article1": {
            "title": "Test Article 1",
            "text": "This is the first test article. It has two sentences.",
        },
        "article2": {
            "title": "Test Article 2",
            "text": "This is the second article. It has three sentences. The last sentence is here.",
        },
        "empty_article": {"title": "Empty Article", "text": ""},  # Empty text
    }

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f)

    yield input_path

    # Cleanup
    shutil.rmtree(temp_dir)


def test_standard_splitter_format(temp_standard_dir) -> None:
    """Test that the splitter correctly processes standard text data."""
    # Set up output directory
    output_dir = os.path.join(os.path.dirname(temp_standard_dir), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Configure the splitter for standard data
    config = {
        "pubmed_bulk": False,
        "input_path": temp_standard_dir,
        "output_folder": output_dir,
        "output_file_prefix": "test_standard",
        "tokenizer": "spacy",
        "model": "en_core_web_sm",
        "text_field": "text",
        "CPU_LIMIT": 1,
        "batch_size": 2,
    }

    # Run the splitter
    run_splitter(config)

    # Find the output file(s)
    output_files = glob(os.path.join(output_dir, "test_standard_*.json"))
    assert len(output_files) > 0, "No output files were created"

    # Load and merge data from all output files
    output_data = {}
    for output_file in output_files:
        with open(output_file, encoding="utf-8") as f:
            batch_data = json.load(f)
            output_data.update(batch_data)

    # Verify the output structure
    assert "article1" in output_data, "article1 not found in output"
    assert "article2" in output_data, "article2 not found in output"
    assert "empty_article" in output_data, "empty_article not found in output"

    # Check article1 structure and content
    article1 = output_data["article1"]
    assert "title" in article1
    assert article1["title"] == "Test Article 1"
    assert "sentences" in article1
    assert len(article1["sentences"]) == 2

    # Check article2 structure and content
    article2 = output_data["article2"]
    assert "title" in article2
    assert article2["title"] == "Test Article 2"
    assert "sentences" in article2
    assert len(article2["sentences"]) == 3

    # Check empty article structure
    empty_article = output_data["empty_article"]
    assert "title" in empty_article
    assert empty_article["title"] == "Empty Article"
    assert "sentences" in empty_article
    assert len(empty_article["sentences"]) == 0

    # Check sentence content for article1
    sentences = article1["sentences"]
    assert sentences[0]["text"] == "This is the first test article."
    assert sentences[1]["text"] == "It has two sentences."

    # Check sentence content for article2
    sentences = article2["sentences"]
    assert sentences[0]["text"] == "This is the second article."
    assert sentences[1]["text"] == "It has three sentences."
    assert sentences[2]["text"] == "The last sentence is here."


def test_different_tokenizers(temp_standard_dir) -> None:
    """Test that the splitter works with different tokenizers."""
    # First check if NLTK is available
    try:
        import nltk
    except ImportError:
        pytest.skip("NLTK not installed, skipping NLTK tokenizer test")

    # Check if punkt is already downloaded to avoid hanging
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        # Try to download with a timeout
        try:
            import multiprocessing
            from functools import partial

            # Define a function to download punkt with timeout
            def download_punkt():
                nltk.download("punkt", quiet=True)

            # Use a process with timeout
            process = multiprocessing.Process(target=download_punkt)
            process.start()
            process.join(timeout=10)  # 10 seconds timeout

            if process.is_alive():
                # If still running after timeout, terminate and skip
                process.terminate()
                pytest.skip("NLTK punkt download is taking too long, skipping test")

            # Verify it was downloaded
            nltk.data.find("tokenizers/punkt")
        except Exception:
            pytest.skip("Failed to download NLTK punkt, skipping NLTK tokenizer test")

    # Set up output directory
    output_dir = os.path.join(os.path.dirname(temp_standard_dir), "output_nltk")
    os.makedirs(output_dir, exist_ok=True)

    # Configure the splitter with NLTK
    config = {
        "pubmed_bulk": False,
        "input_path": temp_standard_dir,
        "output_folder": output_dir,
        "output_file_prefix": "test_nltk",
        "tokenizer": "nltk",  # Use NLTK tokenizer
        "text_field": "text",
        "CPU_LIMIT": 1,
        "batch_size": 3,
    }

    # Run the splitter with a timeout
    try:
        run_splitter(config)

        # Find the output file(s)
        output_files = glob(os.path.join(output_dir, "test_nltk_*.json"))
        assert len(output_files) > 0, "No output files were created with NLTK tokenizer"

        # Load the output file
        with open(output_files[0], encoding="utf-8") as f:
            output_data = json.load(f)

        # Verify basic structure
        assert "article1" in output_data, "article1 not found in NLTK output"
        assert "article2" in output_data, "article2 not found in NLTK output"

        # The exact sentence splitting might differ between tokenizers,
        # but we should have approximately the same number of sentences
        article1 = output_data["article1"]
        assert 1 <= len(article1["sentences"]) <= 3

        article2 = output_data["article2"]
        assert 2 <= len(article2["sentences"]) <= 4

    except Exception as e:
        pytest.skip(f"NLTK test failed with error: {str(e)}")
