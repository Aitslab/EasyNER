import pytest
import json
import os
import tempfile
import shutil
from glob import glob

from easyner.pipeline.splitter.splitter_runner import run_splitter


@pytest.fixture
def temp_pubmed_dir():
    """Create a temporary directory with test PubMed input file"""
    temp_dir = tempfile.mkdtemp()
    input_folder = os.path.join(temp_dir, "input")
    os.makedirs(input_folder, exist_ok=True)

    # Copy the test PubMed input file - this uses the provided example
    input_path = os.path.join(input_folder, "pubmedn0001.json")

    # Get the path to the example file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    example_file = os.path.join(current_dir, "pubmed_input.json")

    # Copy the example data
    with open(example_file, "r", encoding="utf-8") as src:
        test_data = json.load(src)

    with open(input_path, "w", encoding="utf-8") as dest:
        json.dump(test_data, dest)

    yield input_folder

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def expected_output():
    """Load the expected output format"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    example_file = os.path.join(current_dir, "pubmed_output.json")

    with open(example_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_pubmed_splitter_format(temp_pubmed_dir, expected_output):
    """Test that the splitter correctly processes PubMed data"""
    # Set up output directory
    output_dir = os.path.join(os.path.dirname(temp_pubmed_dir), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Configure the splitter for PubMed data
    config = {
        "pubmed_bulk": True,
        "input_path": temp_pubmed_dir + "/",
        "output_folder": output_dir,
        "output_file_prefix": "test_pubmed",
        "tokenizer": "spacy",  # Changed to use spaCy instead of NLTK
        "model": "en_core_web_sm",  # Specify the spaCy model to use
        "text_field": "abstract",  # Specify to use the abstract field
        "CPU_LIMIT": 1,
        "key": "n",
        "file_limit": "ALL",
    }

    # Run the splitter
    run_splitter(config)

    # Find the output file(s)
    output_files = glob(os.path.join(output_dir, "test_pubmed_*.json"))
    assert len(output_files) > 0, "No output files were created"

    # Load the first output file
    with open(output_files[0], "r", encoding="utf-8") as f:
        output_data = json.load(f)

    # Check overall structure
    assert isinstance(output_data, dict), "Output is not a dictionary"
    assert len(output_data) > 0, "Output file is empty"

    # Check that article structure matches expected format
    for article_id, article_data in output_data.items():
        # Check title presence
        assert "title" in article_data, f"Article {article_id} has no 'title' field"

        # Check sentences structure
        assert "sentences" in article_data, f"Article {article_id} has no 'sentences' field"
        assert isinstance(
            article_data["sentences"], list
        ), f"Article {article_id} 'sentences' is not a list"

        # Check sentence format - should be objects with text property
        for sentence in article_data["sentences"]:
            assert isinstance(
                sentence, dict
            ), f"Sentence in article {article_id} is not a dictionary object"
            assert "text" in sentence, f"Sentence in article {article_id} has no 'text' property"
            assert isinstance(
                sentence["text"], str
            ), f"Sentence text in article {article_id} is not a string"

    # Compare with expected output structure for key articles
    for article_id, expected_article in expected_output.items():
        if article_id in output_data:
            assert (
                output_data[article_id]["title"] == expected_article["title"]
            ), f"Title mismatch for article {article_id}"

            # Compare sentence count
            assert len(output_data[article_id]["sentences"]) == len(
                expected_article["sentences"]
            ), f"Sentence count mismatch for article {article_id}"

            # Compare actual sentence content
            for i, sentence in enumerate(output_data[article_id]["sentences"]):
                expected_text = expected_article["sentences"][i]["text"]
                assert (
                    sentence["text"] == expected_text
                ), f"Sentence content mismatch in article {article_id}, sentence {i}"
