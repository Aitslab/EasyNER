import pytest
import json
import os
import tempfile
import shutil
from glob import glob

from easyner.pipeline.splitter.splitter_runner import run_splitter


@pytest.fixture
def temp_invalid_dir():
    """Create a temporary directory with invalid input files"""
    temp_dir = tempfile.mkdtemp()

    # Create an invalid JSON file (not properly formatted)
    invalid_json_path = os.path.join(temp_dir, "invalid.json")
    with open(invalid_json_path, "w", encoding="utf-8") as f:
        f.write("{not valid json")

    # Create an empty file
    empty_file_path = os.path.join(temp_dir, "empty.json")
    with open(empty_file_path, "w", encoding="utf-8") as f:
        f.write("")

    # Create a valid JSON but with unexpected structure
    unexpected_structure_path = os.path.join(temp_dir, "unexpected.json")
    with open(unexpected_structure_path, "w", encoding="utf-8") as f:
        json.dump(["item1", "item2"], f)  # Array instead of object

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


def test_invalid_json_handling(temp_invalid_dir):
    """Test how the splitter handles invalid JSON input"""
    # Set up output directory
    output_dir = os.path.join(temp_invalid_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Configure the splitter with invalid JSON input
    config = {
        "pubmed_bulk": False,
        "input_path": os.path.join(temp_invalid_dir, "invalid.json"),
        "output_folder": output_dir,
        "output_file_prefix": "test_invalid",
        "tokenizer": "spacy",
        "model": "en_core_web_sm",
        "text_field": "text",
        "CPU_LIMIT": 1,
    }

    # Run the splitter - it should handle the error gracefully
    try:
        run_splitter(config)
        # If it doesn't raise an exception, check if output was created
        output_files = glob(os.path.join(output_dir, "test_invalid_*.json"))
        assert (
            len(output_files) == 0
        ), "Output file was created with invalid JSON input"
    except Exception as e:
        # The error should be specific about JSON parsing issues
        assert (
            "JSON" in str(e) or "json" in str(e) or "parse" in str(e)
        ), f"Unexpected error: {str(e)}"


def test_empty_file_handling(temp_invalid_dir):
    """Test how the splitter handles empty file input"""
    # Set up output directory
    output_dir = os.path.join(temp_invalid_dir, "output_empty")
    os.makedirs(output_dir, exist_ok=True)

    # Configure the splitter with empty file input
    config = {
        "pubmed_bulk": False,
        "input_path": os.path.join(temp_invalid_dir, "empty.json"),
        "output_folder": output_dir,
        "output_file_prefix": "test_empty",
        "tokenizer": "spacy",
        "model": "en_core_web_sm",
        "text_field": "text",
        "CPU_LIMIT": 1,
    }

    # Run the splitter - it should handle empty file gracefully
    try:
        run_splitter(config)
        # If it doesn't raise an exception, check if output was created
        output_files = glob(os.path.join(output_dir, "test_empty_*.json"))
        assert (
            len(output_files) == 0
        ), "Output file was created with empty input"
    except Exception as e:
        # Empty file error should be specific
        assert (
            "empty" in str(e).lower() or "no data" in str(e).lower()
        ), f"Unexpected error: {str(e)}"


def test_missing_text_field(temp_invalid_dir):
    """Test how the splitter handles missing text field"""
    # Create a JSON file with missing text field
    missing_field_path = os.path.join(temp_invalid_dir, "missing_field.json")
    with open(missing_field_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "article1": {
                    "title": "Test Article 1",
                    # text field is missing
                },
                "article2": {
                    "title": "Test Article 2",
                    "content": "This is in the wrong field.",  # field with different name
                },
            },
            f,
        )

    # Set up output directory
    output_dir = os.path.join(temp_invalid_dir, "output_missing")
    os.makedirs(output_dir, exist_ok=True)

    # Configure the splitter
    config = {
        "pubmed_bulk": False,
        "input_path": missing_field_path,
        "output_folder": output_dir,
        "output_file_prefix": "test_missing",
        "tokenizer": "spacy",
        "model": "en_core_web_sm",
        "text_field": "text",  # Looking for 'text' field which doesn't exist
        "CPU_LIMIT": 1,
    }

    # Run the splitter - should create output with empty sentences
    run_splitter(config)

    # Should create output file even with missing fields
    output_files = glob(os.path.join(output_dir, "test_missing_*.json"))
    assert len(output_files) > 0, "No output files were created"

    # Load the output file
    with open(output_files[0], "r", encoding="utf-8") as f:
        output_data = json.load(f)

    # Both articles should be present but have empty sentences
    assert "article1" in output_data, "article1 not found in output"
    assert "article2" in output_data, "article2 not found in output"

    # Check that both have empty sentences lists
    assert len(output_data["article1"]["sentences"]) == 0
    assert len(output_data["article2"]["sentences"]) == 0


def test_invalid_tokenizer(temp_invalid_dir):
    """Test how the splitter handles invalid tokenizer selection"""
    # Create a simple valid JSON file
    valid_path = os.path.join(temp_invalid_dir, "valid.json")
    with open(valid_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "article1": {
                    "title": "Test Article",
                    "text": "This is test content.",
                }
            },
            f,
        )

    # Set up output directory
    output_dir = os.path.join(temp_invalid_dir, "output_tokenizer")
    os.makedirs(output_dir, exist_ok=True)

    # Configure the splitter with invalid tokenizer
    config = {
        "pubmed_bulk": False,
        "input_path": valid_path,
        "output_folder": output_dir,
        "output_file_prefix": "test_tokenizer",
        "tokenizer": "non_existent_tokenizer",  # Invalid tokenizer
        "text_field": "text",
        "CPU_LIMIT": 1,
    }

    # Run the splitter - should raise an error about invalid tokenizer
    with pytest.raises(ValueError) as excinfo:
        run_splitter(config)

    # Error should mention the tokenizer
    assert "tokenizer" in str(
        excinfo.value
    ).lower() or "non_existent_tokenizer" in str(excinfo.value)
