import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

from easyner.config.validator import check_absolute_paths, load_schema, validate_config
from easyner.config.generator import generate_template
from easyner.infrastructure.paths import PROJECT_ROOT


@pytest.fixture
def temp_test_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create a sample configuration for testing."""
    return {
        "CPU_LIMIT": 5,
        "TIMEKEEP": True,
        "ignore": {
            "cord_loader": True,
            "downloader": False,
            "text_loader": True,
            "pubmed_bulk_loader": True,
            "splitter": False,
            "ner": True,
            "analysis": True,
            "merger": True,
            "metrics": True,
            "nel": True,
            "result_inspection": True,
        },
        "downloader": {
            "input_path": "data/test.txt",
            "output_path": "results/test/output.json",
            "batch_size": 100,
        },
        "ner": {
            "input_path": "C:/Users/test/Documents/data/",
            "output_path": "results/ner/",
            "output_file_prefix": "ner_test",
            "model_type": "test_model",
            "model_folder": "test_folder",
            "model_name": "test_name",
            "vocab_path": "D:/projects/vocab.txt",
            "store_tokens": "no",
            "labels": "",
            "clear_old_results": True,
            "article_limit": [-1, 100],
            "entity_type": "test",
            "multiprocessing": True,
        },
    }


@pytest.fixture
def config_files(temp_test_dir: str, sample_config: Dict[str, Any]) -> Dict[str, str]:
    """Create config files for testing."""
    config_path = Path(temp_test_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(sample_config, f, indent=2)

    template_path = Path(temp_test_dir) / "config.template.json"

    return {"config_path": str(config_path), "template_path": str(template_path)}


def test_generate_template(
    config_files: Dict[str, str], sample_config: Dict[str, Any]
) -> None:
    """Test the template generation functionality."""
    # Generate the template - now directly using the output path
    generate_template(config_files["template_path"])

    # Check that the template file exists
    assert Path(config_files["template_path"]).exists(), "Template file was not created"

    # Load the template
    with open(config_files["template_path"], "r") as f:
        template = json.load(f)

    # Check that the basic structure is present
    assert "CPU_LIMIT" in template
    assert "TIMEKEEP" in template
    assert "ignore" in template

    # Check that paths are empty strings
    assert template["ner"]["input_path"] == ""
    assert template["ner"]["vocab_path"] == ""

    # Check that the version was added
    assert "CONFIG_VERSION" in template

    # Check that comments were added
    assert "_comments" in template

    # Check that schema reference is included
    assert "$schema" in template


def test_validate_config(config_files: Dict[str, str]) -> None:
    """Test the config validation functionality."""
    # Validate the original config
    result = validate_config(config_files["config_path"])
    assert result, "Valid config should pass validation"

    # Generate the template directly to the template path
    generate_template(config_files["template_path"])

    # Validate the template
    result = validate_config(config_files["template_path"])
    assert result, "Template should pass validation"


def test_validate_config_missing_fields(temp_test_dir: str) -> None:
    """Test validation of configs missing required fields."""
    # Test invalid config - missing required fields
    invalid_config = Path(temp_test_dir) / "invalid_config_missing.json"
    with open(invalid_config, "w") as f:
        # This will fail because it's missing required fields like TIMEKEEP and ignore
        json.dump({"CPU_LIMIT": 5}, f)

    result = validate_config(str(invalid_config))
    assert not result, "Config missing required fields should fail validation"


def test_validate_config_wrong_types(
    temp_test_dir: str, sample_config: Dict[str, Any]
) -> None:
    """Test validation of configs with incorrect data types."""
    # Test invalid config - wrong type
    invalid_config_type = Path(temp_test_dir) / "invalid_config_type.json"
    with open(invalid_config_type, "w") as f:
        # This includes all required fields but has wrong type for CPU_LIMIT
        json.dump(
            {
                "CPU_LIMIT": "not_an_integer",
                "TIMEKEEP": True,
                "ignore": sample_config["ignore"],
            },
            f,
        )

    result = validate_config(str(invalid_config_type))
    assert not result, "Config with wrong types should fail validation"


def test_validate_config_path_types(temp_test_dir: str) -> None:
    """Test that config validation requires paths to be strings."""
    invalid_config = Path(temp_test_dir) / "invalid_config_paths.json"
    with open(invalid_config, "w") as f:
        json.dump(
            {
                "CPU_LIMIT": 5,
                "TIMEKEEP": True,
                "ignore": {
                    "cord_loader": True,
                    "downloader": False,
                    "text_loader": True,
                    "pubmed_bulk_loader": True,
                    "splitter": False,
                    "ner": True,
                    "analysis": True,
                    "merger": True,
                    "metrics": True,
                    "nel": True,
                    "result_inspection": True,
                },
                "downloader": {
                    "input_path": "data/test.txt",
                    "output_path": 42,  # Should be a string
                    "batch_size": 100,
                },
            },
            f,
        )

    result = validate_config(str(invalid_config))
    assert not result, "Config with non-string path should fail validation"


def test_config_path_types() -> None:
    """Test that path fields in the current config are valid strings."""
    # This test should pass with the fixed config.json where all paths are strings
    result = validate_config("config.json")
    assert result is True


def test_check_absolute_paths(config_files: Dict[str, str]) -> None:
    """Test the absolute path checking functionality."""
    # Generate the template
    generate_template(config_files["template_path"])

    # Check the template for absolute paths
    result = check_absolute_paths(config_files["template_path"])
    assert result, "Template should not have absolute paths"


def test_check_absolute_paths_failure(
    config_files: Dict[str, str], temp_test_dir: str
) -> None:
    """Test that absolute paths are correctly detected."""
    # Generate the template
    generate_template(config_files["template_path"])

    # Create a template with an absolute path
    bad_template = Path(temp_test_dir) / "bad_template.json"
    with open(config_files["template_path"], "r") as f:
        template = json.load(f)

    # Introduce an absolute path
    template["ner"]["input_path"] = "C:/some/path"

    with open(bad_template, "w") as f:
        json.dump(template, f, indent=2)

    # Suppress output during this test to avoid confusing error messages
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        result = check_absolute_paths(str(bad_template))

    assert not result, "Template with absolute paths should fail check"


def test_schema_loading() -> None:
    """Test that the schema loads correctly."""
    # Check that the schema is loaded as expected
    schema = load_schema()
    assert isinstance(schema, dict)
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "CPU_LIMIT" in schema["properties"]


def test_validate_current_config() -> None:
    """Test validation of the current config.json file.

    This test ensures that the actual config file being used in the project
    passes validation, including proper path types and format.
    """
    # Check if config.json exists
    config_path = PROJECT_ROOT / "config.json"
    assert config_path.exists(), "Current config.json file not found"
    "Generate using python easyner.config.generate.py"



    # Validate the config file
    result = validate_config(str(config_path))

    # The test should fail if validation fails
    assert (
        result
    ), "Current config.json failed validation - check for path format issues or invalid values"

    # Check for absolute paths in the config
    result = check_absolute_paths(str(config_path))
    assert result, "Current config.json contains absolute paths"
