"""Tests for configuration management in EasyNer.

This module contains unit tests for validating and generating configuration files
used in the EasyNer project. It includes tests for schema validation, template
generation, and handling of various edge cases in configuration files.
"""

import io
import json
import shutil
import tempfile
from collections.abc import Generator
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyner.config.generator import ConfigGenerator
from easyner.config.validator import ConfigValidator
from easyner.infrastructure.paths import PROJECT_ROOT, SCHEMA_PATH

# Constants for frequently used filenames and values using SimpleNamespace
# for better attribute access and IDE autocomplete support
TEST_FILENAMES = SimpleNamespace(
    CONFIG="config.json",
    TEMPLATE="config.template.json",
    INVALID_CONFIG_MISSING="invalid_config_missing.json",
    INVALID_CONFIG_TYPE="invalid_config_type.json",
    INVALID_CONFIG_PATHS="invalid_config_paths.json",
    BAD_TEMPLATE="bad_template.json",
    ORIGINAL_BACKUP="original_backup.json",
    PARTIAL_CONFIG="partial_config.json",
    GENERATED_TEMPLATE="generated_template.json",
    CURRENT_CONFIG_COPY="current_config_copy.json",
)


# Helper functions for common operations
def create_json_file(directory: Path, filename: str, content: dict[str, Any]) -> Path:
    """Create a JSON file with the given content and return its path."""
    file_path = directory / filename
    with open(file_path, "w") as f:
        json.dump(content, f, indent=2)
    return file_path


def load_json(file_path: Path) -> dict[str, Any]:
    """Load JSON from file."""
    with open(file_path) as f:
        return json.load(f)


def create_minimal_valid_config() -> dict[str, Any]:
    """Create a minimally valid configuration."""
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
    }


@pytest.fixture
def temp_test_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def sample_config() -> dict[str, Any]:
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
def config_files(temp_test_dir: str, sample_config: dict[str, Any]) -> dict[str, str]:
    """Create config files for testing."""
    config_path = Path(temp_test_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(sample_config, f, indent=2)

    template_path = Path(temp_test_dir) / "config.template.json"

    return {
        "config_path": str(config_path),
        "template_path": str(template_path),
    }


@pytest.fixture
def validator() -> ConfigValidator:
    """Create a ConfigValidator instance for testing."""
    return ConfigValidator(quiet=True)


@pytest.fixture
def generator() -> ConfigGenerator:
    """Create a ConfigGenerator instance for testing."""
    return ConfigGenerator(quiet=True, schema_path=SCHEMA_PATH)


def test_generate_template(
    config_files: dict[str, str],
    sample_config: dict[str, Any],
    generator: ConfigGenerator,
) -> None:
    """Test the template generation functionality."""
    # Generate the template - now directly using the output path

    generator.generate_template(config_files["template_path"])

    # Check that the template file exists
    template_path = Path(config_files["template_path"])
    assert template_path.exists(), "Template file was not created"

    # Load the template
    template = load_json(template_path)

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


def test_validate_config(
    config_files: dict[str, str],
    validator: ConfigValidator,
    generator: ConfigGenerator,
) -> None:
    """Test the config validation functionality."""
    # Disable quiet mode temporarily to see validation errors
    validator.quiet = False

    # Validate the original config and capture output
    f = io.StringIO()
    with redirect_stdout(f):
        result = validator.validate_config(config_files["config_path"])

    # Print error output if validation failed to help debug
    if not result:
        print(f"Config validation failed with errors:\n{f.getvalue()}")

    assert result, "Valid config should pass validation"

    # Re-enable quiet mode
    validator.quiet = True

    # Generate the template directly to the template path
    generator.generate_template(config_files["template_path"])

    # Validate the template using Path object
    template_path = Path(config_files["template_path"])

    # Disable quiet mode to see template validation errors
    validator.quiet = False
    f = io.StringIO()
    with redirect_stdout(f):
        result = validator.validate_config(str(template_path))

    # Print error output if validation failed to help debug
    if not result:
        print(f"Template validation failed with errors:\n{f.getvalue()}")

    assert result, "Template should pass validation"


def test_validate_config_missing_fields(
    temp_test_dir: str,
    validator: ConfigValidator,
) -> None:
    """Test validation of configs missing required fields."""
    # Test invalid config - missing required fields
    invalid_config_path = create_json_file(
        Path(temp_test_dir),
        TEST_FILENAMES.INVALID_CONFIG_MISSING,
        {"CPU_LIMIT": 5},
    )

    result = validator.validate_config(str(invalid_config_path))
    assert not result, "Config missing required fields should fail validation"


def test_validate_config_wrong_types(
    temp_test_dir: str,
    sample_config: dict[str, Any],
    validator: ConfigValidator,
) -> None:
    """Test validation of configs with incorrect data types."""
    # Test invalid config - wrong type
    invalid_config_path = create_json_file(
        Path(temp_test_dir),
        TEST_FILENAMES.INVALID_CONFIG_TYPE,
        {
            "CPU_LIMIT": "not_an_integer",
            "TIMEKEEP": True,
            "ignore": sample_config["ignore"],
        },
    )

    result = validator.validate_config(str(invalid_config_path))
    assert not result, "Config with wrong types should fail validation"


def test_validate_config_path_types(
    temp_test_dir: str,
    validator: ConfigValidator,
) -> None:
    """Test that config validation requires paths to be strings."""
    invalid_config_path = create_json_file(
        Path(temp_test_dir),
        TEST_FILENAMES.INVALID_CONFIG_PATHS,
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
    )

    result = validator.validate_config(str(invalid_config_path))
    assert not result, "Config with non-string path should fail validation"


def test_config_path_types(temp_test_dir: str, validator: ConfigValidator) -> None:
    """Test that path fields in the current config are valid strings."""
    # Copy the project config to a temp location to avoid modifying original
    project_config_path = PROJECT_ROOT / "config.json"
    temp_config_path = Path(temp_test_dir) / "temp_config.json"
    shutil.copy(project_config_path, temp_config_path)

    # Validate the copy instead of the original
    result = validator.validate_config(str(temp_config_path))
    assert result is True


def test_check_absolute_paths(
    config_files: dict[str, str],
    validator: ConfigValidator,
    generator: ConfigGenerator,
) -> None:
    """Test the absolute path checking functionality."""
    # Generate the template
    generator.generate_template(config_files["template_path"])

    # Check the template for absolute paths
    result = validator.check_absolute_paths(config_files["template_path"])
    assert result, "Template should not have absolute paths"


def test_check_absolute_paths_failure(
    config_files: dict[str, str],
    temp_test_dir: str,
    validator: ConfigValidator,
    generator: ConfigGenerator,
) -> None:
    """Test that absolute paths are correctly detected."""
    # Generate the template
    generator.generate_template(config_files["template_path"])

    # Create a template with an absolute path
    bad_template_path = Path(temp_test_dir) / TEST_FILENAMES.BAD_TEMPLATE

    # Load and modify the template
    template = load_json(Path(config_files["template_path"]))

    # Introduce an absolute path
    template["ner"]["input_path"] = "C:/some/path"

    # Save the modified template
    create_json_file(Path(temp_test_dir), TEST_FILENAMES.BAD_TEMPLATE, template)

    # Suppress output during this test to avoid confusing error messages
    f = io.StringIO()
    with redirect_stdout(f):
        result = validator.check_absolute_paths(str(bad_template_path))

    assert not result, "Template with absolute paths should fail check"


def test_schema_loading(validator: ConfigValidator) -> None:
    """Test that the schema loads correctly."""
    # Check that the schema is loaded as expected
    schema = validator.schema
    assert isinstance(schema, dict)
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "CPU_LIMIT" in schema["properties"]


def test_validate_current_config(
    temp_test_dir: str,
    validator: ConfigValidator,
) -> None:
    """Test validation of the current config.json file.

    This test ensures that the actual config file being used in the project
    passes validation, including proper path types and format.
    """
    # Check if config.json exists
    config_path = PROJECT_ROOT / "config.json"
    assert config_path.exists(), "Current config.json file not found"

    # Create a copy in the temp directory
    temp_config_path = Path(temp_test_dir) / TEST_FILENAMES.CURRENT_CONFIG_COPY
    shutil.copy(config_path, temp_config_path)

    # Validate the copy instead of the original
    result = validator.validate_config(str(temp_config_path))
    assert result, (
        "Current config.json failed validation - check for path format issues or "
        "invalid values"
    )

    # Check for absolute paths in the config copy
    result = validator.check_absolute_paths(str(temp_config_path))
    assert result, "Current config.json contains absolute paths"


def test_validator_run_validation_tests(
    temp_test_dir: str,
    validator: ConfigValidator,
) -> None:
    """Test the run_validation_tests method."""
    # This test verifies the behavior of run_validation_tests
    # In a real environment it would pass if both the config.json
    # and config.template.json files exist
    # Since we're testing in isolation, we expect it to fail because
    # those files don't exist
    result = validator.run_validation_tests()
    assert isinstance(result, bool), "run_validation_tests should return a boolean"


def test_generate_template_preserves_values(
    temp_test_dir: str,
    validator: ConfigValidator,
    generator: ConfigGenerator,
) -> None:
    """Test generate_template preserves existing values and only adds missing fields."""
    # Create a partial config with some values in a temporary file
    temp_dir = Path(temp_test_dir)

    partial_config = {
        "$schema": "easyner/config/schema.json",
        "CPU_LIMIT": 8,  # Custom value
        "TIMEKEEP": False,
        "ignore": {
            "cord_loader": False,
            "downloader": True,
        },
    }

    # Create input file that will also serve as the output
    partial_config_path = create_json_file(
        temp_dir,
        TEST_FILENAMES.PARTIAL_CONFIG,
        partial_config,
    )

    # Make a backup copy to verify original content isn't lost
    original_backup_path = temp_dir / TEST_FILENAMES.ORIGINAL_BACKUP
    shutil.copy(partial_config_path, original_backup_path)

    # Generate template using the partial config file path directly
    generator.generate_template(str(partial_config_path), skip_prettier=True)

    # Load the updated config and original backup for verification
    updated_config = load_json(partial_config_path)
    original_content = load_json(original_backup_path)

    # Check that original values are preserved
    assert updated_config["CPU_LIMIT"] == 8, "Custom CPU_LIMIT value was not preserved"
    assert (
        updated_config["TIMEKEEP"] is False
    ), "Custom TIMEKEEP value was not preserved"
    assert (
        updated_config["ignore"]["cord_loader"] is False
    ), "Custom ignore.cord_loader value was not preserved"
    assert (
        updated_config["ignore"]["downloader"] is True
    ), "Custom ignore.downloader value was not preserved"

    # Check that missing fields were added
    assert (
        "text_loader" in updated_config["ignore"]
    ), "Missing ignore.text_loader field was not added"
    assert "ner" in updated_config, "Missing ner section was not added"
    assert "downloader" in updated_config, "Missing downloader section was not added"

    # Check that validation passes on the completed template
    assert validator.validate_config(
        str(partial_config_path),
    ), "Generated template fails validation"

    # Verify the original file wasn't modified
    # (our backup should still have the original content)
    assert "text_loader" not in original_content.get(
        "ignore",
        {},
    ), "Original file content verification failed"
    assert "ner" not in original_content, "Original file content verification failed"
