import os
import pytest

# Import path constants from the infrastructure package
from easyner.infrastructure.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    SCRIPTS_DIR,
    NER_RESULTS_DIR,
    DATALOADER_RESULTS_DIR,
    SPLITTER_RESULTS_DIR,
    ANALYSIS_RESULTS_DIR,
    CONFIG_PATH,
    TEMPLATE_PATH,
    SCHEMA_PATH,
)


# Fixture to provide absolute paths for testing
@pytest.fixture
def absolute_paths():
    """Convert relative paths to absolute paths based on project root."""
    return {
        "DATA_DIR": PROJECT_ROOT / DATA_DIR,
        "RESULTS_DIR": PROJECT_ROOT / RESULTS_DIR,
        "MODELS_DIR": PROJECT_ROOT / MODELS_DIR,
        "SCRIPTS_DIR": SCRIPTS_DIR,  # Already absolute
        "NER_RESULTS_DIR": PROJECT_ROOT / NER_RESULTS_DIR,
        "DATALOADER_RESULTS_DIR": PROJECT_ROOT / DATALOADER_RESULTS_DIR,
        "SPLITTER_RESULTS_DIR": PROJECT_ROOT / SPLITTER_RESULTS_DIR,
        "ANALYSIS_RESULTS_DIR": PROJECT_ROOT / ANALYSIS_RESULTS_DIR,
        "DEFAULT_CONFIG_PATH": PROJECT_ROOT / CONFIG_PATH,
        "DEFAULT_TEMPLATE_PATH": PROJECT_ROOT / TEMPLATE_PATH,
        "DEFAULT_SCHEMA_PATH": SCHEMA_PATH,  # Already absolute
    }


def test_project_root_exists():
    """Test that the project root directory exists."""
    assert (
        PROJECT_ROOT.exists()
    ), f"Project root does not exist: {PROJECT_ROOT}"
    assert (
        PROJECT_ROOT.is_dir()
    ), f"Project root is not a directory: {PROJECT_ROOT}"


def test_standard_directories_exist(absolute_paths):
    """Test that the standard directories exist."""
    # Check each standard directory
    for name, path in absolute_paths.items():
        if "DIR" in name:  # Only test directories
            # Create the directory if it doesn't exist (for test directories)
            if not path.exists() and "RESULTS" in name:
                os.makedirs(path, exist_ok=True)
                assert path.exists(), f"Failed to create {name}: {path}"

            if path.exists():
                assert path.is_dir(), f"{name} is not a directory: {path}"
            else:
                pytest.skip(f"{name} does not exist: {path}")


def test_config_files(absolute_paths):
    """Test the config file paths."""
    # Check that at least the default config exists
    assert absolute_paths["DEFAULT_CONFIG_PATH"].exists(), (
        f"Default config file does not exist at: {absolute_paths['DEFAULT_CONFIG_PATH']} \n"
        f"Please generate a config file using the generator or restore from personal backups."
    )

    # Check schema path
    assert absolute_paths[
        "DEFAULT_SCHEMA_PATH"
    ].exists(), (
        f"Schema file does not exist: {absolute_paths['DEFAULT_SCHEMA_PATH']}"
    )


def test_path_relationships():
    """Test the relationships between paths."""
    # Test that subdirectories are correctly related to their parent directories
    assert NER_RESULTS_DIR.parent == RESULTS_DIR
    assert DATALOADER_RESULTS_DIR.parent == RESULTS_DIR
    assert SPLITTER_RESULTS_DIR.parent == RESULTS_DIR
    assert ANALYSIS_RESULTS_DIR.parent == RESULTS_DIR
