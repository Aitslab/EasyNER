import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from easyner.infrastructure import paths


def test_paths_defined():
    """Test that all standard paths are defined."""
    # Basic paths
    assert isinstance(paths.PROJECT_ROOT, Path)
    assert isinstance(paths.PACKAGE_ROOT, Path)
    assert isinstance(paths.SCRIPTS_DIR, Path)
    assert isinstance(paths.CONFIG_DIR, Path)
    assert isinstance(paths.INFRASTRUCTURE_DIR, Path)

    # Config files
    assert isinstance(paths.CONFIG_PATH, Path)
    assert isinstance(paths.TEMPLATE_PATH, Path)
    assert isinstance(paths.SCHEMA_PATH, Path)

    # Data directories
    assert isinstance(paths.DATA_DIR, Path)
    assert isinstance(paths.RESULTS_DIR, Path)
    assert isinstance(paths.MODELS_DIR, Path)

    # Results subdirectories
    assert isinstance(paths.NER_RESULTS_DIR, Path)
    assert isinstance(paths.DATALOADER_RESULTS_DIR, Path)
    assert isinstance(paths.SPLITTER_RESULTS_DIR, Path)
    assert isinstance(paths.ANALYSIS_RESULTS_DIR, Path)


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for testing path creation."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@patch("easyner.infrastructure.paths.DATA_DIR")
@patch("easyner.infrastructure.paths.MODELS_DIR")
@patch("easyner.infrastructure.paths.RESULTS_DIR")
@patch("easyner.infrastructure.paths.NER_RESULTS_DIR")
@patch("easyner.infrastructure.paths.DATALOADER_RESULTS_DIR")
@patch("easyner.infrastructure.paths.SPLITTER_RESULTS_DIR")
@patch("easyner.infrastructure.paths.ANALYSIS_RESULTS_DIR")
def test_ensure_paths_exist_creates_base_dirs(
    mock_analysis_dir,
    mock_splitter_dir,
    mock_dataloader_dir,
    mock_ner_dir,
    mock_results_dir,
    mock_models_dir,
    mock_data_dir,
    temp_project_dir,
):
    """Test that ensure_paths_exist creates base directories."""
    # Setup mocks to point to temp directory
    mock_data_dir.exists.return_value = False
    mock_data_dir.mkdir = MagicMock()

    mock_models_dir.exists.return_value = False
    mock_models_dir.mkdir = MagicMock()

    # Call the function
    results = paths.ensure_paths_exist(create_results_dirs=False)

    # Verify the function attempted to create directories
    mock_data_dir.mkdir.assert_called_once_with(exist_ok=True)
    mock_models_dir.mkdir.assert_called_once_with(exist_ok=True)

    # Verify results structure
    assert "DATA_DIR" in results
    assert "MODELS_DIR" in results
    assert results["DATA_DIR"]["created"] is True
    assert results["MODELS_DIR"]["created"] is True


@patch("easyner.infrastructure.paths.DATA_DIR")
@patch("easyner.infrastructure.paths.MODELS_DIR")
@patch("easyner.infrastructure.paths.RESULTS_DIR")
@patch("easyner.infrastructure.paths.NER_RESULTS_DIR")
@patch("easyner.infrastructure.paths.DATALOADER_RESULTS_DIR")
@patch("easyner.infrastructure.paths.SPLITTER_RESULTS_DIR")
@patch("easyner.infrastructure.paths.ANALYSIS_RESULTS_DIR")
def test_ensure_paths_exist_creates_results_dirs(
    mock_analysis_dir,
    mock_splitter_dir,
    mock_dataloader_dir,
    mock_ner_dir,
    mock_results_dir,
    mock_models_dir,
    mock_data_dir,
):
    """Test that ensure_paths_exist creates results directories."""
    # Setup mocks
    mock_data_dir.exists.return_value = True
    mock_models_dir.exists.return_value = True

    mock_results_dir.exists.return_value = False
    mock_results_dir.mkdir = MagicMock()

    mock_ner_dir.exists.return_value = False
    mock_ner_dir.mkdir = MagicMock()

    mock_dataloader_dir.exists.return_value = False
    mock_dataloader_dir.mkdir = MagicMock()

    mock_splitter_dir.exists.return_value = False
    mock_splitter_dir.mkdir = MagicMock()

    mock_analysis_dir.exists.return_value = False
    mock_analysis_dir.mkdir = MagicMock()

    # Call the function
    results = paths.ensure_paths_exist(create_results_dirs=True)

    # Verify the function attempted to create directories
    mock_ner_dir.mkdir.assert_called_once_with(exist_ok=True)
    mock_dataloader_dir.mkdir.assert_called_once_with(exist_ok=True)
    mock_splitter_dir.mkdir.assert_called_once_with(exist_ok=True)
    mock_analysis_dir.mkdir.assert_called_once_with(exist_ok=True)

    # Verify results structure for results dirs
    assert "RESULTS_DIR" in results
    assert "NER_RESULTS_DIR" in results
    assert "DATALOADER_RESULTS_DIR" in results
    assert "SPLITTER_RESULTS_DIR" in results
    assert "ANALYSIS_RESULTS_DIR" in results

    assert results["RESULTS_DIR"]["created"] is True
    assert results["NER_RESULTS_DIR"]["created"] is True

    # Verify RESULTS_DIR was called multiple times (once for itself, plus once for each subdirectory)
    assert mock_results_dir.mkdir.call_count > 0  # It's called at least once
    assert mock_results_dir.mkdir.call_args_list[0] == call(
        exist_ok=True
    )  # With the correct args

    # Verify the subdirectories were each called once
    mock_ner_dir.mkdir.assert_called_once_with(exist_ok=True)
    mock_dataloader_dir.mkdir.assert_called_once_with(exist_ok=True)
    mock_splitter_dir.mkdir.assert_called_once_with(exist_ok=True)
    mock_analysis_dir.mkdir.assert_called_once_with(exist_ok=True)


@patch("easyner.infrastructure.paths.DATA_DIR")
@patch("easyner.infrastructure.paths.MODELS_DIR")
def test_ensure_paths_exist_dry_run(mock_models_dir, mock_data_dir):
    """Test ensure_paths_exist in dry run mode."""
    # Setup mocks
    mock_data_dir.exists.return_value = False
    mock_data_dir.mkdir = MagicMock()

    mock_models_dir.exists.return_value = False
    mock_models_dir.mkdir = MagicMock()

    # Call the function in dry_run mode
    results = paths.ensure_paths_exist(create_results_dirs=False, dry_run=True)

    # Verify no directories were created
    mock_data_dir.mkdir.assert_not_called()
    mock_models_dir.mkdir.assert_not_called()

    # Verify results structure
    assert results["DATA_DIR"]["exists"] is False
    assert results["MODELS_DIR"]["exists"] is False
    assert results["DATA_DIR"]["created"] is False
    assert results["MODELS_DIR"]["created"] is False
