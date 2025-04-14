import pytest
from pathlib import Path  # Import Path

# Removed unused import: from easyner.io import get_io_handler
from easyner.pipeline.utils import get_batch_index_from_filename, construct_output_path


# Using parametrize for get_batch_index tests as suggested before
@pytest.mark.parametrize(
    "filename, expected_index",
    [
        ("file-123.json", 123),
        ("/path/to/output_456.parquet", 456),
        ("batch_789_index.txt", 789),
        ("prefix-001.json", 1),
        ("data_with_version_1.2-batch-99.csv", 99),
        ("only_index_10.dat", 10),
        ("file-123-456.json", 456),  # Test multiple numbers
    ],
)
def test_get_batch_index_valid(filename, expected_index):
    """Tests valid filenames for batch index extraction."""
    assert get_batch_index_from_filename(filename) == expected_index


@pytest.mark.parametrize(
    "filename",
    [
        "file-abc.json",
    ],
)
def test_get_batch_index_convert_error(filename):
    """Tests filenames where extraction finds non-numeric part causing conversion error."""
    with pytest.raises(ValueError, match="Could not find batch index number"):
        get_batch_index_from_filename(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "file.json",
        "prefix-.txt",
        "nonumberhere.txt",
    ],
)
def test_get_batch_index_not_found_error(filename):
    """Tests filenames where no index part can be found."""
    with pytest.raises(ValueError, match="Could not find batch index number"):
        get_batch_index_from_filename(filename)


# --- Tests for construct_output_path ---


def test_construct_output_path_basic(tmp_path: Path):  # Use tmp_path fixture
    """Test basic path construction in a temporary directory."""
    output_dir = tmp_path / "output"  # Create path within tmp_path
    output_path = construct_output_path(
        output_dir=str(output_dir),  # Convert Path object to string
        file_prefix="result",
        input_filename="input-123.json",
        output_extension="json",
    )
    expected_path = output_dir / "result-123.json"
    assert output_path == str(expected_path)
    # Check directory was created by the function
    assert output_dir.is_dir()


def test_construct_output_path_different_ext(tmp_path: Path):  # Use tmp_path fixture
    """Test path construction with different extensions."""
    output_dir = tmp_path / "ner_out"
    output_path = construct_output_path(
        output_dir=str(output_dir),
        file_prefix="result",
        input_filename="input-123.json",
        output_extension=".parquet",  # Test with leading dot
    )
    expected_path = output_dir / "result-123.parquet"
    assert output_path == str(expected_path)
    assert output_dir.is_dir()


# You can add more tests for construct_output_path using tmp_path if needed
# e.g., test_construct_output_path_complex_input_name(tmp_path: Path): ...
# e.g., test_construct_output_path_invalid_input_filename(tmp_path: Path): ...
