import pytest
from easyner.io.utils import get_batch_file_index, filter_batch_files


@pytest.fixture
def create_batch_filename():
    def _create_filename(batch_number, file_extension=".txt"):
        return f"batch_{batch_number}{file_extension}"

    return _create_filename


@pytest.mark.parametrize(
    "batch_number,expected,file_extension",
    [
        # Regular positive numbers
        ("1", 1, ".txt"),
        ("22", 22, ".txt"),
        ("33", 33, ".json"),
        ("123", 123, ".csv"),
        # Edge cases
        ("0", 0, ".txt"),
        ("01", 1, ".txt"),
        ("02", 2, ".json"),
        ("-5", 5, ".txt"),
        ("any_name-10", 10, ".json"),
        # Mixed numeric and non-numeric - should raise ValueError
        ("batch_1-2", 2, ".txt"),
        ("batch_2_2", 2, ".json"),
        # However leading non-numeric characters should be allowed
        ("batch_33", 33, ".csv"),
        ("batch_123", 123, ".txt"),
        # Mixed alphanumeric - should raise ValueError
        ("123abc", pytest.raises(ValueError), ".txt"),
        ("abc123", 123, ".csv"),
        # No batch number - should raise ValueError
        ("no_number", pytest.raises(ValueError), ".txt"),
        # Empty string - should raise ValueError
        ("", pytest.raises(ValueError), ".txt"),
    ],
)
def test_get_batch_number(
    create_batch_filename, batch_number, expected, file_extension
):
    filename = create_batch_filename(batch_number, file_extension)

    if isinstance(expected, int):
        assert get_batch_file_index(filename) == expected
    else:
        with expected:
            get_batch_file_index(filename)

    # Empty string special case handled separately
    if batch_number == "":
        with pytest.raises(ValueError):
            get_batch_file_index("")


@pytest.fixture
def batch_files():
    return [
        "batch_1.txt",
        "batch_2.txt",
        "batch_3.txt",
    ]


def test_filter_batch_files_with_empty_list():
    """Test filter_batch_files with an empty list."""
    with pytest.raises(ValueError):
        filter_batch_files([], start=1, end=3)


def test_filter_batch_files_with_start_greater_than_end():
    """Test filter_batch_files with start > end."""
    files = ["batch_1.txt", "batch_2.txt", "batch_3.txt"]
    with pytest.raises(ValueError):
        filter_batch_files(files, start=3, end=1)


def test_filter_batch_files_with_empty_list_exclude_batches(caplog):
    files = ["batch_1.txt", "batch_2.txt", "batch_3.txt"]
    with caplog.at_level("WARNING"):
        filtered_files = filter_batch_files(files, exclude_batches=[])
        assert filtered_files == files  # No files should be excluded
        # Check that some warning was logged, without requiring exact text
        assert len(caplog.records) > 0
        assert any(record.levelname == "WARNING" for record in caplog.records)


@pytest.mark.parametrize(
    "files,start,end,exclude_batches,expected",
    [
        # Basic filter tests
        (
            ["batch_1.txt", "batch_2.txt", "batch_3.txt"],
            1,
            3,
            None,
            ["batch_1.txt", "batch_2.txt", "batch_3.txt"],
        ),
        (
            ["batch_1.txt", "batch_2.txt", "batch_3.txt"],
            1,
            3,
            [2],
            ["batch_1.txt", "batch_3.txt"],
        ),
        (
            ["batch_1.txt", "batch_2.txt", "batch_3.txt"],
            None,
            2,
            None,
            ["batch_1.txt", "batch_2.txt"],
        ),
        (
            ["batch_1.txt", "batch_2.txt", "batch_3.txt"],
            2,
            None,
            None,
            ["batch_2.txt", "batch_3.txt"],
        ),
        # Test with different filename formats
        (
            ["batch-1.json", "batch-2.json", "batch-3.json"],
            1,
            3,
            None,
            ["batch-1.json", "batch-2.json", "batch-3.json"],
        ),
        # Test with non-sequential batch numbers
        (
            ["batch_1.txt", "batch_5.txt", "batch_10.txt"],
            1,
            10,
            None,
            ["batch_1.txt", "batch_5.txt", "batch_10.txt"],
        ),
        # Test excluding multiple batches
        (
            ["batch_1.txt", "batch_2.txt", "batch_3.txt", "batch_4.txt"],
            None,
            None,
            [1, 3],
            ["batch_2.txt", "batch_4.txt"],
        ),
        # Test with mixed filename patterns
        (
            ["batch_1.txt", "batch-2.json", "batch_3.csv"],
            1,
            3,
            None,
            ["batch_1.txt", "batch-2.json", "batch_3.csv"],
        ),
        # Test with start > end
        (
            ["batch_1.txt", "batch_2.txt", "batch_3.txt"],
            3,
            1,
            None,
            pytest.raises(ValueError),
        ),
        # Test with start > end and exclude_batches
        (
            ["batch_1.txt", "batch_2.txt", "batch_3.txt"],
            3,
            1,
            [2],
            pytest.raises(ValueError),
        ),
        # Test with absolute paths
        (
            [
                "/path/to/batch_1.txt",
                "/path/to/batch_2.txt",
                "/path/to/batch_3.txt",
            ],
            1,
            3,
            None,
            [
                "/path/to/batch_1.txt",
                "/path/to/batch_2.txt",
                "/path/to/batch_3.txt",
            ],
        ),
        # Test with nested directories
        (
            ["dir1/batch_1.txt", "dir2/batch_2.txt", "dir3/batch_3.txt"],
            1,
            3,
            None,
            ["dir1/batch_1.txt", "dir2/batch_2.txt", "dir3/batch_3.txt"],
        ),
        # Test with mixed absolute and relative paths
        (
            ["/abs/path/batch_1.txt", "rel/path/batch_2.txt", "./batch_3.txt"],
            1,
            3,
            None,
            ["/abs/path/batch_1.txt", "rel/path/batch_2.txt", "./batch_3.txt"],
        ),
        # Test with deeply nested paths
        (
            ["/root/dir1/subdir/batch_1.txt", "/root/dir2/subdir/batch_2.txt"],
            1,
            2,
            None,
            ["/root/dir1/subdir/batch_1.txt", "/root/dir2/subdir/batch_2.txt"],
        ),
        # Test excluding batches with mixed paths
        (
            ["/path/to/batch_1.txt", "dir/batch_2.txt", "./batch_3.txt"],
            1,
            3,
            [2],
            ["/path/to/batch_1.txt", "./batch_3.txt"],
        ),
        # Test with Windows-style paths (should still work on Linux)
        (
            ["C:\\Users\\user\\batch_1.txt", "D:\\data\\batch_2.txt"],
            1,
            2,
            None,
            ["C:\\Users\\user\\batch_1.txt", "D:\\data\\batch_2.txt"],
        ),
    ],
)
def test_filter_batch_files_parametrized(
    files, start, end, exclude_batches, expected
):
    """Test filter_batch_files with various input combinations."""
    # If expected is a context manager (like pytest.raises), use it directly
    if not isinstance(expected, list):
        with expected:
            filter_batch_files(
                files, start=start, end=end, exclude_batches=exclude_batches
            )
    else:
        # Otherwise, compare the function's output with the expected result
        filtered_files = filter_batch_files(
            files, start=start, end=end, exclude_batches=exclude_batches
        )
        assert filtered_files == expected
