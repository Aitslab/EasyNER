import pytest
from easyner.io.utils import (
    extract_batch_index,
)


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
        assert extract_batch_index(filename) == expected
    else:
        with expected:
            extract_batch_index(filename)

    # Empty string special case handled separately
    if batch_number == "":
        with pytest.raises(ValueError):
            extract_batch_index("")
