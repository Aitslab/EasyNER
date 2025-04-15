import pytest
import time
from unittest.mock import MagicMock, patch

from easyner.pipeline.splitter.progress import ProgressReporter


@pytest.fixture
def mock_queue():
    """Create a mock result queue for testing."""
    return MagicMock()


@pytest.fixture
def progress_reporter(mock_queue):
    """Create a ProgressReporter instance for testing."""
    return ProgressReporter(
        result_queue=mock_queue, worker_id=1, batch_idx=5, total_items=1000
    )


def test_initialization(progress_reporter, mock_queue):
    """Test ProgressReporter initialization."""
    assert progress_reporter.result_queue == mock_queue
    assert progress_reporter.worker_id == 1
    assert progress_reporter.batch_idx == 5
    assert progress_reporter.total_items == 1000
    assert progress_reporter.processed_items == 0

    # Check that report frequency is calculated correctly
    assert progress_reporter.report_frequency == 50  # 1000 // 20


def test_initialization_small_batch():
    """Test ProgressReporter initialization with a small batch."""
    mock_queue = MagicMock()
    reporter = ProgressReporter(
        result_queue=mock_queue,
        worker_id=1,
        batch_idx=1,
        total_items=50,  # Small batch
    )

    # For small batches, report_frequency should be 10
    assert reporter.report_frequency == 10


def test_initialization_very_large_batch():
    """Test ProgressReporter initialization with a very large batch."""
    mock_queue = MagicMock()
    reporter = ProgressReporter(
        result_queue=mock_queue,
        worker_id=1,
        batch_idx=1,
        total_items=20000,  # Very large batch
    )

    # For very large batches, report_frequency should be capped at 500
    assert reporter.report_frequency == 500


def test_update_below_threshold(progress_reporter, mock_queue):
    """Test update method when below reporting threshold."""
    # Update with small increment
    progress_reporter.update(5)

    # No message should be sent yet
    mock_queue.put.assert_not_called()

    # Check state was updated
    assert progress_reporter.processed_items == 5


def test_update_report_frequency_trigger(progress_reporter, mock_queue):
    """Test update method when hitting report frequency threshold."""
    # Update with exactly the report frequency
    progress_reporter.update(50)  # Matches report_frequency

    # Message should be sent
    mock_queue.put.assert_called_once()
    args = mock_queue.put.call_args[0][0]

    # Check message format
    assert args[0] == "PROGRESS"
    assert args[1] == 5  # batch_idx
    assert args[2] == 50  # processed_items
    assert args[3] == 1000  # total_items
    assert args[4] == 1  # worker_id


# Use a more direct approach to test the time-based trigger
def test_update_time_trigger(progress_reporter, mock_queue):
    """Test update method when time interval threshold is reached."""
    # Set initial time reference
    progress_reporter.last_update_time = (
        time.time() - 0.6
    )  # Time passed > min_update_interval (0.5)

    # Update should trigger message due to time passed
    progress_reporter.update(5)

    # Verify message was sent
    mock_queue.put.assert_called_once()

    # Verify correct message content
    args = mock_queue.put.call_args[0][0]
    assert args[0] == "PROGRESS"
    assert args[2] == 5  # processed_items


def test_multiple_updates(progress_reporter, mock_queue):
    """Test multiple update calls."""
    # First update
    progress_reporter.update(25)
    assert mock_queue.put.call_count == 0

    # Second update - hits report frequency
    progress_reporter.update(25)  # Total = 50
    assert mock_queue.put.call_count == 1

    # Reset the mock to clear call history
    mock_queue.reset_mock()

    # More updates, but not hitting threshold
    progress_reporter.update(10)
    progress_reporter.update(20)
    assert mock_queue.put.call_count == 0

    # Hit threshold again
    progress_reporter.update(20)  # Total = 100
    assert mock_queue.put.call_count == 1


def test_report_completion(progress_reporter, mock_queue):
    """Test report_completion method."""
    # Report completion
    progress_reporter.report_completion(num_articles=1000, processing_time=5.2)

    # Should send two messages: one for progress and one for completion
    assert mock_queue.put.call_count == 2

    # First message should be a progress update with 100% completion
    progress_args = mock_queue.put.call_args_list[0][0][0]
    assert progress_args[0] == "PROGRESS"
    assert progress_args[2] == 1000  # processed_items = total_items
    assert progress_args[3] == 1000  # total_items

    # Second message should be a completion notification
    complete_args = mock_queue.put.call_args_list[1][0][0]
    assert complete_args[0] == "COMPLETE"
    assert complete_args[1] == 5  # batch_idx
    assert complete_args[2] == 1000  # num_articles
    assert complete_args[3] == 5.2  # processing_time
    assert complete_args[4] == 1  # worker_id

    # Processed items should be updated to match total
    assert progress_reporter.processed_items == 1000
