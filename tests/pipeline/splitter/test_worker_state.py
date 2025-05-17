# ruff: noqa: ANN001,
import time
from unittest.mock import MagicMock, patch

import pytest

from easyner.pipeline.splitter.worker_state import WorkerStateManager


@pytest.fixture
def mock_queue():
    """Create a mock result queue for testing."""
    return MagicMock()


@pytest.fixture
def worker_state_manager(mock_queue):
    """Create a WorkerStateManager instance for testing."""
    return WorkerStateManager(worker_id=1, result_queue=mock_queue)


def test_init(worker_state_manager, mock_queue) -> None:
    """Test WorkerStateManager initialization."""
    assert worker_state_manager.worker_id == 1
    assert worker_state_manager.result_queue == mock_queue
    assert worker_state_manager.current_batch is None
    assert worker_state_manager.stats == {
        "batches": 0,
        "articles": 0,
        "processing_time": 0,
    }


def test_start_batch(worker_state_manager) -> None:
    """Test start_batch method."""
    batch_idx = 1
    batch = {"article1": {}, "article2": {}}

    start_time = worker_state_manager.start_batch(batch_idx, batch)

    assert isinstance(start_time, float)
    assert worker_state_manager.current_batch["idx"] == batch_idx
    assert worker_state_manager.current_batch["size"] == 2
    assert worker_state_manager.current_batch["start_time"] == start_time


def test_complete_batch_with_active_batch(worker_state_manager, mock_queue) -> None:
    """Test complete_batch method with an active batch."""
    # Start a batch first
    batch_idx = 1
    batch = {"article1": {}, "article2": {}}
    start_time = worker_state_manager.start_batch(batch_idx, batch)

    # Mock time.time() to return a predictable elapsed time
    with patch("time.time", return_value=start_time + 0.5):
        result_idx, elapsed = worker_state_manager.complete_batch(2)

    # Check return values
    assert result_idx == batch_idx
    assert elapsed == 0.5

    # Check stats updates
    assert worker_state_manager.stats["batches"] == 1
    assert worker_state_manager.stats["articles"] == 2
    assert worker_state_manager.stats["processing_time"] == 0.5

    # Check message was sent to the queue
    mock_queue.put.assert_called_once_with(("COMPLETE", batch_idx, 2, 0.5, 1))

    # Check current_batch was reset
    assert worker_state_manager.current_batch is None


def test_complete_batch_without_active_batch(worker_state_manager, mock_queue) -> None:
    """Test complete_batch method without an active batch."""
    result_idx, elapsed = worker_state_manager.complete_batch(2)

    # Check return values
    assert result_idx is None
    assert elapsed == 0

    # Check no message was sent to the queue
    mock_queue.put.assert_not_called()


def test_report_error_with_batch_idx(worker_state_manager, mock_queue) -> None:
    """Test report_error method with explicit batch_idx."""
    worker_state_manager.report_error("Test error", 5)

    mock_queue.put.assert_called_once()
    message_type, error_msg, worker_id = mock_queue.put.call_args[0][0]

    assert message_type == "ERROR"
    assert "Worker 1 error processing batch 5: Test error" in error_msg
    assert worker_id == 1


def test_report_error_with_current_batch(worker_state_manager, mock_queue) -> None:
    """Test report_error method using current batch index."""
    # Start a batch first
    batch_idx = 3
    batch = {"article1": {}}
    worker_state_manager.start_batch(batch_idx, batch)

    worker_state_manager.report_error("Test error")

    mock_queue.put.assert_called_once()
    message_type, error_msg, worker_id = mock_queue.put.call_args[0][0]

    assert message_type == "ERROR"
    assert "Worker 1 error processing batch 3: Test error" in error_msg
    assert worker_id == 1


def test_signal_ready(worker_state_manager, mock_queue) -> None:
    """Test signal_ready method."""
    worker_state_manager.signal_ready(True, False)

    mock_queue.put.assert_called_once_with(("WORKER_READY", 1, True, False))


def test_signal_done(worker_state_manager, mock_queue) -> None:
    """Test signal_done method."""
    peak_memory = 256.5
    worker_state_manager.signal_done(peak_memory)

    mock_queue.put.assert_called_once_with(("WORKER_DONE", 1, peak_memory))


def test_get_stats_empty(worker_state_manager) -> None:
    """Test get_stats method with no batches processed."""
    stats = worker_state_manager.get_stats()

    assert stats["worker_id"] == 1
    assert stats["batches"] == 0
    assert stats["articles"] == 0
    assert stats["time"] == 0
    assert stats["avg_batch_size"] == 0
    assert stats["avg_time_per_batch"] == 0
    assert stats["avg_time_per_article"] == 0


def test_get_stats_with_activity(worker_state_manager, mock_queue) -> None:
    """Test get_stats method after processing batches."""
    # Start and complete a batch
    worker_state_manager.start_batch(1, {"article1": {}, "article2": {}})
    with patch("time.time", return_value=time.time() + 0.5):
        worker_state_manager.complete_batch(2)

    # Start and complete another batch
    worker_state_manager.start_batch(
        2,
        {"article3": {}, "article4": {}, "article5": {}},
    )
    with patch("time.time", return_value=time.time() + 1.5):
        worker_state_manager.complete_batch(3)

    stats = worker_state_manager.get_stats()

    # Check basic stats
    assert stats["worker_id"] == 1
    assert stats["batches"] == 2
    assert stats["articles"] == 5
    assert stats["time"] == pytest.approx(
        2.0,
        abs=0.01,
    )  # Allow small difference from 2.0

    # Check derived metrics
    assert stats["avg_batch_size"] == 2.5  # 5 / 2
    assert stats["avg_time_per_batch"] == pytest.approx(
        1.0,
        abs=0.01,
    )  # Allow small difference from 1.0
    assert stats["avg_time_per_article"] == pytest.approx(
        0.4,
        abs=0.01,
    )  # Allow small difference from 0.4
