import time
from unittest.mock import MagicMock, patch

import pytest

from easyner.pipeline.splitter.splitter_runner import (
    SplitterRunner,
    make_batches,
    run_splitter,
    worker_process,
)


def test_make_batches() -> None:
    """Test the make_batches function properly divides lists into chunks."""
    # Test with exact multiples of batch size
    assert list(make_batches([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    # Test with non-exact division
    assert list(make_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    # Test with batch size larger than list
    assert list(make_batches([1, 2, 3], 5)) == [[1, 2, 3]]

    # Test with empty list
    assert list(make_batches([], 3)) == []


@patch("easyner.pipeline.splitter.splitter_runner.SpacyTokenizer")
@patch("easyner.pipeline.splitter.splitter_runner.JSONWriter")
@patch("easyner.pipeline.splitter.splitter_runner.SplitterProcessor")
@patch("easyner.pipeline.splitter.splitter_runner.WorkerStateManager")
@patch("easyner.pipeline.splitter.splitter_runner.psutil.Process")
def test_worker_process_initialization(
    mock_process,
    mock_state_manager,
    mock_processor,
    mock_writer,
    mock_tokenizer,
) -> None:
    """Test worker process initialization with SpaCy tokenizer."""
    # Setup mocks
    task_queue = MagicMock()
    result_queue = MagicMock()
    config = {
        "tokenizer": "spacy",
        "model": "en_core_web_sm",
        "output_folder": "/path/to/output",
        "output_file_prefix": "test",
        "pubmed_bulk": False,
    }
    worker_id = 1

    # Setup task queue behavior to return None (sentinel) on first get
    task_queue.get.return_value = None

    # Setup mock tokenizer instance
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.SUPPORTS_BATCH_PROCESSING = True
    mock_tokenizer_instance.SUPPORTS_BATCH_GENERATOR = False
    mock_tokenizer.return_value = mock_tokenizer_instance

    # Setup mock state manager
    mock_state_manager_instance = MagicMock()
    mock_state_manager.return_value = mock_state_manager_instance

    # Setup mock process to return a concrete value for memory info
    memory_mock = MagicMock()
    memory_mock.rss = 100 * 1024 * 1024  # 100 MB in bytes
    mock_process.return_value.memory_info.return_value = memory_mock

    # Call worker process
    worker_process(task_queue, result_queue, config, worker_id)

    # Verify proper initialization
    mock_tokenizer.assert_called_once_with(model_name="en_core_web_sm")
    mock_writer.assert_called_once()
    mock_processor.assert_called_once()

    # Verify state manager calls
    mock_state_manager.assert_called_once_with(worker_id, result_queue)
    mock_state_manager_instance.signal_ready.assert_called_once_with(
        mock_tokenizer_instance.SUPPORTS_BATCH_PROCESSING,
        mock_tokenizer_instance.SUPPORTS_BATCH_GENERATOR,
    )
    mock_state_manager_instance.signal_done.assert_called_once()

    # Verify task queue was accessed
    task_queue.get.assert_called_once()


@patch("easyner.pipeline.splitter.splitter_runner.NLTKTokenizer")
@patch("easyner.pipeline.splitter.splitter_runner.JSONWriter")
@patch("easyner.pipeline.splitter.splitter_runner.SplitterProcessor")
@patch("easyner.pipeline.splitter.splitter_runner.WorkerStateManager")
@patch("easyner.pipeline.splitter.splitter_runner.psutil.Process")
def test_worker_process_with_nltk(
    mock_process,
    mock_state_manager,
    mock_processor,
    mock_writer,
    mock_tokenizer,
) -> None:
    """Test worker process initialization with NLTK tokenizer."""
    # Setup mocks
    task_queue = MagicMock()
    result_queue = MagicMock()
    config = {
        "tokenizer": "nltk",
        "output_folder": "/path/to/output",
        "output_file_prefix": "test",
    }
    worker_id = 1

    # Setup task queue behavior
    task_queue.get.return_value = None

    # Setup mock tokenizer
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.SUPPORTS_BATCH_PROCESSING = False
    mock_tokenizer_instance.SUPPORTS_BATCH_GENERATOR = False
    mock_tokenizer.return_value = mock_tokenizer_instance

    # Setup mock process to return a concrete value for memory info
    memory_mock = MagicMock()
    memory_mock.rss = 100 * 1024 * 1024  # 100 MB in bytes
    mock_process.return_value.memory_info.return_value = memory_mock

    # Call worker process
    worker_process(task_queue, result_queue, config, worker_id)

    # Verify NLTK tokenizer was used
    mock_tokenizer.assert_called_once()
    mock_writer.assert_called_once()


@patch("easyner.pipeline.splitter.splitter_runner.SpacyTokenizer")
@patch("easyner.pipeline.splitter.splitter_runner.JSONWriter")
@patch("easyner.pipeline.splitter.splitter_runner.SplitterProcessor")
@patch("easyner.pipeline.splitter.splitter_runner.WorkerStateManager")
@patch("easyner.pipeline.splitter.splitter_runner.psutil.Process")
@patch("easyner.pipeline.splitter.splitter_runner.logger")  # Also patch logger
def test_worker_process_batch_processing(
    mock_logger,  # Add logger mock
    mock_process,
    mock_state_manager,
    mock_processor,
    mock_writer,
    mock_tokenizer,
) -> None:
    """Test worker process batch processing."""
    # Setup mocks
    task_queue = MagicMock()
    result_queue = MagicMock()
    config = {
        "tokenizer": "spacy",
        "model": "en_core_web_sm",
        "output_folder": "/path/to/output",
        "output_file_prefix": "test",
        "pubmed_bulk": True,
        "spacy_batch_size": 32,
    }
    worker_id = 1

    # Setup task queue behavior - first return a batch, then a sentinel
    batch_idx = 1
    batch_data = {"article1": {"text": "Test"}, "article2": {"text": "Test 2"}}
    task_queue.get.side_effect = [(batch_idx, batch_data, True), None]

    # Setup mock tokenizer with nlp attribute for batch size
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.SUPPORTS_BATCH_PROCESSING = True
    mock_tokenizer_instance.SUPPORTS_BATCH_GENERATOR = True
    mock_tokenizer_instance.nlp = MagicMock()
    mock_tokenizer.return_value = mock_tokenizer_instance

    # Setup mock processor
    mock_processor_instance = MagicMock()
    mock_processor_instance.process_batch.return_value = (
        batch_idx,
        2,
    )  # 2 articles processed
    mock_processor.return_value = mock_processor_instance

    # Setup mock state manager
    mock_state_manager_instance = MagicMock()
    mock_state_manager_instance.start_batch.return_value = time.time()
    mock_state_manager_instance.complete_batch.return_value = (
        batch_idx,
        0.5,
    )  # 0.5s processing time
    mock_state_manager.return_value = mock_state_manager_instance

    # Setup mock process to return a concrete value for memory info
    memory_mock = MagicMock()
    memory_mock.rss = 100 * 1024 * 1024  # 100 MB in bytes
    mock_process.return_value.memory_info.return_value = memory_mock

    # Call worker process
    worker_process(task_queue, result_queue, config, worker_id)

    # Verify batch processing
    assert mock_tokenizer_instance.nlp.batch_size == 32
    mock_processor_instance.process_batch.assert_called_once_with(
        batch_idx,
        batch_data,
        True,
    )
    mock_state_manager_instance.start_batch.assert_called_once_with(
        batch_idx,
        batch_data,
    )
    mock_state_manager_instance.complete_batch.assert_called_once_with(2)


@patch("easyner.pipeline.splitter.splitter_runner.PubMedLoader")
@patch("easyner.pipeline.splitter.splitter_runner.MessageHandler")
@patch("easyner.pipeline.splitter.splitter_runner.Queue")
@patch("easyner.pipeline.splitter.splitter_runner.Process")
@patch("easyner.pipeline.splitter.splitter_runner.Value")
def test_splitter_runner_pubmed_init(
    mock_value,
    mock_process,
    mock_queue,
    mock_message_handler,
    mock_pubmed_loader,
) -> None:
    """Test SplitterRunner initialization with PubMed configuration."""
    # Setup mocks
    mock_task_queue = MagicMock()
    mock_result_queue = MagicMock()
    mock_queue.side_effect = [mock_task_queue, mock_result_queue]

    # Setup mock counters
    mock_counter = MagicMock()
    mock_value.return_value = mock_counter

    # Setup PubMed loader with batch files
    mock_pubmed_loader_instance = MagicMock()
    mock_pubmed_loader_instance.load_data.return_value = [
        "/path/to/batch1.json",
        "/path/to/batch2.json",
    ]
    mock_pubmed_loader_instance.get_batch_index.side_effect = [1, 2]
    mock_pubmed_loader_instance.load_batch.side_effect = [
        {"article1": {"text": "Test 1"}},
        {"article2": {"text": "Test 2"}},
    ]
    mock_pubmed_loader.return_value = mock_pubmed_loader_instance

    # Setup config
    config = {
        "input_path": "/path/to/input",
        "output_folder": "/path/to/output",
        "output_file_prefix": "test",
        "pubmed_bulk": True,
        "tokenizer": "spacy",
        "model": "en_core_web_sm",
        "stats_update_interval": 0.1,
    }

    # Create SplitterRunner
    runner = SplitterRunner(config, cpu_limit=2)

    # Mock message handler behavior for run method
    mock_message_handler_instance = MagicMock()
    mock_message_handler.return_value = mock_message_handler_instance

    # Mock message processing
    def mock_handle_message(result) -> None:
        if result[0] == "WORKER_READY":
            runner.workers_ready += 1

    mock_message_handler_instance.handle.side_effect = mock_handle_message

    # Setup process instances
    mock_process_instances = [MagicMock(), MagicMock()]
    mock_process.side_effect = mock_process_instances

    # Need to patch some methods that would be called during run
    with (
        patch("easyner.pipeline.splitter.splitter_runner.tqdm") as mock_tqdm,
        patch("threading.Thread"),
    ):

        mock_tqdm_instance = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance

        # Call run method, but stop after worker initialization
        with patch.object(runner, "_stats_updater"):
            # Mock result queue to return WORKER_READY messages during initialization
            mock_result_queue.get.side_effect = [
                ("WORKER_READY", 0, True, False),  # Worker 0 ready
                ("WORKER_READY", 1, True, False),  # Worker 1 ready
                Exception("Stop processing"),  # To break the loop
            ]

            try:
                runner.run()
            except Exception:
                pass  # Expected to fail due to our mocked exception

    # Verify PubMed loader was initialized correctly
    mock_pubmed_loader.assert_called_once_with(
        input_folder=config["input_path"],
        limit=config.get("file_limit", "ALL"),
        key=config.get("key", "n"),
        io_format="json",
    )

    # Verify PubMed loader methods were called
    mock_pubmed_loader_instance.load_data.assert_called_once()


@patch("easyner.pipeline.splitter.splitter_runner.StandardLoader")
@patch("easyner.pipeline.splitter.splitter_runner.MessageHandler")
@patch("easyner.pipeline.splitter.splitter_runner.Queue")
@patch("easyner.pipeline.splitter.splitter_runner.Process")
@patch("easyner.pipeline.splitter.splitter_runner.Value")
def test_splitter_runner_standard_init(
    mock_value,
    mock_process,
    mock_queue,
    mock_message_handler,
    mock_standard_loader,
) -> None:
    """Test SplitterRunner initialization with standard text configuration."""
    # Setup mocks
    mock_task_queue = MagicMock()
    mock_result_queue = MagicMock()
    mock_queue.side_effect = [mock_task_queue, mock_result_queue]

    # Setup mock counters
    mock_counter = MagicMock()
    mock_value.return_value = mock_counter

    # Setup standard loader
    mock_standard_loader_instance = MagicMock()
    mock_standard_loader_instance.load_data.return_value = {
        "article1": {"text": "Test 1"},
        "article2": {"text": "Test 2"},
        "article3": {"text": "Test 3"},
    }
    mock_standard_loader.return_value = mock_standard_loader_instance

    # Setup config
    config = {
        "input_path": "/path/to/input",
        "output_folder": "/path/to/output",
        "output_file_prefix": "test",
        "pubmed_bulk": False,
        "batch_size": 2,
        "tokenizer": "spacy",
        "model": "en_core_web_sm",
    }

    # Create SplitterRunner
    runner = SplitterRunner(config, cpu_limit=2)

    # Mock message handler behavior for run method
    mock_message_handler_instance = MagicMock()
    mock_message_handler.return_value = mock_message_handler_instance

    # Mock message processing
    def mock_handle_message(result) -> None:
        if result[0] == "WORKER_READY":
            runner.workers_ready += 1

    mock_message_handler_instance.handle.side_effect = mock_handle_message

    # Setup process instances
    mock_process_instances = [MagicMock(), MagicMock()]
    mock_process.side_effect = mock_process_instances

    # Need to patch some methods that would be called during run
    with (
        patch("easyner.pipeline.splitter.splitter_runner.tqdm") as mock_tqdm,
        patch("threading.Thread"),
    ):

        mock_tqdm_instance = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance

        # Call run method, but stop after worker initialization
        with patch.object(runner, "_stats_updater"):
            # Mock result queue to return WORKER_READY messages during initialization
            mock_result_queue.get.side_effect = [
                ("WORKER_READY", 0, True, False),  # Worker 0 ready
                ("WORKER_READY", 1, True, False),  # Worker 1 ready
                Exception("Stop processing"),  # To break the loop
            ]

            try:
                runner.run()
            except Exception:
                pass  # Expected to fail due to our mocked exception

    # Verify StandardLoader was initialized correctly
    mock_standard_loader.assert_called_once_with(
        input_path=config["input_path"],
        io_format="json",
    )

    # Verify StandardLoader methods were called
    mock_standard_loader_instance.load_data.assert_called_once()


@patch("easyner.pipeline.splitter.splitter_runner.SplitterRunner")
@patch("easyner.pipeline.splitter.splitter_runner.cpu_count")
def test_run_splitter(mock_cpu_count, mock_splitter_runner) -> None:
    """Test the main run_splitter function."""
    # Setup mocks
    mock_splitter_instance = MagicMock()
    mock_splitter_runner.return_value = mock_splitter_instance
    mock_cpu_count.return_value = 4

    # Test with specific CPU limit
    config = {
        "tokenizer": "spacy",
        "input_path": "/path/to/input",
        "output_folder": "/path/to/output",
        "output_file_prefix": "test",
    }
    result = run_splitter(config, ignore=False, cpu_limit=2)

    # Verify SplitterRunner was initialized correctly
    mock_splitter_runner.assert_called_once_with(config, 2)
    mock_splitter_instance.run.assert_called_once()
    assert result == {}

    # Reset mocks
    mock_splitter_runner.reset_mock()
    mock_splitter_instance.reset_mock()

    # Test ignore flag
    result = run_splitter(config, ignore=True, cpu_limit=2)
    mock_splitter_runner.assert_not_called()
    mock_splitter_instance.run.assert_not_called()
    assert result == {}

    # Test invalid tokenizer
    config["tokenizer"] = "invalid"
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        run_splitter(config, ignore=False, cpu_limit=2)
