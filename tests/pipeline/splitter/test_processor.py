import pytest
import time
import logging
from unittest.mock import MagicMock, patch

from easyner.pipeline.splitter.splitter_processor import SplitterProcessor
from easyner.pipeline.splitter.tokenizers import TokenizerBase


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock(spec=TokenizerBase)
    tokenizer.SUPPORTS_BATCH_PROCESSING = True
    tokenizer.SUPPORTS_BATCH_GENERATOR = False
    tokenizer.__class__.__name__ = "MockTokenizer"
    return tokenizer


@pytest.fixture
def mock_writer():
    """Create a mock output writer for testing."""
    writer = MagicMock()
    writer.write.return_value = (1, 3)  # batch_idx, articles_written
    return writer


@pytest.fixture
def default_config():
    """Create a default configuration for testing."""
    return {
        "worker_id": "test-worker",
        "pubmed_bulk": False,
        "max_tokenizer_batch_size": 100,
        "io_format": "json",
        "text_field": "custom_text",
    }


@pytest.fixture
def sample_batch():
    """Create a sample batch of articles for testing."""
    return {
        "article1": {
            "title": "Article 1",
            "text": "This is article 1. It has two sentences.",
        },
        "article2": {
            "title": "Article 2",
            "text": "This is article 2. It also has two sentences.",
        },
        "empty_article": {"title": "Empty Article", "text": ""},
    }


@pytest.fixture
def processed_articles():
    """Create a sample of processed articles."""
    return {
        "article1": {
            "title": "Article 1",
            "sentences": [
                {"text": "This is article 1."},
                {"text": "It has two sentences."},
            ],
        },
        "article2": {
            "title": "Article 2",
            "sentences": [
                {"text": "This is article 2."},
                {"text": "It also has two sentences."},
            ],
        },
        "empty_article": {"title": "Empty Article", "sentences": []},
    }


def test_processor_initialization(mock_tokenizer, mock_writer, default_config):
    """Test SplitterProcessor initialization."""
    processor = SplitterProcessor(
        tokenizer=mock_tokenizer,
        output_writer=mock_writer,
        config=default_config,
    )

    # Check that instance variables are set correctly
    assert processor.tokenizer == mock_tokenizer
    assert processor.output_writer == mock_writer
    assert processor.config == default_config
    assert processor.worker_id == "test-worker"
    assert processor.max_batch_size == 100
    assert processor.io_format == "json"


def test_record_timing_methods():
    """Test the timing recording methods."""
    processor = SplitterProcessor(
        tokenizer=MagicMock(),
        output_writer=MagicMock(),
        config={"worker_id": "test-worker"},
    )

    # Test record_start_time
    start_time = processor.record_start_time()
    assert isinstance(start_time, float)

    # Test record_elapsed_time
    # Sleep a short time to ensure elapsed time is measurable
    time.sleep(0.01)
    elapsed = processor.record_elapsed_time(start_time)
    assert elapsed > 0
    assert isinstance(elapsed, float)


# Fix patch paths to target where they're imported
@patch(
    "easyner.pipeline.splitter.splitter_processor.ProcessingStrategySelector.select_strategy"
)
@patch("easyner.pipeline.splitter.splitter_processor.create_strategy")
def test_process_batch_standard_text(
    mock_create_strategy,
    mock_select_strategy,
    mock_tokenizer,
    mock_writer,
    default_config,
    sample_batch,
    processed_articles,
):
    """Test processing a batch of standard text articles."""
    # Setup mocks
    mock_strategy = MagicMock()
    mock_strategy.process.return_value = processed_articles
    mock_create_strategy.return_value = mock_strategy
    mock_select_strategy.return_value = (
        "single_document"  # Use correct strategy name
    )

    progress_callback = MagicMock()

    processor = SplitterProcessor(
        tokenizer=mock_tokenizer,
        output_writer=mock_writer,
        config=default_config,
        progress_callback=progress_callback,
    )

    # Process the batch
    batch_idx = 1
    result_batch_idx, num_articles = processor.process_batch(
        batch_idx, sample_batch
    )

    # Verify strategy selection and processing
    mock_select_strategy.assert_called_once_with(sample_batch)
    mock_create_strategy.assert_called_once_with("single_document")
    mock_strategy.process.assert_called_once_with(
        processor, batch_idx, sample_batch, "custom_text", None
    )

    # Verify writer was called with processed articles
    mock_writer.write.assert_called_once_with(
        processed_articles, batch_idx, "MockTokenizer"
    )

    # Verify progress callback was called
    progress_callback.assert_called_once_with(batch_idx, 3, 3)

    # Verify return values
    assert result_batch_idx == 1
    assert num_articles == 3


# Fix patch paths to target where they're imported
@patch(
    "easyner.pipeline.splitter.splitter_processor.ProcessingStrategySelector.select_strategy"
)
@patch("easyner.pipeline.splitter.splitter_processor.create_strategy")
def test_process_batch_pubmed_data(
    mock_create_strategy,
    mock_select_strategy,
    mock_tokenizer,
    mock_writer,
    processed_articles,
):
    """Test processing a batch of PubMed data."""
    # Setup mocks
    mock_strategy = MagicMock()
    mock_strategy.process.return_value = processed_articles
    mock_create_strategy.return_value = mock_strategy
    mock_select_strategy.return_value = (
        "batch_generator"  # Use correct strategy name
    )

    pubmed_config = {
        "worker_id": "test-worker",
        "pubmed_bulk": True,  # Set to PubMed mode
        "max_tokenizer_batch_size": 100,
        "io_format": "json",
    }

    sample_pubmed_batch = {
        "12345": {
            "title": "PubMed Article 1",
            "abstract": "This is abstract 1.",
        },
        "67890": {
            "title": "PubMed Article 2",
            "abstract": "This is abstract 2.",
        },
    }

    processor = SplitterProcessor(
        tokenizer=mock_tokenizer,
        output_writer=mock_writer,
        config=pubmed_config,
    )

    # Process the batch
    batch_idx = 1
    full_articles = True
    result_batch_idx, num_articles = processor.process_batch(
        batch_idx, sample_pubmed_batch, full_articles
    )

    # Verify strategy processing with PubMed settings
    mock_strategy.process.assert_called_once_with(
        processor, batch_idx, sample_pubmed_batch, "abstract", full_articles
    )

    # Verify return values
    assert result_batch_idx == 1
    assert num_articles == 3


# Fix patch paths to target where they're imported
@patch(
    "easyner.pipeline.splitter.splitter_processor.ProcessingStrategySelector.select_strategy"
)
@patch("easyner.pipeline.splitter.splitter_processor.create_strategy")
@patch("gc.collect")
def test_garbage_collection(
    mock_gc_collect,
    mock_create_strategy,
    mock_select_strategy,
    mock_tokenizer,
    mock_writer,
    default_config,
    sample_batch,
    processed_articles,
):
    """Test that garbage collection is triggered after batch processing."""
    # Setup mocks
    mock_strategy = MagicMock()
    mock_strategy.process.return_value = processed_articles
    mock_create_strategy.return_value = mock_strategy
    mock_select_strategy.return_value = (
        "single_document"  # Use correct strategy name
    )

    processor = SplitterProcessor(
        tokenizer=mock_tokenizer,
        output_writer=mock_writer,
        config=default_config,
    )

    # Process the batch
    batch_idx = 1
    processor.process_batch(batch_idx, sample_batch)

    # Verify garbage collection was triggered
    mock_gc_collect.assert_called_once()
