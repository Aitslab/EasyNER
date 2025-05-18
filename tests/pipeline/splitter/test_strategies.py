import time
from unittest.mock import MagicMock, patch

import pytest

from easyner.pipeline.splitter.strategies import (
    BatchGeneratorStrategy,
    BatchOptimizedStrategy,
    ProcessingStrategySelector,
    SingleDocumentStrategy,
    create_strategy,
)


@pytest.fixture
def mock_processor():
    """Create a mock processor for testing strategies."""
    processor = MagicMock()
    processor.worker_id = 1
    processor.tokenizer = MagicMock()
    processor.record_start_time = MagicMock(return_value=time.time())
    processor.record_elapsed_time = MagicMock(return_value=0.1)
    return processor


@pytest.fixture
def sample_batch():
    """Create a sample batch of articles for testing."""
    return {
        "article1": {"text": "This is article 1. It has two sentences."},
        "article2": {"text": "This is article 2. It also has two sentences."},
        "empty_article": {"text": ""},  # Empty article
    }


# We can't directly test the abstract BaseProcessingStrategy, so let's use a concrete subclass
def test_prepare_batch_functionality(sample_batch) -> None:
    """Test prepare_batch method functionality (using a concrete subclass)."""
    # Use BatchOptimizedStrategy which inherits from BaseProcessingStrategy
    strategy = BatchOptimizedStrategy()

    texts, article_ids, empty_articles, original_order = strategy.prepare_batch(
        sample_batch,
        "text",
    )

    # Check texts extraction
    assert len(texts) == 2  # Two non-empty articles
    assert "This is article 1. It has two sentences." in texts
    assert "This is article 2. It also has two sentences." in texts

    # Check article IDs
    assert len(article_ids) == 2  # Two non-empty articles
    assert "article1" in article_ids
    assert "article2" in article_ids

    # Check empty articles
    assert len(empty_articles) == 1
    assert "empty_article" in empty_articles

    # Check original order
    assert len(original_order) == 3  # All articles
    assert original_order == ["article1", "article2", "empty_article"]


def test_reconstruct_output_functionality() -> None:
    """Test reconstruct_output method functionality (using a concrete subclass)."""
    # Use BatchOptimizedStrategy which inherits from BaseProcessingStrategy
    strategy = BatchOptimizedStrategy()

    processed_articles = {
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
    }

    empty_articles = {"empty_article": {"title": "Empty Article", "sentences": []}}

    original_order = ["article1", "empty_article", "article2"]

    output = strategy.reconstruct_output(
        processed_articles,
        empty_articles,
        original_order,
    )

    # Check that output has all articles
    assert len(output) == 3
    assert "article1" in output
    assert "article2" in output
    assert "empty_article" in output

    # Check that empty article is included
    assert output["empty_article"] == {
        "title": "Empty Article",
        "sentences": [],
    }

    # Check ordering is preserved
    keys = list(output.keys())
    assert keys[0] == "article1"
    assert keys[1] == "empty_article"
    assert keys[2] == "article2"


def test_batch_optimized_strategy(mock_processor, sample_batch) -> None:
    """Test BatchOptimizedStrategy process method."""
    strategy = BatchOptimizedStrategy()
    batch_idx = 1
    text_field = "text"
    full_articles = False

    # Mock tokenizer behavior with the format it actually returns
    mock_processor.tokenizer.segment_sentences_batch.return_value = [
        ["This is article 1.", "It has two sentences."],
        ["This is article 2.", "It also has two sentences."],
    ]

    result = strategy.process(
        mock_processor,
        batch_idx,
        sample_batch,
        text_field,
        full_articles,
    )

    # Check that process calls tokenizer
    mock_processor.tokenizer.segment_sentences_batch.assert_called_once()

    # Check result structure
    assert len(result) == 3
    assert "article1" in result
    assert "article2" in result
    assert "empty_article" in result

    # Check sentence structure
    assert len(result["article1"]["sentences"]) == 2
    assert result["article1"]["sentences"][0]["text"] == "This is article 1."

    # Check empty article handling
    assert result["empty_article"]["sentences"] == []


@patch("logging.Logger.debug")
def test_batch_generator_strategy(mock_debug, mock_processor, sample_batch) -> None:
    """Test BatchGeneratorStrategy process method."""
    strategy = BatchGeneratorStrategy()
    batch_idx = 1
    text_field = "text"
    full_articles = False

    # Mock tokenizer generator behavior with the format it actually returns
    mock_processor.tokenizer.segment_sentences_batch_generator.return_value = [
        ["This is article 1.", "It has two sentences."],
        ["This is article 2.", "It also has two sentences."],
    ]

    result = strategy.process(
        mock_processor,
        batch_idx,
        sample_batch,
        text_field,
        full_articles,
    )

    # Check that process calls generator
    mock_processor.tokenizer.segment_sentences_batch_generator.assert_called_once()

    # Check result structure
    assert len(result) == 3
    assert "article1" in result
    assert "article2" in result
    assert "empty_article" in result

    # Check sentence structure
    assert len(result["article1"]["sentences"]) == 2
    assert result["article1"]["sentences"][0]["text"] == "This is article 1."

    # Check empty article handling
    assert result["empty_article"]["sentences"] == []


def test_single_document_strategy(mock_processor, sample_batch) -> None:
    """Test SingleDocumentStrategy process method."""
    strategy = SingleDocumentStrategy()
    batch_idx = 1
    text_field = "text"
    full_articles = False

    # Mock tokenizer behavior for single document processing
    mock_processor.tokenizer.segment_sentences.side_effect = lambda text: text.split(
        ". ",
    )

    result = strategy.process(
        mock_processor,
        batch_idx,
        sample_batch,
        text_field,
        full_articles,
    )

    # Check result structure
    assert len(result) == 3
    assert "article1" in result
    assert "article2" in result
    assert "empty_article" in result

    # Check correct calls were made to the tokenizer
    assert mock_processor.tokenizer.segment_sentences.call_count == 2

    # Check sentence structure
    assert len(result["article1"]["sentences"]) == 2
    assert result["article1"]["sentences"][0]["text"] == "This is article 1"

    # Check empty article handling
    assert result["empty_article"]["sentences"] == []


def test_processing_strategy_selector() -> None:
    """Test ProcessingStrategySelector selection logic."""
    # Create mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.SUPPORTS_BATCH_PROCESSING = True
    mock_tokenizer.SUPPORTS_BATCH_GENERATOR = True

    # Create config
    config = {"worker_id": "test"}

    # Create selector
    selector = ProcessingStrategySelector(mock_tokenizer, config)

    # Test with non-PubMed data
    small_batch = {f"article{i}": {"text": f"Text {i}"} for i in range(5)}
    strategy_name = selector.select_strategy(small_batch)
    assert strategy_name == "single_document"

    # Test with PubMed data, should use generator when available
    selector.is_pubmed = True
    strategy_name = selector.select_strategy(small_batch)
    assert strategy_name == "batch_generator"

    # Test with PubMed data but no generator support
    mock_tokenizer.SUPPORTS_BATCH_GENERATOR = False
    strategy_name = selector.select_strategy(small_batch)
    assert strategy_name == "batch_optimized"


def test_create_strategy() -> None:
    """Test create_strategy factory function."""
    optimized = create_strategy("batch_optimized")
    assert isinstance(optimized, BatchOptimizedStrategy)

    generator = create_strategy("batch_generator")
    assert isinstance(generator, BatchGeneratorStrategy)

    single = create_strategy("single_document")
    assert isinstance(single, SingleDocumentStrategy)

    # Test invalid strategy
    with pytest.raises(ValueError):
        create_strategy("unknown_strategy")
