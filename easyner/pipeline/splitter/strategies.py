import logging
from abc import ABC, abstractmethod

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.strategies")


class ProcessingStrategySelector:
    """
    Selects the most appropriate processing strategy based on tokenizer capabilities and data.
    Encapsulates the decision logic for choosing between different processing approaches.
    """

    def __init__(self, tokenizer, config):
        """
        Initialize the strategy selector.

        Args:
            tokenizer: The tokenizer to use for processing
            config: Configuration dictionary
        """
        self.tokenizer = tokenizer
        self.config = config
        self.is_pubmed = config.get("pubmed_bulk", False)
        self.worker_id = config.get("worker_id", "unknown")

    def select_strategy(self, batch):
        """
        Select the most appropriate processing strategy based on tokenizer capabilities and data.

        Args:
            batch: The batch of articles to process

        Returns:
            str: The name of the selected strategy
        """
        total_articles = len(batch)

        # Decision tree for strategy selection
        if self.tokenizer.SUPPORTS_BATCH_GENERATOR and self.is_pubmed and total_articles > 1:
            logger.debug(
                f"[Worker {self.worker_id}] Selected batch generator strategy for batch with {total_articles} articles"
            )
            return "batch_generator"
        elif self.tokenizer.SUPPORTS_BATCH_PROCESSING and self.is_pubmed and total_articles > 1:
            logger.debug(
                f"[Worker {self.worker_id}] Selected batch optimized strategy for batch with {total_articles} articles"
            )
            return "batch_optimized"
        else:
            logger.debug(
                f"[Worker {self.worker_id}] Selected single document strategy for batch with {total_articles} articles"
            )
            return "single_document"


class ProcessingStrategy(ABC):
    """Base class for all processing strategies"""

    @abstractmethod
    def process(self, processor, batch_idx, batch, text_field, full_articles):
        """
        Process a batch using this strategy.

        Args:
            processor: The SplitterProcessor instance
            batch_idx: The index of the current batch
            batch: The batch of articles to process
            text_field: The field containing the text to process
            full_articles: Whether full articles are being processed

        Returns:
            dict: A dictionary of processed articles
        """
        pass


class BaseProcessingStrategy(ProcessingStrategy):
    """
    Base implementation of processing strategy with common functionality
    that specific strategies can inherit from.
    """

    def prepare_batch(self, batch, text_field):
        """
        Extract texts and handle empty articles.

        Args:
            batch: The batch of articles to process
            text_field: The field containing the text to process

        Returns:
            tuple: A tuple of (texts_to_process, article_ids_in_order, empty_articles, original_order)
        """
        texts_to_process = []
        article_ids_in_order = []
        empty_articles = {}
        original_order = list(batch.keys())  # Preserve original order

        for article_id, article_data in batch.items():
            text_to_split = article_data.get(text_field, "")
            title = article_data.get("title", "")
            if text_to_split:
                texts_to_process.append(text_to_split)
                article_ids_in_order.append(article_id)
            else:
                empty_articles[article_id] = {"title": title, "sentences": []}

        return texts_to_process, article_ids_in_order, empty_articles, original_order

    def reconstruct_output(self, processed_articles, empty_articles, original_order):
        """
        Ensure all articles in original batch appear in output.

        Args:
            processed_articles: Articles that have been processed
            empty_articles: Articles with no content to process
            original_order: Original order of articles in the batch

        Returns:
            dict: A dictionary of all articles in the original order
        """
        # Add empty articles
        for article_id, article_data in empty_articles.items():
            processed_articles[article_id] = article_data

        # Ensure original order is preserved
        return {
            article_id: processed_articles.get(article_id, {"sentences": []})
            for article_id in original_order
        }


class BatchGeneratorStrategy(BaseProcessingStrategy):
    """Strategy for processing with batch generators (memory efficient)"""

    def process(self, processor, batch_idx, batch, text_field, full_articles):
        """Process articles using the batch generator approach"""
        start_time = processor.record_start_time()
        worker_id = processor.worker_id
        tokenizer = processor.tokenizer

        processed_articles = {}
        total_articles = len(batch)

        # Extract texts and prepare batch
        texts_to_process, article_ids_in_order, empty_articles, original_order = self.prepare_batch(
            batch, text_field
        )

        logger.debug(
            f"[Worker {worker_id}] Tokenizing {len(texts_to_process)} non-empty articles using batch generator."
        )

        # Process texts using the generator and zip with IDs
        try:
            # The generator yields lists of sentences, one list per input text
            sentence_lists_generator = tokenizer.segment_sentences_batch_generator(texts_to_process)

            for article_id, sentences in zip(article_ids_in_order, sentence_lists_generator):
                # Format output based on pubmed_bulk setting (assuming generator is used mainly for pubmed)
                # Get the original title from the input batch dictionary
                original_title = batch[article_id].get("title", "")
                processed_articles[article_id] = {
                    "title": original_title,
                    "sentences": [{"text": s} for s in sentences],
                }

            # Add articles with empty abstracts
            for article_id, article_data in empty_articles.items():
                processed_articles[article_id] = article_data

        except Exception as e:
            logger.error(
                f"[Worker {worker_id}] Error during batch generation for batch {batch_idx}: {e}",
                exc_info=True,
            )
            # Handle error: potentially mark all articles in this batch as failed or return partial results
            # For simplicity, returning currently processed + empty ones
            for article_id, article_data in empty_articles.items():
                if article_id not in processed_articles:
                    processed_articles[article_id] = article_data

        # Reconstruct the output in original order
        ordered_processed_articles = self.reconstruct_output(
            processed_articles, empty_articles, original_order
        )

        elapsed = processor.record_elapsed_time(start_time)
        logger.debug(
            f"[Worker {worker_id}] Batch {batch_idx} tokenization (generator) finished in {elapsed:.2f}s for {total_articles} articles."
        )
        return ordered_processed_articles


class BatchOptimizedStrategy(BaseProcessingStrategy):
    """Strategy for standard batch processing"""

    def process(self, processor, batch_idx, batch, text_field, full_articles):
        """Process multiple articles at once using standard batch tokenization"""
        start_time = processor.record_start_time()
        worker_id = processor.worker_id
        tokenizer = processor.tokenizer

        processed_articles = {}
        total_articles = len(batch)

        # Extract texts and prepare batch
        texts_to_process, article_ids_in_order, empty_articles, original_order = self.prepare_batch(
            batch, text_field
        )

        logger.debug(
            f"[Worker {worker_id}] Tokenizing {len(texts_to_process)} non-empty articles using standard batch."
        )

        # Process non-empty texts in a batch
        if texts_to_process:
            try:
                # This returns a list of lists of sentences
                results = tokenizer.segment_sentences_batch(texts_to_process)

                # Map results back to article IDs
                for article_id, sentences in zip(article_ids_in_order, results):
                    # Format output based on pubmed_bulk setting
                    title = batch[article_id].get("title", "")  # Ensure title is included
                    processed_articles[article_id] = {
                        "title": title,
                        "sentences": [{"text": s} for s in sentences],
                    }

            except Exception as e:
                logger.error(
                    f"[Worker {worker_id}] Error during standard batch processing for batch {batch_idx}: {e}",
                    exc_info=True,
                )
                # Handle error: Mark articles in this batch as failed or return partial results
                # Add successfully processed ones before error, if any, plus empty ones
                for article_id, article_data in empty_articles.items():
                    processed_articles[article_id] = article_data

        # Add articles with empty abstracts
        for article_id, article_data in empty_articles.items():
            processed_articles[article_id] = article_data

        # Reconstruct the output in original order
        ordered_processed_articles = self.reconstruct_output(
            processed_articles, empty_articles, original_order
        )

        elapsed = processor.record_elapsed_time(start_time)
        logger.debug(
            f"[Worker {worker_id}] Batch {batch_idx} tokenization (standard batch) finished in {elapsed:.2f}s for {total_articles} articles."
        )
        return ordered_processed_articles


class SingleDocumentStrategy(BaseProcessingStrategy):
    """Strategy for processing one document at a time"""

    def process(self, processor, batch_idx, batch, text_field, full_articles):
        """Process articles one at a time"""
        start_time = processor.record_start_time()
        worker_id = processor.worker_id
        tokenizer = processor.tokenizer
        is_pubmed = processor.config.get("pubmed_bulk", False)

        processed_articles = {}
        total_articles = len(batch)

        logger.debug(f"[Worker {worker_id}] Processing {total_articles} articles individually.")

        # Process each article individually
        for article_id, article_data in batch.items():
            try:
                text_to_split = article_data.get(text_field, "")
                title = article_data.get("title", "")

                if text_to_split:
                    sentences = tokenizer.segment_sentences(text_to_split)
                    # Format output based on pubmed_bulk setting
                    if is_pubmed:
                        processed_articles[article_id] = {
                            "title": title,
                            "sentences": [{"text": s} for s in sentences],
                        }
                    else:
                        processed_articles[article_id] = {
                            "sentences": sentences,
                        }
                else:
                    # Handle empty text
                    if is_pubmed:
                        processed_articles[article_id] = {
                            "title": title,
                            "sentences": [],
                        }
                    else:
                        processed_articles[article_id] = {
                            "sentences": [],
                        }
            except Exception as e:
                logger.error(
                    f"[Worker {worker_id}] Error processing article {article_id} in batch {batch_idx}: {e}",
                    exc_info=True,
                )
                # Ensure entry exists even on error
                if is_pubmed:
                    processed_articles[article_id] = {
                        "title": article_data.get("title", ""),
                        "sentences": [],
                    }
                else:
                    processed_articles[article_id] = {
                        "sentences": [],
                    }

        elapsed = processor.record_elapsed_time(start_time)
        logger.debug(
            f"[Worker {worker_id}] Batch {batch_idx} individual processing finished in {elapsed:.2f}s for {total_articles} articles."
        )
        return processed_articles


def create_strategy(strategy_name):
    """
    Factory function to create a strategy instance based on name.

    Args:
        strategy_name: Name of the strategy to create

    Returns:
        ProcessingStrategy: An instance of the requested strategy
    """
    strategies = {
        "batch_generator": BatchGeneratorStrategy(),
        "batch_optimized": BatchOptimizedStrategy(),
        "single_document": SingleDocumentStrategy(),
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategies[strategy_name]
