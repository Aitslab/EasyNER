import time
import gc
import logging

from .tokenizers import TokenizerBase
from .strategies import ProcessingStrategySelector, create_strategy

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.processor")


class SplitterProcessor:
    def __init__(self, tokenizer: TokenizerBase, output_writer, config, progress_callback=None):
        """
        Initialize a processor with the given tokenizer, writer and configuration.

        Args:
            tokenizer: The tokenizer to use for processing
            output_writer: The writer to use for output
            config: Configuration dictionary
            progress_callback: Optional callback for progress reporting
        """
        self.tokenizer = tokenizer
        self.output_writer = output_writer
        self.config = config
        self.progress_callback = progress_callback
        self.worker_id = config.get("worker_id", "unknown")

        # Initialize the strategy selector
        self.strategy_selector = ProcessingStrategySelector(tokenizer, config)

        # Log tokenizer capabilities
        logger.debug(
            f"[Worker {self.worker_id}] Tokenizer batch processing support: {self.tokenizer.SUPPORTS_BATCH_PROCESSING}"
        )
        logger.debug(
            f"[Worker {self.worker_id}] Tokenizer batch generator support: {self.tokenizer.SUPPORTS_BATCH_GENERATOR}"
        )

        self.max_batch_size = self.config.get("max_tokenizer_batch_size", 5000)
        logger.debug(f"[Worker {self.worker_id}] Max tokenizer batch size: {self.max_batch_size}")

    def record_start_time(self):
        """
        Record the start time of an operation.

        Returns:
            float: Current time
        """
        return time.time()

    def record_elapsed_time(self, start_time):
        """
        Calculate elapsed time since start_time.

        Args:
            start_time: Start time from record_start_time()

        Returns:
            float: Elapsed time in seconds
        """
        return time.time() - start_time

    def process_batch(self, batch_idx, batch, full_articles=None):
        """
        Process a batch of articles.

        Args:
            batch_idx: Index of the batch
            batch: Dictionary of articles to process
            full_articles: Whether full articles are being processed

        Returns:
            tuple: (written_batch_idx, num_articles_written)
        """
        total_articles = len(batch)
        start_batch_time = time.time()  # Track batch start time

        is_pubmed = self.config.get("pubmed_bulk", False)
        default_text_field = "abstract" if is_pubmed else "text"
        text_field = self.config.get("text_field", default_text_field)

        logger.debug(
            f"[Worker {self.worker_id}] Processing batch {batch_idx} with {total_articles} articles"
        )
        logger.debug(
            f"[Worker {self.worker_id}] Using text field: {text_field} (pubmed={is_pubmed})"
        )

        # --- Select and use appropriate processing strategy ---
        strategy_name = self.strategy_selector.select_strategy(batch)
        strategy = create_strategy(strategy_name)

        # Process using the selected strategy
        processed_articles = strategy.process(self, batch_idx, batch, text_field, full_articles)

        # --- Report Progress During Processing ---
        if self.progress_callback:
            # Report progress periodically during processing if the strategy doesn't handle it
            self.progress_callback(batch_idx, total_articles, total_articles)  # Ensure 100%

        # --- Write Output ---
        logger.debug(
            f"[Worker {self.worker_id}] Writing {len(processed_articles)} articles from batch {batch_idx}"
        )
        write_start_time = time.time()
        written_batch_idx, num_articles_written = self.output_writer.write(
            processed_articles, batch_idx, self.tokenizer.__class__.__name__
        )
        write_duration = time.time() - write_start_time
        logger.debug(
            f"[Worker {self.worker_id}] Completed writing batch {batch_idx}: {num_articles_written} articles written in {write_duration:.2f}s"
        )

        # --- Final Stats ---
        batch_duration = time.time() - start_batch_time

        # Log batch completion statistics for the SummaryHandler
        stats = {
            "batch_complete": True,
            "batch_id": batch_idx,
            "articles": total_articles,
            "batch_size": total_articles,
            "processing_time": batch_duration,
            "worker_id": self.worker_id,
        }
        logger.debug(f"Batch {batch_idx} complete.", extra={"statistics": stats})

        # Explicit garbage collection after processing and writing a batch
        del processed_articles
        del batch  # Assuming batch is not needed after this point
        gc.collect()
        logger.debug(
            f"[Worker {self.worker_id}] Garbage collection triggered after batch {batch_idx}"
        )

        return written_batch_idx, num_articles_written
