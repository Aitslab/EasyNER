import logging
import time
import sys
from collections import defaultdict
from .splitter_runner import run_splitter

# Configure logging - Add filtering capabilities


class BatchProcessingFilter(logging.Filter):
    """Filter to remove repetitive batch processing messages"""

    def __init__(self):
        super().__init__()
        self.last_batch_message_time = 0
        self.batch_message_count = 0
        self.suppressed_count = 0
        # Track different message types to report specific suppression counts
        self.message_types = defaultdict(int)

    def filter(self, record):
        message = record.getMessage()

        # Allow all non-batch messages through
        if not any(
            pattern in message
            for pattern in [
                "Using batch processing for",
                "Batch",
                "Processed batch",
                "Processing batch",
            ]
        ):
            return True

        # Track message type
        msg_type = "batch_processing"
        if "Processed batch" in message:
            msg_type = "batch_completion"
        elif "Processing batch" in message:
            msg_type = "batch_start"

        self.message_types[msg_type] += 1

        current_time = time.time()
        # Only let batch messages through occasionally
        if current_time - self.last_batch_message_time > 10:  # Every 10 seconds
            if self.suppressed_count > 0:
                # Add a summary of suppressed messages
                logger = logging.getLogger("easyner.pipeline.splitter")
                logger.debug(
                    f"Suppressed {self.suppressed_count} batch processing messages ({', '.join(f'{k}: {v}' for k, v in self.message_types.items())})"
                )
                self.suppressed_count = 0
                self.message_types.clear()

            self.last_batch_message_time = current_time
            self.batch_message_count += 1
            return True
        else:
            self.suppressed_count += 1
            return False


class SummaryHandler(logging.Handler):
    """Handler that collects statistics for summary reporting"""

    def __init__(self):
        super().__init__()
        self.statistics = {
            "batches_processed": 0,
            "articles_processed": 0,
            "processing_times": [],
            "batch_sizes": [],
            "start_time": time.time(),
        }

    def emit(self, record):
        if not hasattr(record, "statistics"):
            return

        stats = record.statistics
        if "batch_complete" in stats:
            self.statistics["batches_processed"] += 1
            self.statistics["articles_processed"] += stats.get("articles", 0)

            if "processing_time" in stats:
                self.statistics["processing_times"].append(stats["processing_time"])

            if "batch_size" in stats:
                self.statistics["batch_sizes"].append(stats["batch_size"])

    def get_summary(self):
        """Return formatted summary statistics"""
        elapsed = time.time() - self.statistics["start_time"]

        if not self.statistics["processing_times"]:
            return "No statistics collected"

        avg_time = sum(self.statistics["processing_times"]) / len(
            self.statistics["processing_times"]
        )
        avg_batch_size = (
            sum(self.statistics["batch_sizes"]) / len(self.statistics["batch_sizes"])
            if self.statistics["batch_sizes"]
            else 0
        )

        summary = [
            "\n========== SPLITTER SUMMARY ==========",
            f"Total runtime: {elapsed:.2f} seconds",
            f"Batches processed: {self.statistics['batches_processed']}",
            f"Articles processed: {self.statistics['articles_processed']}",
            f"Average batch size: {avg_batch_size:.1f} articles",
            f"Average processing time: {avg_time:.2f} seconds per batch",
            f"Processing speed: {self.statistics['articles_processed']/elapsed:.2f} articles/second",
            "======================================",
        ]

        return "\n".join(summary)


# Configure logging
logger = logging.getLogger("easyner.pipeline.splitter")
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level if not already configured
if not logger.handlers:
    # Force stderr to flush frequently
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    # Set a small buffer size to ensure frequent flushing
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add batch message filter
    batch_filter = BatchProcessingFilter()
    console_handler.addFilter(batch_filter)

    # Create and add statistics handler (doesn't output logs directly)
    stats_handler = SummaryHandler()
    stats_handler.setLevel(logging.INFO)
    logger.addHandler(stats_handler)

    # Make the stats handler available to other modules
    logger.stats_handler = stats_handler

    # Add handler to the logger
    logger.addHandler(console_handler)

__all__ = ["run_splitter"]
