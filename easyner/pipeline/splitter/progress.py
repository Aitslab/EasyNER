import time
import logging

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.progress")


class ProgressReporter:
    """
    Handles reporting progress to the main process.
    Separates progress reporting concerns from processing logic.
    """

    def __init__(self, result_queue, worker_id, batch_idx, total_items):
        """
        Initialize a progress reporter.

        Args:
            result_queue: Queue to send progress messages to
            worker_id: ID of the worker reporting progress
            batch_idx: Index of the batch being processed
            total_items: Total number of items to process
        """
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.batch_idx = batch_idx
        self.total_items = total_items
        self.processed_items = 0
        self.last_update_time = time.time()
        self.min_update_interval = 0.5  # seconds
        self.report_frequency = max(
            10, min(500, total_items // 20)
        )  # Report at appropriate intervals

    def update(self, increment=1):
        """
        Update progress by the specified increment.

        Args:
            increment: Number of items processed since last update
        """
        self.processed_items += increment
        current_time = time.time()

        # Report progress if enough time has passed or we've processed enough items
        if (
            current_time - self.last_update_time > self.min_update_interval
            or self.processed_items % self.report_frequency == 0
        ):
            self._send_progress_message()
            self.last_update_time = current_time

    def _send_progress_message(self):
        """Send a progress update message to the main process"""
        self.result_queue.put(
            ("PROGRESS", self.batch_idx, self.processed_items, self.total_items, self.worker_id)
        )
        logger.debug(
            f"[Worker {self.worker_id}] Progress report: {self.processed_items}/{self.total_items} "
            f"({self.processed_items / self.total_items * 100:.1f}%)"
        )

    def report_completion(self, num_articles, processing_time):
        """
        Report task completion.

        Args:
            num_articles: Number of articles processed
            processing_time: Time taken to process the batch
        """
        # Ensure we report 100% completion
        self.processed_items = self.total_items
        self._send_progress_message()

        # Report completion
        self.result_queue.put(
            ("COMPLETE", self.batch_idx, num_articles, processing_time, self.worker_id)
        )
        logger.debug(
            f"[Worker {self.worker_id}] Completed batch {self.batch_idx}: "
            f"{num_articles} articles in {processing_time:.2f}s"
        )
