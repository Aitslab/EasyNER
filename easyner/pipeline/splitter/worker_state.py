import logging
import time

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.worker")


class WorkerStateManager:
    """Manages the state of a worker process.

    Centralizes worker state management and communication with main process.
    """

    def __init__(self, worker_id, result_queue):
        """Initialize a worker state manager.

        Args:
            worker_id: ID of the worker
            result_queue: Queue to send messages to the main process

        """
        self.worker_id = worker_id
        self.result_queue = result_queue
        self.current_batch = None
        self.stats = {"batches": 0, "articles": 0, "processing_time": 0}

    def start_batch(self, batch_idx, batch):
        """Record start of batch processing.

        Args:
            batch_idx: Index of the batch being processed
            batch: The batch data to process

        Returns:
            float: Start time of batch processing

        """
        self.current_batch = {
            "idx": batch_idx,
            "size": len(batch),
            "start_time": time.time(),
        }
        logger.debug(
            f"[Worker {self.worker_id}] Starting batch {batch_idx} with {len(batch)} items",
        )
        return self.current_batch["start_time"]

    def complete_batch(self, num_articles_processed):
        """Record batch completion and update stats.

        Args:
            num_articles_processed: Number of articles processed in this batch

        Returns:
            tuple: (batch_idx, elapsed_time)

        """
        if not self.current_batch:
            logger.warning(
                f"[Worker {self.worker_id}] Attempted to complete a batch, but no batch was started",
            )
            return None, 0

        elapsed = time.time() - self.current_batch["start_time"]
        batch_idx = self.current_batch["idx"]

        # Update statistics
        self.stats["batches"] += 1
        self.stats["articles"] += num_articles_processed
        self.stats["processing_time"] += elapsed

        # Report completion
        self.result_queue.put(
            ("COMPLETE", batch_idx, num_articles_processed, elapsed, self.worker_id),
        )

        logger.debug(
            f"[Worker {self.worker_id}] Completed batch {batch_idx} in {elapsed:.2f}s: "
            f"{num_articles_processed} articles",
        )

        # Reset current batch
        self.current_batch = None
        return batch_idx, elapsed

    def report_error(self, error_message, batch_idx=None) -> None:
        """Report error to main process.

        Args:
            error_message: Error message to report
            batch_idx: Optional batch index

        """
        if batch_idx is None and self.current_batch:
            batch_idx = self.current_batch["idx"]

        self.result_queue.put(
            (
                "ERROR",
                f"Worker {self.worker_id} error processing batch {batch_idx}: {error_message}",
                self.worker_id,
            ),
        )
        logger.error(f"[Worker {self.worker_id}] Error: {error_message}")

    def signal_ready(self, supports_batch, supports_generator) -> None:
        """Signal worker is initialized and ready to process batches.

        Args:
            supports_batch: Whether the worker supports batch processing
            supports_generator: Whether the worker supports batch generators

        """
        self.result_queue.put(
            ("WORKER_READY", self.worker_id, supports_batch, supports_generator),
        )
        logger.debug(
            f"[Worker {self.worker_id}] Signaled ready (batch={supports_batch}, generator={supports_generator})",
        )

    def signal_done(self, peak_memory_mb=0.0) -> None:
        """Signal worker has completed all work.

        Args:
            peak_memory_mb (float): Peak memory usage in MiB observed by the worker.

        """
        # Include peak_memory_mb in the message tuple
        self.result_queue.put(("WORKER_DONE", self.worker_id, peak_memory_mb))
        logger.debug(
            f"[Worker {self.worker_id}] Signaled done (Peak Mem: {peak_memory_mb:.1f} MiB)",
        )

    def get_stats(self):
        """Get worker statistics.

        Returns:
            dict: Worker statistics

        """
        # Calculate derived metrics
        if self.stats["batches"] > 0:
            avg_batch_size = self.stats["articles"] / self.stats["batches"]
            avg_time_per_batch = self.stats["processing_time"] / self.stats["batches"]
        else:
            avg_batch_size = 0
            avg_time_per_batch = 0

        if self.stats["articles"] > 0:
            avg_time_per_article = (
                self.stats["processing_time"] / self.stats["articles"]
            )
        else:
            avg_time_per_article = 0

        return {
            "worker_id": self.worker_id,
            "batches": self.stats["batches"],
            "articles": self.stats["articles"],
            "time": self.stats["processing_time"],
            "avg_batch_size": avg_batch_size,
            "avg_time_per_batch": avg_time_per_batch,
            "avg_time_per_article": avg_time_per_article,
        }
