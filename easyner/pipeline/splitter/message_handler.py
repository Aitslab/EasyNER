import time
import logging
import datetime

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.message_handler")


class MessageHandler:
    """
    Handles messages from worker processes in the main process.
    Implements a message handler pattern for different message types.
    """

    def __init__(self, runner):
        """
        Initialize a message handler.

        Args:
            runner: The SplitterRunner instance
        """
        self.runner = runner
        self.handlers = {
            "PROGRESS": self.handle_progress,
            "COMPLETE": self.handle_complete,
            "ERROR": self.handle_error,
            "WORKER_READY": self.handle_worker_ready,
            "WORKER_DONE": self.handle_worker_done,
        }

    def handle(self, message):
        """
        Dispatch message to appropriate handler.

        Args:
            message: The message to handle

        Returns:
            bool: Whether the message was handled
        """
        if not message or not isinstance(message, tuple) or len(message) < 1:
            logger.warning(f"Received invalid message: {message}")
            return False

        message_type = message[0]
        if message_type in self.handlers:
            self.handlers[message_type](message)
            return True
        else:
            logger.warning(f"Unknown message type: {message_type}")
            return False

    def handle_progress(self, message):
        """
        Handle progress update from worker.

        Args:
            message: The progress message (PROGRESS, batch_idx, processed, total, worker_id)
        """
        _, batch_idx, processed, total, worker_id = message
        self.runner.batch_progress[batch_idx] = (processed, total)
        self.runner.worker_batch_map[worker_id] = batch_idx

        # Log progress at appropriate intervals
        if total > 0:
            progress_percent = processed / total * 100
            if progress_percent % 10 == 0 and processed > 0:  # Log at 10% intervals
                logger.debug(f"Batch {batch_idx}: {processed}/{total} ({progress_percent:.1f}%)")

    def handle_complete(self, message):
        """
        Handle batch completion from worker.

        Args:
            message: The complete message (COMPLETE, batch_idx, num_articles, processing_time, worker_id)
        """
        _, batch_idx, num_articles, processing_time, worker_id = message

        # Log receipt of completion message for debugging
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        logger.debug(
            f"[{timestamp}] Received COMPLETE message for batch {batch_idx} from worker {worker_id}"
        )

        # Update counters
        with self.runner.articles_processed_val.get_lock():
            self.runner.articles_processed_val.value += num_articles
            self.runner.articles_processed = self.runner.articles_processed_val.value

        with self.runner.batches_completed_val.get_lock():
            self.runner.batches_completed_val.value += 1
            self.runner.batches_completed = self.runner.batches_completed_val.value

        # Update worker statistics
        if worker_id in self.runner.worker_stats:
            self.runner.worker_stats[worker_id]["batches"] += 1
            self.runner.worker_stats[worker_id]["articles"] += num_articles
            self.runner.worker_stats[worker_id]["time"] += processing_time

        # Update processing times for summaries
        self.runner.processing_times.append(processing_time)

        # Update progress bar - with debug logging
        current_completion = self.runner.batches_completed_val.value
        logger.debug(f"[{timestamp}] Progress update: batches_completed_val={current_completion}")

        try:
            # Force refresh of progress bar
            self.runner.pbar.update(1)
            logger.debug(
                f"[{timestamp}] Progress bar updated by 1 unit (now at {self.runner.pbar.n}/{self.runner.pbar.total})"
            )
        except Exception as e:
            logger.error(f"[{timestamp}] Error updating progress bar: {str(e)}")

        # Update progress description with stats including timestamp
        elapsed = time.time() - self.runner.start_time
        total_articles = self.runner.articles_processed_val.value
        speed = total_articles / elapsed if elapsed > 0 else 0

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        description = (
            f"[{timestamp}] Processed: {self.runner.batches_completed_val.value}/{self.runner.total_batches} | "
            f"Articles: {total_articles:,} | "
            f"Speed: {speed:.1f} art/s"
        )

        try:
            self.runner.pbar.set_description(description)
            # Force a refresh by calling update with 0
            self.runner.pbar.update(0)
            logger.debug(f"[{timestamp}] Progress bar description updated")
        except Exception as e:
            logger.error(f"[{timestamp}] Error updating progress bar description: {str(e)}")

        # Remove from in-progress tracking if needed
        if worker_id in self.runner.worker_batch_map:
            batch_idx = self.runner.worker_batch_map[worker_id]
            if batch_idx in self.runner.batch_progress:
                del self.runner.batch_progress[batch_idx]
            del self.runner.worker_batch_map[worker_id]

    def handle_error(self, message):
        """
        Handle error from worker.

        Args:
            message: The error message (ERROR, error_msg, worker_id)
        """
        _, error_msg, worker_id = message
        logger.error(f"Error from worker {worker_id}: {error_msg}")

        # Could implement additional error handling here, like retry logic

    def handle_worker_ready(self, message):
        """
        Handle worker ready notification.

        Args:
            message: The worker ready message (WORKER_READY, worker_id, supports_batch, supports_generator)
        """
        _, worker_id, supports_batch, supports_generator = message

        # Record capability support
        if supports_batch:
            self.runner.workers_with_batch_support += 1

        if supports_generator:
            self.runner.workers_with_generator_support += 1

        # Update ready count and notify runner
        self.runner.workers_ready += 1

        # Log progress at appropriate intervals or when all workers are ready
        if (
            self.runner.workers_ready == self.runner.cpu_limit
            or self.runner.workers_ready % 5 == 0
            or self.runner.workers_ready == 1
        ):
            logger.info(
                f"Worker initialization: {self.runner.workers_ready}/{self.runner.cpu_limit} workers ready"
            )

    def handle_worker_done(self, message):
        """
        Handle worker completion notification.

        Args:
            message: The worker done message (WORKER_DONE, worker_id)
        """
        _, worker_id = message
        with self.runner.active_processes_val.get_lock():
            self.runner.active_processes_val.value = max(
                0, self.runner.active_processes_val.value - 1
            )
        self.runner.remaining_workers -= 1

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        logger.debug(
            f"[{timestamp}] Worker {worker_id} has completed all tasks. {self.runner.remaining_workers} workers remaining."
        )
