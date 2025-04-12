import time
import logging
from multiprocessing import Queue, Process, Value, cpu_count
from tqdm import tqdm
from queue import Empty  # Import Empty exception
from typing import Union

from .tokenizers import SpacyTokenizer, NLTKTokenizer
from .writers import JSONWriter
from .loaders import StandardLoader, PubMedLoader
from .splitter_processor import SplitterProcessor

from easyner.config import load_config

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.runner")
config = load_config()


def make_batches(list_id, n):
    """Yield n-size batches from list of ids"""
    for i in range(0, len(list_id), n):
        yield list_id[i : i + n]


def worker_process(task_queue, result_queue, config, worker_id):
    """Worker process that processes tasks from the queue"""
    tokenizer = None
    writer = None
    processor = None
    logger.debug(f"[Worker {worker_id}] Starting")  # Debug log instead of print

    try:
        # Initialize components for this worker
        logger.debug(f"[Worker {worker_id}] Initializing tokenizer")
        if config["tokenizer"].lower() == "spacy":
            tokenizer = SpacyTokenizer(model_name=config.get("model", "en_core_web_sm"))
            # Set optimal batch size for PubMed documents if using batch processing
            if config.get("pubmed_bulk", False):
                # For PubMed, we can use a larger batch size since documents are small
                tokenizer.nlp.batch_size = config.get("spacy_batch_size", 100)
                logger.debug(
                    f"[Worker {worker_id}] Set spaCy batch size to {tokenizer.nlp.batch_size} for PubMed"
                )
        elif config["tokenizer"].lower() == "nltk":
            tokenizer = NLTKTokenizer()
        else:
            raise ValueError(f"Unknown tokenizer: {config['tokenizer']}")

        logger.debug(f"[Worker {worker_id}] Initializing writer")
        writer = JSONWriter(
            output_folder=config["output_folder"],
            output_file_prefix=config["output_file_prefix"],
        )

        # Define progress callback function
        def report_progress(batch_idx, processed, total):
            result_queue.put(("PROGRESS", batch_idx, processed, total, worker_id))

        logger.debug(f"[Worker {worker_id}] Initializing processor")
        processor = SplitterProcessor(
            tokenizer=tokenizer,
            output_writer=writer,
            config=config,
            progress_callback=report_progress,
        )
        logger.debug(f"[Worker {worker_id}] Initialization complete")

        # Process tasks from the queue
        while True:
            logger.debug(f"[Worker {worker_id}] Waiting for task...")
            task = task_queue.get()
            logger.debug(f"[Worker {worker_id}] Got task from queue")

            if task is None:  # None is our sentinel value to stop
                logger.debug(f"[Worker {worker_id}] Received sentinel. Exiting.")
                break

            batch_idx, batch_data, full_articles = task
            logger.debug(
                f"[Worker {worker_id}] Processing batch {batch_idx} with {len(batch_data)} items"
            )

            try:
                # Add inner try-except to catch errors during processing a specific batch
                start_time = time.time()

                # Process the batch
                processed_batch_idx, num_articles_processed = processor.process_batch(
                    batch_idx, batch_data, full_articles
                )
                processing_time = time.time() - start_time

                logger.debug(
                    f"[Worker {worker_id}] Processed batch {batch_idx} in {processing_time:.2f}s, sending result"
                )
                # Send results back (batch_idx, num_articles, processing_time)
                result_queue.put(
                    (
                        "COMPLETE",
                        processed_batch_idx,
                        num_articles_processed,
                        processing_time,
                        worker_id,
                    )
                )
                logger.debug(f"[Worker {worker_id}] Result sent for batch {processed_batch_idx}")

            except Exception as batch_exc:
                logger.error(
                    f"[Worker {worker_id}] Error processing batch {batch_idx}: {batch_exc}",
                    exc_info=True,
                )
                # If an error occurs during batch processing, report it
                result_queue.put(
                    ("ERROR", f"Error processing batch {batch_idx}: {batch_exc}", worker_id)
                )
                logger.debug(f"[Worker {worker_id}] Error report sent")

    except Exception as init_exc:
        logger.error(f"[Worker {worker_id}] Critical error: {init_exc}", exc_info=True)
        # If an error occurs during initialization or getting from queue, report it
        result_queue.put(("ERROR", f"Worker {worker_id} failed: {init_exc}", worker_id))
    finally:
        # Signal that this worker is done, regardless of success or failure
        logger.debug(f"[Worker {worker_id}] Signaling WORKER_DONE")
        result_queue.put(("WORKER_DONE", worker_id))
        logger.debug(f"[Worker {worker_id}] Exit complete")


class SplitterRunner:
    def __init__(self, config, cpu_limit):
        """
        Initialize runner with configuration

        Args:
            config: Dictionary containing all configuration parameters
            cpu_limit: Integer specifying the number of CPU cores to use
        """
        self.cpu_limit = cpu_limit
        self.config = config
        self.update_interval = config.get("stats_update_interval", 1.0)  # Update stats every second

    def _initialize_components(self):
        """Initialize loader component"""
        # Initialize loader based on config
        if self.config.get("pubmed_bulk", False):
            self.loader = PubMedLoader(
                input_folder=self.config["input_path"],
                limit=self.config.get("file_limit", "ALL"),
                key=self.config.get("key", "n"),
            )
            self.is_pubmed = True
            logger.info(f"Initialized PubMed loader for {self.config['input_path']}")
        else:
            self.loader = StandardLoader(input_path=self.config["input_path"])
            self.is_pubmed = False
            logger.info(f"Initialized Standard loader for {self.config['input_path']}")

    def _stats_updater(
        self,
        articles_processed_val,
        batches_completed_val,
        active_processes_val,
        pbar,
        running,
        total_batches,
        start_time,
    ):
        """Update statistics in the progress bar using Value objects"""
        while running.value:
            # Read values from shared memory
            current_articles = articles_processed_val.value
            batches_completed = batches_completed_val.value
            active_processes = active_processes_val.value

            # Calculate speed
            elapsed = max(0.001, time.time() - start_time)  # Avoid division by zero
            speed = current_articles / elapsed

            # Calculate ETA - this is approximate since we don't know total articles
            if batches_completed > 0 and total_batches > 0:
                articles_per_batch = current_articles / batches_completed
                remaining_batches = total_batches - batches_completed
                eta_seconds = (remaining_batches * articles_per_batch) / max(0.1, speed)

                # Format ETA nicely
                if eta_seconds < 60:
                    eta = f"{eta_seconds:.0f}s"
                elif eta_seconds < 3600:
                    eta = f"{eta_seconds/60:.1f}m"
                else:
                    eta = f"{eta_seconds/3600:.1f}h"
            else:
                eta = "calculating..."

            # If pbar exists, update it with stats
            if pbar is not None:
                pbar.set_description(
                    f"Files: {batches_completed}/{total_batches} | "
                    f"Articles: {current_articles} | "
                    f"Speed: {speed:.2f} art/s | "
                    f"ETA: {eta}"
                )

            # Log statistics periodically (every 10 update intervals)
            if batches_completed % 10 == 0 and batches_completed > 0:
                logger.info(
                    f"Progress: {batches_completed}/{total_batches} batches "
                    f"({(batches_completed/total_batches*100):.1f}%), "
                    f"{current_articles} articles processed, "
                    f"{active_processes} active workers, "
                    f"speed: {speed:.2f} articles/s"
                )

            # Sleep for update interval
            time.sleep(self.update_interval)

    def run(self):
        """Run the splitter pipeline"""
        self._initialize_components()
        logger.info(f"Starting splitter with {self.cpu_limit} CPU cores")

        # Create queues for task distribution and result collection
        task_queue = Queue()
        result_queue = Queue()

        # Create shared counters for statistics
        articles_processed_val = Value("i", 0)
        batches_completed_val = Value("i", 0)
        active_processes_val = Value("i", 0)
        running = Value("i", 1)  # Flag to indicate whether processing is ongoing

        # Keep track of in-progress batches
        batch_progress = {}  # {batch_idx: (processed, total)}
        worker_batch_map = {}  # {worker_id: batch_idx}

        # Get the number of batches to process
        if self.is_pubmed:
            batch_files = self.loader.load_data()
            total_batches = len(batch_files)
            logger.info(f"Found {total_batches} PubMed batch files to process")
            # Adjust cpu_limit if it's higher than the number of files
            if total_batches < self.cpu_limit:
                logger.info(
                    f"Adjusting CPU limit from {self.cpu_limit} to {total_batches} (number of PubMed files)"
                )
                self.cpu_limit = total_batches
        else:
            articles = self.loader.load_data()
            batch_size = self.config.get("batch_size", 100)
            batches = list(make_batches(list(articles.keys()), batch_size))
            total_batches = len(batches)
            logger.info(
                f"Creating {total_batches} batches from {len(articles)} articles (batch size: {batch_size})"
            )

        # Re-log the final CPU core count after potential adjustment
        logger.info(f"Starting splitter workers with {self.cpu_limit} CPU cores")

        # Start workers (moved after potential cpu_limit adjustment)
        processes = []
        for i in range(self.cpu_limit):
            p = Process(target=worker_process, args=(task_queue, result_queue, self.config, i))
            p.daemon = True
            p.start()
            processes.append(p)
            with active_processes_val.get_lock():
                active_processes_val.value += 1
            logger.debug(f"Started worker process {i} (PID: {p.pid})")

        # Start stats updater in a separate thread
        import threading

        start_time = time.time()
        stats_thread = threading.Thread(
            target=self._stats_updater,
            args=(
                articles_processed_val,
                batches_completed_val,
                active_processes_val,
                None,  # We'll replace this with the progress bar once created
                running,
                total_batches,
                start_time,
            ),
        )
        stats_thread.daemon = True

        # Create progress bar
        with tqdm(total=total_batches, desc="Processing", unit="batch") as pbar:
            # Update the stats thread with the progress bar
            stats_thread.start()

            # Enqueue tasks
            if self.is_pubmed:
                for batch_file in batch_files:
                    batch_idx = self.loader.get_batch_index(batch_file)
                    batch_data = self.loader.load_batch(batch_file)
                    task_queue.put((batch_idx, batch_data, True))
                    logger.debug(f"Enqueued PubMed batch {batch_idx}: {batch_file}")
            else:
                for batch_idx, batch_ids in enumerate(batches, 1):
                    batch_data = {article_id: articles[article_id] for article_id in batch_ids}
                    task_queue.put((batch_idx, batch_data, False))
                    if batch_idx % 100 == 0 or batch_idx == 1:  # Log first batch and every 100th
                        logger.debug(f"Enqueued batch {batch_idx} with {len(batch_data)} articles")

            # Add sentinel values to signal workers to stop
            for _ in range(self.cpu_limit):
                task_queue.put(None)
            logger.debug(f"Added {self.cpu_limit} sentinel values to task queue")

            # Process results
            remaining_workers = self.cpu_limit
            last_update_time = time.time()

            while remaining_workers > 0:
                try:
                    # Wait for a result with timeout
                    result = result_queue.get(timeout=1.0)
                    last_update_time = time.time()

                    # Process different types of messages
                    if result[0] == "PROGRESS":
                        # Handle progress update
                        _, batch_idx, processed, total, worker_id = result
                        batch_progress[batch_idx] = (processed, total)
                        worker_batch_map[worker_id] = batch_idx

                        # Calculate in-progress articles
                        in_progress_articles = sum(
                            processed for processed, _ in batch_progress.values()
                        )

                        # Update progress display with both columns
                        pbar.set_description(
                            f"Articles: {articles_processed_val.value + in_progress_articles} | "
                            f"Active processes: {active_processes_val.value}/{self.cpu_limit}"
                        )

                    elif result[0] == "COMPLETE":
                        # Handle completed batch
                        _, batch_idx, num_articles, processing_time, worker_id = result

                        # Update counters
                        with articles_processed_val.get_lock():
                            articles_processed_val.value += num_articles
                        with batches_completed_val.get_lock():
                            batches_completed_val.value += 1

                        # Remove from in-progress tracking
                        if batch_idx in batch_progress:
                            del batch_progress[batch_idx]
                        if worker_id in worker_batch_map:
                            del worker_batch_map[worker_id]

                        # Update progress bar
                        pbar.update(1)

                        # Log completion of batch at debug level
                        logger.debug(
                            f"Completed batch {batch_idx} with {num_articles} articles in {processing_time:.2f}s"
                        )

                    elif result[0] == "ERROR":
                        # Handle error
                        error_msg = result[1]
                        logger.error(f"Error received: {error_msg}")

                    elif result[0] == "WORKER_DONE":
                        # Handle worker completion
                        worker_id = result[1]
                        with active_processes_val.get_lock():
                            active_processes_val.value = max(0, active_processes_val.value - 1)
                        remaining_workers -= 1

                        logger.debug(
                            f"Worker {worker_id} has completed all tasks. {remaining_workers} workers remaining."
                        )

                        # Remove from in-progress tracking if needed
                        if worker_id in worker_batch_map:
                            batch_idx = worker_batch_map[worker_id]
                            del worker_batch_map[worker_id]

                except Empty:
                    # Check if workers are still alive but not reporting
                    if time.time() - last_update_time > 10:  # No updates for 10 seconds
                        active_count = sum(1 for p in processes if p.is_alive())
                        logger.warning(
                            f"No results received from workers for 10 seconds. "
                            f"Active worker processes reported by OS: {active_count}"
                        )
                        last_update_time = time.time()  # Reset timeout

        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=1.0)

        # Stop the stats updater
        with running.get_lock():
            running.value = 0
        stats_thread.join(timeout=1.0)

        logger.info(
            f"Processing complete! Processed {articles_processed_val.value} articles "
            f"in {batches_completed_val.value} batches."
        )


def run_splitter(
    splitter_config: dict, ignore: bool = False, cpu_limit: Union[int, None] = None
) -> dict:
    """
    Run the splitter pipeline with the given configuration.

    Args:
        splitter_config: Dictionary containing all configuration parameters
        ignore: If True, skip execution
        cpu_limit: Integer specifying the number of CPU cores to use, or None for default

    Returns:
        Dict: Empty dict (for compatibility with pipeline architecture)
    """
    if ignore:
        logger.info("Ignoring script: splitter.")
        return {}

    cpu_limit_to_use = min(cpu_limit, cpu_count()) if cpu_limit else cpu_count()
    logger.info(
        f"CPU_LIMIT: {cpu_limit} (cpu_count: {cpu_count()}), using {cpu_limit_to_use} cores"
    )

    logger.info(f"Starting splitter with {cpu_limit_to_use} CPU cores")

    runner = SplitterRunner(splitter_config, cpu_limit_to_use)
    runner.run()

    logger.info("Finished running splitter script.")
    return {}
