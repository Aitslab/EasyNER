import time
import logging
import sys
from multiprocessing import Queue, Process, Value, cpu_count
from tqdm import tqdm
from queue import Empty  # Import Empty exception
from typing import Union
import tabulate  # Import tabulate for nice tables
import datetime

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

    # Add worker_id to config for logging purposes
    config["worker_id"] = worker_id

    logger.debug(f"[Worker {worker_id}] Starting")

    try:
        # Initialize components for this worker
        logger.debug(f"[Worker {worker_id}] Initializing tokenizer")
        if config["tokenizer"].lower() == "spacy":
            tokenizer = SpacyTokenizer(model_name=config.get("model", "en_core_web_sm"))
            # Set optimal batch size for PubMed documents if using batch processing
            if config.get("pubmed_bulk", False):
                # For PubMed, use a moderate batch size to balance performance and memory usage
                spacy_batch_size = config.get("spacy_batch_size", 64)  # Reduced from default 100
                tokenizer.nlp.batch_size = spacy_batch_size
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

        # Signal that worker is initialized and ready
        result_queue.put(
            (
                "WORKER_READY",
                worker_id,
                tokenizer.SUPPORTS_BATCH_PROCESSING,
                tokenizer.SUPPORTS_BATCH_GENERATOR,
            )
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
        # Add counters for batch processing statistics
        self.batch_processing_started = 0
        self.batch_processing_completed = 0
        self.log_verbosity = config.get("log_verbosity", "normal")  # normal, verbose, quiet

        # Statistics for summary
        self.start_time = None
        self.end_time = None
        self.articles_processed = 0
        self.batches_completed = 0
        self.processing_times = []
        self.worker_stats = {}

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
        last_log_time = time.time()
        log_interval = 30  # seconds between summary logs

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
                articles_per_batch = (
                    current_articles / batches_completed if batches_completed > 0 else 0
                )
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

            # Update progress bar with clear, concise message
            if pbar is not None:
                pbar.set_description(
                    f"Progress: {batches_completed}/{total_batches} batches | "
                    f"Articles: {current_articles:,} | "
                    f"Speed: {speed:.1f} art/s | "
                    f"ETA: {eta}"
                )
                pbar.update(0)  # Force refresh

            # Log statistics periodically to file, only with summary information
            current_time = time.time()
            if current_time - last_log_time >= log_interval and batches_completed > 0:
                # Write a concise progress summary to the log
                percent_done = (batches_completed / total_batches * 100) if total_batches > 0 else 0
                logger.debug(
                    f"Progress: {batches_completed}/{total_batches} batches "
                    f"({percent_done:.1f}%) | "
                    f"{current_articles:,} articles | "
                    f"{active_processes} workers | "
                    f"{speed:.1f} art/s"
                )
                last_log_time = current_time

            # Sleep for update interval
            time.sleep(self.update_interval)

    def _print_summary(
        self, elapsed_time, articles_processed, batches_completed, worker_stats=None
    ):
        """Print a formatted summary of the processing run"""
        # Convert elapsed time to a readable format
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.1f} seconds"
        elif elapsed_time < 3600:
            time_str = f"{elapsed_time/60:.1f} minutes"
        else:
            time_str = f"{elapsed_time/3600:.1f} hours"

        # Calculate articles per second
        speed = articles_processed / elapsed_time if elapsed_time > 0 else 0

        # Create summary table
        summary_data = [
            ["Total runtime", time_str],
            ["Batches processed", f"{batches_completed:,}"],
            ["Articles processed", f"{articles_processed:,}"],
            ["Processing speed", f"{speed:.1f} articles/second"],
        ]

        # Add average batch size if we have data
        if batches_completed > 0:
            avg_batch_size = articles_processed / batches_completed
            summary_data.append(["Average batch size", f"{avg_batch_size:.1f} articles"])

        # Format as a nice table
        summary_table = tabulate.tabulate(
            summary_data, headers=["Metric", "Value"], tablefmt="simple"
        )

        # Log the summary table
        logger.info(
            f"\n===== PROCESSING SUMMARY =====\n{summary_table}\n============================="
        )

        # Also print to stdout for immediate feedback
        print(f"\n===== PROCESSING SUMMARY =====\n{summary_table}\n=============================")

        # If we have worker stats, print detailed worker performance
        if worker_stats and len(worker_stats) > 0:
            worker_data = [
                [
                    f"Worker {worker_id}",
                    stats["batches"],
                    f"{stats['articles']:,}",
                    f"{stats['time']:.1f}s",
                ]
                for worker_id, stats in sorted(worker_stats.items())
            ]

            # Only show this in debug mode as it can be lengthy
            if logger.level <= logging.DEBUG:
                worker_table = tabulate.tabulate(
                    worker_data,
                    headers=["Worker ID", "Batches", "Articles", "Total Time"],
                    tablefmt="simple",
                )
                logger.debug(
                    f"\n==== WORKER PERFORMANCE ====\n{worker_table}\n==========================="
                )

    def run(self):
        """Run the splitter pipeline"""
        # Record start time for total execution time
        self.start_time = time.time()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Starting splitter processing at {timestamp}")

        self._initialize_components()
        logger.info(f"Using {self.cpu_limit} CPU cores")

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
        worker_stats = {}  # {worker_id: {'batches': count, 'articles': count, 'time': seconds}}

        # Get the number of batches to process
        if self.is_pubmed:
            batch_files = self.loader.load_data()
            total_batches = len(batch_files)
            logger.info(f"Found {total_batches:,} PubMed batch files to process")
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
                f"Creating {total_batches:,} batches from {len(articles):,} articles (batch size: {batch_size})"
            )

        # Re-log the final CPU core count after potential adjustment
        logger.info(f"Starting splitter workers with {self.cpu_limit} CPU cores")

        # Start workers
        processes = []
        for i in range(self.cpu_limit):
            p = Process(target=worker_process, args=(task_queue, result_queue, self.config, i))
            p.daemon = True
            p.start()
            processes.append(p)
            with active_processes_val.get_lock():
                active_processes_val.value += 1
            logger.debug(f"Started worker process {i} (PID: {p.pid})")
            # Initialize worker statistics
            worker_stats[i] = {"batches": 0, "articles": 0, "time": 0.0}

        # Wait for worker initialization and report summary
        workers_ready = 0
        workers_with_batch_support = 0
        worker_init_start = time.time()
        worker_init_timeout = 60  # seconds

        # Process worker initialization messages until all workers are ready
        while workers_ready < self.cpu_limit:
            try:
                result = result_queue.get(timeout=0.5)

                if result[0] == "WORKER_READY":
                    workers_ready += 1
                    worker_id = result[1]

                    # Check if worker supports batch processing
                    if len(result) > 2 and result[2]:
                        workers_with_batch_support += 1

                    # Progress log at intervals
                    if workers_ready == self.cpu_limit:
                        logger.info(f"All {self.cpu_limit} workers initialized successfully")
                    elif workers_ready == 1 or workers_ready % 10 == 0:
                        # Only log at significant milestones to reduce verbosity
                        logger.debug(
                            f"Worker initialization: {workers_ready}/{self.cpu_limit} workers ready"
                        )

                # Handle any unexpected early messages
                elif result[0] == "ERROR":
                    logger.error(f"Error during worker initialization: {result[1]}")

            except Empty:
                # Check for timeout during initialization
                if time.time() - worker_init_start > worker_init_timeout:
                    logger.warning(
                        f"Some workers failed to initialize: {workers_ready}/{self.cpu_limit} ready after {worker_init_timeout}s"
                    )
                    break

        # Log batch processing capability if PubMed mode
        if self.is_pubmed:
            if workers_with_batch_support > 0:
                logger.info(
                    f"Using optimized batch processing: {workers_with_batch_support}/{self.cpu_limit} workers support it"
                )
            else:
                logger.warning(
                    "No workers support batch processing. PubMed processing will be slower."
                )

        # Start stats updater in a separate thread
        import threading

        # Create a clean progress bar with a proper description
        with tqdm(
            total=total_batches,
            desc="Starting...",
            unit="batch",
            # Make progress bar more log-friendly
            disable=False,  # Force enable the progress bar even if stderr is not a TTY
            miniters=1,  # Update when actual progress is made
            file=sys.stdout,  # Explicitly write to stderr (which Slurm redirects to .err)
            dynamic_ncols=True,  # Adjust progress bar width as terminal changes (less relevant for file output)
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            # Start the stats updater thread with the progress bar
            start_time = time.time()
            stats_thread = threading.Thread(
                target=self._stats_updater,
                args=(
                    articles_processed_val,
                    batches_completed_val,
                    active_processes_val,
                    pbar,
                    running,
                    total_batches,
                    start_time,
                ),
            )
            stats_thread.daemon = True
            stats_thread.start()

            # Enqueue tasks
            if self.is_pubmed:
                for batch_file in batch_files:
                    batch_idx = self.loader.get_batch_index(batch_file)
                    batch_data = self.loader.load_batch(batch_file)
                    task_queue.put((batch_idx, batch_data, True))
                    # Log only first few batches to reduce log volume
                    if batch_idx <= 3:
                        logger.debug(f"Enqueued PubMed batch {batch_idx}: {batch_file}")
            else:
                for batch_idx, batch_ids in enumerate(batches, 1):
                    batch_data = {article_id: articles[article_id] for article_id in batch_ids}
                    task_queue.put((batch_idx, batch_data, False))
                    if batch_idx == 1:  # Only log the first batch
                        logger.debug(f"Enqueued first batch with {len(batch_data)} articles")

            # Add sentinel values to signal workers to stop
            for _ in range(self.cpu_limit):
                task_queue.put(None)
            logger.debug(f"Added {self.cpu_limit} sentinel values to task queue")

            # Process results
            remaining_workers = self.cpu_limit
            last_update_time = time.time()
            last_summary_time = time.time()
            summary_interval = 60  # Create a progress summary every minute

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

                    elif result[0] == "COMPLETE":
                        # Handle completed batch
                        _, batch_idx, num_articles, processing_time, worker_id = result

                        # Update counters
                        with articles_processed_val.get_lock():
                            articles_processed_val.value += num_articles
                            self.articles_processed = articles_processed_val.value
                        with batches_completed_val.get_lock():
                            batches_completed_val.value += 1
                            self.batches_completed = batches_completed_val.value

                        # Update worker statistics
                        if worker_id in worker_stats:
                            worker_stats[worker_id]["batches"] += 1
                            worker_stats[worker_id]["articles"] += num_articles
                            worker_stats[worker_id]["time"] += processing_time

                        # Track processing times for summaries
                        self.processing_times.append(processing_time)

                        # Calculate current stats
                        elapsed = time.time() - start_time
                        completed = batches_completed_val.value
                        total_articles = articles_processed_val.value
                        speed = total_articles / elapsed if elapsed > 0 else 0

                        # Update progress description with stats
                        pbar.set_description(
                            f"Processed: {completed}/{total_batches} | "
                            f"Articles: {total_articles:,} | "
                            f"Speed: {speed:.1f} art/s"
                        )

                        # Update progress bar position - THIS IS CRUCIAL
                        pbar.update(1)

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

                    # Periodically log a concise summary
                    current_time = time.time()
                    if current_time - last_summary_time >= summary_interval:
                        elapsed = current_time - start_time
                        completed = batches_completed_val.value
                        total_articles = articles_processed_val.value

                        if completed > 0:  # Only log if we have some progress
                            progress_pct = (
                                completed / total_batches * 100 if total_batches > 0 else 0
                            )
                            speed = total_articles / elapsed if elapsed > 0 else 0
                            eta_seconds = (
                                (total_batches - completed) * (elapsed / completed)
                                if completed > 0
                                else 0
                            )

                            # Format ETA nicely
                            if eta_seconds < 60:
                                eta = f"{eta_seconds:.0f} seconds"
                            elif eta_seconds < 3600:
                                eta = f"{eta_seconds/60:.1f} minutes"
                            else:
                                eta = f"{eta_seconds/3600:.1f} hours"

                            # Log a summary line that's clearly distinguished from the progress bar
                            logger.info(
                                f"PROGRESS SUMMARY: {completed}/{total_batches} batches ({progress_pct:.1f}%), "
                                f"{total_articles:,} articles, {speed:.1f} art/s, "
                                f"ETA: {eta}"
                            )

                        last_summary_time = current_time

                except Empty:
                    # Check if workers are still alive but not reporting
                    if time.time() - last_update_time > 30:  # No updates for 30 seconds
                        active_count = sum(1 for p in processes if p.is_alive())
                        if active_count < remaining_workers:
                            logger.warning(
                                f"Some workers seem to have died. Only {active_count}/{remaining_workers} workers active."
                            )
                        last_update_time = time.time()  # Reset timeout

        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=1.0)

        # Stop the stats updater
        with running.get_lock():
            running.value = 0
        stats_thread.join(timeout=1.0)

        # Record end time and calculate total elapsed time
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time

        # Print final summary with nice formatting
        self._print_summary(
            elapsed_time, articles_processed_val.value, batches_completed_val.value, worker_stats
        )

        # Also access the statistics handler if available
        root_logger = logging.getLogger("easyner.pipeline.splitter")
        if hasattr(root_logger, "stats_handler"):
            logger.info(root_logger.stats_handler.get_summary())

        # Save worker statistics
        self.worker_stats = worker_stats


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
    logger.debug(
        f"CPU_LIMIT: {cpu_limit} (cpu_count: {cpu_count()}), using {cpu_limit_to_use} cores"
    )

    logger.info(f"Starting splitter with {cpu_limit_to_use} CPU cores")

    runner = SplitterRunner(splitter_config, cpu_limit_to_use)
    runner.run()

    logger.info("Finished running splitter script.")
    return {}
