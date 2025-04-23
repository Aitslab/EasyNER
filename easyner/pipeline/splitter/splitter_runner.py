import time
import logging
import sys
from multiprocessing import Queue, Process, Value, cpu_count
from tqdm import tqdm
from queue import Empty  # Import Empty exception
from typing import Union
import tabulate  # Import tabulate for nice tables
import datetime
import psutil  # <-- Add psutil
import os  # <-- Add os

from .tokenizers import SpacyTokenizer, NLTKTokenizer
from .writers import JSONWriter
from .loaders import StandardLoader, PubMedLoader
from .splitter_processor import SplitterProcessor
from .worker_state import WorkerStateManager
from .progress import ProgressReporter
from .message_handler import MessageHandler
from .stats_manager import StatsManager

from easyner.config import config_manager

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.runner")
config = config_manager.load_config()


def make_batches(list_id, n):
    """Yield n-size batches from list of ids"""
    for i in range(0, len(list_id), n):
        yield list_id[i : i + n]


def worker_process(task_queue, result_queue, config, worker_id):
    """
    Worker process that processes tasks from the queue.
    Uses WorkerStateManager for state management and ProgressReporter for progress reporting.

    Args:
        task_queue: Queue to get tasks from
        result_queue: Queue to send results to
        config: Configuration dictionary
        worker_id: ID of this worker
    """
    # Add worker_id to config for logging purposes
    config["worker_id"] = worker_id

    # Initialize state manager
    state_manager = WorkerStateManager(worker_id, result_queue)

    logger.debug(f"[Worker {worker_id}] Starting")

    # --- Memory Tracking ---
    peak_memory_mb = 0
    try:
        process = psutil.Process(os.getpid())
        # Initial memory reading
        peak_memory_mb = process.memory_info().rss / (1024 * 1024)
    except Exception as e:
        logger.warning(f"[Worker {worker_id}] Could not initialize memory tracking: {e}")
        process = None
    # --- End Memory Tracking ---

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
        # Get IO format from config
        io_format = config.get("io_format", "json")
        writer = JSONWriter(
            output_folder=config["output_folder"],
            output_file_prefix=config["output_file_prefix"],
            io_format=io_format,
        )

        # Define progress callback function that uses ProgressReporter
        def report_progress(batch_idx, processed, total):
            # Create a progress reporter on-demand if needed
            if not hasattr(report_progress, "reporters"):
                report_progress.reporters = {}

            # Get or create reporter for this batch
            if batch_idx not in report_progress.reporters:
                report_progress.reporters[batch_idx] = ProgressReporter(
                    result_queue, worker_id, batch_idx, total
                )

            # Update progress
            reporter = report_progress.reporters[batch_idx]
            reporter.update(processed - reporter.processed_items)

        logger.debug(f"[Worker {worker_id}] Initializing processor")
        processor = SplitterProcessor(
            tokenizer=tokenizer,
            output_writer=writer,
            config=config,
            progress_callback=report_progress,
        )

        # Signal that worker is initialized and ready
        state_manager.signal_ready(
            tokenizer.SUPPORTS_BATCH_PROCESSING,
            tokenizer.SUPPORTS_BATCH_GENERATOR,
        )

        # Process tasks from the queue
        while True:
            logger.debug(f"[Worker {worker_id}] Waiting for task...")
            task = task_queue.get()
            logger.debug(f"[Worker {worker_id}] Got task from queue")

            if task is None:  # None is our sentinel value to stop
                logger.debug(f"[Worker {worker_id}] Received sentinel. Exiting.")
                break

            batch_idx, batch_data, full_articles = task

            # Use state manager to track batch processing
            state_manager.start_batch(batch_idx, batch_data)

            logger.debug(
                f"[Worker {worker_id}] Processing batch {batch_idx} with {len(batch_data)} items"
            )

            try:
                # Process the batch
                processed_batch_idx, num_articles_processed = processor.process_batch(
                    batch_idx, batch_data, full_articles
                )

                # Mark batch as complete
                _, processing_time = state_manager.complete_batch(num_articles_processed)

                # If we created a progress reporter for this batch, report 100% completion
                if hasattr(report_progress, "reporters") and batch_idx in report_progress.reporters:
                    reporter = report_progress.reporters[batch_idx]
                    reporter.report_completion(num_articles_processed, processing_time)
                    # Clean up reporter
                    del report_progress.reporters[batch_idx]

                # --- Update Peak Memory ---
                if process:
                    try:
                        current_memory_mb = process.memory_info().rss / (1024 * 1024)
                        peak_memory_mb = max(peak_memory_mb, current_memory_mb)
                    except psutil.NoSuchProcess:
                        logger.warning(
                            f"[Worker {worker_id}] Process disappeared during memory check."
                        )
                        process = None  # Stop trying if process is gone
                    except Exception as mem_e:
                        logger.warning(f"[Worker {worker_id}] Error getting memory usage: {mem_e}")
                # --- End Update Peak Memory ---

            except Exception as batch_exc:
                logger.error(
                    f"[Worker {worker_id}] Error processing batch {batch_idx}: {batch_exc}",
                    exc_info=True,
                )
                # Report error using state manager
                state_manager.report_error(str(batch_exc), batch_idx)

    except Exception as init_exc:
        logger.error(f"[Worker {worker_id}] Critical error: {init_exc}", exc_info=True)
        # Report error using state manager
        state_manager.report_error(str(init_exc))
    finally:
        # --- Final Memory Check ---
        if process:
            try:
                current_memory_mb = process.memory_info().rss / (1024 * 1024)
                peak_memory_mb = max(peak_memory_mb, current_memory_mb)
            except Exception:
                pass  # Ignore errors on final check
        # --- End Final Memory Check ---

        # Signal that this worker is done, passing peak memory
        state_manager.signal_done(peak_memory_mb)
        logger.debug(f"[Worker {worker_id}] Final peak memory: {peak_memory_mb:.2f} MiB")


class SplitterRunner:
    def __init__(self, config, cpu_limit):
        """
        Initialize runner with configuration

        Args:
            config: Dictionary containing all configuration parameters
            cpu_limit: Integer specifying the number of CPU cores to use
        """
        self.config = config
        self.cpu_limit = cpu_limit
        self.update_interval = config.get("stats_update_interval", 1.0)
        self.log_verbosity = config.get("log_verbosity", "normal")

        # Initialize counters and state
        self.is_pubmed = False
        self.workers_ready = 0
        self.workers_with_batch_support = 0
        self.workers_with_generator_support = 0
        self.remaining_workers = 0
        self.total_batches = 0

        # Statistics
        self.start_time = None
        self.end_time = None
        self.articles_processed = 0
        self.batches_completed = 0
        self.processing_times = []

        # Runtime state
        self.batch_progress = {}  # {batch_idx: (processed, total)}
        self.worker_batch_map = {}  # {worker_id: batch_idx}
        self.worker_stats = (
            {}
        )  # {worker_id: {'batches': count, 'articles': count, 'time': seconds}}

        # Initialize the stats manager
        self.stats_manager = StatsManager()

    def _initialize_components(self):
        """Initialize loader component based on configuration"""
        # Get IO format from config
        io_format = self.config.get("io_format", "json")

        # Initialize loader based on config
        if self.config.get("pubmed_bulk", False):
            self.loader = PubMedLoader(
                input_folder=self.config["input_path"],
                limit=self.config.get("file_limit", "ALL"),
                key=self.config.get("key", "n"),
                io_format=io_format,
            )
            self.is_pubmed = True
            logger.info(
                f"Initialized PubMed loader for {self.config['input_path']} with format {io_format}"
            )
        else:
            self.loader = StandardLoader(input_path=self.config["input_path"], io_format=io_format)
            self.is_pubmed = False
            logger.info(
                f"Initialized Standard loader for {self.config['input_path']} with format {io_format}"
            )

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
        """
        Update statistics in the progress bar using Value objects

        Args:
            articles_processed_val: Shared counter for processed articles
            batches_completed_val: Shared counter for completed batches
            active_processes_val: Shared counter for active processes
            pbar: Progress bar to update
            running: Flag indicating whether processing is ongoing
            total_batches: Total number of batches to process
            start_time: Start time of processing
        """
        last_log_time = time.time()
        log_interval = 30  # seconds between summary logs
        update_count = 0

        while running.value:
            update_count += 1
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            # Read values from shared memory
            current_articles = articles_processed_val.value
            batches_completed = batches_completed_val.value
            active_processes = active_processes_val.value

            # Calculate speed
            elapsed = max(0.001, time.time() - start_time)  # Avoid division by zero
            speed = current_articles / elapsed

            # Calculate ETA
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

            # Log current progress every 10 updates for debugging
            if update_count % 10 == 0:
                logger.debug(
                    f"[{timestamp}] Stats updater: batches_completed_val={batches_completed}, "
                    f"articles_processed_val={current_articles}, "
                    f"pbar.n={pbar.n}/{pbar.total if pbar else 'N/A'}"
                )

            # Update progress bar with clear, concise message including timestamp
            if pbar is not None:
                try:
                    description = (
                        f"[{timestamp}] Progress: {batches_completed}/{total_batches} batches | "
                        f"Articles: {current_articles:,} | "
                        f"Speed: {speed:.1f} art/s | "
                        f"ETA: {eta}"
                    )
                    pbar.set_description(description)

                    # Ensure position is correct (synchronize tqdm counter with our counter)
                    if pbar.n != batches_completed:
                        # Only fix if needed to avoid excessive refreshes
                        logger.debug(
                            f"[{timestamp}] Fixing progress bar position: {pbar.n} → {batches_completed}"
                        )

                        # Reset position to match our counter
                        pbar.n = batches_completed
                        # Force refresh
                        pbar.refresh()
                except Exception as e:
                    logger.error(
                        f"[{timestamp}] Error updating progress bar in stats_updater: {str(e)}"
                    )

            # Log statistics periodically to file
            current_time = time.time()
            if current_time - last_log_time >= log_interval and batches_completed > 0:
                # Write a concise progress summary to the log
                percent_done = (batches_completed / total_batches * 100) if total_batches > 0 else 0
                logger.info(
                    f"[{timestamp}] Progress: {batches_completed}/{total_batches} batches "
                    f"({percent_done:.1f}%) | "
                    f"{current_articles:,} articles | "
                    f"{active_processes} workers | "
                    f"{speed:.1f} art/s"
                )
                last_log_time = current_time

            # Sleep for update interval
            time.sleep(self.update_interval)

    def run(self):
        """Run the splitter pipeline"""
        # Record start time for total execution time
        self.start_time = time.time()
        self.stats_manager.start_time = self.start_time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Starting splitter processing at {timestamp}")

        # Initialize components
        self._initialize_components()
        logger.info(f"Using {self.cpu_limit} CPU cores")

        # Create queues for task distribution and result collection
        task_queue = Queue()
        result_queue = Queue()

        # Create shared counters for statistics
        self.articles_processed_val = Value("i", 0)
        self.batches_completed_val = Value("i", 0)
        self.active_processes_val = Value("i", 0)
        running = Value("i", 1)  # Flag to indicate whether processing is ongoing

        # Initialize message handler
        message_handler = MessageHandler(self)

        # Get the number of batches to process
        if self.is_pubmed:
            batch_files = self.loader.load_data()
            self.total_batches = len(batch_files)
            logger.info(f"Found {self.total_batches:,} PubMed batch files to process")

            # Adjust cpu_limit if it's higher than the number of files
            if self.total_batches < self.cpu_limit:
                logger.info(
                    f"Adjusting CPU limit from {self.cpu_limit} to {self.total_batches} (number of PubMed files)"
                )
                self.cpu_limit = self.total_batches
        else:
            articles = self.loader.load_data()
            batch_size = self.config.get("batch_size", 100)
            batches = list(make_batches(list(articles.keys()), batch_size))
            self.total_batches = len(batches)
            logger.info(
                f"Creating {self.total_batches:,} batches from {len(articles):,} articles (batch size: {batch_size})"
            )

        # Start workers
        processes = []
        self.remaining_workers = self.cpu_limit

        for i in range(self.cpu_limit):
            p = Process(
                target=worker_process,
                args=(task_queue, result_queue, self.config, i),
            )
            p.daemon = True
            p.start()
            processes.append(p)
            with self.active_processes_val.get_lock():
                self.active_processes_val.value += 1
            logger.debug(f"Started worker process {i} (PID: {p.pid})")

            # Initialize worker statistics
            self.worker_stats[i] = {"batches": 0, "articles": 0, "time": 0.0}

        # Wait for worker initialization
        worker_init_start = time.time()
        worker_init_timeout = 60  # seconds

        # Process worker initialization messages until all workers are ready
        while self.workers_ready < self.cpu_limit:
            try:
                result = result_queue.get(timeout=0.5)
                message_handler.handle(result)
            except Empty:
                # Check for timeout during initialization
                if time.time() - worker_init_start > worker_init_timeout:
                    logger.warning(
                        f"Some workers failed to initialize: {self.workers_ready}/{self.cpu_limit} ready after {worker_init_timeout}s"
                    )
                    break

        # Log batch processing capability if in PubMed mode
        if self.is_pubmed and self.workers_with_batch_support > 0:
            logger.info(
                f"Using optimized batch processing: {self.workers_with_batch_support}/{self.cpu_limit} workers support it"
            )
        elif self.is_pubmed:
            logger.warning("No workers support batch processing. PubMed processing will be slower.")

        # Start stats updater in a separate thread
        import threading

        # Create a clean progress bar with a proper description
        with tqdm(
            total=self.total_batches,
            desc="Starting...",
            unit="batch",
            # Make progress bar more log-friendly
            disable=False,  # Force enable the progress bar even if stderr is not a TTY
            miniters=1,  # Update when actual progress is made
            file=sys.stdout,  # Write to stdout
            dynamic_ncols=True,  # Adjust progress bar width as terminal changes
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as self.pbar:
            # Start the stats updater thread with the progress bar
            stats_thread = threading.Thread(
                target=self._stats_updater,
                args=(
                    self.articles_processed_val,
                    self.batches_completed_val,
                    self.active_processes_val,
                    self.pbar,
                    running,
                    self.total_batches,
                    self.start_time,
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
            last_update_time = time.time()
            last_summary_time = time.time()
            summary_interval = 60  # Create a progress summary every minute

            while self.remaining_workers > 0:
                try:
                    # Wait for a result with timeout
                    result = result_queue.get(timeout=1.0)
                    last_update_time = time.time()

                    # Process the message using the handler
                    message_handler.handle(result)

                    # Periodically log a concise summary
                    current_time = time.time()
                    if current_time - last_summary_time >= summary_interval:
                        elapsed = current_time - self.start_time
                        completed = self.batches_completed_val.value
                        total_articles = self.articles_processed_val.value

                        if completed > 0:  # Only log if we have some progress
                            progress_pct = (
                                completed / self.total_batches * 100
                                if self.total_batches > 0
                                else 0
                            )
                            speed = total_articles / elapsed if elapsed > 0 else 0
                            eta_seconds = (
                                (self.total_batches - completed) * (elapsed / completed)
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

                            # Log a summary line
                            logger.info(
                                f"PROGRESS SUMMARY: {completed}/{self.total_batches} batches ({progress_pct:.1f}%), "
                                f"{total_articles:,} articles, {speed:.1f} art/s, "
                                f"ETA: {eta}"
                            )

                        last_summary_time = current_time

                except Empty:
                    # Check if workers are still alive but not reporting
                    if time.time() - last_update_time > 30:  # No updates for 30 seconds
                        active_count = sum(1 for p in processes if p.is_alive())
                        if active_count < self.remaining_workers:
                            logger.warning(
                                f"Some workers seem to have died. Only {active_count}/{self.remaining_workers} workers active."
                            )
                        last_update_time = time.time()  # Reset timeout

        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=1.0)

        # Stop the stats updater
        with running.get_lock():
            running.value = 0
        stats_thread.join(timeout=1.0)

        # Record end time
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time

        # Print final summary using the StatsManager
        for worker_id, stats in self.worker_stats.items():
            self.stats_manager.update_stats(
                stats.get("articles", 0),
                stats.get("time", 0),
                stats.get("batches", 0),
            )

        summary = self.stats_manager.format_summary(self.worker_stats)
        logger.info(summary)
        print(summary)

        return (
            self.articles_processed_val.value,
            self.batches_completed_val.value,
        )


def run_splitter(
    splitter_config: dict,
    ignore: bool = False,
    cpu_limit: Union[int, None] = None,
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

    # Validate tokenizer configuration before spawning processes
    tokenizer_name = splitter_config.get("tokenizer", "").lower()
    if tokenizer_name not in ["spacy", "nltk"]:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

    cpu_limit_to_use = min(cpu_limit, cpu_count()) if cpu_limit else cpu_count()
    logger.debug(
        f"CPU_LIMIT: {cpu_limit} (cpu_count: {cpu_count()}), using {cpu_limit_to_use} cores"
    )

    logger.info(f"Starting splitter with {cpu_limit_to_use} CPU cores")

    runner = SplitterRunner(splitter_config, cpu_limit_to_use)
    runner.run()

    logger.info("Finished running splitter script.")
    return {}
