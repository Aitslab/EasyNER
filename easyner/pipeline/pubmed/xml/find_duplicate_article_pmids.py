#!/usr/bin/env python3
"""Uses PubMedXMLParser to find duplicate articles in PubMed XML files.

Multiprocessing to find all PMIDs which are aggregated across multiple files.
Each worker process maintains one parser instance that processes multiple documents.
Memory management avoids system crashes due to OOM conditions.
Cross-system compatibility with resource monitoring and safe shutdown procedures.
"""

import atexit
import csv
import gc
import glob
import logging
import multiprocessing as mp
import os
import queue
import signal
import tempfile
import time
from collections import Counter
from functools import partial
from types import FrameType
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pmid_deduplication")

# Import EasyNER memory utilities for cross-platform memory management
# Import parser after other imports
from pubmed_xml_parser import PubMedXMLParser
from tqdm import tqdm

from easyner.core.memory_utils import (
    DEFAULT_MEMORY_CRITICAL_THRESHOLD,
    DEFAULT_MEMORY_HIGH_THRESHOLD,
    MemoryUsageDetectionError,
    get_memory_info,
    get_memory_status,
    get_memory_usage,
)

# Constants for memory management - these will be adjusted based on system
MEMORY_HIGH_THRESHOLD = DEFAULT_MEMORY_HIGH_THRESHOLD
MEMORY_CRITICAL_THRESHOLD = DEFAULT_MEMORY_CRITICAL_THRESHOLD
MEMORY_CHECK_INTERVAL = 5  # Check memory every 5 seconds
BACKPRESSURE_DELAY = 1.0  # Delay in seconds when memory pressure is high
MAX_FILES_PER_PROCESS = 100  # Maximum files to process before forced GC
DEFAULT_MAX_OPEN_FILES = 1024  # Default max open files if we can't detect

# Global variables for cleanup
_temp_files = []
_temp_dirs = []
_processes = []
_stop_requested = False


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful termination."""

    def handle_terminate(sig: int, frame: FrameType | None) -> None:
        global _stop_requested
        logger.warning(f"Received signal {sig}, initiating graceful shutdown...")
        _stop_requested = True
        # Allow the main process to handle cleanup

    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_terminate)
    signal.signal(signal.SIGINT, handle_terminate)


def cleanup_resources() -> None:
    """Clean up temporary resources on exit."""
    # Clean up temp files
    for temp_file in _temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            logger.error(f"Error removing temp file {temp_file}: {e}")

    # Clean up temp directories
    for temp_dir in _temp_dirs:
        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.error(f"Error removing temp directory {temp_dir}: {e}")

    # Terminate any remaining processes
    for process in _processes:
        if process and process.is_alive():
            try:
                process.terminate()
                process.join(timeout=2)
            except Exception as e:
                logger.error(f"Error terminating process: {e}")


def get_max_open_files() -> int:
    """Get maximum number of open files allowed by the system.

    Returns:
        int: Maximum number of file descriptors allowed

    """
    try:
        import resource

        return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except (AttributeError, ValueError, ImportError):
        # Resource module not available or doesn't support RLIMIT_NOFILE
        try:
            # Try ulimit command on UNIX systems
            import subprocess

            output = subprocess.check_output(["ulimit", "-n"], shell=True, text=True)
            limit = int(output.strip())
            return limit
        except Exception:
            # Return a conservative default
            return DEFAULT_MAX_OPEN_FILES


def get_process_count() -> int:
    """Get optimal process count based on available CPUs and Slurm allocation.

    Returns:
        int: Number of processes to use

    """
    # Check for Slurm environment variables first
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return min(32, int(os.environ["SLURM_CPUS_PER_TASK"]))
    elif "SLURM_JOB_CPUS_PER_NODE" in os.environ:
        # Parse complex Slurm CPU specs if needed
        cpus_str = os.environ["SLURM_JOB_CPUS_PER_NODE"]
        if "(" in cpus_str:
            # Handle complex spec like "16(x2)"
            return min(32, int(cpus_str.split("(")[0]))
        return min(32, int(cpus_str))

    # Check system memory to avoid over-allocation
    try:
        memory_info = get_memory_info()
        if memory_info["total_mb"]:
            mem_gb = memory_info["total_mb"] / 1024
            # Rough heuristic - allocate 1 process per 2GB of RAM, up to CPU count
            # Each parsed XML tree can take 5-10 X memory than the file size
            # -> 1 file potentially ~ 50 MB x 10 = 500 MB
            mem_based_procs = max(1, int(mem_gb / 2))
            return min(mem_based_procs, mp.cpu_count())
    except Exception as e:
        logger.warning(f"Error detecting memory-based process count: {e}")

    # Fallback to a more conservative default
    return min(8, max(1, mp.cpu_count() - 1))


def dump_pmids_to_temp_file(pmid_tuples, temp_dir):
    """Dump PMID tuples to a temporary file to free memory.

    Args:
        pmid_tuples: List of (pmid, filepath) tuples
        temp_dir: Directory to save temporary files

    Returns:
        str: Path to the created temporary file

    """
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="pmids_",
        suffix=".tsv",
        dir=temp_dir,
        delete=False,
    )

    temp_path = temp_file.name
    _temp_files.append(temp_path)  # Track for cleanup

    for pmid, filepath in pmid_tuples:
        temp_file.write(f"{pmid}\t{filepath}\n")

    temp_file.close()
    return temp_path


def process_files_batch(
    files_batch,
    progress_queue=None,
    memory_status=None,
    file_count_limiter=None,
):
    """Process a batch of XML files and extract PMIDs with memory management.

    Args:
        files_batch: List of paths to XML files to process
        progress_queue: Optional multiprocessing.Manager.Queue for progress
        memory_status: Optional shared memory status flag (0=OK, 1=HIGH, 2=CRITICAL)
        file_count_limiter: Optional shared counter to limit total files processed

    Returns:
        List of (pmid, filepath) tuples found across all files in the batch

    """
    # Create a single parser instance for all files in this batch
    parser = PubMedXMLParser()
    pmid_filepath_tuples_in_batch = []

    # Track file processing count for forced GC
    files_processed = 0

    for filepath in files_batch:
        # Check if we need to stop due to signal
        if _stop_requested:
            break

        # Check memory status if provided and apply backpressure
        if memory_status is not None and memory_status.value >= 1:
            delay_time = BACKPRESSURE_DELAY
            # For critical memory usage, wait longer
            if memory_status.value >= 2:
                delay_time = BACKPRESSURE_DELAY * 3

            time.sleep(delay_time)  # Pause to allow memory clearing

        # Force garbage collection periodically
        files_processed += 1
        if files_processed % MAX_FILES_PER_PROCESS == 0:
            gc.collect()

        try:
            # Load document
            root = parser.load_document(filepath)

            # Extract PMIDs using the already loaded document
            pmids = parser.extract_pmids(root)

            # Create tuples and extend the result list
            new_tuples = [(pmid, filepath) for pmid in pmids]
            pmid_filepath_tuples_in_batch.extend(new_tuples)

            # Explicitly clear the root object to free memory
            root = None
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
        finally:
            # Signal progress
            if progress_queue is not None:
                progress_queue.put(1)

            # Update global file counter if provided
            if file_count_limiter is not None:
                with file_count_limiter.get_lock():
                    file_count_limiter.value += 1

    # Final garbage collection before returning
    gc.collect()
    return pmid_filepath_tuples_in_batch


def memory_monitor_process(memory_status, stop_flag) -> None:
    """Dedicated process to monitor memory usage.

    Args:
        memory_status: Shared memory status value (0=OK, 1=HIGH, 2=CRITICAL)
        stop_flag: Event to signal when to stop monitoring

    """
    # Register this process for cleanup
    _processes.append(mp.current_process())

    while not stop_flag.is_set() and not _stop_requested:
        try:
            # Use memory_utils to get memory status code
            status_code = get_memory_status(
                high_threshold=MEMORY_HIGH_THRESHOLD,
                critical_threshold=MEMORY_CRITICAL_THRESHOLD,
            )

            memory_status.value = status_code

            # Log warnings for high or critical memory usage
            if status_code >= 1:
                try:
                    memory_percent = get_memory_usage()
                    level = "HIGH" if status_code == 1 else "CRITICAL"
                    logger.warning(f"{level} memory usage: {memory_percent:.1f}%")
                except MemoryUsageDetectionError as e:
                    level = "HIGH" if status_code == 1 else "CRITICAL"
                    logger.warning(
                        f"{level} memory usage detected, but couldn't get percentage: {e}",
                    )
        except MemoryUsageDetectionError as e:
            # Set to OK (0) if we can't detect memory status - just log the issue
            logger.warning(f"Memory monitoring failed: {e}")
            memory_status.value = 0

        time.sleep(MEMORY_CHECK_INTERVAL)


def find_duplicate_pmids(
    directory_path,
    batch_size: Optional[int] = None,
    limit: Optional[int] = None,
    checkpoint_file=None,
    max_memory_percent: Optional[float] = None,
):
    """Find duplicate PMIDs across multiple XML files with memory management.

    Args:
        directory_path: Path to directory containing XML files
        batch_size: Optional custom batch size (files per worker)
        limit: Optional limit on the number of files to process
        checkpoint_file: Optional path to save/load checkpoint data
        max_memory_percent: Optional memory usage limit (0-100)

    Returns:
        Tuple containing:
        - Counter of PMIDs with their occurrence counts
        - Dictionary of duplicate PMIDs with their counts
        - List of (pmid, filepath) tuples for duplicate PMIDs

    """
    # Setup signal handlers for external interruptions
    setup_signal_handlers()

    # Adjust memory thresholds if specified
    global MEMORY_HIGH_THRESHOLD, MEMORY_CRITICAL_THRESHOLD
    if max_memory_percent is not None:
        # Set high threshold to 80% of max and critical to 90% of max
        MEMORY_HIGH_THRESHOLD = max(30, int(max_memory_percent * 0.8))
        MEMORY_CRITICAL_THRESHOLD = max(50, int(max_memory_percent * 0.9))
        logger.info(
            f"Memory thresholds set to {MEMORY_HIGH_THRESHOLD}% (high) and "
            f"{MEMORY_CRITICAL_THRESHOLD}% (critical)",
        )

    # Get all XML files
    xml_files = glob.glob(os.path.join(directory_path, "*.xml.gz"))
    if limit is not None:
        xml_files = xml_files[:limit]

    if not xml_files:
        logger.warning(f"No XML files found in {directory_path}")
        return Counter(), {}, []

    # Setup function-level cleanup
    global _temp_files, _temp_dirs
    _temp_files = []
    _temp_dirs = []

    # Set up multiprocessing with optimal number of CPUs
    num_processes = get_process_count()
    max_files = get_max_open_files()

    # Limit processes based on max open files
    # (each process may have multiple open files)
    safe_process_limit = max(1, min(num_processes, max_files // 10))
    if safe_process_limit < num_processes:
        logger.warning(
            f"Limiting processes to {safe_process_limit} due to file descriptor limits",
        )
        num_processes = safe_process_limit

    # Calculate batch size if not provided
    total_files = len(xml_files)
    if batch_size is None:
        # Set a reasonable batch size based on file count and process count
        batch_size = max(1, total_files // (num_processes * 4))
        # Ensure we don't have empty batches at the end
        if total_files % batch_size != 0:
            batch_size = max(1, total_files // num_processes)

    batches = [xml_files[i : i + batch_size] for i in range(0, total_files, batch_size)]

    logger.info(f"Using {num_processes} processes to process {total_files} files")
    logger.info(f"Work divided into {len(batches)} batches (batch size: {batch_size})")

    # Create a temporary directory for intermediate results
    temp_dir = tempfile.mkdtemp(prefix="pmids_temp_")
    _temp_dirs.append(temp_dir)

    # Set up multiprocessing resources
    manager = mp.Manager()
    progress_queue = manager.Queue()
    memory_status = mp.Value("i", 0)  # 0=OK, 1=HIGH, 2=CRITICAL
    stop_monitor = mp.Event()  # Event to signal the monitor to stop
    file_counter = mp.Value("i", 0)  # Count files processed for status reporting

    # Start memory monitor in a separate process
    monitor_process = mp.Process(
        target=memory_monitor_process,
        args=(memory_status, stop_monitor),
    )
    monitor_process.daemon = True
    _processes.append(monitor_process)
    monitor_process.start()

    # Use functools.partial to pass the queue and memory status to workers
    worker_func = partial(
        process_files_batch,
        progress_queue=progress_queue,
        memory_status=memory_status,
        file_count_limiter=file_counter,
    )

    # Process batches and incrementally store results
    pmid_counts = Counter()
    all_pmid_tuples = []  # Only store if memory allows

    # Track whether we're storing all PMIDs in memory or on disk
    using_disk_storage = False
    memory_last_checked = time.time()
    temp_files = []

    # Try loading from checkpoint if provided
    pmid_counts_from_checkpoint = Counter()
    processed_files = set()

    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            logger.info(f"Loading checkpoint from {checkpoint_file}")
            with open(checkpoint_file) as f:
                # Format: First line is comma-separated list of processed files
                # Remaining lines are tab-separated pmid/count pairs
                processed_files_line = f.readline().strip()
                processed_files = (
                    set(processed_files_line.split(","))
                    if processed_files_line
                    else set()
                )

                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        pmid, count = parts
                        pmid_counts_from_checkpoint[pmid] = int(count)

            logger.info(
                f"Loaded {len(processed_files)} processed files and "
                f"{len(pmid_counts_from_checkpoint)} PMIDs from checkpoint",
            )

            # Filter batches to only include unprocessed files
            new_batches = []
            for batch in batches:
                new_batch = [f for f in batch if f not in processed_files]
                if new_batch:  # Only keep non-empty batches
                    new_batches.append(new_batch)

            batches = new_batches
            logger.info(f"After filtering: {len(batches)} batches remaining")

            # Update counts with checkpoint data
            pmid_counts.update(pmid_counts_from_checkpoint)

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            # Continue with full processing if checkpoint loading fails

    try:
        with mp.Pool(processes=num_processes) as pool:
            # Submit all batches as asynchronous tasks
            async_results = [
                pool.apply_async(worker_func, (batch,)) for batch in batches
            ]

            with tqdm(
                total=total_files,
                desc="Processing XML files",
                unit="file",
            ) as pbar:
                tasks_to_complete = len(async_results)
                completed_tasks = 0
                pending_futures = list(async_results)
                last_checkpoint_time = time.time()

                # Update progress bar with files already processed
                pbar.update(len(processed_files))

                while completed_tasks < tasks_to_complete and not _stop_requested:
                    # Drain the progress queue
                    try:
                        drained_items = 0
                        while True:
                            try:
                                progress_queue.get_nowait()
                                drained_items += 1
                                if drained_items >= 100:
                                    # Update in chunks to avoid progress bar overhead
                                    pbar.update(drained_items)
                                    drained_items = 0
                            except queue.Empty:
                                if drained_items > 0:
                                    pbar.update(drained_items)
                                break
                    except Exception as e:
                        logger.error(f"Error updating progress: {e}")

                    # Check memory periodically
                    if time.time() - memory_last_checked > MEMORY_CHECK_INTERVAL:
                        try:
                            memory_percent = get_memory_usage()
                        except MemoryUsageDetectionError as e:
                            logger.warning(
                                f"Could not check memory usage: {e}, proceeding cautiously",
                            )
                            # Assume high usage to be cautious
                            memory_percent = MEMORY_HIGH_THRESHOLD
                        memory_last_checked = time.time()

                    # Check for completed tasks and retrieve results
                    next_pending = []
                    for future in pending_futures:
                        if future.ready():
                            try:
                                batch_results = future.get(timeout=30)

                                # Check memory - use memory_utils to determine if high
                                try:
                                    memory_status_code = get_memory_status(
                                        high_threshold=MEMORY_HIGH_THRESHOLD,
                                        critical_threshold=MEMORY_CRITICAL_THRESHOLD,
                                    )
                                except MemoryUsageDetectionError as e:
                                    logger.warning(
                                        f"Could not determine memory status: {e}. Using cautious approach.",
                                    )
                                    memory_status_code = (
                                        1  # Assume HIGH memory if we can't detect
                                    )

                                if memory_status_code >= 1 or using_disk_storage:
                                    # If we haven't switched to disk yet, dump what we have
                                    if not using_disk_storage and all_pmid_tuples:
                                        temp_file = dump_pmids_to_temp_file(
                                            all_pmid_tuples,
                                            temp_dir,
                                        )
                                        temp_files.append(temp_file)
                                        all_pmid_tuples = []  # Clear memory
                                        using_disk_storage = True
                                        # Log memory info when switching to disk storage
                                        current_memory_percent_str = "Unknown"
                                        try:
                                            memory_info = get_memory_info()
                                            current_memory_percent_str = (
                                                f"{memory_info['percent']:.1f}%"
                                            )
                                        except MemoryUsageDetectionError as e:
                                            logger.warning(
                                                f"Could not get memory info when switching to disk: {e}",
                                            )

                                        logger.info(
                                            f"Memory high ({current_memory_percent_str}): Switched to disk storage",
                                        )

                                    # Store this batch to disk
                                    if batch_results:
                                        temp_file = dump_pmids_to_temp_file(
                                            batch_results,
                                            temp_dir,
                                        )
                                        temp_files.append(temp_file)

                                        # Update counters while we have the data
                                        for pmid, _ in batch_results:
                                            pmid_counts[pmid] += 1

                                        # Force garbage collection
                                        batch_results = None
                                        gc.collect()

                                else:
                                    # Memory is fine, keep in memory
                                    all_pmid_tuples.extend(batch_results)

                                    # Update counters
                                    for pmid, _ in batch_results:
                                        pmid_counts[pmid] += 1

                            except Exception as e:
                                logger.error(f"A batch processing task failed: {e}")

                            completed_tasks += 1
                        else:
                            next_pending.append(future)

                    pending_futures = next_pending

                    # Save checkpoint periodically (every 10 minutes)
                    if (
                        checkpoint_file
                        and time.time() - last_checkpoint_time > 600
                        and file_counter.value > 0
                    ):
                        try:
                            save_checkpoint(
                                checkpoint_file,
                                processed_files,
                                pmid_counts,
                            )
                            last_checkpoint_time = time.time()
                        except Exception as e:
                            logger.error(f"Failed to save checkpoint: {e}")

                    # Avoid busy-waiting
                    if pending_futures:
                        time.sleep(0.1)

                # Save final checkpoint if requested and not complete
                if checkpoint_file and _stop_requested and file_counter.value > 0:
                    try:
                        save_checkpoint(checkpoint_file, processed_files, pmid_counts)
                        logger.info("Saved checkpoint before early termination")
                    except Exception as e:
                        logger.error(f"Failed to save final checkpoint: {e}")

        # Find duplicate PMIDs (appear more than once)
        duplicates = {pmid: count for pmid, count in pmid_counts.items() if count > 1}

        # If we used disk storage, return minimal data or read from disk as needed
        if using_disk_storage:
            logger.info(
                f"Used disk storage for intermediate results: {len(temp_files)} files",
            )

            # If all_pmid_tuples is empty and we need to return all data,
            # we need to read from disk - only if caller needs this data
            if not all_pmid_tuples and temp_files:
                duplicate_pmid_set = set(duplicates.keys())

                # Read temp files and filter to only keep duplicates
                for temp_file_path in temp_files:
                    with open(temp_file_path) as f:
                        for line in f:
                            try:
                                pmid, filepath = line.strip().split("\t")
                                if pmid in duplicate_pmid_set:
                                    all_pmid_tuples.append((pmid, filepath))
                            except ValueError:
                                # Skip malformed lines
                                continue

    finally:
        # Stop monitoring and clean up resources
        stop_monitor.set()
        if monitor_process.is_alive():
            monitor_process.join(timeout=2)

        # Clean up temp files - this will be also done by atexit handler
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    _temp_files.remove(temp_file)  # Remove from global list
            except Exception as e:
                logger.error(f"Error removing temp file {temp_file}: {e}")

        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
                _temp_dirs.remove(temp_dir)  # Remove from global list
        except Exception as e:
            logger.error(f"Error removing temp directory {temp_dir}: {e}")

    # Report completion
    if _stop_requested:
        logger.warning("Processing stopped early due to external signal")

    return pmid_counts, duplicates, all_pmid_tuples


def save_checkpoint(checkpoint_file, processed_files, pmid_counts) -> None:
    """Save current progress to a checkpoint file.

    Args:
        checkpoint_file: Path to save checkpoint data
        processed_files: Set of files already processed
        pmid_counts: Counter with current PMID counts

    """
    # Create temp file first to avoid corrupting existing checkpoint
    temp_checkpoint = f"{checkpoint_file}.tmp"

    with open(temp_checkpoint, "w") as f:
        # Write processed files as first line, comma separated
        f.write(",".join(processed_files))
        f.write("\n")

        # Write pmid counts, tab separated
        for pmid, count in pmid_counts.items():
            f.write(f"{pmid}\t{count}\n")

    # Rename temp file to target file (atomic operation)
    os.replace(temp_checkpoint, checkpoint_file)
    logger.info(
        f"Checkpoint saved with {len(processed_files)} files and {len(pmid_counts)} PMIDs",
    )


def write_results(
    pmid_counts,
    duplicates,
    all_pmid_tuples,
    output_file,
    csv_output,
) -> None:
    """Write results to output files.

    Args:
        pmid_counts: Counter with PMID occurrence counts
        duplicates: Dict of duplicate PMIDs with counts
        all_pmid_tuples: List of (pmid, filepath) tuples
        output_file: Path to output summary file
        csv_output: Path to CSV output file for duplicate sources

    """
    # Write duplicates summary to file
    if duplicates:
        with open(output_file, "w") as f:
            for pmid, count in sorted(
                duplicates.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                f.write(f"{pmid}\t{count}\n")
        logger.info(f"Duplicate PMIDs summary written to {output_file}")

    # Write all instances of duplicate PMIDs with their source files to CSV
    if duplicates and csv_output:
        # Get the set of duplicate PMIDs
        duplicate_pmid_set = set(duplicates.keys())

        with open(csv_output, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["pmid", "file"])  # CSV header

            # Write all instances of duplicate PMIDs with their source files
            row_count = 0
            for pmid, filepath in all_pmid_tuples:
                # Only include rows for PMIDs that are duplicates
                if pmid in duplicate_pmid_set:
                    writer.writerow([pmid, filepath])
                    row_count += 1

        logger.info(f"{row_count} rows written to {csv_output}")


if __name__ == "__main__":
    import argparse

    # Register global cleanup handler
    atexit.register(cleanup_resources)

    parser = argparse.ArgumentParser(
        description="Find duplicate PMIDs across PubMed XML files with memory management",
    )
    parser.add_argument("directory", help="Directory containing XML files")
    parser.add_argument(
        "--output",
        help="Output file for duplicate PMIDs",
        default="duplicate_pmids.txt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Custom batch size (files per worker)",
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of files to process (for testing)",
        default=None,
    )
    parser.add_argument(
        "--csv-output",
        help="CSV file to store duplicate PMIDs with their source files",
        default="duplicate_pmids_with_sources.csv",
    )
    parser.add_argument(
        "--memory-high",
        type=int,
        help="Memory percentage threshold for high memory warning",
        default=MEMORY_HIGH_THRESHOLD,
    )
    parser.add_argument(
        "--memory-critical",
        type=int,
        help="Memory percentage threshold for critical memory warning",
        default=MEMORY_CRITICAL_THRESHOLD,
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        help="Maximum memory percent to use (0-100)",
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint file for resumable processing",
        default=None,
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file (default: log to console)",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ),
        )
        logger.addHandler(file_handler)

    # Update memory thresholds if provided
    if args.memory_high is not None:
        MEMORY_HIGH_THRESHOLD = args.memory_high
    if args.memory_critical is not None:
        MEMORY_CRITICAL_THRESHOLD = args.memory_critical

    # Log basic system info
    try:
        memory_info = get_memory_info()
        if memory_info["total_mb"]:
            total_memory_gb = memory_info["total_mb"] / 1024
            logger.info(f"Total system memory: {total_memory_gb:.1f} GB")
    except MemoryUsageDetectionError as e:
        logger.warning(f"Could not get system memory information: {e}")

    max_files = get_max_open_files()
    logger.info(f"Maximum open files: {max_files}")

    # Process files
    start_time = time.time()
    pmid_counts, duplicates, all_pmid_tuples = find_duplicate_pmids(
        args.directory,
        args.batch_size,
        args.limit,
        args.checkpoint,
        args.max_memory,
    )

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(f"Processing completed in {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    logger.info(f"Found {len(pmid_counts)} total unique PMIDs")
    logger.info(f"Found {len(duplicates)} PMIDs that appear in multiple places")

    # Write results to files
    write_results(
        pmid_counts,
        duplicates,
        all_pmid_tuples,
        args.output,
        args.csv_output,
    )
