# ruff: noqa : E501, D100
import gc
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
from ctypes import c_bool, c_int
from multiprocessing import Event, JoinableQueue, Process, Queue, Value
from queue import Empty

import duckdb
import pandas as pd
import psutil
import spacy
from dotenv import load_dotenv
from spacy.language import Language
from spacy.tokens import Doc, Span
from tqdm import tqdm

# Import memory utilities
from easyner.core import memory_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv()
DB_PATH = os.getenv("DB_PATH")
if DB_PATH is None or DB_PATH.strip() == "":
    # Provide a default path for testing if not set
    DB_PATH = "test_database.duckdb"
    logger.warning(f"DB_PATH environment variable is not set. Using default: {DB_PATH}")
else:
    logger.info(f"Using database path: {DB_PATH}")

logger.info(
    f"---Memory usage at start: {memory_utils.get_memory_usage()}%---Total memory: {memory_utils.get_total_memory_bytes() / (1024 * 1024)}MB",
)


def get_memory_details() -> dict:
    """Get detailed memory information."""
    import psutil

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    return {
        "physical_mb": vm.total / (1024 * 1024),
        "available_mb": vm.available / (1024 * 1024),
        "used_mb": vm.used / (1024 * 1024),
        "percent_used": vm.percent,
        "swap_total_mb": swap.total / (1024 * 1024),
        "swap_used_mb": swap.used / (1024 * 1024),
        "swap_percent": swap.percent,
    }


# Use it in your startup log
mem_details = get_memory_details()
logger.info(
    f"Memory details: Physical RAM: {mem_details['physical_mb']:.0f}MB, "
    f"Used: {mem_details['used_mb']:.0f}MB ({mem_details['percent_used']:.1f}%), "
    f"Swap: {mem_details['swap_total_mb']:.0f}MB (used: {mem_details['swap_percent']:.1f}%)",
)

TEXT_SEGMENTS_TABLE = "abstract_segments"
TEMP_TABLE = "segments_to_process"
SENTENCES_TABLE = "sentences"
BATCH_SIZE = 10000  # Reduced batch size for better memory management
WORKER_BATCH_SIZE = 500  # Small batches for workers to process
SPACY_MODEL = "en_core_web_sm"
# Number of parallel worker processes - adjust based on your machine
NUM_WORKERS = min(16, max(1, mp.cpu_count() - 1))
# Memory threshold in MB - adjust based on your system
MEMORY_HIGH_THRESHOLD = 75  # When to start applying backpressure
MEMORY_CRITICAL_THRESHOLD = 80  # When to temporarily pause processing


def monitor_memory() -> dict:
    """Monitor memory usage and provide usage statistics.

    Returns:
        dict: Memory usage information including percentage and MB values

    """
    try:
        return memory_utils.get_memory_info()
    except Exception as e:
        logger.warning(f"Failed to get memory info from memory_utils: {e}")
        # Fallback to psutil directly
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        return {"percent": 0, "used_mb": mem_mb, "total_mb": 0, "available_mb": 0}


def check_memory_pressure() -> int:
    """Check current memory pressure level.

    Returns:
        int: 0=OK, 1=HIGH, 2=CRITICAL

    """
    try:
        return memory_utils.get_memory_status(
            high_threshold=MEMORY_HIGH_THRESHOLD,
            critical_threshold=MEMORY_CRITICAL_THRESHOLD,
        )
    except Exception as e:
        logger.warning(f"Failed to get memory status: {e}")
        # Fallback to direct process memory if memory_utils fails
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        if mem_mb > 25000:  # 25GB
            return 2  # CRITICAL
        elif mem_mb > 20000:  # 20GB
            return 1  # HIGH
        return 0  # OK


def process_batch(nlp: Language, batch: list[tuple]) -> pd.DataFrame:
    """Process a text segment batch using spaCy and preserves sentence order within segments.

    Args:
        nlp: The loaded spaCy NLP object.
        batch: A list of tuples, where each tuple is (pmid, segment_number, segment_text).

    Returns:
        A Pandas DataFrame with columns: pmid, segment_number, sentence_in_segment_order, sentence.

    """
    # Prepare texts and corresponding metadata for spaCy pipe
    texts = [item[2] for item in batch]
    metadata = [(item[0], item[1]) for item in batch]  # (pmid, segment_number)

    sentences_data = []

    # Process texts through spaCy pipeline - don't use n_process here since we're already
    # in a worker process
    for doc, (pmid, segment_number) in zip(
        nlp.pipe(texts, batch_size=WORKER_BATCH_SIZE),
        metadata,
    ):
        sentence_in_segment_order = 1
        for sent in doc.sents:
            sentences_data.append(
                {
                    "pmid": pmid,
                    "segment_number": segment_number,
                    "sentence_in_segment_order": sentence_in_segment_order,
                    "sentence": sent.text,
                    "start_char": sent.start_char,
                    "end_char": sent.end_char,
                },
            )
            sentence_in_segment_order += 1
        # Explicitly clear the doc to free up memory
        del doc

    return pd.DataFrame(sentences_data)


def worker_process(
    worker_id: int,
    task_queue: JoinableQueue,
    result_queue: Queue,
    stop_event: Event,
    pause_event: Event,
    active_workers_counter: Value,
) -> None:
    """Worker process function that handles text processing with spaCy.

    Args:
        worker_id: The ID of this worker process
        task_queue: Queue containing batches of text segments to process
        result_queue: Queue to put processed sentence dataframes
        stop_event: Event to signal worker to stop
        pause_event: Event to signal worker to pause processing
        active_workers_counter: Shared counter for active workers

    """
    logger.info(f"Worker {worker_id} starting...")

    try:
        # Each worker loads its own spaCy model with specific components disabled
        logger.info(f"Worker {worker_id} loading spaCy model: {SPACY_MODEL}...")

        # Load the model with tagger to avoid lemmatizer warnings
        # Explicitly disable components we don't need for sentence segmentation
        nlp = spacy.load(
            SPACY_MODEL,
            # Disable components we don't need for sentence splitting
            # Keep the tagger since lemmatizer needs it
            disable=[
                "ner",
                "entity_ruler",
                "entity_linker",
                "textcat",
                "textcat_multilabel",
            ],
        )

        # If we want to fully disable the lemmatizer to avoid warnings:
        if "lemmatizer" in nlp.pipe_names:
            nlp.remove_pipe("lemmatizer")

        logger.info(f"Worker {worker_id} loaded spaCy model successfully")

        while not stop_event.is_set():
            # Check if processing should be paused due to memory pressure
            if pause_event.is_set():
                logger.info(f"Worker {worker_id} paused due to memory pressure")
                time.sleep(1)  # Wait before checking again
                continue

            try:
                # Get pre-fetched segments data from the queue with a timeout
                segments_batch = task_queue.get(timeout=1)

                # Increment active workers counter
                with active_workers_counter.get_lock():
                    active_workers_counter.value += 1

                # Process this batch with spaCy
                sentences_df = process_batch(nlp, segments_batch)

                # Send result to the writer process along with the count
                result_queue.put((len(segments_batch), sentences_df))

                # Mark the task as done
                task_queue.task_done()

                # Decrement active workers counter
                with active_workers_counter.get_lock():
                    active_workers_counter.value = max(
                        0,
                        active_workers_counter.value - 1,
                    )

                # Clean up to help with memory
                del segments_batch
                del sentences_df
                gc.collect()

            except Empty:
                # No tasks available - just continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                # Put an error message in the result queue
                result_queue.put(("ERROR", str(e), worker_id))
                task_queue.task_done()  # Still mark as done even if it failed

    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {str(e)}")
    finally:
        logger.info(f"Worker {worker_id} exiting")


def result_writer_process(  # noqa: C901
    result_queue: Queue,
    stop_event: Event,
    processed_counter: Value,
    sentences_counter: Value,
) -> None:
    """Process that handles writing results back to the database.

    This runs as a separate process to ensure database writes don't block processing.

    Args:
        result_queue: Queue with processed sentence dataframes
        stop_event: Event to signal process to stop
        processed_counter: Shared counter for processed segments
        sentences_counter: Shared counter for sentences found

    """
    logger.info("Result writer starting")

    try:
        # Create a single database connection with read_only=False for the writer
        logger.info(f"Writer connecting to database: {DB_PATH}")
        # Assert DB_PATH not None for type checking
        assert DB_PATH is not None, "DB_PATH should not be None"
        writer_con = main_con.cursor()  # Thread local connection
        logger.info("Writer connected to database")

        COMMIT_EVERY = 10
        append_count = 0
        total_sentences_since_commit = 0

        while not stop_event.is_set() or not result_queue.empty():
            try:
                # Get a processed batch from the queue with a timeout
                result = result_queue.get(timeout=1)

                # Check if this is an error message
                if (
                    isinstance(result, tuple)
                    and len(result) == 3
                    and result[0] == "ERROR"
                ):
                    error_msg, worker_id = result[1], result[2]
                    logger.error(f"Error from worker {worker_id}: {error_msg}")
                    continue

                # Result should be a tuple of (count, DataFrame)
                if isinstance(result, tuple) and len(result) == 2:
                    count, sentences_df = result

                    # Update processed counter
                    with processed_counter.get_lock():
                        processed_counter.value += count

                    # Add sentences to database without immediate commit
                    if (
                        isinstance(sentences_df, pd.DataFrame)
                        and not sentences_df.empty
                    ):
                        writer_con.append(SENTENCES_TABLE, sentences_df)
                        append_count += 1
                        total_sentences_since_commit += len(sentences_df)

                        # Only commit periodically
                        if append_count >= COMMIT_EVERY:
                            logger.debug(
                                f"Committing after {append_count} appends ({total_sentences_since_commit} sentences)",
                            )
                            writer_con.commit()

                            # Update sentence counter after commit
                            with sentences_counter.get_lock():
                                sentences_counter.value += total_sentences_since_commit

                            # Reset counters
                            append_count = 0
                            total_sentences_since_commit = 0

                    # Log progress occasionally
                    with processed_counter.get_lock(), sentences_counter.get_lock():
                        current_processed = processed_counter.value
                        current_sentences = sentences_counter.value

                    if current_processed % 1000 == 0:
                        logger.info(
                            f"Progress: {current_processed} segments processed, {current_sentences} sentences found",
                        )

                # Clean up
                del result
                gc.collect()

            except Empty:
                # Queue empty - wait a bit
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"Result writer error: {str(e)}")
                time.sleep(1)  # Wait before trying again

    except Exception as e:
        logger.error(f"Result writer failed: {str(e)}")
    finally:
        try:
            # Add the final commit code here
            if append_count > 0:
                logger.info(
                    f"Final commit of {append_count} pending appends ({total_sentences_since_commit} sentences)",
                )
                writer_con.commit()

                # Update sentence counter
                with sentences_counter.get_lock():
                    sentences_counter.value += total_sentences_since_commit

            if "writer_con" in locals():
                writer_con.close()
                logger.info("Writer closed database connection")
        except Exception as e:
            logger.error(f"Error closing writer database connection: {e}")
        logger.info("Result writer exiting")


def memory_monitor_thread(pause_event: Event) -> None:
    """Thread that monitors memory usage and sets pause event when memory is high.

    Args:
        pause_event: Event to set when memory pressure is high

    """
    logger.info("Memory monitor starting")

    while True:
        try:
            # Get current memory usage as percentage (0-100)
            memory_percent = memory_utils.get_memory_usage()

            # Handle different memory pressure levels based on percentage
            if memory_percent >= MEMORY_CRITICAL_THRESHOLD:  # CRITICAL
                if not pause_event.is_set():
                    logger.warning(
                        f"Memory CRITICAL: {memory_percent:.1f}%. Pausing processing.",
                    )
                    pause_event.set()
                    gc.collect()  # Force garbage collection
                time.sleep(5)  # Check less frequently when critical

            elif memory_percent >= MEMORY_HIGH_THRESHOLD:  # HIGH
                # Just log warning but don't pause yet
                logger.info(f"Memory HIGH: {memory_percent:.1f}%. Monitoring.")
                time.sleep(2)  # Check more frequently when memory is high

            else:  # OK
                # Clear pause event if set
                if pause_event.is_set():
                    logger.info(
                        f"Memory recovered: {memory_percent:.1f}%. Resuming processing.",
                    )
                    pause_event.clear()
                time.sleep(5)  # Normal check interval

        except Exception as e:
            logger.error(f"Memory monitor error: {str(e)}")
            time.sleep(5)  # Wait before trying again  # Longer sleep on error


def progress_reporter_thread(
    total_segments: int,
    processed_counter: Value,
    sentences_counter: Value,
    stop_event: Event,
    active_workers_counter: Value,
) -> None:
    """Thread that reports progress.

    Args:
        total_segments: Total number of segments to process
        processed_counter: Shared counter for processed segments
        sentences_counter: Shared counter for sentences found
        stop_event: Event to check if processing should stop
        active_workers_counter: Shared counter for active workers

    """
    logger.info("Progress reporter starting")

    start_time = time.time()

    with tqdm(total=total_segments, unit="segment") as pbar:
        last_processed = 0

        while not stop_event.is_set() or last_processed < total_segments:
            try:
                with processed_counter.get_lock():
                    current_processed = processed_counter.value

                with sentences_counter.get_lock():
                    current_sentences = sentences_counter.value

                # Update progress bar
                if current_processed > last_processed:
                    increment = current_processed - last_processed
                    pbar.update(increment)
                    last_processed = current_processed

                # Update progress display
                elapsed = time.time() - start_time
                speed = current_processed / elapsed if elapsed > 0 else 0

                # Get active workers count
                with active_workers_counter.get_lock():
                    active_workers = active_workers_counter.value

                # Get memory percentage directly
                try:
                    memory_percent = memory_utils.get_memory_usage()
                    memory_display = f"{memory_percent:.1f}%"
                except Exception:
                    # Fallback to basic process memory if get_memory_usage fails
                    process = psutil.Process(os.getpid())
                    mem_mb = process.memory_info().rss / (1024 * 1024)
                    memory_display = f"{mem_mb:.0f}MB"

                pbar.set_postfix(
                    {
                        "Processed": f"{current_processed}/{total_segments}",
                        "Sentences": current_sentences,
                        "Speed": f"{speed:.1f} seg/s",
                        "Workers": f"{active_workers}/{NUM_WORKERS}",
                        "Memory": memory_display,
                    },
                )

                if current_processed >= total_segments:
                    break

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Progress reporter error: {str(e)}")
                time.sleep(1)  # Longer sleep on error

    # Final report
    with processed_counter.get_lock(), sentences_counter.get_lock():
        total_processed = processed_counter.value
        total_sentences = sentences_counter.value

    time_taken = time.time() - start_time
    logger.info(f"Finished processing {total_processed} segments")
    logger.info(f"Total sentences inserted: {total_sentences}")
    logger.info(f"Total time: {time_taken:.2f} seconds")
    if total_processed > 0 and time_taken > 0:
        logger.info(f"Speed: {total_processed / time_taken:.2f} segments/second")


def setup_database():
    """Set up the database connection and tables."""
    logger.info(f"Connecting to database: {DB_PATH}")
    con = duckdb.connect(database=DB_PATH, read_only=False)

    # Ensure sentences table exists with proper schema
    logger.info(f"Ensuring {SENTENCES_TABLE} table exists")
    con.execute(
        f"""--sql
        CREATE TABLE IF NOT EXISTS {SENTENCES_TABLE} (
            pmid INTEGER,
            segment_number INTEGER,
            sentence_in_segment_order INTEGER,
            sentence VARCHAR,
            start_char INTEGER,
            end_char INTEGER
        );
    """,
    )

    # Create temporary table to track segments needing processing
    logger.info("Creating temporary table for unprocessed segments")
    con.execute(
        f"""--sql
        CREATE TEMPORARY TABLE {TEMP_TABLE} AS
        SELECT s.pmid, s.segment_number
        FROM {TEXT_SEGMENTS_TABLE} s
        WHERE NOT EXISTS (
            SELECT 1 FROM {SENTENCES_TABLE} t
            WHERE t.pmid = s.pmid AND t.segment_number = s.segment_number
        )
        AND s.is_header = FALSE
    """,
    )

    # Get count of segments to process
    count_result = con.execute(f"SELECT COUNT(*) FROM {TEMP_TABLE}").fetchone()
    total_segments = count_result[0] if count_result else 0
    logger.info(f"Total segments to process: {total_segments}")

    return con, total_segments


def main() -> None:
    """Coordinate multiprocessing of sentence splitting.

    This function manages the overall workflow of fetching text segments from
    the database and distributing them to worker processes for NLP processing.
    """
    try:
        msg = (
            "---------------------------------\n"
            "Starting sentence splitting with multiprocessing. "
            f"Using {NUM_WORKERS} workers."
            "\n---------------------------------"
        )
        logger.info(msg)

        # Initial database setup using a context manager to ensure proper cleanup
        assert DB_PATH is not None, "DB_PATH must be set"

        logger.info(f"Creating global DuckDB connection: {DB_PATH}")
        main_con = duckdb.connect(database=DB_PATH, read_only=False)
        # Configure DuckDB with memory settings
        main_con.execute("PRAGMA memory_limit='4GB'")

        # Ensure sentences table exists with proper schema
        logger.info(f"Ensuring {SENTENCES_TABLE} table exists")
        main_con.execute(
            f"""--sql
            CREATE TABLE IF NOT EXISTS {SENTENCES_TABLE} (
                pmid INTEGER,
                segment_number INTEGER,
                sentence_in_segment_order INTEGER,
                sentence VARCHAR,
                start_char INTEGER,
                end_char INTEGER
            );
            """,
        )

        # Create temporary table to track unprocessed segments
        logger.info("Creating temporary table for unprocessed segments")
        main_con.execute(
            f"""--sql
            CREATE TEMPORARY TABLE {TEMP_TABLE} AS
            SELECT s.pmid, s.segment_number
            FROM {TEXT_SEGMENTS_TABLE} s
            WHERE NOT EXISTS (
                SELECT 1 FROM {SENTENCES_TABLE} t
                WHERE t.pmid = s.pmid AND t.segment_number = s.segment_number
            )
            AND s.is_header = FALSE
            """,
        )

        # Count total segments to process
        count_result = main_con.execute(
            f"SELECT COUNT(*) FROM {TEMP_TABLE}",
        ).fetchone()
        total_segments = count_result[0] if count_result else 0
        logger.info(f"Total segments to process: {total_segments}")

        if total_segments == 0:
            logger.info("No segments to process. Exiting.")
            return

        # Set up shared queues and events for worker coordination
        task_queue = JoinableQueue(
            maxsize=NUM_WORKERS * 3,
        )  # Limited queue size for backpressure
        result_queue = Queue(maxsize=NUM_WORKERS * 5)  # Larger result queue

        # Events for signaling
        stop_event = Event()
        pause_event = Event()

        # Shared counters for progress tracking
        processed_counter = Value(c_int, 0)
        sentences_counter = Value(c_int, 0)
        active_workers_counter = Value(c_int, 0)  # New counter for active workers

        # Start worker processes - these will only do NLP processing
        workers = []
        for i in range(NUM_WORKERS):
            p = Process(
                target=worker_process,
                args=(
                    i,
                    task_queue,
                    result_queue,
                    stop_event,
                    pause_event,
                    active_workers_counter,
                ),
                name=f"worker-{i}",
            )
            p.daemon = True
            p.start()
            workers.append(p)
            logger.info(f"Started worker process {i}")

        # Start result writer process - this handles all database writes
        writer = Process(
            target=result_writer_process,
            args=(
                result_queue,
                stop_event,
                processed_counter,
                sentences_counter,
            ),
            name="writer",
        )
        writer.daemon = True
        writer.start()
        logger.info("Started result writer process")

        # Start memory monitor thread
        memory_monitor = threading.Thread(
            target=memory_monitor_thread,
            args=(pause_event,),
            name="memory-monitor",
        )
        memory_monitor.daemon = True
        memory_monitor.start()
        logger.info("Started memory monitor thread")

        # Start progress reporter thread
        progress_reporter = threading.Thread(
            target=progress_reporter_thread,
            args=(
                total_segments,
                processed_counter,
                sentences_counter,
                stop_event,
                active_workers_counter,
            ),
            name="progress-reporter",
        )
        progress_reporter.daemon = True
        progress_reporter.start()
        logger.info("Started progress reporter thread")

        # Main loop to feed tasks to workers
        try:
            offset = 0
            while offset < total_segments and not stop_event.is_set():
                # Check if processing is paused due to memory pressure
                if pause_event.is_set():
                    logger.info("Processing paused due to memory pressure")
                    time.sleep(2)  # Wait before trying again
                    continue

                # Get a batch of segment IDs from the temp table
                batch_ids = main_con.execute(
                    f"""--sql
                    SELECT pmid, segment_number
                    FROM {TEMP_TABLE}
                    LIMIT {BATCH_SIZE} OFFSET {offset}
                    """,
                ).fetchall()

                if not batch_ids:
                    break

                # Fetch all text segments for this batch at once
                pmids = [id[0] for id in batch_ids]
                seg_nums = [id[1] for id in batch_ids]

                # Get the actual text segments
                segments_data = main_con.execute(
                    f"""--sql
                    SELECT s.pmid, s.segment_number, s.segment
                    FROM {TEXT_SEGMENTS_TABLE} s
                    WHERE (pmid, segment_number) IN (
                        SELECT UNNEST(?), UNNEST(?)
                    )
                    """,
                    [pmids, seg_nums],
                ).fetchall()

                # Split into smaller batches for workers
                for i in range(0, len(segments_data), WORKER_BATCH_SIZE):
                    worker_batch = segments_data[i : i + WORKER_BATCH_SIZE]

                    # Send pre-fetched data directly to workers
                    task_queue.put(worker_batch)

                # Update offset for next batch
                offset += len(batch_ids)

                # Help with memory pressure
                del batch_ids
                del segments_data
                del pmids
                del seg_nums
                gc.collect()

            # Wait for all tasks to be processed
            logger.info("Waiting for task queue to empty")
            task_queue.join()

            # Wait for result queue to empty
            while not result_queue.empty():
                logger.info(
                    f"Waiting for result queue to empty ({result_queue.qsize()} items remaining)",
                )
                time.sleep(1)

            # Signal processes to stop
            logger.info("Setting stop event")
            stop_event.set()

            # Wait for worker processes to finish
            for i, p in enumerate(workers):
                p.join(timeout=5)
                logger.info(
                    f"Worker {i} {'finished' if not p.is_alive() else 'still running'}",
                )

            # Wait for writer process to finish
            writer.join(timeout=10)
            logger.info(
                f"Writer {'finished' if not writer.is_alive() else 'still running'}",
            )

            # Wait for progress reporter to finish
            progress_reporter.join(timeout=5)

            # Final cleanup
            main_con.execute(f"DROP TABLE IF EXISTS {TEMP_TABLE}")
            logger.info("Dropped temporary table")

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt: Setting stop event")
            stop_event.set()

            # Give processes a chance to finish gracefully
            time.sleep(2)

        logger.info("Sentence splitting with multiprocessing completed")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
    finally:
        # Connection is auto-closed by context manager
        logger.info("Main process completed")


if __name__ == "__main__":
    # Set up signal handler for graceful termination
    def signal_handler(sig, frame) -> None:
        logger.warning(f"Received signal {sig}, initiating shutdown")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main()
