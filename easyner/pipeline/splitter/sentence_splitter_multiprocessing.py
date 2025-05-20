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
SPACY_N_PROCESSES = 1  # Set to 1 for multiprocessing
SPACY_BATCH_SIZE = 200  # Batch size for spaCy processing not same as worker batch size
# Number of parallel worker processes - adjust based on your machine
NUM_WORKERS = min(16, max(1, mp.cpu_count() - 1))
# Memory threshold in MB - adjust based on your system
MEMORY_HIGH_THRESHOLD = 75  # When to start applying backpressure
MEMORY_CRITICAL_THRESHOLD = 80  # When to temporarily pause processing
COMMIT_EVERY = 50


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
        nlp.pipe(texts, batch_size=SPACY_BATCH_SIZE, n_process=SPACY_N_PROCESSES),
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
            # We need the parser
            disable=[
                "ner",
                "entity_ruler",
                "entity_linker",
                "textcat",
                "textcat_multilabel",
                "attribute_ruler",
                "lemmatizer",
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

    with tqdm(total=total_segments, unit="seg") as pbar:
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
                        "Avg Speed": f"{speed:.1f} seg/s",
                        "Sent": current_sentences,
                        "Active Workers": f"{active_workers}/{NUM_WORKERS}",
                        "Used Memory": memory_display,
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


def setup_database_tables(con: duckdb.DuckDBPyConnection) -> int:
    """Set up the database tables using the provided connection."""
    logger.info("Setting up database tables")

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

    return total_segments


def main() -> None:
    """Coordinate multiprocessing of sentence splitting."""
    try:
        msg = (
            "---------------------------------\n"
            "Starting sentence splitting with multiprocessing. "
            f"Using {NUM_WORKERS} workers."
            "\n---------------------------------"
        )
        logger.info(msg)
        assert DB_PATH is not None, "DB_PATH must be set"

        # --- SINGLE DATABASE CONNECTION FOR ALL THREADS ---
        duckdb_con = duckdb.connect(database=DB_PATH, read_only=False)
        duckdb_con.execute("PRAGMA memory_limit='4GB'")

        reader_con = duckdb_con.cursor()

        # Database setup using the main connection
        total_segments = setup_database_tables(
            reader_con,
        )  # Since reader needs the temp table

        # Create shared counters and events
        processed_counter = Value(c_int, 0)
        sentences_counter = Value(c_int, 0)
        active_workers_counter = Value(c_int, 0)
        stop_event = Event()
        pause_event = Event()

        # Create queues for worker processes
        task_queue = JoinableQueue(maxsize=NUM_WORKERS * 3)
        result_queue = Queue(maxsize=NUM_WORKERS * 5)

        # Start worker processes for NLP processing only
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
            )
            p.daemon = True
            p.start()
            workers.append(p)

        # Start THREAD for database writing - NOT a process
        writer_thread = threading.Thread(
            target=result_writer_thread,
            args=(
                duckdb_con,
                result_queue,
                stop_event,
                processed_counter,
                sentences_counter,
            ),
            name="writer-thread",
        )
        writer_thread.daemon = True
        writer_thread.start()

        # Start THREAD for memory monitoring
        memory_thread = threading.Thread(
            target=memory_monitor_thread,
            args=(pause_event,),
            name="memory-monitor",
        )
        memory_thread.daemon = True
        memory_thread.start()

        # Start THREAD for progress reporting
        progress_thread = threading.Thread(
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
        progress_thread.daemon = True
        progress_thread.start()

        # === MAIN THREAD READS FROM DATABASE AND FEEDS WORKERS ===
        try:
            # Create a thread-local cursor from the main connection

            offset = 0
            while offset < total_segments:
                # Check memory pressure
                if pause_event.is_set():
                    logger.info("Main thread paused due to memory pressure")
                    time.sleep(2)  # Wait before checking again
                    continue

                # Get a batch of segment IDs - using thread-local cursor
                batch_ids = reader_con.execute(
                    f"""--sql
                    SELECT pmid, segment_number
                    FROM {TEMP_TABLE}
                    LIMIT {BATCH_SIZE} OFFSET {offset}
                    """,
                ).fetchall()

                if not batch_ids:
                    break

                # Convert to native Python lists to avoid serialization issues
                pmids = [int(id[0]) for id in batch_ids]
                seg_nums = [int(id[1]) for id in batch_ids]

                # Get segments with thread-local cursor
                segments_data = reader_con.execute(
                    f"""--sql
                    SELECT s.pmid, s.segment_number, s.segment
                    FROM {TEXT_SEGMENTS_TABLE} s
                    WHERE (pmid, segment_number) IN (
                        SELECT UNNEST(?), UNNEST(?)
                    )
                    """,
                    [pmids, seg_nums],
                ).fetchall()

                # Add to worker queue in smaller batches
                for i in range(0, len(segments_data), WORKER_BATCH_SIZE):
                    worker_batch = segments_data[i : i + WORKER_BATCH_SIZE]
                    task_queue.put(worker_batch)

                offset += len(batch_ids)

                # Memory management
                del batch_ids, segments_data, pmids, seg_nums
                gc.collect()

            # Signal that no more tasks will be added
            logger.info("All segments queued for processing")

        except Exception as e:
            logger.error(f"Error in main thread: {str(e)}")

        # Wait for all workers to finish
        for _ in range(NUM_WORKERS):
            task_queue.put(None)  # Sentinel to stop workers

        for w in workers:
            w.join()

        # Signal other threads to stop
        stop_event.set()

        # Wait for threads to finish
        writer_thread.join()
        progress_thread.join()

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
    finally:
        # Clean up
        logger.info("Main process completed")


def result_writer_thread(
    duckdb_con: duckdb.DuckDBPyConnection,
    result_queue: Queue,
    stop_event: Event,
    processed_counter: Value,
    sentences_counter: Value,
) -> None:
    """Thread that handles writing results to the database using thread-local cursor."""
    logger.info("Result writer thread starting")

    try:
        # Create thread-local cursor from shared connection
        writer_con = duckdb_con.cursor()
        logger.info("Writer thread connected to database")

        append_count = 0
        total_sentences_since_commit = 0

        while not stop_event.is_set() or not result_queue.empty():
            try:
                result = result_queue.get(timeout=1)

                # Handle error messages
                if (
                    isinstance(result, tuple)
                    and len(result) == 3
                    and result[0] == "ERROR"
                ):
                    error_msg, worker_id = result[1], result[2]
                    logger.error(f"Error from worker {worker_id}: {error_msg}")
                    continue

                # Process valid results
                if isinstance(result, tuple) and len(result) == 2:
                    count, sentences_df = result

                    # Update processed counter
                    with processed_counter.get_lock():
                        processed_counter.value += count

                    # Add sentences to database
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

                            # Update sentence counter
                            with sentences_counter.get_lock():
                                sentences_counter.value += total_sentences_since_commit

                            # Reset counters
                            append_count = 0
                            total_sentences_since_commit = 0

                # Clean up
                del result
                gc.collect()

            except Empty:
                # Commit any pending changes while idle
                if append_count > 0:
                    writer_con.commit()
                    with sentences_counter.get_lock():
                        sentences_counter.value += total_sentences_since_commit
                    append_count = 0
                    total_sentences_since_commit = 0
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Writer thread error: {str(e)}")
                time.sleep(1)

    except Exception as e:
        logger.error(f"Writer thread failed: {str(e)}")
    finally:
        # Final commit if needed
        try:
            if append_count > 0:
                logger.info(
                    f"Final commit of {append_count} pending appends ({total_sentences_since_commit} sentences)",
                )
                writer_con.commit()
                with sentences_counter.get_lock():
                    sentences_counter.value += total_sentences_since_commit
        except Exception as e:
            logger.error(f"Error during final commit: {str(e)}")

        logger.info("Writer thread exiting")


if __name__ == "__main__":
    # Set up signal handler for graceful termination
    def signal_handler(sig, frame) -> None:
        logger.warning(f"Received signal {sig}, initiating shutdown")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main()
