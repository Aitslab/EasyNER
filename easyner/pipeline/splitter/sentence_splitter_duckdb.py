# ruff: noqa : E501, D100
import gc
import os
import sys  # Added for sys.exit
import time

import duckdb
import pandas as pd
import psutil
import spacy
import spacy.tokens
from dotenv import load_dotenv
from spacy.language import Language
from spacy.tokens import Doc, Span
from tqdm import tqdm  # Added tqdm

# --- Configuration ---
load_dotenv()
DB_PATH = os.getenv("DB_PATH")
if DB_PATH is None or DB_PATH.strip() == "":
    msg = "DB_PATH environment variable is not set."
    raise ValueError(msg)
else:
    print(f"Using database path: {DB_PATH}")
TEXT_SEGMENTS_TABLE = "abstract_segments"  # Use the view you created
TEMP_TABLE = "segments_to_process"
SENTENCES_TABLE = "sentences"  # New table name to reflect order
BATCH_SIZE = 20000  # Batch size for reading from DuckDB
SPACY_MODEL = "en_core_web_sm"  # Same sentence performance as en_core_web_md
N_PROCESS = 2  # Number of parallel spaCy processes
SPACY_BATCH_SIZE = 50
MEM_THRESHOLD_MB = 25000  # 25GB threshold


def monitor_memory() -> float:
    """Check memory usage and perform garbage collection if above threshold."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)

    if mem_mb > MEM_THRESHOLD_MB:
        print(f"Memory usage high ({mem_mb:.1f} MB). Forcing garbage collection...")
        gc.collect()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Memory after collection: {mem_mb:.1f} MB")

    return mem_mb


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

    # Process texts in parallel using nlp.pipe
    # We iterate through the docs and their original metadata simultaneously
    doc: Doc
    for doc, (pmid, segment_number) in zip(
        nlp.pipe(
            texts,
            batch_size=SPACY_BATCH_SIZE,
            n_process=N_PROCESS,
            disable=["tagger", "ner"],
        ),
        metadata,
    ):
        sentence_in_segment_order = 1  # Index same as segment_number starts from 1
        sent: Span
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


def main() -> None:
    """Process text segments from a DuckDB database.

    Split them into sentences and store the sentences back into the database, with progress reporting.
    """
    con = None
    try:
        # This assertion confirms DB_PATH is not None. It serves two main purposes:
        # 1. Runtime check: Ensures DB_PATH is valid before use, though an earlier
        #    module-level check should already guarantee this.
        # 2. Static analysis hint: Informs type checkers (e.g., Mypy) that DB_PATH
        #    can be treated as `str` (not `Optional[str]`) beyond this point,
        #    preventing potential false positive type errors.
        assert (
            DB_PATH is not None
        ), "DB_PATH cannot be None at this point due to module-level check."
        con = duckdb.connect(database=DB_PATH, read_only=False)

        # Configure DuckDB with modest memory settings
        con.execute("PRAGMA memory_limit='8GB'")

        # --- Ensure sentences table exists with new schema ---
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

        # --- Load spaCy model once ---
        print(f"Loading spaCy model: {SPACY_MODEL}...")
        # Load only components needed for sentence segmentation
        nlp = spacy.load(
            SPACY_MODEL,
            exclude=["ner", "attribute_ruler", "lemmatizer", "tagger"],
        )
        print("SpaCy model loaded.")
        monitor_memory()  # Check memory after loading model

        print("Creating temporary table for processing...")
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

        # Calculate total number of segments to process for tqdm
        count_result = con.execute(f"SELECT COUNT(*) FROM {TEMP_TABLE}").fetchone()
        total_segments = count_result[0] if count_result else 0
        print(f"Total segments to process: {total_segments}")

        if total_segments == 0:
            print("No segments to process. Exiting.")
            return

        total_processed = 0
        total_sentences = 0
        start_time = time.time()

        with tqdm(total=total_segments, unit="segment") as pbar:
            offset = 0
            while offset < total_segments:
                # Explicitly force garbage collection
                gc.collect()

                # Get a batch of segment IDs from our temp table
                # This avoids the expensive NOT EXISTS query
                batch_ids = con.execute(
                    f"""--sql
                    SELECT pmid, segment_number
                    FROM {TEMP_TABLE}
                    LIMIT {BATCH_SIZE} OFFSET {offset}
                """,
                ).fetchall()

                if not batch_ids:
                    break

                # For each segment ID, fetch the actual text content
                # Note: We use a separate query to fetch text only for segments we'll process
                # This uses much less memory than fetching all columns at once
                pmids = [id[0] for id in batch_ids]
                seg_nums = [id[1] for id in batch_ids]

                # Use parameters to avoid SQL injection with UNNEST
                segments_data = con.execute(
                    f"""--sql
                    SELECT s.pmid, s.segment_number, s.segment
                    FROM {TEXT_SEGMENTS_TABLE} s
                    WHERE (pmid, segment_number) IN (
                        SELECT UNNEST(?), UNNEST(?)
                    )
                """,
                    [pmids, seg_nums],
                ).fetchall()

                # Process this batch
                sentences_df = process_batch(nlp, segments_data)

                con.append(SENTENCES_TABLE, sentences_df)
                con.commit()
                # # Insert in smaller chunks to reduce memory pressure
                # if not sentences_df.empty:
                #     for i in range(0, len(sentences_df), 500):
                #         chunk = sentences_df.iloc[i : i + 500]
                #         con.append(SENTENCES_TABLE, chunk)
                #         con.commit()

                #     total_sentences += len(sentences_df)

                # Update counters
                batch_size = len(segments_data)
                total_processed += batch_size
                offset += batch_size
                pbar.update(batch_size)

                # Update progress display
                pbar.set_postfix(
                    {
                        "Processed": f"{total_processed}/{total_segments}",
                        "Sentences": total_sentences,
                        "Memory": f"{monitor_memory():.1f}MB",
                    },
                )

                # Clean up
                del segments_data
                del sentences_df
                gc.collect()

        # Final report
        print(f"Finished processing {total_processed} segments")
        print(f"Total sentences inserted: {total_sentences}")
        time_taken = time.time() - start_time
        print(f"Total time: {time_taken:.2f} seconds")
        if total_processed > 0 and time_taken > 0:
            print(f"Speed: {total_processed / time_taken:.2f} segments/second")

        # Clean up temp table
        con.execute(f"DROP TABLE IF EXISTS {TEMP_TABLE}")

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting the script.")
        # The 'finally' block below will be executed before the script terminates.
        sys.exit(130)  # Exit with status 130 (standard for SIGINT)
    except duckdb.Error as e:
        print(f"DuckDB Error: {e}")
    except spacy.errors as e:
        print(f"SpaCy Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if con:
            con.close()
            print("DuckDB connection closed.")


if __name__ == "__main__":
    # Check if the database path is provided as a command-line argument
    # If not, prompt the user for the database path

    main()
