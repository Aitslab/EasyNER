# ruff: noqa : E501, D100
import os
import time

import duckdb
import pandas as pd
import spacy
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    msg = "DB_PATH environment variable is not set."
    raise ValueError(msg)
else:
    print(f"Using database path: {DB_PATH}")
SOURCE_VIEW = "abstract_segments"  # Use the view you created
SENTENCES_TABLE = "sentences"  # New table name to reflect order
BATCH_SIZE = 5000  # Batch size for reading from DuckDB
SPACY_MODEL = "en_core_web_sm"
N_PROCESS = 3  # Number of parallel spaCy processes
SPACY_BATCH_SIZE = 100  # Batch size for spaCy processing


def process_batch(nlp, batch: list[tuple]) -> pd.DataFrame:
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
        for sent in doc.sents:
            sentences_data.append(
                {
                    "pmid": pmid,
                    "segment_number": segment_number,
                    "sentence_in_segment_order": sentence_in_segment_order,
                    "sentence": sent.text,
                },
            )
            sentence_in_segment_order += 1

    return pd.DataFrame(sentences_data)


def main() -> None:
    con = None
    try:
        con = duckdb.connect(database=DB_PATH, read_only=False)

        # --- Ensure sentences table exists with new schema ---
        # Added segment_number and sentence_in_segment_order
        con.execute(
            f"""--sql
            CREATE TABLE IF NOT EXISTS {SENTENCES_TABLE} (
                pmid INTEGER,
                segment_number INTEGER,
                sentence_in_segment_order INTEGER,
                sentence VARCHAR
            );
        """,
        )

        # --- Get already processed segment IDs (pmid, segment_number) ---
        # Check for existing entries based on both pmid and segment_number
        print(f"Fetching already processed segment IDs from {SENTENCES_TABLE}...")
        processed_segments_df = con.execute(
            f"SELECT DISTINCT pmid, segment_number FROM {SENTENCES_TABLE}",
        ).df()
        # Create a set of tuples for efficient lookup
        processed_segments = set(
            zip(processed_segments_df["pmid"], processed_segments_df["segment_number"]),
        )
        print(f"Found {len(processed_segments)} already processed segments.")

        # --- Load spaCy model once ---
        print(f"Loading spaCy model: {SPACY_MODEL}...")
        # Load only components needed for sentence segmentation
        nlp = spacy.load(
            SPACY_MODEL,
            exclude=["ner", "attribute_ruler", "lemmatizer"],
        )
        print("SpaCy model loaded.")

        offset = 0
        total_segments_processed = 0
        total_sentences_inserted = 0

        while True:
            # --- Fetch a batch of text segments from the view ---
            # Select pmid, segment_number, and segment text
            query = f"""--sql
                SELECT pmid, segment_number, segment
                FROM {SOURCE_VIEW}
                ORDER BY pmid, segment_number
                -- Maintains consistent batching, however not strictly necessary
                -- The nlp pipe can handle potential differences in batchs sizes
                -- For each batch we still check for already processed segments
                -- PMIDs are already highly sorted although not garanteed to be in order
                LIMIT {BATCH_SIZE} OFFSET {offset}
            """
            batch = con.execute(query).fetchall()

            if not batch:
                print("No more segments to fetch. Exiting.")
                break

            # --- Filter out already processed segments ---
            # A segment is processed if its (pmid, segment_number) pair is in our set
            segments_to_process = [
                item for item in batch if (item[0], item[1]) not in processed_segments
            ]
            num_to_process = len(segments_to_process)

            if num_to_process > 0:
                print(
                    f"Processing {num_to_process} new segments (batch offset: {offset})...",
                )

                # --- Process batch and get sentences with order ---
                sentences_df = process_batch(nlp, segments_to_process)

                if not sentences_df.empty:
                    # --- Bulk insert using Pandas DataFrame ---
                    # Use append which is optimized for DataFrames
                    con.append(SENTENCES_TABLE, sentences_df)
                    total_segments_processed += num_to_process
                    total_sentences_inserted += len(sentences_df)
                    print(
                        f"Inserted {len(sentences_df)} new sentences for {num_to_process} segments.",
                    )
                else:
                    print(
                        f"No sentences generated for the {num_to_process} segments in this batch.",
                    )

            # Add the newly processed segments to our set to avoid reprocessing in case of script interruption
            for item in segments_to_process:
                processed_segments.add((item[0], item[1]))

            offset += BATCH_SIZE

        print(
            f"Finished processing. Total new segments processed: {total_segments_processed}.",
        )
        print(f"Total new sentences inserted: {total_sentences_inserted}.")
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt: Exiting the script.")
        if con:
            con.close()
            print("DuckDB connection closed.")
        raise e

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
