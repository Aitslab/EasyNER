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

        # Instead of fetching all processed segments upfront,
        # directly filter in the database during each batch:
        while True:
            query = f"""--sql
                SELECT s.pmid, s.segment_number, s.segment
                FROM {SOURCE_VIEW} s
                WHERE NOT EXISTS (
                    SELECT 1 FROM {SENTENCES_TABLE} t
                    WHERE t.pmid = s.pmid AND t.segment_number = s.segment_number
                )
                AND s.is_header = FALSE -- Do not process header segments
                ORDER BY s.pmid, s.segment_number
                LIMIT {BATCH_SIZE} OFFSET {offset}
            """
            start_time = time.time()
            segments_to_process = con.execute(query).fetchall()
            end_time = time.time()
            print(
                f"Fetched filtered {len(segments_to_process)} segments in "
                f"{end_time - start_time:.2f} seconds.",
            )
            num_to_process = len(segments_to_process)

            if not segments_to_process:
                print("No more segments to process. Exiting.")
                break

            if num_to_process > 0:
                print(
                    (
                        f"Processing {num_to_process} new segments "
                        f"(batch offset: {offset})..."
                    ),
                )
                # --- Process batch and get sentences with order ---
                start_time = time.time()
                sentences_df = process_batch(nlp, segments_to_process)
                end_time = time.time()
                print(
                    f"Processed {num_to_process} segments in "
                    f"{end_time - start_time:.2f} seconds.",
                )
                if not sentences_df.empty:
                    # --- Bulk insert using Pandas DataFrame ---
                    # Use append which is optimized for DataFrames
                    con.append(SENTENCES_TABLE, sentences_df)
                    con.commit()
                    total_segments_processed += num_to_process
                    total_sentences_inserted += len(sentences_df)
                    print(
                        f"Inserted {len(sentences_df)} new sentences for "
                        f"{num_to_process} segments.",
                    )
                else:
                    print(
                        f"No sentences generated for the {num_to_process} segments in this batch.",
                    )

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
