# ruff: noqa : E501, D100
import os
import sys  # Added for sys.exit
import time

import duckdb
import pandas as pd
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
SOURCE_VIEW = "abstract_segments"  # Use the view you created
SENTENCES_TABLE = "sentences"  # New table name to reflect order
BATCH_SIZE = 5000  # Batch size for reading from DuckDB
SPACY_MODEL = "en_core_web_sm"
N_PROCESS = 3  # Number of parallel spaCy processes
SPACY_BATCH_SIZE = 100  # Batch size for spaCy processing


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
            exclude=["ner", "attribute_ruler", "lemmatizer"],
        )
        print("SpaCy model loaded.")

        # Calculate total number of segments to process for tqdm
        count_query = f"""--sql
            SELECT COUNT(*)
            FROM {SOURCE_VIEW} s
            WHERE NOT EXISTS (
                SELECT 1 FROM {SENTENCES_TABLE} t
                WHERE t.pmid = s.pmid AND t.segment_number = s.segment_number
            )
            AND s.is_header = FALSE;
        """
        count_result = con.execute(count_query).fetchone()
        total_segments_to_process_initially = count_result[0] if count_result else 0
        print(f"Total segments to process: {total_segments_to_process_initially}")

        total_segments_processed = 0
        total_sentences_inserted = 0
        overall_start_time = time.time()  # For overall SPS calculation

        with tqdm(
            total=total_segments_to_process_initially,
            unit="segment",
            desc="Processing Segments",
        ) as pbar:
            while True:
                query = f"""--sql
                    SELECT s.pmid, s.segment_number, s.segment
                    FROM {SOURCE_VIEW} s
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {SENTENCES_TABLE} t
                        WHERE t.pmid = s.pmid AND t.segment_number = s.segment_number
                    )
                    AND s.is_header = FALSE -- Do not process header segments
                    LIMIT {BATCH_SIZE}
                """
                segments_to_process = con.execute(query).fetchall()
                num_to_process = len(segments_to_process)

                if not segments_to_process:
                    # Check if the initial count was zero and we never entered the processing logic
                    if total_segments_to_process_initially == 0:
                        print(
                            "\nNo segments to process based on initial count. Exiting.",
                        )
                        break
                    # If segments were processed, but now no more are found, it's a normal exit.
                    # Or, if the progress bar already reflects completion.
                    if total_segments_processed >= pbar.total:
                        print("\nNo more segments to process. Exiting.")
                        break
                    else:
                        # If no segments are fetched but tqdm indicates more are expected,
                        # re-calculate total and update tqdm.
                        current_total_recheck_result = con.execute(
                            count_query,
                        ).fetchone()
                        current_total_recheck = (
                            current_total_recheck_result[0]
                            if current_total_recheck_result
                            else 0
                        )
                        if current_total_recheck > total_segments_processed:
                            pbar.total = current_total_recheck
                            pbar.refresh()
                            if (
                                not segments_to_process
                            ):  # Still no segments after refresh
                                time.sleep(
                                    1,
                                )  # Wait a bit before trying again, maybe data is incoming
                                continue
                        else:  # No more segments even after recheck
                            print(
                                "\nNo more segments to process after re-check. Exiting.",
                            )
                            break

                if num_to_process > 0:
                    # --- Process batch and get sentences with order ---
                    sentences_df = process_batch(nlp, segments_to_process)
                    if not sentences_df.empty:
                        # --- Bulk insert using Pandas DataFrame ---
                        con.append(SENTENCES_TABLE, sentences_df)
                        con.commit()  # Commit after each batch append

                        total_sentences_inserted += len(sentences_df)

                    total_segments_processed += num_to_process
                    pbar.update(num_to_process)
                    elapsed_time = time.time() - overall_start_time
                    current_sps = (
                        total_segments_processed / elapsed_time
                        if elapsed_time > 0
                        else 0
                    )
                    pbar.set_postfix(
                        {
                            "Processed": f"{total_segments_processed}/{pbar.total}",
                            "Sentences": f"{total_sentences_inserted}",
                            "SPS": f"{current_sps:.2f}",
                        },
                    )

        print(
            f"\nFinished processing. Total segments processed: {total_segments_processed}.",
        )
        print(f"Total sentences inserted: {total_sentences_inserted}.")
        total_time_taken = time.time() - overall_start_time
        if (
            total_segments_processed > 0 and total_time_taken > 0
        ):  # Avoid division by zero if no segments or no time
            print(f"Total time taken: {total_time_taken:.2f} seconds.")
            overall_sps = total_segments_processed / total_time_taken
            print(f"Overall Segments Per Second (SPS): {overall_sps:.2f}")
        elif total_segments_processed == 0:
            print("No segments were processed.")
        else:  # total_time_taken is 0 or less, which is unlikely but good to handle
            print(
                f"Total time taken: {total_time_taken:.2f} seconds (SPS not calculable).",
            )

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
