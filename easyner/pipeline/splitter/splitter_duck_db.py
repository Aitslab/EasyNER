import duckdb
import pandas as pd
import spacy

# --- Configuration ---
DB_PATH = "your_database.db"  # Replace with your DuckDB database path
SOURCE_TABLE = "your_source_table"  # Replace with your source table name
TEXT_COLUMN = "text_segment"  # Replace with the name of your text column
ID_COLUMN = "id"  # Replace with the name of your ID column
SENTENCES_TABLE = "sentences"
BATCH_SIZE = 50000
SPACY_MODEL = "en_core_web_sm"
N_PROCESS = 4  # Number of parallel spaCy processes


def process_batch(con, nlp, batch):
    """Processes a batch of text segments using spaCy."""
    texts = [item[1] for item in batch]
    ids = [item[0] for item in batch]
    sentences_data = []
    for doc, doc_id in zip(
        nlp.pipe(
            texts,
            batch_size=1000,
            n_process=N_PROCESS,
            disable=["tagger", "parser", "ner"],
        ),
        ids,
    ):
        for sent in doc.sents:
            sentences_data.append({"doc_id": doc_id, "sentence": sent.text})
    return pd.DataFrame(sentences_data)


def main() -> None:
    con = None
    try:
        con = duckdb.connect(database=DB_PATH, read_only=False)

        # --- Ensure sentences table exists ---
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SENTENCES_TABLE} (
                doc_id INTEGER,
                sentence VARCHAR
            );
        """,
        )

        # --- Get already processed IDs ---
        processed_ids_df = con.execute(
            f"SELECT DISTINCT doc_id FROM {SENTENCES_TABLE}",
        ).df()
        processed_ids = set(processed_ids_df["doc_id"].tolist())
        print(f"Found {len(processed_ids)} already processed documents.")

        # --- Load spaCy model ---
        nlp = spacy.load(SPACY_MODEL)

        offset = 0
        total_processed = 0

        while True:
            # --- Fetch a batch of text segments ---
            query = f"SELECT {ID_COLUMN}, {TEXT_COLUMN} FROM {SOURCE_TABLE} LIMIT {BATCH_SIZE} OFFSET {offset}"
            batch = con.execute(query).fetchall()

            if not batch:
                break

            # --- Filter out already processed segments ---
            segments_to_process = [
                item for item in batch if item[0] not in processed_ids
            ]
            num_to_process = len(segments_to_process)

            if num_to_process > 0:
                print(
                    f"Processing {num_to_process} new segments (batch offset: {offset})...",
                )
                sentences_df = process_batch(con, nlp, segments_to_process)

                if not sentences_df.empty:
                    # --- Bulk insert using Pandas DataFrame ---
                    con.append(SENTENCES_TABLE, sentences_df)
                    total_processed += num_to_process
                    print(f"Inserted {len(sentences_df)} new sentences.")

            offset += BATCH_SIZE

        print(f"Finished processing. Total new segments processed: {total_processed}.")

    except duckdb.Error as e:
        print(f"DuckDB Error: {e}")
    except spacy.Error as e:
        print(f"SpaCy Error: {e}")
    finally:
        if con:
            con.close()


if __name__ == "__main__":
    main()
