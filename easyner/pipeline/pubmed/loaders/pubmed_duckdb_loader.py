"""Loader for converting PubMed XML data to DuckDB format.

This module defines PubMedDuckDBLoader, which processes PubMed XML files
and loads them into a DuckDB database.
"""

import os
from typing import Any

import pandas as pd

from easyner.io.database import DatabaseConnection
from easyner.pipeline.pubmed.loaders import BasePubMedLoader
from easyner.pipeline.utils import get_batch_index_from_filename


class PubMedDuckDBLoader(BasePubMedLoader):
    """PubMed XML to DuckDB."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        baseline: str,
        file_start: int | None = None,
        file_end: int | None = None,
        batch_size: int = 100000,  # Larger batch size for better performance
    ) -> None:
        """Initialize the PubMedDuckDBLoader.

        Args:
            input_path: Directory containing input XML files
            output_path: DuckDB database file path
            baseline: Baseline identifier used in filename parsing
            file_start: Optional start index for file range processing
            file_end: Optional end index for file range processing
            batch_size: Size of batches for database insertion

        """
        self.conn = DatabaseConnection(
            db_path=output_path,
            threads=8,
            memory_limit="4GB",
        )

        # Initialize schema once
        self._initialize_schema()

        super().__init__(input_path, output_path, baseline, file_start, file_end)

        self.input_files = self._get_input_files(input_path)
        self.total_articles = 0
        self.processed_files = 0

    def _initialize_schema(self) -> None:
        """Initialize the database schema with all possible fields from pubmed_parser."""
        self.conn.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS pubmed (
                pmid INTEGER,
                title VARCHAR,
                abstract VARCHAR,
                authors VARCHAR,
                journal VARCHAR,
                pubdate VARCHAR,
                volume VARCHAR,
                issue VARCHAR,
                pages VARCHAR,
                doi VARCHAR,
                mesh_terms VARCHAR,
                publication_types VARCHAR,
                chemical_list VARCHAR,
                keywords VARCHAR,
                affiliations VARCHAR,
                "references" VARCHAR,
                pmc VARCHAR,
                other_id VARCHAR,
                country VARCHAR,
                medline_ta VARCHAR,
                nlm_unique_id VARCHAR,
                issn_linking VARCHAR,
                "delete" BOOLEAN,
                _created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                _source_batch INTEGER,

            )
        """,
        )

    def _write_output(self, data: Any, input_file: str) -> None:
        """Write the processed data to the DuckDB database using register method.

        Args:
            data: List of article dictionaries from pubmed_parser
            input_file: Original input file path

        """
        # Only proceed if we have articles to insert
        if not data:
            return

        try:
            # Start a transaction for better performance
            source_file = os.path.basename(input_file)
            batch = get_batch_index_from_filename(source_file)
            df = pd.DataFrame(data)
            self.conn.execute("BEGIN TRANSACTION")

            # Register the data as a temporary table
            self.conn._conn.register("temp_articles", df)

            # Insert from the registered table to the target table
            self.conn.execute(
                f"""--sql
                INSERT INTO pubmed BY NAME
                SELECT *, '{batch}' as _source_batch
                FROM temp_articles
                """,
            )

            # Commit the transaction
            self.conn.execute("COMMIT")

        except Exception as e:
            # Rollback on error
            self.conn.execute("ROLLBACK")
            print(f"Error during bulk insert for file {input_file}: {str(e)}")

            # Try with smaller batches if the list is large
            if len(data) > 1000:
                print("Trying with smaller batches...")
                try:
                    self.conn.execute("BEGIN TRANSACTION")

                    # Process in batches of 1000
                    batch_size = 1000
                    for i in range(0, len(data), batch_size):
                        batch = data[i : i + batch_size]
                        batch_df = pd.DataFrame(batch)
                        self.conn._conn.register("batch_data", batch_df)
                        self.conn.execute("INSERT INTO pubmed SELECT * FROM batch_data")

                    self.conn.execute("COMMIT")
                    print(f"Successfully loaded {len(data)} articles in batches")

                except Exception as batch_error:
                    self.conn.execute("ROLLBACK")
                    print(f"Batch processing also failed: {str(batch_error)}")
                    print(f"Skipping file: {input_file}")
                    return

        # Update statistics
        self.total_articles += len(data)
        self.processed_files += 1

        # Progress update every 10 files
        if self.processed_files % 10 == 0:
            print(
                f"Processed {self.processed_files} files, {self.total_articles} articles so far",
            )
