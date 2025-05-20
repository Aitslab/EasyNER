"""Resolve incomplete article publication dates in the PubMed dataset.

Only works with the DuckDB format

Article pubdates are not standardized to YYYY-MM-DD format as some contain only
YYYY or YYYY-MM format.

Thus they are saved as strings in the database.
"""

import os
import sys
from pathlib import Path

import duckdb


def _resolve_varchar_to_date(conn: duckdb.DuckDBPyConnection) -> None:
    """Resolve incomplete pubmed date data in duckdb.

    Simple heuristic:
    - Get pubdate column
    - Any missing day AND/OR month data is set to 01
    - If nothing is missing, new date = old date
    - New date is saved as DATE YYYY-MM-DD

    """
    # First, add a new column to store the resolved dates
    conn.execute("ALTER TABLE pubmed ADD COLUMN IF NOT EXISTS date_resolved DATE")

    # Update the new column based on the pattern of the existing pubdate
    conn.execute(
        """--sql
        UPDATE pubmed
        SET date_resolved =
            CASE
                WHEN LENGTH(pubdate) = 4 THEN CAST(pubdate || '-01-01' AS DATE)
                WHEN LENGTH(pubdate) = 7 THEN CAST(pubdate || '-01' AS DATE)
                WHEN LENGTH(pubdate) = 10 THEN CAST(pubdate AS DATE)
                ELSE NULL
            END
    """,
    )


if __name__ == "__main__":
    try:
        db_path_str = None
        if len(sys.argv) == 2:
            db_path_str = sys.argv[1]
        else:
            print("---------------------------------------------------")
            print("Resolve incomplete publication dates in the PubMed dataset")
            print("---------------------------------------------------")
            print(
                "This script resolves incomplete article publication dates "
                "by updating them to a YYYY-MM-DD format.",
            )
            print("It can accept the database path as a command-line argument.")
            print(
                "Example: python resolve_incomplete_publication_dates.py /path/to/pubmed.db",  # noqa: E501
            )
            print("---------------------------------------------------")
            user_input = input(
                "Press enter to use DB_PATH env variable "
                "or enter the path to the DuckDB database: ...",
            ).strip()
            if user_input:
                db_path_str = user_input
            else:
                print(
                    "No path provided via input, attempting to use environment variable DB_PATH.",  # noqa: E501
                )
                db_path_str = os.getenv("DB_PATH")

        if not db_path_str:
            print(
                "Error: Database path not provided via command-line argument, "
                "user input, or DB_PATH environment variable.",
            )
            sys.exit(1)

        db_path = Path(db_path_str)
        if not db_path.exists() or not db_path.is_file():
            print(f"Error: Database file not found at {db_path}")
            sys.exit(1)

        conn = duckdb.connect(db_path)
        try:
            _resolve_varchar_to_date(conn)
            print(f"Resolved pubdate values in {db_path}")
        finally:
            conn.close()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting the script.")
        sys.exit(0)
    except duckdb.Error as e:
        print(f"A DuckDB error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
