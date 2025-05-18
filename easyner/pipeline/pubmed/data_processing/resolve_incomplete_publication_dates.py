"""Resolve incomplete article publication dates in the PubMed dataset.

Only works with the DuckDB format

Article pubdates are not standardized to YYYY-MM-DD format as some contain only
YYYY or YYYY-MM format.

Thus they are saved as strings in the database.
"""

import sys
from pathlib import Path

import duckdb


def _resolve_varchar_to_date(db_path: Path) -> None:
    """Resolve incomplete pubmed date data in duckdb.

    Simple heuristic:
    - Get pubdate column
    - Any missing day AND/OR month data is set to 01
    - If nothing is missing, new date = old date
    - New date is saved as DATE YYYY-MM-DD

    """
    conn = duckdb.connect(db_path)

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

    conn.close()


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print("---------------------------------------------------")
            print("Resolve duplicate PMIDs in the PubMed dataset")
            print("---------------------------------------------------")
            print("Accepts a single argument: the path to the DuckDB database")
            print("Example: python resolve_duplicate_pmid.py /path/to/pubmed.db")
            print("---------------------------------------------------")
            db_path = input("Enter the path to the DuckDB database: ...").strip()
            if not db_path:
                print("No path provided. Exiting.")
                sys.exit(1)

        db_path = Path(sys.argv[1])
        _resolve_varchar_to_date(db_path)
        print(f"Resolved pubdate values in {db_path}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting the script.")
        sys.exit(0)
