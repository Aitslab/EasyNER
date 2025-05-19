"""Module for marking and processing truncated abstracts in PubMed data.

This module provides functionality to identify and handle abstracts that are truncated,
specifically those marked with '(ABSTRACT TRUNCATED AT 250 WORDS)'.
"""

import sys
from pathlib import Path

import duckdb


def find_truncated_abstracts(db_path: Path) -> None:
    """Find mark in abstract text and remove it, marking the abstract as truncated."""
    pattern = "(ABSTRACT TRUNCATED AT 250 WORDS)"  # Appears at end of abstract
    conn = duckdb.connect(db_path)
    if conn is None:
        print(f"Failed to connect to the database at {db_path}")
        return
    # Check for the pattern in all abstracts
    result = conn.execute(
        """--sql
        SELECT COUNT(*)
        FROM pubmed
        WHERE abstract LIKE '%' || ? || '%'
    """,
        (pattern,),
    ).fetchone()
    count = result[0] if result is not None else 0
    print(f"Found {count} abstracts with the truncation pattern '{pattern}'")


def remove_truncation_mark_and_add_truncated(db_path: Path) -> None:
    """Remove truncation mark from abstracts and add a 'truncated' flag."""
    pattern = "(ABSTRACT TRUNCATED AT 250 WORDS)"
    conn = duckdb.connect(db_path)
    if conn is None:
        print(f"Failed to connect to the database at {db_path}")
        return
    # Add a new column to mark truncated abstracts
    conn.execute(
        """--sql
        ALTER TABLE pubmed
        ADD COLUMN IF NOT EXISTS
        is_truncated BOOLEAN DEFAULT FALSE""",
    )
    conn.execute(
        """--sql
        UPDATE pubmed
        SET abstract = REPLACE(abstract, ?, ''),
            is_truncated = TRUE
        WHERE abstract LIKE '%' || ? || '%'
    """,
        (pattern, pattern),
    )
    print(
        f"Removed truncation mark and marked abstracts as truncated "
        f"for pattern '{pattern}'",
    )


if __name__ == "__main__":
    try:
        db_path = Path("/home/callebalik/EasyNER/data/temp/pubmed.db")
        find_truncated_abstracts(db_path)
        remove_truncation_mark_and_add_truncated(db_path)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting the script.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
