"""Module for marking and processing truncated abstracts in PubMed data.

This module provides functionality to identify and handle abstracts that are truncated,
specifically those marked with for example '(ABSTRACT TRUNCATED AT 250 WORDS)'.

I believe from the pubmed docs that they started truncating abstracts in the olden days
when uploading. Then the limit has changed over time and now they are not truncating...
anymore.

TODO: Check if truncation is surronded by newline.
"""

import sys
from pathlib import Path

import duckdb


def find_truncated_abstracts(db_path: Path) -> None:
    """Find mark in abstract text and remove it, marking the abstract as truncated."""
    descriptive_pattern = "(ABSTRACT TRUNCATED AT <number> WORDS)"
    regex_pattern = r"\(ABSTRACT TRUNCATED AT \d+ WORDS\)"
    conn = duckdb.connect(db_path)
    if conn is None:
        print(f"Failed to connect to the database at {db_path}")
        return
    # Check for the pattern in all abstracts using regex
    result = conn.execute(
        """--sql
        SELECT COUNT(*)
        FROM pubmed
        WHERE regexp_matches(abstract, ?)
    """,
        (regex_pattern,),
    ).fetchone()
    count = result[0] if result is not None else 0
    print(
        f"Found {count} abstracts with the truncation pattern matching '{descriptive_pattern}'",
    )


def remove_truncation_mark_and_add_truncated(db_path: Path) -> None:
    """Remove truncation mark from abstracts and add a 'truncated' flag."""
    descriptive_pattern = "(ABSTRACT TRUNCATED AT <number> WORDS)"
    regex_pattern = r"\(ABSTRACT TRUNCATED AT \d+ WORDS\)"  # Regex for "digits"
    conn = duckdb.connect(str(db_path))  # Ensure db_path is string for DuckDB < 0.9.0
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
        SET abstract = regexp_replace(abstract, ?, '', 'g'),
            is_truncated = TRUE
        WHERE regexp_matches(abstract, ?)
    """,
        (regex_pattern, regex_pattern),
    )
    print(
        f"Removed truncation mark and marked abstracts as truncated "
        f"for patterns matching '{descriptive_pattern}'",
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
