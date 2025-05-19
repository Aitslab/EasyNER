"""Segment PubMed abstracts into non-empty segments and create a DuckDB view.

This module splits abstracts on newlines, removes empty segments, and stores
the results in a view.
"""

import os
import sys

import duckdb
from dotenv import load_dotenv


def _create_abstract_segments_view(conn: duckdb.DuckDBPyConnection) -> None:
    """Create a view of non-empty abstract segments.

    splits abstracts on newlines and returns non-empty segments as rows.
    """
    print("Creating abstract_segments view...")

    # Fixed query with proper table references
    conn.execute(
        """--sql
        CREATE OR REPLACE VIEW abstract_segments AS
        SELECT
            subq.pmid,
            TRIM(segment) AS segment,
            ROW_NUMBER() OVER (PARTITION BY subq.pmid ORDER BY segment_id) AS segment_number,
            -- Check if the segment is a header
            -- A header is defined as a less than 10 words all in uppercase
            (
                len(string_split(segment, ' ')) < 10 AND UPPER(TRIM(segment)) = TRIM(segment)
            ) AS is_header
        FROM (
            SELECT
                pmid,
                unnest(string_split(abstract, '\n')) AS segment,
                generate_subscripts(string_split(abstract, '\n'), 1) AS segment_id
            FROM pubmed
            WHERE abstract IS NOT NULL
        ) subq
        WHERE LENGTH(TRIM(segment)) > 0
    """,
    )

    # Check the number of segments
    result = conn.execute("SELECT COUNT(*) FROM abstract_segments").fetchone()
    segment_count = result[0] if result is not None else 0
    print(f"Created view with {segment_count} non-empty segments from the abstracts")

    # Show sample segments
    print("\nSample segments:")
    samples = conn.execute(
        """
        SELECT pmid, segment_number, segment
        FROM abstract_segments
        LIMIT 5
    """,
    ).fetchall()

    for pmid, segment_number, segment in samples:
        print(f"PMID {pmid}, Segment {segment_number}: {segment[:50]}...")

    conn.close()


if __name__ == "__main__":
    try:
        load_dotenv()
        DB_PATH = os.getenv("DB_PATH")
        if DB_PATH is None or DB_PATH.strip() == "":
            msg = "DB_PATH environment variable is not set."
            raise ValueError(msg)
        else:
            print(f"Using database path: {DB_PATH}")

        conn = duckdb.connect(DB_PATH)

        _create_abstract_segments_view(conn)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting the script.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
