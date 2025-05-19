"""Segment PubMed abstracts into non-empty segments and create a DuckDB view.

This module splits abstracts on newlines, removes empty segments, and stores
the results in a view.
"""

import os
import random
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
        """--sql
        SELECT pmid, segment_number, segment, is_header
        FROM abstract_segments
        WHERE is_header = FALSE
        LIMIT 5
    """,
    ).fetchall()

    samples += conn.execute(
        """--sql
        SELECT pmid, segment_number, segment, is_header
        FROM abstract_segments
        WHERE is_header = TRUE
        LIMIT 5
    """,
    ).fetchall()

    random.shuffle(samples)
    print("PMID   | Segment nbr | Header | Segment")
    print("--------------------------------------------------")
    for pmid, segment_number, segment, is_header in samples:
        if len(segment) > 50:
            print(f"{pmid} | {segment_number} | {is_header} | {segment[:50]}..")
        else:
            print(f"{pmid} | {segment_number} | {is_header} | {segment}")

    # Test that headers are correctly identified
    header_samples = conn.execute(
        """--sql
        SELECT pmid, segment_number, segment
        FROM abstract_segments
        WHERE is_header = TRUE
        LIMIT 5
    """,
    ).fetchall()
    print("\nSample header segments:")

    for pmid, segment_number, segment in header_samples:
        seg_text = segment
        if len(seg_text) > 50:
            print(f"PMID {pmid}, Segment {segment_number}: {seg_text[:50]}..")
        else:
            print(f"PMID {pmid}, Segment {segment_number}: {seg_text}")

    # segments with uppercase LIKE BACKGROUND
    unmarked_headers = conn.execute(
        """--sql
        SELECT DISTINCT segment
        FROM abstract_segments
        WHERE is_header = FALSE
        AND UPPER(segment) = segment
        -- this would inlcude non structured segments with "BACKGROUND Ventilarors..."
        -- AND segment LIKE '%BACKGROUND%'
        LIMIT 50
    """,
    ).fetchall()
    print("\nPotentially unmarked headers (ALL UPPERCASE and not marked):")

    for segment in unmarked_headers:
        seg_text = segment[0]
        if len(seg_text) > 50:
            print(f"Segment: {seg_text[:50]}..")
        else:
            print(f"Segment: {seg_text}")

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
