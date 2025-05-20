import os
import sys

import duckdb
from dotenv import load_dotenv


def _check_newlines_in_abstracts(conn: duckdb.DuckDBPyConnection) -> None:
    """Check what kind of newline characters exist in abstracts."""
    # Check for actual newlines (ASCII 10)
    result = conn.execute(
        """--sql
        SELECT COUNT(*)
        FROM pubmed
        WHERE abstract LIKE '%' || chr(10) || '%'
    """,
    ).fetchone()
    newline_count = result[0] if result is not None else 0

    # Get a sample with newlines
    if newline_count > 0:
        sample = conn.execute(
            """
            SELECT abstract
            FROM pubmed
            WHERE abstract LIKE '%' || chr(10) || '%'
            LIMIT 1
        """,
        ).fetchone()

        if sample:
            abstract = sample[0]
            print(f"Found {newline_count} abstracts with actual newlines")
            print("Sample abstract (showing first 100 chars):")
            print(repr(abstract[:100]))
            print("\nSplitting this abstract on newlines would yield:")
            segments = abstract.split("\n")
            segments_to_show = 20
            for i, segment in enumerate(segments[:segments_to_show]):
                if len(segment) > 50:
                    print(f"  Segment {i+1}: {segment[:50]} [...] " + segment[-40:])
                elif len(segment) == 0:
                    print(f"  Segment {i+1}: <empty>")
                else:
                    print(f"  Segment {i+1}: {segment}")
            if len(segments) > segments_to_show:
                print(f"  (and {len(segments)-segments_to_show} more segments)")
    else:
        print("No abstracts with actual newlines found")

        # Check if abstracts have \r instead (ASCII 13)
        cr_result = conn.execute(
            """--sql
            SELECT COUNT(*)
            FROM pubmed
            WHERE abstract LIKE '%' || chr(13) || '%'
        """,
        ).fetchone()
        cr_count = cr_result[0] if cr_result is not None else 0

        if cr_count > 0:
            print(f"Found {cr_count} abstracts with carriage returns (\\r)")

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

        _check_newlines_in_abstracts(conn)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting the script.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
