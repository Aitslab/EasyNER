import sys
from pathlib import Path

import duckdb


def _check_newlines_in_abstracts(db_path: Path) -> None:
    """Check what kind of newline characters exist in abstracts."""
    conn = duckdb.connect(db_path)
    if conn is None:
        print(f"Failed to connect to the database at {db_path}")
        return

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
        db_path = Path("/home/callebalik/EasyNER/data/temp/pubmed.db")
        _check_newlines_in_abstracts(db_path)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting the script.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
