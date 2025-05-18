import duckdb
import pytest


def test_insert_on_conflict_returning_generic():
    """Tests the core behavior of INSERT ON CONFLICT DO NOTHING RETURNING
    to return IDs of successfully inserted rows (non-duplicates).
    """
    # Connect to an in-memory DuckDB database
    con = duckdb.connect(database=":memory:", read_only=False)

    table_name = "my_data"
    id_col = "item_id"
    value_col = "item_value"

    try:
        # 1. Setup Target Table with Primary Key (for UNIQUE constraint/ART index)
        con.execute(
            f"CREATE TABLE {table_name} ({id_col} INTEGER PRIMARY KEY, {value_col} VARCHAR)"
        )

        # 2. Insert Some Initial Data (These will be duplicates in the source)
        initial_data = [(1, "original_A"), (2, "original_B")]
        # --- FIX IS HERE ---
        # Use executemany for inserting multiple rows from a list of tuples
        con.executemany(
            f"INSERT INTO {table_name} ({id_col}, {value_col}) VALUES (?, ?)",
            initial_data,
        )
        # -------------------

        initial_count = con.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        assert initial_count == len(initial_data)

        # 3. Define Source Data (using VALUES for simplicity in the test)
        # Contains duplicates (IDs 1, 2) and non-duplicates (IDs 3, 4)
        # Includes an internal duplicate in source (ID 1 appears twice)
        source_data_values = [
            (1, "new_A_v1"),  # Duplicate of existing ID 1
            (3, "new_C_v1"),  # New ID
            (2, "new_B_v1"),  # Duplicate of existing ID 2
            (4, "new_D_v1"),  # New ID
            (
                1,
                "new_A_v2",
            ),  # Duplicate of existing ID 1, also internal duplicate in source
        ]
        # Based on initial_data, IDs 3 and 4 are the unique IDs in source that are NEW to the target.
        # The rows successfully inserted are those from source with IDs 3 and 4.
        expected_inserted_ids = sorted([3, 4])

        # 4. Execute INSERT ON CONFLICT DO NOTHING RETURNING
        # Use VALUES directly in the INSERT statement for conciseness
        print("\n--- Executing INSERT ON CONFLICT DO NOTHING RETURNING ---")
        insert_sql = f"""
        INSERT INTO {table_name} ({id_col}, {value_col})
        SELECT * FROM (VALUES {', '.join([str(row) for row in source_data_values])}) AS t({id_col}, {value_col})
        ON CONFLICT ({id_col}) DO NOTHING
        RETURNING {id_col}; -- Return the ID of each successfully inserted row
        """

        returned_ids_result = con.execute(insert_sql).fetchall()
        # Extract IDs from the list of tuples and sort for consistent comparison
        returned_ids = sorted([row[0] for row in returned_ids_result])

        print(f"Returned (Inserted) IDs: {returned_ids}")

        # 5. Assertions
        # Assert that the IDs returned are correct
        assert (
            returned_ids == expected_inserted_ids
        ), "Mismatch in returned IDs"

        # Assert the final count in the target table is correct
        # Initial count + count of successfully inserted unique IDs
        final_count = con.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        expected_final_count = initial_count + len(expected_inserted_ids)
        assert (
            final_count == expected_final_count
        ), "Mismatch in final table row count"

        # Assert that the values of initial rows were not updated
        initial_rows_final_state = con.execute(
            f"SELECT {id_col}, {value_col} FROM {table_name} WHERE {id_col} IN (1, 2) ORDER BY {id_col}"
        ).fetchall()
        # Values for IDs 1 and 2 should still be 'original_A' and 'original_B'
        assert (
            initial_rows_final_state == initial_data
        ), "Original rows were unexpectedly updated by DO NOTHING"

    finally:
        # Clean up
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.close()
