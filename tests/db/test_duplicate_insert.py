# Test if a error is raised if duplicate keys are attempted to be inserted
# Create articles table

from collections.abc import Generator

import duckdb
import pandas as pd
import pytest

from easyner.io.database import (
    ArticleRepository,
    DatabaseConnection,
    DuckDBHandler,
)
from easyner.io.database.repositories.entity_repository import EntityRepository
from easyner.io.database.repositories.sentence_repository import (
    SentenceRepository,
)


@pytest.fixture(scope="class")
def db_conn() -> Generator[DatabaseConnection, None, None]:
    """Set up/tear down a database connection for a test class."""
    ddb_handler = DuckDBHandler(":memory:")
    conn = ddb_handler.conn
    ddb_handler.create_base_tables()  # Create article, sentences and entities tables
    yield conn  # Provide the connection to the test
    if conn:
        conn.close()  # Teardown: close connection


@pytest.fixture(scope="class")
def article_repo(db_conn: DatabaseConnection) -> ArticleRepository:
    """Fixture to create an ArticleRepository instance and its table."""
    repo = ArticleRepository(db_conn)
    repo._create_table()  # Ensure table is created
    return repo


@pytest.fixture(scope="class")
def sentence_repo(
    db_conn: DatabaseConnection,
    article_repo: ArticleRepository,
) -> SentenceRepository:
    """Fixture to create a SentenceRepository instance and its table."""
    repo = SentenceRepository(db_conn)
    repo._create_table()  # Ensure table is created
    return repo


@pytest.fixture(scope="class")
def entity_repo(
    db_conn: DatabaseConnection,
    sentence_repo: SentenceRepository,
) -> EntityRepository:
    """Fixture to create an EntityRepository instance and its table."""
    repo = EntityRepository(db_conn)
    repo._create_table()  # Ensure table is created
    return repo


def test_insert_duplicate_key(article_repo: ArticleRepository):
    """Test that inserting a duplicate key raises a ConstraintException."""
    # Create a sample DataFrame with duplicate keys
    data = {
        "article_id": [1, 1],
        "title": ["Title 1", "Title 1"],
    }
    df = pd.DataFrame(data)

    # Attempt to insert the DataFrame into the database
    # using the injected repository
    with pytest.raises(duckdb.ConstraintException) as excinfo:
        article_repo.insert_many_transactional(df)  # Use the injected fixture

    assert (
        "duplicate key"
        in str(excinfo.value).lower()
        # or "primary key constraint failed" in str(excinfo.value).lower()
    )


def test_insert_duplicate_key_with_logging(article_repo: ArticleRepository):
    """Test that insert with log_duplicates_to_duplicates_table=True.

    handles duplicates without errors.
    """
    # Clear any existing data
    conn = article_repo.conn  # Use DatabaseConnection directly
    conn.execute(f"DELETE FROM {article_repo.table_name}")
    if hasattr(article_repo, "duplicate_table_name"):
        conn.execute(f"DELETE FROM {article_repo.duplicate_table_name}")

    # Create initial record
    article_repo.insert({"article_id": 100, "title": "Original Title"})

    # Create a sample DataFrame with duplicate keys -
    # both against DB and internal
    data = {
        "article_id": [
            100,
            101,
            101,
        ],  # 100 conflicts with DB, 101 is internal duplicate
        "title": ["Duplicate Title", "Title A", "Title B"],
    }
    df = pd.DataFrame(data)

    # This should NOT raise an exception
    article_repo.insert_many_non_transactional(df, log_duplicates=True)

    # Verify main table has 2 records: original 100 + first 101
    main_records = conn.execute(
        f"SELECT article_id, title FROM {article_repo.table_name} "
        f"ORDER BY article_id",
    ).fetchall()
    assert len(main_records) == 2
    assert main_records[0] == (100, "Original Title")  # Original not changed
    assert main_records[1] == (101, "Title A")  # First occurrence added

    # Verify duplicates table has 2 records: duplicate 100 + second 101
    dup_records = conn.execute(
        f"SELECT article_id, title FROM {article_repo.duplicate_table_name} "
        f"ORDER BY article_id",
    ).fetchall()
    assert len(dup_records) == 2
    assert dup_records[0][0] == 100  # ID match
    assert dup_records[0][1] == "Duplicate Title"  # Content match
    assert dup_records[1][0] == 101  # ID match
    assert dup_records[1][1] == "Title B"  # Content match


def test_insert_log_duplicates_to_duplicates_table(
    article_repo: ArticleRepository,
):
    """Test _insert_log_duplicates_to_duplicates_table correctly handles duplicates.

    This test verifies that:
    1. Existing records in DB are not overwritten
    2. Conflicts with existing records are logged to duplicates table
    3. Internal duplicates within batch are detected and logged
    4. Only unique new records are inserted to main table
    """
    # Step 1: Add some initial data to the database
    initial_data = {
        "article_id": [1, 2, 3],
        "title": ["Existing Title 1", "Existing Title 2", "Existing Title 3"],
    }
    initial_df = pd.DataFrame(initial_data)

    # Use standard insert without duplicate handling to set up test data
    article_repo.insert_many_non_transactional(initial_df)

    # Step 2: Create a test batch with different types of records:
    # - Conflicts with DB (article_id: 1, 3)
    # - Internal duplicates (article_id: 5 appears twice)
    # - Unique new records (article_id: 4, 6)
    test_data = {
        "article_id": [1, 3, 4, 5, 5, 6],
        "title": [
            "New Title 1 (DB Conflict)",
            "New Title 3 (DB Conflict)",
            "New Title 4 (Unique)",
            "New Title 5A (Internal Duplicate)",
            "New Title 5B (Internal Duplicate)",
            "New Title 6 (Unique)",
        ],
    }
    test_df = pd.DataFrame(test_data)

    # Step 3: Insert with duplicate logging enabled
    article_repo.insert_many_non_transactional(test_df, log_duplicates=True)

    # Step 4: Verify results
    conn = article_repo.conn  # Use DatabaseConnection directly

    # Check main table - should have original 3 records + 2 new unique
    # records (total 5)
    main_table_records = conn.execute(
        f"SELECT article_id, title FROM {article_repo.table_name} "
        f"ORDER BY article_id",
    ).fetchall()

    # Check duplicates table - should have 2 DB conflicts + 1 internal
    # duplicate (total 3)
    duplicates_table_records = conn.execute(
        f"SELECT article_id, title FROM {article_repo.duplicate_table_name} "
        f"ORDER BY article_id",
    ).fetchall()

    # Assertions
    assert (
        len(main_table_records) == 6
    ), "Main table should contain 6 [1,2,3,4,5,6] records"

    # Check specific records in main table
    main_ids = [r[0] for r in main_table_records]
    assert set(main_ids) == {
        1,
        2,
        3,
        4,
        5,
        6,
    }, "Main table should contain IDs 1,2,3,4,6"

    # Original records shouldn't be changed
    assert main_table_records[0] == (
        1,
        "Existing Title 1",
    ), "Existing record should not be modified"

    # Verify unique new records were added
    assert (
        4,
        "New Title 4 (Unique)",
    ) in main_table_records, "New unique record ID 4 should be in main table"
    assert (
        6,
        "New Title 6 (Unique)",
    ) in main_table_records, "New unique record ID 6 should be in main table"

    # Only one version of article_id 5 should be in the main table
    # (the first one)
    assert main_ids.count(5) == 1, "Only one version of ID 5 should be in main table"

    # Duplicates table checks
    assert (
        len(duplicates_table_records) == 3
    ), "Duplicates table should contain 3 records"

    dup_records = [(r[0], r[1]) for r in duplicates_table_records]
    # Check for specific duplicate records
    assert (
        1,
        "New Title 1 (DB Conflict)",
    ) in dup_records, "DB conflict record ID 1 should be in duplicates table"
    assert (
        3,
        "New Title 3 (DB Conflict)",
    ) in dup_records, "DB conflict record ID 3 should be in duplicates table"
    assert (
        5,
        "New Title 5B (Internal Duplicate)",
    ) in dup_records, "Internal duplicate ID 5B should be in duplicates table"


def test_insert_duplicate_sentences(
    sentence_repo: SentenceRepository,
    article_repo: ArticleRepository,
):
    """Test duplicate handling in SentenceRepository."""
    # Clear any existing data
    conn = sentence_repo.conn
    conn.execute(f"DELETE FROM {sentence_repo.table_name}")
    if hasattr(sentence_repo, "duplicate_table_name"):
        conn.execute(f"DELETE FROM {sentence_repo.duplicate_table_name}")
    conn.execute(f"DELETE FROM {article_repo.table_name}")

    # First, create necessary articles to satisfy foreign key constraints
    articles_data = {
        "article_id": [1, 2, 3],
        "title": ["Article One", "Article Two", "Article Three"],
    }
    article_repo.insert_many_non_transactional(pd.DataFrame(articles_data))

    # Now proceed with sentence testing as before
    # Step 1: Add initial sentences
    initial_data = {
        "sentence_id": [1, 2, 3],
        "article_id": [1, 1, 2],  # These now reference existing articles
        "text": ["Sentence One", "Sentence Two", "Sentence Three"],
        "position": [0, 1, 0],
    }
    initial_df = pd.DataFrame(initial_data)
    sentence_repo.insert_many_non_transactional(initial_df)

    # Step 2: Create test data with duplicates
    test_data = {
        "sentence_id": [1, 3, 4, 5, 5, 6],
        "article_id": [1, 2, 2, 3, 3, 3],
        "text": [
            "New Sentence One (DB Conflict)",
            "New Sentence Three (DB Conflict)",
            "New Sentence Four (Unique)",
            "New Sentence Five A (Internal Duplicate)",
            "New Sentence Five B (Internal Duplicate)",
            "New Sentence Six (Unique)",
        ],
        "position": [0, 0, 1, 0, 0, 1],
    }
    test_df = pd.DataFrame(test_data)

    # Step 3: Insert with duplicate logging
    sentence_repo.insert_many_non_transactional(test_df, log_duplicates=True)

    # Step 4: Verify results
    main_table_records = conn.execute(
        f"SELECT sentence_id, article_id, text FROM {sentence_repo.table_name} "
        f"ORDER BY sentence_id",
    ).fetchall()

    duplicates_records = conn.execute(
        f"SELECT sentence_id, article_id, text FROM {sentence_repo.duplicate_table_name} "
        f"ORDER BY sentence_id",
    ).fetchall()

    # Assertions
    assert len(main_table_records) == 6, "Main table should contain 6 records"

    # Check specific records
    main_ids = [r[0] for r in main_table_records]
    assert set(main_ids) == {
        1,
        2,
        3,
        4,
        5,
        6,
    }, "Main table should contain expected IDs"

    # Original records shouldn't be changed
    assert main_table_records[0][0:2] == (
        1,
        1,
    ), "Existing record should not be modified"
    assert main_table_records[0][2] == "Sentence One", "Text should not change"

    # Check duplicates table
    assert len(duplicates_records) == 3, "Duplicates table should contain 3 records"

    dup_records = [(r[0], r[2]) for r in duplicates_records]
    assert (
        1,
        "New Sentence One (DB Conflict)",
    ) in dup_records, "DB conflict should be in duplicates"
    assert (
        3,
        "New Sentence Three (DB Conflict)",
    ) in dup_records, "DB conflict should be in duplicates"
    assert (
        5,
        "New Sentence Five B (Internal Duplicate)",
    ) in dup_records, "Internal duplicate should be logged"
