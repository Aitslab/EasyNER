"""Hierarchical data insertion manager for article, sentence, and entity data.

This module provides functionality to insert hierarchical data into a database,
handling relationships between articles, sentences, and entities.

Separate DataFrames (articles, sentences, entities) are used for insertion speed
Therefore since the data is hierarchical, we need to ensure that:

There are two types of duplicates:
1. Only-Hierarchical duplicates (i.e. by association)
2. Non-hierarchical duplicates (i.e. by value, can also be hierarchical at the same time)

Hierarchical duplicates are not necessarily duplicates in the database sense
i.e. by value, but they are duplicates in the context of being added as part
of a duplicate parent and should therefore not be added to the main database

Expected behavior is cascading:
Steps:
Insert  Articles:
if duplicates ->
     articles_duplicates table += article_duplicates
     articles table += article_non_duplicates
        any sentence or entity within the insert session with the same article_id
        is added to respective duplicates table as they are hierarchical duplicates
Insert Sentences:
if duplicate articles_id inserted in same session ->
     sentences_duplicates table += hierarchical_sentence_duplicates
if sentence_value_duplicates_remaining > 0:
     sentences_duplicates table += sentence_value_duplicates
sentences table += sentence_non_duplicates

Insert Entities:
if duplicate sentences_id inserted in same session ->
     entities_duplicates table += hierarchical_entity_duplicates
if entity_value_duplicates_remaining > 0:
     entities_duplicates table += entity_value_duplicates
sentences table += sentence_non_duplicates

"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from easyner.io.database.conn import DatabaseConnection

logger = logging.getLogger(__name__)


@dataclass
class InsertionCounts:
    """Holds counts of inserted and duplicate records."""

    inserted: int
    duplicates: int

    def __add__(self, other: "InsertionCounts") -> "InsertionCounts":
        """Add two InsertionCounts objects together."""
        return InsertionCounts(
            inserted=self.inserted + other.inserted,
            duplicates=self.duplicates + other.duplicates,
        )

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for backwards compatibility."""
        return {"inserted": self.inserted, "duplicates": self.duplicates}


class HierarchicalDataManager:
    """Manages hierarchical relationships between articles, sentences, and entities during insertion."""

    def __init__(self, conn: DatabaseConnection) -> None:
        """Initialize the hierarchical data manager.

        Args:
            conn: Database conn to use for operations

        """
        self.conn = conn
        self.logger = logging.getLogger(__name__)

    def insert_hierarchical_data(
        self,
        articles: Union[pd.DataFrame, list[dict[str, Any]]],
        sentences: Union[pd.DataFrame, list[dict[str, Any]]],
        entities: Union[pd.DataFrame, list[dict[str, Any]]],
        log_duplicates: bool = True,
        ignore_duplicates: bool = False,
        use_optimized: bool = True,  # Added parameter to control optimization
        transaction_active: bool = False,
    ) -> dict[str, int]:
        """Insert hierarchical data with proper parent-child relationship handling.

        This method will:
        1. Insert non-duplicate articles into the articles table
        2. Move duplicate articles to the articles_duplicates table if log_duplicates=True
        3. Only insert sentences that belong to non-duplicate articles
        4. Only insert entities that belong to non-duplicate sentences

        Args:
            articles: DataFrame or list of dictionaries with article data
            sentences: DataFrame or list of dictionaries with sentence data
            entities: DataFrame or list of dictionaries with entity data
            log_duplicates: Whether to log duplicates to _duplicates tables
            ignore_duplicates: Whether to silently ignore duplicates
            use_optimized: Whether to use the optimized CTE-based approach (recommended for large datasets)

        Returns:
            Dictionary with counts of inserted and duplicate records

        """
        if log_duplicates and ignore_duplicates:
            msg = "Cannot both log and ignore duplicates"
            raise ValueError(msg)

        # Convert to DataFrame if needed
        articles_df = (
            articles
            if isinstance(articles, pd.DataFrame)
            else pd.DataFrame(articles) if articles else pd.DataFrame()
        )
        sentences_df = (
            sentences
            if isinstance(sentences, pd.DataFrame)
            else pd.DataFrame(sentences) if sentences else pd.DataFrame()
        )
        entities_df = (
            entities
            if isinstance(entities, pd.DataFrame)
            else pd.DataFrame(entities) if entities else pd.DataFrame()
        )

        # Skip empty datasets
        if articles_df.empty and sentences_df.empty and entities_df.empty:
            return InsertionCounts(inserted=0, duplicates=0).to_dict()

        # Generate unique view names
        articles_view = f"temp_articles_{id(articles_df)}"
        sentences_view = f"temp_sentences_{id(sentences_df)}"
        entities_view = f"temp_entities_{id(entities_df)}"

        try:
            # Register views for all data
            if not articles_df.empty:
                self.conn.register(articles_view, articles_df)
            if not sentences_df.empty:
                self.conn.register(sentences_view, sentences_df)
            if not entities_df.empty:
                self.conn.register(entities_view, entities_df)

            # Check if we're already in a transaction
            transaction_started_here = False

            if not self.conn.is_in_transaction():
                # Start a transaction if not already in one
                self.conn.begin_transaction()
                transaction_started_here = True

            try:
                if use_optimized:
                    # Use the optimized CTE-based approach
                    results = self._execute_hierarchical_insert_optimized(
                        articles_view=(
                            articles_view if not articles_df.empty else None
                        ),
                        sentences_view=(
                            sentences_view if not sentences_df.empty else None
                        ),
                        entities_view=(
                            entities_view if not entities_df.empty else None
                        ),
                        log_duplicates=log_duplicates,
                    )
                else:
                    # Use the original approach
                    results = self._execute_hierarchical_insert(
                        articles_view=(
                            articles_view if not articles_df.empty else None
                        ),
                        sentences_view=(
                            sentences_view if not sentences_df.empty else None
                        ),
                        entities_view=(
                            entities_view if not entities_df.empty else None
                        ),
                        log_duplicates=log_duplicates,
                        ignore_duplicates=ignore_duplicates,
                    )

                # Only commit if we started the transaction
                if transaction_started_here:
                    self.conn.commit()

                return results.to_dict()
            except Exception:
                # Only rollback if we started the transaction
                if transaction_started_here:
                    self.conn.rollback()
                raise

        finally:
            # Clean up views
            for view in [articles_view, sentences_view, entities_view]:
                try:
                    self.conn.unregister(view)
                except Exception as e:
                    self.logger.warning(
                        f"Error unregistering view {view}: {e}",
                    )

    def _execute_hierarchical_insert_optimized(
        self,
        articles_view: Optional[str] = None,
        sentences_view: Optional[str] = None,
        entities_view: Optional[str] = None,
        log_duplicates: bool = True,
    ) -> InsertionCounts:
        """Execute hierarchical insert using optimized CTE approach.

        This method uses Common Table Expressions (CTEs) to optimize the insertion process by:
        1. Identifying duplicates once using a CTE
        2. Logging duplicates directly from that CTE (if enabled)
        3. Inserting non-duplicates in the same query
        4. Returning counts in a single operation
        5. Avoiding creating temporary tables

        This approach is significantly faster for large datasets.
        """
        total_counts = InsertionCounts(inserted=0, duplicates=0)

        # Create duplicates tables if they don't exist
        if log_duplicates:
            self._ensure_duplicates_tables_exist()

        # Step 1: Handle articles
        if articles_view:
            article_counts = self._handle_articles_optimized(
                articles_view,
                log_duplicates,
            )
            total_counts = total_counts + article_counts

        # Step 2: Handle sentences
        if sentences_view:
            sentence_counts = self._handle_sentences_optimized(
                sentences_view,
                log_duplicates,
            )
            total_counts = total_counts + sentence_counts

        # Step 3: Handle entities
        if entities_view:
            entity_counts = self._handle_entities_optimized(
                entities_view,
                log_duplicates,
            )
            total_counts = total_counts + entity_counts

        return total_counts

    def _handle_articles_optimized(
        self,
        articles_view: str,
        log_duplicates: bool,
    ) -> InsertionCounts:
        """Handle article insertion using optimized CTE approach."""
        # Start with the duplicates CTE
        query = f"""--sql
        WITH duplicates AS (
            SELECT v.article_id
            FROM {articles_view} v
            JOIN articles t ON t.article_id = v.article_id
        )"""

        # Add duplicate logging if needed
        if log_duplicates:
            query += f"""
        , logged_duplicates AS (
            INSERT INTO articles_duplicates
            SELECT v.*, NOW() as duplicate_detection_timestamp
            FROM {articles_view} v
            JOIN duplicates d ON v.article_id = d.article_id
            RETURNING v.article_id
        )"""

        # Always include the comma for proper CTE syntax
        query += f"""
        , inserted AS (
            INSERT INTO articles
            SELECT v.* FROM {articles_view} v
            WHERE NOT EXISTS (SELECT 1 FROM duplicates d WHERE d.article_id = v.article_id)
            RETURNING v.article_id
        )
        SELECT
            (SELECT COUNT(*) FROM inserted) as inserted_count,
            (SELECT COUNT(*) FROM duplicates) as duplicates_count
        """

        result = self.conn.execute(query).fetchone()
        return InsertionCounts(inserted=result[0], duplicates=result[1])

    def _handle_sentences_optimized(
        self,
        sentences_view: str,
        log_duplicates: bool,
    ) -> InsertionCounts:
        """Handle sentence insertion using optimized CTE approach."""
        query = f"""--sql
        WITH sentence_duplicates AS (
            SELECT v.sentence_id, v.article_id
            FROM {sentences_view} v
            WHERE EXISTS (
                SELECT 1 FROM articles_duplicates d WHERE d.article_id = v.article_id
            )
            OR EXISTS (
                SELECT 1 FROM sentences s
                WHERE s.article_id = v.article_id AND s.sentence_id = v.sentence_id
            )
        )"""

        # Add duplicate logging if needed
        if log_duplicates:
            query += f"""--sql
        , logged_duplicates AS (
            INSERT INTO sentences_duplicates
            SELECT v.*, NOW() as duplicate_detection_timestamp
            FROM {sentences_view} v
            JOIN sentence_duplicates d ON v.sentence_id = d.sentence_id AND v.article_id = d.article_id
            RETURNING v.sentence_id
        )"""

        # Always include the comma for proper CTE syntax
        query += f"""--sql
        , inserted AS (
            INSERT INTO sentences
            SELECT v.* FROM {sentences_view} v
            WHERE NOT EXISTS (
                SELECT 1 FROM sentence_duplicates d
                WHERE d.sentence_id = v.sentence_id AND d.article_id = v.article_id
            )
            RETURNING v.sentence_id
        )
        SELECT
            (SELECT COUNT(*) FROM inserted) as inserted_count,
            (SELECT COUNT(*) FROM sentence_duplicates) as duplicates_count
        """

        result = self.conn.execute(query).fetchone()
        return InsertionCounts(inserted=result[0], duplicates=result[1])

    def _handle_entities_optimized(
        self,
        entities_view: str,
        log_duplicates: bool,
    ) -> InsertionCounts:
        """Handle entity insertion using optimized CTE approach."""
        query = f"""--sql
        WITH entity_duplicates AS (
            SELECT v.entity_id
            FROM {entities_view} v
            WHERE
                -- Only check for duplicate based on natural business key
                EXISTS (
                    SELECT 1 FROM entities e
                    WHERE e.article_id = v.article_id
                      AND e.sentence_id = v.sentence_id
                      -- Position checks first as they're likely more selective for entity detection
                      AND e.start_char = v.start_char
                      AND e.end_char = v.end_char
                      AND e.text = v.text
                )
        )"""

        # Add duplicate logging if needed
        if log_duplicates:
            query += f"""
        , logged_duplicates AS (
            INSERT INTO entities_duplicates
            SELECT v.*, NOW() as duplicate_detection_timestamp
            FROM {entities_view} v
            JOIN entity_duplicates d ON v.entity_id = d.entity_id
            RETURNING v.entity_id
        )"""

        # Always include the comma for proper CTE syntax
        query += f"""
        , inserted AS (
            INSERT INTO entities (
                article_id, sentence_id, text, start_char, end_char,
                inference_model, inference_model_metadata
            )
            SELECT
                v.article_id, v.sentence_id, v.text, v.start_char, v.end_char,
                v.inference_model, v.inference_model_metadata
            FROM {entities_view} v
            WHERE NOT EXISTS (SELECT 1 FROM entity_duplicates d WHERE d.entity_id = v.entity_id)
            RETURNING entity_id
        )
        SELECT
            (SELECT COUNT(*) FROM inserted) as inserted_count,
            (SELECT COUNT(*) FROM entity_duplicates) as duplicates_count
        """

        result = self.conn.execute(query).fetchone()
        return InsertionCounts(inserted=result[0], duplicates=result[1])

    def _execute_hierarchical_insert(
        self,
        articles_view: Optional[str] = None,
        sentences_view: Optional[str] = None,
        entities_view: Optional[str] = None,
        log_duplicates: bool = True,
        ignore_duplicates: bool = False,
    ) -> InsertionCounts:
        """Execute the hierarchical insert with proper duplicate handling."""
        total_counts = InsertionCounts(inserted=0, duplicates=0)

        # Create duplicates tables if they don't exist
        if log_duplicates:
            self._ensure_duplicates_tables_exist()

        # Step 1: Handle articles
        if articles_view:
            article_counts = self._handle_articles(
                articles_view,
                log_duplicates,
            )
            total_counts = total_counts + article_counts

        # Step 2: Handle sentences - only insert those belonging to non-duplicate articles
        if sentences_view:
            sentence_counts = self._handle_sentences(
                sentences_view,
                log_duplicates,
            )
            total_counts = total_counts + sentence_counts

        # Step 3: Handle entities - only insert those belonging to non-duplicate sentences
        if entities_view:
            entity_counts = self._handle_entities(
                entities_view,
                log_duplicates,
            )
            total_counts = total_counts + entity_counts

        # Clean up temporary tables
        for table in [
            "temp_duplicate_articles",
            "temp_duplicate_sentences",
            "temp_duplicate_entities",
        ]:
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")

        return total_counts

    def _handle_articles(
        self,
        articles_view: str,
        log_duplicates: bool,
    ) -> InsertionCounts:
        """Handle article insertion and duplicate detection."""
        # Identify duplicates
        self._identify_duplicates(
            "articles",
            articles_view,
            "article_id",
            "temp_duplicate_articles",
        )

        # Count duplicates
        duplicates_count = self._count_duplicates("temp_duplicate_articles")

        # Log duplicates if needed
        if log_duplicates and duplicates_count > 0:
            self._log_duplicates(
                "articles_duplicates",
                articles_view,
                "article_id",
                "temp_duplicate_articles",
            )

        # Insert non-duplicates
        self._insert_non_duplicates(
            "articles",
            articles_view,
            "article_id",
            "temp_duplicate_articles",
        )

        # Count inserted
        inserted_count = self._count_non_duplicates(
            articles_view,
            "article_id",
            "temp_duplicate_articles",
        )

        return InsertionCounts(
            inserted=inserted_count,
            duplicates=duplicates_count,
        )

    def _handle_sentences(
        self,
        sentences_view: str,
        log_duplicates: bool,
    ) -> InsertionCounts:
        """Handle sentence insertion and duplicate detection."""
        # Identify duplicates (including those with duplicate article parents)
        self._identify_sentence_duplicates(sentences_view)

        # Count duplicates
        duplicates_count = self._count_duplicates("temp_duplicate_sentences")

        # Log duplicates if needed
        if log_duplicates and duplicates_count > 0:
            self._log_duplicates(
                "sentences_duplicates",
                sentences_view,
                "sentence_id",
                "temp_duplicate_sentences",
                "article_id",  # Add secondary key column
            )

        # Insert non-duplicates
        self._insert_non_duplicates(
            "sentences",
            sentences_view,
            "sentence_id",
            "temp_duplicate_sentences",
            "article_id",  # Add secondary key column
        )

        # Count inserted
        inserted_count = self._count_non_duplicates(
            sentences_view,
            "sentence_id",
            "temp_duplicate_sentences",
            "article_id",  # Add secondary key column
        )

        return InsertionCounts(
            inserted=inserted_count,
            duplicates=duplicates_count,
        )

    def _handle_entities(
        self,
        entities_view: str,
        log_duplicates: bool,
    ) -> InsertionCounts:
        """Handle entity insertion and duplicate detection."""
        # Identify duplicates (including those with duplicate sentence parents)
        self._identify_entity_duplicates(entities_view)

        # Count duplicates
        duplicates_count = self._count_duplicates("temp_duplicate_entities")

        # Log duplicates if needed
        if log_duplicates and duplicates_count > 0:
            self._log_duplicates(
                "entities_duplicates",
                entities_view,
                "entity_id",
                "temp_duplicate_entities",
            )

        # Insert non-duplicates
        self._insert_non_duplicates(
            "entities",
            entities_view,
            "entity_id",
            "temp_duplicate_entities",
        )

        # Count inserted
        inserted_count = self._count_non_duplicates(
            entities_view,
            "entity_id",
            "temp_duplicate_entities",
        )

        return InsertionCounts(
            inserted=inserted_count,
            duplicates=duplicates_count,
        )

    def _identify_duplicates(
        self,
        table: str,
        view: str,
        id_col: str,
        temp_table: str,
    ) -> None:
        """Identify duplicates by comparing with existing records."""
        self.conn.execute(
            f"""
            CREATE TEMPORARY TABLE {temp_table} AS
            SELECT v.{id_col}
            FROM {view} v
            WHERE EXISTS (
                SELECT 1 FROM {table} t WHERE t.{id_col} = v.{id_col}
            )
        """,
        )

    def _identify_sentence_duplicates(self, sentences_view: str) -> None:
        """Identify duplicate sentences or those with duplicate article parents.

        Sentences have a composite primary key (article_id, sentence_id), so we need to
        check for duplicates using both fields.
        """
        self.conn.execute(
            f"""
            CREATE TEMPORARY TABLE temp_duplicate_sentences AS
            SELECT s.sentence_id, s.article_id
            FROM {sentences_view} s
            WHERE EXISTS (
                SELECT 1 FROM temp_duplicate_articles d
                WHERE d.article_id = s.article_id
            )
            OR EXISTS (
                SELECT 1 FROM sentences m
                WHERE m.article_id = s.article_id AND m.sentence_id = s.sentence_id
            )
        """,
        )

    def _identify_entity_duplicates(self, entities_view: str) -> None:
        """Identify duplicate entities or those with duplicate sentence parents."""
        self.conn.execute(
            f"""
            CREATE TEMPORARY TABLE temp_duplicate_entities AS
            SELECT e.entity_id
            FROM {entities_view} e
            WHERE EXISTS (
                SELECT 1 FROM temp_duplicate_sentences d
                WHERE d.sentence_id = e.sentence_id
            )
            OR EXISTS (
                SELECT 1 FROM entities m
                WHERE m.entity_id = e.entity_id
            )
        """,
        )

    def _count_duplicates(self, temp_table: str) -> int:
        """Count records in a temporary duplicates table."""
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {temp_table}",
        ).fetchone()[0]

    def _log_duplicates(
        self,
        duplicates_table: str,
        view: str,
        id_col: str,
        temp_table: str,
        secondary_id_col: Optional[str] = None,
    ) -> None:
        """Insert duplicates into the appropriate duplicates table."""
        if secondary_id_col:
            # Composite key version
            self.conn.execute(
                f"""
                INSERT INTO {duplicates_table}
                SELECT v.*, NOW() as duplicate_detection_timestamp
                FROM {view} v
                WHERE EXISTS (
                    SELECT 1 FROM {temp_table} d
                    WHERE d.{id_col} = v.{id_col} AND d.{secondary_id_col} = v.{secondary_id_col}
                )
            """,
            )
        else:
            # Original version for single column keys
            self.conn.execute(
                f"""
                INSERT INTO {duplicates_table}
                SELECT v.*, NOW() as duplicate_detection_timestamp
                FROM {view} v
                WHERE EXISTS (
                    SELECT 1 FROM {temp_table} d
                    WHERE d.{id_col} = v.{id_col}
                )
            """,
            )

    def _insert_non_duplicates(
        self,
        table: str,
        view: str,
        id_col: str,
        temp_table: str,
        secondary_id_col: Optional[str] = None,
    ) -> None:
        """Insert non-duplicate records into the target table."""
        if secondary_id_col:
            # Composite key version
            self.conn.execute(
                f"""
                INSERT INTO {table}
                SELECT v.* FROM {view} v
                WHERE NOT EXISTS (
                    SELECT 1 FROM {temp_table} d
                    WHERE d.{id_col} = v.{id_col} AND d.{secondary_id_col} = v.{secondary_id_col}
                )
            """,
            )
        else:
            # Original version for single column keys
            self.conn.execute(
                f"""
                INSERT INTO {table}
                SELECT v.* FROM {view} v
                WHERE NOT EXISTS (
                    SELECT 1 FROM {temp_table} d
                    WHERE d.{id_col} = v.{id_col}
                )
            """,
            )

    def _count_non_duplicates(
        self,
        view: str,
        id_col: str,
        temp_table: str,
        secondary_id_col: Optional[str] = None,
    ) -> int:
        """Count non-duplicate records."""
        if secondary_id_col:
            # Composite key version
            return self.conn.execute(
                f"""
                SELECT COUNT(*) FROM {view} v
                WHERE NOT EXISTS (
                    SELECT 1 FROM {temp_table} d
                    WHERE d.{id_col} = v.{id_col} AND d.{secondary_id_col} = v.{secondary_id_col}
                )
            """,
            ).fetchone()[0]
        else:
            # Original version for single column keys
            return self.conn.execute(
                f"""
                SELECT COUNT(*) FROM {view} v
                WHERE NOT EXISTS (
                    SELECT 1 FROM {temp_table} d
                    WHERE d.{id_col} = v.{id_col}
                )
            """,
            ).fetchone()[0]

    def _ensure_duplicates_tables_exist(self) -> None:
        """Make sure all needed duplicates tables exist."""
        # Reuse code from Repository.create_duplicates_table but apply to all tables
        for table in ["articles", "sentences", "entities"]:
            duplicate_table = f"{table}_duplicates"

            # Check if table exists
            table_exists = (
                self.conn.execute(
                    f"""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_name = '{duplicate_table}'
            """,
                ).fetchone()[0]
                > 0
            )

            if not table_exists:
                # Get original table schema
                columns_info = self.conn.execute(
                    f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                """,
                ).fetchall()

                if not columns_info:
                    msg = f"Table {table} does not exist"
                    raise ValueError(msg)

                # Construct create table statement
                columns_sql = ", ".join(
                    [
                        f"{col[0]} {col[1]} {'NULL' if col[2] == 'YES' else 'NOT NULL'}"
                        for col in columns_info
                    ],
                )

                self.conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {duplicate_table} (
                        {columns_sql},
                        duplicate_detection_timestamp TIMESTAMP DEFAULT NOW()
                    )
                """,
                )

                self.logger.info(
                    f"Created duplicates table: {duplicate_table}",
                )
