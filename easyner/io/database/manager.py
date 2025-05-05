import logging
from typing import List, Optional
from pathlib import Path

from .connection import DatabaseConnection
from .schemas import (
    ARTICLES_TABLE_SQL,
    ENTITIES_TABLE_SQL,
    SENTENCES_TABLE_SQL,
)
from .utils.transaction import transactional


class TableManager:
    """
    Manages database tables for the EasyNer application.

    This class is responsible for creating and maintaining database tables,
    including tables for articles, sentences, entities, and their indices.
    """

    def __init__(
        self,
        connection: DatabaseConnection,
    ):
        """
        Initialize the TableManager.

        Args:
            connection: A database connection object
            sql_reader: A SQL file reader object (optional, for backward compatibility)
        """
        self.logger = logging.getLogger(__name__)
        self.connection = connection

    @transactional
    def create_base_tables(self) -> None:
        """
        Create all database tables using SQL statements.

        Creates tables for articles, sentences, and entities,
        as well as necessary sequences.

        All operations are executed within a transaction and rolled back if any operation fails.
        """
        try:
            # Execute the SQL statements in order (articles -> sentences -> entities)
            self.connection.execute(ARTICLES_TABLE_SQL)
            self.connection.execute(SENTENCES_TABLE_SQL)
            self.connection.execute(ENTITIES_TABLE_SQL)

            self.logger.info("Successfully created all database tables")
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    @transactional
    def create_indices(self) -> None:
        """
        Create database indices for performance optimization.

        Creates indices on commonly queried fields to improve query performance.

        All operations are executed within a transaction and rolled back if any operation fails.
        """
        try:
            # Create indices for performance
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_article_id ON articles(article_id)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sentence_article_id ON sentences(article_id)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_article_sentence ON entities(article_id, sentence_id)"
            )

            self.logger.info("Successfully created all database indices")
        except Exception as e:
            self.logger.error(f"Error creating indices: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if the table exists, False otherwise
        """
        result = self.connection.execute(
            f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
        )
        return result.fetchone()[0] > 0

    def get_table_count(self, table_name: str) -> int:
        """
        Get the count of rows in a table.

        Args:
            table_name: Name of the table to count rows in

        Returns:
            Number of rows in the table
        """
        result = self.connection.execute(f"SELECT COUNT(*) FROM {table_name}")
        return result.fetchone()[0]

    @transactional
    def export_to_csv(self, table_name: str, output_path: str) -> None:
        """
        Export a table to CSV file.

        Args:
            table_name: Name of the table to export
            output_path: Path to save the CSV file

        All operations are executed within a transaction and rolled back if any operation fails.
        """
        try:
            self.connection.execute(
                f"COPY (SELECT * FROM {table_name}) TO '{output_path}' (HEADER, DELIMITER ',');"
            )
            self.logger.info(
                f"Successfully exported {table_name} to {output_path}"
            )
        except Exception as e:
            self.logger.error(f"Error exporting table to CSV: {e}")
            raise
