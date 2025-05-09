import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import pandas as pd

from easyner.io.handlers.base import IOHandler

from .connection import DatabaseConnection
from .manager import TableManager
from .repositories import (
    ArticleRepository,
    EntityRepository,
    SentenceRepository,
)
from .utils.transaction import transactional


class DuckDBHandler:
    """DuckDBHandler is a class that provides methods to read and write data to and from DuckDB databases.

    This class serves as a facade for the database subsystem, coordinating access to various
    repository classes that handle specific database operations.
    """

    def __init__(
        self,
        db_path: Optional[str] = ":memory:",
        threads: int = 4,
        memory_limit: str = "4GB",
    ):
        """Initialize the DuckDB handler.

        Args:
            db_path: Path to the database file, or ":memory:" for in-memory database
            threads: Number of threads to use
            memory_limit: Memory limit for DuckDB
            encoding: Encoding to use for file operations

        """
        self.logger = logging.getLogger(__name__)

        # Create component instances using dependency injection
        self.conn = DatabaseConnection(db_path, threads, memory_limit)
        self.conn.connect()

        # No need to specify SQL directory since schemas are imported directly
        self.table_manager = TableManager(self.conn)

        # Initialize repositories
        self.article_repository = ArticleRepository(self.conn)
        self.sentence_repository = SentenceRepository(self.conn)
        self.entity_repository = EntityRepository(self.conn)

    def read(self, file_path: str, **kwargs):
        """Read data from DuckDB database.

        Args:
            file_path: Path to the database file
            **kwargs: Additional arguments for reading
                - query: SQL query to execute
                - as_df: Whether to return as DataFrame (default: True)

        Returns:
            The result of the query

        """
        query = kwargs.get("query")
        if not query:
            msg = "Query parameter is required for reading from database"
            raise ValueError(
                msg,
            )

        # Create a temporary connection if a different DB path is provided
        if file_path != self.conn.db_path:
            temp_conn = DatabaseConnection(file_path)
            temp_conn.connect()
            result = temp_conn.execute(query)
            temp_conn.close()
        else:
            result = self.conn.execute(query)

        # Return as DataFrame by default
        as_df = kwargs.get("as_df", True)
        return result.fetchdf() if as_df else result.fetchall()

    @transactional
    def write(self, data, file_path: str, **kwargs) -> None:
        """Write data to DuckDB database.

        Args:
            data: Data to write (can be DataFrame or list of dictionaries)
            file_path: Path to the database file
            **kwargs: Additional arguments for writing
                - table_name: Name of the table to write to
                - if_exists: What to do if table exists ('fail', 'replace', 'append')

        Note:
            This operation is wrapped in a transaction to ensure data integrity.
            If the write operation fails, the database will remain unchanged.

        """
        table_name = kwargs.get("table_name")
        if not table_name:
            msg = "table_name parameter is required for writing to database"
            raise ValueError(
                msg,
            )

        if_exists = kwargs.get("if_exists", "fail")

        # Create a temporary conn if a different DB path is provided
        if file_path != self.conn.db_path:
            temp_conn = DatabaseConnection(file_path)
            temp_conn.connect()
            try:
                # Begin transaction manually since we're not using self.conn
                temp_conn.begin_transaction()
                self._write_data(temp_conn, data, table_name, if_exists)
                temp_conn.commit()
            except Exception as e:
                temp_conn.rollback()
                self.logger.error(f"Error writing data to {file_path}: {e}")
                raise
            finally:
                temp_conn.close()
        else:
            self._write_data(self.conn, data, table_name, if_exists)

    def _write_data(
        self,
        conn: DatabaseConnection,
        data,
        table_name: str,
        if_exists: str,
    ):
        """Helper method to write data to a database connection.

        Args:
            conn: Database connection to write to
            data: Data to write
            table_name: Name of the table to write to
            if_exists: What to do if table exists ('fail', 'replace', 'append')

        """
        if isinstance(data, pd.DataFrame):
            # Register DataFrame as a view
            conn.register(f"{table_name}_temp", data)

            if if_exists == "replace":
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_temp",
                )
            elif if_exists == "append":
                conn.execute(
                    f"INSERT INTO {table_name} SELECT * FROM {table_name}_temp",
                )
            else:  # 'fail'
                query = f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
                table_exists = conn.execute(query).fetchone()[0]
                if table_exists:
                    msg = f"Table {table_name} already exists"
                    raise ValueError(msg)
                conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_temp",
                )
        else:
            # Convert to DataFrame if necessary
            df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
            self._write_data(conn, df, table_name, if_exists)

    def create_base_tables(self) -> None:
        """Create all database tables using SQL files.

        This method delegates to the TableManager to create
        the necessary tables in the database.
        """
        self.table_manager.create_base_tables()

    def create_indices(self) -> None:
        """Create database indices for performance optimization.

        This method delegates to the TableManager to create
        performance indices in the database.
        """
        self.table_manager.create_indices()

    def get_table(
        self,
        table_name: str,
        as_df: bool = True,
    ) -> Union[pd.DataFrame, list[tuple]]:
        """Fetch data from a table, either as DataFrame or list of tuples.

        Args:
            table_name: Name of the table to fetch
            as_df: Return as DataFrame if True, otherwise as list of tuples

        Returns:
            DataFrame or list of tuples containing table data

        """
        result = self.conn.execute(f"SELECT * FROM {table_name}")
        return result.fetchdf() if as_df else result.fetchall()

    def get_table_count(self, table_name: str) -> int:
        """Get the count of rows in a table.

        Args:
            table_name: Name of the table to count rows in

        Returns:
            Number of rows in the table

        """
        return self.table_manager.get_table_count(table_name)

    def export_to_csv(
        self,
        table_name: str,
        output_path: Union[str, Path],
    ) -> None:
        """Export a table to CSV file.

        Args:
            table_name: Name of the table to export
            output_path: Path to save the CSV file

        """
        self.table_manager.export_to_csv(table_name, str(output_path))

    def get_entities_by_article(self, article_id: int) -> pd.DataFrame:
        """Get all entities for a specific article.

        Args:
            article_id: ID of the article

        Returns:
            DataFrame containing entities for the specified article

        """
        return self.entity_repository.get_by_article_id(article_id)

    # Legacy methods for backward compatibility

    def get_articles_df(self) -> pd.DataFrame:
        """Get all articles from the database as a DataFrame.

        Returns:
            DataFrame containing article data

        """
        return self.article_repository.get_all_df()

    def get_articles_as_dict_list(self) -> list[dict[str, Any]]:
        """Get all articles from the database as a list of dictionaries.

        Returns:
            List of dictionaries containing article data

        """
        return self.article_repository.get_all_dict_list()

    def get_articles(
        self,
        as_df: bool = True,
    ) -> Union[pd.DataFrame, list[dict[str, Any]]]:
        """Get all articles from the database.
        This is a wrapper method for backward compatibility.

        Args:
            as_df: Return as DataFrame if True, otherwise as list of dicts (default: True)

        Returns:
            DataFrame or list of article dictionaries

        """
        return self.article_repository.get_all(as_df=as_df)

    def get_sentences_df(self) -> pd.DataFrame:
        """Get all sentences from the database as a DataFrame.

        Returns:
            DataFrame containing sentence data

        """
        return self.sentence_repository.get_all_df()

    def get_sentences_as_dict_list(self) -> list[dict[str, Any]]:
        """Get all sentences from the database as a list of dictionaries.

        Returns:
            List of dictionaries containing sentence data

        """
        return self.sentence_repository.get_all_dict_list()

    def get_sentences(
        self,
        as_df: bool = True,
    ) -> Union[pd.DataFrame, list[dict[str, Any]]]:
        """Get all sentences from the database.
        This is a wrapper method for backward compatibility.

        Args:
            as_df: Return as DataFrame if True, otherwise as list of dicts (default: True)

        Returns:
            DataFrame or list of sentence dictionaries

        """
        return self.sentence_repository.get_all(as_df=as_df)

    def get_entities_df(self) -> pd.DataFrame:
        """Get all entities from the database as a DataFrame.

        Returns:
            DataFrame containing entity data

        """
        return self.entity_repository.get_all_df()

    def get_entities_as_dict_list(self) -> list[dict[str, Any]]:
        """Get all entities from the database as a list of dictionaries.

        Returns:
            List of dictionaries containing entity data

        """
        return self.entity_repository.get_all_dict_list()

    def get_entities(
        self,
        as_df: bool = True,
    ) -> Union[pd.DataFrame, list[dict[str, Any]]]:
        """Get all entities from the database.
        This is a wrapper method for backward compatibility.

        Args:
            as_df: Return as DataFrame if True, otherwise as list of dicts (default: True)

        Returns:
            DataFrame or list of entity dictionaries

        """
        return self.entity_repository.get_all(as_df=as_df)

    def get_conversion_log_df(self) -> pd.DataFrame:
        """Get the conversion log from the database as a DataFrame.

        Returns:
            DataFrame containing conversion log data, or an empty DataFrame if the log is empty or table doesn't exist.

        """
        try:
            # Check if the table exists to prevent errors if it hasn't been created yet
            # (though create_base_tables should handle this in normal flow)
            query_exists = "SELECT count(*) FROM information_schema.tables WHERE table_name = 'conversion_log'"
            table_exists = self.conn.execute(query_exists).fetchone()[0]
            if not table_exists:
                self.logger.warning(
                    "Conversion log table does not exist. Returning empty DataFrame.",
                )
                return pd.DataFrame()

            return self.conn.execute(
                "SELECT * FROM conversion_log",
            ).fetchdf()
        except Exception as e:
            self.logger.error(f"Error fetching conversion log: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
