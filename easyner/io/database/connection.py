"""Database connection handling and management for DuckDB.

This module provides classes for managing connections to DuckDB databases,
including transaction management and query execution.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import duckdb


class IDatabaseConnection(ABC):
    """Interface for database connection management."""

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the database."""
        pass

    @abstractmethod
    def execute(
        self,
        query: str,
        parameters: Optional[list[Any]] = None,
    ) -> Any:
        """Execute a query on the database."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass


class DatabaseConnection(IDatabaseConnection):
    """Manages connections to a DuckDB database.

    This class is responsible for establishing, configuring and
    maintaining connections to a DuckDB database.
    """

    def __init__(
        self,
        db_path: Optional[str] = ":memory:",
        threads: int = 4,
        memory_limit: str = "1GB",
    ) -> None:
        """Initialize a database connection.

        Args:
            db_path: Path to the database file, or ":memory:" for in-memory database
            threads: Number of threads to use
            memory_limit: Memory limit for DuckDB

        """
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.threads = threads
        self.memory_limit = memory_limit
        self._connection: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> None:
        """Establish a connection to the database."""
        # If using a file database (not in-memory), ensure the directory exists
        if self.db_path != ":memory:":
            db_dir = os.path.dirname(self.db_path)
            if db_dir:  # Only try to create if there's a directory part
                os.makedirs(db_dir, exist_ok=True)

        try:
            self._connection = duckdb.connect(database=self.db_path)
            if self._connection is not None:
                self._connection.execute(f"PRAGMA threads={self.threads}")
                self._connection.execute(
                    f"PRAGMA memory_limit='{self.memory_limit}'",
                )
        except Exception as e:
            self.logger.error(f"Error connecting to DuckDB database: {e}")
            raise

    def close(self) -> None:
        """Close the database connection if it exists."""
        if self._connection:
            try:
                self._connection.close()
                self._connection = None
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}")
                raise

    def execute(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Execute a SQL query with parameters and return the result.

        Args:
            query: SQL query to execute
            parameters: Parameters for the query (optional)

        Returns:
            Query result

        """
        if self._connection is None:
            self.connect()
            if self._connection is None:
                raise RuntimeError("Failed to establish database connection")

        try:
            if parameters:
                return self._connection.execute(query, parameters)
            else:
                return self._connection.execute(query)
        except Exception as e:
            # Enhanced error logging with query details and stack trace
            import textwrap
            import traceback

            # Get the stack trace for better debugging
            stack_trace = traceback.format_stack()[
                :-1
            ]  # Exclude current frame

            # Truncate very long queries for readability in logs
            max_query_length = 1000
            query_for_log = (
                query
                if len(query) <= max_query_length
                else query[:max_query_length] + "... [truncated]"
            )

            # Format query with line breaks for better readability
            formatted_query = textwrap.indent(query_for_log, "    ")

            # Log detailed error information
            error_msg = f"Error executing query: {e}\n"
            error_msg += f"Query:\n{formatted_query}\n"

            error_msg += "Stack trace:\n" + "".join(stack_trace)

            self.logger.error(error_msg)
            raise

    def register(self, name: str, obj: Any) -> None:
        """Register an object (like DataFrame) with the connection.

        Args:
            name: Name to register the object under
            obj: Object to register (typically a DataFrame)

        """
        if self._connection is None:
            self.connect()
            if self._connection is None:
                raise RuntimeError("Failed to establish database connection")

        try:
            self._connection.register(name, obj)
        except Exception as e:
            self.logger.error(f"Error registering object: {e}")
            raise

    def unregister(self, name: str) -> None:
        """Unregister an object from the connection.

        Args:
            name: Name of the object to unregister

        """
        if self._connection is None:
            self.connect()
            if self._connection is None:
                raise RuntimeError("Failed to establish database connection")

        try:
            self._connection.unregister(name)
        except Exception as e:
            self.logger.error(f"Error unregistering object: {e}")
            raise

    def is_in_transaction(self) -> bool:
        """Check if the current connection is in a transaction.

        Returns:
            True if in a transaction, False otherwise

        """
        if self._connection is None:
            self.logger.warning(
                "Attempted to check transaction status with no active connection",
            )
            return False

        try:
            # For DuckDB, use a more reliable approach
            # Try to begin a transaction and see if it raises an exception about already being in one
            try:
                self._connection.execute("BEGIN TRANSACTION")
                # If we get here, we weren't in a transaction
                # But now we are, so roll it back
                self._connection.execute("ROLLBACK")
                return False
            except Exception as e:
                # If error message contains "already in a transaction", we're in a transaction
                if "transaction" in str(e).lower():
                    return True
                # For any other error, re-raise
                raise
        except Exception as e:
            self.logger.error(f"Error checking transaction status: {e}")
            raise

    def begin_transaction(self) -> None:
        """Begin a database transaction.

        This marks the beginning of a transaction block that can be
        committed or rolled back as a single unit of work.
        """
        if self._connection is None:
            self.connect()
            if self._connection is None:
                raise RuntimeError("Failed to establish database connection")

        try:
            self._connection.execute("BEGIN TRANSACTION")
            self.logger.debug("Transaction started")
        except Exception as e:
            self.logger.error(f"Error beginning transaction: {e}")
            raise

    def commit(self) -> None:
        """Commit the current transaction.

        This permanently applies all changes made within the current transaction.
        """
        if self._connection is None:
            self.logger.warning(
                "Attempted to commit with no active connection",
            )
            return

        try:
            self._connection.execute("COMMIT")
            self.logger.debug("Transaction committed")
        except Exception as e:
            self.logger.error(f"Error committing transaction: {e}")
            raise

    def rollback(self) -> None:
        """Rollback the current transaction.

        This discards all changes made within the current transaction.
        """
        if self._connection is None:
            self.logger.warning(
                "Attempted to rollback with no active connection",
            )
            return

        try:
            self._connection.execute("ROLLBACK")
            self.logger.debug("Transaction rolled back")
        except Exception as e:
            self.logger.error(f"Error rolling back transaction: {e}")
            raise

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get the underlying DuckDB connection object."""
        if self._connection is None:
            self.connect()
        return self._connection
