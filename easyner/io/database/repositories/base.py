"""
Base repository interfaces for database access.

This module defines abstract base classes that serve as interfaces
for the concrete repository implementations.
"""

from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Dict, List, Any, Union, Set

from easyner.io.database.utils.transaction import transactional
from ..connection import DatabaseConnection


class Repository(ABC):
    """
    Abstract base class for all repositories.
    Provides common methods and defines the interface for repository implementations.
    """

    def __init__(self, connection: DatabaseConnection):
        """
        Initialize a repository with a database connection.

        Args:
            connection: Database connection object
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connection = connection

    @property
    @abstractmethod
    def table_name(self) -> str:
        """The name of the table this repository manages."""
        pass

    @property
    @abstractmethod
    def table_sql_stmt(self) -> str:
        """The SQL statement to create the table."""
        pass

    @property
    @abstractmethod
    def required_columns(self) -> Set[str]:
        """The set of required columns for insertion operations."""
        pass

    def _create_table(self) -> None:
        """Create the table if it does not exist."""
        self.connection.execute(self.table_sql_stmt)

    @abstractmethod
    def _build_insert_query(self, view_name: str) -> str:
        """
        Build SQL query for insertion from a temporary view.

        Args:
            view_name: Name of the temporary view

        Returns:
            SQL query string for insertion
        """
        pass

    @abstractmethod
    def insert(self, item: Dict[str, Any]) -> None:
        """
        Insert a single item into the database.

        Args:
            item: Data dictionary with required fields
        """
        pass

    def _execute_insert_many(
        self, items: Union[List[Dict[str, Any]], pd.DataFrame]
    ) -> None:
        """
        Core logic to insert multiple items. Not transactional by itself.

        Args:
            items: List of dictionaries or DataFrame containing data

        Raises:
            TypeError: If items is not a list of dictionaries or DataFrame
            ValueError: If items format is invalid or missing required columns
        """
        # Validate input type
        if not isinstance(items, (list, pd.DataFrame)):
            self.logger.error(
                f"Invalid type for {self.table_name}: Expected list of dictionaries or DataFrame."
            )
            raise TypeError(
                f"Items must be a list of dictionaries or a pandas DataFrame."
            )

        # Convert to DataFrame if needed
        if isinstance(items, list):
            if not items:  # Handle empty list
                return
            # Ensure all elements are dictionaries
            if not all(isinstance(item, dict) for item in items):
                self.logger.error(
                    f"Invalid format for {self.table_name} list: Expected list of dictionaries."
                )
                raise ValueError(f"All items in list must be dictionaries.")
            df = pd.DataFrame(items)
        else:  # isinstance(items, pd.DataFrame)
            df = items

        if df.empty:
            return

        # Validate required columns
        required_cols = self.required_columns
        if not required_cols.issubset(df.columns):
            missing_cols = required_cols - set(df.columns)
            self.logger.error(
                f"DataFrame is missing required columns for {self.table_name}: {missing_cols}"
            )
            raise ValueError(f"DataFrame is missing columns: {missing_cols}")

        # Perform insertion
        view_name = f"temp_{self.table_name}_df_{id(df)}"

        try:
            # Register view with only the columns we need
            self._register_view(view_name, df, required_cols)
            # Execute the repository-specific SQL query
            self.connection.execute(self._build_insert_query(view_name))
        finally:
            self.connection.unregister(view_name)  # Always clean up

    def _register_view(
        self, view_name: str, df: pd.DataFrame, columns: Set[str]
    ) -> None:
        """
        Register a temporary view with specific columns.

        Args:
            view_name: Name for the temporary view
            df: DataFrame to register
            columns: Set of column names to include
        """
        self.connection.register(view_name, df[list(columns)])

    @transactional
    def insert_many_transactional(
        self, items: Union[List[Dict[str, Any]], pd.DataFrame]
    ) -> None:
        """
        Insert multiple items with transaction support.
        Use this for standalone batch insertions.

        Args:
            items: List of dictionaries or DataFrame containing data

        Raises:
            Exception: If there is an error during insertion
        """
        self.insert_many_non_transactional(items)

    def insert_many_non_transactional(
        self, items: Union[List[Dict[str, Any]], pd.DataFrame]
    ) -> None:
        """
        Insert multiple items without transaction management.
        Use within externally managed transactions.

        Args:
            items: List of dictionaries or DataFrame containing data

        Raises:
            Exception: If there is an error during insertion
        """
        try:
            self._execute_insert_many(items)
        except Exception as e:
            self.logger.error(
                f"Error batch inserting {self.table_name} within an existing transaction: {e}"
            )
            raise
