"""
Base repository interfaces for database access.

This module defines abstract base classes that serve as interfaces
for the concrete repository implementations.
"""

from abc import ABC, abstractmethod
import duckdb
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
    def primary_key_columns(self) -> List[str]:
        """The primary key of the table."""
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
        self,
        items: Union[List[Dict[str, Any]], pd.DataFrame],
        log_duplicates_to_duplicates_table: bool = False,
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
            # ToDo this should only be done if set to True since
            # most of the time we want to insert all columns
            self._register_view(view_name, df, required_cols)
            # Execute the repository-specific SQL query
            if log_duplicates_to_duplicates_table:
                self._insert_log_duplicates_to_duplicates_table(view_name)
            else:
                self.connection.execute(self._build_insert_query(view_name))
        except duckdb.ConstraintException as e:
            # Handle constraint violation
            # When running withh log_duplicates_to_duplicates_table=True
            # this exception instead gets caughht in the
            # _insert_log_duplicates_to_duplicates_table
            # method and is not raised here
            self.logger.error(
                f"Constraint violation while inserting into {self.table_name}: {e}"
            )
            raise

        except Exception as e:
            self.logger.error(f"Error batch inserting {self.table_name}: {e}")
            raise
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
        self,
        items: Union[List[Dict[str, Any]], pd.DataFrame],
        log_duplicates_to_duplicates_table: bool = False,
    ) -> None:
        """
        Insert multiple items without transaction management.
        Use within externally managed transactions.

        Args:
            items: List of dictionaries or DataFrame containing data
            log_duplicates_to_duplicates_table: Flag to log duplicates to duplicates table
                (default: False)
        Raises:
            Exception: If there is an error during insertion
        """
        try:
            self._execute_insert_many(
                items, log_duplicates_to_duplicates_table
            )
        except Exception as e:
            self.logger.error(
                f"Error batch inserting {self.table_name} within an existing transaction: {e}"
            )
            raise

    def create_duplicates_table(self) -> None:
        """
        Create a duplicates table for the repository if not exists.
        This is a complete copy of the original table with a different name.
        """
        duplicate_table_name = f"{self.table_name}_duplicates"
        # replace table_name with duplicate_table_name in the SQL statement
        create_table_sql = self.table_sql_stmt.replace(
            self.table_name, duplicate_table_name
        )
        try:
            self.connection.execute(create_table_sql)
            # Add timestamp column for when the duplicate was detected
            self.connection.execute(
                f"""
                ALTER TABLE {duplicate_table_name}
                ADD COLUMN duplicate_detection_timestamp TIMESTAMP DEFAULT NOW()
            """
            )
            self.duplicate_table_name = duplicate_table_name
            self.logger.info(
                f"Created duplicates table: {duplicate_table_name}"
            )
        except Exception as e:
            self.logger.error(
                f"Error creating duplicates table {duplicate_table_name}: {e}"
            )
            raise

    def _insert_log_duplicates_to_duplicates_table(
        self, view_name: str
    ) -> None:
        """
        Insert records from view_name, handling duplicates by logging them to a duplicates table.

        This method:
        1. Identifies records in the view that conflict with existing table data
        2. Logs these conflicts to the duplicates table
        3. Identifies any internal duplicates within the view itself
        4. Logs these internal duplicates (keeping the first occurrence)
        5. Inserts the remaining unique records into the main table

        Args:
            view_name: Name of the temporary view containing records to insert

        Raises:
            Exception: If there is an error during the duplicate handling process
        """
        # Create the duplicates table if it doesn't exist
        self.create_duplicates_table()

        # Need to use self.duplicate_table_name which is set in create_duplicates_table()
        duplicates_table = self.duplicate_table_name

        try:
            # Step 1: Identify records in view that conflict with existing table data
            self.logger.info(
                f"Identifying conflicts between view and existing {self.table_name} records"
            )

            # This query identifies records in the view that have matching primary keys in the main table
            conflicts_query = f"""
                WITH view_keys AS (
                    SELECT * FROM {view_name}
                )
                SELECT v.*
                FROM view_keys v
                WHERE EXISTS (
                    SELECT 1
                    FROM {self.table_name} t
                    WHERE {self._build_pk_match_condition('t', 'v')}
                )
            """

            # Execute query to find conflicts and save as temporary view
            conflicts_view_name = f"{view_name}_conflicts"
            self.connection.execute(
                f"CREATE TEMPORARY VIEW {conflicts_view_name} AS {conflicts_query}"
            )

            # Step 2: Log these conflicts to the duplicates table
            # Add timestamp column for when the duplicate was detected
            self.connection.execute(
                f"""
                INSERT INTO {duplicates_table} ({self._build_columns_list()}, duplicate_detection_timestamp)
                SELECT {self._build_columns_list()}, NOW()
                FROM {conflicts_view_name}
                """
            )

            # Count and log how many conflicts were found
            conflict_count = self.connection.execute(
                f"SELECT COUNT(*) FROM {conflicts_view_name}"
            ).fetchone()[0]

            self.logger.info(
                f"Found {conflict_count} records in view that conflict with existing {self.table_name}"
            )

            # Step 3 & 4: Handle internal duplicates (duplicates within the batch itself)
            # This creates a view of unique records from the input that don't conflict with the DB
            non_conflicts_view = f"{view_name}_non_conflicts"
            self.connection.execute(
                f"""
                CREATE TEMPORARY VIEW {non_conflicts_view} AS
                SELECT v.* FROM {view_name} v
                WHERE NOT EXISTS (
                    SELECT 1 FROM {conflicts_view_name} c
                    WHERE {self._build_pk_match_condition('c', 'v')}
                )
            """
            )

            # Identify internal duplicates (keeping the first occurrence of each PK)
            internal_dups_view = f"{view_name}_internal_dups"
            self.connection.execute(
                f"""
                CREATE TEMPORARY VIEW {internal_dups_view} AS
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER(PARTITION BY {self._build_pk_columns_list()} ORDER BY (SELECT 1)) as rn
                    FROM {non_conflicts_view}
                ) t
                WHERE rn > 1
            """
            )

            # Log internal duplicates to duplicates table
            self.connection.execute(
                f"""
                INSERT INTO {duplicates_table} ({self._build_columns_list()}, duplicate_detection_timestamp)
                SELECT {self._build_columns_list()}, NOW()
                FROM {internal_dups_view}
            """
            )

            # Count internal duplicates
            internal_dup_count = self.connection.execute(
                f"SELECT COUNT(*) FROM {internal_dups_view}"
            ).fetchone()[0]

            self.logger.info(
                f"Found {internal_dup_count} internal duplicate records in batch"
            )

            # Step 5: Insert the remaining unique records into the main table
            unique_records_view = f"{view_name}_unique"
            self.connection.execute(
                f"""
                CREATE TEMPORARY VIEW {unique_records_view} AS
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER(PARTITION BY {self._build_pk_columns_list()} ORDER BY (SELECT 1)) as rn
                    FROM {non_conflicts_view}
                ) t
                WHERE rn = 1
            """
            )

            # Build columns list for the insert (excluding any temp columns like rn)
            insert_query = f"""
                INSERT INTO {self.table_name} ({self._build_columns_list()})
                SELECT {self._build_columns_list()}
                FROM {unique_records_view}
            """
            self.connection.execute(insert_query)

            # Count records actually inserted
            inserted_count = self.connection.execute(
                f"SELECT COUNT(*) FROM {unique_records_view}"
            ).fetchone()[0]

            self.logger.info(
                f"Successfully inserted {inserted_count} unique records into {self.table_name}"
            )

        except Exception as e:
            self.logger.error(
                f"Error in _insert_log_duplicates_to_duplicates_table: {e}"
            )
            raise

        finally:
            # Clean up temporary views
            for temp_view in [
                f"{view_name}_conflicts",
                f"{view_name}_non_conflicts",
                f"{view_name}_internal_dups",
                f"{view_name}_unique",
            ]:
                try:
                    self.connection.execute(f"DROP VIEW IF EXISTS {temp_view}")
                except Exception as e:
                    self.logger.warning(
                        f"Error dropping temp view {temp_view}: {e}"
                    )

    def _build_pk_match_condition(
        self, table1_alias: str, table2_alias: str
    ) -> str:
        """Builds SQL condition using the primary key columns property."""
        conditions = []
        for col in self.primary_key_columns:
            conditions.append(f"{table1_alias}.{col} = {table2_alias}.{col}")
        return " AND ".join(conditions)

    def _build_pk_columns_list(self) -> str:
        """
        Build a comma-separated list of primary key column names for the table.

        Returns:
            String of primary key column names to use in SQL queries
        """
        # Default implementation uses all columns from primary_key_columns
        # Repository subclasses might want to override this if they need specific column ordering
        return ", ".join(self.primary_key_columns)

    def _build_columns_list(self) -> str:
        """
        Build a comma-separated list of column names for the table.

        Returns:
            String of column names to use in SQL queries
        """
        # Default implementation uses all columns from required_columns
        # Repository subclasses might want to override this if they need specific column ordering
        return ", ".join(self.required_columns)

    def _get_columns(self) -> None:
        """Placeholder. Return the set of columns in the table."""

        pass
