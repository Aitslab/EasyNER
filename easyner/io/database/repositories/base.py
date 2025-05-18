"""Base repository interfaces for database access.

This module defines abstract base classes that serve as interfaces
for the concrete repository implementations.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Union

import duckdb
import pandas as pd

from easyner.io.database.schemas.python_mappings import (
    HIERARCHICAL_DUPLICATE,
    KEY_DUPLICATE,
)
from easyner.io.database.utils.transaction import transactional

from ..connection import DatabaseConnection


class Repository(ABC):
    """Abstract base class for all repositories.

    Provides common methods and defines the interface for repository implementations.
    """

    def __init__(self, conn: DatabaseConnection) -> None:
        """Initialize a repository with a database conn.

        Args:
            conn: Database conn object

        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conn = conn

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
    def primary_key_columns(self) -> list[str]:
        """The primary key of the table."""
        pass

    @property
    @abstractmethod
    def required_columns(self) -> set[str]:
        """The set of required columns for insertion operations."""
        pass

    @property
    def duplicate_table_name(self) -> str:
        """Return the name of the repository's duplicates table."""
        return f"{self.table_name}_duplicates"

    def _create_table(self) -> None:
        """Create the table if it does not exist."""
        self.conn.execute(self.table_sql_stmt)

    def _create_duplicates_table(self) -> None:
        """Create a duplicates table for the repository if not exists.

        This is a complete copy of the original table with a different name
        and an additional column for the timestamp of when the duplicate was detected.
        """
        # TODO: Early exit if duplicates table already exists

        # replace table_name with duplicate_table_name in the SQL statement
        create_table_sql = self.table_sql_stmt.replace(
            self.table_name,
            self.duplicate_table_name,
        )

        try:
            self.conn.execute(create_table_sql)
            # Add timestamp column for when the duplicate was detected
            # self.conn.execute(
            #     f"""
            #     ALTER TABLE {self.duplicate_table_name}
            #     ADD COLUMN IF NOT EXISTS duplicate_detection_timestamp TIMESTAMP DEFAULT NOW()
            # """,
            # )
            self.conn.execute(
                f"""
                ALTER TABLE {self.duplicate_table_name}
                ADD COLUMN IF NOT EXISTS {HIERARCHICAL_DUPLICATE} BOOLEAN DEFAULT FALSE
            """,
            )
            self.conn.execute(
                f"""
                ALTER TABLE {self.duplicate_table_name}
                ADD COLUMN IF NOT EXISTS {KEY_DUPLICATE} BOOLEAN DEFAULT FALSE
            """,
            )

            self.logger.info(
                f"Created duplicates table: {self.duplicate_table_name}",
            )
        except Exception as e:
            self.logger.error(
                f"Error creating duplicates table {self.duplicate_table_name}: {e}",
            )
            raise
            raise

    @abstractmethod
    def _build_insert_query(self, view_name: str) -> str:
        """Build SQL query for insertion from a temporary view.

        Args:
            view_name: Name of the temporary view

        Returns:
            SQL query string for insertion

        """
        pass

    @abstractmethod
    def insert(self, item: dict[str, Any]) -> None:
        """Insert a single item into the database.

        Args:
            item: Data dictionary with required fields

        """
        pass

    def _execute_insert_many(
        self,
        items: Union[list[dict[str, Any]], pd.DataFrame],
        log_duplicates: bool = False,
        ignore_duplicates: bool = False,
    ) -> None:
        """Core logic to insert multiple items. Not transactional by itself.

        Args:
            items: List of dictionaries or DataFrame containing data

        Raises:
            TypeError: If items is not a list of dictionaries or DataFrame
            ValueError: If items format is invalid or missing required columns

        """
        # Validate input type
        if not isinstance(items, (list, pd.DataFrame)):
            self.logger.error(
                f"Invalid type for {self.table_name}: "
                "Expected list of dictionaries or DataFrame.",
            )
            msg = "Items must be a list of dictionaries or a pandas DataFrame."
            raise TypeError(
                msg,
            )

        # Convert to DataFrame if needed
        if isinstance(items, list):
            if not items:  # Handle empty list
                return
            # Ensure all elements are dictionaries
            if not all(isinstance(item, dict) for item in items):
                self.logger.error(
                    f"Invalid format for {self.table_name} list: "
                    "Expected list of dictionaries.",
                )
                msg = "All items in list must be dictionaries."
                raise ValueError(msg)
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
                f"DataFrame is missing required columns for {self.table_name}: "
                f"{missing_cols}",
            )
            msg = f"DataFrame is missing columns: {missing_cols}"
            raise ValueError(msg)

        # Perform insertion, unique view name to avoid conflicts
        view_name = f"temp_{self.table_name}_df_{id(df)}_{str(uuid.uuid4())[:8]}"

        try:
            # Register view with only the columns we need
            # ToDo this should only be done if set to True since
            # most of the time we want to insert all columns
            self._register_view(view_name, df, required_cols)
            # Execute the repository-specific SQL query
            if log_duplicates:
                self._insert_log_duplicates_to_duplicates_table(view_name)
            else:
                query = self._build_insert_query(view_name)
                if ignore_duplicates:
                    # Replace INSERT with INSERT OR IGNORE in sql query
                    query = query.replace("INSERT", "INSERT OR IGNORE")

                self.conn.execute(query)

        except duckdb.ConstraintException as e:
            # Handle constraint violation
            # When running withh log_duplicates_to_duplicates_table=True
            # this exception instead gets caughht in the
            # _insert_log_duplicates_to_duplicates_table
            # method and is not raised here
            self.logger.error(
                f"Constraint violation while inserting into {self.table_name}: {e}",
            )
            raise

        except Exception as e:
            self.logger.error(f"Error batch inserting {self.table_name}: {e}")
            raise
        finally:
            self.conn.unregister(view_name)  # Always clean up

    def _register_view(
        self,
        view_name: str,
        df: pd.DataFrame,
        columns: set[str],
    ) -> None:
        """Register a temporary view with specific columns.

        Args:
            view_name: Name for the temporary view
            df: DataFrame to register
            columns: Set of column names to include

        """
        self.conn.register(view_name, df[list(columns)])

    @transactional
    def insert_many_transactional(
        self,
        items: Union[list[dict[str, Any]], pd.DataFrame],
        log_duplicates: bool = False,
        ignore_duplicates: bool = False,
    ) -> None:
        """Transactional wrapper for insert many.

        Use this for standalone batch insertions.
        """
        self.insert_many_non_transactional(
            items,
            log_duplicates,
            ignore_duplicates,
        )

    def insert_many_non_transactional(
        self,
        items: Union[list[dict[str, Any]], pd.DataFrame],
        log_duplicates: bool = False,
        ignore_duplicates: bool = False,
    ) -> None:
        """Insert multiple items without transaction management.

        Use within externally managed transactions that catch
        exceptions and rollback if needed.

        Default behavior is INSERT INTO, which will raise an error
        on duplicate entries.
        If log_duplicates is set to True, it will log duplicates
        to the duplicates table instead of raising an error.
        If ignore_duplicates is set to True, it will silently
        ignore duplicate entries. This is not compatible with log_duplicates.

        Args:
            items: List of dictionaries or DataFrame containing data
            log_duplicates: Flag to log duplicates to duplicates table
                (default: False)
            ignore_duplicates: Flag to ignore duplicate entries
                (default: False)

        Raises:
            Exception: If there is an error during insertion

        """
        # We can't both log and ignore duplicates
        if log_duplicates and ignore_duplicates:
            msg = (
                "Log duplicates is not compatible with ignore duplicates. "
                "Please choose whether to log or ignore duplicates. "
                "If not logging to table, use ignore duplicates to either "
                "raise errors on duplicate insertions or silently ignore them."
            )
            raise ValueError(
                msg,
            )
        try:
            self._execute_insert_many(
                items,
                log_duplicates,
                ignore_duplicates,
            )
        except Exception as e:
            self.logger.error(
                f"Error batch inserting {self.table_name} "
                f"within an existing transaction: {e}",
            )

    def _insert_log_duplicates_to_duplicates_table(
        self,
        view_name: str,
    ) -> None:
        """Insert records from view_name, log duplicates to separate table.

        Handling duplicates by logging them to a duplicates table.

        This method:
        1. Identifies records in the view that conflict with existing table data
        2. Logs these conflicts to the duplicates table
        3. Identifies any internal duplicates within the view itself
        4. Logs these internal duplicates (keeping the first occurrence)
        5. Inserts the remaining unique records into the main table

        # TODO: Improve duplicate handling logic so that if a parent object
        # is duplicated, all child objects are also moveed to the duplicates table
        # e.g. if a article is duplicate, we should not add it's sentences to the main
        # table, but rather move them to the duplicates table as well but mark wheter,
        # they themself are duplicates or not
        # An alternative strategy would be to simply skip them and analyze them later
        # in the source file.
        # For now this is not an issue as we only really importing articles with the
        # log to  duplicates table option

        Args:
            view_name: Name of the temporary view containing records to insert

        Raises:
            Exception: If there is an error during the duplicate handling process

        """
        # Create the duplicates table if it doesn't exist
        self._create_duplicates_table()

        # Need to use self.duplicate_table_name which is set
        # in create_duplicates_table()
        duplicates_table = self.duplicate_table_name

        try:
            # Step 1: Identify records in view that conflict with existing table data
            self.logger.info(
                "Identifying conflicts between view and existing "
                f"{self.table_name} records",
            )

            # This query identifies records in the view that have matching primary keys
            # in the main table
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
            self.conn.execute(
                f"CREATE TEMPORARY VIEW {conflicts_view_name} AS {conflicts_query}",
            )

            # Step 2: Log these conflicts to the duplicates table
            # Add timestamp column for when the duplicate was detected
            self.conn.execute(
                f"""
                INSERT INTO {duplicates_table} ({self._build_columns_list()},
                                             duplicate_detection_timestamp)
                SELECT {self._build_columns_list()}, NOW()
                FROM {conflicts_view_name}
                """,
            )

            # Count and log how many conflicts were found
            conflict_count_result = self.conn.execute(
                f"SELECT COUNT(*) FROM {conflicts_view_name}",
            ).fetchone()
            conflict_count = conflict_count_result[0] if conflict_count_result else 0

            self.logger.info(
                f"Found {conflict_count} records in view that conflict with "
                f"existing {self.table_name}",
            )

            # Step 3 & 4: Handle internal duplicates
            # (duplicates within the batch itself)
            # This creates a view of unique records from the input that don't conflict
            # with the DB
            non_conflicts_view = f"{view_name}_non_conflicts"
            self.conn.execute(
                f"""
                CREATE TEMPORARY VIEW {non_conflicts_view} AS
                SELECT v.* FROM {view_name} v
                WHERE NOT EXISTS (
                    SELECT 1 FROM {conflicts_view_name} c
                    WHERE {self._build_pk_match_condition('c', 'v')}
                )
            """,
            )

            # Identify internal duplicates (keeping the first occurrence of each PK)
            internal_dups_view = f"{view_name}_internal_dups"
            self.conn.execute(
                f"""
                CREATE TEMPORARY VIEW {internal_dups_view} AS
                SELECT * FROM (
                    SELECT *,
                        ROW_NUMBER() OVER(PARTITION
                            BY {self._build_pk_columns_list()}
                            ORDER BY (SELECT 1)) as rn
                    FROM {non_conflicts_view}
                ) t
                WHERE rn > 1
            """,
            )

            # Log internal duplicates to duplicates table
            self.conn.execute(
                f"""
                INSERT INTO {duplicates_table} ({self._build_columns_list()},
                                             duplicate_detection_timestamp)
                SELECT {self._build_columns_list()}, NOW()
                FROM {internal_dups_view}
            """,
            )

            # Count internal duplicates
            internal_dup_count_result = self.conn.execute(
                f"SELECT COUNT(*) FROM {internal_dups_view}",
            ).fetchone()
            internal_dup_count = (
                internal_dup_count_result[0] if internal_dup_count_result else 0
            )
            self.logger.info(
                f"Found {internal_dup_count} internal duplicate records in batch",
            )

            # Step 5: Insert the remaining unique records into the main table
            unique_records_view = f"{view_name}_unique"
            self.conn.execute(
                f"""
                CREATE TEMPORARY VIEW {unique_records_view} AS
                SELECT * FROM (
                    SELECT *,
                        ROW_NUMBER() OVER(PARTITION
                            BY {self._build_pk_columns_list()}
                            ORDER BY (SELECT 1)) as rn
                    FROM {non_conflicts_view}
                ) t
                WHERE rn = 1
            """,
            )

            # Build columns list for the insert (excluding any temp columns like rn)
            insert_query = f"""
                INSERT INTO {self.table_name} ({self._build_columns_list()})
                SELECT {self._build_columns_list()}
                FROM {unique_records_view}
            """
            self.conn.execute(insert_query)

            # Count records actually inserted
            inserted_count_result = self.conn.execute(
                f"SELECT COUNT(*) FROM {unique_records_view}",
            ).fetchone()
            inserted_count = inserted_count_result[0] if inserted_count_result else 0

            self.logger.info(
                f"Successfully inserted {inserted_count} unique records "
                f"into {self.table_name}",
            )

        except Exception as e:
            self.logger.error(
                f"Error in _insert_log_duplicates_to_duplicates_table: {e}",
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
                    self.conn.execute(f"DROP VIEW IF EXISTS {temp_view}")
                except Exception as e:
                    self.logger.warning(
                        f"Error dropping temp view {temp_view}: {e}",
                    )

    def _build_pk_match_condition(
        self,
        table1_alias: str,
        table2_alias: str,
    ) -> str:
        """Build SQL condition using the primary key columns property."""
        conditions = [
            f"{table1_alias}.{col} = {table2_alias}.{col}"
            for col in self.primary_key_columns
        ]
        return " AND ".join(conditions)

    def _build_pk_columns_list(self) -> str:
        """Build a comma-separated list of primary key column names for the table.

        Returns:
            String of primary key column names to use in SQL queries

        """
        # Default implementation uses all columns from primary_key_columns
        # Repository subclasses might want to override this
        # if they need specific column ordering
        return ", ".join(self.primary_key_columns)

    def _build_columns_list(self) -> str:
        """Build a comma-separated list of column names for the table.

        Returns:
            String of column names to use in SQL queries

        """
        # Default implementation uses all columns from required_columns
        # Repository subclasses might want to override this if they
        # need specific column ordering
        return ", ".join(self.required_columns)

    def _get_columns(self) -> None:
        """Placeholder. Return the set of columns in the table."""
        pass

    def insert_new(
        self,
        items: pd.DataFrame,
        log_duplicates: bool = False,
        ignore_duplicates: bool = False,
    ) -> None:
        """Insert new items into the database.

        Args:
            items: DataFrame containing data
            log_duplicates: Flag to log duplicates to duplicates table
                (default: False)
            ignore_duplicates: Flag to ignore duplicate entries
                (default: False)

        """
        self._create_table()

        self._create_duplicates_table()

        session_duplicates_table = f"{self.duplicate_table_name}_session"

        # Make empty copy of duplicates table
        self.conn.execute(
            f"""
            CREATE TEMPORARY TABLE {session_duplicates_table} AS
            SELECT * FROM {self.duplicate_table_name}
            WHERE 1=0
        """,
        )

        view_name = f"temp_{self.table_name}_df_{id(items)}_{str(uuid.uuid4())[:8]}"

        try:
            self._register_view(view_name, items, self.required_columns)

            self._insert_sql_duplicate_handling(
                self,
                view_name,
                session_duplicates_table,
            )
        except Exception as e:
            self.logger.error(
                f"Error inserting new items into {self.table_name}: {e}",
            )
            raise

        # self._insert_with_duplicate_handling(session_duplicates_table, items)

    # @abstractmethod
    # def _insert_with_duplicate_handling(
    #     self,
    #     session_duplicates_table: str,
    #     items: pd.DataFrame,
    # ) -> None:
    #     pass

    @abstractmethod
    def _insert_sql_duplicate_handling(
        self,
        view_name: str,
        session_duplicates_table: str,
    ) -> None:
        """Insert records with SQL duplicate handling.

        Args:
            view_name: Name of the temporary view
            session_duplicates_table: Name of the session duplicates table

        """
        pass
