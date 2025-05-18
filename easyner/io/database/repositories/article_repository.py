from typing import Any, Optional, Union

import pandas as pd

from easyner.io.database.schemas import ARTICLES_TABLE_SQL
from easyner.io.database.schemas.python_mappings import (
    ARTICLE_ID,
    ARTICLES_TABLE,
    KEY_DUPLICATE,
    RETURNED_IDS_TABLE,
    TITLE,
)

from .base import Repository


class ArticleRepository(Repository):
    """Repository for managing article data in the database.

    Provides methods to retrieve, insert, and query article records.
    """

    @property
    def table_name(self) -> str:
        return ARTICLES_TABLE

    @property
    def table_sql_stmt(self) -> str:
        return ARTICLES_TABLE_SQL

    @property
    def required_columns(self) -> set[str]:
        return {ARTICLE_ID, TITLE}

    @property
    def primary_key_columns(self) -> list[str]:
        """Return the primary key of the articles table."""
        return [ARTICLE_ID]

    def _get_required_columns(self) -> set[str]:
        """Return the required columns for article insertion."""
        return {ARTICLE_ID, TITLE}

    def _build_insert_query(self, view_name: str) -> str:
        """Build SQL insert query for articles.

        Args:
            view_name: Name of the temporary view

        Returns:
            SQL query for article insertion

        """
        return f"INSERT INTO {ARTICLES_TABLE} ({ARTICLE_ID}, {TITLE}) SELECT {ARTICLE_ID}, {TITLE} FROM {view_name}"

    def get_all(
        self,
        as_df: bool = True,
    ) -> Union[pd.DataFrame, list[dict[str, Any]]]:
        """Get all articles from the database.

        Args:
            as_df: Return as DataFrame if True, otherwise as list of dicts (default: True)

        Returns:
            DataFrame or list of article dictionaries

        """
        if as_df:
            return self.get_all_df()
        else:
            return self.get_all_dict_list()

    def get_all_df(self) -> pd.DataFrame:
        """Get all articles from the database as a DataFrame.

        Returns:
            DataFrame containing article data

        """
        try:
            query = f"SELECT {ARTICLE_ID}, {TITLE} FROM {ARTICLES_TABLE}"
            return self.conn.execute(query).fetchdf()
        except Exception as e:
            self.logger.error(f"Error retrieving articles as DataFrame: {e}")
            raise

    def get_all_dict_list(self) -> list[dict[str, Any]]:
        """Get all articles from the database as a list of dictionaries.

        Returns:
            List of dictionaries containing article data

        """
        try:
            query = f"SELECT {ARTICLE_ID}, {TITLE} FROM {ARTICLES_TABLE}"
            rows = self.conn.execute(query).fetchall()
            return [{ARTICLE_ID: row[0], TITLE: row[1]} for row in rows]
        except Exception as e:
            self.logger.error(
                f"Error retrieving articles as dictionary list: {e}",
            )
            raise

    def get_by_id(self, article_id: int) -> Optional[dict[str, Any]]:
        """Get an article by its ID.

        Args:
            article_id: ID of the article to retrieve

        Returns:
            Article data as a dictionary or None if not found

        """
        try:
            result = self.conn.execute(
                f"SELECT {ARTICLE_ID}, {TITLE} FROM {ARTICLES_TABLE} WHERE {ARTICLE_ID} = ?",
                [article_id],
            )
            row = result.fetchone()
            if row:
                return {ARTICLE_ID: row[0], TITLE: row[1]}
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving article by ID: {e}")
            raise

    def insert(self, item: dict[str, Any]) -> None:
        """Insert an article into the database.

        Args:
            item: Article data as a dictionary with 'article_id' and 'title' keys

        """
        try:
            self.conn.execute(
                f"INSERT INTO {ARTICLES_TABLE} ({ARTICLE_ID}, {TITLE}) VALUES (?, ?)",
                [item[ARTICLE_ID], item[TITLE]],
            )
        except Exception as e:
            self.logger.error(f"Error inserting article: {e}")
            raise

    def _insert_sql_duplicate_handling(
        self,
        view_name: str,
        session_duplicates_table: str,
    ) -> None:
        """Insert view into db.

        Extract duplicates from the view and insert them into the session duplicates table.
        """
        # All article duplicates are key duplicates, no hierarchical duplicates
        # Since article_id is the primary key DuckDB will use a ART index to accelerate
        # This should be the fastest way to insert non-duplicates
        insert_non_duplicates_and_return_ids = f"""--sql
            -- Create a table to hold the IDs of the successfully inserted rows
            -- The column name in this new table will be whatever is aliased in RETURNING,
            -- or the original column name if no alias is used.
            CREATE TEMPORARY TABLE {RETURNED_IDS_TABLE} AS
            WITH InsertResult AS (
                INSERT INTO {ARTICLES_TABLE} BY NAME
                SELECT v.* FROM {view_name} v
                ON CONFLICT ({ARTICLE_ID}) DO NOTHING
                -- Return the ID(s) of the successfully inserted rows
                RETURNING {ARTICLE_ID} AS inserted_article_id
            )
            -- Select the results from the RETURNING clause (now in the CTE) into the new table
            SELECT * FROM InsertResult;
            """

        insert_duplicates = f"""--sql
            INSERT INTO {session_duplicates_table} BY NAME
            -- this is in memory
            -- used to find hierarchical duplicates of sentences and entities
            SELECT  v.*,  TRUE AS {KEY_DUPLICATE} -- all article duplicates are key duplicates
            FROM {view_name} v
            LEFT JOIN {RETURNED_IDS_TABLE} r ON v.{ARTICLE_ID} = r.inserted_article_id
            -- here we are a little clever, only joining the small inserted id set and
            -- relatively small view instead of joining against full articles table
            WHERE r.inserted_article_id IS NULL
        """
