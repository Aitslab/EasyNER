from typing import Any, Optional, Union

import pandas as pd

from easyner.io.database.schemas import ARTICLES_TABLE_SQL
from easyner.io.database.utils.column_names import (
    ARTICLE_ID,
    ARTICLES_TABLE,
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
            return self.connection.execute(query).fetchdf()
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
            rows = self.connection.execute(query).fetchall()
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
            result = self.connection.execute(
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
            self.connection.execute(
                f"INSERT INTO {ARTICLES_TABLE} ({ARTICLE_ID}, {TITLE}) VALUES (?, ?)",
                [item[ARTICLE_ID], item[TITLE]],
            )
        except Exception as e:
            self.logger.error(f"Error inserting article: {e}")
            raise
