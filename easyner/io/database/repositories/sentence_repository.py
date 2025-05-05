import pandas as pd
from typing import Dict, List, Any, Union, Optional, Set

from easyner.io.database.utils.column_names import (
    ARTICLE_ID,
    SENTENCE_ID,
    TEXT,
    SENTENCES_TABLE,
)

from easyner.io.database.schemas import SENTENCES_TABLE_SQL

from .base import Repository
from ..connection import DatabaseConnection


class SentenceRepository(Repository):
    """
    Repository for managing sentence data in the database.
    Provides methods to retrieve, insert, and query sentence records.
    """

    @property
    def table_name(self) -> str:
        return SENTENCES_TABLE

    @property
    def table_sql_stmt(self) -> str:
        return SENTENCES_TABLE_SQL

    @property
    def required_columns(self) -> Set[str]:
        return {ARTICLE_ID, SENTENCE_ID, TEXT}

    def _build_insert_query(self, view_name: str) -> str:
        """
        Build SQL query for insertion from a temporary view.

        Args:
            view_name: Name of the temporary view

        Returns:
            SQL query string for insertion
        """
        return f"INSERT OR IGNORE INTO {SENTENCES_TABLE} ({ARTICLE_ID}, {SENTENCE_ID}, {TEXT}) SELECT {ARTICLE_ID}, {SENTENCE_ID}, {TEXT} FROM {view_name}"

    def insert(self, item: Dict[str, Any]) -> None:
        """
        Insert a single sentence into the database.

        Args:
            item: Sentence data dictionary with required fields
        """
        try:
            self.connection.execute(
                f"INSERT INTO {SENTENCES_TABLE} ({ARTICLE_ID}, {SENTENCE_ID}, {TEXT}) VALUES (?, ?, ?)",
                [item[ARTICLE_ID], item[SENTENCE_ID], item[TEXT]],
            )
        except Exception as e:
            self.logger.error(f"Error inserting sentence: {e}")
            raise

    # Repository-specific methods
    def get_all(
        self, as_df: bool = True
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Get all sentences from the database."""
        if as_df:
            return self.get_all_df()
        else:
            return self.get_all_dict_list()

    def get_all_df(self) -> pd.DataFrame:
        """Get all sentences as DataFrame."""
        try:
            result = self.connection.execute(
                f"SELECT {ARTICLE_ID}, {SENTENCE_ID}, {TEXT} FROM {SENTENCES_TABLE}"
            )
            return result.fetchdf()
        except Exception as e:
            self.logger.error(f"Error retrieving sentences as DataFrame: {e}")
            raise

    def get_all_dict_list(self) -> List[Dict[str, Any]]:
        """Get all sentences as list of dictionaries."""
        try:
            rows = self.connection.execute(
                f"SELECT {ARTICLE_ID}, {SENTENCE_ID}, {TEXT} FROM {SENTENCES_TABLE}"
            ).fetchall()
            return [
                {ARTICLE_ID: row[0], SENTENCE_ID: row[1], TEXT: row[2]}
                for row in rows
            ]
        except Exception as e:
            self.logger.error(
                f"Error retrieving sentences as dictionary list: {e}"
            )
            raise

    def get_by_article_id(
        self, article_id: int, as_df: bool = True
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """Get sentences for a specific article."""
        try:
            result = self.connection.execute(
                f"SELECT {ARTICLE_ID}, {SENTENCE_ID}, {TEXT} FROM {SENTENCES_TABLE} WHERE {ARTICLE_ID} = ?",
                [article_id],
            )

            if as_df:
                return result.fetchdf()
            else:
                rows = result.fetchall()
                return [
                    {ARTICLE_ID: row[0], SENTENCE_ID: row[1], TEXT: row[2]}
                    for row in rows
                ]
        except Exception as e:
            self.logger.error(f"Error retrieving sentences by article ID: {e}")
            raise

    def get_by_id(
        self, article_id: int, sentence_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get a specific sentence by its article ID and sentence ID."""
        try:
            result = self.connection.execute(
                f"SELECT {ARTICLE_ID}, {SENTENCE_ID}, {TEXT} FROM {SENTENCES_TABLE} WHERE {ARTICLE_ID} = ? AND {SENTENCE_ID} = ?",
                [article_id, sentence_id],
            )
            row = result.fetchone()
            if row:
                return {ARTICLE_ID: row[0], SENTENCE_ID: row[1], TEXT: row[2]}
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving sentence by IDs: {e}")
            raise

    def get_sentence_count_by_article(self) -> pd.DataFrame:
        """Get count of sentences per article."""
        try:
            result = self.connection.execute(
                f"SELECT {ARTICLE_ID}, COUNT(*) as sentence_count FROM {SENTENCES_TABLE} GROUP BY {ARTICLE_ID}"
            )
            return result.fetchdf()
        except Exception as e:
            self.logger.error(f"Error getting sentence count by article: {e}")
            raise
