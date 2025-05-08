from typing import Any, Union

import pandas as pd

from easyner.io.database.schemas import ENTITIES_TABLE_SQL
from easyner.io.database.utils.column_names import (
    ARTICLE_ID,
    END_CHAR,
    ENTITIES_TABLE,
    ENTITY_ID,
    INFERENCE_MODEL,
    INFERENCE_MODEL_METADATA,
    SENTENCE_ID,
    START_CHAR,
    TEXT,
)

from .base import Repository


class EntityRepository(Repository):
    """Repository for managing entity data in the database.

    Provides methods to retrieve, insert, and query entity records.
    """

    @property
    def table_name(self) -> str:
        """Return the name of the table this repository manages."""
        return ENTITIES_TABLE

    @property
    def table_sql_stmt(self) -> str:
        return ENTITIES_TABLE_SQL

    @property
    def columns(self) -> set[str]:
        """Return the set of columns in the entities table."""
        return {
            ENTITY_ID,
            ARTICLE_ID,
            SENTENCE_ID,
            TEXT,
            START_CHAR,
            END_CHAR,
            INFERENCE_MODEL,
            INFERENCE_MODEL_METADATA,
        }

    @property
    def required_columns(self) -> set[str]:
        """Return the set of required columns for insertion operations."""
        return {
            ARTICLE_ID,
            SENTENCE_ID,
            TEXT,
            START_CHAR,
            END_CHAR,
        }

    @property
    def primary_key_columns(self) -> list[str]:
        """Return the primary key of the entities table."""
        return [ENTITY_ID, ARTICLE_ID, SENTENCE_ID]

    def _build_insert_query(self, view_name: str) -> str:
        """Build SQL query for insertion from a temporary view.

        Args:
            view_name: Name of the temporary view

        Returns:
            SQL query string for insertion

        """
        cols = ", ".join(self.required_columns)
        return f"INSERT INTO {ENTITIES_TABLE} ({cols}) SELECT {cols} FROM {view_name}"

    def insert(self, item: dict[str, Any]) -> None:
        """Insert a single entity into the database.

        Args:
            item: Entity data dictionary with required fields

        """
        try:
            required_cols = self.required_columns
            placeholders = ", ".join(["?"] * len(required_cols))
            cols = ", ".join(required_cols)

            self.connection.execute(
                f"INSERT INTO {ENTITIES_TABLE} ({cols}) VALUES ({placeholders})",
                [item[col] for col in required_cols],
            )
        except Exception as e:
            self.logger.error(f"Error inserting entity: {e}")
            raise

    # Repository-specific methods
    def get_all(
        self,
        as_df: bool = True,
    ) -> Union[pd.DataFrame, list[dict[str, Any]]]:
        """Get all entities from the database."""
        if as_df:
            return self.get_all_df()
        else:
            return self.get_all_dict_list()

    def get_all_df(self) -> pd.DataFrame:
        """Get all entities as DataFrame."""
        try:
            cols = ", ".join(self.required_columns)
            result = self.connection.execute(
                f"SELECT {cols} FROM {ENTITIES_TABLE}",
            )
            return result.fetchdf()
        except Exception as e:
            self.logger.error(f"Error retrieving entities as DataFrame: {e}")
            raise

    def get_all_dict_list(self) -> list[dict[str, Any]]:
        """Get all entities as list of dictionaries."""
        try:
            cols = list(self.required_columns)
            result = self.connection.execute(
                f"SELECT {', '.join(cols)} FROM {ENTITIES_TABLE}",
            )
            rows = result.fetchall()

            return [
                {cols[i]: row[i] for i in range(len(cols))} for row in rows
            ]
        except Exception as e:
            self.logger.error(
                f"Error retrieving entities as dictionary list: {e}",
            )
            raise

    def get_by_sentence(
        self,
        article_id: int,
        sentence_id: int,
        as_df: bool = True,
    ) -> Union[pd.DataFrame, list[dict[str, Any]]]:
        """Get entities for a specific sentence."""
        try:
            cols = ", ".join(self.required_columns)
            query = f"""
                SELECT {cols} FROM {ENTITIES_TABLE}
                WHERE {ARTICLE_ID} = ? AND {SENTENCE_ID} = ?
            """
            result = self.connection.execute(query, [article_id, sentence_id])

            if as_df:
                return result.fetchdf()
            else:
                cols_list = list(self.required_columns)
                rows = result.fetchall()
                return [
                    {cols_list[i]: row[i] for i in range(len(cols_list))}
                    for row in rows
                ]
        except Exception as e:
            self.logger.error(f"Error retrieving entities by sentence: {e}")
            raise
