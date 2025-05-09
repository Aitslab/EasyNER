"""Service for importing and exporting data across multiple repositories."""

import logging
from typing import Any, Union

import pandas as pd

from easyner.io.database.connection import DatabaseConnection
from easyner.io.database.repositories import (
    ArticleRepository,
    EntityRepository,
    SentenceRepository,
)
from easyner.io.database.utils.transaction import transactional


class DataExchanger:
    """Orchestrates data exchange between multiple repositories.

    This service handles operations that span multiple repositories,
    such as importing complete articles with their sentences and entities.
    It maintains referential integrity and provides hierarchical duplicate handling.
    """

    def __init__(
        self,
        connection: DatabaseConnection,
    ) -> None:
        """Initialize with database connection and repositories.

        Args:
            connection: Database connection to use
            article_repo: ArticleRepository instance (created if None)
            sentence_repo: SentenceRepository instance (created if None)
            entity_repo: EntityRepository instance (created if None)

        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conn = connection

    @transactional
    def import_article_with_sentences_and_entities(
        self,
        articles: Union[pd.DataFrame, list[dict[str, Any]]],
        sentences: Union[pd.DataFrame, list[dict[str, Any]]],
        entities: Union[pd.DataFrame, list[dict[str, Any]]],
    ) -> None:
        """Import complete article data with duplicate handling.

        All operations are performed in a single transaction to ensure data integrity.
        Everything except passing flags is handled within the database for performance
        Uses a single connection across the script and temporary tables to track
        duplicates found in session, which is used to
        find hierarchical duplicates.

        This means session duplicates will be stored in memory for performance
        If problems with OOM airse batch data in smaller chunks, or use
        SET temp_directory to a location with more space allowing db to off-load to disk

        Handles both hierarchical and pure duplicates

        This method:
        1. Imports the article(s), tracking any duplicates in a session table.
        2. Imports sentences, using session article duplicates to find hierarchical sentence duplicates. Tracks sentence duplicates in a new session table.
        3. Imports entities, using session sentence duplicates to find hierarchical entity duplicates.
        4. All operations occur within a single transaction.

        Args:
            article_data: Article data dictionary or list of dictionaries.
            sentences_data: List of sentence dictionaries.
            entities_data: List of entity dictionaries.

        """
        # Early failure if no data is provided
        if not articles and not sentences and not entities:
            msg = "No data provided for import."
            self.logger.error(msg)
            raise ValueError(msg)

        # Convert to DataFrames if needed
        articles_df = (
            articles
            if isinstance(articles, pd.DataFrame)
            else pd.DataFrame(articles) if articles else pd.DataFrame()
        )
        sentences_df = (
            sentences
            if isinstance(sentences, pd.DataFrame)
            else pd.DataFrame(sentences) if sentences else pd.DataFrame()
        )
        entities_df = (
            entities
            if isinstance(entities, pd.DataFrame)
            else pd.DataFrame(entities) if entities else pd.DataFrame()
        )

        summary_input_stat_df = pd.DataFrame(
            {
                "type": ["articles", "sentences", "entities"],
                "count": [
                    len(articles_df),
                    len(sentences_df),
                    len(entities_df),
                ],
            },
        )

        msg = f"Importing \n{summary_input_stat_df.to_markdown()}"
        self.logger.debug(msg)

        self._process_data(
            articles_df=articles_df,
            sentences_df=sentences_df,
            entities_df=entities_df,
        )

    @transactional
    def _process_data(
        self,
        articles_df: pd.DataFrame,
        sentences_df: pd.DataFrame,
        entities_df: pd.DataFrame,
    ) -> None:

        try:
            if articles_df:
                session_duplicate_article_repo = ArticleRepository(
                    conn=self.conn,
                ).insert_new(articles_df)

            if sentences_df:
                session_duplicate_sentence_repo = SentenceRepository(
                    conn=self.conn,
                ).insert_new(sentences_df)

            if entities_df:
                session_duplicate_entity_repo = EntityRepository(
                    conn=self.conn,
                ).insert_new(entities_df)

        except Exception as e:
            msg = f"Error during data processing: {e}"
            self.logger.error(msg)
            raise
        finally:
            self._cleanup_temp_tables()

    def _cleanup_temp_tables(self) -> None:
        pass
