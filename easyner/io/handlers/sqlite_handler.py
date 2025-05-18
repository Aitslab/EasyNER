import json
import sqlite3
from typing import Dict, List, Optional, Union
from .base import IOHandler
import logging

logger = logging.getLogger(__name__)


class SQLiteHandler(IOHandler):
    """Handles reading and writing data to SQLite database for relational NER storage."""

    EXTENSION = "sqlite"

    def __init__(self, encoding: Optional[str] = None):
        super().__init__(encoding)
        self.tables_created = False

    def _get_article_sentences(self, cursor: sqlite3.Cursor, article_id):
        cursor.execute(
            "SELECT * FROM sentences WHERE article_id = ?",
            (article_id,),
        )
        return cursor.fetchall()

    def _get_sentence_entities(self, cursor: sqlite3.Cursor, sentence_id):
        cursor.execute(
            "SELECT * FROM entities WHERE sentence_id = ?",
            (sentence_id,),
        )
        return cursor.fetchall()

    def read(self, file_path: str, **kwargs):
        """Reads NER data from a SQLite database."""
        self.check_file_exists(file_path)

        try:
            conn = sqlite3.connect(file_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Fetch all articles with their sentences and entities
            articles = {}

            # Get all articles
            cursor.execute("SELECT * FROM articles")
            article_rows = cursor.fetchall()

            for article_row in article_rows:
                article_id = article_row["id"]
                article = dict(article_row)

                sentences = [
                    dict(row)
                    for row in self._get_article_sentences(cursor, article_id)
                ]

                # For each sentence, get its entities
                for sentence in sentences:
                    sentence_id = sentence["id"]
                    entities = self._get_sentence_entities(cursor, sentence_id)
                    sentence["entities"] = [dict(row) for row in entities]

                article["sentences"] = sentences
                articles[article_id] = article

            conn.close()
            return articles

        except Exception as e:
            error_msg = f"Error reading SQLite database {file_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def write(self, data: Union[List[Dict], Dict], file_path: str, **kwargs):
        """
        Writes NER data to a SQLite database with tables for articles, sentences, and entities.

        Parameters:
        -----------
        data: Union[List[Dict], Dict]
            Either a list of article dictionaries or a single article dictionary with the following structure:
            {
                "id": "article_id",
                "text": "article full text",
                "metadata": {...},
                "sentences": [
                    {
                        "id": "sentence_id",
                        "text": "sentence text",
                        "entities": [
                            {
                                "id": "entity_id",
                                "text": "entity text",
                                "start": 10,
                                "end": 15,
                                "label": "PERSON",
                                "score": 0.95
                            },
                            ...
                        ]
                    },
                    ...
                ]
            }
        file_path: str
            Path to the SQLite database file
        **kwargs:
            Additional arguments specific to SQLite operations
        """
        self.ensure_dir_exists(file_path)

        # Normalize data to list format
        articles = data if isinstance(data, list) else [data]

        try:
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()

            # Create tables if they don't exist
            if not self.tables_created:
                self._create_tables(cursor)
                self.tables_created = True

            # Begin a transaction for better performance
            conn.execute("BEGIN TRANSACTION")

            for article in articles:
                # Insert article
                article_id = article.get("id", None)
                if not article_id:
                    continue

                article_data = {
                    "id": article_id,
                    "text": article.get("text", ""),
                    "metadata": json.dumps(article.get("metadata", {})),
                }

                self._insert_or_update_article(cursor, article_data)

                # Process sentences
                for sentence in article.get("sentences", []):
                    sentence_id = sentence.get("id", None)
                    if not sentence_id:
                        continue

                    sentence_data = {
                        "id": sentence_id,
                        "article_id": article_id,
                        "text": sentence.get("text", ""),
                        "position": sentence.get("position", 0),
                    }

                    self._insert_or_update_sentence(cursor, sentence_data)

                    # Process entities
                    for entity in sentence.get("entities", []):
                        entity_id = entity.get("id", None)
                        if not entity_id:
                            continue

                        entity_data = {
                            "id": entity_id,
                            "sentence_id": sentence_id,
                            "text": entity.get("text", ""),
                            "start": entity.get("start", 0),
                            "end": entity.get("end", 0),
                            "label": entity.get("label", ""),
                            "score": entity.get("score", 0.0),
                        }

                        self._insert_or_update_entity(cursor, entity_data)

            # Commit the transaction
            conn.commit()
            conn.close()

            logger.debug(
                f"Successfully wrote data to SQLite database {file_path}"
            )

        except Exception as e:
            error_msg = f"Error writing SQLite database {file_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _create_tables(self, cursor):
        """Create the necessary tables if they don't exist."""
        # Articles table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            text TEXT,
            metadata TEXT
        )
        """
        )

        # Sentences table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS sentences (
            id TEXT PRIMARY KEY,
            article_id TEXT,
            text TEXT,
            position INTEGER,
            FOREIGN KEY (article_id) REFERENCES articles (id)
        )
        """
        )

        # Entities table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            sentence_id TEXT,
            text TEXT,
            start INTEGER,
            end INTEGER,
            label TEXT,
            score REAL,
            FOREIGN KEY (sentence_id) REFERENCES sentences (id)
        )
        """
        )

    def _insert_or_update_article(self, cursor, article_data):
        """Insert or update an article."""
        cursor.execute(
            """
        INSERT OR REPLACE INTO articles (id, text, metadata)
        VALUES (:id, :text, :metadata)
        """,
            article_data,
        )

    def _insert_or_update_sentence(self, cursor, sentence_data):
        """Insert or update a sentence."""
        cursor.execute(
            """
        INSERT OR REPLACE INTO sentences (id, article_id, text, position)
        VALUES (:id, :article_id, :text, :position)
        """,
            sentence_data,
        )

    def _insert_or_update_entity(self, cursor, entity_data):
        """Insert or update an entity."""
        cursor.execute(
            """
        INSERT OR REPLACE INTO entities (id, sentence_id, text, start, end, label, score)
        VALUES (:id, :sentence_id, :text, :start, :end, :label, :score)
        """,
            entity_data,
        )
