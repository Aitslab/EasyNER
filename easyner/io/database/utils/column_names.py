"""
SQL Column name and table name constants extracted from schema definitions.

This module provides constants for all database table names and column names,
extracted from the actual SQL schema files for consistency.
"""

import re
from pathlib import Path
from typing import Dict, List

# Path to schema directory
SCHEMA_DIR = Path(__file__).parent.parent / "schemas"

# Table name constants
ARTICLES_TABLE = "articles"
SENTENCES_TABLE = "sentences"
ENTITIES_TABLE = "entities"


def extract_column_names_from_sql(file_path: Path) -> List[str]:
    """
    Extract column names from a SQL CREATE TABLE statement.

    Args:
        file_path: Path to SQL schema file

    Returns:
        List of column names found in the schema
    """
    try:
        with open(file_path, "r") as f:
            sql_content = f.read()

        # Extract text between CREATE TABLE and closing parenthesis
        match = re.search(r"CREATE TABLE.*?\((.*?)\);", sql_content, re.DOTALL)
        if not match:
            return []

        table_def = match.group(1)

        # Extract column definitions (name and type)
        column_defs = re.findall(r"(\w+)\s+[A-Za-z0-9()]+", table_def)
        return column_defs
    except Exception as e:
        print(f"Error extracting column names from {file_path}: {e}")
        return []


# Extract column names from schema files
articles_columns = extract_column_names_from_sql(SCHEMA_DIR / "articles.sql")
sentences_columns = extract_column_names_from_sql(SCHEMA_DIR / "sentences.sql")
entities_columns = extract_column_names_from_sql(SCHEMA_DIR / "entities.sql")

# Shared column names
ARTICLE_ID = "article_id"
SENTENCE_ID = "sentence_id"
TEXT = "text"
START_CHAR = "start_char"
END_CHAR = "end_char"
METADATA = "metadata"
# These use timezone-aware timestamps from beginning of transaction
CREATED_AT = "created_at"
UPDATED_AT = "updated_at"

# Article column constants
TITLE = "title"

# Sentence column constants
SENTENCIZER_MODEL = "sentencizer_model"
SENTENCIZER_MODEL_METADATA = "sentencizer_model_metadata"

# Entity column constants
ENTITY_ID = "entity_id"
ENTITY_CLASS = "entity_class"
INFERENCE_MODEL = "inference_model"
INFERENCE_MODEL_METADATA = "inference_model_metadata"


# Duplicates column constants
HIERARCHICAL_DUPLICATE = "hierarchical_duplicate"  # bool
KEY_DUPLICATE = "key_duplicate"  # bool
RETURNED_IDS_TABLE = "returned_ids_table"
