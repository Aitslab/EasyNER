"""
Schema definitions for EasyNer database tables.

This package contains SQL schema definitions for tables used in the EasyNer application.
"""

from pathlib import Path

import easyner.io.database.schemas.python_mappings as python_mappings

from ..utils.sql_utils import read_sql_file

SCHEMA_DIR = Path(__file__).parent

# Schema file paths
ARTICLES_SCHEMA = SCHEMA_DIR / "articles.sql"
SENTENCES_SCHEMA = SCHEMA_DIR / "sentences.sql"
ENTITIES_SCHEMA = SCHEMA_DIR / "entities.sql"

CONVERSION_LOG_SCHEMA = SCHEMA_DIR / "conversion_log.sql"

# SQL schema statements - loaded at import time using the utility function
ARTICLES_TABLE_SQL = read_sql_file(ARTICLES_SCHEMA)
SENTENCES_TABLE_SQL = read_sql_file(SENTENCES_SCHEMA)
ENTITIES_TABLE_SQL = read_sql_file(ENTITIES_SCHEMA)
# Conversion log schema
CONVERSION_LOG_TABLE_SQL = read_sql_file(CONVERSION_LOG_SCHEMA)

__all__ = [
    # SQL statements (preferred usage)
    "ARTICLES_TABLE_SQL",
    "SENTENCES_TABLE_SQL",
    "ENTITIES_TABLE_SQL",
    # File paths (for backward compatibility)
    "SCHEMA_DIR",
    "ARTICLES_SCHEMA",
    "SENTENCES_SCHEMA",
    "ENTITIES_SCHEMA",
]
