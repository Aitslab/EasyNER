# Database module initialization
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

# Import schema definitions
from .schemas import (
    ARTICLES_SCHEMA,
    SENTENCES_SCHEMA,
    ENTITY_SEQUENCE_SCHEMA,
    ENTITIES_SCHEMA,
)

# Import core components
from .connection import DatabaseConnection
from .manager import TableManager
from .repositories import (
    ArticleRepository,
    SentenceRepository,
    EntityRepository,
)
from .duckdb_handler import DuckDBHandler

# Set the module base directory
MODULE_DIR = Path(__file__).parent

__all__ = [
    "DatabaseConnection",
    "TableManager",
    "ArticleRepository",
    "SentenceRepository",
    "EntityRepository",
    "DuckDBHandler",
    # Schema paths
    "ARTICLES_SCHEMA",
    "SENTENCES_SCHEMA",
    "ENTITY_SEQUENCE_SCHEMA",
    "ENTITIES_SCHEMA",
]
