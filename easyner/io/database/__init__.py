# Database module initialization
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import core components
from .connection import DatabaseConnection
from .duckdb_handler import DuckDBHandler
from .manager import TableManager
from .repositories import (
    ArticleRepository,
    EntityRepository,
    SentenceRepository,
)

# Import schema definitions
from .schemas import (
    ARTICLES_SCHEMA,
    ENTITIES_SCHEMA,
    SENTENCES_SCHEMA,
)

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
    "ENTITIES_SCHEMA",
]
