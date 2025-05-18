"""
Repository package for database access.

This package provides repository classes for accessing article,
sentence, and entity data in the database.
"""

# Import the base Repository interface
from .base import Repository

# Import concrete implementations
from .article_repository import ArticleRepository
from .sentence_repository import SentenceRepository
from .entity_repository import EntityRepository

__all__ = [
    "Repository",
    "ArticleRepository",
    "SentenceRepository",
    "EntityRepository",
]
