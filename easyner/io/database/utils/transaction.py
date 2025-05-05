"""
Transaction management utilities for the database operations.
"""

import functools
import logging
from typing import Callable, Any, TypeVar

# Define a generic type for the decorator
F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def transactional(method: F) -> F:
    """
    Decorator to wrap database methods in transactions.

    This decorator ensures that database operations are executed within a transaction
    and handles commits and rollbacks appropriately based on whether the operation succeeds.

    Args:
        method: The method to wrap with transaction management

    Returns:
        Wrapped method that executes within a transaction context

    Example:
        @transactional
        def create_tables(self) -> None:
            # Operations will be automatically wrapped in a transaction
            self.connection.execute(ARTICLES_TABLE_SQL)
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Access the connection object from self
        conn = self.connection

        try:
            # Start transaction
            conn.begin_transaction()

            # Execute the method
            result = method(self, *args, **kwargs)

            # Commit the transaction if successful
            conn.commit()

            return result

        except Exception as e:
            # Roll back the transaction on error
            logger.error(f"Transaction failed in {method.__name__}: {e}")
            conn.rollback()

            # Re-raise the exception after rollback
            raise

    return wrapper
