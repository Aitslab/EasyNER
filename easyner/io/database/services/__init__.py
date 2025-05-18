"""Services for cross-repository operations and advanced data handling.

This package provides services that operate across multiple repositories,
orchestrating complex operations while maintaining data integrity.
"""

from .data_exchanger import DataExchanger

__all__ = [
    "DataExchanger",
]
