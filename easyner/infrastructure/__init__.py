"""
EasyNER Infrastructure Package

This package contains core infrastructure components used across the
EasyNER project, including path management, environment settings, and
other low-level utilities that are foundational to the application but
not tied to specific business logic.
"""

from easyner.infrastructure.paths import (
    # Base paths
    PROJECT_ROOT,
    SCRIPTS_DIR,
    # Config files
    CONFIG_PATH,
    TEMPLATE_PATH,
    SCHEMA_PATH,
    # Data directories
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    # Results subdirectories
    NER_RESULTS_DIR,
    DATALOADER_RESULTS_DIR,
    SPLITTER_RESULTS_DIR,
    ANALYSIS_RESULTS_DIR,
    # Utility functions
    ensure_paths_exist,
)

__all__ = [
    "PROJECT_ROOT",
    "SCRIPTS_DIR",
    "CONFIG_PATH",
    "TEMPLATE_PATH",
    "SCHEMA_PATH",
    "DATA_DIR",
    "RESULTS_DIR",
    "MODELS_DIR",
    "NER_RESULTS_DIR",
    "DATALOADER_RESULTS_DIR",
    "SPLITTER_RESULTS_DIR",
    "ANALYSIS_RESULTS_DIR",
    "ensure_paths_exist",
]
