"""Standard paths used throughout the EasyNER project.

This module defines all standard paths and directories used by the project,
ensuring consistency across different modules and components.

Use these to comply with DRY and do not hardcode paths in your scripts.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PACKAGE_ROOT = Path(__file__).parent.parent

# This is the main directory for scripts and modules
# which are not part of the package
# TODO: refactor all into package
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

CONFIG_DIR = PACKAGE_ROOT / "config"
INFRASTRUCTURE_DIR = PACKAGE_ROOT / "infrastructure"

# Config files
CONFIG_PATH = PROJECT_ROOT / "config.json"
TEMPLATE_PATH = PROJECT_ROOT / "config.template.json"
SCHEMA_PATH = CONFIG_DIR / "schema.json"
BACKUPS_DIR = PROJECT_ROOT / ".backups"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Results subdirectories
NER_RESULTS_DIR = RESULTS_DIR / "ner"
DATALOADER_RESULTS_DIR = RESULTS_DIR / "dataloader"
SPLITTER_RESULTS_DIR = RESULTS_DIR / "splitter"
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis"


# Ensure paths exist function
def ensure_paths_exist(
    create_results_dirs: bool = True, dry_run: bool = False
) -> dict:
    """Ensure that all required directories exist.

    Args:
        paths_to_check: List of paths to check/create
        create_results_dirs: Create missing results directories if True
        dry_run: Only check existence without creating if True

    Returns:
        bool: True if all paths exist or were created successfully
    """
    results: dict = {}

    # Base directories
    for name, dir_path in {
        "DATA_DIR": DATA_DIR,
        "MODELS_DIR": MODELS_DIR,
    }.items():
        exists = dir_path.exists()
        created = False

        if not exists and not dry_run:
            dir_path.mkdir(exist_ok=True)
            created = True

        results[name] = {
            "exists": exists or created,
            "created": created,
            "path": dir_path,
        }

    # Results directories
    if create_results_dirs:
        for name, dir_path in {
            "RESULTS_DIR": RESULTS_DIR,
            "NER_RESULTS_DIR": NER_RESULTS_DIR,
            "DATALOADER_RESULTS_DIR": DATALOADER_RESULTS_DIR,
            "SPLITTER_RESULTS_DIR": SPLITTER_RESULTS_DIR,
            "ANALYSIS_RESULTS_DIR": ANALYSIS_RESULTS_DIR,
        }.items():
            exists = dir_path.exists()
            created = False

            if not exists and not dry_run:
                # For subdirectories, ensure parent exists first
                if name != "RESULTS_DIR":
                    RESULTS_DIR.mkdir(exist_ok=True)
                dir_path.mkdir(exist_ok=True)
                created = True

            results[name] = {
                "exists": exists or created,
                "created": created,
                "path": dir_path,
            }

    return results
