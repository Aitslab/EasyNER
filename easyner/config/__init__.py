"""
EasyNER Configuration Management

Tools for creating, validating, and working with the config.

Main benefit is to validate the configuration files against a schema on
load ensure that the configuration is correct and complete.

Example Usage:
-------------
1. Load and validate a configuration file:
   ```python
   from scripts.config import load_config

   # Load the default config file (config.json)
   config = load_config()

   # Access a specific module's configuration
   ner_config = config["ner"]
   print(f"Using NER model: {ner_config['model_type']}")
   ```

2. Generate a template from the schema:
   ```python
   from scripts.config import generate_template_from_schema,
   DEFAULT_TEMPLATE_PATH

   # Generate a template based on the schema
   generate_template_from_schema(DEFAULT_TEMPLATE_PATH)
   print(f"Template generated at {DEFAULT_TEMPLATE_PATH}")
   ```

3. Validate a configuration file:
   ```python
   from scripts.config import validate_config

   # Validate a specific config file
   is_valid = validate_config("my_custom_config.json")
   if not is_valid:
       print("Configuration has issues that need to be fixed")
   ```

4. Format a JSON file with prettier:
   ```python
   from scripts.config import format_with_prettier

   # Format a JSON file using prettier (requires npm prettier to be
   # installed)
   format_with_prettier("config.template.json")
   ```
"""

__all__ = [
    # Variables
    # "config",
    # Functions
    "load_config",
]


import json
from pathlib import Path
from typing import Dict, Any, Union

# Import paths from the infrastructure package
from easyner.infrastructure.paths import CONFIG_PATH


def load_config(config_path: Union[str, Path] = CONFIG_PATH) -> Dict[str, Any]:
    """Load and validate a configuration file.

    Args:
        config_path: Path to the configuration file to load

    Returns:
        Dict containing the loaded configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        json.JSONDecodeError: If the config file contains invalid JSON
    """
    from . import validator

    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Validate the config
    is_valid = validator.validate_config(config_path)
    if not is_valid:
        print(f"Warning: Configuration at {config_path} has validation issues")

    # Load the configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


# config: Dict[str, Any] = load_config()
