"""
EasyNER Configuration Management

This module provides tools for creating, validating, and working with configuration files.

Example Usage:
-------------
1. Load and use the configuration:
   ```python
   from easyner.config import config_manager

   # Get the entire config
   config = config_manager.get_config()
   print(f"Using {config['CPU_LIMIT']} CPU cores")

   # Access specific sections using dictionary syntax
   ner_config = config_manager["ner"]
   print(f"Using NER model: {ner_config['model_type']}")

   # Check if a config key exists
   if "experimental_features" in config_manager:
       print("Experimental features are configured")

   # Iterate through config sections
   for section_name, section_data in config_manager.items():
       print(f"Section: {section_name}")
   ```

2. Working with configuration templates and validation:
   ```python
   from easyner.config import config_manager

   # Generate a template
   config_manager.generate_template()

   # Validate configuration files
   is_valid = config_manager.validate_all()
   ```

3. For more advanced use cases:
   ```python
   from easyner.config.manager import ConfigManager
   from easyner.config.validator import ConfigValidator
   from easyner.config.generator import ConfigGenerator

   # Create custom instances for specific needs
   custom_manager = ConfigManager(config_path="custom_config.json")

   # Access config using dictionary syntax
   custom_config = custom_manager["ner"]
   ```
"""

from easyner.infrastructure.paths import (
    CONFIG_PATH,
    TEMPLATE_PATH,
    SCHEMA_PATH,
)
from easyner.config.manager import ConfigManager

# Create a default instance of ConfigManager for convenient access
config_manager = ConfigManager(
    config_path=CONFIG_PATH,
    template_path=TEMPLATE_PATH,
    schema_path=SCHEMA_PATH,
)
