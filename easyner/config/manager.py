"""Configuration Manager Module.

This module provides a central class for managing all
configuration-related operations, including loading,
validation, generation and backup.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Optional, Union

from easyner.config.backup import ConfigBackup, ConfigBackupManager
from easyner.config.generator import ConfigGenerator
from easyner.config.validator import ConfigValidator
from easyner.infrastructure.paths import (
    CONFIG_PATH,
    SCHEMA_PATH,
    TEMPLATE_PATH,
)


class ConfigManager:
    """Central manager for all configuration operations.

    Unified interface for working with configuration files,
    including loading, validating, generating templates,
    and ensuring config existence.

    This class supports dictionary-like access to configuration sections:

    ```python
    # Access config sections using dictionary syntax
    ner_config = config_manager["ner"]
    cpu_limit = config_manager["CPU_LIMIT"]
    ```

    Attributes:
        config_path: Path to the main configuration file
        template_path: Path to the template configuration file
        schema_path: Path to the JSON schema file
        quiet: Whether to suppress standard output messages

    """

    def __init__(
        self,
        config_path: Path = CONFIG_PATH,
        template_path: Path = TEMPLATE_PATH,
        schema_path: Path = SCHEMA_PATH,
        quiet: bool = False,
    ) -> None:
        """Initialize a ConfigManager instance.

        Args:
            config_path: Path to the main configuration file
            template_path: Path to the template configuration file
            schema_path: Path to the JSON schema file
            quiet: Whether to suppress standard output messages

        """
        self.config_path = Path(config_path)
        self.template_path = Path(template_path)
        self.schema_path = Path(schema_path)
        self.quiet = quiet

        # Create component instances
        self.validator = ConfigValidator(schema_path, quiet)
        self.generator = ConfigGenerator(schema_path, quiet)
        self.backup_manager = ConfigBackupManager(
            config_path=self.config_path,
            template_path=self.template_path,
        )

        # Cache for loaded configuration
        self._config: Optional[dict[str, Any]] = None

    def _print_message(self, message: str) -> None:
        """Print a message if quiet mode is not enabled.

        Args:
            message: The message to print

        """
        if not self.quiet:
            print(message)

    def load_config(self, validate: bool = True) -> dict[str, Any]:
        """Load and optionally validate the configuration file.

        Args:
            validate: Whether to validate the configuration against the schema

        Returns:
            Dict containing the loaded configuration

        Raises:
            FileNotFoundError: If the config file doesn't exist
            json.JSONDecodeError: If the config file contains invalid JSON

        """
        # Check if config exists, if not try to create it
        if not self.config_path.exists():
            self._print_message(f"Configuration file not found: {self.config_path}")
            if self.template_path.exists():
                self._print_message("Attempting to create from template...")
                success = self.generator.copy_template_to_config(
                    self.template_path,
                    self.config_path,
                )
                if not success:
                    msg = f"Configuration file not found: {self.config_path}"
                    raise FileNotFoundError(
                        msg,
                    )
            else:
                msg = f"Configuration file not found: {self.config_path}"
                raise FileNotFoundError(
                    msg,
                )

        # Validate the config if requested
        if validate:
            is_valid = self.validator.validate_config(self.config_path)
            if not is_valid:
                self._print_message("Warning: Configuration has validation issues")

        # Load the configuration
        with open(self.config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Cache the loaded config
        self._config = config
        return config

    def get_config(self, reload: bool = False) -> dict[str, Any]:
        """Get the configuration, loading it if necessary.

        Args:
            reload: Whether to force reloading the configuration

        Returns:
            Dict containing the configuration

        """
        if self._config is None or reload:
            return self.load_config()
        return self._config

    def ensure_config_exists(self) -> bool:
        """Ensure that the configuration file exists, creating it if necessary.

        Returns:
            bool: True if the config file exists or was created, False on error

        """
        if self.config_path.exists():
            self._print_message(f"Config file already exists at: {self.config_path}")
            return True

        # Check if template exists, if not generate it
        if not self.template_path.exists():
            self._print_message(
                f"Template file not found at {self.template_path}, generating...",
            )
            success = self.generator.generate_template(self.template_path)
            if not success:
                self._print_message(
                    f"Failed to generate template at {self.template_path}",
                )
                return False

        # Copy template to config
        return self.generator.copy_template_to_config(
            self.template_path,
            self.config_path,
        )

    def generate_template(self, skip_prettier: bool = False) -> bool:
        """Generate a template configuration file from the schema.

        Args:
            skip_prettier: Whether to skip formatting with prettier

        Returns:
            bool: True if template generation succeeded, False otherwise

        """
        return self.generator.generate_template(self.template_path, skip_prettier)

    def validate_all(self) -> bool:
        """Validate all configuration files.

        Returns:
            bool: True if all validation tests pass, False otherwise

        """
        return self.validator.run_validation_tests(self.config_path, self.template_path)

    def save_config(self, config_data: dict[str, Any]) -> bool:
        """Save configuration data to the config file.

        Args:
            config_data: The configuration data to save

        Returns:
            bool: True if saving succeeded, False otherwise

        """
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, sort_keys=False)

            # Update cached config
            self._config = config_data

            # Validate the saved config
            is_valid = self.validator.validate_config(self.config_path)
            if not is_valid:
                self._print_message(
                    "Warning: Saved configuration has validation issues",
                )

            return True
        except Exception as e:
            self._print_message(f"Error saving configuration: {e}")
            return False

    # Backup management methods
    def create_backup(self, description: Optional[str] = None) -> Path:
        """Create a backup of the current configuration.

        Args:
            description: Optional description of the backup

        Returns:
            Path to the created backup file

        Raises:
            FileNotFoundError: If the config file doesn't exist
            IOError: If there's an issue creating the backup

        """
        return self.backup_manager.create_backup(description)

    def list_backups(self) -> list[ConfigBackup]:
        """List all available configuration backups.

        Returns:
            List of ConfigBackup objects representing available backups

        """
        return self.backup_manager.list_backups()

    def restore_backup(self, backup_identifier: Union[str, Path, ConfigBackup]) -> bool:
        """Restore configuration from a backup.

        Args:
            backup_identifier: Can be a backup name, path,
            or ConfigBackup object

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If the backup identifier is invalid
            FileNotFoundError: If the backup file doesn't exist

        """
        return self.backup_manager.restore_backup(backup_identifier)

    def delete_backup(self, backup_identifier: Union[str, Path, ConfigBackup]) -> bool:
        """Delete a backup file.

        Args:
            backup_identifier: Can be a backup name, path,
            or ConfigBackup object

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            ValueError: If the backup identifier is invalid
            FileNotFoundError: If the backup file doesn't exist

        """
        return self.backup_manager.delete_backup(backup_identifier)

    # Dictionary-like methods
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dictionary syntax.

        Args:
            key: The configuration key to retrieve

        Returns:
            The value associated with the key

        Raises:
            KeyError: If the key doesn't exist in the configuration

        """
        config = self.get_config()
        try:
            return config[key]
        except KeyError as e:
            msg = f"Key '{key}' not found in configuration: {e}"
            raise KeyError(msg) from e

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a configuration value using dictionary syntax.

        Args:
            key: The configuration key to set
            value: The value to set

        Note:
            This doesn't automatically save the configuration to disk.
            Call save_config() to persist changes.

        """
        config = self.get_config()
        config[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the configuration.

        Args:
            key: The key to check

        Returns:
            True if the key exists, False otherwise

        """
        config = self.get_config()
        return key in config

    def __iter__(self) -> Iterator[str]:
        """Iterate over configuration keys.

        Returns:
            Iterator over configuration keys

        """
        config = self.get_config()
        return iter(config)

    def __len__(self) -> int:
        """Get the number of configuration entries.

        Returns:
            The number of top-level configuration entries

        """
        config = self.get_config()
        return len(config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default fallback.

        Args:
            key: The configuration key to retrieve
            default: Value to return if key doesn't exist

        Returns:
            The value associated with the key, or the default if not found

        """
        config = self.get_config()
        return config.get(key, default)

    def keys(self) -> Iterable[str]:
        """Get configuration keys.

        Returns:
            Iterable over configuration keys

        """
        config = self.get_config()
        return config.keys()

    def values(self) -> Iterable[Any]:
        """Get configuration values.

        Returns:
            Iterator over configuration values

        """
        config = self.get_config()
        return config.values()

    def items(self) -> Iterable[tuple[str, Any]]:
        """Get configuration key-value pairs.

        Returns:
            Iterator over configuration key-value pairs

        """
        config = self.get_config()
        return config.items()
