"""
Configuration Backup Utility

This module provides tools for creating, listing, and restoring configuration
backups.
It follows OOP principles to provide a clean, maintainable interface.

Example Usage:
-------------
1. Create a backup:
   ```python
   from easyner.config.backup import ConfigBackupManager

   backup_manager = ConfigBackupManager()
   backup_path = backup_manager.create_backup("pre-experiment-changes")
   print(f"Backup created at: {backup_path}")
   ```

2. List available backups:
   ```python
   backups = backup_manager.list_backups()
   for backup in backups:
       print(f"{backup.name} - created on {backup.timestamp}")
   ```

3. Restore from a backup:
   ```python
   success = backup_manager.restore_backup("backup-20250423-120530")
   if success:
       print("Configuration successfully restored")
   ```
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from easyner.infrastructure.paths import (
    CONFIG_PATH,
    TEMPLATE_PATH,
    BACKUPS_DIR,
)


@dataclass
class ConfigBackup:
    """Represents a configuration backup with metadata."""

    path: Path
    name: str
    timestamp: datetime
    description: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if the backup file exists and is valid."""
        return self.path.exists() and self.path.is_file()


class ConfigBackupManager:
    """Manages configuration file backups.

    This class handles creating, listing, and restoring config backups,
    following the Single Responsibility Principle by focusing only on
    backup management operations.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        template_path: Optional[Path] = None,
        backups_dir: Optional[Path] = None,
    ):
        """Initialize the backup manager.

        Args:
            config_path: Path to the config file (default: from paths)
            template_path: Path to the template file (default: from paths)
            backups_dir: Directory to store backups (default: from paths)
        """
        self.config_path = config_path or CONFIG_PATH
        self.template_path = template_path or TEMPLATE_PATH
        self.backups_dir = backups_dir or BACKUPS_DIR

        # Ensure the backups directory exists
        self._ensure_backup_dir()

    def _ensure_backup_dir(self) -> None:
        """Create the backups directory if it doesn't exist."""
        if not self.backups_dir.exists():
            self.backups_dir.mkdir(parents=True, exist_ok=True)

    def _generate_backup_filename(self, label: Optional[str] = None) -> str:
        """Generate a backup filename with timestamp.

        Args:
            label: Optional label to include in the filename

        Returns:
            A filename string for the backup
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if label:
            # Sanitize the label to be filename-safe
            safe_label = "".join(c if c.isalnum() else "-" for c in label)
            return f"config-backup-{timestamp}-{safe_label}.json"
        return f"config-backup-{timestamp}.json"

    def create_backup(
        self, description: Optional[str] = None, include_metadata: bool = True
    ) -> Path:
        """Create a backup of the current configuration.

        Args:
            description: Optional description of the backup
            include_metadata: Whether to include metadata in the backup file

        Returns:
            Path to the created backup file

        Raises:
            FileNotFoundError: If the config file doesn't exist
            IOError: If there's an issue creating the backup
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {self.config_path}"
            )

        # Read the current config
        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                config_data = json.load(f)
            except json.JSONDecodeError:
                raise IOError(
                    f"Invalid JSON in config file: {self.config_path}"
                )

        # Generate backup filename and path
        backup_filename = self._generate_backup_filename(description)
        backup_path = self.backups_dir / backup_filename

        # Add metadata if requested
        if include_metadata:
            backup_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "description": description,
                    "source_path": str(self.config_path),
                },
                "config": config_data,
            }
        else:
            backup_data = config_data

        # Write the backup file
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, indent=2)

        return backup_path

    def list_backups(self) -> List[ConfigBackup]:
        """List all available configuration backups.

        Returns:
            List of ConfigBackup objects representing available backups
        """
        backups = []

        # Ensure backup directory exists
        self._ensure_backup_dir()

        # Find all backup files
        backup_files = list(self.backups_dir.glob("config-backup-*.json"))

        for backup_file in backup_files:
            try:
                # Extract information from filename
                filename = backup_file.name
                # Parse timestamp from the filename
                # (format: config-backup-YYYYMMDD-HHMMSS-label.json)
                timestamp_str = filename.split("-")[2:4]
                timestamp_str = "-".join(timestamp_str)

                try:
                    timestamp = datetime.strptime(
                        timestamp_str, "%Y%m%d-%H%M%S"
                    )
                except ValueError:
                    # If timestamp parsing fails, use file creation time
                    timestamp = datetime.fromtimestamp(
                        backup_file.stat().st_ctime
                    )

                # Extract description from metadata if available
                description = None
                try:
                    with open(backup_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "metadata" in data:
                            description = data["metadata"].get("description")
                except (json.JSONDecodeError, IOError):
                    # Can't read metadata -> continue without description
                    pass

                backups.append(
                    ConfigBackup(
                        path=backup_file,
                        name=backup_file.stem,
                        timestamp=timestamp,
                        description=description,
                    )
                )

            except Exception as e:
                # Skip invalid backups but don't crash
                print(f"Warning: Could not process backup {backup_file}: {e}")

        # Sort backups by timestamp, newest first
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)

    def restore_backup(
        self, backup_identifier: Union[str, Path, ConfigBackup]
    ) -> bool:
        """Restore configuration from a backup.

        Args:
            backup_identifier: Can be a backup name, path, or
            ConfigBackup object

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If the backup identifier is invalid
            FileNotFoundError: If the backup file doesn't exist
        """
        # Determine the backup path
        backup_path = self._resolve_backup_path(backup_identifier)

        # Read the backup
        with open(backup_path, "r", encoding="utf-8") as f:
            try:
                backup_data = json.load(f)
            except json.JSONDecodeError:
                raise IOError(f"Invalid JSON in backup file: {backup_path}")

        # Extract config data (handle both with and without metadata)
        config_data = backup_data.get("config", backup_data)

        # Create a backup of the current config before restoring
        self.create_backup(description="auto-backup-before-restore")

        # Restore the config
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        return True

    def delete_backup(
        self, backup_identifier: Union[str, Path, ConfigBackup]
    ) -> bool:
        """Delete a backup file.

        Args:
            backup_identifier: Can be a backup name, path, or
            ConfigBackup object

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            ValueError: If the backup identifier is invalid
            FileNotFoundError: If the backup file doesn't exist
        """
        # Determine the backup path (reusing logic from restore_backup)
        backup_path = self._resolve_backup_path(backup_identifier)

        # Delete the backup file
        backup_path.unlink()
        return True

    def _resolve_backup_path(
        self, backup_identifier: Union[str, Path, ConfigBackup]
    ) -> Path:
        """Resolve a backup identifier to an actual file path.

        Args:
            backup_identifier: Can be a backup name, path, or
            ConfigBackup object

        Returns:
            Path to the backup file

        Raises:
            ValueError: If the backup identifier is invalid or
            matches multiple backups
            FileNotFoundError: If the backup file doesn't exist
        """
        backup_path = None

        if isinstance(backup_identifier, ConfigBackup):
            backup_path = backup_identifier.path

        elif isinstance(backup_identifier, Path):
            backup_path = backup_identifier

        elif isinstance(backup_identifier, str):
            # Check if it's a full path
            path = Path(backup_identifier)
            if path.exists() and path.is_file():
                backup_path = path
            else:
                # Try to find by name in the backups directory
                if not backup_identifier.endswith(".json"):
                    backup_identifier += ".json"

                potential_path = self.backups_dir / backup_identifier
                if potential_path.exists():
                    backup_path = potential_path
                else:
                    # Search by partial name
                    matches = list(
                        self.backups_dir.glob(f"*{backup_identifier}*.json")
                    )
                    if len(matches) == 1:
                        backup_path = matches[0]
                    elif len(matches) > 1:
                        match_names = [m.name for m in matches]
                        raise ValueError(
                            "Multiple matching backups found: "
                            f"{', '.join(match_names)}"
                        )

        if not backup_path:
            raise ValueError(f"Could not find backup: {backup_identifier}")

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        return backup_path
