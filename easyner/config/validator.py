"""
Configuration Validation Module

This module provides classes for validating configuration files against
a JSON schema.
"""

import json
import re
import sys
import jsonschema
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Iterable
import argparse

from easyner.infrastructure.paths import (
    SCHEMA_PATH,
    CONFIG_PATH,
    TEMPLATE_PATH,
)

import jsonschema.exceptions


class ConfigValidator:
    """Validates configuration files against a JSON schema.

    This class encapsulates all validation logic for configuration files,
    including schema validation, path type checking, and absolute path
    detection.

    Attributes:
        schema_path: Path to the JSON schema file
        quiet: Whether to suppress standard output messages
    """

    # Format constants to avoid duplication
    HEADER_DIVIDER = "=" * 80
    SECTION_DIVIDER = "-" * 80

    def __init__(
        self, schema_path: Path = SCHEMA_PATH, quiet: bool = False
    ) -> None:
        """Initialize a ConfigValidator instance.

        Args:
            schema_path: Path to the JSON schema file
            quiet: Whether to suppress standard output messages
        """
        self.schema_path = schema_path
        self.quiet = quiet
        self._schema: Optional[Dict[str, Any]] = None
        self._errors: List[str] = []
        self._warnings: List[str] = []
        self._empty_paths: List[str] = []

    @property
    def schema(self) -> Dict[str, Any]:
        """Load and return the JSON schema, caching it for future use.

        Returns:
            Dict containing the loaded schema

        Raises:
            FileNotFoundError: If schema file not found
            json.JSONDecodeError: If schema contains invalid JSON
        """
        if self._schema is None:
            self._schema = self._load_schema()
        return self._schema

    def _load_schema(self) -> Dict[str, Any]:
        """Load the schema from file.

        Returns:
            Dict containing the schema

        Raises:
            FileNotFoundError: If schema file not found
            json.JSONDecodeError: If schema contains invalid JSON
        """
        try:
            with open(self.schema_path, "r") as f:
                schema = json.load(f)
            return schema
        except FileNotFoundError:
            print(f"Error: Schema file not found at {self.schema_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in schema file: {e}")
            sys.exit(1)

    def _print_section(self, title: str, is_header: bool = False) -> None:
        """Print a formatted section header.

        Args:
            title: The title to display
            is_header: Whether this is a main header (True) or section (False)
        """
        if self.quiet:
            return

        divider = self.HEADER_DIVIDER if is_header else self.SECTION_DIVIDER
        print(f"\n{divider}")
        print(title)
        print(f"{divider}")

    def _print_message(self, message: str, prefix: str = "") -> None:
        """Print a formatted message.

        Args:
            message: The message to display
            prefix: Optional prefix for the message (e.g., indentation)
        """
        if self.quiet:
            return

        print(f"{prefix}{message}")

    def validate_config(self, config_file: Union[str, Path]) -> bool:
        """Validates the config file against the schema.

        Args:
            config_file: Path to the config file to validate

        Returns:
            bool: True if validation succeeds, False otherwise
        """
        # Reset validation state
        self._errors = []
        self._warnings = []
        self._empty_paths = []

        # Initialize config_data to None to prevent unbound variable issues
        config_data = None

        try:
            config_file_path = Path(config_file)
            config_file_name = config_file_path.name

            if not self.quiet:
                self._print_section(
                    f"VALIDATING: {config_file_name}", is_header=True
                )

            # Load config data
            with open(config_file, "r") as f:
                config_data = json.load(f)

            # Check path types first
            self._check_path_types(config_data, self.schema)

            # Check schema reference
            self._check_schema_reference(config_data)

            # Print errors if any
            if self._errors:
                self._print_section("VALIDATION ERRORS:")
                for error in self._errors:
                    self._print_message(error, prefix="  ")
                return False

            # Print warnings
            if self._warnings and not self.quiet:
                self._print_section("VALIDATION WARNINGS:")
                for warning in self._warnings:
                    self._print_message(warning, prefix="  ")

            # Print empty paths
            if self._empty_paths and not self.quiet:
                self._print_section(
                    "INFO: The following path fields are left empty:"
                )
                for path in self._empty_paths:
                    self._print_message(path, prefix="  • ")

            # Perform schema validation
            try:
                jsonschema.validate(instance=config_data, schema=self.schema)
            except jsonschema.exceptions.ValidationError as e:
                # Enhanced validation error reporting
                error_path = (
                    ".".join([str(p) for p in e.path]) if e.path else "root"
                )
                error_msg = (
                    f"Schema validation error at '{error_path}': {e.message}"
                )

                # Show the invalid value
                if e.instance is not None:
                    error_msg += f"\nInvalid value: {e.instance}"

                # Show what was expected from the schema
                if e.schema:
                    validation_info = {}
                    for key in [
                        "type",
                        "required",
                        "minimum",
                        "maximum",
                        "pattern",
                        "enum",
                    ]:
                        if key in e.schema:
                            validation_info[key] = e.schema[key]
                    if validation_info:
                        error_msg += f"\nExpected: {validation_info}"

                self._print_section("SCHEMA VALIDATION ERROR:")
                self._print_message(error_msg, prefix="  ")
                return False

            if not self.quiet:
                self._print_section(
                    "RESULT: Configuration validation successful!"
                )

            return True

        except jsonschema.exceptions.ValidationError as e:
            return self._handle_validation_error(e, config_data)
        except FileNotFoundError as e:
            self._print_section("FILE ERROR:")
            self._print_message(f"File not found: {e}", prefix="  ")
            return False
        except json.JSONDecodeError as e:
            self._print_section("JSON ERROR:")
            self._print_message(f"JSON decode error: {e}", prefix="  ")
            return False

    def _handle_validation_error(
        self,
        error: jsonschema.exceptions.ValidationError,
        config_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Handle validation errors from jsonschema.

        Args:
            error: The validation error
            config_data: The configuration data (if available)

        Returns:
            bool: Always False (validation failed)
        """
        error_path = (
            ".".join([str(p) for p in error.path]) if error.path else "root"
        )

        self._print_section("VALIDATION ERROR:")

        if "pattern" in str(error) and "does not match" in str(error):
            # This is a pattern validation error, likely for a path
            if config_data:
                error_value = self._get_value_at_path(config_data, error.path)
                error_type = type(error_value).__name__
            else:
                self._print_message(
                    "Unexpected error: config_data not available", prefix="  "
                )
                return False

            if isinstance(error_value, (int, float, bool)):
                # The value is not even a string
                self._print_message(
                    f"Path field '{error_path}' must be a string, "
                    f"but got {error_type}: {error_value}",
                    prefix="  ",
                )
            elif error_value == "":
                # Empty strings are allowed with an info message
                self._print_section("INFO:")
                self._print_message(
                    f"Path field '{error_path}' is empty", prefix="  "
                )
                # Return True to allow empty paths
                return True
            else:
                # The value is a string but doesn't match the pattern
                self._print_message(
                    f"Path field '{error_path}' "
                    f"contains an invalid path format: {error_value}",
                    prefix="  ",
                )
        else:
            # For other validation errors
            self._print_message(
                f"Error at '{error_path}': {error.message}", prefix="  "
            )

        return False

    def _check_schema_reference(self, config_data: Dict[str, Any]) -> None:
        """Check if the $schema field is correctly set in the configuration.

        Args:
            config_data: The configuration data
        """
        if "$schema" not in config_data:
            self._warnings.append(
                f"$schema field is missing. It should be set to \n"
                f"{self.schema_path} to provide type hints"
            )
        elif config_data["$schema"] != str(self.schema_path):
            self._warnings.append(
                f"$schema has unexpected value: "
                f"'{config_data['$schema']}'. "
                f"Expected: '{self.schema_path}'"
            )

    def _check_path_types(
        self,
        data: Any,
        schema_part: Dict[str, Any],
        path: Optional[List[Any]] = None,  # Changed List[str] to List[Any]
    ) -> None:
        """Recursively check for path fields with non-string values.

        Args:
            data: The data to check (can be dict or list)
            schema_part: The schema part to check against
            path: Current path in the data structure (list of keys/indices)
        """
        if path is None:
            path = []

        # Only process if data is a dictionary
        if isinstance(data, dict) and isinstance(schema_part, dict):
            # Check each property in this object
            for key, value in data.items():
                current_path: List[Any] = path + [
                    key
                ]  # Ensure current_path is List[Any]
                path_str = ".".join(str(p) for p in current_path)

                # Check if this property exists in the schema
                if key in schema_part.get("properties", {}):
                    prop_schema = schema_part["properties"][key]

                    # If this is a path reference
                    if prop_schema.get("$ref") == "#/definitions/path":
                        if not isinstance(value, str):
                            self._errors.append(
                                f"Path field '{path_str}' must be a string, "
                                f"but got {type(value).__name__}: {value}"
                            )
                        elif value == "":
                            self._empty_paths.append(path_str)
                    # If this is an object, recurse
                    elif (
                        isinstance(value, dict)
                        and prop_schema.get("type") == "object"
                    ):
                        self._check_path_types(
                            value,
                            prop_schema,
                            current_path,
                        )
                    # If this is an array, recurse through items
                    elif (
                        isinstance(value, list)
                        and prop_schema.get("type") == "array"
                    ):
                        item_schema = prop_schema.get("items", {})
                        # If the array items are paths
                        if item_schema.get("$ref") == "#/definitions/path":
                            for i, item in enumerate(value):
                                item_path = path_str + f"[{i}]"
                                if not isinstance(item, str):
                                    self._errors.append(
                                        f"Path field '{item_path}' must be a string, "
                                        f"but got {type(item).__name__}: {item}"
                                    )
                                elif item == "":
                                    self._empty_paths.append(item_path)
                        # Otherwise, if it's an array of objects or arrays, recurse
                        else:
                            for i, item in enumerate(value):
                                # Only recurse if item is dict or list and schema expects it
                                if isinstance(item, (dict, list)):
                                    self._check_path_types(
                                        item,
                                        item_schema,
                                        current_path + [i],
                                    )
        # Handle case where data is a list (e.g., recursive call from list item)
        elif isinstance(data, list) and isinstance(schema_part, dict):
            # This case might occur if the schema expects a list of objects/arrays
            # We iterate through the list items and check them against the item schema
            item_schema = schema_part.get("items", {})
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    self._check_path_types(
                        item,
                        item_schema,
                        path + [i],  # This now correctly passes List[Any]
                    )

    def _get_value_at_path(self, data: Any, path: Iterable[Any]) -> Any:
        """Get a value from a nested structure using a jsonschema error path.

        Args:
            data: The nested data structure
            path: The path from a jsonschema ValidationError

        Returns:
            The value at the specified path
        """
        current = data
        for part in path:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and isinstance(part, int):
                if part < len(current):
                    current = current[part]
                else:
                    return None
            else:
                return None
        return current

    def check_absolute_paths(
        self,
        config_file: Union[str, Path],
        section_keys: Optional[List[str]] = None,
    ) -> bool:
        """Check for absolute paths in a config file.

        Args:
            config_file: Path to the config file to check
            section_keys: Optional list of section keys to check specifically

        Returns:
            bool: True if no absolute paths are found, False otherwise
        """
        try:
            config_path = Path(config_file)
            with open(config_path, "r") as f:
                config_data = json.load(f)

            # If looking at template file, we need to check for placeholders
            if "template" in config_path.name:
                found_paths: List[Tuple[str, str]] = []
                self._find_absolute_paths(config_data, found_paths)

                if found_paths:
                    self._print_section("ABSOLUTE PATHS ERROR:")
                    self._print_message(
                        "Absolute paths found in template file:", prefix="  "
                    )
                    for path, value in found_paths:
                        self._print_message(
                            f"{path}: {value}", prefix="    • "
                        )
                    return False

            return True
        except Exception as e:
            self._print_section("PATH CHECK ERROR:")
            self._print_message(f"Error checking paths: {e}", prefix="  ")
            return False

    def _find_absolute_paths(
        self, obj: Any, found_paths: List[Tuple[str, str]], path: str = ""
    ) -> None:
        """Recursively find absolute paths in an object.

        Args:
            obj: The object to search
            found_paths: List to store found absolute paths
            path: Current path string for reporting
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "_comments" or key == "CONFIG_VERSION":
                    continue
                new_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and (
                    Path(value).is_absolute() or re.search(r"[A-Z]:/.*", value)
                ):
                    if value != "PATH_PLACEHOLDER":
                        found_paths.append((new_path, value))
                elif isinstance(value, (dict, list)):
                    self._find_absolute_paths(value, found_paths, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                if isinstance(item, str) and (
                    Path(item).is_absolute() or re.search(r"[A-Z]:/.*", item)
                ):
                    if item != "PATH_PLACEHOLDER":
                        found_paths.append((new_path, item))
                elif isinstance(item, (dict, list)):
                    self._find_absolute_paths(item, found_paths, new_path)

    def run_validation_tests(
        self,
        config_path: Path = CONFIG_PATH,
        template_path: Path = TEMPLATE_PATH,
    ) -> bool:
        """Run validation tests on the config files. Defaults to
        default paths.

        Args:
            config_path: Path to the configuration file
            template_path: Path to the template file

        Returns:
            bool: True if all validation tests pass, False otherwise
        """
        # Check if the config file exists
        if not config_path.exists():
            self._print_section("FILE ERROR:")
            self._print_message(f"Error: {config_path} not found", prefix="  ")
            return False

        # Check if the template file exists
        if not template_path.exists():
            self._print_section("FILE ERROR:")
            self._print_message(
                f"Error: {template_path} not found", prefix="  "
            )
            return False

        # Validate both files against the schema
        config_valid = self.validate_config(str(config_path))
        template_valid = self.validate_config(str(template_path))

        # Check for absolute paths in the template
        template_paths_valid = self.check_absolute_paths(str(template_path))

        return config_valid and template_valid and template_paths_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate configuration files"
    )
    parser.add_argument(
        "file", nargs="?", default=None, help="Config file to validate"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress success messages"
    )
    args = parser.parse_args()

    if args.file:
        # Validate a specific file
        validator = ConfigValidator(quiet=args.quiet)
        success = validator.validate_config(
            args.file
        ) and validator.check_absolute_paths(args.file)
    else:
        # Run all validation tests using default paths
        validator = ConfigValidator()
        success = validator.run_validation_tests()

    sys.exit(0 if success else 1)
