"""
Configuration Template Generator Module

This module provides a class for generating configuration templates from a JSON schema.
The generator ensures templates are properly formatted and preserves existing values
when updating templates.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Union, Optional

from easyner.config.validator import ConfigValidator
from easyner.infrastructure.paths import (
    PROJECT_ROOT,
    SCHEMA_PATH,
)


class ConfigGenerator:
    """Generates configuration templates from a JSON schema.

    This class provides functionality to create template configuration files
    based on a schema, with options to preserve existing values and format
    the output.

    Attributes:
        schema_path: Path to the JSON schema file
        quiet: Whether to suppress standard output messages
        validator: The validator instance used to validate templates
    """

    def __init__(
        self, schema_path: Path = SCHEMA_PATH, quiet: bool = False
    ) -> None:
        """Initialize a ConfigGenerator instance.

        Args:
            schema_path: Path to the JSON schema file
            quiet: Whether to suppress standard output messages
        """
        self.schema_path: str = str(SCHEMA_PATH.relative_to(PROJECT_ROOT))
        self.quiet = quiet
        self.validator = ConfigValidator(schema_path, quiet)
        self._schema: Optional[Dict[str, Any]] = None

    @property
    def schema(self) -> Dict[str, Any]:
        """Get the schema, loading it if necessary.

        Returns:
            Dict containing the loaded schema
        """
        if self._schema is None:
            self._schema = self.validator.schema
        return self._schema

    def _print_message(self, message: str) -> None:
        """Print a message if quiet mode is not enabled.

        Args:
            message: The message to print
        """
        if not self.quiet:
            print(message)

    def format_with_prettier(self, file_path: Union[str, Path]) -> bool:
        """Format a JSON file using Prettier.

        Args:
            file_path: Path to the file to format

        Returns:
            bool: True if formatting succeeded, False otherwise
        """
        try:
            # Check if prettier is installed first
            check_result = subprocess.run(
                ["npx", "--no-install", "prettier", "--version"],
                capture_output=True,
                text=True,
                check=False,
            )

            if check_result.returncode != 0:
                self._print_message(
                    "  - Prettier not installed or not found in PATH"
                )
                self._print_message(
                    "  - To install prettier: npm install --global prettier"
                )
                self._print_message("  - Skipping formatting step")
                return False

            # Run prettier on the file
            file_path_str = str(file_path)
            result = subprocess.run(
                ["npx", "prettier", "--write", file_path_str],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Unknown error"
                self._print_message(
                    f"  - Failed to format with prettier: {error_msg}"
                )
                self._print_message("  - To fix prettier issues:")
                self._print_message(
                    "    1. Ensure Node.js is installed (https://nodejs.org/)"
                )
                self._print_message(
                    "    2. Install prettier: npm install --global prettier"
                )
                self._print_message(
                    "    3. Check that the file exists and is accessible"
                )
                self._print_message("  - Skipping formatting step")
                return False

            self._print_message(f"  - Successfully formatted: {file_path_str}")
            return True
        except Exception as e:
            self._print_message(f"  - Error running prettier: {e}")
            self._print_message("  - To fix prettier issues:")
            self._print_message(
                "    1. Ensure Node.js is installed (https://nodejs.org/)"
            )
            self._print_message(
                "    2. Install prettier: npm install --global prettier"
            )
            self._print_message(
                "    3. Make sure the npx command is available in your PATH"
            )
            self._print_message("  - Skipping formatting step")
            return False

    def create_default_value(
        self, schema_property: Dict[str, Any], property_path: str = ""
    ) -> Any:
        """Create a default value based on a schema property definition.

        Args:
            schema_property: The schema property definition
            property_path: The path of the property in the schema

        Returns:
            The default value for the property
        """
        if "default" in schema_property:  # Top-level default value
            return schema_property["default"]

        if "$ref" in schema_property:
            if schema_property["$ref"] == "#/definitions/path":
                return ""

        # Handle oneOf schema type
        # (used for file_limit which can be array or string)
        if "oneOf" in schema_property:
            # Special case for file_limit
            if property_path.endswith("file_limit"):
                return "ALL"

            # For other oneOf schemas, use the first option as default
            for option in schema_property["oneOf"]:
                if "type" in option:
                    schema_property["type"] = option["type"]
                    if option["type"] == "array" and "items" in option:
                        schema_property["items"] = option["items"]
                    break

        property_type = schema_property.get("type")

        if property_type == "string":
            return ""
        elif property_type == "integer":
            return schema_property.get("minimum", 0)
        elif property_type == "number":
            return schema_property.get("minimum", 0.0)
        elif property_type == "boolean":
            # Default ignore properties to True, all other booleans to False
            if property_path.startswith("ignore."):
                return True
            return False
        elif property_type == "array":
            if "items" in schema_property:
                if schema_property["items"].get("type") == "string":
                    return []
                elif schema_property["items"].get("type") == "integer":
                    # Special handling for range arrays
                    if (
                        property_path.endswith("subset_range")
                        or property_path.endswith("update_file_range")
                        or property_path.endswith("article_limit")
                    ):
                        return [0, 999999]  # Large range to represent "all"
                    # Use "ALL" for file_limit since it accepts either format
                    elif property_path.endswith("file_limit"):
                        return "ALL"
                    return [0, 0]  # Default for other integer arrays
                elif (
                    "$ref" in schema_property["items"]
                    and schema_property["items"]["$ref"]
                    == "#/definitions/path"
                ):
                    return []
                else:
                    return []
            else:
                return []
        elif property_type == "object":
            return {}
        else:
            return None

    def generate_template(
        self,
        template_path: Union[str, Path],
        skip_prettier: bool = False,
    ) -> bool:
        """Generate a template config file from the JSON schema.

        This method creates a config.template.json file based on the schema definition.
        If the template already exists, existing values will be preserved and only missing fields added.

        Args:
            template_path: Path where the template file will be saved
            skip_prettier: If True, skips formatting with prettier

        Returns:
            bool: True if template generation succeeded, False otherwise
        """
        # Print header
        self._print_message(
            "\n=== EasyNER Configuration Template Generator ==="
        )

        # Step 1: Generate the template
        template_path_str = str(template_path)
        self._print_message(
            f"Generating template from schema to: {template_path_str}"
        )

        # Check if template already exists, if so load existing values
        existing_template = {}
        if Path(template_path).exists():
            try:
                self._print_message(
                    f"Found existing template at {template_path_str}, preserving values..."
                )
                with open(template_path, "r") as f:
                    existing_template = json.load(f)
            except json.JSONDecodeError:
                self._print_message(
                    "Warning: Existing template could not be parsed, creating a new one"
                )
            except Exception as e:
                self._print_message(
                    f"Warning: Error reading existing template: {e}, creating a new one"
                )

        # Create a template based on the schema
        template = {}

        # Add schema reference to enable IDE Code validation and autocompletion
        template["$schema"] = existing_template.get(
            "$schema", self.schema_path
        )

        # Process all properties
        for prop_name, prop_schema in self.schema.get(
            "properties", {}
        ).items():
            # Skip schema metadata properties
            if prop_name.startswith("$"):
                continue

            # Create default value or use existing value
            if prop_name in existing_template:
                if (
                    prop_schema.get("type") == "object"
                    and "properties" in prop_schema
                ):
                    # Handle nested objects - preserve existing values but ensure all required fields exist
                    nested_obj = {}
                    existing_nested = existing_template.get(prop_name, {})

                    for nested_prop, nested_schema in prop_schema.get(
                        "properties", {}
                    ).items():
                        nested_path = f"{prop_name}.{nested_prop}"
                        if nested_prop in existing_nested:
                            nested_obj[nested_prop] = existing_nested[
                                nested_prop
                            ]
                        else:
                            # Add missing fields in the nested object
                            nested_obj[nested_prop] = (
                                self.create_default_value(
                                    nested_schema, nested_path
                                )
                            )
                    template[prop_name] = nested_obj
                else:
                    # For non-object types, preserve the existing value
                    template[prop_name] = existing_template[prop_name]
            elif prop_name == "CONFIG_VERSION":
                template[prop_name] = "1.0.0"
            elif prop_name == "_comments":
                template[prop_name] = {
                    "general": (
                        "This is a template configuration file. "
                        "Copy to config.json and update values as needed."
                    ),
                    "paths": (
                        "Provide actual file system paths appropriate "
                        "for your environment."
                    ),
                    "schema": (
                        f"Schema defined in {self.schema_path}. "
                        "Type hints should be provided in most editors."
                    ),
                }
            elif (
                prop_schema.get("type") == "object"
                and "properties" in prop_schema
            ):
                # Handle nested objects
                nested_obj = {}
                existing_nested = existing_template.get(prop_name, {})

                for nested_prop, nested_schema in prop_schema.get(
                    "properties", {}
                ).items():
                    nested_path = f"{prop_name}.{nested_prop}"
                    if nested_prop in existing_nested:
                        nested_obj[nested_prop] = existing_nested[nested_prop]
                    else:
                        nested_obj[nested_prop] = self.create_default_value(
                            nested_schema, nested_path
                        )
                template[prop_name] = nested_obj
            else:
                template[prop_name] = self.create_default_value(
                    prop_schema, prop_name
                )

        # Write the template to a file with nice formatting
        with open(template_path, "w") as f:
            json.dump(template, f, indent=2, sort_keys=False)

        # Step 2: Format with prettier if not skipped
        if not skip_prettier:
            self._print_message("Formatting with prettier...")
            success = self.format_with_prettier(template_path)
            if not success:
                self._print_message(
                    "Note: Template was generated but not formatted with prettier"
                )
                self._print_message(
                    "      The template will work fine without formatting"
                )

        # Step 3: Validate the template against the schema
        validation_result = self.validator.validate_config(template_path)

        # Final result
        if validation_result:
            self._print_message("✓ Template generated successfully")
            return True
        else:
            self._print_message(
                "✗ Template generation completed with validation issues"
            )
            return False

    def copy_template_to_config(
        self, template_path: Path, config_path: Path
    ) -> bool:
        """Copy a template file to a config file.

        Args:
            template_path: Path to the template file
            config_path: Path where the config file will be saved

        Returns:
            bool: True if the copy was successful, False otherwise
        """
        try:
            self._print_message(f"Creating config file at: {config_path}")

            # Read template content
            with open(template_path, "r") as src:
                template_content = src.read()

            # Write to config.json
            with open(config_path, "w") as dst:
                dst.write(template_content)

            self._print_message(f"✓ Config file created at: {config_path}")
            self._print_message(
                "  Edit this file to customize your configuration"
            )
            return True

        except Exception as e:
            self._print_message(f"Error creating config file: {e}")
            return False


if __name__ == "__main__":
    import argparse
    from easyner.infrastructure.paths import CONFIG_PATH, TEMPLATE_PATH

    parser = argparse.ArgumentParser(description="Generate a config template")
    parser.add_argument(
        "--output", default=TEMPLATE_PATH, help="Output template file path"
    )
    parser.add_argument(
        "--skip-prettier", action="store_true", help="Skip prettier formatting"
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Skip creating config.json even if it doesn't exist",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress output messages"
    )
    args = parser.parse_args()

    # Create generator
    generator = ConfigGenerator(quiet=args.quiet)

    # Generate template first
    result = generator.generate_template(args.output, args.skip_prettier)

    # Automatically create config.json from template if it doesn't exist
    if not args.no_config and result and not CONFIG_PATH.exists():
        generator.copy_template_to_config(Path(args.output), CONFIG_PATH)

    sys.exit(0 if result else 1)
