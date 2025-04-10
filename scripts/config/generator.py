import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add the project root to sys.path to make scripts package importable
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent  # Go up two directories to reach project root
sys.path.insert(0, str(project_root))

# Import validation function from config package
from scripts.config.validator import load_schema, validate_config

# Import standard paths directly from infrastructure package
from scripts.infrastructure.paths import (
    DATA_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_SCHEMA_PATH,
    DEFAULT_TEMPLATE_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
)


def format_with_prettier(file_path: Union[str, Path]) -> bool:
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
            print("  - Prettier not installed or not found in PATH")
            print("  - To install prettier: npm install --global prettier")
            print("  - Skipping formatting step")
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
            print(f"  - Failed to format with prettier: {error_msg}")
            print("  - To fix prettier issues:")
            print("    1. Ensure Node.js is installed (https://nodejs.org/)")
            print("    2. Install prettier: npm install --global prettier")
            print("    3. Check that the file exists and is accessible")
            print("  - Skipping formatting step")
            return False

        print(f"  - Successfully formatted: {file_path_str}")
        return True
    except Exception as e:
        print(f"  - Error running prettier: {e}")
        print("  - To fix prettier issues:")
        print("    1. Ensure Node.js is installed (https://nodejs.org/)")
        print("    2. Install prettier: npm install --global prettier")
        print("    3. Make sure the npx command is available in your PATH")
        print("  - Skipping formatting step")
        return False


def create_default_value(
    schema_property: Dict[str, Any], property_path: str = ""
) -> Any:
    """Create a default value based on a schema property definition."""
    if "$ref" in schema_property:
        if schema_property["$ref"] == "#/definitions/path":
            return ""

    # Handle oneOf schema type (used for file_limit which can be array or string)
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
                and schema_property["items"]["$ref"] == "#/definitions/path"
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
    template_path: Union[str, Path],
    skip_prettier: bool = False,
) -> bool:
    """Generate a template config file from the JSON schema.

    This script creates a config.template.json file based on the schema definition.

    Args:
        template_path: Path where the template file will be saved
        skip_prettier: If True, skips formatting with prettier

    Returns:
        bool: True if template generation succeeded, False otherwise
    """
    # Print header
    print("\n=== EasyNER Configuration Template Generator ===")

    # Step 1: Generate the template
    template_path_str = str(template_path)
    print(f"Generating template from schema to: {template_path_str}")

    # Load the schema
    schema = load_schema()

    # Create a template based on the schema
    template = {}

    # Add schema reference to enable VS Code validation and autocompletion
    template["$schema"] = "./scripts/config/schema.json"

    # Process required properties first to maintain order
    required_props = schema.get("required", [])

    # Process all properties
    for prop_name, prop_schema in schema.get("properties", {}).items():
        # Skip schema metadata properties
        if prop_name.startswith("$"):
            continue

        # Create default value
        if prop_name == "CONFIG_VERSION":
            template[prop_name] = "1.0.0"
        elif prop_name == "_comments":
            template[prop_name] = {
                "general": "This is a template configuration file. Copy to config.json and update values as needed.",
                "paths": "Provide actual file system paths appropriate for your environment",
                "schema": "The structure of this file follows the schema defined in scripts/config/schema.json. Type hints should be provided in most editors.",
            }
        elif prop_schema.get("type") == "object" and "properties" in prop_schema:
            # Handle nested objects
            nested_obj = {}
            for nested_prop, nested_schema in prop_schema.get("properties", {}).items():
                nested_path = f"{prop_name}.{nested_prop}"
                nested_obj[nested_prop] = create_default_value(
                    nested_schema, nested_path
                )
            template[prop_name] = nested_obj
        else:
            template[prop_name] = create_default_value(prop_schema, prop_name)

    # Write the template to a file with nice formatting
    with open(template_path, "w") as f:
        json.dump(template, f, indent=2, sort_keys=False)

    # Step 2: Format with prettier if not skipped
    if not skip_prettier:
        print("Formatting with prettier...")
        success = format_with_prettier(template_path)
        if not success:
            print("Note: Template was generated but not formatted with prettier")
            print("      The template will work fine without formatting")

    # Step 3: Validate the template against the schema
    # Use quiet=True to suppress warnings about empty path values
    validation_result = validate_config(template_path, quiet=True)

    # Final result
    if validation_result:
        print(f"✓ Template generated successfully")
        return True
    else:
        print(f"✗ Template generation completed with validation issues")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a config template")
    parser.add_argument(
        "--output", default=str(DEFAULT_TEMPLATE_PATH), help="Output template file path"
    )
    parser.add_argument(
        "--skip-prettier", action="store_true", help="Skip prettier formatting"
    )
    args = parser.parse_args()

    result = generate_template(args.output, args.skip_prettier)
    sys.exit(0 if result else 1)
