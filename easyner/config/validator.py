import json
import jsonschema
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Iterator

import jsonschema.exceptions

from easyner.infrastructure.paths import (
    CONFIG_PATH,
    TEMPLATE_PATH,
    SCHEMA_PATH,
)


def load_schema() -> Dict[str, Any]:
    """Load the JSON schema for config validation.

    Returns:
        dict: The loaded schema
    """
    try:
        with open(SCHEMA_PATH, "r") as f:
            schema = json.load(f)
        return schema
    except FileNotFoundError:
        print(f"Error: Schema file not found at {SCHEMA_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in schema file: {e}")
        sys.exit(1)


def validate_config(
    config_file: Union[str, Path], quiet: bool = False
) -> bool:
    """Validates the config file against the schema.

    Args:
        config_file: Path to the config file to validate
        quiet: If True, suppresses success messages and warnings

    Returns:
        bool: True if validation succeeds, False otherwise
    """
    try:
        config_file_path = Path(config_file)
        config_file_name = config_file_path.name

        if not quiet:
            print(f"\n{'='*80}")
            print(f"VALIDATING: {config_file_name}")
            print(f"{'='*80}")

        with open(config_file, "r") as f:
            config_data = json.load(f)

        schema = load_schema()

        # Pre-validation check for non-string path values
        validation_errors: List[str] = []
        warnings: List[str] = []
        empty_paths: List[str] = []

        check_path_types(
            config_data, schema, validation_errors, warnings, empty_paths
        )

        # Check if the $schema field is present in the config
        if "$schema" not in config_data:
            warnings.append(
                f"$schema field is missing. It should be set to \n"
                f"{SCHEMA_PATH} to provide type hints"
            )
        elif config_data["$schema"] != str(SCHEMA_PATH):
            warnings.append(
                f"$schema has unexpected value: "
                f"'{config_data['$schema']}'. "
                f"Expected: '{SCHEMA_PATH}'"
            )

        if validation_errors:
            if not quiet:
                print(f"\n{'-'*80}")
                print("VALIDATION ERRORS:")
                print(f"{'-'*80}")
            for error in validation_errors:
                print(f"  {error}")
            return False

        # Print warnings but don't fail validation for them
        if warnings and not quiet:
            print(f"\n{'-'*80}")
            print("VALIDATION WARNINGS:")
            print(f"{'-'*80}")
            for warning in warnings:
                print(f"  {warning}")

        # Print empty paths in a more concise way
        if empty_paths and not quiet:
            print(f"\n{'-'*80}")
            print("INFO: The following path fields are left empty:")
            print(f"{'-'*80}")
            for path in empty_paths:
                print(f"  • {path}")

        jsonschema.validate(instance=config_data, schema=schema)
        if not quiet:
            print(f"\n{'-'*80}")
            print("RESULT: Configuration validation successful!")
            print(f"{'-'*80}\n")
        return True

    except jsonschema.exceptions.ValidationError as e:
        # Create a more informative error message
        error_path = ".".join([str(p) for p in e.path]) if e.path else "root"

        if not quiet:
            print(f"\n{'-'*80}")
            print("VALIDATION ERROR:")
            print(f"{'-'*80}")

        if "pattern" in str(e) and "does not match" in str(e):
            # This is a pattern validation error, likely for a path
            if "config_data" in locals():
                error_value = get_value_at_path(config_data, e.path)
                error_type = type(error_value).__name__
            else:
                print("Unexpected error: config_data not available")
                return False

            if isinstance(error_value, (int, float, bool)):
                # The value is not even a string
                print(
                    f"  Path field '{error_path}' must be a string, "
                    f"but got {error_type}: {error_value}"
                )
            elif error_value == "":
                # Empty strings are allowed with an info message
                if not quiet:
                    print(f"\n{'-'*80}")
                    print("INFO:")
                    print(f"{'-'*80}")
                    print(f"  Path field '{error_path}' is empty")
                # Return True to allow empty paths
                return True
            else:
                # The value is a string but doesn't match the pattern
                print(
                    f"  Path field '{error_path}' "
                    f"contains an invalid path format:  {error_value}"
                )
        else:
            # For other validation errors
            print(f"  Error at '{error_path}': {e.message}")

        return False
    except FileNotFoundError as e:
        if not quiet:
            print(f"\n{'-'*80}")
            print("FILE ERROR:")
            print(f"{'-'*80}")
        print(f"  File not found: {e}")
        return False
    except json.JSONDecodeError as e:
        if not quiet:
            print(f"\n{'-'*80}")
            print("JSON ERROR:")
            print(f"{'-'*80}")
        print(f"  JSON decode error: {e}")
        return False


def check_path_types(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    errors: List[str],
    warnings: List[str],
    empty_paths: List[str],
    path: Optional[List[str]] = None,
) -> None:
    """Recursively check for path fields with non-string values.

    Args:
        data: The data to check
        schema: The schema to check against
        errors: List to collect error messages
        warnings: List to collect warning messages
        empty_paths: List to collect empty path fields
        path: Current path in the data structure
    """
    if path is None:
        path = []

    # If this is a reference to the path definition, check the value
    if isinstance(data, dict) and isinstance(schema, dict):
        # Check each property in this object
        for key, value in data.items():
            current_path = path + [key]
            path_str = ".".join(str(p) for p in current_path)

            # Check if this property exists in the schema
            if key in schema.get("properties", {}):
                prop_schema = schema["properties"][key]

                # If this is a path reference
                if prop_schema.get("$ref") == "#/definitions/path":
                    if not isinstance(value, str):
                        errors.append(
                            f"Path field '{path_str}' must be a string, "
                            f"but got {type(value).__name__}: {value}"
                        )
                    elif value == "":
                        empty_paths.append(path_str)
                # If this is an object or array, recurse
                elif (
                    isinstance(value, dict)
                    and prop_schema.get("type") == "object"
                ):
                    check_path_types(
                        value,
                        prop_schema,
                        errors,
                        warnings,
                        empty_paths,
                        current_path,
                    )
                elif (
                    isinstance(value, list)
                    and prop_schema.get("type") == "array"
                ):
                    # If the array items are paths
                    if (
                        prop_schema.get("items", {}).get("$ref")
                        == "#/definitions/path"
                    ):
                        for i, item in enumerate(value):
                            if not isinstance(item, str):
                                item_path = path_str + f"[{i}]"
                                errors.append(
                                    f"Path field '{item_path}' must be a string, "
                                    f" but got {type(item).__name__}: {item}"
                                )
                            elif item == "":
                                item_path = path_str + f"[{i}]"
                                empty_paths.append(item_path)
                    # Otherwise, if it's an array of objects or arrays, recurse
                    else:
                        for i, item in enumerate(value):
                            if isinstance(item, (dict, list)):
                                check_path_types(
                                    item,
                                    prop_schema.get("items", {}),
                                    errors,
                                    warnings,
                                    empty_paths,
                                    current_path + [i],
                                )


def get_value_at_path(data: Any, path: Iterator[Any]) -> Any:
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
    config_file: Union[str, Path],
    section_keys: Optional[List[str]] = None,
    quiet: bool = False,
) -> bool:
    """Check for absolute paths in a config file.

    Args:
        config_file: Path to the config file to check
        section_keys: Optional list of section keys to check specifically
        quiet: If True, suppresses messages

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

            def find_abs_paths(obj: Any, path: str = "") -> None:
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == "_comments" or key == "CONFIG_VERSION":
                            continue
                        new_path = f"{path}.{key}" if path else key
                        if isinstance(value, str) and (
                            Path(value).is_absolute()
                            or re.search(r"[A-Z]:/.*", value)
                        ):
                            if value != "PATH_PLACEHOLDER":
                                found_paths.append((new_path, value))
                        elif isinstance(value, (dict, list)):
                            find_abs_paths(value, new_path)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        new_path = f"{path}[{i}]"
                        if isinstance(item, str) and (
                            Path(item).is_absolute()
                            or re.search(r"[A-Z]:/.*", item)
                        ):
                            if item != "PATH_PLACEHOLDER":
                                found_paths.append((new_path, item))
                        elif isinstance(item, (dict, list)):
                            find_abs_paths(item, new_path)

            find_abs_paths(config_data)

            if found_paths:
                if not quiet:
                    print(f"\n{'-'*80}")
                    print("ABSOLUTE PATHS ERROR:")
                    print(f"{'-'*80}")
                    print("  Absolute paths found in template file:")
                    for path, value in found_paths:
                        print(f"    • {path}: {value}")
                return False
            return True
        return True
    except Exception as e:
        if not quiet:
            print(f"\n{'-'*80}")
            print("PATH CHECK ERROR:")
            print(f"{'-'*80}")
        print(f"  Error checking paths: {e}")
        return False


def run_validation_tests() -> bool:
    """Run validation tests on the config files.

    Returns:
        bool: True if all validation tests pass, False otherwise
    """
    # Use the standard path constants from the config package
    config_file = CONFIG_PATH
    template_file = TEMPLATE_PATH

    # Check if the config file exists
    if not config_file.exists():
        print(f"\n{'-'*80}")
        print("FILE ERROR:")
        print(f"{'-'*80}")
        print(f"  Error: {config_file} not found")
        return False

    # Check if the template file exists
    if not template_file.exists():
        print(f"\n{'-'*80}")
        print("FILE ERROR:")
        print(f"{'-'*80}")
        print(f"  Error: {template_file} not found")
        return False

    # Validate both files against the schema
    config_valid = validate_config(str(config_file))
    template_valid = validate_config(str(template_file))

    # Check for absolute paths in the template
    template_paths_valid = check_absolute_paths(str(template_file))

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
        success = validate_config(
            args.file, args.quiet
        ) and check_absolute_paths(args.file, quiet=args.quiet)
    else:
        # Run all validation tests
        success = run_validation_tests()

    sys.exit(0 if success else 1)
