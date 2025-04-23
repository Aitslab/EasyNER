"""
Tests for schema validation issues with oneOf and anyOf structures.

These tests verify that the schema follows best practices with complex type definitions.
"""

import json
import pytest
from pathlib import Path

from easyner.infrastructure.paths import SCHEMA_PATH


def test_complex_types_have_defaults():
    """Test that all oneOf and anyOf structures in the schema have top-level default values."""

    # Load the schema
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    # Keep track of problematic fields
    issues = []

    # Recursively check all schema properties
    def check_properties(props, path=""):
        for prop_name, prop_schema in props.items():
            current_path = f"{path}.{prop_name}" if path else prop_name

            # Check if this property has oneOf or anyOf without a default
            if (
                "oneOf" in prop_schema or "anyOf" in prop_schema
            ) and "default" not in prop_schema:
                issues.append(
                    f"Field '{current_path}' has {'oneOf' if 'oneOf' in prop_schema else 'anyOf'} without a top-level default"
                )

            # Recurse into nested objects
            if "properties" in prop_schema and isinstance(
                prop_schema["properties"], dict
            ):
                check_properties(prop_schema["properties"], current_path)

    # Check all top-level properties
    check_properties(schema.get("properties", {}))

    # Report any issues found
    if issues:
        pytest.fail(f"Schema validation issues found:\n" + "\n".join(issues))
