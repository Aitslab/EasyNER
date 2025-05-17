#!/usr/bin/env python3
"""
JSON File Structure Validator for EasyNer

This utility examines JSON files to detect and diagnose structural issues
that might cause processing errors in the ArticleCounter class.
"""

import json
import os
import sys
from glob import glob
import argparse
from pathlib import Path
from tqdm import tqdm

# Resolve path to parent
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
# Import necessary modules


# Handle compatibility with different Python versions
try:
    # Python 3.5+
    JSONDecodeError = json.JSONDecodeError
except AttributeError:
    # Python < 3.5
    JSONDecodeError = ValueError


class JsonFileValidator:
    """Validate JSON files and provide detailed diagnostics about their structure."""

    def __init__(self, json_dir=None):
        """Initialize with directory path containing JSON files."""
        self.json_dir = json_dir
        self.json_files = []
        if json_dir:
            self.json_files = sorted(glob(os.path.join(json_dir, "*.json")))
            print(f"Found {len(self.json_files)} JSON files in {json_dir}")

        # Error types and statistics
        self.error_types = {
            "not_found": 0,
            "empty": 0,
            "decode_error": 0,
            "not_dict": 0,
            "no_title_field": 0,
            "valid": 0,
        }

        # For aggregating summary statistics
        self.total_docs = 0
        self.docs_with_title = 0

    def validate_file(self, file_path, show_details=False):
        """Validate a single JSON file and return diagnostic information."""
        result = {
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "size_mb": 0,
            "status": "unknown",
            "error": None,
            "structure": {},
            "sample": None,
        }

        # Check if file exists
        if not os.path.exists(file_path):
            result["status"] = "not_found"
            result["error"] = f"File not found: {file_path}"
            self.error_types["not_found"] += 1
            return result

        # Check file size
        file_size = os.path.getsize(file_path)
        result["size_mb"] = file_size / (1024 * 1024)  # Convert to MB

        if file_size == 0:
            result["status"] = "empty"
            result["error"] = "File is empty"
            self.error_types["empty"] += 1
            return result

        # Try to parse JSON
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # First check the initial character to determine structure
                first_char = f.read(1)
                f.seek(0)  # Reset to file beginning

                # Read full content for small files, or just enough for large files
                if file_size < 10 * 1024 * 1024:  # 10MB
                    data = json.load(f)
                else:
                    # For large files, just check structure
                    data = None
                    if first_char == "{":
                        result["structure"]["type"] = "object"
                    elif first_char == "[":
                        result["structure"]["type"] = "array"
                    else:
                        result["status"] = "decode_error"
                        result["error"] = (
                            f"File doesn't start with valid JSON (starts with '{first_char}')"
                        )
                        self.error_types["decode_error"] += 1
                        return result

                    # Read a sample of the beginning
                    f.seek(0)
                    sample = f.read(10000)  # First X kB
                    result["sample"] = sample

                    # Try to parse just the beginning
                    try:
                        json.loads(sample + "]" if first_char == "[" else sample + "}")
                    except JSONDecodeError as je:
                        result["status"] = "decode_error"
                        result["error"] = f"JSON parse error in sample: {str(je)}"
                        self.error_types["decode_error"] += 1
                        return result

        except JSONDecodeError as je:
            result["status"] = "decode_error"
            result["error"] = f"JSON decode error: {str(je)}"
            self.error_types["decode_error"] += 1
            return result
        except Exception as e:
            result["status"] = "error"
            result["error"] = f"Error reading file: {str(e)}"
            self.error_types["decode_error"] += 1
            return result

        # If we've loaded the full data, analyze structure
        if data is not None:
            # Check if data is a dictionary (object)
            if isinstance(data, dict):
                result["structure"]["type"] = "object"
                result["structure"]["keys_count"] = len(data)
                self.total_docs += len(data)

                # Check if doc entries have titles
                has_title = False
                for key, value in list(data.items())[:5]:  # Check first 5 entries
                    if isinstance(value, dict) and "title" in value:
                        has_title = True
                        self.docs_with_title += 1
                        if show_details:
                            result["sample"] = {key: value}
                        break

                if not has_title:
                    result["status"] = "no_title_field"
                    result["error"] = "No 'title' field found in document entries"
                    self.error_types["no_title_field"] += 1
                    if show_details:
                        # Include a sample entry for debugging
                        sample_key = next(iter(data), None)
                        if sample_key:
                            result["sample"] = {sample_key: data[sample_key]}
                    return result

                result["status"] = "valid"
                self.error_types["valid"] += 1

            else:  # Not a dict
                result["structure"]["type"] = (
                    "array" if isinstance(data, list) else type(data).__name__
                )
                result["status"] = "not_dict"
                result["error"] = (
                    f"Expected JSON object/dict, got {type(data).__name__}"
                )
                self.error_types["not_dict"] += 1
                return result

        return result

    def validate_all_files(self, show_progress=True, show_details=False):
        """Validate all JSON files in the directory."""
        results = []

        if show_progress:
            files_iter = tqdm(self.json_files, desc="Validating JSON files")
        else:
            files_iter = self.json_files

        for json_file in files_iter:
            result = self.validate_file(json_file, show_details)
            results.append(result)

        return results

    def print_summary(self):
        """Print summary of validation results."""
        print("\nValidation Summary:")
        print(f"Total files examined: {len(self.json_files)}")
        print(f"Valid files: {self.error_types['valid']}")
        print(
            f"Files with errors: {sum(count for error, count in self.error_types.items() if error != 'valid')}"
        )
        print("\nError types:")
        for error_type, count in self.error_types.items():
            if error_type != "valid":
                print(f"  {error_type}: {count}")

        print(f"\nTotal documents found: {self.total_docs}")
        print(f"Documents with 'title' field: {self.docs_with_title}")

    def fix_json_file(self, input_path, output_path=None):
        """
        Attempt to fix a JSON file by:
        1. Handling common formatting issues
        2. Converting arrays to objects if needed
        3. Adding missing title fields

        Returns tuple: (success, message)
        """
        if not output_path:
            # Create a backup of the original
            output_path = input_path + ".fixed"

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                first_char = f.read(1)
                f.seek(0)

                # Try to load the JSON data
                try:
                    data = json.load(f)
                except JSONDecodeError as je:
                    # For JSON decode errors, try to fix common issues
                    f.seek(0)
                    content = f.read()

                    # Try some common fixes:
                    # 1. Remove trailing commas
                    content = content.replace(",]", "]").replace(",}", "}")

                    # 2. Fix unquoted keys
                    # This is a simplified fix and may not handle all cases
                    import re

                    content = re.sub(
                        r"([{,])\s*([a-zA-Z0-9_]+)\s*:", r'\1"\2":', content
                    )

                    try:
                        data = json.loads(content)
                    except JSONDecodeError:
                        return False, f"Could not fix JSON decode error: {str(je)}"

            # Convert array to object if needed
            if isinstance(data, list):
                converted_data = {}
                for i, item in enumerate(data):
                    converted_data[str(i)] = item
                data = converted_data

            # Ensure all entries have a title field
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict) and "title" not in value:
                        # Add a placeholder title using the key
                        data[key]["title"] = f"Untitled document {key}"

            # Write the fixed data
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            return True, f"Fixed JSON saved to {output_path}"

        except Exception as e:
            return False, f"Error fixing JSON file: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Validate JSON files for EasyNer")
    parser.add_argument(
        "path", help="Directory containing JSON files or a specific JSON file"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix problematic JSON files"
    )
    parser.add_argument(
        "--details", action="store_true", help="Show detailed information for each file"
    )
    args = parser.parse_args()

    path = args.path

    if os.path.isdir(path):
        validator = JsonFileValidator(path)
        results = validator.validate_all_files(show_details=args.details)
        validator.print_summary()

        # Attempt to fix problematic files if requested
        if args.fix:
            print("\nAttempting to fix problematic files:")
            fixed_count = 0
            for result in results:
                if result["status"] != "valid" and result["status"] != "not_found":
                    success, message = validator.fix_json_file(result["file_path"])
                    if success:
                        fixed_count += 1
                        print(f"✓ Fixed: {result['filename']} - {message}")
                    else:
                        print(f"✗ Failed to fix: {result['filename']} - {message}")
            print(
                f"\nFixed {fixed_count} out of {len(results) - validator.error_types['valid']} problematic files"
            )

    elif os.path.isfile(path):
        validator = JsonFileValidator()
        result = validator.validate_file(path, show_details=True)
        print(f"\nFile: {os.path.basename(path)}")
        print(f"Status: {result['status']}")
        if result["error"]:
            print(f"Error: {result['error']}")
        print(f"Structure: {result['structure']}")

        if args.fix and result["status"] != "valid" and result["status"] != "not_found":
            success, message = validator.fix_json_file(path)
            if success:
                print(f"Fix result: ✓ {message}")
            else:
                print(f"Fix result: ✗ {message}")

    else:
        print(f"Error: '{path}' is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
