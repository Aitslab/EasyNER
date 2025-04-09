#!/usr/bin/env python3
"""
Enhanced JSON Diagnostics and Repair Tool for EasyNer

This utility provides detailed diagnostics for JSON files and offers
more robust repair options for common JSON formatting issues.
"""

import argparse
import datetime
import json
import os
import re
from glob import glob

from tqdm import tqdm


class JsonDiagnostic:
    """Advanced JSON file analysis and repair tool."""

    def __init__(self, json_dir=None):
        """Initialize with directory path containing JSON files."""
        self.json_dir = json_dir
        self.json_files = []
        if json_dir:
            self.json_files = sorted(glob(os.path.join(json_dir, "*.json")))
            print(f"Found {len(self.json_files)} JSON files in {json_dir}")

        # Error categories and statistics
        self.error_types = {
            "not_found": 0,
            "empty": 0,
            "decode_error": 0,
            "malformed_brace": 0,
            "unquoted_keys": 0,
            "trailing_comma": 0,
            "control_chars": 0,
            "not_dict": 0,
            "no_title_field": 0,
            "encoding_error": 0,
            "valid": 0,
        }

        # For detailed error analysis
        self.error_samples = {
            error_type: [] for error_type in self.error_types
        }

        # For aggregating summary statistics
        self.total_docs = 0
        self.docs_with_title = 0

    def analyze_file(self, file_path, detailed=False):
        """Perform detailed analysis of JSON file format and structure."""
        result = {
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "size_mb": 0,
            "status": "unknown",
            "error_type": None,
            "error_details": None,
            "line_number": None,
            "position": None,
            "context": None,
            "structure": {},
            "sample": None,
        }

        # Check if file exists
        if not os.path.exists(file_path):
            result["status"] = "error"
            result["error_type"] = "not_found"
            result["error_details"] = f"File not found: {file_path}"
            self.error_types["not_found"] += 1
            return result

        # Check file size
        file_size = os.path.getsize(file_path)
        result["size_mb"] = file_size / (1024 * 1024)  # Convert to MB

        if file_size == 0:
            result["status"] = "error"
            result["error_type"] = "empty"
            result["error_details"] = "File is empty"
            self.error_types["empty"] += 1
            self._add_error_sample("empty", result)
            return result

        # Try to parse JSON with more detailed error handling
        try:
            # First try simple textual analysis
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    first_char = f.read(1)
                    f.seek(0)

                    # Sample the first few KB for analysis
                    sample_size = min(10000, file_size)
                    sample = f.read(sample_size)
                    result["sample"] = (
                        sample[:200] + "..." if len(sample) > 200 else sample
                    )

                    # Check for common issues before parsing
                    if self._has_control_chars(sample):
                        result["status"] = "error"
                        result["error_type"] = "control_chars"
                        result["error_details"] = (
                            "File contains invalid control characters"
                        )
                        self.error_types["control_chars"] += 1
                        self._add_error_sample("control_chars", result)
                        return result

                    if first_char not in "{[\"'0123456789":
                        result["status"] = "error"
                        result["error_type"] = "malformed_brace"
                        result["error_details"] = (
                            f"File doesn't start with valid JSON (starts with '{first_char}')"
                        )
                        self.error_types["malformed_brace"] += 1
                        self._add_error_sample("malformed_brace", result)
                        return result

                    # Try to do full parse for smaller files
                    if file_size < 5 * 1024 * 1024:  # 5MB
                        f.seek(0)
                        data = json.load(f)
                        self._analyze_structure(data, result)
                    else:
                        # For large files, just check the basic structure
                        result["structure"]["type"] = (
                            "object" if first_char == "{" else "array"
                        )
                        result["status"] = "valid"
                        self.error_types["valid"] += 1

                except UnicodeDecodeError as ue:
                    result["status"] = "error"
                    result["error_type"] = "encoding_error"
                    result["error_details"] = (
                        f"File has encoding issues: {str(ue)}"
                    )
                    self.error_types["encoding_error"] += 1
                    self._add_error_sample("encoding_error", result)

                except json.JSONDecodeError as je:
                    # Get detailed information about the JSON error
                    result["status"] = "error"
                    result["error_type"] = "decode_error"
                    result["error_details"] = str(je)
                    result["line_number"] = getattr(je, "lineno", None)
                    result["position"] = getattr(je, "pos", None)

                    # Extract more context around the error if possible
                    if (
                        detailed
                        and result["line_number"]
                        and result["position"]
                    ):
                        with open(
                            file_path, "r", encoding="utf-8", errors="replace"
                        ) as cf:
                            lines = cf.readlines()
                            if result["line_number"] <= len(lines):
                                line = lines[result["line_number"] - 1]
                                result["context"] = line.rstrip()
                                # Mark the position
                                pos = result["position"]
                                if pos < len(line):
                                    pointer = " " * pos + "^"
                                    result["context"] += f"\n{pointer}"

                    # Categorize the JSON error more specifically
                    error_msg = str(je).lower()
                    if (
                        "expecting property name" in error_msg
                        or "keys must be quoted" in error_msg
                    ):
                        result["error_type"] = "unquoted_keys"
                        self.error_types["unquoted_keys"] += 1
                        self._add_error_sample("unquoted_keys", result)
                    elif "trailing comma" in error_msg:
                        result["error_type"] = "trailing_comma"
                        self.error_types["trailing_comma"] += 1
                        self._add_error_sample("trailing_comma", result)
                    elif (
                        "unexpected character" in error_msg
                        or "control character" in error_msg
                    ):
                        result["error_type"] = "control_chars"
                        self.error_types["control_chars"] += 1
                        self._add_error_sample("control_chars", result)
                    elif "expecting" in error_msg and (
                        "}" in error_msg or "]" in error_msg
                    ):
                        result["error_type"] = "malformed_brace"
                        self.error_types["malformed_brace"] += 1
                        self._add_error_sample("malformed_brace", result)
                    else:
                        self.error_types["decode_error"] += 1
                        self._add_error_sample("decode_error", result)

        except Exception as e:
            result["status"] = "error"
            result["error_type"] = "decode_error"
            result["error_details"] = f"Error analyzing file: {str(e)}"
            self.error_types["decode_error"] += 1
            self._add_error_sample("decode_error", result)

        return result

    def _add_error_sample(self, error_type, result):
        """Add sample to error_samples, limiting to 5 samples per error type."""
        if len(self.error_samples[error_type]) < 5:
            self.error_samples[error_type].append(result)

    def _has_control_chars(self, text):
        """Check if text contains illegal JSON control characters."""
        control_chars = set(range(0, 32)) - {
            9,
            10,
            13,
        }  # Tab, LF, CR are allowed
        return any(ord(char) in control_chars for char in text)

    def _analyze_structure(self, data, result):
        """Analyze JSON structure once parsed successfully."""
        if isinstance(data, dict):
            result["structure"]["type"] = "object"
            result["structure"]["keys_count"] = len(data)
            self.total_docs += len(data)

            # Check if doc entries have titles
            has_title = False
            for key, value in list(data.items())[:5]:
                if isinstance(value, dict) and "title" in value:
                    has_title = True
                    self.docs_with_title += 1
                    result["sample"] = {key: value}
                    break

            if not has_title:
                result["status"] = "error"
                result["error_type"] = "no_title_field"
                result["error_details"] = (
                    "No 'title' field found in document entries"
                )
                self.error_types["no_title_field"] += 1
                self._add_error_sample("no_title_field", result)
            else:
                result["status"] = "valid"
                self.error_types["valid"] += 1

        else:
            result["structure"]["type"] = (
                "array" if isinstance(data, list) else type(data).__name__
            )
            result["status"] = "error"
            result["error_type"] = "not_dict"
            result["error_details"] = (
                f"Expected JSON object/dict, got {type(data).__name__}"
            )
            self.error_types["not_dict"] += 1
            self._add_error_sample("not_dict", result)

    def analyze_all_files(self, show_progress=True, detailed=False):
        """Analyze all JSON files in the directory."""
        results = []

        if show_progress:
            files_iter = tqdm(self.json_files, desc="Analyzing JSON files")
        else:
            files_iter = self.json_files

        for json_file in files_iter:
            result = self.analyze_file(json_file, detailed=detailed)
            results.append(result)

        return results

    def print_summary(self):
        """Print summary of analysis results with examples."""
        print("\nAnalysis Summary:")
        print(f"Total files examined: {len(self.json_files)}")
        print(f"Valid files: {self.error_types['valid']}")
        print(
            f"Files with errors: {sum(count for error, count in self.error_types.items() if error != 'valid')}"
        )
        print("\nError types:")
        for error_type, count in sorted(
            self.error_types.items(), key=lambda x: x[1], reverse=True
        ):
            if error_type != "valid" and count > 0:
                print(f"  {error_type}: {count}")

    def print_error_samples(self):
        """Print samples of each error type to help diagnose issues."""
        print("\nSample errors by type:")
        for error_type, samples in sorted(self.error_samples.items()):
            if error_type != "valid" and samples:
                print(f"\n=== {error_type.upper()} ===")
                sample = samples[0]  # Show the first sample of each error type
                print(f"File: {sample['filename']}")
                print(f"Error: {sample['error_details']}")

                if sample.get("line_number"):
                    print(
                        f"Line {sample['line_number']}, Position {sample['position']}"
                    )

                if sample.get("context"):
                    print("Context:")
                    print(sample["context"])

                if sample.get("sample"):
                    print("Sample content:")
                    print(sample["sample"])
                print("-" * 60)

    def repair_file(self, file_path, output_path=None, repair_strategy="auto"):
        """
        Attempt to repair a JSON file using the specified strategy.

        Args:
            file_path (str): Path to the file to repair
            output_path (str, optional): Path for the repaired file. If None, uses original path + .fixed
            repair_strategy (str): Strategy for repair ('auto', 'minimal', 'aggressive')

        Returns:
            tuple: (success, message, error_type)
        """
        if not output_path:
            output_path = f"{file_path}.fixed"

        # First analyze the file to determine the issue
        analysis = self.analyze_file(file_path, detailed=True)
        error_type = analysis.get("error_type")

        if not error_type or error_type == "valid":
            return True, "File is already valid JSON, no repair needed.", None

        if error_type == "empty":
            # Create a minimal valid JSON object
            with open(output_path, "w", encoding="utf-8") as f:
                f.write('{"empty_file_fixed": true}')
            return (
                True,
                "Empty file replaced with minimal valid JSON object.",
                "empty",
            )

        if error_type == "not_found":
            return (
                False,
                f"Cannot repair: {analysis['error_details']}",
                "not_found",
            )

        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Apply repairs based on error type and strategy
            if error_type == "control_chars":
                content = self._remove_control_chars(content)

            if error_type in [
                "unquoted_keys",
                "trailing_comma",
                "malformed_brace",
            ] or repair_strategy in ["auto", "aggressive"]:
                content = self._fix_common_json_issues(content)

            # For more aggressive repair, try JSONLint-style reconstruction
            if repair_strategy == "aggressive":
                content = self._aggressive_json_repair(content)

            # Write the repaired content
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Validate the repaired file
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    json.load(f)
                return (
                    True,
                    f"Successfully repaired file. Saved to {output_path}",
                    error_type,
                )
            except Exception as e:
                return (
                    False,
                    f"Repair attempt failed validation: {str(e)}",
                    error_type,
                )

        except Exception as e:
            return False, f"Error during repair: {str(e)}", error_type

    def _remove_control_chars(self, content):
        """Remove illegal control characters from JSON content."""
        return "".join(
            char for char in content if ord(char) >= 32 or char in "\t\r\n"
        )

    def _fix_common_json_issues(self, content):
        """Fix common JSON syntax issues."""
        # Remove trailing commas in arrays and objects
        content = re.sub(r",\s*}", "}", content)
        content = re.sub(r",\s*]", "]", content)

        # Fix unquoted keys
        content = re.sub(r"([{,])\s*([a-zA-Z0-9_]+)\s*:", r'\1"\2":', content)

        # Fix single-quoted strings (replace with double quotes)
        # This is more complex as we need to handle escaped quotes
        # Simple version that works for basic cases:
        content = re.sub(r'\'([^\'"]*)\'(\s*:)', r'"\1"\2', content)

        # Ensure proper object/array closure
        # (This is a simple fix; might not work for complex nested structures)
        open_braces = content.count("{")
        close_braces = content.count("}")
        if open_braces > close_braces:
            content += "}" * (open_braces - close_braces)

        open_brackets = content.count("[")
        close_brackets = content.count("]")
        if open_brackets > close_brackets:
            content += "]" * (open_brackets - close_brackets)

        return content

    def _aggressive_json_repair(self, content):
        """
        Attempt more aggressive JSON repair for severely malformed files.
        This is a last resort that tries to reconstruct valid JSON.
        """
        # Try to extract what looks like JSON data
        if content.lstrip().startswith("{"):
            # It's trying to be an object
            if "{" not in content[1:]:
                # Looks like a single-level object with issues
                # Extract key-value pairs that look valid
                pairs = []
                for match in re.finditer(
                    r'"([^"]+)"\s*:\s*("[^"]*"|[0-9]+|true|false|null|\{[^}]*\}|\[[^\]]*\])',
                    content,
                ):
                    pairs.append(f'"{match.group(1)}": {match.group(2)}')

                if pairs:
                    return "{" + ", ".join(pairs) + "}"

        # If we can't intelligently repair it, create a minimal valid JSON object
        # with the original content as a string property
        safe_content = content.replace('"', '\\"').replace("\n", "\\n")
        return f'{{"original_malformed_content": "{safe_content[:1000]}..."}}'

    def repair_all_files(
        self, output_dir=None, repair_strategy="auto", show_progress=True
    ):
        """
        Attempt to repair all files with errors.

        Args:
            output_dir (str, optional): Directory for repaired files. If None, uses original filename + .fixed
            repair_strategy (str): Strategy for repair ('auto', 'minimal', 'aggressive')

        Returns:
            dict: Summary of repair operations
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        repair_results = {
            "total_files": len(self.json_files),
            "repaired": 0,
            "failed": 0,
            "already_valid": 0,
            "by_error_type": {},
        }

        # First analyze all files if not already done
        if sum(self.error_types.values()) == 0:
            self.analyze_all_files(show_progress=show_progress)

        if show_progress:
            files_iter = tqdm(self.json_files, desc="Repairing JSON files")
        else:
            files_iter = self.json_files

        for json_file in files_iter:
            file_name = os.path.basename(json_file)

            if output_dir:
                output_path = os.path.join(output_dir, file_name)
            else:
                output_path = f"{json_file}.fixed"

            success, message, error_type = self.repair_file(
                json_file, output_path, repair_strategy
            )

            if not error_type:
                repair_results["already_valid"] += 1
                continue

            # Track results by error type
            if error_type not in repair_results["by_error_type"]:
                repair_results["by_error_type"][error_type] = {
                    "total": 0,
                    "fixed": 0,
                    "failed": 0,
                }

            repair_results["by_error_type"][error_type]["total"] += 1

            if success:
                repair_results["repaired"] += 1
                repair_results["by_error_type"][error_type]["fixed"] += 1
            else:
                repair_results["failed"] += 1
                repair_results["by_error_type"][error_type]["failed"] += 1

        return repair_results

    def print_repair_summary(self, repair_results):
        """Print a summary of the repair operations."""
        print("\nRepair Summary:")
        print(f"Total files processed: {repair_results['total_files']}")
        print(f"Already valid: {repair_results['already_valid']}")
        print(f"Successfully repaired: {repair_results['repaired']}")
        print(f"Failed to repair: {repair_results['failed']}")

        print("\nResults by error type:")
        for error_type, stats in repair_results["by_error_type"].items():
            print(
                f"  {error_type}: {stats['fixed']} fixed, {stats['failed']} failed"
            )

    def save_results(
        self,
        results,
        result_type="analysis",
        output_dir="results/json_diagnostics",
    ):
        """
        Save analysis or repair results to files in the specified directory.

        Args:
            results: Results data to save (analysis results or repair summary)
            result_type: Type of results ('analysis' or 'repair')
            output_dir: Directory to save results to

        Returns:
            str: Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filenames based on result type
        if result_type == "analysis":
            detailed_file = os.path.join(
                output_dir, f"json_analysis_detailed_{timestamp}.json"
            )
            summary_file = os.path.join(
                output_dir, f"json_analysis_summary_{timestamp}.json"
            )

            # Save detailed analysis results
            with open(detailed_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            # Save summary statistics
            summary = {
                "timestamp": timestamp,
                "total_files": len(self.json_files),
                "error_types": self.error_types,
                "total_documents": self.total_docs,
                "documents_with_title": self.docs_with_title,
            }
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            return detailed_file

        elif result_type == "repair":
            repair_file = os.path.join(
                output_dir, f"json_repair_results_{timestamp}.json"
            )

            # Save repair results
            with open(repair_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            return repair_file

        return None


def main():
    """Main entry point for the JSON diagnostics tool."""
    parser = argparse.ArgumentParser(
        description="Enhanced JSON Diagnostics and Repair Tool for EasyNer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all JSON files in a directory
  python json_diagnostics.py analyze data/

  # Analyze with detailed error information
  python json_diagnostics.py analyze data/ --detailed

  # Repair a specific file
  python json_diagnostics.py repair data/problematic.json

  # Repair all files in a directory with aggressive strategy
  python json_diagnostics.py repair data/ --strategy aggressive

  # Repair files and save to a different directory
  python json_diagnostics.py repair data/ --output-dir fixed_data/

  # Save analysis results to custom location
  python json_diagnostics.py analyze data/ --save-results --results-dir my_results
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute"
    )

    # Parser for the 'analyze' command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze JSON files for issues"
    )
    analyze_parser.add_argument(
        "path", help="Path to JSON file or directory of JSON files"
    )
    analyze_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed error information",
    )
    analyze_parser.add_argument(
        "--sample", action="store_true", help="Show sample errors by type"
    )
    analyze_parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save analysis results to file",
    )
    analyze_parser.add_argument(
        "--results-dir",
        default="results/json_diagnostics",
        help="Directory to save results (default: results/json_diagnostics)",
    )

    # Parser for the 'repair' command
    repair_parser = subparsers.add_parser(
        "repair", help="Attempt to repair problematic JSON files"
    )
    repair_parser.add_argument(
        "path", help="Path to JSON file or directory of JSON files"
    )
    repair_parser.add_argument(
        "--strategy",
        choices=["auto", "minimal", "aggressive"],
        default="auto",
        help="Repair strategy to use",
    )
    repair_parser.add_argument(
        "--output-dir", help="Directory to save repaired files"
    )
    repair_parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save repair results summary to file",
    )
    repair_parser.add_argument(
        "--results-dir",
        default="results/json_diagnostics",
        help="Directory to save results (default: results/json_diagnostics)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup diagnostics
    path = args.path
    if os.path.isdir(path):
        diagnostic = JsonDiagnostic(path)
    else:
        diagnostic = JsonDiagnostic()
        diagnostic.json_files = [path]

    # Execute the appropriate command
    if args.command == "analyze":
        if len(diagnostic.json_files) == 1:
            # Analyze a single file
            result = diagnostic.analyze_file(
                diagnostic.json_files[0], detailed=args.detailed
            )
            print(f"\nFile: {os.path.basename(result['file_path'])}")
            print(f"Status: {result['status']}")
            if result["error_type"]:
                print(f"Error type: {result['error_type']}")
                print(f"Error details: {result['error_details']}")

            if result.get("line_number"):
                print(
                    f"Line {result['line_number']}, Position {result['position']}"
                )

            if args.detailed and result.get("context"):
                print("\nContext:")
                print(result["context"])

            if args.detailed and result.get("sample"):
                print("\nSample content:")
                print(result["sample"][:500])

            # Save single file analysis result if requested
            if args.save_results:
                results_file = diagnostic.save_results(
                    [result], "analysis", args.results_dir
                )
                print(f"\nResults saved to: {results_file}")
        else:
            # Analyze multiple files
            results = diagnostic.analyze_all_files(detailed=args.detailed)
            diagnostic.print_summary()

            if args.sample:
                diagnostic.print_error_samples()

            # Save analysis results if requested
            if args.save_results:
                results_file = diagnostic.save_results(
                    results, "analysis", args.results_dir
                )
                print(f"\nResults saved to: {results_file}")

    elif args.command == "repair":
        if len(diagnostic.json_files) == 1:
            # Repair a single file
            output_path = (
                os.path.join(
                    args.output_dir, os.path.basename(diagnostic.json_files[0])
                )
                if args.output_dir
                else None
            )
            success, message, error_type = diagnostic.repair_file(
                diagnostic.json_files[0], output_path, args.strategy
            )
            print(f"\nRepair result: {'✓' if success else '✗'} {message}")

            # Save repair result if requested
            if args.save_results:
                repair_result = {
                    "file": diagnostic.json_files[0],
                    "success": success,
                    "message": message,
                    "error_type": error_type,
                    "output_path": output_path,
                }
                results_file = diagnostic.save_results(
                    repair_result, "repair", args.results_dir
                )
                print(f"\nResults saved to: {results_file}")
        else:
            # Repair multiple files
            repair_results = diagnostic.repair_all_files(
                args.output_dir, args.strategy
            )
            diagnostic.print_repair_summary(repair_results)

            # Save repair results if requested
            if args.save_results:
                repair_results["source_directory"] = path
                repair_results["output_directory"] = args.output_dir
                repair_results["strategy"] = args.strategy
                results_file = diagnostic.save_results(
                    repair_results, "repair", args.results_dir
                )
                print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
