import argparse
import datetime
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from multiprocessing import cpu_count

import orjson
from tqdm import tqdm

# Get system capabilities
SYSTEM_CPU_COUNT = cpu_count()

# Try to load CPU_LIMIT from config.json, with fallback
CONFIG_CPU_LIMIT = 16  # Default fallback
try:
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config.json",
    )
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = orjson.loads(f.read())
            if "CPU_LIMIT" in config:
                if isinstance(config["CPU_LIMIT"], int):
                    CONFIG_CPU_LIMIT = config["CPU_LIMIT"]
                else:
                    print(
                        "Warning: CPU_LIMIT in config.json is not an integer. "
                        "Using default value.",
                    )
            else:
                print(
                    "Warning: CPU_LIMIT not found in config.json. "
                    "Using default value.",
                )
except Exception as e:
    print(f"Warning: Could not load CPU_LIMIT from config.json: {e}")

# Define constants respecting both system capabilities and config limit
DEFAULT_THREAD_COUNT = min(SYSTEM_CPU_COUNT, CONFIG_CPU_LIMIT)
DEFAULT_PROCESS_COUNT = min(SYSTEM_CPU_COUNT, CONFIG_CPU_LIMIT)


class ArticleCounter:
    """Efficiently count articles and unique doc_id-title combinations across JSON files."""

    def __init__(
        self,
        json_dir: str,
        num_threads=DEFAULT_THREAD_COUNT,
        num_processes=DEFAULT_PROCESS_COUNT,
        sample_mode=False,
    ):
        """Initialize with directory path and thread/process count."""
        self.json_dir = json_dir
        # Ensure thread/process counts never exceed system capabilities
        self.num_threads = min(num_threads, SYSTEM_CPU_COUNT)
        self.num_processes = min(num_processes, SYSTEM_CPU_COUNT)
        self.json_files = sorted(glob(os.path.join(json_dir, "*.json")))

        # If sample mode is enabled, only process the first 2 files
        if sample_mode and len(self.json_files) > 5:
            self.json_files = self.json_files[:10]

        self.results = {}
        self.processed_files = []
        self.error_files = []

    def _process_file(self, json_file):
        """Process a single JSON file and return statistics."""
        try:
            file_size_mb = os.path.getsize(json_file) / (1024 * 1024)  # MB
            start_time = time.time()

            # Read and parse JSON file
            with open(json_file, encoding="utf-8") as f:
                try:
                    data = orjson.loads(f.read())
                except orjson.JSONDecodeError as je:
                    return {
                        "filename": os.path.basename(json_file),
                        "error": f"JSON decode error: {str(je)}",
                        "file_size_mb": file_size_mb,
                    }

            # Calculate statistics
            doc_count = len(data)
            unique_combos = set()

            # Extract unique document-title combinations
            if isinstance(data, dict):
                for doc_id, doc_data in data.items():
                    if isinstance(doc_data, dict) and "title" in doc_data:
                        unique_combos.add((str(doc_id), doc_data["title"]))
            else:
                return {
                    "filename": os.path.basename(json_file),
                    "error": (f"Expected JSON object, got {type(data).__name__}"),
                    "file_size_mb": file_size_mb,
                }

            processing_time = time.time() - start_time

            # Clean up memory
            del data
            gc.collect()

            return {
                "filename": os.path.basename(json_file),
                "file_path": json_file,
                "file_size_mb": file_size_mb,
                "doc_count": doc_count,
                "unique_combinations": unique_combos,
                "processing_time": processing_time,
            }

        except Exception as e:
            # Get more detailed error information
            import traceback

            error_details = traceback.format_exc()
            return {
                "filename": os.path.basename(json_file),
                "file_path": json_file,
                "error": f"{str(e)}\n{error_details}",
            }

    def _process_file_streaming(self, json_file):
        """Process a single JSON file using streaming parser to reduce memory usage."""
        try:
            file_size_mb = os.path.getsize(json_file) / (1024 * 1024)  # MB
            start_time = time.time()

            # Use ijson for streaming JSON parsing
            import ijson

            doc_count = 0
            unique_combos = set()

            with open(json_file, "rb") as f:
                # Check first character to determine if it's an object or array
                first_char = f.read(1).decode("utf-8")
                f.seek(0)

                if first_char == "{":
                    # It's a dictionary/object
                    for prefix, event, value in ijson.parse(f):
                        if event == "map_key":
                            doc_id = value
                        elif prefix.endswith(".title") and event == "string":
                            unique_combos.add((doc_id, value))
                            doc_count += 1
                else:
                    return {
                        "filename": os.path.basename(json_file),
                        "error": f"Expected JSON object, got {first_char}",
                        "file_size_mb": file_size_mb,
                    }

            processing_time = time.time() - start_time

            return {
                "filename": os.path.basename(json_file),
                "file_path": json_file,
                "file_size_mb": file_size_mb,
                "doc_count": doc_count,
                "unique_combinations": unique_combos,
                "processing_time": processing_time,
            }
        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return {
                "filename": os.path.basename(json_file),
                "file_path": json_file,
                "error": f"{str(e)}\n{error_details}",
            }

    def run_threaded_counter(self):
        """Run counting operation using Python's ThreadPoolExecutor."""
        print(f"Found {len(self.json_files)} JSON files in directory")
        print(f"Processing with {self.num_threads} threads...")

        start_time = time.time()
        total_articles = 0
        all_combinations = set()
        file_stats = []
        errors = []

        try:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(self._process_file, json_file): json_file
                    for json_file in self.json_files
                }

                # Process results as they complete
                for future in tqdm(
                    as_completed(future_to_file),
                    total=len(future_to_file),
                ):
                    result = future.result()
                    json_file = future_to_file[future]
                    self.processed_files.append(json_file)

                    if "error" in result:
                        errors.append(result)
                        self.error_files.append(json_file)
                        continue

                    total_articles += result["doc_count"]
                    all_combinations.update(result["unique_combinations"])
                    file_stats.append(result)

        except KeyboardInterrupt:
            print(
                "\nProcess interrupted by user. Generating report with "
                "processed files...",
            )
        finally:
            # Calculate total processing time
            total_time = time.time() - start_time

            # Prepare results
            self.results = {
                "total_articles": total_articles,
                "unique_combinations": len(all_combinations),
                "duplicate_rate": (
                    100 * (1 - len(all_combinations) / total_articles)
                    if total_articles > 0
                    else 0
                ),
                "processing_time": total_time,
                "processed_files": len(self.processed_files),
                "total_files": len(self.json_files),
                "completed": len(self.processed_files) == len(self.json_files),
                "errors": len(errors),
                "error_details": (
                    errors[:5] if errors else []
                ),  # Include first 5 errors
                "file_stats": file_stats,
                "timestamp": datetime.datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S",
                ),
            }

            # Print detailed error information if all processed files failed
            if len(errors) > 0 and len(errors) == len(self.processed_files):
                print(
                    "\nAll processed files failed processing. " "First error details:",
                )
                if errors:
                    err_msg = errors[0]["error"]
                    print(
                        f"{err_msg[:500]}..." if len(err_msg) > 500 else err_msg,
                    )

                    # Check for common JSON structure issues
                    if self.processed_files:
                        sample_file = self.processed_files[0]
                        print(
                            f"\nChecking structure of first file: "
                            f"{os.path.basename(sample_file)}",
                        )
                        try:
                            with open(sample_file, encoding="utf-8") as f:
                                first_char = f.read(1)
                                if first_char == "{":
                                    print(
                                        "File appears to start with a JSON object (expected).",
                                    )
                                elif first_char == "[":
                                    print(
                                        "File starts with an array, but code expects a "
                                        "dictionary/object.",
                                    )
                                else:
                                    print(
                                        "File doesn't start with a valid JSON structure "
                                        f"(starts with '{first_char}').",
                                    )
                        except Exception as e:
                            print(f"Error examining file structure: {str(e)}")

        return self.results

    def run_multiprocessing_counter(self):
        """Run counting using multiprocessing for better CPU utilization."""
        from multiprocessing import Pool, cpu_count

        print(f"Found {len(self.json_files):,} JSON files in directory")
        available_cores = cpu_count()
        print(f"System has {available_cores} CPU cores available")
        print(
            f"Processing with {self.num_processes} processes "
            f"({self.num_processes/available_cores:.1f}x CPU utilization)...",
        )

        start_time = time.time()
        total_articles = 0
        all_combinations = set()
        file_stats = []
        errors = []
        processed_files = []

        try:
            with Pool(processes=self.num_processes) as pool:
                # Enhanced tqdm with more information
                progress_bar = tqdm(
                    total=len(self.json_files),
                    desc="Processing JSON files",
                    unit="file",
                    bar_format=(
                        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                    ),
                )

                # Use a callback to update our progress
                def update_progress(result):
                    progress_bar.update(1)
                    if "error" not in result:
                        progress_bar.set_postfix(
                            articles=f"{total_articles:,}",
                            errors=f"{len(errors)}",
                        )
                    return result

                # Submit all jobs and collect results
                results = []
                for file_path in self.json_files:
                    results.append(
                        pool.apply_async(
                            self._process_file,
                            args=(file_path,),
                            callback=update_progress,
                        ),
                    )

                # Wait for all tasks to complete
                for result in results:
                    result.wait()

                progress_bar.close()

                # Process the actual results
                results = [r.get() for r in results]
                for i, result in enumerate(results):
                    json_file = self.json_files[i]
                    processed_files.append(json_file)

                    if "error" in result:
                        errors.append(result)
                        self.error_files.append(json_file)
                        continue

                    total_articles += result["doc_count"]
                    all_combinations.update(result["unique_combinations"])
                    file_stats.append(result)

        except KeyboardInterrupt:
            print(
                "\nProcess interrupted by user. Generating report with "
                "processed files...",
            )
        finally:
            # Calculate total processing time
            total_time = time.time() - start_time

            # Prepare results
            self.processed_files = processed_files
            self.results = {
                "total_articles": total_articles,
                "unique_combinations": len(all_combinations),
                "duplicate_rate": (
                    100 * (1 - len(all_combinations) / total_articles)
                    if total_articles > 0
                    else 0
                ),
                "processing_time": total_time,
                "processed_files": len(processed_files),
                "total_files": len(self.json_files),
                "completed": len(processed_files) == len(self.json_files),
                "errors": len(errors),
                "error_details": (
                    errors[:5] if errors else []
                ),  # Include first 5 errors
                "file_stats": file_stats,
                "timestamp": datetime.datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S",
                ),
                "method": "multiprocessing",
            }

            # Calculate and display processing speed metrics
            files_per_second = (
                len(processed_files) / total_time if total_time > 0 else 0
            )
            articles_per_second = total_articles / total_time if total_time > 0 else 0

            print("\nProcessing complete:")
            print(
                f"- Speed: {files_per_second:.2f} files/sec, "
                f"{articles_per_second:.2f} articles/sec",
            )

            # Calculate and format file processing percentage
            files_processed_pct = 100 * len(processed_files) / len(self.json_files)
            print(
                f"- Files: {len(processed_files):,} of "
                f"{len(self.json_files):,} ({files_processed_pct:.1f}%)",
            )

            print(
                f"- Articles: {total_articles:,} "
                f"({len(all_combinations):,} unique, "
                f"{self.results['duplicate_rate']:.2f}% duplicates)",
            )
            print(f"- Errors: {len(errors):,}")

            # Print detailed error information if all processed files failed
            if len(errors) > 0 and len(errors) == len(processed_files):
                print(
                    "\nAll processed files failed processing. " "First error details:",
                )
                if errors:
                    err_msg = errors[0]["error"]
                    print(
                        f"{err_msg[:500]}..." if len(err_msg) > 500 else err_msg,
                    )

                    # Check for common JSON structure issues
                    if processed_files:
                        sample_file = processed_files[0]
                        print(
                            f"\nChecking structure of first file: "
                            f"{os.path.basename(sample_file)}",
                        )
                        try:
                            with open(sample_file, encoding="utf-8") as f:
                                first_char = f.read(1)
                                if first_char == "{":
                                    print(
                                        "File appears to start with a JSON object (expected).",
                                    )
                                elif first_char == "[":
                                    print(
                                        "File starts with an array, but code expects a "
                                        "dictionary/object.",
                                    )
                                else:
                                    print(
                                        "File doesn't start with a valid JSON structure "
                                        f"(starts with '{first_char}').",
                                    )
                        except Exception as e:
                            print(f"Error examining file structure: {str(e)}")

        return self.results

    def print_summary(self):
        """Print a summary of the results."""
        if not self.results:
            print("No results available. Run a counting method first.")
            return

        print("\nResults:")
        print(f"Total articles: {self.results['total_articles']:,}")
        print(
            f"Unique doc_id-title combinations: {self.results['unique_combinations']:,}",
        )
        print(f"Duplicate rate: {self.results['duplicate_rate']:.2f}%")
        print(
            f"Processing time: {self.results['processing_time']:.2f} seconds",
        )
        print(
            f"Processed {self.results['processed_files']} of {self.results.get('total_files', '?')} files",
        )

        if "errors" in self.results and self.results["errors"] > 0:
            print(
                f"Encountered {self.results['errors']} errors during processing",
            )

        if not self.results.get("completed", True):
            print("Note: Processing was interrupted before completion")

    def generate_report(self, output_path=None):
        """Generate a detailed report of the results in JSON format."""
        if not self.results:
            print("No results available. Run a counting method first.")
            return False

        if not output_path:
            timestamp = self.results.get(
                "timestamp",
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
            # Create results directory if it doesn't exist
            results_dir = os.path.join(
                os.getcwd(),
                "results",
                "json_diagnostics",
            )
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(
                results_dir,
                f"article_count_report_{timestamp}.json",
            )

        # Create a clean version of the results for the report
        report_data = {
            "summary": {
                "total_articles": self.results["total_articles"],
                "unique_combinations": self.results["unique_combinations"],
                "duplicate_rate": self.results["duplicate_rate"],
                "processing_time_seconds": self.results["processing_time"],
                "files_processed": self.results["processed_files"],
                "total_files": self.results.get(
                    "total_files",
                    len(self.json_files),
                ),
                "completed": self.results.get("completed", True),
                "errors_encountered": self.results.get("errors", 0),
                "generated_at": datetime.datetime.now().isoformat(),
            },
        }

        # Add error details if present
        if "error_details" in self.results and self.results["error_details"]:
            report_data["errors"] = [
                {
                    "filename": err["filename"],
                    "error_message": err["error"].split("\n")[
                        0
                    ],  # Only include the main error message
                }
                for err in self.results["error_details"]
            ]
            if len(self.results.get("error_details", [])) < self.results.get(
                "errors",
                0,
            ):
                report_data["errors"].append(
                    {
                        "note": f"Showing only first {len(self.results['error_details'])} of {self.results['errors']} errors",
                    },
                )

        # Add file statistics if available
        if "file_stats" in self.results:
            report_data["file_statistics"] = [
                {
                    "filename": stat["filename"],
                    "size_mb": round(stat["file_size_mb"], 2),
                    "document_count": stat["doc_count"],
                    "unique_combinations": len(stat["unique_combinations"]),
                    "processing_time_seconds": round(
                        stat["processing_time"],
                        3,
                    ),
                }
                for stat in self.results["file_stats"]
            ]

        # Write the report to file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(
                    orjson.dumps(
                        report_data,
                        option=orjson.OPT_INDENT_2,
                    ).decode("utf-8"),
                )
            print(f"Report saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return False

    def generate_duplicate_report(self, output_path=None):
        """Generate a detailed report of duplicates in JSON format."""
        if not self.results or "file_stats" not in self.results:
            print(
                "No results available to generate duplicate report. Run a counting method first.",
            )
            return False

        # Build a dictionary to track occurrences of each doc_id-title combination
        combo_occurrences = {}
        combo_files = {}

        # Collect all unique combinations and their file locations
        for stat in self.results["file_stats"]:
            file_path = stat.get("file_path", "unknown_path")
            unique_combos = stat.get("unique_combinations", set())

            for combo in unique_combos:
                pmid, title = combo
                key = (pmid, title)

                # Count occurrences
                if key not in combo_occurrences:
                    combo_occurrences[key] = 1
                    combo_files[key] = [file_path]
                else:
                    combo_occurrences[key] += 1
                    combo_files[key].append(file_path)

        # Filter only duplicates (combinations that appear more than once)
        duplicates = []
        for (pmid, title), count in combo_occurrences.items():
            if count > 1:
                duplicates.append(
                    {
                        "count": count,
                        "pmid": pmid,
                        "title": title,
                        "files": combo_files[(pmid, title)],
                    },
                )

        # Sort by count (highest first) and then by pmid
        duplicates.sort(key=lambda x: (-x["count"], x["pmid"]))

        # Generate report even if no duplicates were found
        duplicate_report = {
            "duplicates": duplicates,
            "total_duplicate_combinations": len(duplicates),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

        # Determine output path
        if not output_path:
            timestamp = self.results.get(
                "timestamp",
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
            # Create results directory if it doesn't exist
            results_dir = os.path.join(
                os.getcwd(),
                "results",
                "json_diagnostics",
            )
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(
                results_dir,
                f"duplicate_report_{timestamp}.json",
            )

        # Write the duplicate report to file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(
                    orjson.dumps(
                        duplicate_report,
                        option=orjson.OPT_INDENT_2,
                    ).decode("utf-8"),
                )

            if duplicates:
                print(
                    f"Duplicate report saved to: {output_path} ({len(duplicates)} duplicate combinations found)",
                )
            else:
                print(
                    f"Duplicate report saved to: {output_path} (No duplicates found)",
                )
            return True
        except Exception as e:
            print(f"Error saving duplicate report: {str(e)}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count articles and unique doc_id-title combinations across JSON files",
    )
    parser.add_argument(
        "json_dir",
        help="Directory containing the JSON files to analyze",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREAD_COUNT,
        help=f"Number of threads to use for thread mode (default: {DEFAULT_THREAD_COUNT})",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=DEFAULT_PROCESS_COUNT,
        help=f"Number of processes to use for process mode (default: {DEFAULT_PROCESS_COUNT})",
    )
    parser.add_argument(
        "--method",
        choices=["thread", "process"],
        default="thread",
        help="Processing method: 'thread' for Python threads, 'process' for Python multiprocessing (default: thread)",
    )
    parser.add_argument(
        "--report",
        metavar="FILEPATH",
        help="Generate a detailed JSON report (default: article_count_report_<timestamp>.json)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process only the first 5 files (for testing)",
    )
    parser.add_argument(
        "--duplicate_report",
        metavar="FILEPATH",
        help=(
            "Generate a detailed JSON duplicate report "
            "(default: duplicate_report_<timestamp>.json)"
        ),
    )

    args = parser.parse_args()

    # Initialize the counter with path to JSON files and thread/process counts
    counter = ArticleCounter(
        args.json_dir,
        num_threads=args.threads,
        num_processes=args.processes,
        sample_mode=args.sample,
    )

    # Run the appropriate counter method
    if args.method == "thread":
        results = counter.run_threaded_counter()
    elif args.method == "process":
        results = counter.run_multiprocessing_counter()

    # Print summary of results
    counter.print_summary()

    # Generate main report
    counter.generate_report(args.report)

    # Generate duplicate report by default
    if args.duplicate_report:
        duplicate_report_path = args.duplicate_report
    else:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), "results", "json_diagnostics")
        os.makedirs(results_dir, exist_ok=True)
        duplicate_report_path = os.path.join(
            results_dir,
            f"duplicate_report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
        )
    counter.generate_duplicate_report(duplicate_report_path)
