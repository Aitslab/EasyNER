#!/usr/bin/env python3
"""
Duplicate Sentence Analyzer

This script analyzes duplicate articles identified in a duplicate_report.json file
to check if their sentences are also complete matches. It helps determine if articles
are true duplicates (exact copies) or just share the same title but have different content.
"""

import argparse
import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from multiprocessing import cpu_count

import numpy as np
import orjson
from tqdm import tqdm

# Define similarity threshold for sentences to be considered matching
DEFAULT_SIMILARITY_THRESHOLD = 0.95  # 95% similarity
DEFAULT_MAX_CPU_LIMIT = 32  # Maximum number of CPU cores to use


class DuplicateSentenceAnalyzer:
    """
    Analyzes sentence-level similarity between articles identified as
    duplicates
    Args:
        duplicate_report_filepath (str): Path to the duplicate report JSON
        file
        similarity_threshold (float): Similarity threshold for sentences to be
        considered matching (0.0-1.0)
        max_processes (int): Number of parallel processes to use (default:
        CPU count)
        verbose (bool): Enable verbose output for debugging
    """

    def __init__(
        self,
        duplicate_report_filepath,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        num_processes=None,
        verbose=False,
    ):
        """Initialize with path to duplicate report and similarity threshold."""
        self.duplicate_report_filepath = duplicate_report_filepath
        self.similarity_threshold = similarity_threshold
        self.max_processes = num_processes or min(
            cpu_count(), DEFAULT_MAX_CPU_LIMIT
        )  # Default to CPU count or max limit, whichever is smaller
        self.verbose = verbose
        self.duplicates = []
        self.original_duplicate_count = (
            0  # Added to track original count when sampling
        )
        self.results = {}

    def load_duplicate_report(self):
        """Load the duplicate report JSON file."""
        if self.verbose:
            print(
                f"[DEBUG] Loading duplicate report from "
                f"{self.duplicate_report_filepath}"
            )

        try:
            with open(
                self.duplicate_report_filepath, "r", encoding="utf-8"
            ) as f:
                data = orjson.loads(f.read())
                self.duplicates = data.get("duplicates", [])

            duplicate_count = len(self.duplicates)
            if self.verbose:
                print(
                    f"[DEBUG] Found {duplicate_count} duplicate sets in the "
                    "report"
                )

            return duplicate_count > 0
        except Exception as e:
            print(f"Error loading duplicate report: {str(e)}")
            return False

    def calculate_sentence_similarity(self, sentence1, sentence2):
        """Calculate similarity between two sentences, using fast exact match
        check first."""
        # Fast path: Check for exact equality first
        if sentence1 == sentence2:
            return 1.0

        # Only use the more expensive SequenceMatcher if sentences aren't
        # identical
        return SequenceMatcher(None, sentence1, sentence2).ratio()

    def compare_article_sentences(self, article1_data, article2_data):
        """Compare sentences between two article objects and calculate
        similarity metrics."""
        # Get sentences from both articles
        article1_sentences = article1_data.get("sentences", [])
        article2_sentences = article2_data.get("sentences", [])

        # Handle empty sentences
        if not article1_sentences or not article2_sentences:
            return {
                "sentence_count_match": len(article1_sentences)
                == len(article2_sentences),
                "sentence_count1": len(article1_sentences),
                "sentence_count2": len(article2_sentences),
                "identical_sentences": 0,
                "similar_sentences": 0,
                "average_similarity": 0.0,
                "is_exact_duplicate": False,
                "sentence_similarities": [],
            }

        # Compare each sentence
        similarities = []
        identical_count = 0
        similar_count = 0

        # Use min length to avoid index errors
        min_length = min(len(article1_sentences), len(article2_sentences))

        for i in range(min_length):
            # Get text of each sentence (handling potential formats)
            article1_sentence_text = (
                article1_sentences[i].get("text", "")
                if isinstance(article1_sentences[i], dict)
                else str(article1_sentences[i])
            )
            article2_sentence_text = (
                article2_sentences[i].get("text", "")
                if isinstance(article2_sentences[i], dict)
                else str(article2_sentences[i])
            )

            # Skip empty sentences
            if not article1_sentence_text or not article2_sentence_text:
                continue

            # Calculate similarity
            similarity = self.calculate_sentence_similarity(
                article1_sentence_text, article2_sentence_text
            )  # This is computationally expensive
            similarities.append(similarity)

            # Count identical and similar sentences
            if similarity == 1.0:
                identical_count += 1
            elif similarity >= self.similarity_threshold:
                similar_count += 1

        # Calculate metrics
        avg_similarity = np.mean(similarities) if similarities else 0.0
        is_exact_duplicate = (
            avg_similarity >= self.similarity_threshold
            and len(article1_sentences) == len(article2_sentences)
        )

        return {
            "sentence_count_match": len(article1_sentences)
            == len(article2_sentences),
            "sentence_count1": len(article1_sentences),
            "sentence_count2": len(article2_sentences),
            "identical_sentences": identical_count,
            "similar_sentences": similar_count,
            "average_similarity": float(avg_similarity),
            "is_exact_duplicate": is_exact_duplicate,
            "sentence_similarities": similarities,
        }

    # def analyze_duplicate_pair(self, duplicate_info, file_paths):
    #     """Analyze a single pair of duplicate articles."""
    #     try:
    #         # Extract duplicate info
    #         pmid = duplicate_info["pmid"]
    #         title = duplicate_info["title"]
    #         files = duplicate_info["files"]

    #         # Ensure we have at least 2 files to compare
    #         if len(files) < 2:
    #             return {
    #                 "pmid": pmid,
    #                 "title": title,
    #                 "status": "skipped",
    #                 "reason": "less than 2 files",
    #                 "files": files,
    #             }

    #         # Load content from JSON files
    #         file_contents = []
    #         for file_path in files:
    #             # Check if file exists
    #             if not os.path.exists(file_path):
    #                 if self.verbose:
    #                     print(f"File not found: {file_path}")
    #                 continue

    #             try:
    #                 with open(
    #                     file_path, "r", encoding="utf-8"
    #                 ) as f:  # I/O operations are slow
    #                     data = orjson.loads(f.read())
    #                     # Check if the article with this PMID exists in this
    #                     # file
    #                     if pmid in data:
    #                         file_contents.append((file_path, data[pmid]))
    #                     else:
    #                         if self.verbose:
    #                             print(f"PMID {pmid} not found in {file_path}")
    #             except Exception as e:
    #                 if self.verbose:
    #                     print(f"Error reading {file_path}: {str(e)}")

    #         # Skip if we couldn't load at least 2 files
    #         if len(file_contents) < 2:
    #             return {
    #                 "pmid": pmid,
    #                 "title": title,
    #                 "status": "skipped",
    #                 "reason": "could not load at least 2 files",
    #                 "files": files,
    #             }

    #         # Compare each pair of article versions
    #         comparisons = []
    #         for i in range(len(file_contents)):
    #             for j in range(
    #                 i + 1, len(file_contents)
    #             ):  # Quadratic complexity
    #                 file1_path, article1_data = file_contents[i]
    #                 file2_path, article2_data = file_contents[j]

    #                 comparison_result = self.compare_article_sentences(
    #                     article1_data, article2_data
    #                 )

    #                 comparisons.append(
    #                     {
    #                         "file1": os.path.basename(file1_path),
    #                         "file2": os.path.basename(file2_path),
    #                         "comparison": comparison_result,
    #                     }
    #                 )

    #         # Determine overall duplicate status
    #         is_exact_duplicate_across_all = all(
    #             comp["comparison"]["is_exact_duplicate"]
    #             for comp in comparisons
    #         )

    #         return {
    #             "pmid": pmid,
    #             "title": title,
    #             "status": "analyzed",
    #             "is_exact_duplicate": is_exact_duplicate_across_all,
    #             "comparisons": comparisons,
    #             "files": [os.path.basename(f) for f in files],
    #         }

    #     except Exception as e:
    #         if self.verbose:
    #             print(f"Error analyzing duplicate {pmid}: {str(e)}")
    #         return {
    #             "pmid": pmid,
    #             "title": title if "title" in locals() else "unknown",
    #             "status": "error",
    #             "error": str(e),
    #             "files": (
    #                 duplicate_info["files"]
    #                 if "files" in duplicate_info
    #                 else []
    #             ),
    #         }

    def run_analysis(self):
        """Run the duplicate sentence analysis on all duplicate articles."""
        if not self.duplicates:
            print("No duplicates to analyze. Load the duplicate report first.")
            return False

        start_time = datetime.datetime.now()
        print(f"Started analysis at {start_time}")
        print(
            f"Processing {len(self.duplicates)} duplicate sets with "
            f"{self.max_processes} processes"
        )

        try:
            print("Step 1/3: Grouping duplicates by shared files...")
            # Group duplicates by shared files to optimize file loading
            file_to_duplicates = {}
            for duplicate in tqdm(self.duplicates, desc="Indexing files"):
                for file_path in duplicate.get("files", []):
                    if file_path not in file_to_duplicates:
                        file_to_duplicates[file_path] = []
                    file_to_duplicates[file_path].append(duplicate)

            print(
                f"Found {len(file_to_duplicates)} unique files referenced by "
                "duplicates"
            )

            print("Step 2/3: Creating optimized processing batches...")
            # Create batches of duplicates that share files
            duplicate_batches = self._create_duplicate_batches(
                file_to_duplicates
            )

            print(f"Created {len(duplicate_batches)} processing batches")

            # Log batch sizes to help diagnose issues
            batch_sizes = [len(batch) for batch in duplicate_batches]
            print(
                f"Batch size statistics: "
                f"min={min(batch_sizes)}, max={max(batch_sizes)}, "
                f"  avg={sum(batch_sizes)/len(batch_sizes):.1f}"
            )

            print("Step 3/3: Running parallel analysis...")
            # Use multiprocessing to speed up analysis
            results = []

            # Add timeout to prevent hanging
            TIMEOUT_PER_BATCH = 300  # 5 minutes timeout per batch

            with ProcessPoolExecutor(
                max_workers=self.max_processes
            ) as executor:
                # Submit batches instead of individual duplicates
                futures = []
                for i, batch in enumerate(duplicate_batches):
                    futures.append(
                        executor.submit(self._process_duplicate_batch, batch)
                    )

                # Process results as they complete
                for i, future in enumerate(
                    tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="Processing batches",
                    )
                ):
                    try:
                        batch_results = future.result(
                            timeout=TIMEOUT_PER_BATCH
                        )
                        results.extend(batch_results)

                        # Provide more frequent updates
                        if (i + 1) % 10 == 0 or i == len(futures) - 1:
                            print(
                                f"Processed {i+1}/{len(futures)} batches "
                                f"({(i+1)/len(futures)*100:.1f}%)"
                            )

                    except TimeoutError:
                        print(
                            f"WARNING: Batch timed out after "
                            f"{TIMEOUT_PER_BATCH} seconds. Skipping."
                        )
                    except Exception as e:
                        print(f"Error processing batch: {str(e)}")
                        if self.verbose:
                            import traceback

                            print(traceback.format_exc())

            # Continue with existing code...
            # Analyze results
            exact_duplicates = [
                r
                for r in results
                if r.get("status") == "analyzed"
                and r.get("is_exact_duplicate", False)
            ]
            partial_duplicates = [
                r
                for r in results
                if r.get("status") == "analyzed"
                and not r.get("is_exact_duplicate", False)
            ]
            errors = [
                r for r in results if r.get("status") in ["error", "skipped"]
            ]

            self.results = {
                "summary": {
                    "total_duplicate_sets": len(
                        self.duplicates
                    ),  # This is now correct regardless of sampling
                    "exact_duplicates": len(exact_duplicates),
                    "partial_duplicates": len(partial_duplicates),
                    "errors": len(errors),
                    "processing_time_seconds": (
                        datetime.datetime.now() - start_time
                    ).total_seconds(),
                    "similarity_threshold": self.similarity_threshold,
                    "timestamp": datetime.datetime.now().strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    ),
                },
                "exact_duplicates": exact_duplicates,
                "partial_duplicates": partial_duplicates,
                "errors": errors,
            }

            return True

        except Exception as e:
            print(f"An error occurred during analysis: {str(e)}")
            return False

    def _create_duplicate_batches(self, file_to_duplicates):
        """
        Create optimized batches of duplicates that share files.
        This minimizes redundant file loading across processes.
        """
        # Create a graph where duplicates are connected if they share files
        duplicate_to_id = {
            dup["pmid"]: i for i, dup in enumerate(self.duplicates)
        }
        connections = {}

        if self.verbose:
            print(
                f"Building connection graph for {len(duplicate_to_id)} "
                "duplicates..."
            )

        # Find connections between duplicates that share files
        for file_path, dups in tqdm(
            file_to_duplicates.items(),
            total=len(file_to_duplicates),
            desc="Creating file connections",
        ):
            # Skip files with too many duplicates to avoid memory issues
            if len(dups) > 1000:
                if self.verbose:
                    print(
                        f"Skipping file with {len(dups)} duplicates: "
                        f"{file_path}"
                    )
                continue

            for i in range(len(dups)):
                dup1 = dups[i]
                dup1_id = duplicate_to_id[dup1["pmid"]]

                if dup1_id not in connections:
                    connections[dup1_id] = set()

                for j in range(i + 1, len(dups)):
                    dup2 = dups[j]
                    dup2_id = duplicate_to_id[dup2["pmid"]]
                    connections[dup1_id].add(dup2_id)

                    if dup2_id not in connections:
                        connections[dup2_id] = set()
                    connections[dup2_id].add(dup1_id)

        if self.verbose:
            print(
                f"Finding connected components for {len(connections)} nodes..."
            )

        # Use a more efficient implementation for finding connected components
        from collections import deque

        visited = set()
        batches = []

        # Set maximum batch size to prevent huge batches
        MAX_BATCH_SIZE = 1000

        for i in tqdm(
            range(len(self.duplicates)),
            desc="Creating batches",
        ):
            if i in visited:
                continue

            # Start a new batch with this duplicate
            batch = [self.duplicates[i]]
            visited.add(i)

            # Use deque instead of list for O(1) popleft operation
            queue = deque(connections.get(i, []))

            # Limit batch size to prevent memory/performance issues
            while queue and len(batch) < MAX_BATCH_SIZE:
                next_id = queue.popleft()  # O(1) operation vs O(n) for pop(0)
                if next_id not in visited:
                    batch.append(self.duplicates[next_id])
                    visited.add(next_id)
                    # Add new unvisited nodes to queue
                    for n in connections.get(next_id, []):
                        if n not in visited and n not in queue:
                            queue.append(n)

            batches.append(batch)

            # Progress report for large batches
            if len(batch) > 100 and self.verbose:
                print(f"Created batch with {len(batch)} duplicates")

        if self.verbose:
            print(f"Created {len(batches)} initial batches")

            # Display batch statistics
            batch_sizes = [len(batch) for batch in batches]
            if batch_sizes:
                print(
                    f"Batch size statistics: min={min(batch_sizes)}, "
                    f"max={max(batch_sizes)}, avg={sum(batch_sizes)/len(batch_sizes):.1f}"
                )

        # Balance batches if needed (avoid too many small batches)
        if len(batches) > self.max_processes * 2:
            if self.verbose:
                print(
                    "Merging small batches to optimize workload distribution..."
                )

            batches = self._merge_small_batches(batches)

            if self.verbose:
                print(f"After merging: {len(batches)} balanced batches")

        return batches

    def _merge_small_batches(self, batches, target_batch_count=None):
        """Merge smaller batches to create more balanced workloads."""
        import heapq

        from tqdm import tqdm

        if target_batch_count is None:
            target_batch_count = max(self.max_processes, len(batches) // 2)

        if len(batches) <= target_batch_count:
            return batches

        # Create a heap of (size, index) tuples - don't include the batch
        # objects
        heap = [(len(batch), i) for i, batch in enumerate(batches)]
        heapq.heapify(heap)

        # Track which batches have been merged
        merged = set()
        result = []

        # Initialize progress bar
        total_merges = len(batches) - target_batch_count
        progress_bar = tqdm(total=total_merges, desc="Merging batches")

        # Continue merging until we reach the target count
        while len(heap) + len(result) - len(merged) > target_batch_count:
            # Get the smallest batch not yet processed
            size1, idx1 = heapq.heappop(heap)
            if idx1 in merged:
                continue

            # If the heap is empty or we've reached our target count, add this
            # batch to result
            if not heap:
                result.append(batches[idx1])
                break

            # Get the next smallest batch
            size2, idx2 = heapq.heappop(heap)
            if idx2 in merged:
                # Put batch1 back and continue
                heapq.heappush(heap, (size1, idx1))
                continue

            # Mark both batches as merged
            merged.add(idx1)
            merged.add(idx2)

            # Create a merged batch and add it to the heap with a new index
            merged_batch = batches[idx1] + batches[idx2]
            batches.append(merged_batch)
            heapq.heappush(heap, (len(merged_batch), len(batches) - 1))

            # Update progress bar
            progress_bar.update(1)

        # Add any remaining unmerged batches to the result
        while heap:
            size, idx = heapq.heappop(heap)
            if idx not in merged:
                result.append(batches[idx])

        progress_bar.close()

        if self.verbose:
            print(
                f"Merged {len(merged)//2} batch pairs into {len(result)} batches"
            )

        return result

    def _process_duplicate_batch(self, batch):
        """
        Process a batch of duplicates that potentially share files.
        Uses file caching to avoid redundant file loading.
        """
        results = []
        file_cache = {}  # Cache for file contents

        # Add progress bar for processing duplicates in the batch
        for duplicate in tqdm(
            batch, desc="Processing duplicates in batch", leave=False
        ):
            try:
                # Extract duplicate info
                pmid = duplicate["pmid"]
                title = duplicate["title"]
                files = duplicate["files"]

                # Ensure we have at least 2 files to compare
                if len(files) < 2:
                    results.append(
                        {
                            "pmid": pmid,
                            "title": title,
                            "status": "skipped",
                            "reason": "less than 2 files",
                            "files": files,
                        }
                    )
                    continue

                # Load content from JSON files using cache
                file_contents = []
                for file_path in files:
                    # Check if file exists
                    if not os.path.exists(file_path):
                        if self.verbose:
                            print(f"File not found: {file_path}")
                        continue

                    try:
                        # Use cached file content if available
                        if file_path not in file_cache:
                            with open(file_path, "r", encoding="utf-8") as f:
                                file_cache[file_path] = orjson.loads(f.read())

                        data = file_cache[file_path]

                        # Check if the article with this PMID exists in this
                        # file
                        if pmid in data:
                            file_contents.append((file_path, data[pmid]))
                        else:
                            if self.verbose:
                                print(f"PMID {pmid} not found in {file_path}")
                    except Exception as e:
                        if self.verbose:
                            print(f"Error reading {file_path}: {str(e)}")

                # Skip if we couldn't load at least 2 files
                if len(file_contents) < 2:
                    results.append(
                        {
                            "pmid": pmid,
                            "title": title,
                            "status": "skipped",
                            "reason": "could not load at least 2 files",
                            "files": files,
                        }
                    )
                    continue

                # Compare each pair of article versions
                comparisons = []
                for i in range(len(file_contents)):
                    for j in range(i + 1, len(file_contents)):
                        file1_path, article1_data = file_contents[i]
                        file2_path, article2_data = file_contents[j]

                        comparison_result = self.compare_article_sentences(
                            article1_data, article2_data
                        )

                        comparisons.append(
                            {
                                "file1": os.path.basename(file1_path),
                                "file2": os.path.basename(file2_path),
                                "comparison": comparison_result,
                            }
                        )

                # Determine overall duplicate status
                is_exact_duplicate_across_all = all(
                    comp["comparison"]["is_exact_duplicate"]
                    for comp in comparisons
                )

                results.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "status": "analyzed",
                        "is_exact_duplicate": is_exact_duplicate_across_all,
                        "comparisons": comparisons,
                        "files": [os.path.basename(f) for f in files],
                    }
                )

            except Exception as e:
                if self.verbose:
                    print(
                        f"Error analyzing duplicate "
                        f"{pmid if 'pmid' in locals() else 'unknown'}: {str(e)}"
                    )
                results.append(
                    {
                        "pmid": pmid if "pmid" in locals() else "unknown",
                        "title": title if "title" in locals() else "unknown",
                        "status": "error",
                        "error": str(e),
                        "files": (
                            duplicate["files"] if "files" in duplicate else []
                        ),
                    }
                )

        return results

    def print_summary(self):
        """Print a summary of the analysis results."""
        if not self.results:
            print("No results available. Run analysis first.")
            return

        summary = self.results["summary"]

        print("\n" + "=" * 60)
        print(" DUPLICATE SENTENCE ANALYSIS SUMMARY ")
        print("=" * 60)
        print(
            f"Total duplicate sets analyzed: {summary['total_duplicate_sets']}"
        )
        print(
            f"Exact duplicates (content matches): {summary['exact_duplicates']} "
            + f"({summary['exact_duplicates']/summary['total_duplicate_sets']*100:.1f}%)"
        )
        print(
            f"Partial duplicates (content differs): {summary['partial_duplicates']} "
            + f"({summary['partial_duplicates']/summary['total_duplicate_sets']*100:.1f}%)"
        )
        print(
            f"Errors/skipped: {summary['errors']} "
            + f"({summary['errors']/summary['total_duplicate_sets']*100:.1f}%)"
        )
        print(
            f"Processing time: {summary['processing_time_seconds']:.2f} seconds"
        )
        print(f"Similarity threshold: {summary['similarity_threshold']}")
        print("=" * 60)

        # Print some examples of partial duplicates if available
        if self.results["partial_duplicates"]:
            print(
                "\nEXAMPLES OF PARTIAL DUPLICATES (same title, different content):"
            )
            for i, dup in enumerate(self.results["partial_duplicates"][:3]):
                print(f"\n{i+1}. PMID: {dup['pmid']}")
                print(
                    f"   Title: {dup['title'][:80]}"
                    + ("..." if len(dup["title"]) > 80 else "")
                )
                print(f"   Files: {', '.join(dup['files'])}")
                # Show the first comparison
                if dup["comparisons"]:
                    comp = dup["comparisons"][0]["comparison"]
                    print(
                        f"   Sentence match: "
                        f"{comp['identical_sentences'] + comp['similar_sentences']}"
                        f"/"
                        f"{min(comp['sentence_count1'], comp['sentence_count2'])} "
                        f"({comp['average_similarity']*100:.1f}% similarity)"
                    )

    def save_report(self, output_path=None):
        """Save analysis results to a JSON file."""
        if not self.results:
            print("No results available. Run analysis first.")
            return False

        # Determine output file path
        if not output_path:
            timestamp = self.results["summary"]["timestamp"]
            # Create results directory if it doesn't exist
            results_dir = os.path.join(
                os.getcwd(), "results", "json_diagnostics"
            )
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(
                results_dir, f"duplicate_sentence_analysis_{timestamp}.json"
            )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(
                    orjson.dumps(
                        self.results, option=orjson.OPT_INDENT_2
                    ).decode("utf-8")
                )
            print(f"Analysis report saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return False


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze sentence-level similarities between articles identified "
            "as duplicates."
        )
    )
    parser.add_argument(
        "duplicate_report", help="Path to the duplicate report JSON file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=(
            "Similarity threshold for sentences to be considered matching "
            f"(0.0-1.0, default: {DEFAULT_SIMILARITY_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=min(cpu_count(), DEFAULT_MAX_CPU_LIMIT),
        help=(
            "Number of parallel processes to use (default: "
            f"{min(cpu_count(), DEFAULT_MAX_CPU_LIMIT)})"
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Output path for the analysis report (default: "
            "duplicate_sentence_analysis_<timestamp>.json)"
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run with cProfile to analyze performance",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Analyze only a sample of N duplicates (default: analyze all)",
    )

    args = parser.parse_args()

    # Validate threshold
    if args.threshold < 0.0 or args.threshold > 1.0:
        print("Error: Threshold must be between 0.0 and 1.0")
        return 1

    # Initialize analyzer
    analyzer = DuplicateSentenceAnalyzer(
        args.duplicate_report,
        similarity_threshold=args.threshold,
        num_processes=args.processes,
        verbose=args.verbose,
    )

    # Load duplicate report
    if not analyzer.load_duplicate_report():
        print("Failed to load duplicate report")
        return 1

    # Apply sampling if requested
    if args.sample > 0:
        original_count = len(analyzer.duplicates)
        analyzer.original_duplicate_count = original_count

        # Use random sampling instead of just taking the first N
        if args.sample < original_count:
            import random

            random.seed(42)  # For reproducibility
            analyzer.duplicates = random.sample(
                analyzer.duplicates, args.sample
            )

        print(
            f"Sampling mode: analyzing {len(analyzer.duplicates)} out of "
            f"{original_count} duplicates "
            f"({len(analyzer.duplicates)/original_count*100:.2f}%)"
        )

    # Run analysis with or without profiling
    if args.profile:
        import cProfile
        import pstats
        from datetime import datetime

        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), "results", "json_diagnostics")
        os.makedirs(results_dir, exist_ok=True)
        profile_output = os.path.join(
            results_dir,
            "duplicate_analyzer_profile_"
            + f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof",
        )
        print(
            f"Running with profiler. Results will be saved to {profile_output}"
        )

        cProfile.runctx(
            "analyzer.run_analysis()", globals(), locals(), profile_output
        )

        # Print top 20 functions by cumulative time
        stats = pstats.Stats(profile_output)
        stats.strip_dirs().sort_stats("cumulative").print_stats(20)

        if analyzer.results:  # Check if analysis was successful
            analyzer.print_summary()
            analyzer.save_report(args.output)
            return 0
        else:
            print("Analysis failed")
            return 1
    else:
        # Normal run without profiling
        if analyzer.run_analysis():
            analyzer.print_summary()
            analyzer.save_report(args.output)
            return 0
        else:
            print("Analysis failed")
            return 1


if __name__ == "__main__":
    exit(main())
