"""Script to check for DeleteCitation elements in PubMed XML files.

This script is optimized to efficiently scan through large XML datasets
and identify files containing deleted citations.
"""

import warnings

warnings.warn(
    "This module is deprecated and will be removed in a future version. "
    "Please use the pubmed_xml_parser or the pubmed_parser package instead. "
    "This script is kept for development utility purposes only.",
    DeprecationWarning,
    stacklevel=2,
)

import argparse
import datetime
import glob
import gzip
import json
import os
import subprocess
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from lxml import etree
from tqdm import tqdm

# Path configuration - can be overridden by command line arguments
DEFAULT_DATA_DIR = "/lunarc/nobackup/projects/snic2020-6-41/carl/data/pubmed_raw/"
DEFAULT_FILE_PATTERN = "*.xml.gz"
DEFAULT_OUTPUT_DIR = "./pubmed_analysis/"
DEFAULT_PROCESS_COUNT = min(cpu_count(), 16)  # Limit to reasonable number


def check_delete_citations(file_path):
    """Check for DeleteCitation elements in a PubMed XML file.

    Args:
        file_path: Path to the XML file to check

    Returns:
        tuple: (has_deletions, deleted_pmids)

    """
    deleted_pmids = []
    file_name = os.path.basename(file_path)

    try:
        # Method 1: Use grep for faster initial check (works on compressed files)
        cmd = f"zgrep -l 'DeleteCitation' {file_path}"
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)

        # If grep found nothing, we can return early
        if not result.stdout.strip():
            return {
                "file": file_name,
                "file_path": file_path,
                "has_deletions": False,
                "deleted_pmids": [],
                "count": 0,
            }

        print(f"File {file_name} contains DeleteCitation elements, parsing...")

        # Method 2: If grep found something, parse the XML to get the PMIDs
        with gzip.open(file_path, "rb") as f:
            context = etree.iterparse(f, events=("end",), tag="DeleteCitation")
            for _, element in context:
                for pmid_element in element.findall(".//PMID"):
                    if pmid_element.text:
                        deleted_pmids.append(pmid_element.text)
                element.clear()

        return {
            "file": file_name,
            "file_path": file_path,
            "has_deletions": True,
            "deleted_pmids": deleted_pmids,
            "count": len(deleted_pmids),
        }

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return {
            "file": file_name,
            "file_path": file_path,
            "has_deletions": False,
            "error": f"{str(e)}\n{error_details}",
            "count": 0,
        }


def analyze_delete_citations_multiprocessing(
    data_dir=DEFAULT_DATA_DIR,
    file_pattern=DEFAULT_FILE_PATTERN,
    output_dir=DEFAULT_OUTPUT_DIR,
    limit=None,
    num_processes=DEFAULT_PROCESS_COUNT,
) -> None:
    """Analyze PubMed XML files for DeleteCitation elements using multiprocessing.

    Args:
        data_dir: Directory containing PubMed XML files
        file_pattern: Pattern to match XML files
        output_dir: Directory to save output
        limit: Maximum number of files to process
        num_processes: Number of parallel processes to use

    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all XML files in directory
    file_paths = glob.glob(os.path.join(data_dir, file_pattern))

    if limit:
        file_paths = file_paths[:limit]

    print(f"Found {len(file_paths)} XML files to process")

    if not file_paths:
        print(f"No files found matching pattern {file_pattern} in {data_dir}")
        return

    # Statistics
    files_with_deletions = []
    all_deleted_pmids = []
    processed_files = 0
    errors = []

    start_time = time.time()

    try:
        with Pool(processes=num_processes) as pool:
            # Set up progress bar with additional statistics
            progress_bar = tqdm(
                total=len(file_paths),
                desc="Checking for DeleteCitation elements",
                unit="file",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )

            # Process results as they come in
            results = []

            # Define callback to update progress bar
            def update_progress(result):
                nonlocal processed_files
                progress_bar.update(1)
                processed_files += 1

                if result.get("has_deletions", False):
                    count = result.get("count", 0)
                    progress_bar.set_postfix(
                        found=f"{len(files_with_deletions)}",
                        pmids=f"{len(all_deleted_pmids)}",
                        errors=f"{len(errors)}",
                    )

                return result

            # Submit all jobs
            for file_path in file_paths:
                results.append(
                    pool.apply_async(
                        check_delete_citations,
                        args=(file_path,),
                        callback=update_progress,
                    ),
                )

            # Wait for all tasks to complete and collect results
            all_results = []
            for result_obj in results:
                try:
                    result = result_obj.get()
                    all_results.append(result)

                    if "error" in result:
                        errors.append(result)
                    elif result.get("has_deletions", False):
                        files_with_deletions.append(result)
                        all_deleted_pmids.extend(result.get("deleted_pmids", []))
                except Exception as e:
                    print(f"Error processing result: {e}")

            progress_bar.close()

    except KeyboardInterrupt:
        print(
            "\nProcess interrupted by user. Generating report with processed files...",
        )
    finally:
        total_time = time.time() - start_time

        # Prepare results
        results = {
            "total_files_processed": processed_files,
            "files_with_deletions": len(files_with_deletions),
            "total_deleted_pmids": len(all_deleted_pmids),
            "unique_deleted_pmids": len(set(all_deleted_pmids)),
            "processing_time": total_time,
            "processing_speed": processed_files / total_time if total_time > 0 else 0,
            "errors": len(errors),
            "deletion_details": files_with_deletions,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

        # Save results
        output_path = os.path.join(output_dir, "pubmed_deletion_analysis.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Display summary
    print("\nSummary:")
    print(f"Total files processed: {processed_files} of {len(file_paths)}")
    print(f"Files with DeleteCitation tags: {len(files_with_deletions)}")
    print(f"Total deleted PMIDs: {len(all_deleted_pmids)}")
    print(f"Unique deleted PMIDs: {len(set(all_deleted_pmids))}")
    print(f"Processing time: {total_time:.2f} seconds")
    print(f"Processing speed: {processed_files/total_time:.2f} files/second")

    if files_with_deletions:
        print("\nTop 10 files with most deletions:")
        for item in sorted(
            files_with_deletions,
            key=lambda x: x["count"],
            reverse=True,
        )[:10]:
            print(f"  {item['file']}: {item['count']} deleted PMIDs")


def analyze_delete_citations(
    data_dir=DEFAULT_DATA_DIR,
    file_pattern=DEFAULT_FILE_PATTERN,
    output_dir=DEFAULT_OUTPUT_DIR,
    limit=None,
) -> None:
    """Sequential (non-parallel) version of the analysis function.
    Useful for debugging or when multiprocessing causes issues.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all XML files in directory
    file_paths = glob.glob(os.path.join(data_dir, file_pattern))

    if limit:
        file_paths = file_paths[:limit]

    print(f"Found {len(file_paths)} XML files to process")

    if not file_paths:
        print(f"No files found matching pattern {file_pattern} in {data_dir}")
        return

    # Statistics
    files_with_deletions = []
    all_deleted_pmids = []

    # Process each file sequentially
    for file_path in tqdm(file_paths, desc="Checking for DeleteCitation elements"):
        result = check_delete_citations(file_path)

        if result["has_deletions"]:
            files_with_deletions.append(result)
            all_deleted_pmids.extend(result.get("deleted_pmids", []))

    # Prepare results
    results = {
        "total_files_processed": len(file_paths),
        "files_with_deletions": len(files_with_deletions),
        "total_deleted_pmids": len(all_deleted_pmids),
        "unique_deleted_pmids": len(set(all_deleted_pmids)),
        "deletion_details": files_with_deletions,
    }

    # Save results
    output_path = os.path.join(output_dir, "pubmed_deletion_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Analysis complete. Results saved to {output_path}")

    # Display summary
    print("\nSummary:")
    print(f"Total files processed: {len(file_paths)}")
    print(f"Files with DeleteCitation tags: {len(files_with_deletions)}")
    print(f"Total deleted PMIDs: {len(all_deleted_pmids)}")
    print(f"Unique deleted PMIDs: {len(set(all_deleted_pmids))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for DeleteCitation elements in PubMed XML files",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing PubMed XML files",
    )
    parser.add_argument(
        "--file-pattern",
        default=DEFAULT_FILE_PATTERN,
        help="Pattern to match XML files",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of files to process")
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Use multiprocessing for faster execution (default: True)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=DEFAULT_PROCESS_COUNT,
        help=f"Number of parallel processes (default: {DEFAULT_PROCESS_COUNT})",
    )

    args = parser.parse_args()

    # Run the appropriate analysis function based on arguments
    if args.parallel:
        print(f"Using parallel processing with {args.processes} processes")
        analyze_delete_citations_multiprocessing(
            args.data_dir,
            args.file_pattern,
            args.output_dir,
            args.limit,
            args.processes,
        )
    else:
        print("Using sequential processing")
        analyze_delete_citations(
            args.data_dir,
            args.file_pattern,
            args.output_dir,
            args.limit,
        )
