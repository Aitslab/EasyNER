"""PubMed Bulk Loader.

This script loads PubMed XML files, converts them to JSON format,
and filters articles based on specified criteria.
It also generates statistics about the filtering process and
counts the number of articles in the converted JSON files.
"""

import os
import sys
from glob import glob
from pathlib import Path

import orjson
from tqdm import tqdm

from easyner.pipeline.pubmed.loaders.pubmed_json_loader import PubMedJSONLoader
from easyner.pipeline.pubmed.utils import _resolve_path


def count_articles(input_path: str, baseline: int = 23) -> None:
    """Count articles from converted json files."""
    # Resolve input path
    resolved_input_path = _resolve_path(input_path)
    count = 0
    pmids = []
    # k is used for keyword to split the filename obtained from pubmed.
    # It's different for each annual baseline
    k = str(baseline) + "n"
    count_file = resolved_input_path + "counts.txt"
    pmid_file = resolved_input_path + "pmid_list.txt"
    input_files = sorted(
        glob(f"{resolved_input_path}*.json"),
        key=lambda x: int(
            os.path.splitext(os.path.basename(x))[0].split(k)[-1],
        ),
    )

    # Add debug information about JSON files found
    print(f"Found {len(input_files)} JSON files in {resolved_input_path}")
    if len(input_files) == 0:
        print(f"WARNING: No JSON files found in {resolved_input_path}")
        print("Article counting will not produce meaningful results.")
        return

    count_writer = Path(count_file).open("w", encoding="utf-8")
    pmid_writer = Path(pmid_file).open("w", encoding="utf-8")

    for infile in tqdm(input_files):
        with open(infile, encoding="utf-8") as f:
            full_articles = orjson.loads(f.read())

        count_writer.write(
            f"{os.path.splitext(os.path.basename(infile))[0].split(k)[-1]}\t{len(full_articles)}\n",
        )
        count += len(full_articles)
        pmids.extend([k for k in full_articles])

    count_writer.write(f"total\t{count}")
    count_writer.close()

    for pmid in sorted(pmids, key=int):
        pmid_writer.write(f"{pmid}\n")
    pmid_writer.close()


def load_pubmed_from_xml(config: dict) -> None:
    """Run the PubMed loading process based on the provided configuration.

    Args:
        config: A dictionary containing configuration parameters

    """
    # Get paths from config and resolve them
    input_path = _resolve_path(
        "data/tmp/pubmed/" if len(config["input_path"]) == 0 else config["input_path"],
    )
    output_path = _resolve_path(config["output_path"])

    # Get file range parameters
    file_start = config.get("file_start")
    file_end = config.get("file_end")

    # Validate file range if both are provided
    if file_start is not None and file_end is not None and file_start > file_end:
        msg = f"file_start ({file_start}) cannot be greater than file_end ({file_end})"
        raise ValueError(msg)

    print("Processing PubMed raw files...")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    if file_start is not None:
        print(f"Starting with file: {file_start}")
    if file_end is not None:
        print(f"Ending with file: {file_end}")

    # Pass the require_abstract option to the loader
    loader = PubMedJSONLoader(
        input_path=input_path,
        output_path=output_path,
        baseline=config["baseline"],
        require_abstract=config.get("require_abstract", True),
        file_start=file_start,
        file_end=file_end,
    )

    loader.run_loader()

    # Generate statistics report
    loader._write_statistics_report()

    if config.get("count_articles", False):
        print("Counting articles")
        count_articles(
            input_path=output_path,
            baseline=config["baseline"],
        )

    print("PubMed processing complete")


if __name__ == "__main__":
    try:
        # When run as a standalone script, use the config from the config module
        from easyner.config import config_manager

        # Display help message if requested
        if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
            print("PubMed Bulk Loader")
            print("=================")
            print("Usage:")
            print("  python -m easyner.pipeline.pubmed.pubmed_bulk_loader [OPTIONS]")
            print("\nOptions:")
            print("  --no-count           Skip counting articles")
            print("  --no-abstract-filter Skip filtering by abstract presence")
            print("  -h, --help           Show this help message and exit")
            print("\nConfiguration:")
            print("  This script uses the pubmed_bulk_loader section from config.json.")
            sys.exit(0)

        # Load the config if it isn't already loaded
        config = config_manager.get_config()

        # Check if the required section exists in the config
        if "pubmed_bulk_loader" in config:
            print("Using configuration from config.json")
            loader_config = config["pubmed_bulk_loader"]

            # Check that required configuration options exist
            required_keys = ["input_path", "output_path", "baseline"]
            missing_keys = [key for key in required_keys if key not in loader_config]

            if missing_keys:
                msg = (
                    f"Missing required keys in pubmed_bulk_loader: "
                    f"{', '.join(missing_keys)}"
                )
                raise ValueError(msg)

            # Apply command-line overrides if any
            if "--no-count" in sys.argv:
                loader_config["count_articles"] = False
                print("Article counting has been disabled via command line argument")

            if "--no-abstract-filter" in sys.argv:
                loader_config["require_abstract"] = False
                print("Abstract filtering has been disabled via command line argument")

            # Run the loader
            load_pubmed_from_xml(loader_config)
            print("PubMed bulk loading complete.")
        else:
            msg = (
                "pubmed_bulk_loader section not found in config.json.\n"
                "Please add this section to your config file."
            )
            raise ValueError(msg)

    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
