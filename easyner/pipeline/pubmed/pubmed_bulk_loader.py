"""PubMed Bulk Loader.

This script loads PubMed XML files, converts them to JSON format,
and filters articles based on specified criteria.
It also generates statistics about the filtering process and
counts the number of articles in the converted JSON files.
"""

import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any, Optional

import orjson
import pubmed_parser as pp
from tqdm import tqdm


class PubMedLoader:
    """PubMed XML to JSON loader."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        k: str,
        require_abstract: bool = False,
        file_start: Optional[int] = None,
        file_end: Optional[int] = None,
    ) -> None:
        """Initialize the PubMedLoader."""
        # Resolve paths against project root if they're relative
        self.input_path = _resolve_path(input_path)
        self.output_path = _resolve_path(output_path)
        self.counter = {}
        # Ensure k is a string
        self.k = str(k)
        self.require_abstract = require_abstract
        self.file_start = file_start
        self.file_end = file_end

        # Validate file range if both are provided
        if self.file_start is not None and self.file_end is not None:
            if self.file_start > self.file_end:
                msg = f"file_start ({self.file_start}) cannot be greater than file_end ({self.file_end})"
                raise ValueError(
                    msg,
                )

        self.filter_stats = {
            "total_articles": 0,
            "no_abstract": 0,
            "abstract_not_string": 0,
            "empty_abstract": 0,
            "included_articles": 0,
        }
        os.makedirs(self.output_path, exist_ok=True)

    def _get_input_files(self, input_path: str) -> list[str]:
        """Get input files using path objects for reliable path handling.

        Args:
            input_path: Directory containing the input files

        Returns:
            List of input file paths sorted by file number

        """
        # k is used for keyword to split the filename obtained from pubmed.
        # It's different for each annual baseline
        input_path_obj = Path(input_path)

        # Use Path's glob method which handles path separators correctly
        input_files = sorted(
            [str(p) for p in input_path_obj.glob("*.gz")],
            key=lambda x: int(
                os.path.splitext(os.path.basename(x))[0].split(self.k + "n")[-1][:-4],
            ),
        )

        # Filter files by range if specified
        if input_files and (self.file_start is not None or self.file_end is not None):
            filtered_files = []
            for file_path in input_files:
                try:
                    file_num = int(
                        os.path.splitext(os.path.basename(file_path))[0].split(
                            self.k + "n",
                        )[-1][:-4],
                    )

                    # Apply file_start filter if specified
                    if self.file_start is not None and file_num < self.file_start:
                        continue

                    # Apply file_end filter if specified
                    if self.file_end is not None and file_num > self.file_end:
                        continue

                    filtered_files.append(file_path)
                except (ValueError, IndexError):
                    # Skip files that don't match expected naming pattern
                    continue

            input_files = filtered_files
            print(
                f"After applying range filters (start={self.file_start}, end={self.file_end}): {len(input_files)} files",
            )

        # Add debug output for the number of files found
        print(f"Found {len(input_files)} XML files in {input_path}")
        if len(input_files) == 0:
            print(
                f"WARNING: No XML files found in {input_path} matching pattern '*.gz'",
            )
            print("Make sure the path exists and contains gzipped XML files.")
        return input_files

    def _get_counter(self):
        return self.counter

    def _load_xml_and_convert(self, input_file: str) -> dict[str, Any]:
        """Load XML file and convert to JSON format."""
        data: list[dict[str, Any]] = pp.parse_medline_xml(
            input_file,
            year_info_only=False,
        )

        article: dict[str, Any] = {}
        local_stats: dict[str, int] = {
            "total": len(data),
            "no_abstract": 0,
            "abstract_not_string": 0,
            "empty_abstract": 0,
            "included": 0,
        }

        for art in data:
            self.filter_stats["total_articles"] += 1
            local_stats["total"] += 1

            pmid = art.get("pmid", str(self.filter_stats["total_articles"]))

            # Check if we should include this article based on abstract requirements
            include_article = True

            if self.require_abstract:
                if "abstract" not in art:
                    self.filter_stats["no_abstract"] += 1
                    local_stats["no_abstract"] += 1
                    include_article = False
                elif not isinstance(art["abstract"], str):
                    self.filter_stats["abstract_not_string"] += 1
                    local_stats["abstract_not_string"] += 1
                    include_article = False
                elif len(art["abstract"]) == 0:
                    self.filter_stats["empty_abstract"] += 1
                    local_stats["empty_abstract"] += 1
                    include_article = False

            if include_article:
                self.filter_stats["included_articles"] += 1
                local_stats["included"] += 1

                # Create a default empty abstract if it doesn't exist
                # and we're not requiring it
                abstract = (
                    art.get("abstract", "")
                    if not self.require_abstract
                    else art["abstract"]
                )

                article[pmid] = {
                    "title": art.get("title", ""),
                    "abstract": abstract,
                    "mesh_terms": art.get("mesh_terms", ""),
                    "pubdate": art.get("pubdate", ""),
                    "chemical_list": art.get("chemical_list", ""),
                }

        self.counter[input_file] = local_stats
        return article

    def _write_to_json(self, data: dict[str, Any], input_file: str) -> None:
        outfile = os.path.join(
            self.output_path,
            os.path.basename(input_file.split(".xml")[0]) + ".json",
        )
        with open(outfile, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def run_loader(self) -> None:
        """Run the loader of PubMed files."""
        print(f"Starting to load PubMed files from {self.input_path}")
        input_files_list = self._get_input_files(self.input_path)

        # Add more debug information about the files being processed
        if len(input_files_list) > 0:
            print(f"Processing {len(input_files_list)} XML files")
            print(f"First file: {os.path.basename(input_files_list[0])}")
            print(f"Last file: {os.path.basename(input_files_list[-1])}")
        else:
            print("No files to process. Please check the input path and file pattern.")
            return

        for _, input_file in tqdm(enumerate(input_files_list)):
            data = self._load_xml_and_convert(input_file)
            self._write_to_json(data, input_file)

    def get_filter_statistics(self) -> dict[str, int]:
        """Return statistics about filtered articles."""
        return self.filter_stats

    def _write_statistics_report(
        self,
        output_file: str = "filter_statistics.json",
    ) -> None:
        """Write filtering statistics to a JSON file."""
        report_path = os.path.join(self.output_path, output_file)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {"overall": self.filter_stats, "by_file": self.counter},
                f,
                indent=2,
                ensure_ascii=False,
            )


def _resolve_path(path_str: str) -> str:
    """Resolve a path against the project root if it's relative.

    Args:
        path_str: Path string that might be relative

    Returns:
        Resolved absolute path as string

    """
    path = Path(path_str)
    if path.is_absolute():
        return str(path)

    # Import here to avoid circular imports
    from easyner.infrastructure.paths import PROJECT_ROOT

    # Resolve relative to PROJECT_ROOT
    resolved_path = PROJECT_ROOT / path
    return str(resolved_path)


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


def run_pubmed_loading(config: dict) -> None:
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
    loader = PubMedLoader(
        input_path=input_path,
        output_path=output_path,
        k=config["baseline"],
        require_abstract=config.get("require_abstract", True),
        file_start=file_start,
        file_end=file_end,
    )

    loader.run_loader()

    # Generate statistics report
    loader._write_statistics_report()

    if config["count_articles"]:
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
            run_pubmed_loading(loader_config)
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
