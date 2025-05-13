"""PubMed Bulk Loader.

This script loads PubMed XML files, converts them to JSON format,
and filters articles based on specified criteria.
It also generates statistics about the filtering process and
counts the number of articles in the converted JSON files.
"""

import json
import os
from glob import glob
from pathlib import Path
from typing import Any

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
    ) -> None:
        """Initialize the PubMedLoader."""
        self.input_path = input_path
        self.output_path = output_path
        self.counter = {}
        self.k = k
        self.require_abstract = require_abstract
        self.filter_stats = {
            "total_articles": 0,
            "no_abstract": 0,
            "abstract_not_string": 0,
            "empty_abstract": 0,
            "included_articles": 0,
        }
        os.makedirs(output_path, exist_ok=True)

    def _get_input_files(self, input_path: str) -> list[str]:
        # k is used for keyword to split the filename obtained from pubmed.
        # It's different for each annual baseline
        input_files = sorted(
            glob(f"{input_path}*.gz"),
            key=lambda x: int(
                os.path.splitext(os.path.basename(x))[0].split(self.k + "n")[-1][:-4],
            ),
        )
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
        input_files_list = self._get_input_files(self.input_path)

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


def count_articles(input_path: str, baseline: int = 23) -> None:
    """Count articles from converted json files."""
    count = 0
    pmids = []
    # k is used for keyword to split the filename obtained from pubmed.
    # It's different for each annual baseline
    k = str(baseline) + "n"
    count_file = input_path + "counts.txt"
    pmid_file = input_path + "pmid_list.txt"
    input_files = sorted(
        glob(f"{input_path}*.json"),
        key=lambda x: int(
            os.path.splitext(os.path.basename(x))[0].split(k)[-1],
        ),
    )

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
    # Get paths from config
    download_path = (
        "data/tmp/pubmed/"
        if len(config["raw_download_path"]) == 0
        else config["raw_download_path"]
    )
    output_path = config["output_path"]

    print("Processing PubMed raw files...")

    # Pass the require_abstract option to the loader
    loader = PubMedLoader(
        input_path=download_path,
        output_path=output_path,
        k=config["baseline"],
        require_abstract=config.get("require_abstract", True),
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
