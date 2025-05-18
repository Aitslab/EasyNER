import json
import os
from typing import Any, Optional

import orjson

from .pubmed_base_loader import BasePubMedLoader


class PubMedJSONLoader(BasePubMedLoader):
    """PubMed XML to JSON loader with abstract filtering."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        k: str,
        require_abstract: bool = False,
        file_start: Optional[int] = None,
        file_end: Optional[int] = None,
    ) -> None:
        """Initialize the PubMedJSONLoader.

        Args:
            input_path: Directory containing input XML files
            output_path: Directory where processed JSON files will be written
            k: Baseline identifier used in filename parsing
            require_abstract: If True, articles without abstracts are filtered out
            file_start: Optional start index for file range processing
            file_end: Optional end index for file range processing

        """
        super().__init__(input_path, output_path, k, file_start, file_end)
        self.require_abstract = require_abstract
        self.counter = {}
        self.filter_stats = {
            "total_articles": 0,
            "no_abstract": 0,
            "abstract_not_string": 0,
            "empty_abstract": 0,
            "included_articles": 0,
        }
        self.current_input_file = None

    def _process_article_data(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Process article data and apply filtering based on abstract requirements.

        Args:
            data: List of article data dictionaries from pubmed_parser

        Returns:
            Dictionary of processed articles with PMID as key

        """
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

        self.counter[self.current_input_file] = local_stats
        return article

    def _write_output(self, data: dict[str, Any], input_file: str) -> None:
        """Write processed article data to a JSON file.

        Args:
            data: Dictionary of articles to write
            input_file: Original input file path

        """
        outfile = os.path.join(
            self.output_path,
            os.path.basename(input_file.split(".xml")[0]) + ".json",
        )
        with open(outfile, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def get_filter_statistics(self) -> dict[str, int]:
        """Return statistics about filtered articles."""
        return self.filter_stats

    def get_counter(self) -> dict[str, dict[str, int]]:
        """Return per-file statistics."""
        return self.counter

    def _write_statistics_report(
        self,
        output_file: str = "filter_statistics.json",
    ) -> None:
        """Write filtering statistics to a JSON file.

        Args:
            output_file: Name of the output statistics file

        """
        report_path = os.path.join(self.output_path, output_file)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {"overall": self.filter_stats, "by_file": self.counter},
                f,
                indent=2,
                ensure_ascii=False,
            )
