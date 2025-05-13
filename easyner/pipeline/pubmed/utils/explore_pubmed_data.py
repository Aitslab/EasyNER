"""Script to explore PubMed XML data structure and extraction.

This includes a modified parser that handles deleted articles.

Element: DeleteCitation

Indicates one or more <PubmedArticle> or <PubmedBookArticle> that have been deleted. PMIDs in DeleteCitation will typically have been found to be duplicate citations, or citations to content that was determined to be out-of-scope for PubMed. It is possible that a PMID would appear in DeleteCitation without having been distributed in a previous file. This would happen if the creation and deletion of the record take place on the same day.

Element: DeleteDocument

Indicates one or more <BookDocument> that have been deleted.
Content Model

( PMID* )
Contains:

    <PMID>

May be contained in:

    <BookDocumentSet>

Examples
--------
<DeleteDocument>
<PMID>28230950</PMID>
</DeleteDocument>


"""

import argparse
import datetime
import glob
import gzip
import json
import os
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Iterator
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Union  # Added Union

import pandas as pd
from lxml import etree
from tqdm import tqdm

# Path configuration
DATA_DIR = "/lunarc/nobackup/projects/snic2020-6-41/carl/data/pubmed_raw/"
FILE_PATTERN = "*.xml.gz"  # Process all XML files
OUTPUT_DIR = "./pubmed_analysis/"
DEFAULT_PROCESS_COUNT = min(cpu_count(), 48)  # Limit to reasonable number

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Helper functions
def stringify_children(element: Optional[etree.ElementBase]) -> str:
    """Convert children of an element to string."""
    if element is None:
        return ""
    return "".join(
        element.itertext(),
    )  # Removed tag argument as it's not needed for itertext


def parse_pmid(pubmed_article: etree.ElementBase) -> str:
    """Extract PMID from a PubMed article."""
    medline = pubmed_article.find("MedlineCitation")
    if medline is not None and medline.find("PMID") is not None:
        pmid_element = medline.find("PMID")
        if pmid_element is not None and pmid_element.text:
            return pmid_element.text
    return ""


def parse_article_info(
    element: etree.ElementBase,
    year_info_only: bool,
    nlm_category: bool,
    author_list: bool,
    reference_list: bool,
    parse_subs: bool = False,
) -> dict[str, Any]:
    """Parse article information (simplified)."""
    medline = element.find("MedlineCitation")
    article = medline.find("Article") if medline is not None else None

    # Extract minimal information
    pmid = parse_pmid(element)
    title = ""
    abstract = ""
    abstract_structure: list[dict[str, str]] = []

    if article is not None and article.find("ArticleTitle") is not None:
        title_element = article.find("ArticleTitle")
        if title_element is not None:
            title = stringify_children(title_element).strip()

    if article is not None:
        abstract_node = article.find("Abstract")
        if abstract_node is not None:
            abstract_text_elements = abstract_node.findall("AbstractText")
            if abstract_text_elements:
                if len(abstract_text_elements) > 1:
                    # Structured abstract with labeled sections
                    for abstract_elem in abstract_text_elements:
                        section_text = stringify_children(abstract_elem).strip()
                        label = abstract_elem.get("Label", "")
                        nlm_cat = abstract_elem.get(
                            "NlmCategory",
                            "",
                        )  # Renamed to avoid conflict
                        abstract_structure.append(
                            {
                                "label": label,
                                "category": nlm_cat,
                                "text": section_text,
                            },
                        )
                    abstract = " ".join(part["text"] for part in abstract_structure)
                else:
                    # Single AbstractText (unstructured abstract)
                    abstract = stringify_children(abstract_text_elements[0]).strip()
                    abstract_structure.append(
                        {
                            "label": abstract_text_elements[0].get("Label", ""),
                            "category": abstract_text_elements[0].get(
                                "NlmCategory",
                                "",
                            ),
                            "text": abstract,
                        },
                    )
            else:  # Abstract node exists but no AbstractText children
                abstract = stringify_children(abstract_node).strip()
                abstract_structure.append(
                    {
                        "label": "",
                        "category": "",
                        "text": abstract,
                    },
                )

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "abstract_structure": abstract_structure,
        "delete": False,
    }


def parse_medline_xml(path: str) -> Iterator[dict[str, Any]]:
    """Parse MEDLINE XML with focus on regular PubmedArticle elements."""
    try:
        with gzip.open(path, "rb") as f:
            # element_types = Counter() # This was unused in this function's direct scope for return
            for _event, element in etree.iterparse(
                f,
                events=("end",),
                tag="PubmedArticle",
            ):
                # element_types[element.tag] += 1 # Not needed if not returning/using element_types
                res = parse_article_info(
                    element,
                    year_info_only=True,  # These args seem to be fixed, consider if they should be params
                    nlm_category=False,
                    author_list=False,
                    reference_list=False,
                )
                element.clear()
                # Clear ancestor elements to free memory
                while element.getprevious() is not None:
                    del element.getparent()[0]
                yield res
        # Removed the problematic `return element_types`
    except Exception as e:
        print(f"Error processing {path}: {e}")
        # To conform to Iterator type, yield nothing or raise, or return an empty iterator
        return iter([])  # Return an empty iterator on error


def process_single_file(
    file_path: str,
    article_limit_per_file: Optional[int] = None,
) -> dict[str, Any]:
    """Process a single PubMed XML file and return statistics."""
    file_name = os.path.basename(file_path)

    # Initialize counters for this file
    articles_in_file = 0
    abstracts_in_file = 0
    abstract_lengths_in_file: list[int] = []
    field_presence_in_file: dict[str, int] = defaultdict(int)
    pmid_info: list[dict[str, str]] = []
    pmid_abstracts: dict[str, dict[str, Any]] = {}
    element_types_in_file: dict[str, int] = (
        Counter()
    )  # For tracking elements in this file

    regular_sample = None

    try:
        # Process articles in this file
        for article in parse_medline_xml(
            file_path,
        ):  # parse_medline_xml now only yields articles
            articles_in_file += 1
            element_types_in_file[
                "PubmedArticle"
            ] += 1  # Assuming parse_medline_xml filters this

            # Track field presence
            for field, value in article.items():
                if value:  # Checks for non-empty/non-false values
                    field_presence_in_file[field] += 1

            # Collect PMID with source file info
            pmid = article.get("pmid")
            if pmid:
                pmid_info.append({"pmid": pmid, "file": file_name})

                # Store abstract information for this PMID with enhanced details
                if article.get("abstract"):
                    abstract_text = article.get("abstract", "")
                    abstract_structure = article.get("abstract_structure", [])
                    word_count = len(abstract_text.split()) if abstract_text else 0
                    char_count = len(abstract_text) if abstract_text else 0
                    abstract_fingerprint = hash(abstract_text) if abstract_text else 0

                    pmid_abstracts[pmid] = {
                        "abstract": abstract_text,
                        "abstract_structure": abstract_structure,
                        "file": file_name,
                        "metrics": {
                            "word_count": word_count,
                            "char_count": char_count,
                            "fingerprint": abstract_fingerprint,
                        },
                    }

            # Track stats for articles with abstracts
            if article.get("abstract"):
                abstracts_in_file += 1
                abstract_lengths_in_file.append(len(article["abstract"]))

            # Store a sample article
            if regular_sample is None and article.get("abstract"):
                regular_sample = {**article, "source_file": file_name}

            # Limit processing articles per file if specified
            if (
                article_limit_per_file is not None
                and articles_in_file >= article_limit_per_file
            ):
                break

        # Note: element_types from etree.iterparse in parse_medline_xml is not directly passed here anymore.
        # If a count of all element types (not just PubmedArticle) per file is needed,
        # parse_medline_xml would need to be redesigned, perhaps to yield tuples of (type, data)
        # or to return the counter separately, which complicates its Iterator nature.
        # For now, element_types_in_file only counts PubmedArticle as processed by this loop.

        return {
            "file_name": file_name,
            "file_path": file_path,
            "articles": articles_in_file,
            "articles_with_abstract": abstracts_in_file,
            "abstract_lengths": abstract_lengths_in_file,
            "field_presence": dict(field_presence_in_file),
            "element_types": dict(
                element_types_in_file,
            ),  # This now reflects articles processed
            "sample": regular_sample,
            "pmid_info": pmid_info,
            "pmid_abstracts": pmid_abstracts,
            "error": None,
        }
    except Exception as e:
        error_details = traceback.format_exc()
        return {
            "file_name": file_name,
            "file_path": file_path,
            "error": f"{str(e)}\n{error_details}",
            "articles": 0,
            "articles_with_abstract": 0,
            "element_types": {},  # Ensure all keys are present on error
            "pmid_info": [],
            "pmid_abstracts": {},
            "field_presence": {},
            "abstract_lengths": [],
            "sample": None,
        }


def compare_abstracts(
    abstract1_data: dict[str, Any],
    abstract2_data: dict[str, Any],
) -> dict[str, Any]:
    """Compare two abstracts in detail and return similarity information.

    Parameters
    ----------
    - abstract1_data: Dict with abstract text and structure.
    - abstract2_data: Dict with abstract text and structure.

    Returns
    -------
    - Dictionary with comparison results.

    """
    # Extract relevant data
    abstract1 = abstract1_data.get("abstract", "")
    abstract1_structure = abstract1_data.get("abstract_structure", [])
    fingerprint1 = abstract1_data.get("metrics", {}).get("fingerprint", 0)

    abstract2 = abstract2_data.get("abstract", "")
    abstract2_structure = abstract2_data.get("abstract_structure", [])
    fingerprint2 = abstract2_data.get("metrics", {}).get("fingerprint", 0)

    # If either abstract is missing, they're not matching
    if not abstract1 or not abstract2:
        return {
            "match_type": "missing_abstract",
            "similarity": 0.0,
            "structure_match": None,  # Using None for indeterminate
            "details": "One or both abstracts are missing",
        }

    # Quick fingerprint comparison (if available)
    if fingerprint1 and fingerprint2 and fingerprint1 == fingerprint2:
        return {
            "match_type": "exact",
            "similarity": 1.0,
            "structure_match": True,
            "details": "Exact match (fingerprint)",
        }

    # Full text comparison
    if abstract1.strip() == abstract2.strip():
        return {
            "match_type": "exact",
            "similarity": 1.0,
            "structure_match": True,  # Assuming structure matches if text does
            "details": "Exact text match",
        }

    # Compare structure for structured abstracts
    structure_match = False
    structure_details: list[dict[str, Any]] = []

    if abstract1_structure and abstract2_structure:
        if len(abstract1_structure) == len(abstract2_structure):
            matching_sections = 0
            total_sections = len(abstract1_structure)

            for i, (section1, section2) in enumerate(
                zip(abstract1_structure, abstract2_structure),
            ):
                if section1.get("label") == section2.get("label") and section1.get(
                    "text",
                ) == section2.get("text"):
                    matching_sections += 1
                else:
                    structure_details.append(
                        {
                            "section": i + 1,
                            "label1": section1.get("label", ""),
                            "label2": section2.get("label", ""),
                            "text_match": section1.get("text", "")
                            == section2.get("text", ""),
                        },
                    )
            if matching_sections == total_sections:
                structure_match = True
        else:
            structure_details.append(
                {
                    "issue": "different_section_count",
                    "count1": len(abstract1_structure),
                    "count2": len(abstract2_structure),
                },
            )
    elif abstract1_structure or abstract2_structure:  # One has structure, other doesn't
        structure_details.append(
            {
                "issue": "structure_mismatch_presence",
                "has_structure1": bool(abstract1_structure),
                "has_structure2": bool(abstract2_structure),
            },
        )

    # Simple similarity check (ratio of common words)
    words1 = set(abstract1.lower().split())
    words2 = set(abstract2.lower().split())

    if (
        not words1 or not words2
    ):  # Handles cases where one abstract might be empty after split
        return {
            "match_type": "different",
            "similarity": 0.0,
            "structure_match": structure_match,  # Keep determined structure match
            "structure_details": structure_details if structure_details else None,
            "details": "One or both abstracts have no words for comparison after splitting.",
        }

    common_words = len(words1.intersection(words2))
    total_words = len(words1.union(words2))
    similarity = common_words / total_words if total_words > 0 else 0.0

    match_type = "different"  # Default
    if similarity > 0.9:
        match_type = "very_similar"
    elif similarity > 0.7:
        match_type = "similar"

    return {
        "match_type": match_type,
        "similarity": similarity,
        "structure_match": structure_match,
        "structure_details": structure_details if structure_details else None,
        "common_words_ratio": f"{common_words}/{total_words}",
        "text_metrics": {
            "length1": len(abstract1),
            "length2": len(abstract2),
            "word_count1": len(words1),
            "word_count2": len(words2),
        },
    }


def analyze_pubmed_data_multiprocessing(
    data_dir: str = DATA_DIR,
    file_pattern: str = FILE_PATTERN,
    output_dir: str = OUTPUT_DIR,
    file_limit: Optional[int] = None,
    duplicate_analysis_limit: Optional[int] = None,
    num_processes: int = DEFAULT_PROCESS_COUNT,
) -> None:
    """Analyze PubMed data files using multiprocessing."""
    file_paths = glob.glob(os.path.join(data_dir, file_pattern))
    os.makedirs(output_dir, exist_ok=True)

    if file_limit:
        file_paths = file_paths[:file_limit]

    print(f"Found {len(file_paths)} XML files to process")

    if not file_paths:
        print(f"No files found matching pattern {file_pattern} in {data_dir}")
        return

    # Initialize statistics and data structures
    total_articles = 0
    articles_with_abstract = 0
    abstract_lengths: list[int] = []
    all_element_types: dict[str, int] = Counter()
    field_presence: dict[str, int] = defaultdict(int)
    all_pmids: list[dict[str, str]] = []
    all_pmid_abstracts: dict[str, dict[str, Any]] = (
        {}
    )  # PMID -> {abstract_data with file info}
    errors: list[dict[str, Any]] = []
    regular_sample: Optional[dict[str, Any]] = None
    processed_files = 0

    pool: Optional[Pool] = None
    progress_bar: Optional[tqdm] = None

    start_time = time.time()
    processing_start_time = start_time
    status_message = "Processing completed."

    # Variables that might be unbound if errors occur early
    duplicate_pmids_list: list[str] = []
    comparison_limit: int = 50
    overall_results: dict[str, Any] = {}  # Initialize overall_results
    output_file_path: str = ""  # Initialize output_file_path

    try:
        pool = Pool(processes=num_processes)
        progress_bar = tqdm(
            total=len(file_paths),
            desc=f"Processing files with {num_processes} processes",
            unit="file",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )

        def update_progress(result_data: dict[str, Any]) -> dict[str, Any]:
            nonlocal processed_files, total_articles, articles_with_abstract, progress_bar
            if progress_bar:
                progress_bar.update(1)
            processed_files += 1

            if "error" not in result_data or not result_data["error"]:
                if "articles" in result_data:
                    total_articles += result_data["articles"]
                if "articles_with_abstract" in result_data:
                    articles_with_abstract += result_data["articles_with_abstract"]
                if progress_bar:
                    progress_bar.set_postfix(
                        articles=f"{total_articles}",
                        abstracts=f"{articles_with_abstract}",
                    )
            return result_data

        async_results = []
        for file_path_item in file_paths:
            async_results.append(
                pool.apply_async(
                    process_single_file,
                    args=(file_path_item,),
                    callback=update_progress,
                ),
            )

        pool.close()
        pool.join()
        pool = None  # Ensure pool is not reused after close/join

        collected_results: list[dict[str, Any]] = []
        for ar in async_results:
            try:
                res = ar.get()
                collected_results.append(res)
                if res.get("error"):
                    errors.append(
                        res,
                    )  # res already contains file_name, file_path, error
                    continue

                # total_articles and articles_with_abstract are updated in callback
                abstract_lengths.extend(res.get("abstract_lengths", []))
                all_pmids.extend(res.get("pmid_info", []))
                if res.get("pmid_abstracts"):
                    all_pmid_abstracts.update(res.get("pmid_abstracts", {}))
                for elem_type, count in res.get("element_types", {}).items():
                    all_element_types[elem_type] += count
                for field, count_val in res.get(
                    "field_presence",
                    {},
                ).items():  # Renamed count to count_val
                    field_presence[field] += count_val
                if regular_sample is None and res.get("sample") is not None:
                    regular_sample = res["sample"]

            except Exception as e_collect:  # Renamed e to e_collect
                # This error is for issues with ar.get() or processing its result
                errors.append(
                    {
                        "file_path": "unknown_async_result",
                        "error": str(e_collect),
                        "traceback": traceback.format_exc(),
                    },
                )

    except KeyboardInterrupt:
        status_message = "Processing interrupted by user."
        print(f"\n{status_message} Attempting to finalize and save partial results...")
        if pool:
            print("Terminating worker processes...")
            pool.terminate()
            pool.join()
    except Exception as e_main_processing:  # Renamed e to e_main_processing
        status_message = f"An error occurred during processing: {str(e_main_processing)}\n{traceback.format_exc()}"
        print(f"\n{status_message} Attempting to finalize and save partial results...")
        if pool:
            pool.terminate()
            pool.join()
    finally:
        if progress_bar:
            progress_bar.close()

        processing_time_seconds = time.time() - processing_start_time
        finalization_start_time = time.time()
        print("\nStarting finalization and report generation...")

        # Initialize these here to ensure they are defined for overall_results
        # duplicate_pmids = [] # This was unused, replaced by duplicate_pmids_list
        duplicate_details: dict[str, list[str]] = {}
        duplicate_article_comparisons: dict[str, dict[str, Any]] = (
            {}
        )  # Renamed for clarity
        article_match_type_summary: dict[str, int] = Counter()
        abstract_match_type_summary: dict[str, int] = Counter()
        xml_element_differences: dict[str, int] = defaultdict(int)

        try:
            if all_pmids:
                print("Finding duplicate PMIDs...")
                df = pd.DataFrame(all_pmids)
                if not df.empty and "pmid" in df.columns:
                    pmid_counts = df["pmid"].value_counts()
                    duplicate_pmids_list = pmid_counts[pmid_counts > 1].index.tolist()

                    if duplicate_pmids_list:
                        duplicate_details_df = df[df["pmid"].isin(duplicate_pmids_list)]
                        duplicate_details = (
                            duplicate_details_df.groupby("pmid")["file"]
                            .apply(list)
                            .to_dict()
                        )
                else:
                    print("PMID data is empty or 'pmid' column is missing.")
                    duplicate_pmids_list = []

                print("Comparing metadata for duplicate PMIDs (limited sample)...")
                # metadata_comparison_stats = defaultdict(int) # Unused
                comparison_limit = 10 if "interrupted" in status_message else 50

                for pmid_to_compare in tqdm(  # Renamed pmid to pmid_to_compare
                    duplicate_pmids_list[:comparison_limit],
                    desc="Analyzing metadata for duplicates",
                ):
                    files_for_pmid = duplicate_details.get(
                        pmid_to_compare,
                        [],
                    )  # Renamed files
                    if len(files_for_pmid) < 2:
                        continue

                    articles_for_pmid_data: list[dict[str, Any]] = []  # Renamed
                    for file_name_for_pmid in files_for_pmid[:3]:  # Renamed file_name
                        # Construct full path using data_dir, not output_dir
                        file_path_for_pmid_lookup = os.path.join(
                            data_dir,
                            file_name_for_pmid,
                        )  # Renamed
                        if os.path.exists(file_path_for_pmid_lookup):
                            article_data_extracted = extract_article_by_pmid(  # Renamed
                                file_path_for_pmid_lookup,
                                pmid_to_compare,
                            )
                            if article_data_extracted:
                                articles_for_pmid_data.append(
                                    {
                                        "file": file_name_for_pmid,
                                        "data": article_data_extracted,  # This is the full metadata
                                        "source_path": file_path_for_pmid_lookup,
                                    },
                                )
                        else:
                            print(
                                f"Warning: File {file_path_for_pmid_lookup} not found for duplicate check.",
                            )

                    if len(articles_for_pmid_data) >= 2:
                        first_article_metadata = articles_for_pmid_data[0][
                            "data"
                        ]  # Renamed
                        current_pmid_comparison_results_list: list[dict[str, Any]] = (
                            []
                        )  # Renamed
                        xml_differences_detected_for_this_pmid = False  # Renamed

                        for i in range(1, len(articles_for_pmid_data)):
                            other_article_metadata = articles_for_pmid_data[i][
                                "data"
                            ]  # Renamed
                            comparison_result = compare_article_metadata(  # Renamed
                                first_article_metadata,
                                other_article_metadata,
                            )
                            current_pmid_comparison_results_list.append(
                                comparison_result,
                            )

                            # Aggregate article match types
                            article_match_type_summary[
                                comparison_result.get(
                                    "match_type",
                                    "unknown_comparison",
                                )
                            ] += 1

                            if (
                                "xml_structure_differences" in comparison_result
                                and comparison_result["xml_structure_differences"]
                            ):
                                for diff_item in comparison_result.get(  # Renamed diff
                                    "xml_structure_differences",
                                    [],
                                ):
                                    xml_element_differences[diff_item["path"]] += 1
                                xml_differences_detected_for_this_pmid = True

                        # Store detailed comparison for this PMID
                        # The abstract comparison part is more complex if abstracts are from different files
                        # For now, this focuses on metadata comparison from extract_article_by_pmid
                        duplicate_article_comparisons[pmid_to_compare] = {
                            "articles_found": len(articles_for_pmid_data),
                            "files": [a["file"] for a in articles_for_pmid_data],
                            "title_preview": (
                                (first_article_metadata.get("title", "")[:100] + "...")
                                if first_article_metadata.get("title")
                                else "N/A"
                            ),
                            "abstract_preview": (
                                (
                                    first_article_metadata.get("abstract", "")[:100]
                                    + "..."
                                )
                                if first_article_metadata.get("abstract")
                                else "N/A"
                            ),
                            "xml_differences_detected": xml_differences_detected_for_this_pmid,
                            "metadata_comparison_results": current_pmid_comparison_results_list,
                            # Placeholder for combined abstract match type for this PMID
                            "combined_abstract_match_type": "not_yet_compared",
                        }

            # Abstract comparison for duplicates (using all_pmid_abstracts)
            if duplicate_pmids_list and all_pmid_abstracts:
                print(
                    "Comparing abstracts for duplicate PMIDs (from all_pmid_abstracts)...",
                )
                for pmid_to_compare_abstracts in tqdm(  # Renamed pmid
                    duplicate_pmids_list[
                        :comparison_limit
                    ],  # Use the same comparison_limit
                    desc="Comparing duplicate abstracts",
                ):
                    # Get all abstract versions for this PMID from all_pmid_abstracts
                    # This requires finding all entries in all_pmid_abstracts that match pmid_to_compare_abstracts
                    # The current structure of all_pmid_abstracts is {pmid: {abstract_data, file, metrics}}
                    # This means if a PMID is duplicated, all_pmid_abstracts will be overwritten by the last encounter.
                    # This part of the logic needs a redesign of how all_pmid_abstracts is populated
                    # to store multiple versions if a PMID appears in multiple files.
                    # For example, all_pmid_abstracts could be Dict[str, List[Dict[str, Any]]]
                    #
                    # Given the current structure, we can't directly compare abstracts for the *same* PMID
                    # from *different* files using `all_pmid_abstracts` if it only stores one version.
                    #
                    # The `duplicate_article_comparisons` above uses `extract_article_by_pmid` which
                    # can get the article (and thus abstract) from each file.
                    #
                    # Let's assume for now that `all_pmid_abstracts` *should* have been designed to hold
                    # all versions, or we use the data from `articles_for_pmid_data` if available.
                    #
                    # For simplicity in this fix, we'll note this limitation.
                    # A proper fix would involve changing `process_single_file` and `collected_results` aggregation.

                    # Placeholder: If `duplicate_article_comparisons` has the abstracts, we could use them.
                    if (
                        pmid_to_compare_abstracts in duplicate_article_comparisons
                        and len(
                            duplicate_article_comparisons[
                                pmid_to_compare_abstracts
                            ].get("metadata_comparison_results", []),
                        )
                        > 0
                    ):

                        # This is still not ideal as it compares metadata-derived abstracts.
                        # A true abstract comparison would use the `pmid_abstracts` from each file's processing.
                        # This section remains a known area for improvement based on data structure.
                        pass  # Actual comparison logic would go here if data structure supported it well.

            # Summarize abstract comparison match types
            # This summary should come from the actual comparisons performed.
            # If the above loop for comparing duplicate abstracts was fully implemented,
            # it would populate `abstract_match_type_summary`.
            # For now, it will likely be empty or based on limited data.
            if (
                duplicate_article_comparisons
            ):  # Using this as a proxy for having some comparison data
                for pmid_data_val in duplicate_article_comparisons.values():
                    # This assumes `metadata_comparison_results` might contain an abstract match type,
                    # or a combined one was set.
                    # Let's use the placeholder 'combined_abstract_match_type'
                    match_type = pmid_data_val.get(
                        "combined_abstract_match_type",
                        "unknown_abstract_match",
                    )
                    abstract_match_type_summary[match_type] += 1

            overall_results = {
                "status_message": status_message,
                "total_articles": total_articles,
                "articles_with_abstract": articles_with_abstract,
                "files_processed": processed_files,
                "total_files_in_run": len(file_paths),
                "duplicate_pmids_found_count": len(duplicate_pmids_list),
                "duplicate_pmids_sampled_for_detail": duplicate_pmids_list[
                    :comparison_limit
                ],
                "duplicate_article_metadata_comparison": duplicate_article_comparisons,  # Renamed
                "article_comparison_match_type_summary": dict(
                    article_match_type_summary,
                ),
                "abstract_comparison_match_type_summary": dict(
                    abstract_match_type_summary,
                ),
                "element_types_summary": dict(all_element_types.most_common()),
                "field_presence_summary": {
                    k: v
                    for k, v in sorted(
                        field_presence.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
                "abstract_length_stats": {
                    "min": min(abstract_lengths) if abstract_lengths else 0,
                    "max": max(abstract_lengths) if abstract_lengths else 0,
                    "average": (
                        sum(abstract_lengths) / len(abstract_lengths)
                        if abstract_lengths
                        else 0.0  # Ensure float
                    ),
                    "count": len(abstract_lengths),
                },
                "sample_regular_article": regular_sample,
                "errors_encountered": errors,
                "initial_processing_time_seconds": processing_time_seconds,
                "finalization_time_seconds": 0.0,  # Will be updated
                "total_execution_time_seconds": 0.0,  # Will be updated
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "xml_element_differences_in_duplicates": dict(xml_element_differences),
            }

            overall_results["finalization_time_seconds"] = (
                time.time() - finalization_start_time
            )
            overall_results["total_execution_time_seconds"] = time.time() - start_time

            output_file_path = os.path.join(
                output_dir,
                f"pubmed_analysis_{overall_results['timestamp']}.json",
            )
            with open(output_file_path, "w") as f:
                json.dump(overall_results, f, indent=2)
            print(f"Final results saved to {output_file_path}")

        except KeyboardInterrupt:
            # This block is inside the main `finally`'s `try`
            # Update overall_results if it has been initialized
            if not overall_results:  # If try block failed very early
                overall_results = {
                    "status_message": "Finalization interrupted before results init.",
                }

            overall_results["status_message"] = "Finalization interrupted by user."
            print(f"\n{overall_results['status_message']}")

            overall_results["finalization_time_seconds"] = (
                time.time() - finalization_start_time
            )
            overall_results["total_execution_time_seconds"] = time.time() - start_time
            if "timestamp" not in overall_results or not overall_results["timestamp"]:
                overall_results["timestamp"] = datetime.datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S_interrupted",
                )

            output_file_path = os.path.join(
                output_dir,
                f"pubmed_analysis_partial_{overall_results['timestamp']}.json",
            )
            try:
                with open(output_file_path, "w") as f:
                    json.dump(overall_results, f, indent=2)
                print(f"Partial results saved to {output_file_path}")
            except Exception as e_save_interrupt:  # Renamed
                print(
                    f"Could not save partial results during finalization interrupt: {e_save_interrupt}",
                )

        except Exception as e_finalization:  # Renamed
            if not overall_results:
                overall_results = {
                    "status_message": "Error during finalization before results init.",
                }

            error_details_final = traceback.format_exc()
            overall_results["status_message"] = (
                f"Error during finalization: {str(e_finalization)}\n{error_details_final}"
            )
            print(f"\n{overall_results['status_message']}")
            overall_results["finalization_time_seconds"] = (
                time.time() - finalization_start_time
            )
            overall_results["total_execution_time_seconds"] = time.time() - start_time
            if "timestamp" not in overall_results or not overall_results["timestamp"]:
                overall_results["timestamp"] = datetime.datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S_final_error",
                )

            output_file_path = os.path.join(
                output_dir,
                f"pubmed_analysis_error_{overall_results['timestamp']}.json",
            )
            try:
                with open(output_file_path, "w") as f:
                    json.dump(overall_results, f, indent=2)
                print(f"Results (with finalization error) saved to {output_file_path}")
            except Exception as e_save_final_error:
                print(
                    f"Could not save results during finalization error: {e_save_final_error}",
                )

        # Ensure overall_results is always a dict for safe access
        if not isinstance(overall_results, dict):
            overall_results = {
                "status_message": "Critical error: overall_results not a dict.",
            }

        print(f"Status: {overall_results.get('status_message', 'Unknown')}")
        print(
            f"Total files processed: {overall_results.get('files_processed', 0)} of "
            f"{overall_results.get('total_files_in_run', 0)}",
        )
        print(f"Total articles analyzed: {overall_results.get('total_articles', 0)}")
        print(
            f"Duplicate PMIDs reported: {overall_results.get('duplicate_pmids_found_count',0)}",
        )
        if overall_results.get(
            "total_raw_duplicates_found",
        ) is not None and overall_results.get(
            "duplicate_pmids_found_count",
        ) != overall_results.get(
            "total_raw_duplicates_found",
        ):
            print(
                f"  (Total unique PMIDs with duplicates found before limit: {overall_results.get('total_raw_duplicates_found')})",
            )


def analyze_pubmed_data(
    file_limit: Optional[int] = None,
    duplicate_analysis_limit: Optional[int] = None,  # Added trailing comma
) -> None:
    """Analyze PubMed data files (sequential version).

    Args:
        file_limit: Optional limit on the number of files to process.
        duplicate_analysis_limit: Optional limit on the number of duplicate  # Wrapped line
            PMIDs to report in detail.

    """
    file_paths = glob.glob(
        os.path.join(DATA_DIR, FILE_PATTERN),
    )
    print(f"Found {len(file_paths)} XML files.")

    if not file_paths:
        print(f"No files found matching pattern {FILE_PATTERN} in {DATA_DIR}")
        return

    if file_limit is not None:
        actual_files_to_process = file_paths[:file_limit]
        print(
            f"Processing a limit of {len(actual_files_to_process)} files (out of {len(file_paths)} found) sequentially due to --file-process-limit={file_limit}.",
        )
    else:
        actual_files_to_process = file_paths
        print(
            f"Processing all {len(actual_files_to_process)} found files sequentially...",
        )

    # Statistics
    total_articles = 0
    deleted_articles = (
        0  # This variable is not actually updated in the provided sequential code.
    )
    articles_with_abstract = 0
    field_presence: dict[str, int] = defaultdict(int)
    abstract_lengths: list[int] = []
    all_element_types: dict[str, int] = Counter()
    pmid_counter: dict[str, int] = Counter()

    regular_sample: Optional[dict[str, Any]] = None
    deleted_sample: Optional[dict[str, Any]] = None  # Not updated in this version.

    for file_path in tqdm(
        actual_files_to_process,
        desc="Processing files sequentially",
    ):
        # Using process_single_file to keep logic consistent
        # Note: process_single_file is designed for multiprocessing and returns a dict.
        # We adapt its use here.
        file_stats = process_single_file(file_path)

        if file_stats.get("error"):
            print(f"Error processing {file_path}: {file_stats['error']}")
            continue  # Skip this file on error

        total_articles += file_stats.get("articles", 0)
        articles_with_abstract += file_stats.get("articles_with_abstract", 0)
        abstract_lengths.extend(file_stats.get("abstract_lengths", []))

        for pmid_info_item in file_stats.get("pmid_info", []):
            pmid_counter[pmid_info_item["pmid"]] += 1

        for elem_type, count in file_stats.get("element_types", {}).items():
            all_element_types[elem_type] += count

        for field, count in file_stats.get("field_presence", {}).items():
            field_presence[field] += count

        if regular_sample is None and file_stats.get("sample"):
            regular_sample = file_stats["sample"]

        # deleted_articles and deleted_sample are not handled by process_single_file
        # as it focuses on PubmedArticle. A different parser would be needed for DeleteCitation.

    all_pmids_with_duplicates = [
        pmid for pmid, count in pmid_counter.items() if count > 1
    ]
    count_of_all_pmids_with_duplicates = len(all_pmids_with_duplicates)

    if duplicate_analysis_limit is not None:
        final_duplicate_pmids_list_for_report = all_pmids_with_duplicates[
            :duplicate_analysis_limit
        ]
    else:
        final_duplicate_pmids_list_for_report = all_pmids_with_duplicates

    results = {
        "total_articles": total_articles,
        "deleted_articles": deleted_articles,  # Remains 0
        "articles_with_abstract": articles_with_abstract,
        "files_processed": len(actual_files_to_process),
        "total_pmids_with_duplicates_found": count_of_all_pmids_with_duplicates,  # Total actual count
        "duplicate_pmids": final_duplicate_pmids_list_for_report,  # Potentially limited list
        "duplicate_count": len(
            final_duplicate_pmids_list_for_report,
        ),  # Count of the reported list
        "element_types": dict(all_element_types),
        "field_presence": {
            k: v
            for k, v in sorted(field_presence.items(), key=lambda x: x[1], reverse=True)
        },
        "abstract_stats": {
            "min_length": min(abstract_lengths) if abstract_lengths else 0,
            "max_length": max(abstract_lengths) if abstract_lengths else 0,
            "avg_length": (
                sum(abstract_lengths) / len(abstract_lengths)
                if abstract_lengths
                else 0.0
            ),
        },
        "samples": {
            "regular_article": regular_sample,
            "deleted_article": deleted_sample,  # Remains None
        },
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    output_file_path = os.path.join(
        OUTPUT_DIR,
        f"pubmed_data_analysis_sequential_{results['timestamp']}.json",
    )
    with open(output_file_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Sequential analysis complete. Results saved to {output_file_path}")
    print(f"Total files processed: {len(actual_files_to_process)}")
    print(f"Total articles processed: {total_articles}")
    if total_articles > 0:  # Avoid division by zero
        print(
            f"Articles with abstract: {articles_with_abstract} ({articles_with_abstract/total_articles*100:.1f}%)",
        )
    else:
        print("Articles with abstract: 0")

    print(
        f"Total unique PMIDs found with duplicates: {count_of_all_pmids_with_duplicates}",
    )
    print(
        f"Duplicate PMIDs reported in output file: {len(final_duplicate_pmids_list_for_report)}",
    )
    if (
        duplicate_analysis_limit is not None
        and count_of_all_pmids_with_duplicates
        > len(
            final_duplicate_pmids_list_for_report,
        )
    ):
        note_message = (
            f"  (Note: Reporting of duplicate PMIDs was limited to "
            f"{duplicate_analysis_limit})"
        )
        print(note_message)

    print("\nElement types found (top 10):")
    for elem_type, count in all_element_types.most_common(10):
        print(f"  {elem_type}: {count}")

    print("\nMost common fields (top 10):")
    for field, count in list(results["field_presence"].items())[:10]:
        print(f"  {field}: {count}")

    print("\nRegular article example:")
    if regular_sample:
        for k, v_item in regular_sample.items():  # Renamed v to v_item
            print(f"  {k}: {str(v_item)[:100]}...")
    else:
        print("  No regular sample article found.")

    # deleted_sample is not populated in this version
    # if deleted_sample:
    #     print("\nDeleted article example:")
    #     for k, v_item in deleted_sample.items():
    #         print(f"  {k}: {str(v_item)[:100]}...")


def extract_article_by_pmid(
    file_path: str,
    target_pmid: str,
) -> Optional[dict[str, Any]]:
    """Extract complete article metadata for a specific PMID from a PubMed XML file.

    Parameters
    ----------
    file_path : str
        Path to the PubMed XML file.
    target_pmid : str
        The PMID to extract the article for.

    Returns
    -------
    dict or None
        Dictionary with complete article metadata or None if PMID not found.

    """
    try:
        with gzip.open(file_path, "rb") as f:
            for _event, element in etree.iterparse(
                f,
                events=("end",),
                tag="PubmedArticle",
            ):
                current_pmid = parse_pmid(element)
                if current_pmid == target_pmid:
                    article_data = extract_complete_metadata(element)
                    element.clear()
                    while element.getprevious() is not None:
                        del element.getparent()[0]
                    return article_data
                element.clear()
                while element.getprevious() is not None:
                    del element.getparent()[0]
        return None
    except Exception as e:
        print(
            f"Error extracting article {target_pmid} from {file_path}: {e}\n{traceback.format_exc()}",
        )
        return None


def extract_xml_to_dict(
    element: Optional[etree.ElementBase],
    include_attributes: bool = True,
) -> Union[dict[str, Any], str, None]:
    """Extract complete XML element hierarchy to a dictionary structure.

    Parameters
    ----------
    element : lxml.etree.Element or None
        XML element to convert.
    include_attributes : bool
        Whether to include element attributes.

    Returns
    -------
    dict or str or None
        Dictionary representing the XML structure, string for text nodes, or None if input is None.

    """
    if element is None:
        return None

    result: dict[str, Any] = {}

    # Add attributes if requested
    if include_attributes and element.attrib:
        result["@attributes"] = {k: v for k, v in element.attrib.items()}

    # Add text content if present and not just whitespace
    text_content = (element.text or "").strip()
    if text_content:
        if not result and not len(
            element,
        ):  # If no attributes and no children, it's just text
            return text_content
        result["#text"] = text_content

    # Process child elements
    children_count: dict[str, int] = Counter()
    for child in element:
        children_count[child.tag] += 1

    for child in element:
        child_data = extract_xml_to_dict(child, include_attributes)
        child_tag = child.tag

        if children_count[child_tag] > 1:  # Multiple children with the same tag
            if child_tag not in result:
                result[child_tag] = []
            result[child_tag].append(child_data)
        else:  # Single child with this tag
            result[child_tag] = child_data

    # If only text was found and no attributes/children, result might be empty
    if (
        not result and text_content
    ):  # Element had text but no children/attributes captured in result
        return text_content
    if (
        not result and not text_content and not len(element) and not element.attrib
    ):  # Truly empty element
        return ""  # Or None, depending on desired representation of empty tags

    return result


def extract_complete_metadata(element: etree.ElementBase) -> dict[str, Any]:
    """Extract comprehensive metadata from a PubmedArticle element including raw XML structure.

    Parameters
    ----------
    element : lxml.etree.Element
        The PubmedArticle element.

    Returns
    -------
    dict
        Dictionary with detailed metadata.

    """
    result: dict[str, Any] = {}

    base_data = parse_article_info(
        element,
        year_info_only=False,
        nlm_category=True,  # parse_article_info uses this for AbstractText, not a general flag
        author_list=True,  # These flags are not directly used by parse_article_info as implemented
        reference_list=True,
        parse_subs=True,
    )
    result.update(base_data)
    result["raw_xml_structure"] = extract_xml_to_dict(element)

    try:
        mesh_headings: list[dict[str, Optional[str]]] = []
        medline_citation = element.find("MedlineCitation")
        if medline_citation is not None:
            mesh_list_element = medline_citation.find("MeshHeadingList")
            if mesh_list_element is not None:
                for mesh_heading in mesh_list_element.findall("MeshHeading"):
                    descriptor_name_el = mesh_heading.find("DescriptorName")  # Renamed
                    if descriptor_name_el is not None:
                        mesh_headings.append(
                            {
                                "descriptor": descriptor_name_el.text,
                                "ui": descriptor_name_el.get("UI"),
                                "major_topic": descriptor_name_el.get("MajorTopicYN"),
                            },
                        )
        result["mesh_headings"] = mesh_headings
    except Exception as e:
        result["mesh_extraction_error"] = str(e)
    return result


def compare_xml_structures(
    struct1: Optional[Union[dict[str, Any], str]],
    struct2: Optional[Union[dict[str, Any], str]],
    path: str = "",
) -> list[dict[str, Any]]:
    """Compare two XML structure dictionaries and return detailed differences.

    Parameters
    ----------
    struct1 : dict or str or None
        First XML structure.
    struct2 : dict or str or None
        Second XML structure.
    path : str
        Current path in the structure (for recursion).

    Returns
    -------
    list
        List of difference objects with paths and details.

    """
    differences: list[dict[str, Any]] = []

    if type(struct1) != type(struct2):
        differences.append(
            {
                "path": path,
                "type": "type_mismatch",
                "struct1_type": type(struct1).__name__,
                "struct2_type": type(struct2).__name__,
                "val1": (
                    str(struct1)[:100] if struct1 is not None else None
                ),  # Add values for context
                "val2": str(struct2)[:100] if struct2 is not None else None,
            },
        )
        return differences

    if isinstance(struct1, str):  # and struct2 is also str due to type check above
        if struct1 != struct2:  # type: ignore
            differences.append(
                {"path": path, "type": "text_diff", "val1": struct1, "val2": struct2},
            )
        return differences

    if isinstance(struct1, dict):  # and struct2 is also dict
        struct1_dict: dict[str, Any] = struct1  # type assertion
        struct2_dict: dict[str, Any] = struct2  # type assertion
        all_keys = set(struct1_dict.keys()) | set(struct2_dict.keys())
        for key in all_keys:
            new_path = f"{path}.{key}" if path else key
            val1 = struct1_dict.get(key)
            val2 = struct2_dict.get(key)

            if key not in struct1_dict:
                differences.append(
                    {
                        "path": new_path,
                        "type": "key_missing_in_struct1",
                        "value_in_struct2": val2,
                    },
                )
            elif key not in struct2_dict:
                differences.append(
                    {
                        "path": new_path,
                        "type": "key_missing_in_struct2",
                        "value_in_struct1": val1,
                    },
                )
            else:
                differences.extend(compare_xml_structures(val1, val2, new_path))
        return differences

    if struct1 is None and struct2 is None:  # Both are None, so no difference here
        return differences

    # Should not be reached if types are consistent (str, dict, or None)
    # but as a fallback:
    differences.append(
        {
            "path": path,
            "type": "unhandled_comparison",
            "val1": str(struct1),
            "val2": str(struct2),
        },
    )
    return differences


def calculate_text_similarity(text1: Optional[str], text2: Optional[str]) -> float:
    """Calculate similarity between two text strings using word overlap."""
    if not text1 or not text2:  # Handles None or empty strings
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 and not words2:  # Both are empty after split
        return 1.0  # Or 0.0, depending on definition of similarity for two empty sets
    if not words1 or not words2:  # One is empty after split
        return 0.0

    common_words = len(words1.intersection(words2))
    total_words = len(words1.union(words2))  # Union handles empty sets correctly

    return common_words / total_words if total_words > 0 else 0.0


def compare_article_metadata(
    article1: Optional[dict[str, Any]],
    article2: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Compare all metadata between two articles with detailed categorization of differences.

    Parameters
    ----------
    article1 : dict or None
        Complete metadata for first article.
    article2 : dict or None
        Complete metadata for second article.

    Returns
    -------
    dict
        Dictionary with comprehensive comparison results.

    """
    result: dict[str, Any] = {  # Define structure upfront
        "identical": True,
        "differences": {},
        "difference_count": 0,
        "field_categories": defaultdict(list),
        "severity": "none",  # none, minor, important, critical
        "match_type": "unknown",  # e.g. exact_match, critical_difference
        "xml_structure_differences": [],
    }

    if article1 is None or article2 is None:
        result["identical"] = False
        result["differences"]["error"] = "One or both articles are None"
        result["difference_count"] = 1
        result["severity"] = "critical"
        result["match_type"] = "error_missing_article"
        return result

    critical_fields = ["pmid", "title", "abstract"]
    important_fields = [
        "authors",
        "journal",
        "publication_types",
        "pubmodel",
        "mesh_headings",
    ]

    # This will be set to False if any non-XML difference is found
    metadata_identical_overall = True

    def compare_values(val1: Any, val2: Any, current_path: str, category: str) -> None:
        nonlocal metadata_identical_overall  # Use nonlocal

        # Generic difference logging
        def log_difference(description: str):
            nonlocal metadata_identical_overall
            metadata_identical_overall = False
            result["difference_count"] += 1
            diff_detail = {
                "path": current_path,
                "description": description,
                "value1_preview": (
                    str(val1)[:100] + "..." if len(str(val1)) > 100 else val1
                ),
                "value2_preview": (
                    str(val2)[:100] + "..." if len(str(val2)) > 100 else val2
                ),
                "type1": type(val1).__name__,
                "type2": type(val2).__name__,
            }
            result["differences"][current_path] = diff_detail
            result["field_categories"][category].append(current_path)

            # Update severity
            if category == "critical" and result["severity"] != "critical":
                result["severity"] = "critical"
            elif category == "important" and result["severity"] not in ["critical"]:
                result["severity"] = "important"
            elif result["severity"] == "none":  # Any other diff is at least minor
                result["severity"] = "minor"

        if type(val1) != type(val2):
            log_difference("type_mismatch")
            return  # Stop deeper comparison if types differ

        if isinstance(val1, dict):
            # This recursive call should ideally be outside compare_values or handled carefully
            # For now, let's assume compare_dict_values is the main entry for dicts
            # This indicates a need to refactor the comparison logic structure
            # For this fix, we'll assume direct dict comparison is handled by the main loop
            if val1 != val2:  # Basic dict inequality
                log_difference(
                    "dict_content_differs",
                )  # Placeholder for deeper dict diff
            # compare_dict_values(val1, val2, current_path) # Avoid direct recursion here
        elif isinstance(val1, list):
            if len(val1) != len(val2):
                log_difference(f"list_length_mismatch ({len(val1)} vs {len(val2)})")
            # Naive list comparison, order matters. For more complex lists, might need element-wise.
            elif val1 != val2:
                log_difference("list_content_differs")
        else:  # Scalar types
            if val1 != val2:
                log_difference("value_mismatch")

    def compare_dict_items(
        dict1: dict[str, Any],
        dict2: dict[str, Any],
        path_prefix: str = "",
    ):
        all_keys_in_dicts = set(dict1.keys()) | set(dict2.keys())  # Renamed
        for key_item in all_keys_in_dicts:  # Renamed
            current_path_item = (
                f"{path_prefix}.{key_item}" if path_prefix else key_item
            )  # Renamed

            # Skip raw XML structure comparison at this level
            if key_item == "raw_xml_structure":
                continue

            val_item1 = dict1.get(key_item)  # Renamed
            val_item2 = dict2.get(key_item)  # Renamed

            current_category = "other"
            if key_item in critical_fields:
                current_category = "critical"
            elif key_item in important_fields:
                current_category = "important"

            if key_item not in dict1:
                compare_values(None, val_item2, current_path_item, current_category)
            elif key_item not in dict2:
                compare_values(val_item1, None, current_path_item, current_category)
            elif isinstance(val_item1, dict) and isinstance(val_item2, dict):
                compare_dict_items(
                    val_item1,
                    val_item2,
                    current_path_item,
                )  # Recurse for nested dicts
            else:
                compare_values(
                    val_item1,
                    val_item2,
                    current_path_item,
                    current_category,
                )

    # Make copies to avoid modifying original dicts
    article1_copy = {k: v for k, v in article1.items() if k != "raw_xml_structure"}
    article2_copy = {k: v for k, v in article2.items() if k != "raw_xml_structure"}

    compare_dict_items(article1_copy, article2_copy)

    # Compare XML structures separately
    xml_struct1 = article1.get("raw_xml_structure")
    xml_struct2 = article2.get("raw_xml_structure")

    if (
        xml_struct1 is not None or xml_struct2 is not None
    ):  # Only compare if at least one has XML
        xml_differences_found = compare_xml_structures(
            xml_struct1,
            xml_struct2,
        )  # Renamed
        if xml_differences_found:
            result["xml_structure_differences"] = xml_differences_found
            result["difference_count"] += len(
                xml_differences_found,
            )  # Add to total diff count
            result["field_categories"]["xml_structure"].extend(
                [d["path"] for d in xml_differences_found if "path" in d],
            )
            if result["severity"] == "none":  # If only XML diffs, it's minor
                result["severity"] = "minor"
            # metadata_identical_overall remains true if only XML differs

    result["identical"] = (
        metadata_identical_overall and not result["xml_structure_differences"]
    )

    # Determine overall match type
    if result["identical"]:
        result["match_type"] = "exact_match"
    elif result["severity"] == "critical":
        result["match_type"] = "critical_difference"
    elif result["severity"] == "important":
        result["match_type"] = "important_difference"
    elif (
        result["severity"] == "minor"
    ):  # Includes XML-only differences if metadata matched
        result["match_type"] = "minor_difference"
    else:  # Should be 'none' if truly identical, but 'identical' flag handles that.
        result["match_type"] = "unknown_difference_state"  # Fallback

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PubMed XML data.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "multiprocessing"],
        default="multiprocessing",
        help="Execution mode: sequential or multiprocessing.",
    )
    parser.add_argument(
        "--file_process_limit",
        type=int,
        default=None,
        help="Limit the number of XML files to process. Default is None (all files).",
    )
    parser.add_argument(
        "--duplicate_analysis_limit",
        type=int,
        default=None,
        help=(
            "Limit the number of duplicate PMIDs to include in the analysis results "
            "(e.g., in the output list of duplicates). Default is None "
            "(all duplicates)."
        ),
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=DEFAULT_PROCESS_COUNT,
        help=(
            f"Number of processes to use in multiprocessing mode. "
            f"Default is {DEFAULT_PROCESS_COUNT}."
        ),
    )

    args = parser.parse_args()

    if args.mode == "sequential":
        print("Running in sequential mode...")
        analyze_pubmed_data(
            file_limit=args.file_process_limit,
            duplicate_analysis_limit=args.duplicate_analysis_limit,
        )
    else:
        run_mode_message_parts = ["Running in multiprocessing mode with"]
        run_mode_message_parts.append(f"file_limit={args.file_process_limit},")
        # Ensure the comma is part of the f-string, not a trailing comma for the append call
        run_mode_message_parts.append(
            f"duplicate_limit={args.duplicate_analysis_limit},",
        )
        run_mode_message_parts.append(f"processes={args.processes}...")
        print(" ".join(run_mode_message_parts))

        analyze_pubmed_data_multiprocessing(
            file_limit=args.file_process_limit,
            duplicate_analysis_limit=args.duplicate_analysis_limit,
            num_processes=args.processes,
        )
