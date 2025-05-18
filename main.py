import json  # noqa: D100
import os
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from multiprocessing import cpu_count

from easyner.pipeline.pubmed import (
    bulk_download_pubmed_baseline,
    bulk_unload_pubmed,
    get_abstracts_by_pmids,
)
from scripts import (
    analysis,
    cord_loader,
    entity_merger,
    metrics,
    nel,
    ner_main,
    search,
    text_loader,
)


def run_cord_loader(cord_loader_config: dict, ignore: bool) -> None:  # noqa: D103
    if ignore:
        print("Ignoring script: cord_loader.")
        return

    print("Running cord_loader script.")
    cord_loader.run(
        input_file=cord_loader_config["input_path"],
        output_file=cord_loader_config["output_path"],
        subset=cord_loader_config["subset"],
        subset_file=cord_loader_config["subset_file"],
    )
    print("Finished running cord_loader script.")


def run_download(dl_config: dict, ignore: bool) -> None:  # noqa: D103
    if ignore:
        print("Ignoring script: downloader.")
        return

    print("Running downloader script.")
    get_abstracts_by_pmids.run(
        input_file=dl_config["input_path"],
        output_file=dl_config["output_path"],
        batch_size=dl_config["batch_size"],
    )
    print("Finished running downloader script.")


def run_text_loader(tl_config: dict, ignore: bool) -> None:  # noqa: D103

    if ignore:
        print("Ignoring script: free text loader")
        return

    print("Running free text loader script")

    text_loader.run(tl_config)

    print("Finished running freetext loader script.")


def run_pubmed_bulk_downloader(pbd_config: dict, ignore: bool) -> None:
    """Run the PubMed bulk downloader for baseline files.

    Args:
        pbd_config: Configuration for the PubMed bulk downloader
        ignore: Whether to skip this module

    """
    if ignore:
        print("Ignoring script: pubmed bulk downloader")
        return

    print("Running pubmed bulk downloader script.")
    bulk_download_pubmed_baseline.download_pubmed_in_bulk(pbd_config)
    print("Finished running pubmed bulk downloader.")


def run_pubmed_bulk_updates_downloader(pbu_config: dict, ignore: bool) -> None:
    """Run the PubMed bulk updates downloader for nightly update files.

    Args:
        pbu_config: Configuration for the PubMed bulk updates downloader
        ignore: Whether to skip this module

    """
    if ignore:
        print("Ignoring script: pubmed bulk updates downloader")
        return

    print("Running pubmed bulk updates downloader script.")
    # Use the dedicated pubmed_bulk_updates_downloader configuration
    # The download_updates flag is no longer needed as we're using a separate function
    bulk_download_pubmed_baseline.download_pubmed_updates_in_bulk(pbu_config)
    print("Finished running pubmed bulk updates downloader.")


def run_pubmed_loader(pbl_config: dict, ignore: bool) -> None:
    """Run the PubMed bulk loader to process downloaded XML files.

    Args:
        pbl_config: Configuration for the PubMed bulk loader
        ignore: Whether to skip this module

    """
    if ignore:
        print("Ignoring script: pubmed loader")
        return

    print("Running pubmed loader script.")
    bulk_unload_pubmed.load_pubmed_from_xml(pbl_config)
    print("Finished running pubmed loader script.")


def run_ner(ner_config: dict, ignore: bool) -> None:  # noqa: C901, D103

    if ignore:
        print("Ignoring script: NER.")
        return

    ner_main.run_ner_module(ner_config, CPU_LIMIT)


def run_analysis(analysis_config: dict, ignore: bool) -> None:  # noqa: D103
    if ignore:
        print("Ignoring script: analysis.")
        return

    print("Running analysis script.")

    analysis.run(analysis_config)

    print("Finished running analysis script.")


def run_metrics(config: dict, ignore: bool) -> None:  # noqa: D103
    if ignore:
        print("Ignoring script: metrics.")
        return

    print("Running metrics script.")

    metrics_config = config["metrics"]

    metrics.get_metrics(metrics_config)

    print("Finished running metrics script.")


def run_nel(config: dict, ignore: bool) -> None:  # noqa: D103
    if ignore:
        print("Ignoring script: nel.")
        return

    print("Running nel script.")

    nel_config = config["nel"]

    nel.nel_main(nel_config)

    print("Finished running nel script.")


def run_merger(config: dict, ignore: bool) -> None:  # noqa: D103
    if ignore:
        print("Ignoring script: merger.")
        return

    print("Running merger script.")

    merger_config = config["merger"]

    entity_merger.run_entity_merger(merger_config)

    print("Finished running merger script.")


def run_search(config: dict, ignore: bool) -> None:  # noqa: D103
    if ignore:
        print("Ignoring script: result inspection.")
        return

    print("Running result inspection script.")

    search_config = config["result_inspection"]

    os.makedirs(os.path.dirname(search_config["output_file"]), exist_ok=True)
    searcher = search.EntitySearch(search_config)
    searcher.run()

    print("Finished running result inspection script.")


if __name__ == "__main__":

    print("Please see config.json for configuration!")

    with open("config.json") as f:
        config = json.loads(f.read())

    print("Loaded config:")

    TIMEKEEP = config["TIMEKEEP"]
    if TIMEKEEP:
        start_main = time.time()
        tkff = open("timekeep.txt", "w", encoding="utf8")
        tkff.write(f"start_time at: {start_main}\n")

    os.makedirs("data", exist_ok=True)

    ignore = config["ignore"]
    CPU_LIMIT = config["CPU_LIMIT"]  # for multiprocessing
    print(f"Limited to {CPU_LIMIT} CPUs")

    # Load abstracts from the CORD dataset.
    if not ignore["cord_loader"] and TIMEKEEP:
        start_cordloader = time.time()

    run_cord_loader(config["cord_loader"], ignore=ignore["cord_loader"])

    if not ignore["cord_loader"] and TIMEKEEP:
        end_cordloader = time.time()
        tkff.write(f"Cord Loader time: {end_cordloader-start_cordloader}\n")
    print()

    # Download articles from the PubMed API.
    if not ignore["downloader"] and TIMEKEEP:
        start_downloader = time.time()

    run_download(config["downloader"], ignore=ignore["downloader"])

    if not ignore["downloader"] and TIMEKEEP:
        end_downloader = time.time()
        tkff.write(f"Downloader time: {end_downloader-start_downloader}\n")
    print()

    # Prepare free text for pipelne.
    if not ignore["text_loader"] and TIMEKEEP:
        start_textloader = time.time()

    run_text_loader(config["text_loader"], ignore=ignore["text_loader"])

    if not ignore["text_loader"] and TIMEKEEP:
        end_textloader = time.time()
        tkff.write(f"Text loader time: {end_textloader-start_textloader}\n")
    print()

    # Download PubMed baseline files
    if not ignore.get("pubmed_bulk_downloader", True) and TIMEKEEP:
        start_pbdownloader = time.time()

    run_pubmed_bulk_downloader(
        config["pubmed_bulk_downloader"],
        ignore=ignore.get("pubmed_bulk_downloader", True),
    )

    if not ignore.get("pubmed_bulk_downloader", True) and TIMEKEEP:
        end_pbdownloader = time.time()
        tkff.write(
            f"Pubmed bulk downloader time: {end_pbdownloader-start_pbdownloader}\n",
        )
    print()

    # Download PubMed update files
    start_pbudownloader = 0  # Initialize time variables

    if not ignore.get("pubmed_bulk_updates_downloader", True):
        if TIMEKEEP:
            start_pbudownloader = time.time()

        # Check if the config section exists
        if "pubmed_bulk_updates_downloader" not in config:
            print(
                "Warning: pubmed_bulk_updates_downloader section not found in config.json",
            )
            print("Skipping PubMed bulk updates download.")
        else:
            run_pubmed_bulk_updates_downloader(
                config["pubmed_bulk_updates_downloader"],
                ignore=False,
            )

        if TIMEKEEP:
            end_pbudownloader = time.time()
            tkff.write(
                f"Pubmed bulk updates downloader time: "
                f"{end_pbudownloader-start_pbudownloader}\n",
            )
    else:
        print("Ignoring pubmed bulk updates downloader as specified in config.")
    print()

    # Process downloaded PubMed XML files
    if not ignore["pubmed_bulk_loader"] and TIMEKEEP:
        start_pbloader = time.time()

    run_pubmed_loader(
        config["pubmed_bulk_loader"],
        ignore=ignore["pubmed_bulk_loader"],
    )

    if not ignore["pubmed_bulk_loader"] and TIMEKEEP:
        end_pbloader = time.time()
        tkff.write(f"Pubmed bulk loader time: {end_pbloader-start_pbloader}\n")
    print()

    # Extract sentences from each article.
    if TIMEKEEP:
        start_splitter = time.time()

    if not ignore["splitter"]:
        from easyner.pipeline.splitter import run_splitter

    run_splitter(
        config["splitter"],
        ignore=ignore["splitter"],
        cpu_limit=CPU_LIMIT,
    )
    # run_splitter_pubmed(config["splitter_pubmed"], ignore=ignore["splitter_pubmed"])
    if TIMEKEEP:
        end_splitter = time.time()
        tkff.write(f"Splitter time: {end_splitter-start_splitter}\n")
    print()

    # Run NER inference on each sentence for each article.
    if TIMEKEEP:
        start_ner = time.time()

    run_ner(config["ner"], ignore=ignore["ner"])

    if TIMEKEEP:
        end_ner = time.time()
        tkff.write(f"NER time: {end_ner-start_ner}\n")
        tkff.write(f"Total time till NER: {end_ner-start_main}\n")
    print()

    # Run analysis on the entities that were found by NER.
    if not ignore["analysis"] and TIMEKEEP:
        start_analysis = time.time()

    run_analysis(config["analysis"], ignore=ignore["analysis"])

    if not ignore["analysis"] and TIMEKEEP:
        end_analysis = time.time()
        tkff.write(f"Analysis time: {end_analysis-start_analysis}\n")
    print()

    # Run metrics on models and gold-standard set
    run_metrics(config, ignore=ignore["metrics"])
    print()

    # Run nel on models and gold-standard set
    run_nel(config, ignore=ignore["nel"])
    print()

    # Run merger on specified output folders
    run_merger(config, ignore=ignore["merger"])
    print()

    # Run result inspection on specified NER folder
    run_search(config, ignore=ignore["result_inspection"])
    print()

    print("Program finished successfully.")

    if TIMEKEEP:
        end_main = time.time()
        tkff.write(f"end_time at: {end_main}\n")

        tkff.write(f"Total runtime: {end_main-start_main}\n")
        tkff.close()
