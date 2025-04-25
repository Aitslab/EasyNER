import json  # noqa: D100
import os
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from glob import glob
from multiprocessing import cpu_count

import torch
from tqdm import tqdm

from scripts import (
    analysis,
    cord_loader,
    downloader,
    entity_merger,
    metrics,
    nel,
    ner_main,
    pubmed_bulk,
    search,
    splitter,
    splitter_pubmed,
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
    downloader.run(
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


def run_pubmed_bulk_loader(pbl_config: dict, ignore: bool) -> None:  # noqa: D103
    if ignore:
        print("Ignoring script: pubmed bulk downloader")
        return

    print("Running pubmed bulk downloader script.")
    pubmed_bulk.run_pbl(pbl_config)


def run_splitter(splitter_config: dict, ignore: bool) -> dict:  # noqa: D103
    if ignore:
        print("Ignoring script: splitter.")
        return {}

    os.makedirs(splitter_config["output_folder"], exist_ok=True)

    if splitter_config["pubmed_bulk"] == True:
        if splitter_config["file_limit"] == "ALL":
            input_files_list = splitter_pubmed.load_pre_batched_files(
                splitter_config["input_path"],
            )
        else:
            input_files_list = splitter_pubmed.load_pre_batched_files(
                splitter_config["input_path"],
                limit=splitter_config["file_limit"],
            )
            # split each batch
        if splitter_config["tokenizer"] == "spacy":
            print("Running splitter script with spacy")

            with ProcessPoolExecutor(min(CPU_LIMIT, cpu_count())) as executor:

                futures = [
                    executor.submit(
                        splitter_pubmed.split_prebatch,
                        splitter_config,
                        input_file,
                        tokenizer="spacy",
                    )
                    for input_file in input_files_list
                ]

                for future in as_completed(futures):
                    # print(future.result)
                    i = future.result()

        elif splitter_config["tokenizer"] == "nltk":
            print("Running splitter script with nltk")

            # import nltk
            # nltk.download("punkt")

            with ProcessPoolExecutor(min(CPU_LIMIT, cpu_count())) as executor:

                futures = [
                    executor.submit(
                        splitter_pubmed.split_prebatch,
                        splitter_config,
                        input_file,
                        tokenizer="nltk",
                    )
                    for input_file in input_files_list
                ]

                for future in as_completed(futures):
                    i = future.result()

    else:
        with open(splitter_config["input_path"], encoding="utf-8") as f:
            full_articles = json.loads(f.read())

        article_batches = splitter.make_batches(
            list(full_articles),
            splitter_config["batch_size"],
        )

        # split each batch
        if splitter_config["tokenizer"] == "spacy":
            print("Running splitter script with spacy")

            with ProcessPoolExecutor(min(CPU_LIMIT, cpu_count())) as executor:

                futures = [
                    executor.submit(
                        splitter.split_batch,
                        splitter_config,
                        idx,
                        art,
                        full_articles,
                        tokenizer="spacy",
                    )
                    for idx, art in enumerate(article_batches)
                ]

                for future in as_completed(futures):
                    # print(future.result)
                    i = future.result()

        elif splitter_config["tokenizer"] == "nltk":
            print("Running splitter script with nltk")

            # import nltk
            # nltk.download("punkt")

            with ProcessPoolExecutor(min(CPU_LIMIT, cpu_count())) as executor:

                futures = [
                    executor.submit(
                        splitter.split_batch,
                        splitter_config,
                        idx,
                        art,
                        full_articles,
                        tokenizer="nltk",
                    )
                    for idx, art in enumerate(article_batches)
                ]

                for future in as_completed(futures):
                    i = future.result()

    print("Finished running splitter script.")


def run_ner(ner_config: dict, ignore: bool) -> None:  # noqa: C901, D103

    if ignore:
        print("Ignoring script: NER.")
        return

    ner_main.run_ner_pipeline(ner_config, CPU_LIMIT)


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

    # Bulk download pubmed baseline though ftp
    if not ignore["pubmed_bulk_loader"] and TIMEKEEP:
        start_pbloader = time.time()

    run_pubmed_bulk_loader(
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

    run_splitter(config["splitter"], ignore=ignore["splitter"])

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
