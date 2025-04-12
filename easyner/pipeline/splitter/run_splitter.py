from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import cpu_count
import os

from scripts import splitter
from scripts import splitter_pubmed

from easyner.config import load_config

config = load_config()
CPU_LIMIT = config["CPU_LIMIT"] if "CPU_LIMIT" in config else 4


def run_splitter(splitter_config: dict, ignore: bool) -> dict:
    if ignore:
        print("Ignoring script: splitter.")
        return {}

    os.makedirs(splitter_config["output_folder"], exist_ok=True)

    if splitter_config["pubmed_bulk"]:
        if splitter_config["file_limit"] == "ALL":
            input_files_list = splitter_pubmed.load_pre_batched_files(splitter_config["input_path"])
        else:
            input_files_list = splitter_pubmed.load_pre_batched_files(
                splitter_config["input_path"], limit=splitter_config["file_limit"]
            )
        print(f"Found {len(input_files_list)} files to process.")
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
        with open(splitter_config["input_path"], "r", encoding="utf-8") as f:
            full_articles = json.loads(f.read())

        article_batches = splitter.make_batches(list(full_articles), splitter_config["batch_size"])

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
