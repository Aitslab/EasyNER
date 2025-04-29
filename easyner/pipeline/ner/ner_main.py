# coding=utf-8

from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
import os
import json
import torch
from tqdm import tqdm

from easyner.io.utils import extract_batch_index

from easyner import util


OUTPUT_FILE_TEMPLATE = "{output_path}/{output_file_prefix}-{batch_index}.json"


def _get_input_files_sorted(ner_config: dict) -> list:
    """
    Find and sort input files based on configuration.

    Parameters:
    -----------
    ner_config: dict
        Configuration settings including input path and filters

    Returns:
    --------
    list: Sorted list of input files to process
    """
    input_file_list = sorted(
        glob(f'{ner_config["input_path"]}*.json'),
        key=lambda x: int(
            os.path.splitext(os.path.basename(x))[0].split("-")[-1]
        ),
    )

    # Apply file range filtering if configured
    if "article_limit" in ner_config and isinstance(
        ner_config["article_limit"], list
    ):
        from easyner.io.utils import filter_files

        start = ner_config["article_limit"][0]
        end = ner_config["article_limit"][1]

        input_file_list = filter_files(input_file_list, start, end)
        print(f"Processing articles in range {start} to {end}")

    return input_file_list


def _build_output_filepath(ner_config: dict, batch_index: int) -> str:
    """
    Generate the output file path based on the configuration and batch index.

    Parameters:
    -----------
    ner_config: dict
        Configuration with output paths and prefixes
    batch_index: int
        The batch index to include in the filename

    Returns:
    --------
    str: The formatted output file path
    """
    return OUTPUT_FILE_TEMPLATE.format(
        output_path=ner_config["output_path"],
        output_file_prefix=ner_config["output_file_prefix"],
        batch_index=batch_index,
    )


def process_files_in_parallel(
    input_file_list: list, ner_config: dict, cpu_limit: int = 1
):
    """
    Process multiple batch files in parallel using a process pool.

    Parameters:
    -----------
    input_file_list: list
        List of files to process
    ner_config: dict
        Configuration for NER processing
    cpu_limit: int
        Maximum number of CPUs to use for multiprocessing
    """
    print(
        f"Processing files in parallel with {ner_config['model_type']} using {cpu_limit} CPUs"
    )
    from multiprocessing import cpu_count

    with ProcessPoolExecutor(min(cpu_limit, cpu_count())) as executor:
        futures = [
            executor.submit(process_batch_file, ner_config, batch_file)
            for batch_file in input_file_list
        ]

        # Process results as they complete
        for i, future in enumerate(as_completed(futures)):
            batch_index = future.result()
            print(f"Completed batch {batch_index} ({i+1}/{len(futures)})")


def process_batch_file(ner_config: dict, batch_file: str, device=None) -> int:
    """
    Process a single batch file with the appropriate NER technique.

    Parameters:
    -----------
    ner_config: dict
        Configuration for NER processing
    batch_file: str
        Path to the batch file to process
    device: torch.device or int or str, optional
        Device to use for processing

    Returns:
    --------
    int: The batch index of the processed file
    """
    # Load the batch file
    with open(batch_file, "r", encoding="utf-8") as f:
        articles = json.loads(f.read())

    # Extract batch index from filename
    batch_index = extract_batch_index(batch_file)

    # Prepare output file path
    output_file = _build_output_filepath(ner_config, batch_index)

    # Handle empty articles case
    if len(articles) == 0:
        util.append_to_json_file(output_file, articles)
        return batch_index

    # Process with appropriate NER model
    if ner_config["model_type"] == "spacy_phrasematcher":
        from .dictionary_based.ner_spacy import (
            run_ner_with_spacy_phrasematcher,
        )

        articles = run_ner_with_spacy_phrasematcher(
            articles, ner_config, batch_index
        )
    elif ner_config["model_type"] == "biobert_finetuned":
        from .transformer_based.ner_biobert import (
            run_ner_with_biobert_finetuned,
        )

        articles = run_ner_with_biobert_finetuned(
            articles, ner_config, batch_index, device
        )
    else:
        raise ValueError(
            f"Unknown model type: {ner_config['model_type']}. "
            "Supported types are 'spacy_phrasematcher' and 'biobert_finetuned'."
        )

    # Save results to output file
    util.append_to_json_file(output_file, articles)
    return batch_index


def run_ner_module(ner_config: dict, cpu_limit: int):
    """
    Main entry point for the NER pipeline that handles:
    - Output directory setup
    - File discovery and filtering
    - Managing parallel or sequential processing

    Parameters:
    -----------
    ner_config: dict
        Configuration for NER processing
    cpu_limit: int
        Maximum number of CPUs to use for multiprocessing
    """
    print("Starting NER pipeline...")

    if ner_config.get("clear_old_results", True):
        from easyner.io.utils import _remove_all_files_from_dir

        _remove_all_files_from_dir(
            ner_config["output_path"]
        )  # Todo change config to output_dir for claritys
    else:
        os.makedirs(ner_config["output_path"], exist_ok=True)

    input_file_list = _get_input_files_sorted(ner_config)

    # Process files (in parallel or sequentially)
    if ner_config["multiprocessing"]:
        process_files_in_parallel(input_file_list, ner_config, cpu_limit)
    else:
        device = torch.device(0 if torch.cuda.is_available() else "cpu")
        print(
            f"Processing files sequentially with {ner_config['model_type']} on device: {device}"
        )

        for batch_file in tqdm(input_file_list, desc="Processing batches"):
            process_batch_file(ner_config, batch_file, device)

    print("----NER pipeline processing complete----")


if __name__ == "__main__":
    pass
