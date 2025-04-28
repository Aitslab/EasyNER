# coding=utf-8

from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
import spacy
import os
import re
import json
import pandas as pd
import torch
from tqdm import tqdm
from spacy.matcher import PhraseMatcher
from datasets import Dataset, load_dataset
from . import ner_biobert, util
from .ner_inference import NERInferenceSession_biobert_onnx


def run_ner_pipeline(ner_config: dict, cpu_limit: int):
    """
    Main entry point for the NER pipeline that handles:
    - Output directory setup
    - Input file gathering and filtering
    - Managing parallel or sequential processing

    Parameters:
    -----------
    ner_config: dict
        Configuration for NER processing
    cpu_limit: int
        Maximum number of CPUs to use for multiprocessing
    """
    print("Running NER script.")

    output_path = ner_config["output_path"]
    if ner_config.get("clear_old_results", True):
        try:
            os.remove(output_path)
        except OSError:
            pass

    os.makedirs(output_path, exist_ok=True)

    input_file_list = sorted(
        glob(f'{ner_config["input_path"]}*.json'),
        key=lambda x: int(
            os.path.splitext(os.path.basename(x))[0].split("-")[-1]
        ),
    )

    # Sort files on range
    if "article_limit" in ner_config:
        if isinstance(ner_config["article_limit"], list):
            start = ner_config["article_limit"][0]
            end = ner_config["article_limit"][1]

            input_file_list = filter_files(input_file_list, start, end)

            print(
                "processing articles between {} and {} range".format(
                    start, end
                )
            )

    # Run prediction on each sentence in each article.
    if ner_config["multiprocessing"]:
        from multiprocessing import cpu_count

        with ProcessPoolExecutor(min(cpu_limit, cpu_count())) as executor:

            futures = [
                executor.submit(run_ner_main, ner_config, batch_file)
                for batch_file in input_file_list
            ]

            for future in as_completed(futures):
                i = future.result()
    else:
        device = torch.device(0 if torch.cuda.is_available() else "cpu")

        for batch_file in tqdm(input_file_list):
            run_ner_main(ner_config, batch_file, device)

    print("Finished running NER script.")


def run_ner_main(ner_config: dict, batch_file, device=-1):
    """
    run NER in batches from sentence splitter output
    """

    with open(batch_file, "r", encoding="utf-8") as f:
        articles = json.loads(f.read())

    # get batch IDs
    regex = re.compile(r"\d+")
    try:
        batch_index = int(regex.findall(os.path.basename(batch_file))[-1])
    except:
        print(batch_file)
        raise Exception("Filenames not numbered!")

    if len(articles) == 0:
        util.append_to_json_file(
            f'{ner_config["output_path"]}/{ner_config["output_file_prefix"]}-{batch_index}.json',
            articles,
        )
        return batch_index

    # Prepare spacy, if it is needed
    if ner_config["model_type"] == "spacy_phrasematcher":
        if not ner_config["multiprocessing"]:
            spacy.prefer_gpu()

        print("Running NER with spacy")
        nlp = spacy.load(ner_config["model_name"])
        terms = []
        with open(ner_config["vocab_path"], "r") as f:
            for line in f:
                x = line.strip()
                terms.append(x)
        print("Phraselist complete")

        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(term) for term in terms]
        matcher.add(ner_config["entity_type"], patterns)

        # Run prediction on each sentence in each article.
        for pmid in tqdm(articles, desc=f"batch:{batch_index}"):

            sentences = articles[pmid]["sentences"]

            # Predict with spacy PhraseMatcher, if it has been selected

            for i, sentence in enumerate(sentences):
                ner_class = ner_config["entity_type"]

                doc = nlp(sentence["text"])
                if ner_config["store_tokens"] == "yes":
                    tokens = []
                    # tokens_idxs = []  #uncomment if you want a list of token character offsets within the sentence
                    for token in doc:
                        tokens.append(
                            token.text
                        )  # to get a list of tokens in the sentence
                    # tokens_idxs.append(token.idx) #uncomment if you want a list of token character offsets within the sentence
                    articles[pmid]["sentences"][i]["tokens"] = tokens

                entities = []
                spans = []
                matches = matcher(doc)

                for match_id, start, end in matches:
                    span = doc[start:end]
                    ent = span.text
                    entities.append(ent)
                    first_char = span.start_char
                    last_char = span.end_char - 1
                    spans.append((first_char, last_char))

                # articles[pmid]["sentences"][i]["NER class"] = ner_class
                articles[pmid]["sentences"][i]["entities"] = entities
                articles[pmid]["sentences"][i]["entity_spans"] = spans

    elif ner_config["model_type"] == "biobert_finetuned":

        # print("Running NER with finetuned BioBERT")

        ner_session = ner_biobert.NER_biobert(
            model_dir=ner_config["model_folder"],
            model_name=ner_config["model_name"],
            device=device,
        )

        def wrapper_predict(batch):
            """
            A wrapper to run map function with predict in batches
            """
            try:
                predictions = ner_session.nlp(
                    batch["text"], batch_size=ner_config.get("batch_size", 8)
                )
            except Exception as e:
                # Create empty lists for failed predictions
                predictions = [[] for _ in range(len(batch["text"]))]

            batch["prediction"] = predictions
            return batch

        articles_dataset = biobert_process_articles(articles)

        articles_dataset_processed = articles_dataset.map(
            wrapper_predict,
            batched=True,
            batch_size=8,  # Adjust based on your memory and CPU capacity
            desc="Batch " + str(batch_index),
        )

        articles_processed = convert_dataset_to_dict(
            articles, articles_dataset_processed
        )
        articles = articles_processed

        # for i, sentence in enumerate(sentences):
        #     try:
        #         # the entities predicted are all uncased but the entity within the sentence is cased
        #         entities = ner_session.predict(sentence["text"])
        #     except:
        #         # exception due to existence of utf tags in the data, which is incomprehensable/non-tokenizable by the model
        #         print("batch {}, sentence no. {} with text [{}] was not predicted".format(batch_index, i, sentence))
        #         entities = []

        #     entities_list = []
        #     entity_spans_list = []
        #     if len(entities)>0:
        #         for ent in entities:
        #             entities_list.append(ent["word"])
        #             entity_spans_list.append([ent["start"],ent["end"]])

        #     articles[pmid]["sentences"][i]["entities"] = entities_list
        #     articles[pmid]["sentences"][i]["entity_spans"] = entity_spans_list

    util.append_to_json_file(
        f'{ner_config["output_path"]}/{ner_config["output_file_prefix"]}-{batch_index}.json',
        articles,
    )
    return batch_index


def filter_files(list_files, start, end):
    """
    filter files based on start and end
    """
    filtered_list_files = []
    for f in list_files:
        f_idx = int(os.path.splitext(os.path.basename(f))[0].split("-")[-1])
        if f_idx >= start and f_idx <= end:
            filtered_list_files.append(f)

    return filtered_list_files


def biobert_process_articles(
    articles, column_names=["pmid", "sent_idx", "text"]
):
    """
    process articles into a huggingface dataset
    articles: sentence split articles from splitter
    column names: column names for the dataframe/dataset

    returns processed hf dataset where each line is a sentence
    """

    articles_processed = []

    for pmid, content in articles.items():
        l = []
        sent_idx = 0
        for sent in content["sentences"]:
            articles_processed.append([pmid, sent_idx, sent["text"]])
            sent_idx += 1

    articles_df = pd.DataFrame(articles_processed)
    articles_df.columns = column_names

    articles_ds = Dataset.from_pandas(articles_df)

    return articles_ds


def convert_dataset_to_dict(articles, ner_dataset):
    """
    adds predictions and spans to expected dictionary/json format articles
    articles: original articles
    ner_dataset: hf dataset with predictions

    returns: articles dictionary with added entities and spans
    """

    for row in ner_dataset:
        pmid = row["pmid"]
        sent_idx = row["sent_idx"]
        text = row["text"]
        prediction = row["prediction"]
        articles[pmid]["sentences"][sent_idx]["entities"] = []
        articles[pmid]["sentences"][sent_idx]["entity_spans"] = []

        if len(prediction) != 0:
            for pred in prediction:
                articles[pmid]["sentences"][sent_idx]["entities"].append(
                    pred["word"]
                )
                articles[pmid]["sentences"][sent_idx]["entity_spans"].append(
                    [pred["start"], pred["end"]]
                )

    return articles


if __name__ == "__main__":
    pass
