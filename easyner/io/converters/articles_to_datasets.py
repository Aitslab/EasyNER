import pandas as pd
from datasets import Dataset


def convert_articles_to_dataset(
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


def convert_articles_to_dataset_optimized(
    articles, column_names=["pmid", "sent_idx", "text"]
):
    """
    Optimized version of convert_articles_to_dataset that uses list comprehension
    for better performance and avoids unnecessary intermediate storage
    """
    # Optimized: Pre-allocate the total size for better memory efficiency
    total_sentences = sum(
        len(content["sentences"]) for content in articles.values()
    )

    # Optimized: Use list comprehension for faster processing
    articles_processed = [
        [pmid, sent_idx, sent["text"]]
        for pmid, content in articles.items()
        for sent_idx, sent in enumerate(content["sentences"])
    ]

    # Create DataFrame directly from the processed list
    articles_df = pd.DataFrame(articles_processed, columns=column_names)

    # Convert to dataset efficiently
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
