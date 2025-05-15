"""
Author: Sonja Aits (partially based in previous script by Rafsan Ahmed)

This script performs NER with a dictionary loaded into spacy Phrasematcher using GPU or CPU (with or without multiprocessing).
"""

import spacy
import orjson
import os
import re
from tqdm import tqdm
from spacy.matcher import PhraseMatcher
import multiprocessing

# Global variables to be used in each worker process
nlp = None
matcher = None

def initialize_spacy(ner_config):
    '''
    Initializes Spacy model and PhraseMatcher
    This should only be called once per worker process
    '''

    # Load Spacy model if not loaded
    print("Initializing Spacy model...")

    if ner_config["multiprocessing"] == False:
        try:
            # Check if GPU is available
            spacy.require_gpu()  # This will raise an error if GPU is not available
            print("SpaCy is using GPU.")
        except Exception as e:
            print(f"No GPU found. spaCy is running on single CPU. Consider switching to multiprocessing:True in config file to run on multiple CPUs. Error: {e}")


    nlp = spacy.load(ner_config["model_name"], disable=["ner", "parser", "tagger"])  

    # Initialize PhraseMatcher if not initialized
    print("Initializing PhraseMatcher...")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # Add the dictionary terms to the PhraseMatcher removing duplicates
    terms = set()
    with open(ner_config["vocab_path"], 'r') as f:
        for line in f:
            x = line.strip()
            terms.add(x)  

    if not terms:
        raise ValueError("Vocabulary file is empty!")
    
    patterns = [nlp.make_doc(term) for term in terms]
    matcher.add(ner_config["entity_type"], patterns)

    return nlp, matcher


def initialize_spacy_worker(ner_config):
    """
    Initialize Spacy model and PhraseMatcher once in each worker for multiprocessing.
    """
    global nlp, matcher
    nlp, matcher = initialize_spacy(ner_config)


def run_ner_main(ner_config: dict, input_file_list, CPU_LIMIT):
    """
    Runs NER in batches, handles multiprocessing.
    """

    if ner_config["multiprocessing"] == True:

        file_batch_size = ner_config["file_batch_size"]
        num_batches = -(-len(input_file_list) // file_batch_size)
        print(f"Number of files in dataset: {len(input_file_list)}")
        print(f"Processing dataset in {num_batches} batches.")
        
        # Create batches of files
        batches = [input_file_list[i * file_batch_size: (i + 1) * file_batch_size] for i in range(num_batches)]
        processes = min(num_batches, CPU_LIMIT)

        # Use Pool to process batches in parallel, initializing Spacy model and matcher once per worker
        with multiprocessing.Pool(processes=processes, initializer=initialize_spacy_worker, initargs=(ner_config,)) as pool:
            pool.starmap(process_batch, [(ner_config, batch) for batch in batches])
    
    else:
        # Fallback to single-process mode (no batching)
        nlp, matcher = initialize_spacy(ner_config)
        for batch_file in tqdm(input_file_list):
            _run_ner_batch(ner_config, batch_file, nlp, matcher)


def process_batch(ner_config, batch_file_list):
    """
    Processes a batch of files
    """
    global nlp, matcher
    for batch_file in tqdm(batch_file_list, desc="Processing Batch"):
        _run_ner_batch(ner_config, batch_file, nlp, matcher)


def _run_ner_batch(ner_config: dict, batch_file, nlp, matcher):
    '''
    Run NER in batches for each batch file
    '''
    # Read articles from the batch file
    with open(batch_file, "r", encoding="utf-8") as f:
        articles = orjson.loads(f.read())

    # Get batch ID from the filename
    regex = re.compile(r'\d+')
    try:
        batch_index = int(regex.findall(os.path.basename(batch_file))[-1])
    except IndexError:  # More specific exception
        raise ValueError(f"Filename format error: '{batch_file}'. Filenames not numbered!")

    if len(articles) == 0:
        # If no articles, just save and return
        with open(f'{ner_config["output_path"]}/{ner_config["output_file_prefix"]}-{batch_index}.json', 'wb') as f:
            f.write(orjson.dumps(articles, option=orjson.OPT_INDENT_2))
        return batch_index

    # Create a list of sentences for processing
    sentence_batch_size = ner_config.get("sentence_batch_size", 500)
    sentence_list = []
    sentence_info = []

    for pmid in tqdm(articles, desc=f'batch:{batch_index}'):

        sentences = articles[pmid]["sentences"]

        # Prepare the sentences and store metadata about them
        for i, sentence in enumerate(sentences):
            sentence_list.append(sentence["text"])
            sentence_info.append((pmid, i))  # Store article and sentence index

    # Use nlp.pipe with batching
    processed_docs = []
    for doc in nlp.pipe(sentence_list, batch_size=sentence_batch_size, n_process=1):
        processed_docs.append(doc)

    # Process the documents and update the articles with NER results
    for doc, (pmid, sentence_idx) in zip(processed_docs, sentence_info):
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

        # Update the article with NER results
        articles[pmid]["sentences"][sentence_idx]["entities"] = entities
        articles[pmid]["sentences"][sentence_idx]["entity_spans"] = spans

        # Optionally store tokens if requested
        if ner_config["store_tokens"] == "yes":
            tokens = [token.text for token in doc]
            articles[pmid]["sentences"][sentence_idx]["tokens"] = tokens

    # Write the processed articles back to the output file
    with open(f'{ner_config["output_path"]}/{ner_config["output_file_prefix"]}-{batch_index}.json', 'wb') as f:
        f.write(orjson.dumps(articles, option=orjson.OPT_INDENT_2))

    return batch_index



