"""
Author: Sonja Aits (partially based in previous script by Rafsan Ahmed)

This script performs NER with BioBERT using GPU or CPU (with or without multiprocessing).
"""

import orjson
import os
import re
import torch
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
import multiprocessing
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pathlib import Path, PurePosixPath
from functools import partial 




def initialize_biobert(ner_config):
    print("Initializing biobert model...")

    device = -1  

    if ner_config["multiprocessing"] == False:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            print("running biobert on GPU")
        else:
            device = -1
            print("running biobert on CPU")
        
    model_dir=ner_config["model_folder"]
    model_name=ner_config["model_name"]
    sentence_batch_size=ner_config["sentence_batch_size"]
    model_path = PurePosixPath(Path(model_dir, model_name))
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
    
    # Convert model to FP16 if GPU is available
    if device >= 0:
        try:
            model.half()  # Convert model weights to FP16
            print("Using FP16 precision.")
        except Exception as e:
            print(f"FP16 not supported: {e}. Falling back to FP32.")

    ner_pipeline = pipeline(
        task='ner',
        model=model, 
        tokenizer=tokenizer,
        aggregation_strategy="max",
        device=device, 
        batch_size=sentence_batch_size) 
    
    return ner_pipeline



def initialize_biobert_worker(ner_config):
    """initizlize biobert once in each worker for multiprocessing.
    """
    global ner_pipeline
    ner_pipeline = initialize_biobert(ner_config)




def run_ner_main(ner_config: dict, input_file_list, CPU_LIMIT):
    """
    Runs NER in batches, handles multiprocessing.
    """

    if ner_config["multiprocessing"] == True:

        file_batch_size = ner_config["file_batch_size"]
        num_batches = -(-len(input_file_list) // file_batch_size)
        print(f"Processing dataset in {num_batches} batches.")
        
        # Create batches of files
        batches = [input_file_list[i * file_batch_size: (i + 1) * file_batch_size] for i in range(num_batches)]
        processes = min(num_batches, CPU_LIMIT)

        # Use Pool to process batches in parallel, initializing Spacy model and matcher once per worker
        with multiprocessing.Pool(processes=processes, initializer=initialize_biobert_worker, initargs=(ner_config,)) as pool:
            pool.starmap(process_batch, [(ner_config, batch) for batch in batches])
    
    else:
        # Fallback to single-process mode (no batching)
        ner_pipeline = initialize_biobert(ner_config)
        for batch_file in tqdm(input_file_list):
            _run_ner_batch(ner_config, batch_file, ner_pipeline)


def process_batch(ner_config, batch_file_list):
    """
    Processes a batch of files
    """
    global ner_pipeline
    for batch_file in tqdm(batch_file_list, desc="Processing Batch"):
        _run_ner_batch(ner_config, batch_file, ner_pipeline)




def _run_ner_batch(ner_config: dict, batch_file, ner_pipeline):
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
    sentence_batch_size=ner_config["sentence_batch_size"]
    sentence_list = []
    sentence_info = []

    for pmid in tqdm(articles, desc=f'batch:{batch_index}'):

        sentences = articles[pmid]["sentences"]

        # Prepare the sentences and store metadata about them
        for i, sentence in enumerate(sentences):
            sentence_list.append(sentence["text"])
            sentence_info.append((pmid, i))  # Store article and sentence index
    

    # Create a DataFrame from the sentence list
    df = pd.DataFrame({"text": sentence_list})

    # Convert the DataFrame into a Dataset
    dataset = Dataset.from_pandas(df)

    def process_batch(batch):
        return {"ner_results": ner_pipeline(batch["text"])}
    
    batched = False

    if ner_config["multiprocessing"] == False and torch.cuda.is_available():
        print("running map function with batched = True")
        batched = True

    predictions = dataset.map(
        process_batch, 
        batched=batched,
        batch_size=sentence_batch_size,
        num_proc=1)
    
    articles_processed = convert_dataset_to_dict(articles, predictions, sentence_info)


    # Write the processed articles back to the output file
    with open(f'{ner_config["output_path"]}/{ner_config["output_file_prefix"]}-{batch_index}.json', 'wb') as f:
        f.write(orjson.dumps(articles_processed, option=orjson.OPT_INDENT_2))

    # Memory cleanup if needed
    #if torch.cuda.is_available():
        #torch.cuda.empty_cache()
        
    return batch_index



def convert_dataset_to_dict(articles, predictions, sentence_info):
    updated_articles = {}
    for pmid, content in articles.items():
        updated_articles[pmid] = content.copy()
        updated_articles[pmid]["sentences"] = [sent.copy() for sent in content["sentences"]]

    for i, prediction_batch in enumerate(predictions["ner_results"]):
        pmid, sent_idx = sentence_info[i]
        updated_articles[pmid]["sentences"][sent_idx]["entities"] = []
        updated_articles[pmid]["sentences"][sent_idx]["entity_spans"] = []
        for prediction in prediction_batch: # Iterate over entities in the sentence
            updated_articles[pmid]["sentences"][sent_idx]["entities"].append(prediction["word"])
            updated_articles[pmid]["sentences"][sent_idx]["entity_spans"].append([prediction["start"], prediction["end"]])

    return updated_articles

