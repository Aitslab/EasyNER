# coding=utf-8

import json
import os
from glob import glob
from tqdm import tqdm
import spacy
import torch
from spacy.matcher import PhraseMatcher
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from scripts import cord_loader
from scripts import downloader
from scripts import splitter
from scripts import splitter_pubmed
from scripts import text_loader
#from scripts import analysis
from scripts import util
from scripts import metrics
from scripts import entity_merger
from scripts import ner_main
from scripts import analysis

CPU_LIMIT=5  #for multiprocessing

def run_cord_loader(cord_loader_config: dict, ignore: bool):
    if ignore:
        print("Ignoring script: cord_loader.")
        return

    print("Running cord_loader script.")
    cord_loader.run(
        input_file=cord_loader_config["input_path"],
        output_file=cord_loader_config["output_path"],
        subset=cord_loader_config["subset"],
        subset_file=cord_loader_config["subset_file"]
    )
    print("Finished running cord_loader script.")


def run_download(dl_config: dict, ignore: bool):
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
    
def run_text_loader(tl_config: dict, ignore:bool):
    
    if ignore:
        print("Ignoring script: free text loader")
        return
    
    print("Running free text loader script")
    
    text_loader.run(tl_config)

    print("Finished running freetext loader script.")

def run_splitter(splitter_config: dict, ignore: bool) -> dict:
    if ignore:
        print("Ignoring script: splitter.")
        return {}
    
    os.makedirs(splitter_config["output_folder"], exist_ok=True)

    if splitter_config["pubmed_pre_batched"]==True:
        if splitter_config["file_limit"]== "ALL":
            input_files_list = splitter_pubmed.load_pre_batched_files(splitter_config["input_path"])
        else:
            input_files_list = splitter_pubmed.load_pre_batched_files(splitter_config["input_path"], limit=splitter_config["file_limit"])
            # split each batch
        if splitter_config["tokenizer"] == 'spacy':
            print("Running splitter script with spacy")
            
            with ProcessPoolExecutor(min(CPU_LIMIT,cpu_count())) as executor:
                
                futures=[executor.submit(splitter_pubmed.split_prebatch,splitter_config, input_file,
                    tokenizer="spacy") for input_file in input_files_list]
                
                for future in as_completed(futures):
                    #print(future.result)
                    i = future.result()
                    
                    

        elif splitter_config["tokenizer"] == 'nltk':
            print("Running splitter script with nltk")

            #import nltk
            #nltk.download("punkt")
            
            with ProcessPoolExecutor(min(CPU_LIMIT,cpu_count())) as executor:
                
                futures=[executor.submit(splitter_pubmed.split_prebatch,splitter_config,input_file,
                    tokenizer="nltk") for input_file in input_files_list]
                
                for future in as_completed(futures):
                    i = future.result()
        


    else:        
        with open(splitter_config["input_path"], "r",encoding="utf-8") as f:
            full_articles = json.loads(f.read())

        article_batches = splitter.make_batches(list(full_articles), splitter_config["batch_size"])

        # split each batch
        if splitter_config["tokenizer"] == 'spacy':
            print("Running splitter script with spacy")
            
            with ProcessPoolExecutor(min(CPU_LIMIT,cpu_count())) as executor:
                
                futures=[executor.submit(splitter.split_batch,splitter_config, idx, art, full_articles,
                    tokenizer="spacy") for idx, art in enumerate(article_batches)]
                
                for future in as_completed(futures):
                    #print(future.result)
                    i = future.result()
                    
                    

        elif splitter_config["tokenizer"] == 'nltk':
            print("Running splitter script with nltk")

            #import nltk
            #nltk.download("punkt")
            
            with ProcessPoolExecutor(min(CPU_LIMIT,cpu_count())) as executor:
                
                futures=[executor.submit(splitter.split_batch,splitter_config,idx, art, full_articles,
                    tokenizer="nltk") for idx, art in enumerate(article_batches)]
                
                for future in as_completed(futures):
                    i = future.result()
                

    print("Finished running splitter script.")


def run_ner(ner_config: dict, ignore: bool):

    if ignore:
        print("Ignoring script: NER.")
        return

    print("Running NER script.")

 
    # For experimentation: limit number of articles to process (and to output)
    # limit = ner_config["article_limit"]
    # if limit > 0:
        # print(f"Limiting NER to {limit} articles.")
        # a = {}
        # i = 0
        # for id in articles:
            # if i >= limit:
                # break
            # a[id] = articles[id]
            # i += 1
        # articles = a

    if ner_config.get("clear_old_results", True):
        try:
            os.remove(ner_config["output_path"])
        except OSError:
            pass
    
    os.makedirs(ner_config["output_path"], exist_ok=True)
    
    input_file_list = sorted(glob(f'{ner_config["input_path"]}*.json'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]))
    
    # Sort files on range
    if "article_limit" in ner_config:
        if isinstance(ner_config["article_limit"], list):
            start=ner_config["article_limit"][0]
            end=ner_config["article_limit"][1]
            
            input_file_list = ner_main.filter_files(input_file_list, start, end)
            
            print("processing articles between {} and {} range".format(start, end))
    


    # Run prediction on each sentence in each article.
    if ner_config["multiprocessing"]:
        with ProcessPoolExecutor(min(CPU_LIMIT,cpu_count())) as executor:
                
            futures=[executor.submit(ner_main.run_ner_main,ner_config,batch_file)
                        for batch_file in input_file_list]
            
            for future in as_completed(futures):
                i = future.result()
    else:
        device=torch.device(0 if torch.cuda.is_available() else "cpu")

        for batch_file in tqdm(input_file_list):
            ner_main.run_ner_main(ner_config,batch_file, device)
            

    print("Finished running NER script.")


def run_analysis(analysis_config: dict, ignore: bool):
    if ignore:
        print("Ignoring script: analysis.")
        return

    print("Running analysis script.")

    analysis.run(analysis_config)

    print("Finished running analysis script.")


def run_metrics(config: dict, ignore: bool):
    if ignore:
        print("Ignoring script: metrics.")
        return

    print("Running metrics script.")

    metrics_config = config["metrics"]
    
    metrics.get_metrics(metrics_config)
    
    print("Finished running metrics script.")

def run_merger(config: dict, ignore: bool):
    if ignore:
        print("Ignoring script: merger.")
        return
    
    print("Running merger script.")

    merger_config = config["merger"]
    
    entity_merger.run_entity_merger(merger_config)
    
    print("Finished running merger script.")

if __name__ == "__main__":
    print("Please see config.json for configuration!")

    with open("config.json", "r") as f:
        config = json.loads(f.read())

    print("Loaded config:")
    # print(json.dumps(config, indent=2, ensure_ascii=False))
    # print()

    os.makedirs("data", exist_ok=True)

    ignore = config["ignore"]


    # Load abstracts from the CORD dataset.
    run_cord_loader(config["cord_loader"], ignore=ignore["cord_loader"])
    print()

    # Download articles from the PubMed API.
    run_download(config["downloader"], ignore=ignore["downloader"])
    print()
    
    # Prepare free text for pipelne.
    run_text_loader(config["text_loader"], ignore=ignore["text_loader"])
    print()

    # Extract sentences from each article.
    run_splitter(config["splitter"], ignore=ignore["splitter"])
    print()

    # Run NER inference on each sentence for each article.
    run_ner(config["ner"], ignore=ignore["ner"])
    print()
    
    # Run analysis on the entities that were found by NER.
    run_analysis(config["analysis"], ignore=ignore["analysis"])
    print()

    # Run metrics on models and gold-standard set
    run_metrics(config, ignore=ignore["metrics"])
    print()

    # Run merger on specified output folders
    run_merger(config, ignore=ignore["merger"])
    print()

    print("Program finished successfully.")
