# coding=utf-8

import json
import os
import re
from tqdm import tqdm, trange
from glob import glob

def read_articles(filename:str):
    with open(filename, encoding="utf-8") as f:
        return json.loads(f.read())

def get_sorted_files(filepath):
    '''
    get a list of sorted file paths using glob
    '''
    return sorted(glob(f'{filepath}/*.json'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]))


def process_articles(articles: dict, entity_tag:str):
    '''
    process the article to contain tag
    '''
    for art in list(articles):

        for i, sent in enumerate(articles[art]["sentences"]):
            if len(sent["entities"])>0:
                articles[art]["sentences"][i]["entities"] = {entity_tag:articles[art]["sentences"][i]["entities"]}
                articles[art]["sentences"][i]["entity_spans"] = {entity_tag:articles[art]["sentences"][i]["entity_spans"]}
                articles[art]["sentences"][i]["ids"] = {entity_tag:articles[art]["sentences"][i]["ids"]}
                articles[art]["sentences"][i]["names"] = {entity_tag:articles[art]["sentences"][i]["names"]}
            else:
                articles[art]["sentences"][i]["entities"] = {}
                articles[art]["sentences"][i]["entity_spans"] = {} 
                articles[art]["sentences"][i]["ids"] = {} 
                articles[art]["sentences"][i]["names"] = {} 

    return articles

def merge_two_articles(articles_1, articles_2):
    '''
    merge articles_2 with article_1 at entity level
    '''
    if len(articles_1)==0:
        articles_1={k:v for k,v in articles_2.items()}
        return articles_1
    
    elif len(articles_2)==0 and len(articles_1)>0: # user exception
        articles_2={k:v for k,v in articles_1.items()}
        return articles_2
    
    else:
        for art1,art2 in zip(list(articles_1), list(articles_2)):
            if art1!=art2:
                raise Exception("ERR!!!!")
            for i, sent in enumerate(articles_1[art1]["sentences"]):
                if len(articles_2[art2]["sentences"][i]["entities"])>0:
                    sent["entities"].update(articles_2[art2]["sentences"][i]["entities"])
                    sent["entity_spans"].update(articles_2[art2]["sentences"][i]["entity_spans"])
                    if "ids" in articles_2[art2]["sentences"][i]:
                        sent["ids"].update(articles_2[art2]["sentences"][i]["ids"])
                    if "names" in articles_2[art2]["sentences"][i]:
                        sent["names"].update(articles_2[art2]["sentences"][i]["names"])
                    
    return articles_1

def entity_merger(paths:list, entities:list, output_file:str):
    '''
    merge same files
    '''
    
    merged_entities = {}
    
    for file_, tag in zip(paths, entities):
        # Read articles
        articles = read_articles(file_)
        
        #process ner parts into dictionaries
        processed_ner_article = process_articles(articles, tag)
        
        #merge entities
        merged_entities=merge_two_articles(merged_entities, processed_ner_article)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(merged_entities, indent=2, ensure_ascii=False))
        
    return

def run_entity_merger(merger_config: dict):
    '''
    merge all files within
    '''
    paths = merger_config["paths"]
    entities = merger_config["entities"]
    output_folder = merger_config["output_path"]
    output_prefix = merger_config["output_prefix"]
    os.makedirs(output_folder, exist_ok=True)

    file_lists = {entity:get_sorted_files(path) for path, entity in zip(paths, entities)}

    if len(set([len(v) for k,v in file_lists.items()]))!=1:
        raise Exception("ERROR! Mismatched number of files in given folders")
    
    for i in trange(len(file_lists[entities[0]])):
        processed_paths = [file_lists[j][i] for j in entities]
        output_file = output_folder+output_prefix + str(get_batch_no_from_filename(processed_paths[0])) +".json"

        entity_merger(paths=processed_paths, entities=entities,output_file=output_file)
        
    return 
    
def get_batch_no_from_filename(filename):
    return re.findall(r'\d+',filename)[-1]

def check_match_batch_index(filename1, filename2):
    return get_batch_no_from_filename(filename1) == get_batch_no_from_filename(filename2)
    
    
