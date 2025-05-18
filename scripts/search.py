# coding=utf-8
## THis script is used to search for entities in a list from output abstracts

import json
import os
from tqdm import tqdm
from glob import glob
from easyner import util


class EntitySearch:

    def __init__(self, search_config:dict):
        
        self.input_folder = search_config["input_folder"]
        self.output_file = search_config["output_file"]
        self.entities = search_config["entities"]
        
        
    def sort_files(self, input_folder):
        
        return sorted(glob(f'{input_folder}*.json'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]))
        
    def read_files(self, input_file):
        
        with open(input_file, encoding="utf8") as f:
            articles = json.loads(f.read())
        
        return articles
        
    def search(self, input_files_list, entities):
        
        '''
        search entities within a file

        '''
        
        main_dict = {}
        for idx, input_file in tqdm(enumerate(input_files_list)):
        
            articles = self.read_files(input_file)
            
            for art, val in tqdm(articles.items()):
                for sent in val["sentences"]:
                    if len(sent["entities"])==0:
                        continue
                    else:
                        for entity in entities:
                            if entity in sent["entities"]:
                                if art not in main_dict:
                                    main_dict[art]={"sentences":[]}
                                main_dict[art]["sentences"].append({"text":sent["text"], "entities": sent["entities"], "entity_spans": sent["entity_spans"]})
        
        return main_dict
        
    
    def run(self):
    
        input_files_list = self.sort_files(self.input_folder)
        
        main_dict = self.search(input_files_list, self.entities)
        
        util.append_to_json_file(self.output_file, main_dict)
        
        


if __name__ == "__main__":
    
    input_folder = "../../NER_pipeline/results_testeval_p50/text-ner-mtorandtsc1_cell/"
    output_file = "../../NER_pipeline/results_testeval_p50/search/text-search-mtorandtsc1_cell.json"
    entities = ["tsc", "mtor", "cell", "cells"]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    searcher = EntitySearch(input_folder, output_file, entities)
    searcher.run()
    
    





