# coding=utf-8
## THis script is used to search for entities in a list from output abstracts

import json
import os
from tqdm.notebook import tqdm
from glob import glob
import utils

class EntitySearch:

    def __init__(self, input_folder, entities):
        
        self.input_folder = input_folder
        self.entities = entities
        
        
    def load_files(self, self.input_folder):
        
        return sorted(glob(f'{input_folder}*.json'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]))
        
    def search(input_files)

        


if __name__ == "__main__":
    
    model_dir = "../../rafsan/models/biobert_pytorch_pretrained/"
    model_name = "HunFlair_chemical_all/"
    seq = "he is feeing very sick nitrous oxide NO nucleus eucaryotic A549 HeLa Cells oxygen."
    
    NER = NER_biobert(model_dir=model_dir, model_name=model_name)
    
    print(NER.predict(seq))
    
    





