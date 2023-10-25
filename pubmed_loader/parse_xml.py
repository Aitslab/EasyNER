# coding=utf-8

import pubmed_parser as pp
import os
import json
import math
from glob import glob
from tqdm import tqdm, trange

class PubMedLoader:
    
    def __init__(self, input_path,  output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.counter = {}
        os.makedirs(output_path, exist_ok=True)
        
    def get_input_files(self, input_path, file_limit, k="23n"):
        # k is used for keyword to split the filename obtained from pubmed. It's different for each annual baseline
        input_files = sorted(glob(f'{input_path}*.gz'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split(k)[-1][:-4]))
    
        if file_limit==None:
            return input_files
        elif isinstance(file_limit, list) and len(file_limit)==2:
            start=file_limit[0]
            end=file_limit[1]

            filtered_files = []
            for f in input_files:
                f_idx = int(os.path.splitext(os.path.basename(f))[0].split(k)[-1][:-4])
                if f_idx>=start and f_idx<=end:
                    filtered_files.append(f)
    
            return filtered_files
        else:
            raise Exception("ERROR: File limit must be in None or [x,y] format where x is lower limit and y is upper limit (both inclusive)")
        
    
    def get_counter(self):
        return self.counter
    
    def load_xml_and_convert(self, input_file):
        data = pp.parse_medline_xml(input_file, year_info_only=False)
        
        count=0
        d_main = {}
        for art in data:
            if "abstract" in art:
                if isinstance(art["abstract"], str):
                    if len(art["abstract"])>0:
                        count+=1
                        pmid = art["pmid"] if "pmid" in art else count_tot
                        d_main[pmid] = {"title": art["title"],
                                    "abstract":art["abstract"],
                                    "mesh_terms":art["mesh_terms"],
                                    "pubdate":art["pubdate"],
                                    "chemical_list":art["chemical_list"]}
        
        self.counter[input_file] = count
        return d_main

    def write_to_json(self, data, input_file):
        outfile = os.path.join(self.output_path, os.path.basename(input_file.split(".xml")[0])+".json")
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False))
            
    def run_loader(self, file_limit=None):
        input_files_list = self.get_input_files(self.input_path, file_limit=file_limit)
        
        for i, input_file in tqdm(enumerate(input_files_list)):
            
            data = self.load_xml_and_convert(input_file)
            self.write_to_json(data, input_file)


if __name__ == "__main__":

    input_path = "../data/tmp/update_files/"
    output_path = "../res/abstracts/update_files/"
    

    loader = PubMedLoader(input_path,output_path)

    loader.run_loader(file_limit=[1179,1371])
    
    # with open(f"{output_path}counts.txt", "w", encoding="utf-8"):
    #     for k, v in loader.get_counter().items():
    #         f.write(f"{k}\t{v}\n")




