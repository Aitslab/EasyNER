# coding=utf-8
import os
import json
from glob import glob

def load_freetext(input_folder, prefix):
    '''
    load free text into the pipeline
    '''
    result={}

    input_files_list = sorted(glob(f'{input_folder}*.txt'))

    for id_, file_ in enumerate(input_files_list):

        with open(file_, encoding="utf-8") as f:
            text = " ".join([l.strip() for l in f.readlines()])

        result[prefix+"_"+str(id_)] = {
                        "title": os.path.splitext(os.path.basename(file_))[0],
                        "abstract": text,
                    }
    
    return result

def convert_to_json(result, output_file):
    '''
    convert results to JSON 
    '''
    with open(output_file,  "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
def run(freetext_config):
    prefix = freetext_config["prefix"] if "prefix" in freetext_config else "freetext"
    
    results = load_freetext(freetext_config["input_path"], prefix=prefix)
    convert_to_json(results, output_file=freetext_config["output_path"])

"""
———————————————————————————————————————————————————————————————————————————————
Load free text from file
Arguments:
    input_file - path to .txt file with list of newline-separated PMIDs.
    batch_size - how many articles to download each API call (default: 10).
———————————————————————————————————————————————————————————————————————————————
"""
if __name__ == "__main__":

    freetext_config={"input_folder":"../../NER_pipeline/data/freetext_trial/",
                    "output_file":"../temp/out_freetext.json"}
    
    run(freetext_config)
    
