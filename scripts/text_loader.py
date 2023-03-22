# coding=utf-8

import json

def load_freetext(input_file, title="", id_="freetext"):
    '''
    load free text into the pipeline
    '''
    result={}
    with open(input_file, encoding="utf-8") as f:
        text=f.read()
        
    result[id_] = {
                    "title": title,
                    "abstract": text,
                }
    
    return result

def convert_to_json(result, output_file):
    '''
    convert results to JSON 
    '''
    with open(output_file,  "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    
"""
———————————————————————————————————————————————————————————————————————————————
Load free text from file
Arguments:
    input_file - path to .txt file with list of newline-separated PMIDs.
    batch_size - how many articles to download each API call (default: 10).
———————————————————————————————————————————————————————————————————————————————
"""
if __name__ == "__main__":

    input_file = "../"
    output_file= "../"
    
    results=load_freetext(input_file)
    
