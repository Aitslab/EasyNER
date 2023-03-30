# coding=utf-8

# Author Rafsan Ahmed
# The script is used to pre-tokenize and convert CRAFT v4.0.0 corpus from pubannotator JSON format to IOB2 format
# The CRAFT concept annotation corpus was used, for example CHEBI, PR, NCBItaxon
# Obtained using scripts from: https://github.com/UCDenver-ccp/CRAFT/wiki/Alternative-annotation-file-formats
# Note that, to use the corpora for biobert, the BioBERT preprocessing step should be run before running predictions.
# The BioBERT preprocess.sh script can be found here: https://github.com/dmis-lab/biobert-pytorch/blob/master/named-entity-recognition/preprocess.sh
# scispacy version 0.5.1


import spacy
import scispacy
import json
import os
import glob
from tqdm.notebook import tqdm

def create_spans_pointer(doc, entities):
    '''
    create a set of spans based on the given text and tokenized entities
    Spans are expanded to match the tokens
    '''
    spans=[]
    limiting_value = 0 #pointer for overlaps
    
    for e in ents:
        start=e["start"]
        end=e["end"]
        label=e["label"]
        span=doc.char_span(start, end, label=label, alignment_mode="expand")
        
        if (span.start_char <limiting_value) or (span.end_char<limiting_value):
            continue
        else:
            limiting_value=span.end_char
            
        spans.append(span)
    
    print(spans)
    return spans
    
if __name__ == "__main__":

    show_err = False
    infolder = "../../../rafsan/data/CRAFT-4.0.0/converted/CHEBI_pubannotation/"
    outfolder = "../../../craft/res/spacy_tok/"
    os.makedirs(outfolder, exist_ok=True)
    outfile = os.path.join(outfolder, "CHEBI_spacy_custom_iob2_.txt")
    out_f = open(outfile, "w", encoding="utf8") #pointer

    files = tqdm(glob.glob(f'{infolder}*.json'))
    nlp = spacy.load("en_core_sci_sm")


    for file in files:
        files.set_description(f"{file}")
        
        # load file
        with open(file, encoding="utf8") as f:
            data = json.load(f)

        # initialize spacy doc and get entities from pubannotation file, remove fragments
        doc = nlp(data["text"])
        ents=[{"start":e["span"]["begin"], "end":e["span"]["end"], "label": e["id"]} for e in data["denotations"] 
              if e["obj"]!='_FRAGMENT']

        # set entities
        spans = create_spans_pointer(doc, ents)
        
        try:
            doc.set_ents(entities=spans)
        except:
            if show_err==True:
                print(file)
                for s in spans:
                    print( "ERR!", s.text, s.start, s.end, s.start_char, s.end_char, s.label_)
            else:
                pass
        # output in iob format
        for t in doc:
            if not t.is_space:
                out_f.write(f"{t}\t{t.ent_iob_}\n")

    out_f.close()