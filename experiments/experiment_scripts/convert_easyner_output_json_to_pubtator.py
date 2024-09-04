#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
A script to convert back into pubtator format the annotated datasets used for HunFLAIR2 (processed BioID, medmentions, and tmvarv3) once the predictions on them are obtained using the EasyNER pipeline. 
The input files are expected to contain annotations.
This script was primarily used in the EasyNER experimentation to convert the JSON formatted outputs from EasyNER into pubtator format in order to run evaluations.
Kindly check the format of the dataset before running the appropriate function

select to run convert_bioid, convert_medmention, convert_tmvar3 or convert biored
    

"""

__author__ = 'Rafsan Ahmed'

import os
import json
import re
from glob import glob

def read_articles(filename):
    "Read articles in json"
    with open(filename, encoding="utf8") as f:
        return json.loads(f.read())
def strip_multi_newline(sentence):
    '''
    Strip multiple newlines occurring after the same sentence
    '''
    return re.sub(r'\n+', ' ', sentence).strip()

## FOR BIOID

def merge_sentences_into_paragraph_bioid(pmid, art):
    # get all sentences of the article and merge them when the output is from the bioID
    
    abstract=[]
    entities=[]
    entity_spans=[]
    span=0
    for sent in art["sentences"]:
        sent_text = sent["text"]
        abstract.append(sent_text)
        
        if len(sent["entities"])!=0:
            for i, ent in enumerate(sent["entities"]):
                entities.append(ent)
                entity_spans.append([int(sent["entity_spans"][i][0])+span, int(sent["entity_spans"][i][1])+span])
        span+=len(sent_text)+1
        
#         print(sent)
        
    abs_ = " ".join(abstract)
    
    assert len(entities)== len(entity_spans)

    return abs_, entities, entity_spans

def convert_bioid(infolder, outfolder_path):
    input_file_list = glob(f'{infolder}**', recursive=True)
    print(input_file_list)

    for idx,infile in enumerate(input_file_list):
        
        # Create pubtator file for eval
        count=0
        basename = os.path.splitext(os.path.basename(infile))[0]
        out_suffix = "_".join(re.split('_|-|\*|\n', basename)[1:-1])
        main_ent = re.split('_|-|\*|\n', basename)[2]
        print(out_suffix, "B", basename, "main", main_ent)
        
        # commented for experimentation. Uncomment
        outfolder = f"{outfolder_path}/easyner_{main_ent}/"
    #     outfolder = f"{os.path.dirname}"

        os.makedirs(outfolder, exist_ok=True)

        outfile = outfolder+f"bioid_{main_ent}_{idx}.txt"

        with open(outfile, "w", encoding="utf8") as f:

            articles = read_articles(infile)

            for pmid, art in articles.items():
                abs_, entities, entity_spans = merge_sentences_into_paragraph_bioid(pmid, art)

                pmid= pmid.split("|")[0]
                f.write(f"{pmid}|t|{abs_}\n")        

                for j, k in zip(entities, entity_spans):
                    f.write(f"{pmid}\t{k[0]}\t{k[1]}\t{j}\t{main_ent}\t-1\t\n")

                f.write("\n")

## For medmentions

def merge_sentences_into_paragraph_medmentions(pmid, art):
    # get all sentences of the article and merge them
    
    abstract=[]
    entities=[]
    entity_spans=[]
    span=0
    for i, sent in enumerate(art["sentences"]):
        sent_text = sent["text"]
        
        if i!=0:
            abstract.append(sent_text)

            if len(sent["entities"])!=0:
                for i, ent in enumerate(sent["entities"]):
                    entities.append(ent)
                    entity_spans.append([int(sent["entity_spans"][i][0])+span, int(sent["entity_spans"][i][1])+span])
            span+=len(sent_text)+1
        
        else:
            # print("INELSE!!!!!!!!!\n")
            if len(sent["entities"])!=0:
                for i, ent in enumerate(sent["entities"]):
                    entities.append(ent)
                    entity_spans.append([int(sent["entity_spans"][i][0])+span, int(sent["entity_spans"][i][1])+span])
            span+=len(sent_text)
        
#         print(sent)
        
    abs_ = " ".join(abstract)
    
    assert len(entities)== len(entity_spans)

    return abs_, entities, entity_spans

def convert_medmentions(infolder,outfolder_path):
    '''
    Convert easyner medmention file to pubtator format
    '''

    input_file_list = glob(f'{infolder}**', recursive=True)
    print(input_file_list)

    for infile in input_file_list:
        # Create pubtator file for eval
        count=0
        basename = os.path.splitext(os.path.basename(infile))[0]
        out_suffix = "_".join(re.split('_|-|\*|\n', basename)[1:-1])
        main_ent = re.split('_|-|\*|\n', basename)[2]
        print(out_suffix, "B", basename, "main", main_ent)
                            
        outfolder = f"{outfolder_path}/easyner_{main_ent}/"
        os.makedirs(outfolder, exist_ok=True)

        outfile = outfolder+f"medmentions.txt"

        with open(outfile, "w", encoding="utf8") as f:

            articles = read_articles(infile)

            for pmid, art in articles.items():
                abs_, entities, entity_spans = merge_sentences_into_paragraph_medmentions(pmid, art)
                print(pmid)
    #             for j, k in zip(entities, entity_spans):
    #                 print(j,k)
                    
                title=art["title"]
    #             print(title)
                pmid= pmid.split("|")[0]
                f.write(f"{pmid}|t|{title}\n")
                f.write(f"{pmid}|a|{abs_}\n")        

                for j, k in zip(entities, entity_spans):
                    f.write(f"{pmid}\t{k[0]}\t{k[1]}\t{j}\t{main_ent}\t-1\t\n")

                f.write("\n")

## FOR TMVAR3

def merge_sentences_into_paragraph_tmvar(pmid, art):
    # get all sentences of the article and merge them
    
    abstract=[]
    entities=[]
    entity_spans=[]
    span=0
    for i, sent in enumerate(art["sentences"]):
        sent_text = sent["text"]
        
        if i!=0:
            abstract.append(sent_text)

            if len(sent["entities"])!=0:
                for i, ent in enumerate(sent["entities"]):
                    entities.append(ent)
                    entity_spans.append([int(sent["entity_spans"][i][0])+span, int(sent["entity_spans"][i][1])+span])
            span+=len(sent_text)+1
        
        else:
            # print("INELSE!!!!!!!!!\n")
            if len(sent["entities"])!=0:
                for i, ent in enumerate(sent["entities"]):
                    entities.append(ent)
                    entity_spans.append([int(sent["entity_spans"][i][0])+span, int(sent["entity_spans"][i][1])+span])
            span+=len(sent_text)+1
        
#         print(sent)
        
    abs_ = " ".join(abstract)
    
    assert len(entities)== len(entity_spans)

    return abs_, entities, entity_spans
    
def convert_tmvar3(infolder,outfolder_path):
    ## convert easyner tmvar3 to pubtator
    input_file_list = glob(f'{infolder}**', recursive=True)
    print(input_file_list)

    for infile in input_file_list:
        # Create pubtator file for eval
        count=0
        basename = os.path.splitext(os.path.basename(infile))[0]
        out_suffix = "_".join(re.split('_|-|\*|\n', basename)[1:-1])
        main_ent = re.split('_|-|\*|\n', basename)[2]
        print(out_suffix, "B", basename, "main", main_ent)
                            
        outfolder = f"{outfolder_path}/easyner_{main_ent}/"
        os.makedirs(outfolder, exist_ok=True)

        outfile = outfolder+f"tmvar_v3.txt"

        with open(outfile, "w", encoding="utf8") as f:

            articles = read_articles(infile)

            for pmid, art in articles.items():
                abs_, entities, entity_spans = merge_sentences_into_paragraph_tmvar(pmid, art)
                print(pmid)
    #             for j, k in zip(entities, entity_spans):
    #                 print(j,k)
                    
                title=art["title"]
    #             print(title)
                pmid= pmid.split("|")[0]
                f.write(f"{pmid}|t|{title}\n")
                f.write(f"{pmid}|a|{abs_}\n")        

                for j, k in zip(entities, entity_spans):
                    f.write(f"{pmid}\t{k[0]}\t{k[1]}\t{j}\t{main_ent}\t-1\t\n")

                f.write("\n")

## FOR BIORED
## Biored has the same format as tmvar3

def convert_biored(infolder, outfolder_path):
    input_file_list = glob(f'{infolder}**', recursive=True)
    print(input_file_list)

    for infile in input_file_list:
        # Create pubtator file for eval
        count=0
        basename = os.path.splitext(os.path.basename(infile))[0]
        out_suffix = "_".join(re.split('_|-|\*|\n', basename)[1:-1])
        main_ent = re.split('_|-|\*|\n', basename)[1]
        print(out_suffix, "B", basename, "main", main_ent)
                            
        outfolder = f"{outfolder_path}/easyner_{main_ent}/"
        os.makedirs(outfolder, exist_ok=True)

        outfile = outfolder+f"biored.txt"

        with open(outfile, "w", encoding="utf8") as f:

            articles = read_articles(infile)

            for pmid, art in articles.items():
                #tmvar applies to the format of biored
                abs_, entities, entity_spans = merge_sentences_into_paragraph_tmvar(pmid, art)
                print(pmid)
    #             for j, k in zip(entities, entity_spans):
    #                 print(j,k)
                    
                title=art["title"]
    #             print(title)
                pmid= pmid.split("|")[0]
                f.write(f"{pmid}|t|{title}\n")
                f.write(f"{pmid}|a|{abs_}\n")        

                for j, k in zip(entities, entity_spans):
                    f.write(f"{pmid}\t{k[0]}\t{k[1]}\t{j}\t{main_ent}\t-1\t\n")

                f.write("\n")

if __name__=="__main__":

    # notice the asterisk in the infolder path to get all biored files. Adjust the script as required
    infolder = "/home/rafsan/aitslab/nlp/EasyNER/results_huner_eval/ner/biored_*/*json"

    outfolder_path = "./annotations/easyner_test/biored/"

    convert_biored(infolder, outfolder_path)