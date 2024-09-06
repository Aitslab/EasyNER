#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
A script to convert the pubtator files (bioid, medmentions and tmvarv3 as well as BioRED) into json format, to be processed by EasyNER

"""

__author__ = 'Rafsan Ahmed'

import os
import json
import re
from glob import glob
from tqdm import tqdm

def convert_to_json(result, output_file):
    '''
    convert results to JSON 
    '''
    with open(output_file,  "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

## FOR BIOID
def detect_bioid_pattern(text):
    # Detect the pattern presented in the processed bioid text in the hunflair2 experiments
    pattern = r'^\d+\|[a-zA-Z]\|'
    
    match = re.match(pattern, text)

    return bool(match)

def split_bioid_paragraph(text):
    # regex
    pattern = r'^(\d+\|[a-zA-Z]\|)(.*)'
    
    # Use re.match to check if the pattern exists at the beginning of the paragraph
    match = re.match(pattern, text)
    
    if match:
        # Extract the matched pattern and the rest of the paragraph
        matched_pattern = match.group(1)
        rest_of_paragraph = match.group(2).strip()
        return matched_pattern, rest_of_paragraph
    else:
        return None, paragraph
    
def convert_bioid_to_json(infile, outfile):
    '''
    Convert bioid pubtator file to json
    '''
    articles = {}
    f = open(infile, encoding="utf8")

    for line in tqdm(f.readlines()):
        if detect_bioid_pattern(line):

            id_, text_ = split_bioid_paragraph(line)
            
            if id_ not in articles:
                articles[id_]= {"title":id_, "abstract":text_}
            else:
                raise Exception("err")

    f.close()

    convert_to_json(articles, outfile)

## For medmentions
def detect_med_pattern(text):
    pattern = r'^\d+\|[a-zA-Z]\|'
    match = re.match(pattern, text)

    return bool(match)

def split_med_paragraph(text):
    # regex
    pattern = r'^(\d+\|[ta]|)(.*)'
#     pattern_a = r'^(\d+\|a|)(.*)'
    
    
    # Use re.match to check if the pattern exists at the beginning of the paragraph
    match = re.match(pattern, text)
    
    if match:
        # Extract the matched pattern and the rest of the paragraph
        matched_pattern = match.group(1)
        rest_of_paragraph = match.group(2).strip()
        
        return matched_pattern, rest_of_paragraph
    else:
        return None, paragraph
    
def join_title_and_abstract_med(articles):
    # format correction
    for id_ in articles:
        articles[id_]["abstract"] = "{} {}".format(articles[id_]["title"],articles[id_]["abstract"])
#         del articles[id_]["title"]
    return articles

def convert_medmention_to_json(infile, outfile):
    """
    Convert pubtator medmentions (hunflair2 experiments) to json
    """
    articles_med = {}
    f = open(infile, encoding="utf8")
    for idx, line in tqdm(enumerate(f.readlines())):
        if detect_med_pattern(line):
            temp_id, text_ = split_med_paragraph(line)
    #         print(temp_id, text_)
            id_, key_ = temp_id.split("|")
            id_ = int(id_)
            text_=text_.strip("|")
            if id_ not in articles_med:
                articles_med[id_]= {}
                
            if key_=="t" and "title" not in articles_med[id_]:
                articles_med[id_]["title"]= text_+"."
            elif key_=="a" and "abstract" not in articles_med[id_]:
                articles_med[id_]["abstract"]= text_
            else:
                raise Exception("ERR!!")

    f.close()
    # Joint title and abstract for medmentions format
    articles_med = join_title_and_abstract_med(articles_med)
    
    convert_to_json(articles_med, outfile)

## FOR TMVAR3
def detect_tmvar_pattern(text):
    #detect re pattern for tmvarv3
    pattern = r'^\d+\|[a-zA-Z]\|'
    match = re.match(pattern, text)

    return bool(match)

def split_tmvar_paragraph(text):
    # regex split text
    pattern = r'^(\d+\|[ta]|)(.*)'
#     pattern_a = r'^(\d+\|a|)(.*)'
    
    
    # Use re.match to check if the pattern exists at the beginning of the paragraph
    match = re.match(pattern, text)
    
    if match:
        # Extract the matched pattern and the rest of the paragraph
        matched_pattern = match.group(1)
        rest_of_paragraph = match.group(2).strip()
        
        return matched_pattern, rest_of_paragraph
    else:
        return None, paragraph

def join_title_and_abstract_tmvar(articles):
    for id_ in articles:
        articles[id_]["abstract"] = "{} {}".format(articles[id_]["title"],articles[id_]["abstract"])
#         del articles[id_]["title"]
    return articles

def convert_tmvar3_to_json(infile, outfile):
    #convert tmvar v3 from pubtator (hunflair2) to json format
    articles_tmvar = {}
    f = open(infile, encoding="utf8")
    for idx, line in tqdm(enumerate(f.readlines())):
        if detect_tmvar_pattern(line):
            temp_id, text_ = split_tmvar_paragraph(line)
    #         print(temp_id, text_)
            id_, key_ = temp_id.split("|")
            id_ = int(id_)
            text_=text_.strip("|")
            if id_ not in articles_tmvar:
                articles_tmvar[id_]= {}
                
            if key_=="t" and "title" not in articles_tmvar[id_]:
                articles_tmvar[id_]["title"]= text_
            elif key_=="a" and "abstract" not in articles_tmvar[id_]:
                articles_tmvar[id_]["abstract"]= text_
            else:
                raise Exception("ERR!!")

    f.close()

    articles_tmvar = join_title_and_abstract_tmvar(articles_tmvar)
    convert_to_json(articles_tmvar, outfile)
    


## FOR BIORED
def detect_biored_pattern(text):
    pattern = r'^\d+\|[a-zA-Z]\|'
    match = re.match(pattern, text)

    return bool(match)

def split_biored_paragraph(text):
    # regex
    pattern = r'^(\d+\|[ta]|)(.*)'
#     pattern_a = r'^(\d+\|a|)(.*)'
    
    
    # Use re.match to check if the pattern exists at the beginning of the paragraph
    match = re.match(pattern, text)
    
    if match:
        # Extract the matched pattern and the rest of the paragraph
        matched_pattern = match.group(1)
        rest_of_paragraph = match.group(2).strip()
        
        return matched_pattern, rest_of_paragraph
    else:
        return None, paragraph
    
def join_title_and_abstract_biored(articles):
    for id_ in articles:
        articles[id_]["abstract"] = "{} {}".format(articles[id_]["title"],articles[id_]["abstract"])
#         del articles[id_]["title"]
    return articles

def convert_biored_to_json(infile, outfile):
    # convert biored pubtator to json

    articles_biored = {}
    f = open(infile, encoding="utf8")
    for idx, line in tqdm(enumerate(f.readlines())):
        if detect_biored_pattern(line):
            temp_id, text_ = split_biored_paragraph(line)
    #         print(temp_id, text_)
            id_, key_ = temp_id.split("|")
            id_ = int(id_)
            text_=text_.strip("|")
            if id_ not in articles_biored:
                articles_biored[id_]= {}
                
            if key_=="t" and "title" not in articles_biored[id_]:
                articles_biored[id_]["title"]= text_+""
            elif key_=="a" and "abstract" not in articles_biored[id_]:
                articles_biored[id_]["abstract"]= text_
            else:
                raise Exception("ERR!!")

    f.close()

    articles_biored = join_title_and_abstract_biored(articles_biored)
    
    convert_to_json(articles_biored, outfile)


if __name__=="__main__":

    # notice the asterisk in the infolder path to get all biored files. Adjust the script as required
    infile = "/home/rafsan/aitslab/nlp/paper_review/hunflair2-experiments/annotations/BIORED/BioRED/Test.PubTator"


    outfile = "./biored.json"

    convert_biored_to_json(infile,outfile)