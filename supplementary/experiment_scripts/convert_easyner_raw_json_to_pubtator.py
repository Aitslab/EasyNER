#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
A script to convert EasyNER raw downloaded files, such as Lund Autophagy 1 & 2 to Pubtator format. The input files are expected to contain no annotation.
This script was primarily used to prepare JSON formatted datasets into pubtator format in order to run HunFlair 2 model and get predictions on the text.

"""

__author__ = 'Rafsan Ahmed'

import os
import json
import re
from glob import glob

def read_articles(filename):
    with open(filename, encoding="utf8") as f:
        return json.loads(f.read())
def strip_multi_newline(sentence):
    return re.sub(r'\n+', ' ', sentence).strip()

if __name__=="__main__":

    infile = "/home/rafsan/aitslab/nlp/EasyNER/results/dataloader/LA2_text.json"

    outfolder = "./lund_autophagy_test/"
    os.makedirs(outfolder, exist_ok=True)

    outfile = outfolder+"lund_autophagy_2.txt"

    with open(outfile, "w", encoding="utf8") as f:
        
        articles = read_articles(infile)
        
        for pmid, art in articles.items():
    #         sents = " ".join([strip_multi_newline(s["abstract"]) for s in art["sentences"]])
    #         sents = sents.replace("\n\n", "\n")
            title=art["title"].strip()
            abstract = strip_multi_newline(art["abstract"])
            f.write(f"{pmid}|t|{title}\n")        
            f.write(f"{pmid}|a|{abstract}\n\n")