# coding=utf-8

import json

def count_frequent_terms_from_ner(input_file, output_file, per_article=False):
    '''
    count term frequencies from detected entities 
    input_file: JSON input file path with entities
    output_file: output file path with frequencies'''
    
    with open(input_file, "r",encoding="utf-8") as f:
        articles = json.loads(f.read())
        
    dict_freq = {}
    
    if per_article:
        
        with open(output_file, "w", encoding="utf-8") as f:
        
            for i, (pmid, art) in enumerate(articles.items()):

                dict_freq[pmid]={}

                for sent in art["sentences"]:
                    if len(sent["entities"])!=0:

                        for ent in sent["entities"]:
                            if ent not in dict_freq[pmid]:
                                dict_freq[pmid][ent] = 0
                            dict_freq[pmid][ent]+=1
                            
                for k, v in sorted(dict_freq[pmid].items(), key=lambda item: item[1], reverse=True):
                    f.write(f"{pmid}\t{k}\t{v}\n")

    
    else:
        
        for i, (pmid, art) in enumerate(articles.items()):
    
            for sent in art["sentences"]:
                if len(sent["entities"])!=0:

                    for ent in sent["entities"]:
                        if ent not in dict_freq:
                            dict_freq[ent] = 0
                        dict_freq[ent]+=1
    
        with open(output_file, "w", encoding="utf-8") as f:
            for k, v in sorted(dict_freq.items(), key=lambda item: item[1], reverse=True):
                f.write(f"{k}\t{v}\n")
                


if __name__=="__main__":
    infile="../../NER_pipeline/results/text-ner_ft_celldeathA-set.json"
    outfile="../../test_predictions/res/test_output2.txt"
    count_frequent_terms_from_ner(infile, outfile,per_article=True)