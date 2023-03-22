# coding=utf-8

import spacy
from nltk.tokenize import sent_tokenize
import json
from tqdm import tqdm

def make_batches(list_id, n):
    #Yield n-size batches from list of ids
    for i in range(0, len(list_id), n):
        yield list_id[i:i + n]
        
def split_into_sentences_nltk(text):
    sentences = sent_tokenize(text)
    return sentences


def split_into_sentences_spacy(text,modelname):
    sentences = []
    nlp = spacy.load(modelname)
    doc = nlp(text)

    for sentence in doc.sents:
        sentences.append(str(sentence))

    return sentences
    
def split_batch(splitter_config, batch_idx, batch, full_articles, tokenizer="spacy", model="en_core_web_sm"):
    '''
    Description:
        split sentences in batches
        
    Parameters:
        batch_idx -> int: batch ID
        batch -> list: full batch with article IDs (for example: pubmed ID)
        full_articles -> dict: the entire collection of input articles with text
        tokenizer -> str: "spacy" or "nltk" sentencer
        model -> str: specific spacy model if needed
        
    Returns:
        batch_idx and split articles TO BE written into JSON files
    '''
    
    articles = {}

    for idx in tqdm(batch, desc=f'batch:{batch_idx}'):
        article=full_articles[idx]
        
        if tokenizer=="spacy":
        
            articles[idx] = {
                # **articles[id], # include other fields
                "title": article["title"],
                "sentences": list(map(
                    lambda sentence: {"text": sentence},
                    split_into_sentences_spacy(article["abstract"],model)
                ))
                }
        elif tokenizer=="nltk":
            articles[idx] = {
                # **articles[id], # include other fields
                "title": article["title"],
                "sentences": list(map(
                    lambda sentence: {"text": sentence},
                    split_into_sentences_nltk(article["abstract"])
                ))
                }
        else:
            raise Exception("ERROR! Proper sentence splitter model not specified!")
            
    
    with open(f'{splitter_config["output_folder"]}/{splitter_config["output_file_prefix"]}_{tokenizer}-split-{batch_idx}.json', "w",encoding="utf-8") as f:
                    f.write(json.dumps(articles, indent=2, ensure_ascii=False))
    
    return batch_idx
    
            
'''if __name__=="__main__":
    
    pass'''