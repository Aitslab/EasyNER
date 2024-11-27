#!/usr/bin/env python
# coding: utf-8


# Author Salma Kazemi Rashed
# The script is used to predict the IOB labels for tokenized text 
# It uses Flair Base models from https://nlp.informatik.hu-berlin.de/resources/models/
# This model could be any model from Flair repo listed in https://github.com/flairNLP/flair/models/sequence_tagger_model.py
# The output file is in the IOB format with three columns where the first column is tokens, The second is gold IOB labels and third is the predicted labels
# e.g., cholesterol B-Chemical B-Chemical
#       clefts O O  
# 

from flair.data import Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, ELMoEmbeddings
from typing import List
import torch
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.data import Sentence
from flair.models import SequenceTagger

# define columns

if __name__ == "__main__":
    '''
    Load dataset, define a ColumnCorpus, Call a SequenceTagger for chemical, gene, species, ..., use predict function or evaluate function.
    Print results 
    Example:
    By class:
                     precision    recall  f1-score   support

    Chemical         0.8827    0.8113    0.8455      6640

    micro avg        0.8827    0.8113    0.8455      6640
    macro avg        0.8827    0.8113    0.8455      6640
    weighted avg     0.8827    0.8113    0.8455      6640
    '''
    columns = {0: 'text', 1: 'ner'}
    
    # If gpu was available
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
    else:
        device = torch.device('cpu') 
        
    # Data is in test.tsv, devel.tsv., and train.tsv
    # A new line is added at the end of each sentences of each file with
    # cat test.tsv | sed s/'^\.\tO'/'\.\tO\n'/g
    data_folder = '../CRAFT-4.0.0/PR_gene_spacy_custom/'  

    # init a corpus using column format, data folder and the names of the train, dev and test files
    # Only test.tsv file have been used here for evaluation
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.tsv',
                                  test_file='test.tsv',
                                  dev_file='devel.tsv')   

    # load tagger (hunflair-species,hunflair-chemical, hunflair-genes)
    tagger = SequenceTagger.load('hunflair-chemical')  

    # The result of prediction will be saved in prediction_chemical.txt file and evaluated by flair script.
    result = tagger.evaluate(corpus.test, mini_batch_size=4, out_path=f"predictions_chemical.txt", gold_label_type="ner")
    print(result.detailed_results)
    
    
    '''
    It is also possible to use tagger's predict function, the input format is Sentence and it should evalute further by converting sentence into IOB format.
    
     
    for s in corpus.test[0]:
        result = tagger.predict(s)
        print(result)
    
   
    '''