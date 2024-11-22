This is a step by step guide to evaluating model predictions with the run_ner.py script provided with BioBERT, which can be used for both NER prediction and evaluation.

https://github.com/dmis-lab/biobert-pytorch/blob/master/named-entity-recognition/run_ner.py

The script calculates precision, recall and F1 scores on token-level.

# Combined prediction and evaluation procedure for BioBERT models
1. Convert the file with gold standard annotations to IOB2 format (if needed, see below). 
2. Convert the IOB2 file to BioBERT input format by running the (BioBERT preprocess.sh script)[https://github.com/dmis-lab/biobert-pytorch/blob/master/named-entity-recognition/preprocess.sh]. We used a max sequence length at the default value of 128, which splits sentences larger than this length into two.
3. Make predictions and do the evaluation at the same time by running run_ner.py

# Conversion to IOB2 format
The CRAFT corpus in Pubannotation format (and any other corpus with this format) needs to be converted to IOB2 format before it can be used with BioBERT. 
This is done with [CRAFT_preprocessing_spacy.py](https://github.com/Aitslab/EasyNER/blob/main/supplementary/experiment_scripts/CRAFT_preprocessing_spacy.py)


