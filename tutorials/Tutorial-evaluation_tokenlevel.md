This is a step by step guide to evaluating model predictions with the run_ner.py script provided with BioBERT, which can be used for both NER prediction and evaluation.

https://github.com/dmis-lab/biobert-pytorch/blob/master/named-entity-recognition/run_ner.py

The script calculates precision, recall and F1 scores on token-level.

# Combined prediction and evaluation procedure for BioBERT models
1. Convert the corpus to IOB2 format, if needed (see below an example conversion procedure for the CRAFT corpus that can be used for PubAnnotation format corpora as well)
2. Convert the IOB2 file to BioBERT input format by running the [BioBERT preprocess.sh script](https://github.com/dmis-lab/biobert-pytorch/blob/master/named-entity-recognition/preprocess.sh). We used a max sequence length at the default value of 128, which splits sentences larger than this length into two. If the corpus is not in IOB2 format
3. The evaluation is run for model and a single class at a time. Make an empty input folder and place your file in it (no othe files should be in the folder).
4. Specify the input (DATA_DIR) and output folder (SAVE_DIR) like this and create the output folder
```console
export SAVE_DIR=/save/directory/for/the/model/
export DATA_DIR=/path/to/training/datasets/
$ mkdir -p $SAVE_DIR
```
4. Specify the maximum sequence length. We set it to 192.
```console
export MAX_LENGTH=192
```
6. Run combined prediction and evaluation with run_ner.py. 
```console

python run_ner.py \
    --data_dir ${DATA_DIR}/${ENTITY}/ \
    --labels ${DATA_DIR}/${ENTITY}/labels.txt \
    --model_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --output_dir ${SAVE_DIR}/${ENTITY}\
    --max_seq_length ${MAX_LENGTH} \
    --do_eval \
    --do_predict \
    --overwrite_output_dir
```

# Preprocessing to IOB2 format for CRAFT corpus and other corpora in PubAnnotation format
1. Download the CRAFT corpus (Version 4.0.0) from https://github.com/UCDenver-ccp/CRAFT/releases/tag/v4.0.0.
2. Convert the downloaded CRAFT corpus to PubAnnotation format by following the conversion instructions: https://github.com/UCDenver-ccp/CRAFT/wiki/Alternative-annotation-file-formats. This produces individual folders for each entity class.
3. Convert the corpus in Pubannotation format to IOB2 format so it can be used with BioBERT. This is done with [CRAFT_preprocessing_spacy.py](https://github.com/Aitslab/EasyNER/blob/main/supplementary/experiment_scripts/CRAFT_preprocessing_spacy.py). Run once for each of the three entity classes by changing the input folder and output file
7.	Prepare each of the three IOB2 files for BioBERT input using the BioBERT preprocessing script ; ;aximum sequence was set to 128.
8.	evaluate the models by running run_ner.py ; run one model at a time with the IOB2 file for the respective class; maximum sequence length was set to 192

When working with other corpora in PubAnnotation format simply start at step 3.

# Evalution of non-BioBERT predictions (e.g. from HunFlair or ScispaCy)
ADD DESCRIPTION
