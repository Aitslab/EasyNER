# EasyNER - Tutorials

![](imgs/pipeline3.png)

The end-to-end NER pipeline for information extraction is designed to take user given input and extract a list of entities. The figure above describes how the given text is processed through the pipeline. Following are step-by-step tutorials on running the NER pipeline

___

# 1. Installation

## 1.1 Download the GitHub repository

If you have git installed on the computer, open a terminal window and download the repository by writing: 

```
cd PATH TO YOUR FOLDER OF CHOICE (e.g. C:/Users/XYZ/)
git clone https://github.com/Aitslab/EasyNER/
```

Alternatively, you can download the repository from github page https://github.com/Aitslab/EasyNER to your designated folder as a zip file (click on 'Code' in the top right corner and then on 'Download ZIP') and unpack it.

![](imgs/github.png)

## 1.2 Install the conda environment

For running the pipeline, anaconda or miniconda must be installed in the computer. The step-by-step installation instructions can be found on: https://docs.anaconda.com/anaconda/install/index.html.

To install the necessary packages for running the environment, open a conda terminal and write the following commands:

```bash
conda env create -f environment.yml
```

After installation, load the environment with this:

```bash
conda activate easyner_env
```

___


# 2. Modify the Config file

The pipeline consists of several modules which are run in a sequential manner. It is also possible to run the modules individually. 

For each pipeline run, the config.json file in the repository needs to be modified with the desired settings. This can be done in any text editor. First, the modules that you want to run, should be switched to "false" in the ignore section. Then, the section for those modules should be modified as required. It is advisable to save a copy of the modified config file somewhere so you have a permanent record of the run. 

```bash
{
  "ignore": {
    "cord_loader": true,
    "downloader": true,
    "text_loader": true,
    "splitter": true,
    "ner": true,
    "analysis": false,
    "merger": true,
    "add_tags": true,
    "re": true,
    "metrics": true
  },
```
In a normal pipeline run, the following modules should be set to false, and the rest to true:

1. One of the data loaders depending on the input type (downloader, cord_loader or free_text loader).
2. splitter
3. ner
4. analysis

The following sections will provide more detail on each of the modules.

___

## 2.1 Data loader modules

The pipeline has three diffent modules for data loading, which handle different input types:

- List of Pubmed IDs => Downloader module
- CORD-19 metadata.csv file => CORD loader module
- Free text => Text loader module

### 2.1.1 Downloader
This downloader variant of the data loader module takes a single .txt file with pubmed IDs (one ID per row) as input and uses an API to retrieve abstracts from the Pubmed database. The output consists of a single JSON file with all titles and abstracts for the selected IDs. 

As example for the input, look at the file ["Lund-Autophagy-1.txt"](/data/Lund-Autophagy-1.txt). The easiest way to create such a file is to perform a search on Pubmed and then save the search results using the "PMID" format option:

![](imgs/pubmed.jpg)

To run the downloader module, change "downloader" in the ignore section to false (cord_loader and text_loader to true) and provide the following arguments in the "downloader" section of the config file:

#### Config file argument:
```console
    "input_path": path to file with pubmed IDs 
    "output_path": path to storage location for output
    "batch_size": number of article records downloaded in each call to API. Note that, too large of a batch size may invalid download requests.
```
#### example: 

![](imgs/downloader_.png)



### 2.1.2 CORD loader

The cord_loader variant of the data loader module processes titles and abstracts in the [CORD-19 dataset](https://github.com/allenai/cord19), a large collection of SARS-CoV2-related articles updated until 2022-06-02. For the CORD loader to work, the CORD19 dataset, which includes the metadata.csv file processed by the pipeline, first needs to be downloaded manually from the CORD-19 website ([direct download link](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2022-06-02.tar.gz)). The file path to the metadata.csv file should then be provided in the config file as input. By default, the module will process all titles and abstracts in the CORD-19 dataset (approximately 700 000 records). If a smaller subset is to be processed, a .txt file with the selected cord UIDs, which can be extracted from the metadata.csv file, needs to be provided. To run the CORD loader script, turn "cord_loader" in the ignore section to false (and data_loader and text_loader to true) and provide the following arguments:

#### Config file argument:
```console
    "input_path": input file path with CORD-19 metadata.csv file
    "output_path": path to storage location for output
    "subset": true or false - whether a subset of the CORD-19 data is to be extracted.
	"subset_file": input file path to a file with cord UIDs if subset option is set to true
```
#### example: 


![](imgs/cord_loader_.png)



### 2.1.3 Freetext loader

The text_loader variant of the dataloader module processess a file with free text and converts it into a JSON file. Similar to data_loader and cord_loader, the file path should be provided in the config files. The output JSON file will contain entries with prefix and a number as IDs and the filename as title. The number randomly assigned. To run the text_loader script, turn "text_loader" in the ignore section to false (and data_loader and cord_loader to true) and provide the following arguments:


#### Config file argument:
```console
    "input_path": input file path with free text. The folder may contain one or several .txt files.
    "output_path": output file (JSON format)
    "prefix": Prefix for the free-text files. 
```
#### example: 

![](imgs/text_loader_.png)

___


## 2.2 Sentence Spliter module

This module loads a JSON file (normally the file produced by the data loader module) and splits the text(s) into sentences with the spaCy or NLTK sentence splitter. The output is stored in one or several JSON files. To run the sentence splitter module set the ignore parameter for splitter in the config file to false. When using the spaCy option, the user needs to choose the model: "en_core_web_sm" or "en_core_web_trf". The number of texts that is processed together and stored in the same JSON output file is specified under "batch size". 

#### Config file argument:
```console
    "input_path": input file path of document collection
    "output_folder": output folder path where each bach will be saved
    "output_file_prefix": user-set prefix for output files
    "tokenizer": "spacy" or "nltk"
    "model_name": "en_core_web_sm" or "en_core_web_trf" for spaCy, for nltk write "" 
    "batch_size": number of texts to be processed together and saved in the same JSON file

```
#### example: 

![](imgs/splitter_.png)

___


## 2.3 Named Entity Recognition module

The NER module performs NER on JSON files containing texts split into sentences (normally the output files from the sentence splitter module). The user can use deep learning models or use the spaCy phrasematcher with dictionaries for NER. Setveral BioBERT-based models fine-tuned on the HUNER corpora collections and several dictionaries are available with the pipeline but the user can also provide their own. To run this module, the ignore argument for ner should be set to false and the following config arguments should be specified in the config file:

#### Config file argument:
```console
    "input_path": input folder path where all JSON batch files with texts split into sentences are located
    "output_folder": output folder path where each batch will be saved
    "output_file_prefix": user-set prefix for tagged output files
    "model_type": type of model; the user can choose between "biobert_finetuned" (deep learning models) and "spacy_phrasematcher" (dictionary-based NER)
    "model_folder": folder where model is located. For huggingface models use the repo name instead. Eg. "aitslab"
    "model_name": name of the model file located in the model folder or repository.
    "vocab_path": path to dictionary (if this option is used)
    "store_tokens":"no",
    "labels": if specific lavels are to be provided, e.g. ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"],
    "clear_old_results": overwrite old results
    "article_limit": if user decides to only choose a range of articles to run the model on, default [-1,9000]
    "entity_type": type of extracted entity, e.g. "gene"
```
#### example: 

![](imgs/ner_.png)

#### models and dictionaries

##### [BioBERT](https://github.com/dmis-lab/biobert-pytorch)-based NER

1. Cell-lines: biobert_huner_cell_v1 
2. Chemical: biobert_huner_chemical_v1
3. Disease: biobert_huner_disease_v1
4. Gene/protein: biobert_huner_gene_v1
5. Species: biobert_huner_species_v1

The BioBERT models above have been fine-tuned using the [HUNER corpora](https://github.com/hu-ner/huner) and uploaded to [huggingface hub](https://huggingface.co/aitslab). These and similar models can be loaded from the huggingface hub by setting the "model_path" to "aitslab" and "model_name" to the model intended for use in the NER section of the config file. For example:

```console
"model_type": "biobert_finetuned",
"model_path": "aitslab",
"model_name": "biobert_huner_chemical_v1"
```

##### Dictionary-based NER
[Spacy Phrasematcher](https://spacy.io/api/phrasematcher) is used to load dictionaries and run NER. COVID-19 related disease and virus dictionaries are provided [here](dictionaries/). 
Dictionary based NER can be run by specifying model_type as "spacy_phrasematcher", "model_name" as the spacy model (like, "en_core_web_sm" model) and specifying the "vocab_path" (path_to_dictionary) in the NER section of the config file. For example:

```console
"model_type": "spacy_phrasematcher",
"model_path": "",
"model_name": "en_core_web_sm",
"vocab_path": "dictionaries/sars-cov-2_synonyms_v2.txt"
```
___


## 2.4 Analysis module

This section uses the extracted entities to generate a file of ranked entities and frequency plots. First, as all the other steps above, set ignore analysis to false. Then use the following input and output config arguments:

#### Config file argument:
```console
    "input_path": input folder path where all batches of NER are located
    "output_path": output folder path where the analysis files will be saved
```
#### example: 

![](imgs/Analysis_.png)


#### output:

1. File with ranked entity list:

The generated output file contains the following columns:

| Column | Description |
| --- | ---|
| entity | Name of the entity |
| total_count | total occurances in the entire document set |
| articles_spanned | no of articles the entity is found |
| batches_spanned | no of batches the entity is found |
| freq_per_article | total_count/articles_spanned |
| freq_per_batch | total_count/batches_spanned |
| batch_set | batch IDs where the entity is found |
| batch_count | no of times the entity is found in each batch |
| articles_set | article IDs where the entity is found |

![](imgs/Analysis_out_01.png)



2. Bar graph of frequencies:

![](imgs/mtorandtsc1_disease_top_50.png)

___

## 2.5 Metrics module

The metrics module can be used to get precision, recall and F1 scores of between a true and a prediction file, as long as both are in IOB2 format. Note that BioBERT raw test prediction file is in IOB2 format. To run metrics, set ignore metrics to false in the config file. Then use the following input and output config arguments:

#### Config file argument:
```console
    "predictions_file": file containing predictions by the chosen model (in IOB2 format),
    "true_file": file containing true (annotated) values (also in IOB2 format),
    "output_file": file containing precision, recall and f1 scores,
    "pred_sep": seperator for predictions file, default is " ",
    "true_sep": seperator for true annotations file, default is " " 

```

#### Output

```console

              precision    recall  f1-score   support

           _    0.67557   0.65274   0.66396      1627

   micro avg    0.67557   0.65274   0.66396      1627
   macro avg    0.67557   0.65274   0.66396      1627
weighted avg    0.67557   0.65274   0.66396      1627

```
___


## 2.6 Merger module (optional)

The merger section combines results from multiple NER module runs into a single file for analysis. First, as all the other steps above, set ignore analysis to false. Then use the following input and output config arguments:

#### Config file argument:
```console
    "input_paths": list of input folder path where the files are saved. for example: ["path/to/cell/model/files/", "path/to/chemical/model/files/", "path/to/disease/model/files/"]
    
    "entities": list of entities correcponding to the models. For example: ["cell", "chemical", "disease"]
    "output_path": output path where the medged file will be saved
```
___

# 3. Run pipeline

When the configuration is saved, the pipeline can be executed by running the main.py file in the conda terminal:
```bash
python main.py
```

