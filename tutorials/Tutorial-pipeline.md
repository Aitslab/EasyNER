# EasyNER Installation and Useage

![](imgs/pipeline3.png)

The end-to-end NER pipeline for information extraction is designed to process text provided by the user and extract a list of entities. The figure above describes how the given text is processed through the pipeline. Following are step-by-step instructions on installing and running the NER pipeline.

___

# 1. Installation

## 1.1 Install Python
EasyNER is written in the Python programming language. Before you use EasyNER, you need to install Python. The easiest way to do this is to use Anacondaa. 
Anaconda can be downloaded from https://www.anaconda.com/ and step-by-step installation instructions can be found on https://docs.anaconda.com/anaconda/install/index.html.  

EasyNER requires very little knowledge of Python but you probably find it easier to use if you watch our [Python tutorial](https://youtu.be/yDNBdB59J1s) on YouTube.
The notebook for the tutorial can be found [here](https://github.com/COMPUTE-LU/PLGroup_PythonforBeginners/blob/main/kickoff_tutorial_20210127.ipynb). You can open the notebook by copying it to google drive and then loading it in colab (see our [tutorial for colab](https://github.com/Aitslab/training/blob/master/tutorials/colab.md)). 

EasyNER was developed using Python version 3.9 but also works with other Python versions. If you do encounter issues, install Python 3.9.



## 1.2 Download the EasyNER GitHub repository

If you have Git installed on the computer, open a terminal window and download the repository by writing: 

```
cd PATH TO YOUR FOLDER OF CHOICE (e.g. C:/Users/XYZ/)
git clone https://github.com/Aitslab/EasyNER/
```

Alternatively, you can download the repository from github page https://github.com/Aitslab/EasyNER to your designated folder as a zip file (click on 'Code' in the top right corner and then on 'Download ZIP') and unpack it.

![](imgs/github.png)

## 1.3 Install the conda environment

Conda environments make it possible to install and manage specific versions of software and their dependencies without interfering with other project. It is best to install EasyNER in a new environment. To install the necessary packages for running the environment, open a conda terminal (called "Anaconda prompt" in the Windows program list) and navigate to the EasyNER folder you downloaded using the change directory command (cd). For example:
```bash
(base) C:\Users\YOURUSERNAME>cd C://Users//YOURUSERNAME//Documents//git_repos//EasyNER
```

Then create a new environment called "easyner_env" by writing the following command:

```bash
conda env create -f environment.yml
```

If you get an error "CondaValueError: prefix already exists" open the environment.yml file which is located in the main folder of the EasyNER folder and change the name of the environment name in the first row.

After installation, load the environment in the conda terminal with this:

```bash
conda activate easyner_env
```

___


# 2. Modify the Config file

EasyNER consists of several modules which are run in a sequential manner. It is also possible to run the modules individually. 

For each run, the config.json file in the EasyNER folder needs to be modified with the desired settings and file paths. This can be done in any text editor. First, the modules that you want to run, should be switched to "false" in the ignore section. Then, the sections for those modules should be modified as required. It is advisable to save a copy of the modified config file elsewhere so you have a permanent record of the run. 

```bash
{
  "ignore": {
    "cord_loader": true,
    "downloader": true,
    "text_loader": true,
    "pubmed_bulk_loader": false,
    "splitter": true,
    "ner": true,
    "analysis": false,
    "merger": true,
    "add_tags": true,
    "re": true,
    "metrics": true
  },
```
In a standard pipeline run, the following modules should be set to false (and the rest to true):

1. One of the data loaders depending on the input type (downloader, cord_loader, free_text loader or pubmed_bulk_loader).
2. splitter
3. ner
4. analysis

When the config file is updated, save it and start the run (see step 3.).

The following sections will provide more detail on each of the modules and instructions for the config file.

___

## 2.1 Data loader modules

The EasyNER pipeline has four diffent modules for data loading, which handle different input types:
- List of PubMed IDs => Downloader module
- PubMed database => Pubmed bulk loader module
- CORD-19 metadata.csv file => CORD loader module
- Plain text => Text loader module

The data loader modules convert the respective input files to one or several JSON files (depending on the number of texts your are processing). These will be placed in the designated output folde and can be processed further by the other EasyNER modules. If the other modules are not run directly, the downloaded files can be saved and the pipeline can be started at the sentence splitter module later.

Example of the generated JSON file:
![](imgs/output_downloader.png)

### 2.1.1 Downloader
The Downloader module takes a single .txt file with pubmed IDs (one ID per row) as input and uses an API to retrieve abstracts from the PubMed database. The output consists of a single JSON file with all titles and abstracts for the selected IDs. 

As example for the input, look at the file ["Lund-Autophagy-1.txt"](/data/Lund-Autophagy-1.txt). The easiest way to create such a file is to perform a search on Pubmed and then save the search results using the "PMID" format option:

![](imgs/pubmed.jpg)

To run the Downloader module, change "downloader" in the ignore section of the config file to false (cord_loader, text_loader and pubmed_bulk_loader to true) and provide the following arguments in the "downloader" section of the config file:

#### Config file arguments:
```console
    "input_path": path to file with pubmed IDs 
    "output_path": path to storage location for output
    "batch_size": number of article records downloaded in each call to API. Note that a too large batch size may result in invalid download requests.
```
#### example: 

![](imgs/downloader_.png)



### 2.1.2 PubMed Bulk loader

The entire PubMed abstract collection is available for download in the form of an annual baseline, updated only once per year, and [nightly update files](https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/). You can read more about this [here](https://ftp.ncbi.nlm.nih.gov/pubmed/README.txt) and [here](https://pubmed.ncbi.nlm.nih.gov/download/). The abstracts are bundled into a large number of gz files. The baselie version number is indicated in the file names after the word "pubmed" and the second number **without the starting zeros** is the file number, e.g. pubmed24n0001.xml.gz has the baseline version 24 and the file number 1.

The PubMed bulk loader variant of the dataloader module can download the annual baseline and nightly update files and converts the gz files into JSON files. In addition a list of the downloaded PMIDs is generated (pmid.txt) and the number of files in each gz file can be counted automatically. Note that the download of the entire article collection requires enough storage space on your computer and may take several hours. An err.txt file is generated to keep track of files that are not downloaded. Missing files can be downloaded in a second EasyNER run or manually from the ftp sites of the baseline and update files. If the err.txt file is empty no errors occured.

Similar to other data loader modules, to run this data loader turn "pubmed_bulk_loader" in the ignore section to "false" (and data_loader, cord_loader and text_loader to "true") and provide the following arguments:

```console
    "output_path": Path to the folder in which the output files (JSON files, PMID list and the files with counts (optional) are to be saved
    "baseline": The PubMed annual baseline version listed in the file names, e.g. 24 in pubmed24n0001.xml.gz
    "subset": Set tp "true" if a subset of the baseline is to be downloaded, otherwise "false" downloads the entire baseline.
    "subset_range": Specify a range of file numbers (without the starting zeros) if a subset of files is to be downloaded, e.g. to download files numbered 1 to 160 (inclusive) add [1,160],
    "get_nightly_update_files": Set to "true" if nightly update files are to be downloaded alongside the annual baseline, otherwise set to "false". Note that a range must be provided under "update_file_range" if this is set to "true".
    "update_file_range": Provide the range of update files to be downloaded if "get_nighly_update" is set to "true", e.g. [1167,1298] to download files 1167 to 1298 (inclusive). To see the available files, check: https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/
    "count_articles": Set to "true" if the number of articles within each file is to be counted and stored in a file called count.txt in the output folder. Otherwise, set to "false".
    "raw_download_path": Path to the folder where the gz files and err.txt file are to be saved. If it is left empty ("raw_download_path": "") the gz files and error file are not saved.
```

If you only want to download the update files, set subset and get_nightly_update_files to "true" and subset_range to [0,0]. Then define the range of update files under update_file_range.

If not all the files you defined were downloaded, there might be an issue with the PubMed server. Try again later.

#### example: 

![](imgs/pubmed_bulk_loader_.png)
___

### 2.1.3 CORD loader

[CORD-19 dataset](https://github.com/allenai/cord19) is a large collection of SARS-CoV2-related articles updated until 2022-06-02. The cord_loader variant of the data loader module processes titles and abstracts in the CORD-19 metadata.csv file. To use the cord_cord loader, CORD-19 needs to be downloaded manually ([direct download link](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2022-06-02.tar.gz)). The downloaded gz file should then be unpacked and the metadata.csv file placed in the desired input folder. The file path to the metadata.csv file should then be provided in the config file as input path. By default, the module will process all titles and abstracts in the CORD-19 dataset (approximately 700 000 records). If a smaller subset is to be processed, a .txt file with the selected cord UIDs, which can be extracted from the metadata.csv file, needs to be provided. To run the CORD loader script, turn "cord_loader" in the ignore section to false (and downloader, pubmed_bulk_loader and text_loader to true) and provide the following arguments:

#### Config file arguments:
```console
    "input_path": input file path for CORD-19 metadata.csv file
    "output_path": path to storage location for output files
    "subset": true or false - whether a subset of the CORD-19 data is to be extracted.
    "subset_file": input file path to a file with cord UIDs if subset option is set to true
```
#### example: 


![](imgs/cord_loader_.png)



### 2.1.4 Freetext loader

The text_loader variant of the dataloader module processess a file with free text and converts it into a JSON file. Similar to data_loader and cord_loader, the file path should be provided in the config files. The output JSON file will contain entries with prefix and a number as IDs and the filename as title. The number randomly assigned. To run the text_loader script, turn "text_loader" in the ignore section to false (and downloader, pubmed_bulk_loader and cord_loader to true) and provide the following arguments:


#### Config file arguments:
```console
    "input_path": input file path with free text. The folder may contain one or several .txt files.
    "output_path": output file (JSON format)
    "prefix": Prefix for the free-text files. 
```
#### example: 

![](imgs/text_loader_.png)





## 2.2 Sentence Spliter module

This module loads a JSON file (normally the file produced by the data loader module) and splits the text(s) into sentences with the spaCy or NLTK sentence splitter. The output is stored in one or several JSON files. To run the sentence splitter module set the ignore parameter for splitter in the config file to false. When using the spaCy option, the user needs to choose the model: "en_core_web_sm" or "en_core_web_trf". The number of texts that is processed together and stored in the same JSON output file is specified under "batch size". 

#### Config file arguments:
```console
    "input_path": input file path of document collection
    "output_folder": output folder path where each bach will be saved
    "output_file_prefix": user-set prefix for output files
    "tokenizer": "spacy" or "nltk"
    "model_name": "en_core_web_sm" or "en_core_web_trf" for spaCy, for nltk write "" 
    "batch_size": number of texts to be processed together and saved in the same JSON file
    "pubmed_bulk": make "true" if pubmed_bulk_loader is used, otherwise use "false"

```
#### example: 

![](imgs/splitter_.png)

___


## 2.3 Named Entity Recognition module

The NER module performs NER on JSON files containing texts split into sentences (normally the output files from the sentence splitter module). The user can use deep learning models or use the spaCy phrasematcher with dictionaries for NER. Setveral BioBERT-based models fine-tuned on the HUNER corpora collections and several dictionaries are available with the pipeline but the user can also provide their own. To run this module, the ignore argument for ner should be set to false and the following config arguments should be specified in the config file:

#### Config file arguments:
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

#### Config file arguments:
```console
    "input_path": input folder path where all batches of NER are located,
    "output_path": output folder path where the analysis files will be saved,
    "entity_type": type of entity, this will be added as a prefix to the output file and bar graph,
    "plot_top_n": plot top n entities. defaults to 50. Note that plotting more than 100 entities can result in a distorted graph

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

#### Config file arguments:
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

The merger module combines results from multiple NER module runs into a single file for analysis. First, as all the other steps above, set ignore analysis to false. Then use the following input and output config arguments:

#### Config file argument:
```console
    "input_paths": list of input folder path where the files are saved. for example: ["path/to/cell/model/files/", "path/to/chemical/model/files/", "path/to/disease/model/files/"]
    
    "entities": list of entities correcponding to the models. For example: ["cell", "chemical", "disease"]
    "output_path": output path where the medged file will be saved
```
___

# 3. Run EasyNER pipeline

When the updated config file is saved, the EasyNER pipeline can be run by activating the easyner_env environment in the conda terminal (Anaconda Prompt), navigating to the easyner folder with the "cd" command, and starting the main.py file in the conda terminal:

```bash
conda activate easyner_env
```


```bash
(easyner_env) C:\Users\YOURUSERNAME>cd C://Users//YOURUSERNAME//Documents//git_repos//EasyNER
```


```bash
(easyner_env) C://Users//YOURUSERNAME//Documents//git_repos//EasyNER>python main.py
```

The output files are placed into your designated output folder. When the run is completed "Program finished successfully will appear in the conda terminal:

![](imgs/easyner_run.png)
