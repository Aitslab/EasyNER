# EasyNER: A Customizable and Easy-to-Use Pipeline for Deep Learning- and Dictionary-based Named Entity Recognition from Medical Text

EasyNER is a easy to use, customizable end-to-end pipeline for extracting named entities from medicine-related texts. The pipeline comes with pre-trained models and dictionaries that can retrieve many biomedical entities: cells, chemicals, diseases, genes/proteins, species, COVID-19-related terms.  

![](tutorials/imgs/pipeline3.png)


## What are the folders in this repo?
- **data**: datasets from the EasyNER article
- **dictionaries**: 3 COVID-19-related dictionaries for the EasyNER pipeline
- **models**: empty folder to copy the models into on your computer
- **pubmed_loader**: everything needed for using EasyNER with the PubMed bulk files
- **results**: example output files; when running EasyNER you can use this to store your output files
- **scripts**: EasyNER pipeline scripts
- **supplementary**: supplementary files for the EasyNER article and all info and scripts to replicate the benchmarking experiments from the article
- **tutorials**: step-by-step instructions on how to install and use EasyNER

## How to use the pipeline

A detailed guide, including installation, configuration and inference tutorial can be found in this [collection of tutorials](tutorials/Tutorial-pipeline.md).

### Quick start guide

Before installation of EasyNER:
Before you use EasyNER, you need to install Python. This easiest way to do this is to use Anaconda. For this, download Anaconda and follow this instructions from https://www.anaconda.com/. 

1. Transfer the EasyNER repo to your computer by downloading it manually or cloning the EasyNER GitHub repository to your target folder.


```console
git clone https://github.com/Aitslab/EasyNER

```

2. Set up an conda environment

```console
cd EasyNER
conda env create -f environment.yml
```


4. After installation activate the environment:
```console

conda activate easyner_env

```


5. Load spacy

```console

python -m spacy download en_core_web_sm
```


6. Choose the input file: list of PubMed IDs, CORD19 metadata.csv file, or file with plain text. If you want to download and process the entire PubMed article collection follow the instructions below.


7. Add the correct paths to your input file in the [config file](config.json). Choose the modules you want to run in the “ignore” section in the beginning of the file and save the desired settings for these modules in the respective sections. 


10. Run the pipeline with the following command:

```python
python main.py
```

The output will consist of a ranked list [(example)](results/sample_output/analysis_mtorandtsc1_chemical/mtorandtsc1_result_chemical.tsv), a graph of the most frequent entities [(example)](results/sample_output/analysis_mtorandtsc1_chemical/mtorandtsc1_chemical_top_50.png) and files with the annotated text.

NOTE: If you want to merge entities on the same corpus, make sure the same batch size is used for all runs.

A reproducable capsule for EasyNER is available on Code Ocean: https://doi.org/10.24433/CO.6908880.v1


___

## PubMed Bulk Download

The EasyNER pipeline includes a data loader module that can download, process and convert (to JSON files) the entire PubMed abstract collection. This is provided as the annual baseline, updated only once per year, and nightly update files. You can read more about this [here](https://ftp.ncbi.nlm.nih.gov/pubmed/README.tx) and [here](https://pubmed.ncbi.nlm.nih.gov/download/). The abstracts are bundled into a large number of gz files. The baselie version number is indicated in the file names after the word "pubmed" and the second number is the file number, e.g. pubmed24n0001.xml.gz. 

Note that the download of the entire article collection requires enough storage space on your computer and may take several hours. An err.txt file is generated in the end (in the folder specified under "raw_download_path") to keep track of files that are not downloaded. Missing files can be downloaded in a second EasyNER run or manually from the ftp sites of the baseline and update files.

The pubmed_bulk_loader section of the config file is as follows:

 ```
 "pubmed_bulk_loader": {
    "output_path": "data/pubmed/",
    "baseline": "24",
    "subset": false,
    "subset_range":[0,0],
    "get_nightly_update_files": false,
    "update_file_range":[0,0],
    "count_articles": true,
    "raw_download_path": ""
  },
 ```

When using this module you need to make the following changes in the config file:

1. In the ignore section, make sure that the downloader, cord_loader and text_loader parameters are set to "true", and pubmed_bulk_loader section is set to "false".
2. In the pubmed_bulk_loader section, specify the desired output path. 
3. In the pubmed_bulk_loader section, specify the baseline version
4. In the splitter section Specify the pubmed folder path in the "input_path" parameter ("data/pubmed/" from the above example).
5. In the splitter section, set "pubmed_bulk" to "true".

Then run the pipeline. 


### Downloading a subset of the PubMed annual baseline

If you want to download only a subset of the files in the the annual baseline, set "subset" to true and indicate the start and end file in the "subset_range" section:

 ```
 "pubmed_bulk_loader": {
    "output_path": "data/pubmed/",
    "baseline": "24",
    "subset": true,
    "subset_range":[300,700],
    "get_nightly_update_files": false,
    "update_file_range":[0,0],
    "count_articles": true,
    "raw_download_path": ""
  },
 ```

### Downloading nightly update files

To download the update files alongside the annual baseline, adjust the config file accordingly:

 1. Set "get_nightly_update_files" to "true"
 2. Provide the range of update file numbers to be downloaded in the "update_file_range" section. The files that have been released so far can be seen at https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/

 ____

## Named Entity Recognition (NER) 

EasyNER can identify entities with BioBERT models (or other models in the same format) or dictionaries (= long lists of the words which are to be detected). Five BioBERT models and three dictionaries are included but the use can also provide their own models or dictionaries.

### [BioBERT](https://github.com/dmis-lab/biobert-pytorch)-based NER

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

### Dictionary-based NER

[Spacy Phrasematcher](https://spacy.io/api/phrasematcher) is used to load dictionaries and run NER. COVID-19 related disease and virus dictionaries are provided [here](dictionaries/). 
Dictionary based NER can be run by specifying model_type as "spacy_phrasematcher", "model_name" as the spacy model (like, "en_core_web_sm" model) and specifying the "vocab_path" (path_to_dictionary) in the NER section of the config file. For example:

```console
"model_type": "spacy_phrasematcher",
"model_path": "",
"model_name": "en_core_web_sm",
"vocab_path": "dictionaries/sars-cov-2_synonyms_v2.txt"
```

#### For BioBERT model training script follow this [tutorial](tutorials/Tutorial-BioBERT_model_training.ipynb)
#### All preprocessing scripts can be found [here](supplementary/preprocessing_scripts/)


## Result inspection

We have included a result inspection module to search a list of entities occuring within the NER results. The result includes the PubMed ID, sentence where the entity occurs, additional entities within that sentence and their spans. In order to run inspection, do the following in the config file:

1. In the ignore section, set "result_inspection" to "true" and everything else to false.
2. Provide the input folder path (with a "/" in the end), the output file path and the list of entities to search in a list as shown below.
```console
"result_inspection":{
    "input_folder": "results/ner/",
    "output_file": "results/ner_search.txt",
    "entities": ["tsc", "mtor", "cell", "cells", "rapamycin"]
```

3. Run the following command as you would do to run the pipeline.
```python
python main.py
```

## Logging time
The runtime for EasyNER and the modules can be obtained by selecting "TIMEKEEP": true in the config file.

## Citation
If you use any of the material in this repository, please cite the following article:

```bibtex
@article{ahmed2023easyner,
  title={EasyNER: A Customizable Easy-to-Use Pipeline for Deep Learning- and Dictionary-based Named Entity Recognition from Medical Text},
  author={Rafsan Ahmed and Petter Berntsson and Alexander Skafte and Salma Kazemi Rashed and Marcus Klang and Adam Barvesten and Ola Olde and William Lindholm and Antton Lamarca Arrizabalaga and Pierre Nugues and Sonja Aits},
  year={2023},
  eprint={2304.07805},
  archivePrefix={arXiv},
  primaryClass={q-bio.QM}
}
```
