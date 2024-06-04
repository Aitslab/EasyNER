# EasyNER: A Customizable and Easy-to-Use Pipeline for Deep Learning- and Dictionary-based Named Entity Recognition from Medical Text

EasyNER is a customizable end-to-end pipeline for extracting named entities from medicine-related texts. The pipeline comes with pre-trained models and dictionaries that can retrieve many biomedical entities: cells, chemicals, diseases, genes/proteins, species, COVID-19-related terms.  

![](tutorials/imgs/pipeline3.png)

## How to use the pipeline

A detailed guide, including installation, configuration and inference tutorial can be found in this [collection of tutorials](tutorials/Tutorial-pipeline.md).

### Quick start guide

1. Before installation: Downnload and install anaconda from https://www.anaconda.com/


2. Clone the repository to your target folder


```console
git clone https://github.com/Aitslab/EasyNER

```

3. Set up an conda environment

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


6. Provide input file: list of PubMed IDs, CORD19 metadata.csv file, or file with free text.


7. Add the correct paths to your input file in the [config file](config.json) and choose the modules you want to run in the “ignore” section in the beginning of the file. 


8. Run the pipeline with the following command:

```python
python main.py
```

9. The output will consist of a ranked list [(example)](results/sample_output/analysis_mtorandtsc1_chemical/mtorandtsc1_result_chemical.tsv) and a graph [(example)](results/sample_output/analysis_mtorandtsc1_chemical/mtorandtsc1_chemical_top_50.png) and files with the annotated text.


A reproducable capsule is available on Code Ocean: https://doi.org/10.24433/CO.6908880.v1


___

## Bulk Download PubMED [NOTE: DUE TO A CHANGE IN THE PUBMED DATACENTER THIS IS NOT WORKING RIGHT NOW. WE ARE UPDATING IT]

The EasyNER pipeline includes a script for bulk downloading PubMed abstracts*. The bulk loader script will download, process and convert (to json) PubMed abstract collection from the annual baseline (currently 2023) (more insight here: https://ftp.ncbi.nlm.nih.gov/pubmed/). The pubmed_bulk_loader section of the config file is as follows:

 ```
 "pubmed_bulk_loader": {
    "output_path": "data/pubmed/",
    "baseline": "23",
    "subset": false,
    "subset_range":[0,0],
    "get_nightly_update_files": false,
    "update_file_range":[0,0],
    "count_articles": true,
    "raw_download_path": ""
  },
 ```

To process the bulk PubMed files through the pipeline, all you need to do is to make the following changes in the config file:

1. In the ignore section, make sure that the downloader, cord_loader and text_loader parameters are set to "true", and pubmed_bulk_loader section is set to "false".
2. In the splitter section Specify the pubmed folder path in the "input_path" parameter ("data/pubmed/" from the above example).
3. In the splitter section, set "pubmed_bulk" to "true".

Then run the pipeline as you would normally.


### Downloading a subset

The PubMed annual baseline files are numbered. If you want to download the files in the range 300 to 700 from the annual baseline, simply update the config file as follows ("subset" and "subset_range" sections) :

 ```
 "pubmed_bulk_loader": {
    "output_path": "data/pubmed/",
    "baseline": "23",
    "subset": true,
    "subset_range":[300,700],
    "get_nightly_update_files": false,
    "update_file_range":[0,0],
    "count_articles": true,
    "raw_download_path": ""
  },
 ```

### Downloading nightly updates

The PubMed annual baseline files are updated every year. However, they provide additional nightly update files during the year. To download the update files alongside the annual baseline, adjust the config file accordingly:

 1. Set "get_nightly_update_files" to "true"
 2. Provide a range in the "update_file_range" section. This needs to be set by the user. The file ranges for 2023 can be seen here: https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/

* Note that bulk downloading files requires enough storage space in your device and may take several hours to download and process millions of articles. An err file is generated in the end to account for files that are not downloaded. Kindly refer to the err file (same folder as raw_download_path) for missing files.
 ____

## Named Entity Recognition (NER) 

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
