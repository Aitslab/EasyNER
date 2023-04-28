# EasyNER: A Customizable and Easy-to-Use Pipeline for Deep Learning- and Dictionary-based Named Entity Recognition from Medical Text

EasyNER is a customizable end-to-end pipeline for extracting named entities from medicine-related texts. The pipeline comes with pre-trained models and dictionaries that can retrieve many biomedical entities: cells, chemicals, diseases, genes/proteins, species, COVID-19-related terms.  

![](tutorials/imgs/pipeline3.png)

## Quick start guide


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
conda activate env_pipeline
```

5. Provide documents in the form of PubMed IDs, CORD metadata, or through free text.


6. Edit the [config file](config.json) with correct paths to your documents. The config file contains an “ignore” section in the beginning to select the modules that the user wants to run. 


7. Run the following command:

```python
python main.py
```

8. The output will be in the form of a ranked list [(example)](results/sample_output/analysis_mtorandtsc1_chemical/mtorandtsc1_result_chemical.tsv) and a graph [(example)](results/sample_output/analysis_mtorandtsc1_chemical/mtorandtsc1_chemical_top_50.png)


### The complete configuration and inference tutorial can be found in this [collection of tutorials](tutorials/Tutorial-pipeline.md)  

___

## NER 

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
