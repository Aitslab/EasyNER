# LUMINER: A Customizable and Easy-to-Use Pipeline for Deep Learning- and Dictionary-based Named Entity Recognition from Medical Text

LUMINER is a customizable End-to-End Information Retrieval Pipeline developed at Lund University for extracting named entities from medcine-related texts. The pipeline comes with pre-trained models and dictionaries that can retrieve many biomedical entities: cell-lines, chemicals, disease, gene/proteins, species, COVID-19 related terms.  

![](tutorials/imgs/pipeline3.png)

## Quick start guide


1. Before installation: Downnload and install anaconda from https://www.anaconda.com/


2. Clone the repository to your target folder


```console
git clone https://github.com/Aitslab/LUMINER

```

3. Set up an conda environment

```console
conda env create -f environment.yml
```

4. After installation activate the environment:
```console
conda activate env_pipeline
```

5. Provide documents in the form of PubMed IDs, CORD metadata, or through free text.


6. Edit the [config file](config.json) with correct paths to your documents.


7. Run the following command:

```python
python main.py
```

8. The output will be in the form of a ranked list [(example)](results/sample_output/analysis_mtorandtsc1_chemical/mtorandtsc1_result_chemical.tsv) and a graph [(example)](results/sample_output/analysis_mtorandtsc1_chemical/mtorandtsc1_chemical_top_50.png)


### The complete configuration and inference tutorial can be found in this [collection of tutorials](tutorials/Tutorial-pipeline.md)  

___

## Models

### [BioBERT](https://github.com/dmis-lab/biobert-pytorch) based

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

#https://github.com/Aitslab/LUMINER/blob/80dd78e21c1fc75ed346897d977b25896f76838e/config.json#L39-L53


### Dictionary based
[Spacy Phrasematcher](https://spacy.io/api/phrasematcher) is used to load dictionaries and run NER. COVID-19 related disease and virus dictionaries are provided [here](dictionaries/). 
Dictionary based NER can be run by specifying model_type as "spacy_phrasematcher", "model_name" as the spacy model (like, "en_core_web_sm" model) and specifying the dictionary path in "vocab_path". For example:

```console
"model_type": "spacy_phrasematcher",
"model_path": "",
"model_name": "en_core_web_sm",
"vocab_path": "dictionaries/LU_GS-disease.txt"
```

#### For BioBERT model training script follow this [tutorial](tutorials/Tutorial-BioBERT_model_training.ipynb)
#### All preprocessing scripts can be found [here](supplementary/preprocessing_scripts/)


## Citation
```bibtex
@article{ahmed2023LUMINER,
  title={LUMINER: A Customizable, Easy-to-Use Pipeline for Deep Learning- and Dictionary-based Named Entity Recognition from Medical Text},
  author={Ahmed, Rafsan and Berntsson, Petter and Skafte, Alexander and Kazemi Rashed, Salma and Klang, Marcus and Barvesten, Adam and Olde, Ola and Lindholm, William and Arrizabalaga, Antton Lamarca and Nugues, Pierre and Aits, Sonja},
  journal={},
  volume={},
  number={},
  pages={},
  year={2023},
  publisher={}
}
```