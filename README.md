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

5. Edit the config file with correct paths to your documents


6. Run the following command:

```python
python main.py
```

___


### The complete configuration and inference tutorial can be found here: [Tutorials](tutorials/Tutorial-pipeline.md)  
#### For BioBERT model training script follow this [tutorial](tutorials/Tutorial-BioBERT_model_training.ipynb)
#### All preprocessing scripts can be found [here](supplementary/preprocessing_scripts/)
