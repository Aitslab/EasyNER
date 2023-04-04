# CIRP: A Customizable Information Retrieval Pipeline

CIRP is a Customizable, domain-specific Information Retrieval Pipeline to extract information from biomedical texts. The pipeline comes with pre-trained models that can retrieve many biomedical entities: cell-lines, chemicals, disease, gene/proteins and species.  

!!Insert Image

## Quick start guide

1. Before installation: Downnload and install anaconda from https://www.anaconda.com/


2. Clone the repository to your target folder


```console
git clone https://github.com/Aitslab/CIRP

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
