*** UNDER CONSTRUCTION ***


All info and scripts to reproduce the benchmarking experiments for the EasyNER article are in this folder.

## Datasets and preprocesing scripts
We used the following datasets and preprocessing scripts

| Corpus                 | File                                                                                                        | Format    | Script| 
|------------------------|-------------------------------------------------------------------------------------------------------------|-----------|-------|
| Lund-Autophagy-1       | [Lund-Autophagy-1.txt](https://github.com/Aitslab/EasyNER/blob/main/data/Lund-Autophagy-1.txt)              | PMID list |       |
| Lund-Autophagy-2       | [Lund-Autophagy-2.txt](https://github.com/Aitslab/EasyNER/blob/main/data/Lund-Autophagy-2.txt)              | PMID list |       |
| Lund-COVID-19          |                                                                                                             | PMID list |       |
|                        | [Lund-COVID-19_plaintext.txt](https://github.com/Aitslab/EasyNER/blob/main/data/Lund-COVID-19_plaintext.txt)| Plain text|       |
| tmVar (Version 3)      ||||
| MedMentions            ||||
| BioID                  |[goldstandard/bioid.txt](https://github.com/hu-ner/hunflair2-experiments/blob/main/annotations/goldstandard/bioid.txt)|||
| BioID                  |[hunflair2/bioid.txt](https://github.com/hu-ner/hunflair2-experiments/blob/main/annotations/hunflair2/bioid.txt)|PubTator (Hunflair2 predictions)||
| BioRED                 |[BIORED.zip](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip)                                         |||

## EasyNER
EasyNER runs included the following modules: Sentence Splitter, NER, Analysis.
Separate runs were conducted for each entity class (WHICH?), using dictionary-based or model-based NER.
For runtimes without the Sentence Splitter or Analysis module the runtime for the respective module (OBTAINED HOW?) was subtracted from the total runtime.
EasyNER output files were converted to PubTator format using the conversion script (FIX AND UPLOAD)

## Hunflair2
Hunflair2 predictions for tmVar, BioID and MedMentions were obtained from the [Hunflair repo](https://github.com/hu-ner/hunflair2-experiments/tree/main/annotations/hunflair2). WHAT WAS DONE FOR OTHER CORPORA PREDICTIONS AND RUNTIMES?

## BERN2
BERN2 local installation was attempted on both laptop and HPC system but failed with multiple errors. Instead, predictions were obtained using the [BERN2 web demo and API](http://bern2.korea.ac.kr/).

## Evaluation
Hardware: DETAILS? 

### Runtime
Timing script:

RESULT TABLE

Evaluation script: [Hunflair2 evaluation script](https://github.com/hu-ner/hunflair2-experiments/blob/main/evaluate.py)

### Performance
Metrics:

RESULT TABLE
