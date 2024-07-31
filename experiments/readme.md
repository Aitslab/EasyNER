*** UNDER CONSTRUCTION ***


All info and scripts to reproduce the benchmarking experiments for the EasyNER article are in this folder.

## 1. Datasets
We used the following datasets:

| Corpus                 | File                      | Format               |
|------------------------|---------------------------------|----------------------|
| Lund-Autophagy-1       | [Lund-Autophagy-1.txt](https://github.com/Aitslab/EasyNER/blob/main/data/Lund-Autophagy-1.txt)            | PMID list            |
| Lund-Autophagy-2       | [Lund-Autophagy-2.txt](https://github.com/Aitslab/EasyNER/blob/main/data/Lund-Autophagy-2.txt)            | PMID list            |
| Lund-COVID-19          | Lund-COVID-19                   | PMID list            |
| Lund-COVID-19          | [Lund-COVID-19_plaintext.txt](https://github.com/Aitslab/EasyNER/blob/main/data/Lund-COVID-19_plaintext.txt)     | Plain text           |
| tmVar (Version 3)      |                                 | PubTator             |
| MedMentions            |                                 | PubTator             |
| BioID                  |                                 | PubTator             |
| BioRED                 |                                 | PubTator             |

## 2. Pre-processing
We used the following scripts to convert PubTator files into document collection json files for EasyNER.

## 3. EasyNER
EasyNER runs included the following modules: Sentence Splitter, NER, Analysis.
Separate runs were conducted for each entity class (WHICH?), using dictionary-based or model-based NER.
For runtimes without the Sentence Splitter or Analysis module the runtime for the respective module (OBTAINED HOW?) was subtracted from the total runtime.

## 4. Hunflair2
Hunflair2 predictions for tmVar, BioID and MedMentions were obtained from the [Hunflair repo] (https://github.com/hu-ner/hunflair2-experiments/tree/main/annotations/hunflair2). OTHER CORPORA PREDICTIONS?

## 5. BERN2
BERN2 local installation was attempted on both laptop and HPC system but failed with multiple errors. Instead, predictions were obtained from the web demo and API.

## 6. Runtime evaluation
Tools: EasyNER, Hunflair2, BERN2 web demo, BERN2 API

Hardware: DETAILS? 

Timing script:

## 7. F1, precision and recall evaluation
Tools: EasyNER, Hunflair2, BERN2 web demo, BERN2 API

Evaluation script:
