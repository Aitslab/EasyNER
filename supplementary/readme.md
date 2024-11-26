*** UNDER CONSTRUCTION ***

This folder contains
1. all the supplementay files for the EasyNER manuscript
2. all info and scripts to reproduce the experiments for the EasyNER article

## Acquired and produced corpora
| Corpus                                                                                                                                     | Format      | Preprocessing | 
|--------------------------------------------------------------------------------------------------------------------------------------------|-------------|-------|
|[BC5CDR_Disease](http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/NERdata.zip)                                                   | IOB2        |       |
|[Bio-ID](https://github.com/hu-ner/hunflair2-experiments/blob/main/annotations/goldstandard/bioid.txt)                                      |PubTator             |       |
|[Bio-ID Hunflair2 predictions](https://github.com/hu-ner/hunflair2-experiments/blob/main/annotations/hunflair2/bioid.txt)                   |PubTator     |       |
|[BIORED.zip](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip)                                                                         |             |       |
|[CRAFT (Version 4.0.0)](https://github.com/UCDenver-ccp/CRAFT/releases/tag/v4.0.0)                                                         | PubAnnotation^ | yes |
|[HUNER](https://github.com/hu-ner/huner/tree/master/ner_scripts)                                                                            | IOB2        |               |
|[Lund-COVID-19 (plain text)](https://github.com/Aitslab/corona/blob/master/manuscript_v2/Supplemental_file4.csv)                            | IOB2        | yes |
|[Lund-COVID-19 (plain text)](https://github.com/Aitslab/EasyNER/blob/main/data/Lund-COVID-19_plaintext.txt)                                 | Plain text  | yes |
|Simplified Lund-COVID-19 (produced by processing Lund-COVID-19)                                                                             | IOB2        |       |
|[MedMentions](https://github.com/chanzuckerberg/MedMentions)                                                                                | PubTator    | yes      |
|[MedMentions Hunflair2 predictions](https://github.com/hu-ner/hunflair2-experiments/blob/main/annotations/hunflair2/medmentions.txt)        | PubTator    |       |
|Simplified MedMentions                                                                                                                      | PubTator    |       |
|[OSIRIS](https://github.com/Rostlab/nala/tree/develop/resources/corpora/osiris)                                                             |             |       | 
|[tmVar (Version 3.0)](https://github.com/hu-ner/hunflair2-experiments/blob/main/annotations/goldstandard/tmvar_v3.txt)                      |PubTator             |       |
|[tmVar (Version 3.0) Hunflair2 predictions](https://github.com/hu-ner/hunflair2-experiments/blob/main/annotations/hunflair2/tmvar_v3.txt)   |PubTator     |       |

^ Follow these instructions to obtain the CRAFT corpus in PubAnnotation format: https://github.com/UCDenver-ccp/CRAFT/wiki/Alternative-annotation-file-formats

  
## Corpus preprocessing
- **process_GS.py**: converts the Lund-COVID-19 corpus to the Simplified Lund-COVID-19 corpus. The script merges the classes "Species_human", "Species_other", "Virus_family", "Virus_other", "Virus_SARS-CoV-2" into "species" and "Disease_COVID_19" and "Disease_other" into "disease", and removes the classes “chemicals” and “cells”.
- **CRAFT_preprocessing_spacy.py**: converts the CRAFT corpus from PubAnnotation to IOB2 format
- **preprocess_pubtatorformat.py**: ensures all annotation in a Pubtator file contain -1 in the end and are detectable with the evaluation script; also remaps classes for MedMention corpus using the mapping file.
- **BioBERT process.sh**: converts an IOB2 file into chunks alligned with the max_len parameter (default 128). If the sentence length is too long, the preprocess.sh file will cut the sentence into these chunks in the size of max_len.

1. convert_hunflair2_pubtator_to_json.py: Convert corpora in raw pubtator format into JSON format. For example, in case of corpora used for HunFlair2 runs like medmentions, bioID, TmVar_v3 etc
2. convert_easyner_raw_json_to_pubtator.py: Convert raw JSON files into Pubtator format.
3. convert_easyner_output_json_to_pubtator.py: Convert output from EasyNER in JSON format into Pubtator format.
4. entity_merger2.py: Merge entities in json formats from multiple input folders with same JSON file but different annotations.
5. get_runtime_for_all_timestamps.ipynb: Get a python dataframe with runtimes from different models. Can be used to both input times manually or from EasyNER output format. This script was used to combine all timestamps and compare different models.
6. evaluate_ner_pubtatorformat.py: run evaluation on annotations in the data folder that are presented in Pubtator format. Each folder name should represent the method and each filename within should contain annotation from that corpus.
7. count_characters_in_each_corpus.ipynb: Count characters, words and articles in each corpus
8. postprocess_separator_merging.py: Post process separators and punctuation marks withins annotated file. Is used to epand EasyNER output from a single detected character, such as "-" into the nearest whole word.

10. remove_NEL_from_biored.ipynb: Remove NEL tags at the end of each annotation for the BioRED corpus.

11. 
## EasyNER
EasyNER runs included the following modules: Sentence Splitter, NER, Analysis.
Separate runs were conducted for each entity class (WHICH?), using dictionary-based or model-based NER.
For runtimes without the Sentence Splitter or Analysis module the runtime for the respective module (OBTAINED HOW?) was subtracted from the total runtime.
EasyNER output files were converted to PubTator format using the conversion script (FIX AND UPLOAD) so that they could be evaluated using the Hunflair2 evaluation script.

## Hunflair2
Hunflair2 predictions for tmVar, BioID and MedMentions were obtained from the [Hunflair repo](https://github.com/hu-ner/hunflair2-experiments/tree/main/annotations/hunflair2). WHAT WAS DONE FOR OTHER CORPORA PREDICTIONS AND RUNTIMES?

## BERN2
BERN2 local installation was attempted on both laptop and HPC system but failed with multiple errors [Github issue](https://github.com/dmis-lab/BERN2/issues/70). Instead, predictions were obtained using the [BERN2 web demo and API](http://bern2.korea.ac.kr/).

## Evaluation
Hardware: The runs and evaluation were done on an ASUS TUF A15 Gaming Laptop with Ubuntu 23.10 OS and AMD Ryzen 7 7735HS with Radeon Graphics X 16 and NVIDIA GeForce RTX4060 dedicated GPU. The runs were conducted on the latter.

### Runtime
Timing script: get_runtime_for_all_timestamps.ipynb

RESULT TABLE

Evaluation script: evaluate_ner_pubtatorformat.py which was modified from [Hunflair2 evaluation script](https://github.com/hu-ner/hunflair2-experiments/blob/main/evaluate.py)

### Performance
Metrics:

RESULT TABLE




