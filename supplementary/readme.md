*** UNDER CONSTRUCTION ***


All info and scripts to reproduce the benchmarking experiments for the EasyNER article are in this folder.

## Datasets and preprocessing scripts

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

## Script descriptions

1. convert_hunflair2_pubtator_to_json.py: Convert corpora in raw pubtator format into JSON format. For example, in case of corpora used for HunFlair2 runs like medmentions, bioID, TmVar_v3 etc
2. convert_easyner_raw_json_to_pubtator.py: Convert raw JSON files into Pubtator format.
3. convert_easyner_output_json_to_pubtator.py: Convert output from EasyNER in JSON format into Pubtator format.
4. entity_merger2.py: Merge entities in json formats from multiple input folders with same JSON file but different annotations.
5. get_runtime_for_all_timestamps.ipynb: Get a python dataframe with runtimes from different models. Can be used to both input times manually or from EasyNER output format. This script was used to combine all timestamps and compare different models.
6. evaluate_ner_pubtatorformat.py: run evaluation on annotations in the data folder that are presented in Pubtator format. Each folder name should represent the method and each filename within should contain annotation from that corpus.
7. count_characters_in_each_corpus.ipynb: Count characters, words and articles in each corpus
8. postprocess_separator_merging.py: Post process separators and punctuation marks withins annotated file. Is used to epand EasyNER output from a single detected character, such as "-" into the nearest whole word.
9. preprocess_pubtatorformat.py: preprocess pubtator formats to make sure all annotations contain -1 in the end and are detectable with the evaluation script.
10. remove_NEL_from_biored.ipynb: Remove NEL tags at the end of each annotation for the BioRED corpus.


