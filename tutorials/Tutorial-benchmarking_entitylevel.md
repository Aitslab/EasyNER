This is a step by step guide to evaluating the performance of EasyNER and other BioNLP tools on corpora in PubTator format on entity-level. 

The evaluation script calculates false-positives and -negatives, single class precision, recall and F1 score, and micro, macro and weighted averages of the precision, recall and F1 score across all entity classes.

# Conversion of PubTator format to EasyNER input and prediction
1. Place corpora files in PubTator format in a folder and convert to single JSON files using the [convert_hunfliar2_pubtator-to-json.py](https://github.com/Aitslab/EasyNER/blob/main/supplementary/experiment_scripts/convert_hunflair2_pubtator_to_json.py)
