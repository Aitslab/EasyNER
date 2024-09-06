#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
This script evaluates named entity recognition predictions against a gold standard and saves the metics in a file called "metrics.tsv". It works with files in PubTator format. 

ATTENTION: The script only loads entities in the goldstandard and prediction files which have an entitity identifier. Use a dummy identifier (e.g. -1) if needed.


The script is partially based on the utils.öy and evaluate.py scripts published with the Hunflair2 paper:
https://github.com/hu-ner/hunflair2-experiments/blob/main/utils.py
https://github.com/hu-ner/hunflair2-experiments/blob/main/evaluate.py
The requirement for these scripts states that they are dependent on bioc==2.1.


Args:
    corpora: A dictionary containing the names of the corpora as keys and the and a list with the entity classes to be evaluated as values.
    tools: A list of tool names to be evaluated.
    data_folder: Folder for input and output files. It should have a subfolder for the gold standard called "goldstandard" and predictions files in PubTator format (one subfolder per tool, names should match corpora dictionary and tool list)
    offset_stride: Allowed offset for entity span matching. With a stride of 1, start and/or end of the detected span can differ by 1 between goldstandard and prediction for a true-positive.
    
    The data_folder structure should match corpora and tools, e.g. PATH/evaluation/goldstandard/medmentions.txt and evaluation/easyner/medmentions.txt

    
PubTator format:
    0|t|Variation in the CXCR1 gene (IL8RA) is not associated with susceptibility to chronic periodontitis.
    0|a|BACKGROUND: The chemokine receptor 1 CXCR-1 (or IL8R-alpha) is a specific receptor for the interleukin 8 (IL-8), which is chemoattractant for neutrophils and has an important role in the inflammatory response. The polymorphism rs2234671 at position Ex2+860G>C of the CXCR1 gene causes a conservative amino acid substitution (S276T). This single nucleotide polymorphism (SNP) seemed to be functional as it was associated with decreased lung cancer risk. Previous studies of our group found association of haplotypes in the IL8 and in the CXCR2 genes with the multifactorial disease chronic periodontitis. In this study we investigated the polymorphism rs2234671 in 395 Brazilian subjects with and without chronic periodontitis. FINDINGS: Similar distribution of the allelic and genotypic frequencies were observed between the groups (p>0.05). CONCLUSIONS: The polymorphism rs2234671 in the CXCR1 gene was not associated with the susceptibility to chronic periodontitis in the studied Brazilian population.
    0	17	22	CXCR1	Gene	NCBI Gene:3577	
    0	29	34	IL8RA	Gene	-1	

    pubmed or document id|t|title text
    pubmed or document id|a|abstract
    pubmed or document id    start of span     end of span     detected text    entity class    entity identifier or dummy value (-1)

    
"""

__author__ = 'Sonja Aits'
__copyright__ = 'Copyright (c) 2024 Sonja Aits'
__license__ = 'Apache 2.0'
__version__ = '0.1.0'



import os
import pandas as pd
from bioc import pubtator


# Function to initialize the metrics dictionary
def initialize_metrics_dictionary(tools, corpora):
    print("INITIALIZING METRICS DICTIONARY")
    metrics = {}
    print()

    # Initialize the metrics dictionary for each tool
    for tool in tools:
        metrics[tool] = {}

        # Initialize the metrics for each corpus
        for corpus in corpora:
            metrics[tool][corpus] = {
                'corpus_metrics': {},
                'class_metrics': {}
            }

            # Initialize corpus_metrics
            for entity_class in corpora[corpus]:
                metrics[tool][corpus]['corpus_metrics'] = {
                    'class_count': 0,
                    'allclass_gold_count': 0,
                    'allclass_pred_count': 0,
                    'allclass_tp': 0,
                    'allclass_fp': 0,
                    'allclass_fn': 0,
                    'macro_precision': 0,
                    'macro_recall': 0,
                    'macro_f1': 0,
                    'micro_precision': 0,
                    'micro_recall': 0,
                    'micro_f1': 0,
                    'weighted_precision': 0,
                    'weighted_recall': 0,
                    'weighted_f1': 0
                }

                # Initialize class_metrics
                metrics[tool][corpus]['class_metrics'][entity_class] = {
                    'gold_count': 0,
                    'pred_count': 0,
                    'class_weight': 0,
                    'tp': 0,
                    'fp': 0,
                    'fn': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0
                }

    print("="*100)
    return metrics




def load_pubtator(path: str, entity_classes: list[str]) -> dict:
    """
    Load annotations from PubTator into nested dict and harmonize type names:
        - text_id:
            - entity_class:
                - span:
        """

    annotations: dict = {}

    with open(path) as fp:
        documents = pubtator.load(fp)

        for d in documents:

            for a in d.annotations: # WARNING: d.annotations only includes annotations that had an identifier!

                # Remapping of entity classes to harmonize predictions and gold standard corpora
                if a.type == "molecule":
                    a.type = "chemical"
                
                if a.type == "ChemicalEntity":
                    a.type = "chemical"

                if a.type == "cellline":
                    a.type = "cell"

                if a.type == "cell_line":
                    a.type = "cell"
                
                if a.type == "CellLine":
                    a.type = "cell"

                if a.type == "protein":
                    a.type = "gene"

                if a.type == "GeneOrGeneProduct":
                    a.type = "gene"

                if a.type == "organism":
                    a.type = "species"

                if a.type == "OrganismTaxon":
                    a.type = "species"
                
                if a.type == "DiseaseOrPhenotypicFeature":
                    a.type = "disease"

                entity_class = a.type.lower()

                if entity_class not in entity_classes:
                    continue

                if entity_class not in annotations:
                    annotations[entity_class] = {}

                if a.pmid not in annotations[entity_class]:
                    annotations[entity_class][a.pmid] = {}

                annotations[entity_class][a.pmid][(a.start, a.end)] = {"text": a.text}

    return annotations




# Function to convert the gold standard into a dictionary for evaluation
def load_gold(corpora, gold_dir):
    print("LOADING GOLDSTANDARDS")
    print()

    gold = {}
    for corpus, entity_types in corpora.items():
        path = os.path.join(gold_dir, f"{corpus}.txt")
        print(f"Loading {corpus} goldstandard: {path}")
        print(f"Evaluated classes: {entity_types}")

        gold[corpus] = load_pubtato
        r(path, entity_types)

        print("-"*100)

    print(gold)
    print("="*100)
    return gold




# Function to convert the predictions into a nested dictionary for evaluation
def load_preds(tools, corpora, dir):
    print("LOADING PREDICTIONS")
    print()

    preds = {}
    for tool in tools:
        tool_dir = os.path.join(dir, tool)
        preds[tool] = {}

        if tool == "bern2":
            load_fn = load_bern # FUNCTION TO BE ADDED
        else:
            load_fn = load_pubtator

        for corpus, entity_class in corpora.items():
            path = os.path.join(tool_dir, f"{corpus}.txt")
            print(f"Loading {tool} {corpus} predictions: {path}")

            if not os.path.exists(path):
                print(f"WARNING: {tool} {corpus} predictions not found")
                print("-"*100)
                continue

            preds[tool][corpus] = load_fn(path, entity_class)

            print("-"*100)

    print(preds)
    print("="*100)
    return preds




#Function to calculate offsets for the spans
def get_offsets(start, end, offset_stride):
    for s, e in [
        (start, end),
        (start + offset_stride, end),
        (start - offset_stride, end),
        (start, end + offset_stride),
        (start, end - offset_stride),
        (start + offset_stride, end + offset_stride),
        (start - offset_stride, end - offset_stride),
        (start + offset_stride, end - offset_stride),
        (start - offset_stride, end + offset_stride)
    ]:
        yield(s, e)




# Function to calculate gold standard corpus statistics
def calculate_gold_stats(gold, preds, metrics):
    print("CALCULATING GOLD STANDARD COUNTS")
    print()

    for corpus, corpus_anno in gold.items():

        for entity_class, class_anno in corpus_anno.items():

            entity_count = 0

            for text_id, entities in class_anno.items():
                entity_count = entity_count + len(entities)

            for tool, pred in preds.items():
                metrics[tool][corpus]['class_metrics'][entity_class]['gold_count'] = entity_count

    print(metrics)
    print("="*100)
    return metrics




# Function to calculate prediction statistics
def calculate_pred_stats(preds, metrics):
    print("CALCULATING PREDICTION COUNTS")
    print()

    for tool, pred in preds.items():

        for corpus, corpus_anno in pred.items():

            for entity_class, class_anno in corpus_anno.items():

                entity_count = 0

                for text_id, entities in class_anno.items():
                    entity_count = entity_count + len(entities)

                    metrics[tool][corpus]['class_metrics'][entity_class]['pred_count'] = entity_count

    print(metrics)
    print("="*100)
    return metrics




# Function to calculate the class-based metrics TP, FP, precision, recall and f1
def calculate_class_metrics(gold, preds, offset_stride, metrics):
    print("CALCULATING CLASS-BASED METRICS")
    print()

    for tool, pred in preds.items():

        # Calculate tp counts
        print(f"Calculating True-positive (TP) counts for tool: {tool}")
        print('-'*100)
        for corpus, corpus_anno in pred.items():

            for entity_class, class_anno in corpus_anno.items():

                tp = 0 # Counter for true positives

                for text_id, entities in class_anno.items():

                    for (start, end), text in entities.items():

                        # Calculate the allowed spans including offset for each span
                        for s, e in get_offsets(start=start, end=end, offset_stride=offset_stride):

                            # Compare predicted entities with gold standard
                            if (s, e) in gold.get(corpus, {}).get(entity_class, {}).get(text_id, {}):
                                tp += 1
                                break # break if a (s, e) pair has a match so that it is not counted multiple times

                print(f"Updating tp metrics for tool: {tool}, corpus: {corpus}, entity_class: {entity_class}")
                metrics[tool][corpus]['class_metrics'][entity_class]['tp'] = tp
                print(f"Updated TP metrics: {metrics[tool][corpus]['class_metrics'][entity_class]}")

                print("-"*100)


        # Calculate fn counts
        print(f"Calculating True-positive (FN) counts for tool: {tool}")
        print('-'*100)
        for corpus, gold_corpus_anno in gold.items():

            for entity_class, class_anno in gold_corpus_anno.items():

                fn = 0 # Counter for false negatives

                for text_id, entities in class_anno.items():

                    for (start, end), text in entities.items():
                        found = False

                        # Calculate the allowed spans including offset for each span
                        for s, e in get_offsets(start=start, end=end, offset_stride=offset_stride):

                            # Compare predicted entities with predictions
                            if (s, e) in pred.get(corpus, {}).get(entity_class, {}).get(text_id, {}):
                                found = True
                                break

                        if found == False:
                            fn += 1

                print(f"Updating FN metrics for tool: {tool}, corpus: {corpus}, entity_class: {entity_class}")
                metrics[tool][corpus]['class_metrics'][entity_class]['fn'] = fn
                print(f"Updated FN metrics: {metrics[tool][corpus]['class_metrics'][entity_class]}")

                print("-"*100)


    # Calculate fp counts
    for tool, corpora in metrics.items():
        print(f"Calculating True-positive (FP) counts for tool: {tool}")
        print('-'*100)
        for corpus, metric_types in corpora.items():
            for entity_class, entry in metric_types['class_metrics'].items():
                print(f"Updating FP metrics for tool: {tool}, corpus: {corpus}, entity_class: {entity_class}")
                entry['fp'] = entry['pred_count'] - entry['tp']
                print(f"Updated FP metrics: {metrics[tool][corpus]['class_metrics'][entity_class]}")

                print("-"*100)


    # calculate precision, recall and f1 score for each class
    for tool, corpora in metrics.items():
        print(f"Calculating prec, rec, f1 for tool: {tool}")
        print('-'*100)
        for corpus, metric_types in corpora.items():
            for entity_class, entry in metric_types['class_metrics'].items():
                print(f"Updating prec, rec, f1 metrics for tool: {tool}, corpus: {corpus}, entity_class: {entity_class}")

                # calculate precision
                if entry['pred_count'] > 0:
                    entry['precision'] = entry['tp'] / entry['pred_count']
                else:
                    entry['precision'] = 0

                # calculate recall
                if entry['gold_count'] > 0:
                    entry['recall'] = entry['tp'] / entry['gold_count']
                else:
                    entry['recall'] = 0

                # calculate f1 score
                if entry['precision'] + entry['recall'] > 0:
                    entry['f1'] = 2 * entry['precision'] * entry['recall'] / (entry['precision'] + entry['recall'])
                else:
                    entry['f1'] = 0

                print(f"Updated prec, rec, f1 metrics: {metrics[tool][corpus]['class_metrics'][entity_class]}")

                print("-"*100)

    print(metrics)
    print("="*100)
    return metrics




# Function to calculate the corpus-based metrics and some class-based metics
def calculate_corpus_metrics(metrics):
    print("CALCULATING CORPUS-BASED METRICS")
    print()

    for tool, corpora in metrics.items():
        for corpus, metric_types in corpora.items():

            # Calculate class count
            class_count = len(metric_types['class_metrics'])
            metric_types['corpus_metrics']['class_count'] = class_count

            # Calculate macro metrics
            summed_gold = 0
            summed_pred = 0

            summed_tp = 0
            summed_fp = 0
            summed_fn = 0

            summed_f1 = 0
            summed_prec = 0
            summed_rec = 0

            for entity_class, entry in metric_types['class_metrics'].items():
                summed_gold += entry['gold_count']
                summed_pred += entry['pred_count']
                summed_tp += entry['tp']
                summed_fp += entry['fp']
                summed_fn += entry['fn']
                summed_prec += entry['precision']
                summed_rec += entry['recall']
                summed_f1 += entry['f1']

            # calculate macro metrics by averaging class_based metrics
            metric_types['corpus_metrics']['macro_precision'] = summed_prec / class_count
            metric_types['corpus_metrics']['macro_recall'] = summed_rec / class_count
            metric_types['corpus_metrics']['macro_f1'] = summed_f1 / class_count

            # calculate summed counts
            metric_types['corpus_metrics']['allclass_gold_count'] = summed_gold
            metric_types['corpus_metrics']['allclass_pred_count'] = summed_pred
            metric_types['corpus_metrics']['allclass_tp'] = summed_tp
            metric_types['corpus_metrics']['allclass_fp'] = summed_fp
            metric_types['corpus_metrics']['allclass_fn'] = summed_fn

            # calculate micro metrics from summed counts
            if summed_pred > 0:
                metric_types['corpus_metrics']['micro_precision'] = summed_tp / summed_pred
            else:
                metric_types['corpus_metrics']['micro_precision']  = 0

            if summed_gold > 0:
                metric_types['corpus_metrics']['micro_recall'] = summed_tp / summed_gold
            else:
                metric_types['corpus_metrics']['micro_recall'] = 0

            if metric_types['corpus_metrics']['micro_precision'] + metric_types['corpus_metrics']['micro_recall'] > 0:
                micro_f1 = 2 * metric_types['corpus_metrics']['micro_precision'] * metric_types['corpus_metrics']['micro_recall'] / (metric_types['corpus_metrics']['micro_precision'] + metric_types['corpus_metrics']['micro_recall'])
                metric_types['corpus_metrics']['micro_f1'] = micro_f1
            else:
                metric_types['corpus_metrics']['micro_f1'] = 0

            # calculate class weights and weighted metrics
            weighted_prec = 0
            weighted_rec = 0
            weighted_f1 = 0

            for entity_class, entry in metric_types['class_metrics'].items():
                entry['class_weight'] = entry['gold_count'] / metric_types['corpus_metrics']['allclass_gold_count']

                weighted_prec += entry['class_weight'] * entry['precision']
                weighted_rec += entry['class_weight'] * entry['recall']
                weighted_f1 += entry['class_weight'] * entry['f1']

            metric_types['corpus_metrics']['weighted_precision'] = weighted_prec
            metric_types['corpus_metrics']['weighted_recall'] = weighted_rec
            metric_types['corpus_metrics']['weighted_f1'] = weighted_f1

    print(metrics)
    print("="*100)
    return metrics




# Function to flatten the nested dictionary for dataframe
def flatten_metrics(metrics):
    flattened_data = []
    for tool, tool_data in metrics.items():
        for corpus, corpus_data in tool_data.items():
            # Corpus metrics
            corpus_metrics = corpus_data['corpus_metrics']
            corpus_metrics['tool'] = tool
            corpus_metrics['corpus'] = corpus
            flattened_data.append(corpus_metrics)

            # Class metrics
            for entity_class, class_metrics in corpus_data['class_metrics'].items():
                class_metrics['tool'] = tool
                class_metrics['corpus'] = corpus
                class_metrics['entity_class'] = entity_class
                flattened_data.append(class_metrics)
    return flattened_data




# Function to run the evaluation
def main(corpora, tools, data_folder, offset_stride):

    # print warning
    print('ATTENTION: The script only evaluates entities which have identifiers in the gold standard and prediction files. Use a dummy identifier (e.g. -1) if needed.')
    print("="*100)

    
    # Initialize the metrics dictionary
    metrics = initialize_metrics_dictionary(tools=tools, corpora=corpora)

    # Load gold standard annotations for all corpora
    gold_dir = os.path.join(os.getcwd(), data_folder, "goldstandard")
    gold = load_gold(corpora, gold_dir)

    # Load predictions from each of the tools for all corpora
    preds = load_preds(tools, corpora, data_folder)

    # Calculate gold standard statistics
    gold_stats = calculate_gold_stats(gold, preds, metrics)

    # Calculate prediction statistics
    pred_stats = calculate_pred_stats(preds, metrics)

    # Calculate the performance of each tool on each corpus for each entity class
    class_metrics = calculate_class_metrics(gold, preds, offset_stride, metrics)

    # Calculate the performance of each tool on each corpus across all entity classes
    corpus_metrics = calculate_corpus_metrics(metrics)

    # Convert to dataframe and save
    flattened_metrics = flatten_metrics(corpus_metrics)
    df = pd.DataFrame(flattened_metrics)
    df.to_csv(f'{data_folder}/metrics.tsv', sep='\t', index=False, encoding = 'utf-8')

    print("="*100)
    print("="*100)
    print('FINISHED')
    print("="*100)





if __name__ == "__main__":

    # Example usage

    # Define name of the folder containing predictions and goldstandard subfolders
    data_folder = C:/Users/sonja/python_runs/evaluation/  

   # Define corpora and entity classes to be evaluated (corpora names should match subfolder names)
    corpora = {
        "medmentions": ["disease", "chemical"],
        "medmentions_original_preprocessed": ["disease", "chemical", "gene", "species", "cell"],
        "tmvar_v3": ["gene", "species", "cell"],
        "tmvar_v3_preprocessed": ["gene", "species", "cell"],
        "bioid": ["gene", "chemical", "cell", "species"],
        "bioid_preprocessed": ["gene", "chemical", "cell", "species"]
    }

    # Define NER prediction tools to be evaluated (should match subfolder names)
    tools = ["hunflair2", "pubtator", "bent", "goldstandard"]
    
    # Run evaluation
    main(corpora=corpora, tools=tools, data_folder=data_folder, offset_stride=1)
