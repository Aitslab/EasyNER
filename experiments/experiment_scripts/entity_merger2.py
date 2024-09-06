#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


"""
    Merges NER files created from different EasyNER NER runs with different models (different entity classes). 

    Args:
      input_folders: List of paths to input folders in json format, one per entity class/EasyNER NER run. The folders should contain the output files of the EasyNER NER module, with a numerical suffix in the name so that corresponding files can be identified.
      entity_classes: List of classes (str) to be used to populate "entity_classes". Order should correspond to input_folders.
      output_folder: Path to folder where merged files are to be saved.
      resolve_conflicts: True if conflicts between annotations should be resolved, leaving only one annotation for each span. False if all annotations should be kept.
"""

__author__ = 'Sonja Aits'
__copyright__ = 'Copyright (c) 2024 Sonja Aits'
__license__ = 'Apache 2.0'
__version__ = '0.1.0'


import os
import json
import re
from collections import defaultdict

# Function to group corresponding files which are to be merged
def group_files(input_folders, entity_classes):

    merged_files = {}
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Helper function to extract the numerical suffix from filenames
    def get_suffix(filename):
        match = re.search(r'-(\d+)\.json$', filename)
        return match.group(1) if match else None

    # Dictionary to collect files by suffix
    files_by_suffix = defaultdict(list)

    # Collect files by suffix
    for folder, entity_class in zip(input_folders, entity_classes):
        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                suffix = get_suffix(filename)
                if suffix:
                    files_by_suffix[suffix].append((os.path.join(folder, filename), entity_class))


    return files_by_suffix


# Function to create a dictionary containing the content of all files in a merge group
def create_merged_dictionary(files):
    merged_dict = {}

    for file_path, entity_class in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            # Iterate over each article in the file
            for article_id, content in data.items():
                if article_id not in merged_dict:
                    # Initialize the structure for each new article
                    merged_dict[article_id] = {
                        "title": content["title"],
                        "sentences": []}

                # Merge entities, entity_spans, and entity_classes for each sentence
                for i, sentence in enumerate(content["sentences"]):
                    if i >= len(merged_dict[article_id]["sentences"]):
                        # Initialize the sentence if it doesn't exist
                        merged_dict[article_id]["sentences"].append({
                            "text": sentence["text"],
                            "entities": [],
                            "entity_spans": [],
                            "entity_classes": []})

                    merged_sentence = merged_dict[article_id]["sentences"][i]

                    # Extend the lists
                    merged_sentence["entities"].extend(sentence["entities"])
                    merged_sentence["entity_spans"].extend(sentence["entity_spans"])
                    merged_sentence["entity_classes"].extend([entity_class] * len(sentence["entities"]))

                    # Sorting the lists based on entity_spans if there are entities
                    if merged_sentence["entity_spans"]:
                        sorted_entities = sorted(
                                zip(merged_sentence["entities"], merged_sentence["entity_spans"], merged_sentence["entity_classes"]),
                                key=lambda x: (x[1][0], x[1][1]))  # Sort by start (first element of span), then by end (second element of span)

                        # Unzip the sorted entities back into their respective lists
                        merged_sentence["entities"], merged_sentence["entity_spans"], merged_sentence["entity_classes"] = map(list, zip(*sorted_entities))

    return merged_dict


# Function to remove conflicting entities
def remove_conflicting_entities(merged_dict):

    for article_id, content in merged_dict.items():
        for index, sentence in enumerate(content["sentences"]):
            merged_sentence = merged_dict[article_id]["sentences"][index]


            i = len(merged_sentence["entity_spans"]) - 1  # Start from the last index

            while i >= 0:

                for j in range(0,len(merged_sentence["entity_spans"])):
                    start_i, end_i = merged_sentence["entity_spans"][i]
                    start_j, end_j = merged_sentence["entity_spans"][j]

                    # Remove entity with lower priority for identical spans
                    if merged_sentence["entity_spans"][i] == merged_sentence["entity_spans"][j] and merged_sentence["entity_classes"][i] != merged_sentence["entity_classes"][j]:
                        prioritization = {"chemical": 1, "disease": 2, "species": 3, "gene": 4, "cell": 5}
                        rank_i = prioritization[merged_sentence["entity_classes"][i]]
                        rank_j = prioritization[merged_sentence["entity_classes"][j]]

                        if rank_i > rank_j:
                            del merged_sentence["entities"][i]
                            del merged_sentence["entity_spans"][i]
                            del merged_sentence["entity_classes"][i]
                            break

                    # Remove entity fully contained within another entity
                    elif merged_sentence["entity_spans"][i] != merged_sentence["entity_spans"][j] and start_i >= start_j and end_i <= end_j:
                        del merged_sentence["entities"][i]
                        del merged_sentence["entity_spans"][i]
                        del merged_sentence["entity_classes"][i]
                        break

                    # Remove shorter entity of two overlapping entities
                    elif (start_i < start_j and end_i > start_j and end_i < end_j) or (start_i > start_j and start_i < end_j and end_i > end_j):
                        if len(merged_sentence["entities"][i]) < len(merged_sentence["entities"][j]):
                            del merged_sentence["entities"][i]
                            del merged_sentence["entity_spans"][i]
                            del merged_sentence["entity_classes"][i]
                            break

                if i> 0:
                    i -= 1
                else:
                    break

    return merged_dict


# Function to save a merged dictionary to a new json file
def save_dict(merged_dict, entity_classes, suffix, output_folder, files):

    # Create output filename
    base_name = re.sub(r'_[^_]+-\d+\.json$', '', os.path.basename(files[0][0]))
    classes = "_".join(entity_classes)
    new_filename = f"{base_name}_{classes}-{suffix}.json"
    output_path = os.path.join(output_folder, new_filename)

    # Save merged dictionary to a new JSON file
    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(merged_dict, out_file, ensure_ascii=False, indent=2)


# Function to merge all files and remove conflicts
def merge_json_files(input_folders, entity_classes, output_folder, resolve_conflicts):
    files_by_suffix = group_files(input_folders, entity_classes)
    
    for suffix, files in files_by_suffix.items():
        merged_dict = create_merged_dictionary(files)
        print(merged_dict)
        
        if resolve_conflicts == True:
            merged_dict = remove_conflicting_entities(merged_dict)

        save_dict(merged_dict, entity_classes, suffix, output_folder, files)


# Example usage:
input_folders = ["C:/Users/sonja/python_runs/easyner_predictions/ner_bioid_cell",
                 "C:/Users/sonja/python_runs/easyner_predictions/ner_bioid_chemical",
                 "C:/Users/sonja/python_runs/easyner_predictions/ner_bioid_gene",
                 "C:/Users/sonja/python_runs/easyner_predictions/ner_bioid_species"]
entity_classes = ['cell', 'chemical', 'gene', 'species']
output_folder = 'C:/Users/sonja/python_runs/easyner_predictions/multiclass'
merge_json_files(input_folders, entity_classes, output_folder, resolve_conflicts = True)
