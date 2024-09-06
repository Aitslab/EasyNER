#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


"""
    Postprocesses all EasyNER NER file in a folder by merging entities separated with hyphens or brackets.
    The output files have the suffix _mergedentities but are otherwise the same as the input files.

    Args:
      input_folder: Path to the input folder containing files in EasyNER json format with annotated entities
"""

__author__ = 'Sonja Aits'
__copyright__ = 'Copyright (c) 2024 Sonja Aits'
__license__ = 'Apache 2.0'
__version__ = '0.1.0'


import os
import json
import re

# Function to extend entities with an unmatched hyphen by merging them with missing parts of the word
def merge_entities(entities, entity_spans, text):

    separators = ['-', '(', ')', '[', ']', '{', '}']
  
    # Loop through the entities and perform merges of entities bordering on separators
    i = len(entities) - 1 # Last index
    while i >= 0:
        entity = entities[i]

        # Merge with preceding entity if entity starts with a separator
        for separator in separators:
            if entity.startswith(separator):

                # Left extension form entity list: look for preceeding entity without gap in the list and merge
                if i > 0 and entity_spans[i-1][1] == entity_spans[i][0]:

                    entities[i-1] = entities[i-1] + entity
                    entity_spans[i-1] = [entity_spans[i-1][0], entity_spans[i][1]]
                    print(f"Merged entity method 1: {entities[i-1]}")
                    print(f"Merged entity method 1 span: {entity_spans[i-1]}")

                    # Remove the merged entity unless it ends with a separator
                    if entity.endswith(separator) == False:
                        del entities[i]
                        del entity_spans[i]

                # Left extension from sentence text: move start of span left until encountering whitespace or punctuation
                else:

                    start = entity_spans[i][0]
                    while start > 0 and not re.match(r"[\s.,:;\n]", text[start-1]):
                        start -= 1
                    entities[i] = text[start:entity_spans[i][1]]
                    entity_spans[i] = [start, entity_spans[i][1]]
                    print(f"Merged entity method 2: {entities[i]}")
                    print(f"Merged entity method 2 span: {entity_spans[i]}")


            # Merge with suceeding entity if entity ends with a separator
            if entity.endswith(separator):

                # Right extension form entity list: look for suceeding entity without gap in the list and merge
                if i + 1 < len(entities) and entity_spans[i][1] == entity_spans[i+1][0]:

                    entities[i] = entities[i] + entities[i+1]
                    entity_spans[i] = [entity_spans[i][0], entity_spans[i+1][1]]
                    print(f"Merged entity method 3: {entities[i]}")
                    print(f"Merged entity method 3 span: {entity_spans[i]}")
                    del entities[i+1]
                    del entity_spans[i+1]

                # Right extension from sentence text: move end of span right until encountering whitespace or punctuation
                else:

                    end = entity_spans[i][1]
                    while end < len(text) and not re.match(r"[\s.,:;\n]", text[end]):
                        end += 1

                    entities[i] = text[entity_spans[i][0]:end]
                    entity_spans[i] = [entity_spans[i][0], end]
                    print(f"Merged entity method 4: {entities[i]}")
                    print(f"Merged entity method 4 span: {entity_spans[i]}")

        i -= 1 # Move index to preceeding entity for next round


    # Merge entities which are only separated by a separator in the text
    i = len(entities) - 1 # Last index
    while i >= 1:
        end_preceeding = entity_spans[i-1][1]
        if ((entity_spans[i][0])-1) == end_preceeding: # For entities with one step between them
            for separator in separators:
                if text[end_preceeding] == separator:
                    entities[i-1] = entities[i-1] + separator  + entities[i]
                    entity_spans[i-1] = [entity_spans[i-1][0], entity_spans[i][1]]
                    print(f"Merged entity method 5: {entities[i-1]}")
                    print(f"Merged entity method 5 span: {entity_spans[i-1]}")
                    del entities[i]
                    del entity_spans[i]

        i -= 1 # Move index to preceeding entity for next round


    # Clean entities and entity_spans lists by removing those where the span is identical with/part of another span
    i = len(entity_spans) - 1 # Last index
    while i > 0:
        span = entity_spans[i]
        for j in range(0,(len(entity_spans))):
            if j !=i:
                if entity_spans[i][0] >= entity_spans[j][0] and entity_spans[i][1] <= entity_spans[j][1]:
                    del entities[i]
                    del entity_spans[i]
                    break
        i -= 1

    return entities, entity_spans


# Function to load text from a json file and update entities before saving to a new file
def postprocess_ner_entities(input_folder):

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(input_folder, f"{os.path.splitext(filename)[0]}_separatormerged.json")
            print(f"Processing file: {input_file}")

            # Load the JSON file
            with open(input_file, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Process each sentence in each document
            for doc_id, doc_data in data.items():
                print(f"Processing document: {doc_id}")
                for sentence in doc_data.get('sentences', []):
                    text = sentence['text']
                    entities = sentence['entities']
                    entity_spans = sentence['entity_spans']

                    # Merge entities as required
                    if entities and entity_spans:
                        sentence['entities'], sentence['entity_spans'] = merge_entities(entities, entity_spans, text)


            # Save to the output file
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=2)


# Test cases for merge_entities
# text = 'ab(de fg]hi jk{lm no-pq rs-tu, vw x-y-z a-b-c. abc d-e-f; gh-ij-kl'
# entities = ['ab','(de', ']hi', 'jk{', 'lm', 'no-', '-', 'vw', 'x-', '-z', 'a-', 'abc', '-e-', 'gh', 'ij-']
# entity_spans = [[0,2], [2, 5], [8, 11], [12, 15], [15, 17], [18, 21], [26, 27], [31, 33], [34, 36], [37, 39], [40, 42], [46, 49], [52, 55], [58, 60], [61, 64]]
# print(merge_entities(entities, entity_spans, text))
# print('='*100)

# text2 = 'Co-immunoprecipitation analysis and glutathione-S-transferase (GST) pull down assay were conducted to analyze the association between EZH2 and H2BY37ph .'
# entities2 = ["glutathione","-"]
# entity_spans2 = [[36,47],[47,48]]
# print(merge_entities(entities2, entity_spans2, text2))
# print('='*100)



# Example usage
input_folder = 'C:/Users/sonja/python_runs/easyner_predictions/'  # Replace with your actual input folder path
postprocess_ner_entities(input_folder)

