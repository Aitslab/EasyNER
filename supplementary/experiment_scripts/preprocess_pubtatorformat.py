#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

    """
    Preprocesses a PubTator file by remapping entity classes based on a tsv file (optional), adding dummy identifiers and writing the preprocessed corpus to a new file.
    The output file has the suffix _preprocessed is otherwise the same as the input PubTator file.

    Args:
      pubtator_file: Path to the input file in PubTator format
      remap: 'yes' to perform entity class remapping, 'no' to skip class remapping and only add identifiers. 
      mapping_file: A tsv file mapping old entity classes (keys, first column) to new ones (values, second column). Required if remap is 'yes'.
    """

__author__ = 'Sonja Aits'
__copyright__ = 'Copyright (c) 2024 Sonja Aits'
__license__ = 'Apache 2.0'
__version__ = '0.1.0'


import os
import csv
from re import split

# Function to create a mapping dictionary from a tsv file (file format: one line as header, column A = old class name, column B = new class name)
def create_mapping_dict(mapping_file):
    mapping = {}
    with open(mapping_file, 'r', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        next(tsv_reader) # Skip the header line

        for key, value in tsv_reader:
            mapping[key] = value
    print(mapping)
    return mapping



# Function to add dummy identifier and remap the entity classes to new names using a mapping dictionary
def preprocess_pubtator(pubtator_file, remap='yes', mapping_file=None):

    # Create the mapping dictionary if remap is 'yes'
    if remap == 'yes':
        if mapping_file is None:
            raise ValueError("mapping_file must be provided when remap is set to 'yes'")
        mapping = create_mapping_dict(mapping_file)
    else:
        mapping = None
     
    
    # Construct the output file name with the _preprocessed suffix
    base_name, ext = os.path.splitext(pubtator_file)
    output_file = f"{base_name}_preprocessed{ext}"

    with open(pubtator_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:

        # Write text lines to file
            split_line = line.strip().split('\t') # this splits the annotations into a list of 5 (without identifier) or 6 elements (with identifier)

            if len(split_line) < 5: # True for the text sections in the beginning
                f_out.write(line)

            else:
                # Add identifier if missing
                if len(split_line) == 5:
                    split_line.append('-1')

                # Remap entity classes
                if remap == 'yes' and split_line[4].strip() in mapping:
                    split_line[4] = mapping[split_line[4]]

                f_out.write('\t'.join(split_line) + '\n')


# Example usage
# mapping_file = '/content/drive/MyDrive/evaluation/original_goldstandard/medmentions_remapping_sonja20240812.tsv'
# pubtator_file = '/content/drive/MyDrive/evaluation/original_goldstandard/medmentions.txt'
# preprocess(pubtator_file=pubtator_file, remap = 'yes', mapping_file=mapping_file)
