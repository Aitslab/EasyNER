# encoding = utf-8
# author: Sonja Aits

"""
This script processes output files from the ner module and adds names and identifiers from a lookup table (tsv file) for each entity.
It creates a corresponding json file with this information added as output file. The prefix "nel" is added to the file name for the output file.
The lookup tsv file should have three columns: entity (the text found in the sentence), id and name (standard name to be assigned)
If the entity does not exist in the lookup table, an identifier is constructed in the format easyner:number (with a consecutive number) and the entity text is used as name. 
This easyner identifier and name is then added to the lookup dictionary so that repeated occurrences receive the same easyner:number. 
The missing entities and their easyner identifiers and names are saved in the lookup table "missing_entities.tsv"
"""


import os
import orjson
import pandas as pd


# Function to load the lookup table from a TSV file into a dictionary
def load_lookup_dict(tsv_path):
    """Load the entity lookup table from a TSV file."""
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)
    lookup_dict = {row['entity']: {'id': row['id'], 'name': row['name']} for _, row in df.iterrows()}
    lookup_keys = set(lookup_dict.keys())  # Keys for fast lookup
    return lookup_dict, lookup_keys

# Function to process a batch of JSON files
def process_batch(files, output_dir, lookup_dict, lookup_keys, new_entries):
    """Process a batch of JSON files."""
    for file_path in files:
        with open(file_path, 'rb') as f:
            data = orjson.loads(f.read())
        
        for doc in data.values():
            for sentence in doc.get("sentences", []):
                entities, entity_spans, ids, names = [], [], [], []
                
                for entity, span in zip(sentence["entities"], sentence["entity_spans"]):
                    if entity in lookup_keys:
                        ids.append(lookup_dict[entity]["id"])
                        names.append(lookup_dict[entity]["name"])
                    else:
                        new_id = f"easyner:{len(new_entries) + 1}"
                        new_name = entity
                        lookup_dict[entity] = {"id": new_id, "name": new_name}
                        lookup_keys.add(entity)
                        new_entries.append({"entity": entity, "id": new_id, "name": new_name})
                        ids.append(new_id)
                        names.append(new_name)

                    entities.append(entity)
                    entity_spans.append(span)

                sentence["entities"], sentence["entity_spans"] = entities, entity_spans
                sentence["ids"], sentence["names"] = ids, names
        
        output_path = os.path.join(output_dir, "nel_" + os.path.basename(file_path))
        with open(output_path, 'wb') as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

# Main function to load the configuration and process all JSON files
def nel_main(config: dict):
    """Process files based on the provided configuration dictionary."""
    input_dir = config["input_path"]  
    output_dir = config["output_path"]
    lookup_tsv = config["lookup_table"]
    os.makedirs(output_dir, exist_ok=True)
    
    lookup_dict, lookup_keys = load_lookup_dict(lookup_tsv)
    new_entries = []
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    batch_size = max(1, os.cpu_count() * 2)  # Adjust batch size based on system performance
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        process_batch(batch_files, output_dir, lookup_dict, lookup_keys, new_entries)
    
    if new_entries:
        pd.DataFrame(new_entries).to_csv(os.path.join(output_dir, "new_entities.tsv"), sep='\t', index=False, encoding='utf-8')
