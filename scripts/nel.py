# author: Sonja Aits

# Named entity linking

# this script matches the detected entities to a lookup table to retrieve their unique identifiers

def NEL(lookupfile, inputfile, outputfile):
    # import
    import pandas as pd
    import json

    # read lookup file into pandas dataframe
    lookup = pd.read_csv(lookupfile, sep='\t')

    # read JSON file
    with open(inputfile) as f:
        data = json.load(f)

    # loop through each document in the JSON data
    for doc_id, doc_data in data.items():
        # loop through each sentence in the document
        for sentence in doc_data['sentences']:
            # find the matching entity IDs
            entity_ids = []
            for entity in sentence['entities']:
                entity_id = lookup.loc[lookup['term'] == entity, 'ID'].tolist()
                if entity_id:
                    entity_ids.extend(entity_id)
                else:
                    entity_ids.append('') #if no entity ID found in lookup table, add an empty string

            # add the entity IDs to the sentence
            sentence['entity_ids'] = entity_ids

    # write the updated JSON data to a file
    with open(outputfile, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)