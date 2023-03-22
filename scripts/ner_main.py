# coding=utf-8

import spacy
import os
import re
import json
from tqdm import tqdm
from spacy.matcher import PhraseMatcher
from . import ner_biobert, util
from .ner_inference import NERInferenceSession_biobert_onnx

def run_ner_main(ner_config: dict, batch_file):
    '''
    run NER in batches from sentence splitter output
    '''

    with open(batch_file, "r",encoding="utf-8") as f:
        articles = json.loads(f.read())
    
    # get batch IDs
    regex=re.compile(r'\d+')
    try:
        batch_index=int(regex.findall(os.path.basename(batch_file))[-1])
    except:
        print(batch_file)
        raise Exception("Filenames not numbered!")
        
        
    # Prepare spacy, if it is needed
    if ner_config["model_type"] == 'spacy_phrasematcher':
        print("Running NER with spacy")
        nlp = spacy.load(ner_config["model_name"])
        terms = []
        with open(ner_config["vocab_path"],'r') as f:
            for line in f:
                x = line.strip()
                terms.append(x)
        print("Phraselist complete")

        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(term) for term in terms]
        matcher.add(ner_config["entity_type"], patterns)
        
    # Run prediction on each sentence in each article.
    for pmid in tqdm(articles, desc=f'batch:{batch_index}'):

        sentences = articles[pmid]["sentences"]

        #Predict with spacy PhraseMatcher, if it has been selected       
        if ner_config["model_type"] == 'spacy_phrasematcher':

            for i, sentence in enumerate(sentences):
                ner_class = ner_config["entity_type"]
                
                doc = nlp(sentence["text"])
                if ner_config["store_tokens"] == "yes":
                    tokens = []
                    # tokens_idxs = []  #uncomment if you want a list of token character offsets within the sentence
                    for token in doc:
                        tokens.append(token.text) #to get a list of tokens in the sentence
                    # tokens_idxs.append(token.idx) #uncomment if you want a list of token character offsets within the sentence
                    articles[pmid]["sentences"][i]["tokens"] = tokens

                entities = []
                spans = []
                matches = matcher(doc)

                for match_id, start, end in matches:
                    span = doc[start:end]
                    ent = span.text
                    entities.append(ent)
                    first_char = span.start_char
                    last_char = span.end_char - 1
                    spans.append((first_char, last_char)) 


                articles[pmid]["sentences"][i]["NER class"] = ner_class
                articles[pmid]["sentences"][i]["entities"] = entities
                articles[pmid]["sentences"][i]["entity spans"] = spans


        #Predict with BioBERT onnx model, if it has been selected
        
        elif ner_config["model_type"] == 'biobert_onnx':
            
            print("Running NER with biobert_onnx")
            ner_session = NERInferenceSession_biobert_onnx(
                model_dir=ner_config["model_folder"],
                model_name=ner_config["model_name"],
                model_vocab=ner_config["vocab_path"],
                labels=ner_config["labels"]
                )
                
            for i, sentence in enumerate(sentences):
                #ner_class = ner_config["entity_type"]


                token_label_pairs = ner_session.predict(sentence["text"])
                if ner_config["store_tokens"] == "yes":
                    tokens = []
                    for pair in token_label_pairs:
                        tokens.append(pair[0])
                    articles[pmid]["sentences"][i]["tokens"] = tokens
                
                x = co_occurrence_extractor(detokenize(token_label_pairs))

                articles[pmid]["sentences"][i]["NER class"] = ner_class
                articles[pmid]["sentences"][i]["entities"] = x["entities"]

                spans = []

                # Run if each entity present in the sentence only once
                if len(x["entities"]) == len(set(x["entities"])): 
                    for ent in x["entities"]:
                        first_char = sentence["text"].find(ent)
                        last_char = first_char + len(ent) - 1
                        spans.append((first_char, last_char))

                # Run if at least one entity present in the sentence more than once
                else:
                    ner_counter = 0
                    text = sentence["text"]

                    while ner_counter < len(x["entities"]):
                        ent = x["entities"][ner_counter]

                        if x["entities"].count(ent) == 1:
                            first_char = sentence["text"].find(ent)
                            last_char = first_char + len(ent) - 1
                            spans.append((first_char, last_char))

                        else:
                            first_char = text.find(ent)
                            last_char = first_char + len(ent) - 1
                            spans.append((first_char, last_char))
                            mask="x"*len(ent)
                            text = text[:first_char]+mask+text[last_char+1:]

                        ner_counter = ner_counter + 1

                articles[pmid]["sentences"][i]["entity spans"] = spans

            
        elif ner_config["model_type"] == 'biobert_finetuned':
            
            #print("Running NER with finetuned BioBERT")
            
            
            ner_session = ner_biobert.NER_biobert(
            model_dir=ner_config["model_folder"],
                model_name=ner_config["model_name"]
            )
            
            for i, sentence in enumerate(sentences):
                try:
                    # the entities predicted are all uncased but the entity within the sentence is cased
                    entities = ner_session.predict(sentence["text"])
                except:
                    # exception due to existence of utf tags in the data, which is incomprehensable/non-tokenizable by the model
                    print("batch {}, sentence no. {} with text [{}] was not predicted".format(batch_index, i, sentence))
                    entities = []
                    
                entities_list = []
                entity_spans_list = []
                if len(entities)>0:
                    for ent in entities:
                        entities_list.append(ent["word"])
                        entity_spans_list.append([ent["start"],ent["end"]])
                        
                articles[pmid]["sentences"][i]["entities"] = entities_list
                articles[pmid]["sentences"][i]["entity_spans"] = entity_spans_list
                
            
    util.append_to_json_file(f'{ner_config["output_path"]}/{ner_config["output_file_prefix"]}-{batch_index}.json', articles)        
    return batch_index
    
def filter_files(list_files, start, end):
    '''
    filter files based on start and end
    '''
    filtered_list_files = []
    for f in list_files:
        f_idx = int(os.path.splitext(os.path.basename(f))[0].split("-")[-1])
        if f_idx>=start and f_idx<=end:
            filtered_list_files.append(f)
    
    return filtered_list_files

if __name__ == "__main__":
    pass




