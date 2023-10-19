# coding=utf-8

import json
from tqdm import tqdm

def read_articles(filename:str):
    with open(filename, encoding="utf-8") as f:
        return json.loads(f.read())

def get_sorted_files(filepath):
    '''
    get a list of sorted file paths using glob
    '''
    return sorted(glob(f'{filepath}*.json'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]))


def process_articles(articles: dict, entity_tag:str):
    '''
    process the article to contain tag
    '''
    for art in list(articles):

        for i, sent in enumerate(articles[art]["sentences"]):
            if len(sent["entities"])>0:
                articles[art]["sentences"][i]["entities"] = {entity_tag:articles[art]["sentences"][i]["entities"]}
                articles[art]["sentences"][i]["entity_spans"] = {entity_tag:articles[art]["sentences"][i]["entity_spans"]}
            else:
                articles[art]["sentences"][i]["entities"] = {}
                articles[art]["sentences"][i]["entity_spans"] = {} 

    return articles

def merge_two_articles(articles_1, articles_2):
    '''
    merge articles_2 with article_1 at entity level
    '''
    if len(articles_1)==0:
        articles_1={k:v for k,v in articles_2.items()}
        return articles_1
    
    elif len(articles_2)==0 and len(articles_1)>0: # user exception
        articles_2={k:v for k,v in articles_1.items()}
        return articles_2
    
    else:
        for art1,art2 in zip(list(articles_1), list(articles_2)):
            if art1!=art2:
                raise Exception("ERR!!!!")
            for i, sent in enumerate(articles_1[art1]["sentences"]):
                if len(articles_2[art2]["sentences"][i]["entities"])>0:
                    sent["entities"].update(articles_2[art2]["sentences"][i]["entities"])
                    sent["entity_spans"].update(articles_2[art2]["sentences"][i]["entity_spans"])
                    
    return articles_1

def run_entity_merger(input_file_paths:list, input_entity_tags:list, output_path:str):
    '''
    merge all files within
    '''
    merged_entities = {}
    
    for file_, tag in tqdm(zip(input_file_paths, input_entity_tags)):
        # Read articles
        articles = read_articles(file_)
        
        #process ner parts into dictionaries
        processed_ner_article = process_articles(articles, tag)
        
        #merge entities
        merged_entities=merge_two_articles(merged_entities, processed_ner_article)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(merged_entities, indent=2, ensure_ascii=False))
        
    return 
    
def get_batch_no_from_filename(filename):
    return re.findall(r'\d+',f)[-1]

def check_match_batch_index(filename1, filename2):
    return get_batch_no_from_filename(filename1) == get_batch_no_from_filename(filename2)
    
    
if __name__ == "__main__":

    input_folders = ["../../NER_pipeline/results/text-ner_cell_mtorandtsc1-set.json",
               "../../NER_pipeline/results/text-ner_ft_chem_mtorandtsc1_1000.json",
               "../../NER_pipeline/results/text-ner_disease_mtorandtsc1-set.json",
                "../../NER_pipeline/results/text-ner_gene_mtorandtsc1-set.json",
              "../../NER_pipeline/results/text-ner_gene_mtorandtsc1-set.json"]
    infile_entity_tags = ["cell", "chemical", "disease","gene", "species"]
    
    run_entity_merger(infile_list, infile_tags)
