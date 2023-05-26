# coding=utf-8
## THis script is used to search for entities in a list from output abstracts

import json
import os
from pathlib import Path, PurePosixPath

def find_test_vs_pred_errors(test_file, pred_file):
    '''
    function to compare errors between annotated IOB2 test file and predicted results
    in IOB2 from BioBERT-like output

    input
        test_file: test file with IOB2 tags
        pred_file: prediction file with IOB2 tags
    
    output
        Dictionary (JSON structure) with sentences and errors
    '''
    with open(test_file, 'r', encoding="utf8") as f:
        lines_test = f.readlines()
    with open(pred_file, 'r', encoding="utf8") as f:
        lines_pred = f.readlines()
    if len(lines_test)!=len(lines_pred):
        raise Exception("ERR")
        
    sentences = []
    current_sentence = []

    for line_t,line_p in zip(lines_test,lines_pred):
        line_t = line_t.strip()
        line_p = line_p.strip()
        if line_t == "" or line_p=="":
            if len(current_sentence) > 0:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            parts_t = line_t.split()
            parts_p = line_p.split()
            
            word = parts_t[0]
            tag_t = parts_t[-1]
            tag_p = parts_p[-1]
            
            current_sentence.append((word, tag_t, tag_p))

    if len(current_sentence) > 0:
        sentences.append(current_sentence)

#     with open(output_file, 'w') as f:
    
    results = {"sentences":[]}
    for sentence in sentences:
        words, tags_t, tags_p = zip(*sentence)
        
        all_words = []
        all_tags_t = []
        all_tags_p = []
        
        current_words = []
        current_tags_t = []
        current_tags_p = []
        
        
        cont=False
        for word, tag_t, tag_p in sentence:
#             print(words,tags_t,tags_p)
            
            if tag_t != "O" or tag_p!="O":
                
                if tag_t=="B" or tag_p=="B":
                    cont=True
                
                if cont==True:
                    current_words.append(word)
                    current_tags_t.append(tag_t)
                    current_tags_p.append(tag_p)
            
            if tag_t=="O" and tag_p=="O":
                cont=False
                if len(current_words)>0:
                    if current_tags_t!=current_tags_p:
                        all_words.append(({"word": " ".join(current_words),
                                         "true_tags": current_tags_t,
                                         "pred_tags": current_tags_p}))

                    current_words=[]
                    current_tags_t=[]
                    current_tags_p=[]
                    
        if len(all_words)!=0:
            results["sentences"].append({"text":" ".join(words), 
                                    "words": all_words
                                    })
        
    return results
                    


if __name__ == "__main__":
    
    input_folder = "../../NER_pipeline/results_testeval_p50/test_results/huner/huner_cell/"
    test_file = PurePosixPath(Path(input_folder, "test.txt"))
    pred_file = PurePosixPath(Path(input_folder, "test_predictions.txt"))
    outfile = PurePosixPath(Path(input_folder, "errors.json"))

    
    
    results = find_test_vs_pred_errors(test_file, pred_file)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
    
    





