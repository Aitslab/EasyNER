This is a step by step guide to evaluating the performance of EasyNER and other BioNLP tools on corpora in PubTator format on entity-level. 

The evaluation script calculates false-positives and -negatives, single class precision, recall and F1 score, and micro, macro and weighted averages of the precision, recall and F1 score across all entity classes.

# Conversion of PubTator format to EasyNER input and prediction
1. Place the MedMention, tmVar 3.0 or BioRED corpus in PubTator format in a folder
2. Modify [convert_hunfliar2_pubtator-to-json.py](https://github.com/Aitslab/EasyNER/blob/main/supplementary/experiment_scripts/convert_hunflair2_pubtator_to_json.py). First, enter the correct path for infile and outfile in the bottom. Then, change the last line to either of these depending on the input:
  
   convert_bioid_to_json(infile,outfile)

   convert_biored_to_json(infile,outfile)
   
   convert_medmention_to_json(infile, outfile)
   
   convert_tmvar3_to_json(infile, outfile)

4. Then run the script in the Anaconda prompt by navigating to its folder (from EasyNER main folder type cd supplementary/experiment_scripts/) and typing the following: python convert_hunflair2_pubtator_to_json.py
5. The output will be a JSON file in the same format as the EasyNER data loaders produce, i.e. it does not contain the PubTator annotations.
6. Use the converted JSON file as input for the EasyNER Sentence Splitter and then run the output through the NER module. If you want to evaluate the effect of the Postprocessing module also run this on the output of the NER module.
7. After the EasyNER run, convert the EasyNER output JSON file back to PubTator format by running [convert_easyner_output_json_to_pubtator.py](https://github.com/Aitslab/EasyNER/blob/main/supplementary/experiment_scripts/convert_easyner_output_json_to_pubtator.py)

If you want to evaluate predictions for multiple entity classes, you have to repeat the above procedure for each class.
If you then want to combine the EasyNER JSON files for multiple classes into a single JSON file, you run the [entity_merger2 script](https://github.com/Aitslab/EasyNER/blob/main/experiments/experiment_scripts/entity_merger2.py). For this you will have to define an order of priority in case of overlapping annotations. Alternatively, you can evaluate each class separately as we have done in the article.

# Evaluation of annotated files in PubTator format

1. Make sure that the gold standard and prediction files contain both the raw texts and the annotation. If the raw texts are missing, as was the case for the [BERN2 annotations](https://github.com/hu-ner/hunflair2-experiments/tree/main/annotations/bern) from the HunFlair2 repo, add them following this [Jupyter notebook](https://github.com/Aitslab/EasyNER/blob/main/supplementary/experiment_scripts/preprocess_BERN2_into_evaluation_ready_format.ipynb)
2. Make sure that the gold standard and prediction files have entity identifiers in their annotations. If they are missing, you can add the dummy identifier "-1" by running the [preprocessing script](https://github.com/Aitslab/EasyNER/blob/main/supplementary/experiment_scripts/preprocess_pubtatorformat.py). In the same script, entity classes can be remapped to suitable class names, as was done for the MedMentions corpus for our experiments.
3. Create a data folder for your input and output (e.g. called "evaluation") and subfolders for the gold standard files ("goldstandard") and one for each tool you want to evaluate named with the name of the tool (e.g. "easyner")
4. Name the gold standard file with the corpus name and place it in the goldstandard subfolder (e.g. PATH/evaluation/goldstandard/medmentions.txt)
5. Name the prediction file for each tool also with the corpus name and place it in the respective tool subfolder (e.g. PATH/evaluation/easyner/medmentions.txt)
6. Run the [evaluation script](https://github.com/Aitslab/EasyNER/blob/main/supplementary/experiment_scripts/evaluate_ner_pubtatorformat.py) with the following input arguments:
    - corpora: A Python dictionary containing the names of the corpora as keys and the and a list with the entity classes to be evaluated as values. The names need to match the PubTator format txt files from step 3/4
    - tools: A Python list of tool names to be evaluated. The names need to match the names of the subfolders you created in step 2
    - data_folder: Folder for input and output files. The path needs to match the data folder you created in step 2 (e.g. PATH/evaluation/)
    - offset_stride: Allowed offset for entity span matching. With a stride of 1, start and/or end of the detected span can differ by 1 between goldstandard and prediction for a true-positive.

The metrics are saved in a file called "metrics.tsv" in your data folder

### PubTator format example:
0|t|Variation in the CXCR1 gene (IL8RA) is not associated with susceptibility to chronic periodontitis.<br>
0|a|BACKGROUND: The chemokine receptor 1 CXCR-1 (or IL8R-alpha) is a specific receptor for the interleukin 8 (IL-8), which is chemoattractant for neutrophils and has an important role in the inflammatory response. The polymorphism rs2234671 at position Ex2+860G>C of the CXCR1 gene causes a conservative amino acid substitution (S276T). This single nucleotide polymorphism (SNP) seemed to be functional as it was associated with decreased lung cancer risk. Previous studies of our group found association of haplotypes in the IL8 and in the CXCR2 genes with the multifactorial disease chronic periodontitis. In this study we investigated the polymorphism rs2234671 in 395 Brazilian subjects with and without chronic periodontitis. FINDINGS: Similar distribution of the allelic and genotypic frequencies were observed between the groups (p>0.05). CONCLUSIONS: The polymorphism rs2234671 in the CXCR1 gene was not associated with the susceptibility to chronic periodontitis in the studied Brazilian population.<br>
0&nbsp;&nbsp;&nbsp;&nbsp;17&nbsp;&nbsp;&nbsp;&nbsp;22&nbsp;&nbsp;&nbsp;&nbsp;CXCR1&nbsp;&nbsp;&nbsp;&nbsp;Gene&nbsp;&nbsp;&nbsp;&nbsp;NCBI Gene:3577<br>
0&nbsp;&nbsp;&nbsp;&nbsp;29&nbsp;&nbsp;&nbsp;&nbsp;34&nbsp;&nbsp;&nbsp;&nbsp;IL8RA&nbsp;&nbsp;&nbsp;&nbsp;Gene&nbsp;&nbsp;&nbsp;&nbsp;-1	

### PubTator format schema:
pubmed or document id|t|title text<br>
pubmed or document id|a|abstract text<br>
pubmed or document id&nbsp;&nbsp;&nbsp;&nbsp;start of span&nbsp;&nbsp;&nbsp;&nbsp;end of span&nbsp;&nbsp;&nbsp;&nbsp;detected text&nbsp;&nbsp;&nbsp;&nbsp;entity class&nbsp;&nbsp;&nbsp;&nbsp;entity identifier or dummy value (-1)

