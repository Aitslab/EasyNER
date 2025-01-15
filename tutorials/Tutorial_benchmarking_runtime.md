This tutorial explains how to repeat the runtime experiments in the article.

# EasyNER runtimes
1. Set the "TIMEKEEP" parameter on top of the config file to "true"
2. Run the EasyNER pipeline repeatedly with different corpora, tokenizers and NER options using the following modules (set them to "false" in the config file)
   - Downloader
   - Sentence Splitter with spaCy or with nltk tokenizer
   - NER with BioBERT model or dictionary
   - Analysis
3. Record runtimes from the timekeep.txt document
4. Subtract/add module runtimes for individual modules to obtain total runtimes
5. Normalize runtimes by dividing with character count

     
# HunFlair2 runtimes

# BERN2 web demo runtimes

# BERN2 API runtimes
