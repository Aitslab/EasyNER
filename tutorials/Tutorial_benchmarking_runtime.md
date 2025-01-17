This tutorial explains how to repeat the runtime experiments in the article.

# EasyNER runtimes
1. Set the "TIMEKEEP" parameter on top of the config file to "true"
2. Run the EasyNER pipeline repeatedly with different corpora, tokenizers and NER options using the following modules (set them to "false" in the config file)
   - Downloader
   - Sentence Splitter with spaCy or with nltk tokenizer
   - NER with BioBERT model or dictionary
   - Analysis
3. Record runtimes from the timekeep.txt document. Note that the timekeep.txt file will be updated each time EasyNER is run. Therefore, either rename the file and/or move it to a different location to avoid overwriting.
4. Subtract/add module runtimes for individual modules to obtain total runtimes
5. Normalize runtimes by dividing with character count

     
# HunFlair2 runtimes
HunFLair2 returns the time taken when the predict.sh file is run on a terminal. When you run the predict.sh script(s), you will see the runtime in miliseconds for each run.

# BERN2 web demo runtimes
The web demo runtime is displayed on the BERN2 website (https://bern2.korea.ac.kr/) when a text or list of PMIDs are run. 

# BERN2 API runtimes
BERN2 API runtimes can be calculated by using the python time module as follows:
```python
import time

start = time.time()

YOUR CODE HERE

end = time.time()

print(end-start)
```