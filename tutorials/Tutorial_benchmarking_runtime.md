This tutorial explains how to repeat the runtime experiments in the article. Runtimes were measured for multiple corpora and then normalized to corpus character count.

Following are the corpora and their character counts

<center>

| Corpus                      |    Characters |
|:---------------------------|---------:|
| BioID             |  4620590 |
| BioRED            |   156351 |
| Lund Autophagy 1                    |  1451436 |
| Lund Autophagy 2                    | 12287696 |
| Lund COVID-19            |    13060 |
| medmentions        |  6464070 |
| tmvar3             |   753565 |

</center>


# EasyNER runtimes
1. Set the "TIMEKEEP" parameter on top of the config file to "true"
2. Run the EasyNER pipeline repeatedly with different corpora, tokenizers and NER options using the following modules (set them to "false" in the config file)
   - Downloader
   - Sentence Splitter with spaCy or with nltk tokenizer
   - NER with BioBERT model or dictionary
   - Analysis
3. Runtimes are recorded in the timekeep.txt document in the main EasyNER folder (same folder as the config file) in seconds.
4. Rename the timekeep.txt file after each run and/or move it to a different location as the next run will overwrite it.
5. Subtract/add module runtimes for individual modules to obtain total runtimes.
     
# HunFlair2 runtimes
1. Run the HunFLair2 the predict.sh file on a terminal with different corpora
2. Record the runtime in milliseconds which is shown in the terminal after the run.
   
# BERN2 web demo runtimes
1. Run the web demo runtime on the BERN2 website (https://bern2.korea.ac.kr/) with plain text or list of PMIDs.
2. Record the runtime in milliseconds which is shown in the web browser with the results after the run.

# BERN2 API runtimes
1. BERN2 API runtimes can be calculated by using the python time module as follows:
```python
import time

start = time.time()

CODE FOR RUNNING THE API

end = time.time()

print(end-start)
```
2. Record the untime which is shown in the terminal after the run.
