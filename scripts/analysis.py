# coding=utf-8

import os
import json
from glob import glob
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_input_files(input_folder_path):
    '''
    get all NER result files within the given input folder in a sorted list
    '''
    return sorted(glob(f'{input_folder_path}*.json'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]))


def run_analysis(input_files_list):
    '''
    Get input files list and return NER results per batch and per article 
    '''
    d_main= {}
    if len(input_files_list) == 0:
        raise Exception ("Error! No input file could be detected. Please provide a correct path!")
        
    count_articles=0
    for batch in tqdm(input_files_list):
        
        #check for file naming errors
        try:
            idx = int(os.path.splitext(os.path.basename(batch))[0].split("-")[-1])
        except:
            raise Exception("Error! NER files do not contain index in the end. Add index to the designated files.")

        with open(batch, encoding="utf8") as f:
            articles = json.loads(f.read())
        
        count_articles+=len(articles)
        #Loop over articles    
        for art in tqdm(articles):
    #         print(art)

            #loop over each sentence
            for sent in articles[art]["sentences"]:
                if len(sent["entities"])!=0:
                    for entity in sent["entities"]:
                        if entity not in d_main:
                            d_main[entity]={"total_count":0,
                                            "articles_set":set(),
                                            "batch_count":{},
                                            "batch_set":set()}

                        d_main[entity]["total_count"]+=1
                        d_main[entity]["articles_set"].update([art])
                        d_main[entity]["batch_set"].update([idx])

                        if idx not in d_main[entity]["batch_count"]:
                            d_main[entity]["batch_count"][idx]=0

                        d_main[entity]["batch_count"][idx]+=1


    #                 print(sent["text"])
    #                 print(sent["entities"])
    
    df = pd.DataFrame.from_dict(d_main, orient="index")

    if df.empty:
        return df
    else:
    #     df.index.name="entity"
        df.sort_values("total_count", ascending=False, inplace=True)
        
        df["articles_spanned"] = df["articles_set"].str.len()
        df["batches_spanned"] = df["batch_set"].str.len()
        df["freq_per_article"] = df["total_count"].astype("float")/df["articles_spanned"].astype("float")
        df["freq_per_batch"] = df["total_count"].astype("float")/df["batches_spanned"].astype("float")
        cols = ["total_count", "articles_spanned", "batches_spanned", "freq_per_article", "freq_per_batch", "batch_set", "batch_count","articles_set"]
        df=df[cols]
        return df

def plot_frequency_barchart(df, entity, n):
    '''
    plot a frequency barchart with the top n entities
    '''
    fig = plt.figure(figsize=(10,10))
    ax = sns.barplot(y=df.index[:n],x="total_count", data=df[:n])
    ax.bar_label(ax.containers[0])
    ax.set_title(f'Top {n} entities for {entity} model', size=15, pad=15)
    return fig, ax
    
def run(analysis_config, n=50):
    '''
    run analysis
    n= number of top entities to plot
    '''
    entity = analysis_config["input_path"].split("_")[-1].strip("\/\n ")
    #prefix = analysis_config["input_path"].split("_")[-2].strip("\/\n ")
    prefix="autophagyANDcancer2020-23"
    
    #if entity not in ["cell", "chemical", "disease", "gene", "species"]:
    #    raise Exception ("Not a valid entity! Make sure the input folder is named properly. If you are running NER on a new/different entity, you may want to add it to the analysis script.")
        
    input_folder = analysis_config["input_path"]
    input_files_list = get_input_files(input_folder)
    df = run_analysis(input_files_list)
    
    if df.empty:
        print("No detected entities exist within the given data!")
    else:
    
        fig,ax = plot_frequency_barchart(df, entity, n)
        
        path_ = analysis_config["output_path"] + "/analysis_{}_{}/".format(prefix,entity)
        os.makedirs(path_, exist_ok=True)
        plt.savefig(path_+"{}_{}_top_{}.png".format(prefix, entity,n), bbox_inches="tight", aspect="auto", format="png")
        df.to_csv(path_+"{}_result_{}.tsv".format(prefix, entity), sep="\t")
    

if __name__ == "__main__":
       
    
    pass




