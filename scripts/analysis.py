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
    d_id = {}

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

                # analyse entitities
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


                # analyse ids
                if "ids" in sent and len(sent.get("ids", [])) != 0:
                    for id in sent["ids"]:
                        for i, id in enumerate(sent["ids"]):
                            entity = sent["entities"][i]
                            name = sent["names"][i]

                            if id not in d_id:
                                d_id[id]={"name": name,
                                          "total_count":0,
                                            "articles_set":set(),
                                            "batch_count":{},
                                            "batch_set":set(),
                                            "entities_list": set()}

                            d_id[id]["entities_list"].add(entity)
                            d_id[id]["total_count"]+=1
                            d_id[id]["articles_set"].update([art])
                            d_id[id]["batch_set"].update([idx])

                            if idx not in d_id[id]["batch_count"]:
                                d_id[id]["batch_count"][idx]=0

                            d_id[id]["batch_count"][idx]+=1


    #                 print(sent["text"])
    #                 print(sent["entities"])
    
    df = pd.DataFrame.from_dict(d_main, orient="index")
    df_id = pd.DataFrame.from_dict(d_id, orient="index")
    
    if df.empty:
        return df, df_id
    else:
    #     df.index.name="entity"
        df.sort_values("total_count", ascending=False, inplace=True)
        df["articles_spanned"] = df["articles_set"].apply(len)
        df["batches_spanned"] = df["batch_set"].apply(len)
        df['batch_set'] = df['batch_set'].apply(lambda x: '; '.join(str(item) for item in x))
        df['articles_set'] = df['articles_set'].apply(lambda x: '; '.join(str(item) for item in x))
        df["freq_per_article"] = df["total_count"].astype("float")/df["articles_spanned"].astype("float")
        df["freq_per_batch"] = df["total_count"].astype("float")/df["batches_spanned"].astype("float")
        cols = ["total_count", "articles_spanned", "batches_spanned", "freq_per_article", "freq_per_batch", "batch_set", "batch_count","articles_set"]
        df=df[cols]

        if not df_id.empty:
            df_id.sort_values("total_count", ascending=False, inplace=True)
            df_id["articles_spanned"] = df_id["articles_set"].apply(len)
            df_id["batches_spanned"] = df_id["batch_set"].apply(len)
            df_id['batch_set'] = df_id['batch_set'].apply(lambda x: '; '.join(str(item) for item in x))
            df_id['articles_set'] = df_id['articles_set'].apply(lambda x: '; '.join(str(item) for item in x))
            df_id['entities_list'] = df_id['entities_list'].apply(lambda x: '; '.join(x))
            df_id["freq_per_article"] = df_id["total_count"].astype("float")/df["articles_spanned"].astype("float")
            df_id["freq_per_batch"] = df_id["total_count"].astype("float")/df["batches_spanned"].astype("float")
            cols = ["name","entities_list", "total_count", "articles_spanned", "batches_spanned", "freq_per_article", "freq_per_batch", "batch_set", "batch_count","articles_set"]
            df_id=df_id[cols]

        return df, df_id

def plot_frequency_barchart(df, entity, n):
    '''
    plot a frequency barchart with the top n entities, names or ids
    '''
    
    if n<=50:
        fig = plt.figure(figsize=(10,10))
        ax = sns.barplot(y=df.index[:n],x="total_count", data=df[:n])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)

        ax.bar_label(ax.containers[0])
        ax.set_title(f'Top {n} for {entity} model', size=20, pad=12)
        return fig, ax
    
    elif n<=100:
        fig = plt.figure(figsize=(20,20))
        ax = sns.barplot(y=df.index[:n],x="total_count", data=df[:n])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)

        ax.bar_label(ax.containers[0])
        ax.set_title(f'Top {n} for {entity} model', size=30, pad=15)
        return fig, ax
    
    else:
        print("Plotting more that 100 entities can result in distorted graph")
        fig = plt.figure(figsize=(2*int(n/10),2*int(n/10)))
        ax = sns.barplot(y=df.index[:n],x="total_count", data=df[:n])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)

        ax.bar_label(ax.containers[0])
        ax.set_title(f'Top {n} for {entity} model', size=4*int(n/10), pad=15)
        return fig, ax
    
    
def run(analysis_config):
    '''
    run analysis
    n= number of top entities to plot
    '''
    
    
    #if entity not in ["cell", "chemical", "disease", "gene", "species"]:
    #    raise Exception ("Not a valid entity! Make sure the input folder is named properly. If you are running NER on a new/different entity, you may want to add it to the analysis script.")
        
    input_folder = analysis_config["input_path"]
    n = int(analysis_config["plot_top_n"]) if "plot_top_n" in analysis_config else 50
    entity=analysis_config["entity_type"]
    input_files_list = get_input_files(input_folder)
    df, df_id = run_analysis(input_files_list)
    
    if df.empty:
        print("No detected entities exist within the given data!")
    else:
        # print entities graph
        fig,ax = plot_frequency_barchart(df, entity, n)
        
        path_ = analysis_config["output_path"] + "/analysis_{}/".format(entity)
        os.makedirs(path_, exist_ok=True)
        plt.savefig(path_+"{}_top_{}_entities.png".format( entity,n), bbox_inches="tight", aspect="auto", format="png")
        df.to_csv(path_+"result_entities_{}.tsv".format(entity), sep="\t")

        if not df_id.empty:
        # print id graph
            fig,ax = plot_frequency_barchart(df_id, entity, n)
            df_id['merged_label'] = df_id['name'] + ' (' + df_id.index.astype(str) + ')'
            ax.set_yticklabels(df_id['merged_label'][:n])
        
            path_ = analysis_config["output_path"] + "/analysis_{}/".format(entity)
            os.makedirs(path_, exist_ok=True)
            plt.savefig(path_+"{}_top_{}_ids.png".format( entity,n), bbox_inches="tight", aspect="auto", format="png")

            df_id.to_csv(path_+"result_ids_{}.tsv".format(entity), sep="\t")
    

if __name__ == "__main__":
       
    
    pass




