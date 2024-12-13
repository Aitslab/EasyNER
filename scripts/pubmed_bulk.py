# coding=utf-8

# import bulk_downloader_pubmed
# import parse_xml
# import count_articles_from_json
import tqdm
from glob import glob
import os
import argparse
import pubmed_parser as pp
import json
import requests
import urllib.request
import time
from tqdm import tqdm, trange

def bulk_download(n_start=0, n_end=10000, nupdate=False, u_start=1167, u_end=3000, save_path="data/tmp/pubmed/", baseline=23):
    '''
    bulk download raw pubmed files
    n_start: start id of baseline files
    n_end: end id of baseline files (included)
    nupdate: include nightly update files (True/False)
    u_start: start id of nightly update files
    u_end: end id of nightly update files
    save_path: temporary save path for raw files


    '''
    os.makedirs(save_path, exist_ok=True)

    f = open(f"{save_path}err.txt", "w", encoding="utf8")
    for i in trange(n_start,n_end+1):
        # url = f'https://data.lhncbc.nlm.nih.gov/public/ii/information/MBR/Baselines/2023/pubmed23n{i:04d}.xml.gz'
        url = f'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed{baseline}n{i:04d}.xml.gz'
        try:
            urllib.request.urlretrieve(url, filename=f"{save_path}pubmed{baseline}n{i:04d}.xml.gz")
        except:
            f.write(f"{i}\n")
            continue

        if i%3==0:
            time.sleep(0.1)

    if nupdate:
        print("Downloading Nightly Update Files...")
        for i in trange(u_start,u_end+1):
            url = f'https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/pubmed{baseline}n{i:04d}.xml.gz'
            try:
                urllib.request.urlretrieve(url, filename=f"{save_path}pubmed{baseline}n{i:04d}.xml.gz")
            except:
                f.write(f"update_{i}\n")
                continue
            if i%3==0:
                time.sleep(0.1)
    f.close()



def count_articles(input_path, baseline=23):
    '''
    count articles from converted json files
    '''
    count=0
    pmids = []
    # k is used for keyword to split the filename obtained from pubmed. It's different for each annual baseline
    k = str(baseline)+"n"
    count_file = input_path + "counts.txt"
    pmid_file = input_path + "pmid_list.txt"
    input_files = sorted(glob(f'{input_path}*.json'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split(k)[-1]))
    # print(input_files)
    count_writer = open(count_file, "w", encoding="utf-8")
    pmid_writer  = open(pmid_file, "w", encoding="utf-8")
    
    for infile in tqdm(input_files):
        with open(infile, "r",encoding="utf-8") as f:
                full_articles = json.loads(f.read())
        
        count_writer.write(f"{os.path.splitext(os.path.basename(infile))[0].split(k)[-1]}\t{len(full_articles)}\n")
        count+=len(full_articles)
        pmids.extend([k for k in full_articles])
    
    count_writer.write(f"total\t{count}")
    count_writer.close()

    for pmid in sorted(pmids, key=int):
         pmid_writer.write(f"{pmid}\n")
    pmid_writer.close()


class PubMedLoader:
    
    def __init__(self, input_path,  output_path, k:str):
        self.input_path = input_path
        self.output_path = output_path
        self.counter = {}
        self.k=k
        os.makedirs(output_path, exist_ok=True)
        
    def get_input_files(self, input_path):
        # k is used for keyword to split the filename obtained from pubmed. It's different for each annual baseline
        input_files = sorted(glob(f'{input_path}*.gz'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split(self.k+"n")[-1][:-4]))
        return input_files

    
    def get_counter(self):
        return self.counter
    
    def load_xml_and_convert(self, input_file):
        data = pp.parse_medline_xml(input_file, year_info_only=False)
        
        count=0
        d_main = {}
        for art in data:
            if "abstract" in art:
                if isinstance(art["abstract"], str):
                    if len(art["abstract"])>0:
                        count+=1
                        pmid = art["pmid"] if "pmid" in art else count
                        d_main[pmid] = {"title": art["title"],
                                    "abstract":art["abstract"],
                                    "mesh_terms":art["mesh_terms"],
                                    "pubdate":art["pubdate"],
                                    "chemical_list":art["chemical_list"]}
        
        self.counter[input_file] = count
        return d_main

    def write_to_json(self, data, input_file):
        outfile = os.path.join(self.output_path, os.path.basename(input_file.split(".xml")[0])+".json")
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False))
            
    def run_loader(self):
        input_files_list = self.get_input_files(self.input_path)
        
        for i, input_file in tqdm(enumerate(input_files_list)):
            
            data = self.load_xml_and_convert(input_file)
            self.write_to_json(data, input_file)

def run_pbl(pbl_config):

    # print("Downloading files...")

    download_path= "data/tmp/pubmed/" if len(pbl_config["raw_download_path"])==0 else pbl_config["raw_download_path"] 

    if pbl_config["subset"] == True:
        if pbl_config["get_nightly_update_files"]:
            bulk_download(n_start = pbl_config["subset_range"][0],
                n_end=pbl_config["subset_range"][1],
                nupdate=True,
                u_start=pbl_config["update_file_range"][0],
                u_end=pbl_config["update_file_range"][1],
                save_path=download_path,
                baseline=pbl_config["baseline"])
            
        else:
            bulk_download(n_start = pbl_config["subset_range"][0],
                            n_end=pbl_config["subset_range"][1],
                            save_path=download_path,
                            baseline=pbl_config["baseline"])

        
    else:
        if pbl_config["get_nightly_update_files"]:
            bulk_download(nupdate=True,
                            u_start=pbl_config["update_file_range"][0],
                            u_end=pbl_config["update_file_range"][1],
                            save_path=download_path,
                            baseline=pbl_config["baseline"])
        else:
            bulk_download(save_path=download_path,
                            baseline=pbl_config["baseline"])
        
    print("Download complete.")
    
    print("Processing raw files...")
    

    loader = PubMedLoader(input_path=download_path,
                            output_path=pbl_config["output_path"],
                            k=pbl_config["baseline"])
     
    loader.run_loader()

    if pbl_config["count_articles"]:
        print("counting articles")
        count_articles(input_path=pbl_config["output_path"],
                        baseline = pbl_config["baseline"])


    print("Pubmed download complete")
    

if __name__ == "__main__":

    # description = "Specify start and end file numbers and save path for downloaded pubmed files"
    run_pbl()

