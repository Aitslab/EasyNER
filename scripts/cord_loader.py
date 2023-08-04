# coding=utf-8

# cord-loader.py
# Build JSON data from CORD CSV data
# https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html

import os
import csv
import hashlib
import json
from tqdm import tqdm


def run(input_file: str, output_file: str, subset: bool, subset_file: str):

    def hash(s: str):
        return hashlib.sha256(s.encode('utf-8')).hexdigest()
    
    
    if subset==True:
        uid_set = set()
        with open(subset_file) as f:
            for line in f.readlines():
                uid_set.update([line.strip()])
            print(uid_set)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    
        

    result = {}

    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=",")

        for row in tqdm(reader):
            title = row["title"]
            abstract = row["abstract"]

            if title == "" or abstract == "":
                continue

            cord_uid = row["cord_uid"]
            doi = row["doi"]
            pmcid = row["pmcid"]
            pubmed_id = row["pubmed_id"]
            mag_id = row["mag_id"]
            who_covidence_id = row["who_covidence_id"]
            arxiv_id = row["arxiv_id"]
            url = row["url"]

            if cord_uid != "":
                id_type = "cord_uid"
                id = cord_uid

            elif doi != "":
                id_type = "doi"
                id = doi

            elif pmcid != "":
                id_type = "pmcid"
                id = pmcid

            elif pubmed_id != "":
                id_type = "pubmed_id"
                id = pubmed_id

            elif mag_id != "":
                id_type = "mag_id"
                id = mag_id

            elif who_covidence_id != "":
                id_type = "who_covidence_id"
                id = who_covidence_id

            elif arxiv_id != "":
                id_type = "arxiv_id"
                id = arxiv_id

            else:
                id_type = "hash"
                id = hash(title)

            if subset and cord_uid.lower() not in uid_set:
                continue
            else:
                result[id] = {
                    "title": title,
                    "abstract": abstract,
                    "id_type": id_type,
                    "cord_uid": cord_uid,
                    "doi": doi,
                    "pmcid": pmcid,
                    "pubmed_id": pubmed_id,
                    "mag_id": mag_id,
                    "who_covidence_id": who_covidence_id,
                    "arxiv_id": arxiv_id,
                    "url": url,
                }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


'''
if __name__ == "__main__":
    basedir = "data/cord/"
    date = "2020-11-28"
    run(
        input_file=f"metadata-{date}.csv",
        output_file=f"metadata-{date}.json",
        subset=True,
        subset_file=""
    )
'''
