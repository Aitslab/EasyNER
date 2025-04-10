# coding=utf-8

# import bulk_downloader_pubmed
# import parse_xml
# import count_articles_from_json
import json
import os
import time
import urllib.request
from glob import glob

import pubmed_parser as pp
from tqdm import tqdm, trange


def bulk_download(
    n_start=0,
    n_end=10000,
    nupdate=False,
    u_start=1167,
    u_end=3000,
    save_path="data/tmp/pubmed/",
    baseline=23,
):
    """
    bulk download raw pubmed files
    n_start: start id of baseline files
    n_end: end id of baseline files (included)
    nupdate: include nightly update files (True/False)
    u_start: start id of nightly update files
    u_end: end id of nightly update files
    save_path: temporary save path for raw files


    """

    print(f"Downloading files to: {os.path.abspath(save_path)}")

    os.makedirs(save_path, exist_ok=True)

    f = open(f"{save_path}err.txt", "w", encoding="utf8")
    for i in trange(n_start, n_end + 1):
        # url = f'https://data.lhncbc.nlm.nih.gov/public/ii/information/MBR/Baselines/2023/pubmed23n{i:04d}.xml.gz'
        url = f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed{baseline}n{i:04d}.xml.gz"
        try:
            full_path = f"{save_path}pubmed{baseline}n{i:04d}.xml.gz"
            print(f"Downloading {url} to {os.path.abspath(full_path)}")
            urllib.request.urlretrieve(url, filename=full_path)
            # Verify file exists and has size
            if os.path.exists(full_path):
                print(f"SUCCESS: {full_path} exists with size: {os.path.getsize(full_path)} bytes")
            else:
                print(f"ERROR: File not created: {full_path}")
        except Exception as e:
            print(f"ERROR downloading {i}: {str(e)}")
            f.write(f"{i}\t{str(e)}\n")
            continue

        if i % 3 == 0:
            time.sleep(0.1)

    if nupdate:
        print("Downloading Nightly Update Files...")
        for i in trange(u_start, u_end + 1):
            url = (
                f"https://ftp.ncbi.nlm.nih.gov/pubmsed/updatefiles/pubmed{baseline}n{i:04d}.xml.gz"
            )
            try:
                urllib.request.urlretrieve(
                    url, filename=f"{save_path}pubmed{baseline}n{i:04d}.xml.gz"
                )
            except:
                f.write(f"update_{i}\n")
                continue
            if i % 3 == 0:
                time.sleep(0.1)
    f.close()


def count_articles(input_path, baseline=23):
    """
    count articles from converted json files
    """
    count = 0
    pmids = []
    # k is used for keyword to split the filename obtained from pubmed.
    # It's different for each annual baseline
    k = str(baseline) + "n"
    count_file = input_path + "counts.txt"
    pmid_file = input_path + "pmid_list.txt"
    input_files = sorted(
        glob(f"{input_path}*.json"),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split(k)[-1]),
    )
    # print(input_files)
    count_writer = open(count_file, "w", encoding="utf-8")
    pmid_writer = open(pmid_file, "w", encoding="utf-8")

    for infile in tqdm(input_files):
        with open(infile, "r", encoding="utf-8") as f:
            full_articles = json.loads(f.read())

        count_writer.write(
            f"{os.path.splitext(os.path.basename(infile))[0].split(k)[-1]}\t{len(full_articles)}\n"
        )
        count += len(full_articles)
        pmids.extend([k for k in full_articles])

    count_writer.write(f"total\t{count}")
    count_writer.close()

    for pmid in sorted(pmids, key=int):
        pmid_writer.write(f"{pmid}\n")
    pmid_writer.close()


class PubMedLoader:

    def __init__(self, input_path, output_path, k: str, require_abstract=False):
        self.input_path = input_path
        self.output_path = output_path
        self.counter = {}
        self.k = k
        self.require_abstract = require_abstract
        self.filter_stats = {
            "total_articles": 0,
            "no_abstract": 0,
            "abstract_not_string": 0,
            "empty_abstract": 0,
            "included_articles": 0,
        }
        os.makedirs(output_path, exist_ok=True)

    def get_input_files(self, input_path):
        # k is used for keyword to split the filename obtained from pubmed.
        # It's different for each annual baseline
        input_files = sorted(
            glob(f"{input_path}*.gz"),
            key=lambda x: int(
                os.path.splitext(os.path.basename(x))[0].split(self.k + "n")[-1][:-4]
            ),
        )
        return input_files

    def get_counter(self):
        return self.counter

    def load_xml_and_convert(self, input_file):
        data = pp.parse_medline_xml(input_file, year_info_only=False)

        d_main = {}
        local_stats = {
            "total": len(data),
            "no_abstract": 0,
            "abstract_not_string": 0,
            "empty_abstract": 0,
            "included": 0,
        }

        for art in data:
            self.filter_stats["total_articles"] += 1
            local_stats["total"] += 1

            pmid = art.get("pmid", str(self.filter_stats["total_articles"]))

            # Check if we should include this article based on abstract requirements
            include_article = True

            if self.require_abstract:
                if "abstract" not in art:
                    self.filter_stats["no_abstract"] += 1
                    local_stats["no_abstract"] += 1
                    include_article = False
                elif not isinstance(art["abstract"], str):
                    self.filter_stats["abstract_not_string"] += 1
                    local_stats["abstract_not_string"] += 1
                    include_article = False
                elif len(art["abstract"]) == 0:
                    self.filter_stats["empty_abstract"] += 1
                    local_stats["empty_abstract"] += 1
                    include_article = False

            if include_article:
                self.filter_stats["included_articles"] += 1
                local_stats["included"] += 1

                # Create a default empty abstract if it doesn't exist and we're not requiring it
                abstract = art.get("abstract", "") if not self.require_abstract else art["abstract"]

                d_main[pmid] = {
                    "title": art.get("title", ""),
                    "abstract": abstract,
                    "mesh_terms": art.get("mesh_terms", ""),
                    "pubdate": art.get("pubdate", ""),
                    "chemical_list": art.get("chemical_list", ""),
                }

                # # Debugging: Check for empty abstract after assignment
                # if abstract == "":
                #     print(
                #         "DEBUG: Empty abstract found after assignment. Successfully included article."
                #     )
                #     print("Article PMID:", pmid)
                #     print("Article Title:", art.get("title", ""))
                #     print("Full Article Data:", art)  # Print the entire article data for inspection
                #     import sys  # Import the sys module

                #     sys.exit(1)  # Exit the program

        self.counter[input_file] = local_stats
        return d_main

    def write_to_json(self, data, input_file):
        outfile = os.path.join(
            self.output_path, os.path.basename(input_file.split(".xml")[0]) + ".json"
        )
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False))

    def run_loader(self):
        input_files_list = self.get_input_files(self.input_path)

        for i, input_file in tqdm(enumerate(input_files_list)):

            data = self.load_xml_and_convert(input_file)
            self.write_to_json(data, input_file)

    def get_filter_statistics(self):
        """Return statistics about filtered articles"""
        return self.filter_stats

    def write_statistics_report(self, output_file="filter_statistics.json"):
        """Write filtering statistics to a JSON file"""
        report_path = os.path.join(self.output_path, output_file)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {"overall": self.filter_stats, "by_file": self.counter},
                f,
                indent=2,
                ensure_ascii=False,
            )


def run_pbl(pbl_config):

    download_path = (
        "data/tmp/pubmed/"
        if len(pbl_config["raw_download_path"]) == 0
        else pbl_config["raw_download_path"]
    )

    # Check if we should skip download
    if not pbl_config.get("skip_download", False):
        print("Downloading files...")
        if pbl_config["subset"]:
            if pbl_config["get_nightly_update_files"]:
                bulk_download(
                    n_start=pbl_config["subset_range"][0],
                    n_end=pbl_config["subset_range"][1],
                    nupdate=True,
                    u_start=pbl_config["update_file_range"][0],
                    u_end=pbl_config["update_file_range"][1],
                    save_path=download_path,
                    baseline=pbl_config["baseline"],
                )
            else:
                bulk_download(
                    n_start=pbl_config["subset_range"][0],
                    n_end=pbl_config["subset_range"][1],
                    save_path=download_path,
                    baseline=pbl_config["baseline"],
                )
        else:
            if pbl_config["get_nightly_update_files"]:
                bulk_download(
                    nupdate=True,
                    u_start=pbl_config["update_file_range"][0],
                    u_end=pbl_config["update_file_range"][1],
                    save_path=download_path,
                    baseline=pbl_config["baseline"],
                )
            else:
                bulk_download(save_path=download_path, baseline=pbl_config["baseline"])
        print("Download complete.")
    else:
        print("Download step skipped based on configuration.")

    print("Processing raw files...")

    # Pass the require_abstract option to the loader
    loader = PubMedLoader(
        input_path=download_path,
        output_path=pbl_config["output_path"],
        k=pbl_config["baseline"],
        require_abstract=pbl_config.get("require_abstract", True),
    )

    loader.run_loader()
    # Generate statistics report
    loader.write_statistics_report()

    if pbl_config["count_articles"]:
        print("Counting articles")
        count_articles(input_path=pbl_config["output_path"], baseline=pbl_config["baseline"])

    print("Pubmed processing complete")


if __name__ == "__main__":

    # description = "Specify start and end file numbers and save path for downloaded pubmed files"
    run_pbl()
