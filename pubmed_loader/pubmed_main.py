# coding=utf-8

import bulk_downloader_pubmed
import parse_xml
import count_articles_from_json
import tqdm
import glob
import os
import argparse
import pubmed_parser as pp



if __name__ == "__main__":

    description = "Specify start and end file numbers and save path for downloaded pubmed files"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', "--start", type=int, required=False, default=0,
                        help="Starting pubmed zip file number")
    parser.add_argument('-e', "--end", type=int, required=False, default=10000000,
                        help="End pubmed zip file number (included)")
    parser.add_argument('-d', "--download_path", type=str, required=False, default="../data/tmp/pubmed/",
                        help="path where raw files are downloaded")
    parser.add_argument('-p', "--output_path", type=str, required=False, default="../data/pubmed/",
                        help="path where processed files are located")
    parser.add_argument('-b', "--baseline", type=int, required=False, default=23,
                        help="pubmed annual baseline, ex: 23 for 2023")
    parser.add_argument('-c', "--count_articles", type=bool, required=False, default=True,
                        help="Count articles and get Pubmed IDs")
    
    args = parser.parse_args()

    print("Downloading files...")

    bulk_downloader_pubmed.bulk_download(n_start = args.start,
                                    n_end=args.end,
                                    save_path=args.download_path,
                                    baseline=args.baseline)
        
    print("Download complete.")
    
    print("Processing raw files...")

    loader = parse_xml.PubMedLoader(input_path=args.download_path,
                                    output_path=args.output_path)
    loader.run_loader(file_limit=[args.start, args.end])

    if args.count_articles:
        print("counting articles")
        count_articles_from_json.count_articles(input_path=args.output_path,
                                                baseline = args.baseline)


    print("Pubmed download complete")

