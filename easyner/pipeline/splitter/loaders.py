from abc import ABC, abstractmethod
import json
import os
from glob import glob


class DataLoaderBase(ABC):
    @abstractmethod
    def load_data(self):
        pass


class StandardLoader(DataLoaderBase):
    def __init__(self, input_path):
        self.input_path = input_path

    def load_data(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            return json.loads(f.read())


class PubMedLoader(DataLoaderBase):
    def __init__(self, input_folder, limit="ALL", key="n"):
        self.input_folder = input_folder
        self.limit = limit
        self.key = key

    def load_data(self):
        """Load pre-batched PubMed files based on configured limits"""
        if self.limit == "ALL":
            return sorted(
                glob(f"{self.input_folder}*.json"),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split(self.key)[-1]),
            )
        elif isinstance(self.limit, list):
            if len(self.limit) == 2:
                if self.limit[0] > self.limit[1]:
                    raise Exception(
                        "Error! Make sure to enter in the format of [#,#] where # represents lower and upper limit numbers respectively"
                    )
                all_files = sorted(
                    glob(f"{self.input_folder}*.json"),
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split(self.key)[-1]),
                )
                processed_files = []
                for f in all_files:
                    fidx = int(os.path.splitext(os.path.basename(f))[0].split(self.key)[-1])
                    if fidx >= self.limit[0] and fidx <= self.limit[1]:
                        processed_files.append(f)
                return processed_files
            else:
                raise Exception(
                    "ERROR!! Invalid limit parameters. Make sure to enter in the format of [#,#] where # represents lower and upper limit numbers respectively"
                )
        else:
            raise Exception(
                "ERROR! Invalid filename or limit parameter! Filename should match pubmed naming convention. Ex: pubmed23n0001.json"
            )

    def load_batch(self, file_path):
        """Load a single batch file"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.loads(f.read())

    def get_batch_index(self, input_file):
        """Extract batch index from filename"""
        return int(os.path.splitext(os.path.basename(input_file))[0].split(self.key)[-1])
