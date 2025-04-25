# coding=utf-8

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
from pathlib import Path, PurePosixPath
import os


class NER_biobert:

    def __init__(
        self, model_dir: str, model_name: str, model_max_length=192, device=-1
    ):
        self.model_path = PurePosixPath(Path(model_dir, model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, model_max_length=model_max_length
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path
        )
        self.nlp = pipeline(
            task="ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="max",
            device=device,
        )

    def predict(self, sequence: str):
        return self.nlp(sequence)


if __name__ == "__main__":
    model_dir = "../../rafsan/models/biobert_pytorch_pretrained/"
    model_name = "HunFlair_chemical_all/"
    seq = "he is feeing very sick nitrous oxide NO nucleus eucaryotic A549 HeLa Cells oxygen."

    NER = NER_biobert(model_dir=model_dir, model_name=model_name)

    print(NER.predict(seq))
