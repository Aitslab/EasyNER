# coding=utf-8

import os
import numpy as np
import onnxruntime
from transformers import BertTokenizer
from typing import List


class NERInferenceSession_biobert_onnx:

    def __init__(
        self,
        model_dir: str,
        model_name: str,
        model_vocab: str,
        labels: List[str],
    ):
        self.model_path = os.path.join(model_dir, model_name)
        self.vocab_path = model_vocab
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained(self.vocab_path)
        self.session = self.create_session()
        onnxruntime.set_default_logger_severity(3)

    def create_session(self) -> onnxruntime.InferenceSession:
        # Allow caller to use symlink to model
        if os.path.islink(self.model_path):
            self.model_path = os.readlink(self.model_path)
        print("Loading model:\n  {}".format(self.model_path))
        session = onnxruntime.InferenceSession(self.model_path)
        print("Model loaded succesfully\n")
        return session

    def encode_sequence(self, sequence: str):
        encoded = self.tokenizer.encode_plus(
            sequence,
            max_length=128,  # TO DO: check if this number should be changed
            add_special_tokens=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        for i, token in enumerate(tokens):
            if token == self.tokenizer.convert_ids_to_tokens(0):
                tokens = tokens[:i]
                break

        token_type_ids = np.array(
            encoded["token_type_ids"].numpy(), dtype=np.int32
        )
        attention_mask = np.array(
            encoded["attention_mask"].numpy(), dtype=np.int32
        )
        input_ids = np.array(encoded["input_ids"].numpy(), dtype=np.int32)
        label_ids = np.array([0], dtype=np.int32)

        return {
            "tokens": tokens,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "label_ids": label_ids,
        }

    def predict(self, sequence: str):
        encodings = self.encode_sequence(sequence)

        _, logits, _ = self.session.run(
            [],
            {
                "segment_ids_1:0": encodings["token_type_ids"],
                "input_mask_1_raw_output___9:0": encodings["attention_mask"],
                "input_ids_1:0": encodings["input_ids"],
                "label_ids_1:0": encodings["label_ids"],
            },
        )

        predicted_labels = []

        for index in logits[0]:
            predicted_labels.append(self.labels[index])

        token_label_pairs = []

        for token, label in zip(encodings["tokens"], predicted_labels):
            token_label_pairs.append((token, label))

        return token_label_pairs


if __name__ == "__main__":

    model_dir = (
        "C:/Users/rafsa/Documents/AITSLAB/nlp_pipeline/rafsan/models/biobert/"
    )
    model_name = "biobert_ner.onnx"
    model_vocab = model_dir + "vocab.txt"
    labels = ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"]
    seq = "he is feeing very sick nitrous oxide NO nucleus eucaryotic A549 HeLa Cells oxygen"

    ner_session = NERInferenceSession_biobert_onnx(
        model_dir=model_dir,
        model_name=model_name,
        model_vocab=model_vocab,
        labels=labels,
    )
