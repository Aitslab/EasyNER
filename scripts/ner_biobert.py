# coding=utf-8

from typing import Dict, List, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    Pipeline,
)
from pathlib import Path, PurePosixPath
import torch  # Ensure torch is imported if not already


class NER_biobert:

    def __init__(
        self, model_dir: str, model_name: str, model_max_length=192, device=-1
    ):
        from easyner.pipeline.ner.utils import get_device_int

        self.model_path = PurePosixPath(Path(model_dir, model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, model_max_length=model_max_length
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path
        )

        self.model.eval()  # Set model to evaluation mode

        print(f"NER_BIOBERT: Creating pipeline with device={device}")
        self.nlp: Pipeline = pipeline(
            task="ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="max",
            device=get_device_int(device),
        )

    @DeprecationWarning
    def predict(self, sequence: str) -> List[Dict[str, Any]]:
        """Process a single text sequence"""
        print(f"Processing text in single sequence: {sequence}")
        return self.nlp(sequence)

    def predict_dataset(
        self, dataset: Dataset, text_column="text", batch_size=None
    ):
        """
        Process an entire HuggingFace dataset for better GPU utilization.

        Args:
            dataset: HuggingFace dataset with text column
            text_column: Name of the column containing text
            batch_size: Batch size to use (None for auto-determination)

        Returns:
            Dataset with predictions added
        """
        print(f"Processing dataset", flush=True)

        # Check if dataset is empty
        if dataset is None or len(dataset) == 0:
            print("Empty dataset provided")
            return dataset

        print(f"Number of texts: {len(dataset)}", flush=True)
        print(f"Processing dataset with batch size: {batch_size}", flush=True)

        if batch_size is None:
            try:
                from easyner.pipeline.ner.utils import (
                    calculate_optimal_batch_size,
                )

                print("Auto-determining optimal batch size", flush=True)

                batch_size = calculate_optimal_batch_size(
                    pipeline=self.nlp,
                    dataset=dataset,
                    text_column=text_column,
                    sample=True,
                )

                print(
                    f"Auto-determined optimal batch size: {batch_size}",
                    flush=True,
                )
            except ImportError as e:
                print(
                    "Error importing calculate_optimal_batch_size function: "
                    f"{e}",
                    flush=True,
                )
                # Fallback to a default batch size
                batch_size = 32
                print(
                    f"Falling back to default batch size: {batch_size}",
                    flush=True,
                )

            except Exception as e:
                print(f"Error in auto-determining batch size: {e}", flush=True)
                # Fallback to a default batch size
                batch_size = 32
                print(
                    f"Falling back to default batch size: {batch_size}",
                    flush=True,
                )

        # Process the entire dataset at once with the pipeline
        with torch.no_grad():
            try:
                # Pass the dataset column directly. This is an iterable view, not a list conversion.
                results = self.nlp(
                    inputs=dataset[text_column], batch_size=batch_size
                )  # Removed input_column kwarg
            except torch.cuda.OutOfMemoryError as oom:
                print(f"Out of memory error during NER processing: {oom}")
                torch.cuda.empty_cache()  # Clear the CUDA memory
                # Create empty results for error cases
                results = [[] for _ in range(len(dataset))]
            except Exception as e:
                print(f"Error in NER processing: {e}")
                # Create empty results for error cases
                results = [[] for _ in range(len(dataset))]

        # Add predictions to the dataset
        dataset = dataset.add_column("prediction", results)
        if "prediction" not in dataset.column_names:
            raise ValueError(
                "Failed to add predictions to the dataset. Check the pipeline output."
            )
        return dataset


if __name__ == "__main__":
    model_dir = "../../rafsan/models/biobert_pytorch_pretrained/"
    model_name = "HunFlair_chemical_all/"
    seq = "he is feeing very sick nitrous oxide NO nucleus eucaryotic A549 HeLa Cells oxygen."

    NER = NER_biobert(model_dir=model_dir, model_name=model_name)

    print(NER.predict(seq))
