# coding=utf-8

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List
import spacy
from spacy.matcher import PhraseMatcher
from tqdm import tqdm

from easyner.pipeline.ner.processor import NERProcessor


class SpacyNERProcessor(NERProcessor):
    """NER processor using SpaCy's phrase matcher."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Load spaCy model once for all processing
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the spaCy model once for all processing."""
        # Implementation-specific initialization
        pass

    def process_dataset(
        self, input_files: List[str], device: Any = None
    ) -> None:
        """
        Process all files using SpaCy's phrase matcher.

        Parameters:
        -----------
        input_files: List[str]
            List of input file paths to process
        device: Any, optional
            Device to use for processing (not used for SpaCy)
        """

        # Process files sequentially or in parallel based on configuration
        if self.config.get("multiprocessing", False):
            self._process_files_in_parallel(input_files)
        else:
            for batch_file in tqdm(input_files, desc="Processing with SpaCy"):
                self._process_single_file(batch_file)

    def _process_single_file(self, batch_file: str) -> int:
        """Process a single file with SpaCy NER."""

        articles, batch_index = self._read_batch_file(batch_file)

        if not articles:
            self._save_processed_articles(articles, batch_index)
            return batch_index

        processed_articles = run_ner_with_spacy_phrasematcher(
            articles, self.config, batch_index
        )

        self._save_processed_articles(processed_articles, batch_index)
        return batch_index

    def _process_files_in_parallel(self, input_files: List[str]) -> None:
        """Process files in parallel using multiprocessing."""
        from multiprocessing import cpu_count

        cpu_limit = self.config.get("cpu_limit", 1)

        with ProcessPoolExecutor(min(cpu_limit, cpu_count())) as executor:
            futures = [
                executor.submit(self._process_single_file, batch_file)
                for batch_file in input_files
            ]

            for i, future in enumerate(as_completed(futures)):
                batch_index = future.result()
                print(
                    f"Completed SpaCy batch {batch_index} ({i+1}/{len(futures)})"
                )


def run_ner_with_spacy_phrasematcher(articles, ner_config, batch_index):
    """
    Run NER with spacy PhraseMatcher
    """
    if not ner_config["multiprocessing"]:
        spacy.prefer_gpu()

    print("Running NER with spacy")
    nlp = spacy.load(ner_config["model_name"])
    terms = []
    with open(ner_config["vocab_path"], "r") as f:
        for line in f:
            x = line.strip()
            terms.append(x)
    print("Phraselist complete")

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(term) for term in terms]
    matcher.add(ner_config["entity_type"], patterns)

    # Run prediction on each sentence in each article.
    for pmid in tqdm(articles, desc=f"batch:{batch_index}"):
        sentences = articles[pmid]["sentences"]

        # Predict with spacy PhraseMatcher, if it has been selected

        for i, sentence in enumerate(sentences):
            ner_class = ner_config["entity_type"]

            doc = nlp(sentence["text"])
            if ner_config["store_tokens"] == "yes":
                tokens = []
                # tokens_idxs = []  #uncomment if you want a list of token character offsets within the sentence
                for token in doc:
                    tokens.append(
                        token.text
                    )  # to get a list of tokens in the sentence
                # tokens_idxs.append(token.idx) #uncomment if you want a list of token character offsets within the sentence
                articles[pmid]["sentences"][i]["tokens"] = tokens

            entities = []
            spans = []
            matches = matcher(doc)

            for match_id, start, end in matches:
                span = doc[start:end]
                ent = span.text
                entities.append(ent)
                first_char = span.start_char
                last_char = span.end_char - 1
                spans.append((first_char, last_char))

            # articles[pmid]["sentences"][i]["NER class"] = ner_class
            articles[pmid]["sentences"][i]["entities"] = entities
            articles[pmid]["sentences"][i]["entity_spans"] = spans
    return articles


@PendingDeprecationWarning
def run_ner_with_spacy(model_name, vocab_path, entity_type, sentences):

    # Prepare spacy, if it is needed
    print("Running NER with spacy")
    nlp = spacy.load(model_name)

    terms = []
    with open(vocab_path) as f:
        for line in f:
            x = line.strip()
            terms.append(x)

    print("Phraselist complete")

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(term) for term in terms]
    matcher.add(entity_type, patterns)

    for i, sentence in enumerate(sentences):
        ner_class = entity_type

        doc = nlp(sentences["text"])
        if store_tokens == "yes":
            tokens = []
            # tokens_idxs = []  #uncomment if you want a list of token character offsets within the sentence
            for token in doc:
                tokens.append(
                    token.text
                )  # to get a list of tokens in the sentence
            # tokens_idxs.append(token.idx) #uncomment if you want a list of token character offsets within the sentence
            articles[pmid]["sentences"][i]["tokens"] = tokens  # type: ignore

        entities = []
        spans = []
        matches = matcher(doc)

        for match_id, start, end in matches:
            span = doc[start:end]
            ent = span.text
            entities.append(ent)
            first_char = span.start_char
            last_char = span.end_char - 1
            spans.append((first_char, last_char))

        articles[pmid]["sentences"][i]["NER class"] = ner_class
        articles[pmid]["sentences"][i]["entities"] = entities
        articles[pmid]["sentences"][i]["entity spans"] = spans


if __name__ == "__main__":
    pass
