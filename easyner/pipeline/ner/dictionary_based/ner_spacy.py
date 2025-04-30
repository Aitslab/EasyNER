# coding=utf-8

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

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
        import spacy
        from spacy.matcher import PhraseMatcher

        if not self.config.get("multiprocessing", False):
            spacy.prefer_gpu()  # Should ideally be called before importing spacy and loading any pipelines

        self.nlp = spacy.load("en_core_web_sm")  # Default model
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        terms = self._load_vocabulary()

        # To create the patterns, each phrase has to be processed with the nlp object. If you have a trained pipeline loaded, doing this in a loop or list comprehension can easily become inefficient and slow. If you only need the tokenization and lexical attributes, you can run nlp.make_doc instead, which will only run the tokenizer. For an additional speed boost, you can also use the nlp.tokenizer.pipe method, which will process the texts as a stream.

        # TODO: Implement choice between full nlp processing and nlp.make_doc/nlp.tokenizer.pipe

        # patterns = [self.nlp.make_doc(term) for term in terms]

        patterns = list(
            self.nlp.tokenizer.pipe(terms)
        )  # Faster will process the texts as a stream.
        self.matcher.add(self.config["entity_type"], patterns)

    def _load_vocabulary(self) -> List[str]:
        """Load vocabulary terms from the specified file."""
        terms = []
        with open(self.config["vocab_path"], "r") as f:
            for line in f:
                term = line.strip()
                if term:  # Skip empty lines
                    terms.append(term)
        return terms

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

        processed_articles = self._process_articles(articles, batch_index)

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

    def _process_articles(self, articles: Dict, batch_index: int) -> Dict:
        """
        Run NER with spacy PhraseMatcher
        """

        # Run prediction on each sentence in each article.
        for pmid in tqdm(articles, desc=f"batch:{batch_index}"):
            sentences = articles[pmid]["sentences"]

            # Predict with spacy PhraseMatcher, if it has been selected

            for i, sentence in enumerate(sentences):
                doc = self.nlp(sentence["text"])

                if self.config.get("store_tokens"):
                    tokens = [token.text for token in doc]
                    articles[pmid]["sentences"][i]["tokens"] = tokens

                entities = []
                spans = []
                matches = self.matcher(doc)

                for match_id, start, end in matches:
                    span = doc[start:end]
                    ent = span.text
                    entities.append(ent)
                    first_char = span.start_char
                    last_char = span.end_char - 1
                    spans.append((first_char, last_char))

                articles[pmid]["sentences"][i]["entities"] = entities
                articles[pmid]["sentences"][i]["entity_spans"] = spans

        return articles


if __name__ == "__main__":
    pass
