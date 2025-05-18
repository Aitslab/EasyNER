"""Tokenizers for sentence segmentation using spaCy and NLTK.

This module provides abstract and concrete classes for splitting text into sentences,
supporting both single and batch processing with spaCy and NLTK backends.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable  # Added List, Tuple

# Consider adding spacy types if using the custom component and type hinting it
# from spacy.tokens import Doc
# from spacy.language import Language


class TokenizerBase(ABC):
    """Abstract base class for text tokenizers."""

    # --- Capability Class Attributes ---
    SUPPORTS_BATCH_PROCESSING: bool = False
    SUPPORTS_BATCH_GENERATOR: bool = False

    @abstractmethod
    def segment_sentences(self, text: str) -> list[str]:
        """Split a single text into sentences."""
        pass

    # @abstractmethod
    # def get_sentences_char_spans(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:  # noqa: E501
    #     """
    #     Splits text into sentences and returns their character spans.
    #     This is usefull for more efficient data strorage where the original text
    #     can be kept, while the spans are used to access the sentences.

    #     Args:
    #         text: The text document to split.

    #     Returns:
    #         A list of tuples, where each tuple contains the sentence text
    #         and a (start_char, end_char) tuple indicating its position.
    #     """
    #     pass

    # Optional batch method - subclasses can override if they support it
    def segment_sentences_batch(self, texts: Iterable[str]) -> list[list[str]]:
        """Split multiple texts into sentences in batch mode."""
        # Default implementation raises error if called but not overridden
        msg = f"{self.__class__.__name__} does not support batch processing."
        raise NotImplementedError(msg)

    # Optional batch generator method - subclasses can override
    def segment_sentences_batch_generator(
        self,
        texts: Iterable[str],
    ) -> Iterable[list[str]]:
        """Split multiple texts into sentences, yielding results incrementally."""
        # Default implementation raises error if called but not overridden
        msg = f"{self.__class__.__name__} does not support batch generator processing."
        raise NotImplementedError(
            msg,
        )


class SpacyTokenizer(TokenizerBase):
    """Tokenizer using spaCy for sentence splitting.

    Attributes:
        model_name (str): The name of the spaCy model to use.
        nlp (spacy.Language): The loaded spaCy language pipeline.

    """

    # --- Capability Class Attributes (Override) ---
    SUPPORTS_BATCH_PROCESSING: bool = True
    SUPPORTS_BATCH_GENERATOR: bool = True

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Initialize the SpacyTokenizer.

        Args:
            model_name (str): The spaCy model to load (e.g., "en_core_web_sm").
                              Defaults to "en_core_web_sm".

        """
        try:
            import spacy

            # If using custom component defined outside:
            # from .custom_components import fix_enumeration_boundaries
        except ImportError as e:
            msg = (
                "SpacyTokenizer requires spaCy. "
                "Install with: "
                "pip install spacy && python -m spacy download en_core_web_sm"
            )
            raise ImportError(msg) from e

        self.model_name = model_name
        # Load only components needed for sentence segmentation
        self.nlp = spacy.load(
            model_name,
            exclude=["ner", "attribute_ruler", "lemmatizer"],
        )

        self.nlp.batch_size = 20

    def segment_sentences(self, text: str) -> list[str]:
        """Split text into sentences using the spaCy pipeline."""
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def segment_sentences_batch(self, texts: Iterable[str]) -> list[list[str]]:
        """Process multiple documents efficiently in a batch using spaCy's nlp.pipe.

        Args:
            texts: An iterable of text documents to process.

        Returns:
            A list of lists, where each inner list contains the sentences
            from the corresponding input document.

        """
        docs = list(self.nlp.pipe(texts))
        return [[sent.text for sent in doc.sents] for doc in docs]

    def segment_sentences_batch_generator(
        self,
        texts: Iterable[str],
    ) -> Iterable[list[str]]:
        """Process multiple documents efficiently, yielding results incrementally.

        Processes documents in batches using spaCy's nlp.pipe but yields
        the list of sentences for each document one by one.

        Args:
            texts: An iterable of text documents to process.

        Yields:
            A list of sentences for each processed document.

        """
        for doc in self.nlp.pipe(texts):
            yield [sent.text for sent in doc.sents]
            del doc  # Explicitly delete the doc to free memory

    def segment_sentences_generator(self, text: str) -> Iterable[str]:
        """Split a single large text into sentences, yielding them one by one.

        More memory-efficient for very large texts than segmenting_sentences().

        Args:
            text: The text document to split.

        Yields:
            Each sentence from the text.

        """
        doc = self.nlp(text)
        for sent in doc.sents:
            yield sent.text


class NLTKTokenizer(TokenizerBase):
    """Tokenizer using NLTK's punkt sentence tokenizer."""

    def __init__(self) -> None:
        """Initialize the NLTKTokenizer."""
        try:
            import nltk
            from nltk.tokenize import sent_tokenize

            # TODO: Check that this works without internet access
            # This checks if they are up to date, and ifso skips the download
            nltk.download("punkt")
            nltk.download("punkt_tab")
        except ImportError as e:
            msg = "NLTKTokenizer requires NLTK. Install with: pip install nltk"
            raise ImportError(msg) from e

        self.sent_tokenize = sent_tokenize

    def segment_sentences(self, text: str) -> list[str]:
        """Split text into sentences using NLTK's sent_tokenize."""
        # Add basic error handling or text type check if needed
        if not isinstance(text, str):
            # Or handle appropriately, e.g., return empty list, log warning
            msg = "Input text must be a string."
            raise TypeError(msg)
        return self.sent_tokenize(text)
