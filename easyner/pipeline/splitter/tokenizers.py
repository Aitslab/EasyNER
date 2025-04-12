from abc import ABC, abstractmethod


class TokenizerBase(ABC):
    @abstractmethod
    def split_text(self, text):
        pass


class SpacyTokenizer(TokenizerBase):
    def __init__(self, model_name="en_core_web_sm"):
        import spacy

        self.model_name = model_name
        # Load only components needed for sentence segmentation
        # Explicitly disable unnecessary components for performance
        self.nlp = spacy.load(
            model_name, exclude=["ner", "parser", "attribute_ruler", "lemmatizer"]
        )

        # Make sure the sentencizer is enabled
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

        # Set smaller batch size for small documents
        self.nlp.batch_size = 50

    def split_text(self, text):
        doc = self.nlp(text)
        return list(sent.text for sent in doc.sents)

    def split_texts_batch(self, texts):
        """
        Process multiple small documents efficiently in a batch.
        More efficient than processing documents individually when you have many small texts.

        Args:
            texts: List of text documents to process

        Returns:
            List of lists, where each inner list contains the sentences from one document
        """
        # Process all texts in an efficient batch
        docs = list(self.nlp.pipe(texts))

        # Extract sentences from each document
        return [[sent.text for sent in doc.sents] for doc in docs]

    def split_texts_batch_generator(self, texts):
        """
        Generator version of batch processing that yields sentences document by document.
        Still processes documents in batches for efficiency but yields results incrementally.

        Args:
            texts: List of text documents to process

        Yields:
            List of sentences for each document, one document at a time
        """
        for doc in self.nlp.pipe(texts):
            yield [sent.text for sent in doc.sents]

    def split_text_generator(self, text):
        """Generator version that's more memory efficient for large texts"""
        doc = self.nlp(text)
        for sent in doc.sents:
            yield sent.text

    def split_text_with_spans(self, text):
        """Return both the text and the character spans"""
        doc = self.nlp(text)
        return [(sent.text, (sent.start_char, sent.end_char)) for sent in doc.sents]


class NLTKTokenizer(TokenizerBase):
    def __init__(self):
        from nltk.tokenize import sent_tokenize

        self.sent_tokenize = sent_tokenize

    def split_text(self, text):
        return self.sent_tokenize(text)
