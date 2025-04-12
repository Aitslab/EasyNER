from abc import ABC, abstractmethod


class TokenizerBase(ABC):
    @abstractmethod
    def split_text(self, text):
        pass


class SpacyTokenizer(TokenizerBase):
    def __init__(self, model_name="en_core_web_sm"):
        import spacy

        self.model_name = model_name
        self.nlp = spacy.load(model_name)

    def split_text(self, text):
        doc = self.nlp(text)
        return [str(sentence) for sentence in doc.sents]


class NLTKTokenizer(TokenizerBase):
    def __init__(self):
        from nltk.tokenize import sent_tokenize

        self.sent_tokenize = sent_tokenize

    def split_text(self, text):
        return self.sent_tokenize(text)
