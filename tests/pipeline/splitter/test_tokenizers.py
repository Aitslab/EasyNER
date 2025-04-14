import pytest
import sys
import os

from easyner.pipeline.splitter.tokenizers import TokenizerBase, SpacyTokenizer, NLTKTokenizer


class TestTokenizers:

    def test_tokenizer_base_abstract(self):
        """Test that TokenizerBase is abstract and cannot be instantiated directly"""
        with pytest.raises(TypeError):
            TokenizerBase()

    @pytest.mark.parametrize(
        "text,expected_count",
        [
            ("This is a test. This is another test.", 2),
            ("Single sentence without period", 1),
            ("Hello world! How are you? I'm fine.", 3),
            ("", 0),  # Empty string test
            ("Mr. Smith went to Washington, D.C. yesterday.", 1),  # Test abbreviations
        ],
    )
    def test_spacy_tokenizer_sentence_count(self, text, expected_count):
        """Test that SpacyTokenizer correctly identifies the number of sentences"""
        try:
            tokenizer = SpacyTokenizer()
            sentences = tokenizer.segment_sentences(text)
            assert len(sentences) == expected_count
        except ImportError:
            pytest.skip("spaCy not installed or en_core_web_sm model not available")

    @pytest.mark.parametrize(
        "text,expected_count",
        [
            ("This is a test. This is another test.", 2),
            ("Single sentence without period", 1),
            ("Hello world! How are you? I'm fine.", 3),
            ("", 0),  # Empty string test
            ("Mr. Smith went to Washington, D.C. yesterday.", 1),  # Test abbreviations
        ],
    )
    def test_nltk_tokenizer_sentence_count(self, text, expected_count):
        """Test that NLTKTokenizer correctly identifies the number of sentences"""
        try:
            tokenizer = NLTKTokenizer()
            sentences = tokenizer.segment_sentences(text)
            assert len(sentences) == expected_count
        except ImportError:
            pytest.skip("NLTK not installed or punkt not available")

    def test_spacy_tokenizer_batch_support(self):
        """Test that SpacyTokenizer correctly supports batch processing"""
        try:
            tokenizer = SpacyTokenizer()
            assert tokenizer.SUPPORTS_BATCH_PROCESSING is True

            # Test batch processing
            texts = ["This is sentence one. This is sentence two.", "Another document here."]
            results = tokenizer.segment_sentences_batch(texts)

            assert len(results) == 2
            assert len(results[0]) == 2  # First document has 2 sentences
            assert len(results[1]) == 1  # Second document has 1 sentence

            # Test batch generator
            assert tokenizer.SUPPORTS_BATCH_GENERATOR is True
            gen_results = list(tokenizer.segment_sentences_batch_generator(texts))
            assert len(gen_results) == 2
            assert len(gen_results[0]) == 2
            assert len(gen_results[1]) == 1

        except ImportError:
            pytest.skip("spaCy not installed or en_core_web_sm model not available")

    def test_tokenizer_content(self):
        """Test that tokenizers preserve content correctly"""
        text = "This is a complete sentence. Another one here!"
        try:
            spacy_tokenizer = SpacyTokenizer()
            sentences = spacy_tokenizer.segment_sentences(text)
            assert sentences[0].strip() == "This is a complete sentence."
            assert sentences[1].strip() == "Another one here!"
        except ImportError:
            pytest.skip("spaCy not installed or en_core_web_sm model not available")

        try:
            nltk_tokenizer = NLTKTokenizer()
            sentences = nltk_tokenizer.segment_sentences(text)
            assert sentences[0].strip() == "This is a complete sentence."
            assert sentences[1].strip() == "Another one here!"
        except ImportError:
            pytest.skip("NLTK not installed or punkt not available")
