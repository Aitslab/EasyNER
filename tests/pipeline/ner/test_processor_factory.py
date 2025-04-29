import pytest
from unittest.mock import patch

from easyner.pipeline.ner.factory import NERProcessorFactory


class TestNERProcessorFactory:
    """Test suite for the NERProcessorFactory."""

    def test_create_spacy_processor(self):
        """Test creating a SpacyNERProcessor"""
        with patch(
            "easyner.pipeline.ner.dictionary_based.ner_spacy.SpacyNERProcessor"
        ) as mock_processor:
            config = {"model_type": "spacy_phrasematcher"}
            processor = NERProcessorFactory.create_processor(config)

            # Verify correct processor was created
            mock_processor.assert_called_once_with(config)
            assert processor == mock_processor.return_value

    def test_create_biobert_processor(self):
        """Test creating a BioBertNERProcessor"""
        with patch(
            "easyner.pipeline.ner.transformer_based.ner_biobert.BioBertNERProcessor"
        ) as mock_processor:
            config = {"model_type": "biobert_finetuned"}
            processor = NERProcessorFactory.create_processor(config)

            # Verify correct processor was created
            mock_processor.assert_called_once_with(config)
            assert processor == mock_processor.return_value

    def test_unknown_processor_type(self):
        """Test error handling for unknown processor type"""
        config = {"model_type": "unknown_type"}

        with pytest.raises(ValueError) as excinfo:
            NERProcessorFactory.create_processor(config)

        # Verify error message
        assert "Unknown model type" in str(excinfo.value)
