from typing import Dict, Any

from easyner.pipeline.ner.processor import NERProcessor


class NERProcessorFactory:
    """Factory class for creating appropriate NER processors."""

    @staticmethod
    def create_processor(config: Dict[str, Any]) -> NERProcessor:
        """
        Create an appropriate NER processor based on configuration.

        Parameters:
        -----------
        config: Dict[str, Any]
            Configuration for NER processing

        Returns:
        --------
        NERProcessor: The appropriate processor instance
        """
        model_type = config.get("model_type", "")

        if model_type == "spacy_phrasematcher":
            from easyner.pipeline.ner.dictionary_based.ner_spacy import (
                SpacyNERProcessor,
            )

            return SpacyNERProcessor(config)
        elif model_type == "biobert_finetuned":
            from easyner.pipeline.ner.transformer_based.ner_biobert import (
                BioBertNERProcessor,
            )

            return BioBertNERProcessor(config)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                "Supported types are 'spacy_phrasematcher' and 'biobert_finetuned'."
            )
