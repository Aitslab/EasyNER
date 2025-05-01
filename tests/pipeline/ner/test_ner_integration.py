import pytest
import torch
from unittest.mock import patch, MagicMock, ANY
import json
import os
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from easyner.pipeline.ner.ner_main import NERPipeline, NERProcessorFactory
from easyner.pipeline.ner.transformer_based.ner_biobert import (
    BioBertNERProcessor,
)


class TestNERIntegration:
    """Integration tests for the NER pipeline components."""

    @pytest.fixture
    def setup_test_environment(self, tmp_path):
        """Create a test environment with sample data files."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create sample test files
        files = []
        for i in range(2):
            file_path = input_dir / f"batch-{i}.json"
            with open(file_path, "w") as f:
                # Create simple valid JSON test data
                data = {
                    f"article-{i}-1": {
                        "pmid": f"test-{i}-1",
                        "title": f"Test Article {i}-1",
                        "abstract": f"This is test abstract {i}-1 with some entities.",
                        "sentences": [
                            {
                                "text": f"This is sentence 1 in article {i}-1.",
                                "entity_spans": [],
                                "entities": [],
                            },
                            {
                                "text": f"This is sentence 2 in article {i}-1.",
                                "entity_spans": [],
                                "entities": [],
                            },
                        ],
                    },
                    f"article-{i}-2": {
                        "pmid": f"test-{i}-2",
                        "title": f"Test Article {i}-2",
                        "abstract": f"This is test abstract {i}-2 with some entities.",
                        "sentences": [
                            {
                                "text": f"This is sentence 1 in article {i}-2.",
                                "entity_spans": [],
                                "entities": [],
                            }
                        ],
                    },
                }
                json.dump(data, f)
            files.append(str(file_path))

        return {
            "files": files,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        }

    @pytest.fixture
    def mock_ner_components(self):
        """Mock the NER components for testing without requiring actual model loading."""
        with (
            patch(
                "transformers.AutoModelForTokenClassification"
            ) as mock_model_class,
            patch("transformers.AutoTokenizer") as mock_tokenizer_class,
            patch("transformers.pipeline") as mock_pipeline_fn,
        ):

            # Set up mocks for the model and tokenizer
            mock_model = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model

            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Important: configure the pipeline mock to handle transformers framework detection
            # This prevents the "Could not infer framework from class <class 'unittest.mock.MagicMock'>" error
            mock_pipeline_instance = MagicMock()

            # Mock the pipeline function to return our pipeline instance
            def mock_pipeline_creator(**kwargs):
                return mock_pipeline_instance

            mock_pipeline_fn.side_effect = mock_pipeline_creator

            # Configure the pipeline instance to return entity predictions
            def predict_entities(inputs, batch_size=None):
                """Generate mock entity predictions for test data."""
                if isinstance(inputs, list):
                    return [
                        [
                            {
                                "entity_group": "CHEMICAL",
                                "score": 0.95,
                                "word": "entity1",
                                "start": 10,
                                "end": 17,
                            },
                            {
                                "entity_group": "DISEASE",
                                "score": 0.87,
                                "word": "entity2",
                                "start": 25,
                                "end": 32,
                            },
                        ]
                        for _ in inputs
                    ]
                else:
                    # Single input case
                    return [
                        {
                            "entity_group": "CHEMICAL",
                            "score": 0.95,
                            "word": "entity1",
                            "start": 10,
                            "end": 17,
                        },
                        {
                            "entity_group": "DISEASE",
                            "score": 0.87,
                            "word": "entity2",
                            "start": 25,
                            "end": 32,
                        },
                    ]

            mock_pipeline_instance.side_effect = predict_entities

            yield {
                "model": mock_model_class.from_pretrained,
                "tokenizer": mock_tokenizer_class.from_pretrained,
                "pipeline": mock_pipeline_fn,
                "pipeline_instance": mock_pipeline_instance,
                "model_instance": mock_model,
                "tokenizer_instance": mock_tokenizer,
            }

    def test_processor_factory_integration(self):
        """Test that the NER processor factory creates the correct processor type."""
        # Test biobert processor creation
        config = {"model_type": "biobert_finetuned"}
        processor = NERProcessorFactory.create_processor(config)
        assert isinstance(processor, BioBertNERProcessor)

        # Test with an invalid processor type
        with pytest.raises(ValueError):
            NERProcessorFactory.create_processor(
                {"model_type": "invalid_model_type"}
            )

    def test_full_pipeline_execution(
        self, setup_test_environment, mock_ner_components
    ):
        """Test that the complete pipeline executes from start to finish."""
        input_dir = setup_test_environment["input_dir"]
        output_dir = setup_test_environment["output_dir"]

        # Create configuration
        config = {
            "model_type": "biobert_finetuned",
            "model_folder": "/dummy/path",
            "model_name": "dummy_model",
            "input_path": input_dir + "/",
            "output_path": output_dir + "/",
            "output_file_prefix": "processed",
            "clear_old_results": True,
            "multiprocessing": False,
            "batch_size": 16,
        }

        # Create and run pipeline with proper mocks
        with (
            patch("easyner.util.append_to_json_file") as mock_append_json,
            patch.object(
                BioBertNERProcessor, "_initialize_model"
            ) as mock_init,
            patch.object(
                BioBertNERProcessor, "_process_single_file", return_value=0
            ) as mock_process,
        ):

            pipeline = NERPipeline(config)
            pipeline.run()

            # Verify initialization was called
            mock_init.assert_called_once()

            # Verify output was saved for each file
            assert mock_process.call_count == len(
                setup_test_environment["files"]
            )

    def test_biobert_dataset_processing(self, mock_ner_components):
        """Test that the BioBertNERProcessor correctly processes datasets."""
        # Create processor
        config = {
            "model_type": "biobert_finetuned",
            "model_folder": "/dummy/path",
            "model_name": "dummy_model",
        }

        processor = BioBertNERProcessor(config)

        # Patch the model initialization
        with patch.object(processor, "_initialize_model"):
            # Create a mock NLP pipeline for the processor with predictable outputs
            mock_nlp = MagicMock()
            mock_nlp.side_effect = lambda inputs, batch_size: [
                [
                    {
                        "word": "entity1",
                        "start": 10,
                        "end": 17,
                        "score": 0.95,
                        "entity_group": "CHEMICAL",
                    },
                    {
                        "word": "entity2",
                        "start": 25,
                        "end": 32,
                        "score": 0.87,
                        "entity_group": "DISEASE",
                    },
                ]
                for _ in inputs
            ]
            processor.nlp = mock_nlp

            # Create test dataset
            from datasets import Dataset

            test_data = {
                "text": [
                    "This is a sample text with entities like aspirin and diabetes.",
                    "Another example mentioning paracetamol and cancer.",
                ]
            }

            dataset = Dataset.from_dict(test_data)

            # Process dataset and verify results
            processed_dataset = processor._predict_dataset(
                dataset, "text", batch_size=8
            )

            # Verify the structure of the processed dataset
            assert "prediction" in processed_dataset.column_names
            assert len(processed_dataset["prediction"]) == len(
                test_data["text"]
            )

            # Check that predictions contain the expected entity structure
            for prediction in processed_dataset["prediction"]:
                assert isinstance(prediction, list)
                assert (
                    len(prediction) == 2
                )  # Based on our mock, each input gets 2 entities
                assert "word" in prediction[0]
                assert "start" in prediction[0]
                assert "end" in prediction[0]
                assert "score" in prediction[0]
                assert prediction[0]["word"] == "entity1"
                assert prediction[1]["word"] == "entity2"

    def test_multiprocessing_execution(
        self, setup_test_environment, mock_ner_components
    ):
        """Test pipeline execution with multiprocessing enabled."""
        input_dir = setup_test_environment["input_dir"]
        output_dir = setup_test_environment["output_dir"]

        # Create configuration with multiprocessing
        config = {
            "model_type": "biobert_finetuned",
            "model_folder": "/dummy/path",
            "model_name": "dummy_model",
            "input_path": input_dir + "/",
            "output_path": output_dir + "/",
            "clear_old_results": True,
            "multiprocessing": True,  # Enable multiprocessing
            "cpu_limit": 2,
        }

        # Create a mock processor that implements the multiprocessing behavior
        class MockMultiprocessingProcessor(BioBertNERProcessor):
            def process_dataset(self, input_files, device=None):
                # Directly use the multiprocessing implementation
                self._process_files_in_parallel(input_files)

            def _process_files_in_parallel(self, input_files):
                self.multiprocessing_called = True
                self.processed_files = input_files
                # Do nothing else - we're just checking if this method is called

        # Use our mock processor
        mock_processor = MockMultiprocessingProcessor(config)

        # Run pipeline with our mock processor
        with patch(
            "easyner.pipeline.ner.ner_main.NERProcessorFactory.create_processor",
            return_value=mock_processor,
        ):

            pipeline = NERPipeline(config)
            pipeline.run()

            # Verify the multiprocessing path was called
            assert (
                mock_processor.multiprocessing_called
            ), "Multiprocessing path was not called"

    def test_article_range_filtering(
        self, setup_test_environment, mock_ner_components
    ):
        """Test processing with article range filtering."""
        input_dir = setup_test_environment["input_dir"]
        output_dir = setup_test_environment["output_dir"]

        # Create configuration with article range filters
        config = {
            "model_type": "biobert_finetuned",
            "model_folder": "/dummy/path",
            "model_name": "dummy_model",
            "input_path": input_dir + "/",
            "output_path": output_dir + "/",
            "article_limit": [1, 1],  # Process only batch-1.json
            "clear_old_results": True,
        }

        # Create a custom processor class with tracking
        class TrackedBioBertProcessor(BioBertNERProcessor):
            def process_dataset(self, input_files, device=None):
                self.processed_files = input_files

        # Create our mocked processor
        mock_processor = TrackedBioBertProcessor(config)

        # Patch the factory to use our custom processor
        with patch(
            "easyner.pipeline.ner.ner_main.NERProcessorFactory.create_processor",
            return_value=mock_processor,
        ):
            # Run pipeline with the actual file system
            pipeline = NERPipeline(config)
            pipeline.run()

            # Verify only the specified batch was processed
            expected_file = next(
                f
                for f in setup_test_environment["files"]
                if "batch-1.json" in f
            )
            assert hasattr(
                pipeline.processor, "processed_files"
            ), "Expected processed_files attribute"
            assert (
                len(pipeline.processor.processed_files) == 1
            ), f"Expected exactly one file to be processed but got {pipeline.processor.processed_files}"
            assert (
                pipeline.processor.processed_files[0] == expected_file
            ), f"Expected {expected_file} to be processed but got {pipeline.processor.processed_files[0]}"
