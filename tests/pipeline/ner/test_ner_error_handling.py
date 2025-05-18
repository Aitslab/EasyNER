import pytest
import torch
from unittest.mock import patch, MagicMock
from datasets import Dataset
import json
import os
from pathlib import Path

from easyner.pipeline.ner.ner_main import NERPipeline
from easyner.pipeline.ner.transformer_based.ner_biobert import (
    BioBertNERProcessor,
)


class TestNERErrorHandling:
    """Test suite for error handling in NER components."""

    @pytest.fixture
    def base_config(self, tmp_path):
        """Base configuration for error handling tests with safe temp paths."""
        output_dir = tmp_path / "output"
        input_dir = tmp_path / "input"
        output_dir.mkdir()
        input_dir.mkdir()

        return {
            "model_type": "biobert_finetuned",
            "model_folder": str(tmp_path / "model_folder"),
            "model_name": "dummy_model",
            "input_path": str(input_dir) + "/",
            "output_path": str(output_dir) + "/",
            "clear_old_results": True,
        }

    @pytest.fixture
    def create_pipeline(self):
        """Factory fixture for creating a NERPipeline with mocked components."""

        def _factory(config, mock_device=None):
            with patch(
                "easyner.pipeline.ner.ner_main.NERProcessorFactory"
            ) as factory_mock:
                # Mock the processor created by the factory
                processor_mock = MagicMock()
                factory_mock.create_processor.return_value = processor_mock

                # Mock torch.device if provided
                if mock_device:
                    with patch("torch.device", return_value=mock_device):
                        pipeline = NERPipeline(config)
                else:
                    pipeline = NERPipeline(config)

                return pipeline, processor_mock

        return _factory

    def test_missing_input_path(self, base_config):
        """Test handling of missing input path."""
        # Remove input path from config
        config = base_config.copy()
        del config["input_path"]

        # Verify pipeline raises appropriate error with key error info
        with pytest.raises(KeyError, match="input_path"):
            with patch("os.makedirs"):  # Avoid file system operations
                pipeline = NERPipeline(config)
                pipeline.run()

    def test_missing_output_path(self, base_config):
        """Test handling of missing output path."""
        # Remove output path from config
        config = base_config.copy()
        del config["output_path"]

        # Verify pipeline raises appropriate error
        with pytest.raises(KeyError):
            with patch("os.makedirs"):  # Avoid file system operations
                pipeline = NERPipeline(config)
                pipeline.run()

    def test_empty_input_directory(self, base_config, create_pipeline):
        """Test handling of an empty input directory."""
        # Create pipeline with mocked components
        pipeline, mock_processor = create_pipeline(
            base_config, mock_device="cuda:0"
        )

        # Mock glob.glob to return empty list (no files)
        with patch("easyner.pipeline.ner.ner_main.glob") as mock_glob:
            mock_glob.return_value = []

            # Run pipeline (should not crash)
            pipeline.run()

            # Verify processor.process_dataset should NOT be called with empty list
            # Our implementation now skips processing when there are no input files
            mock_processor.process_dataset.assert_not_called()

    def test_malformed_input_file(self, tmp_path):
        """Test behavior with malformed JSON input file."""
        input_dir = tmp_path / "malformed_input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create malformed JSON file
        malformed_file = input_dir / "batch-1.json"
        with open(malformed_file, "w") as f:
            f.write("{this is not valid json")

        # Create config with temp dirs
        config = {
            "model_type": "biobert_finetuned",
            "model_folder": str(tmp_path / "model_folder"),
            "model_name": "dummy_model",
            "input_path": str(input_dir) + "/",
            "output_path": str(output_dir) + "/",
        }

        # Create processor
        processor = BioBertNERProcessor(config)

        # Test with mock initialize_model to avoid actual model loading
        with patch.object(processor, "_initialize_model"):
            # Currently, JsonHandler errors propagate all the way up
            # This test verifies the current behavior - that errors are not caught
            with pytest.raises(ValueError) as excinfo:
                processor._process_single_file(
                    str(malformed_file), torch.device("cuda:0")
                )

            # Verify the error message contains details about JSON decoding
            assert "Error decoding JSON" in str(
                excinfo.value
            ) or "JSONDecodeError" in str(excinfo.value)

    def test_cuda_out_of_memory(self):
        """Test handling of CUDA out of memory errors during processing."""
        # Create processor with minimal config
        config = {"model_type": "biobert_finetuned"}
        processor = BioBertNERProcessor(config)

        # Mock the NLP pipeline
        processor.nlp = MagicMock()
        processor.nlp.side_effect = torch.cuda.OutOfMemoryError(
            "CUDA out of memory"
        )

        # Create test dataset
        dataset = Dataset.from_dict(
            {"text": ["Test sentence" for _ in range(10)]}
        )

        # Process dataset (should handle OOM)
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            result_dataset = processor._predict_dataset(dataset, "text", 32)

            # Should clear CUDA cache
            mock_empty_cache.assert_called_once()

            # Should return dataset with empty predictions
            assert "prediction" in result_dataset.column_names
            assert all(len(pred) == 0 for pred in result_dataset["prediction"])

    def test_prediction_errors(self):
        """Test handling of general errors during prediction."""
        # Create processor with minimal config
        config = {"model_type": "biobert_finetuned"}
        processor = BioBertNERProcessor(config)

        # Mock the NLP pipeline to raise an error
        processor.nlp = MagicMock()
        processor.nlp.side_effect = RuntimeError("Some random error")

        # Create test dataset
        dataset = Dataset.from_dict(
            {"text": ["Test sentence" for _ in range(10)]}
        )

        # Process dataset (should handle the error)
        with patch("easyner.pipeline.ner.transformer_based.ner_biobert.print"):
            result_dataset = processor._predict_dataset(dataset, "text", 32)

            # Should return dataset with empty predictions
            assert "prediction" in result_dataset.column_names
            assert all(len(pred) == 0 for pred in result_dataset["prediction"])

    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        # Create processor with minimal config
        config = {"model_type": "biobert_finetuned"}
        processor = BioBertNERProcessor(config)

        # Mock the NLP pipeline
        processor.nlp = MagicMock()

        # Create empty dataset
        empty_dataset = Dataset.from_dict({"text": []})

        # Process empty dataset with print suppressed
        with patch("easyner.pipeline.ner.transformer_based.ner_biobert.print"):
            result_dataset = processor._predict_dataset(
                empty_dataset, "text", 32
            )

            # For empty datasets, we should check that it's still empty
            # and doesn't throw an exception, rather than checking for columns
            assert len(result_dataset) == 0
            # An empty dataset might not have prediction column added, and that's OK

    @patch(
        "easyner.pipeline.ner.utils.calculate_optimal_batch_size",
        side_effect=Exception("Test error"),
    )
    def test_batch_size_calculation_failure(self, mock_calc, base_config):
        """Test handling of failures during batch size calculation."""
        # Create processor with dependencies
        processor = BioBertNERProcessor(base_config)

        # Mock the processor's NLP pipeline
        processor.nlp = MagicMock()

        # Create test dataset
        dataset = Dataset.from_dict({"text": ["Test sentence"]})

        with patch("easyner.pipeline.ner.transformer_based.ner_biobert.print"):
            # Should fall back to default batch size
            batch_size = processor._get_optimal_batch_size(dataset, "text")
            assert batch_size == 32  # Default fallback value
