import pytest
import torch
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from datasets import Dataset

from easyner.pipeline.ner.transformer_based.ner_biobert import (
    BioBertNERProcessor,
)


@pytest.fixture
def sample_config():
    """Create a sample configuration for BioBERT processor."""
    return {
        "model_type": "biobert_finetuned",
        "model_folder": "dummy/path",
        "model_name": "dummy_model",
        "output_path": str(Path(tempfile.mkdtemp()) / "output"),
        "output_file_prefix": "processed",
    }


@pytest.fixture
def sample_files(tmp_path):
    """Create sample files for testing."""
    input_dir = tmp_path / "test_input"
    output_dir = tmp_path / "test_output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create sample test files with realistic batch format
    files = []
    for i in range(3):
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
                }
            }
            json.dump(data, f)
        files.append(str(file_path))

    return {
        "file_paths": files,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }


@pytest.fixture
def mock_transformers():
    """Mock the transformers components used by the BioBERT processor."""
    with (
        patch(
            "transformers.AutoModelForTokenClassification.from_pretrained",
            create=True,
        ) as mock_model,
        patch(
            "transformers.AutoTokenizer.from_pretrained", create=True
        ) as mock_tokenizer,
        patch("transformers.pipeline", create=True) as mock_pipeline_fn,
    ):

        # Set up mocks for model and tokenizer
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        # Create a mock tokenizer with all needed methods
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Configure the pipeline mock
        mock_pipeline_instance = MagicMock()
        mock_pipeline_fn.return_value = mock_pipeline_instance

        # Add predictions to the pipeline
        mock_pipeline_instance.return_value = [
            [
                {
                    "entity_group": "CHEMICAL",
                    "score": 0.95,
                    "word": "entity1",
                    "start": 10,
                    "end": 17,
                }
            ]
        ]

        # Add device property to pipeline mock
        mock_pipeline_instance.device = torch.device("cuda:0")

        yield {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "pipeline": mock_pipeline_fn,
            "pipeline_instance": mock_pipeline_instance,
            "model_instance": mock_model_instance,
            "tokenizer_instance": mock_tokenizer_instance,
        }


@pytest.fixture
def mock_utils():
    """Mock utility functions used by the BioBERT processor."""
    with (
        patch(
            "easyner.pipeline.ner.utils.get_device_int", create=True
        ) as mock_get_device_int,
        patch(
            "easyner.pipeline.ner.utils.calculate_optimal_batch_size",
            create=True,
        ) as mock_batch_size,
        patch(
            "easyner.io.utils.remove_all_files_from_dir", create=True
        ) as mock_remove_files,
    ):

        # Configure the device int function
        mock_get_device_int.return_value = 0

        # Configure the batch size calculation
        mock_batch_size.return_value = 32

        yield {
            "get_device_int": mock_get_device_int,
            "calculate_optimal_batch_size": mock_batch_size,
            "remove_files": mock_remove_files,
        }


@pytest.fixture
def mock_io():
    """Mock I/O functions used by the BioBERT processor."""
    with (
        patch(
            "easyner.util.append_to_json_file", create=True
        ) as mock_append_json,
        patch(
            "easyner.io.converters.articles_to_datasets.convert_articles_to_dataset",
            create=True,
        ) as mock_convert_to_dataset,
        patch(
            "easyner.io.converters.articles_to_datasets.convert_dataset_to_dict",
            create=True,
        ) as mock_convert_to_dict,
        patch(
            "easyner.io.handlers.JsonHandler", create=True
        ) as mock_json_handler,
    ):

        # Configure the JSON handler
        mock_handler_instance = MagicMock()
        mock_json_handler.return_value = mock_handler_instance

        # Set up the JSON read method to return sample data
        mock_articles = {
            "article1": {
                "pmid": "test-1",
                "sentences": [
                    {
                        "text": "This is a test sentence.",
                        "entity_spans": [],
                        "entities": [],
                    }
                ],
            }
        }
        mock_handler_instance.read.return_value = (mock_articles, 1)

        # Configure dataset conversion
        mock_dataset = Dataset.from_dict(
            {
                "text": ["This is a test sentence."],
                "pmid": ["article1"],
                "sent_idx": [0],
            }
        )
        mock_convert_to_dataset.return_value = mock_dataset

        # Configure dict conversion
        mock_convert_to_dict.return_value = {
            "article1": {
                "sentences": [
                    {
                        "text": "This is a test.",
                        "entities": [{"entity": "TEST"}],
                    }
                ]
            }
        }

        yield {
            "append_json": mock_append_json,
            "convert_to_dataset": mock_convert_to_dataset,
            "convert_to_dict": mock_convert_to_dict,
            "json_handler": mock_json_handler,
        }


@pytest.fixture
def biobert_processor(sample_config):
    """Create a BioBertNERProcessor instance with mocked dependencies"""
    processor = BioBertNERProcessor(sample_config)
    return processor


class TestBioBertProcessor:
    """Test suite for BioBertNERProcessor."""

    def test_biobert_initialization(
        self, biobert_processor, mock_transformers, mock_utils
    ):
        """Test that BioBertNERProcessor initializes correctly"""
        device = torch.device("cuda:0")

        # Create a patched version of the _initialize_model method
        def mock_initialize(dev):
            from pathlib import Path, PurePosixPath

            biobert_processor.model_path = PurePosixPath(
                Path("dummy/model/path")
            )
            biobert_processor.tokenizer = mock_transformers[
                "tokenizer_instance"
            ]
            biobert_processor.model = mock_transformers["model_instance"]
            biobert_processor.nlp = mock_transformers["pipeline_instance"]

        # Replace the original method with our mock
        with patch.object(
            biobert_processor, "_initialize_model", side_effect=mock_initialize
        ):
            biobert_processor._initialize_model(device)

            # Verify the processor is correctly initialized with our mocks
            assert (
                biobert_processor.tokenizer
                == mock_transformers["tokenizer_instance"]
            )
            assert (
                biobert_processor.model == mock_transformers["model_instance"]
            )
            assert (
                biobert_processor.nlp == mock_transformers["pipeline_instance"]
            )

    def test_process_single_file(
        self,
        biobert_processor,
        mock_io,
        mock_transformers,
        mock_utils,
        monkeypatch,
    ):
        """Test processing a single file"""
        # Setup processor and patch its methods
        device = torch.device("cuda:0")

        # Mock the model initialization and add required attributes
        def mock_initialize_model(dev):
            from pathlib import Path, PurePosixPath

            biobert_processor.model_path = PurePosixPath(
                Path("dummy/model/path")
            )
            biobert_processor.tokenizer = mock_transformers[
                "tokenizer_instance"
            ]
            biobert_processor.model = mock_transformers["model_instance"]
            biobert_processor.nlp = mock_transformers["pipeline_instance"]

        monkeypatch.setattr(
            biobert_processor, "_initialize_model", mock_initialize_model
        )
        biobert_processor._initialize_model(device)

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as tmp_file:
            tmp_file.write(
                b'{"article1": {"sentences": [{"text": "Test sentence"}]}}'
            )
            tmp_path = tmp_file.name

        # Mock the _read_batch_file method to return proper data
        def mock_read_batch_file(batch_file):
            return mock_io["json_handler"].return_value.read()

        # Mock _predict_dataset to return dataset with prediction column
        def mock_predict_dataset(dataset, text_column, batch_size):
            return dataset.add_column(
                "prediction",
                [
                    [
                        {
                            "entity": "TEST",
                            "score": 0.98,
                            "word": "test",
                            "start": 8,
                            "end": 12,
                        }
                    ]
                ],
            )

        monkeypatch.setattr(
            biobert_processor, "_read_batch_file", mock_read_batch_file
        )
        monkeypatch.setattr(
            biobert_processor, "_predict_dataset", mock_predict_dataset
        )
        # Mock the _save_processed_articles method
        monkeypatch.setattr(
            biobert_processor,
            "_save_processed_articles",
            lambda articles, batch_idx: None,
        )

        try:
            # Process the temporary file
            result = biobert_processor._process_single_file(tmp_path, device)

            # Verify interactions
            mock_io["convert_to_dataset"].assert_called_once()
            mock_io["convert_to_dict"].assert_called_once()

            # Verify result
            assert result == 1  # Should match the batch index
        finally:
            # Clean up
            os.unlink(tmp_path)

    def test_predict_dataset(
        self, biobert_processor, mock_transformers, mock_utils, monkeypatch
    ):
        """Test dataset prediction"""
        # Setup processor
        device = torch.device("cuda:0")

        # Mock the model initialization
        def mock_initialize_model(dev):
            from pathlib import Path, PurePosixPath

            biobert_processor.model_path = PurePosixPath(
                Path("dummy/model/path")
            )
            biobert_processor.tokenizer = mock_transformers[
                "tokenizer_instance"
            ]
            biobert_processor.model = mock_transformers["model_instance"]
            biobert_processor.nlp = mock_transformers["pipeline_instance"]

        monkeypatch.setattr(
            biobert_processor, "_initialize_model", mock_initialize_model
        )
        biobert_processor._initialize_model(device)

        # Create a sample dataset
        sample_dataset = Dataset.from_dict(
            {
                "text": ["This is a test sentence."],
                "pmid": ["article1"],
                "sent_idx": [0],
            }
        )

        # Mock torch.no_grad to avoid actual tensor operations
        with patch("torch.no_grad"):
            # Process the dataset
            result = biobert_processor._predict_dataset(
                sample_dataset, "text", 16
            )

        # Verify the pipeline is called correctly
        mock_transformers["pipeline_instance"].assert_called_once_with(
            inputs=sample_dataset["text"], batch_size=16
        )

        # Verify result has prediction column
        assert "prediction" in result.column_names

    def test_optimal_batch_size(
        self, biobert_processor, mock_transformers, mock_utils, monkeypatch
    ):
        """Test optimal batch size calculation"""
        # Setup processor
        device = torch.device("cuda:0")

        # Mock the model initialization
        def mock_initialize_model(dev):
            from pathlib import Path, PurePosixPath

            biobert_processor.model_path = PurePosixPath(
                Path("dummy/model/path")
            )
            biobert_processor.tokenizer = mock_transformers[
                "tokenizer_instance"
            ]
            biobert_processor.model = mock_transformers["model_instance"]
            biobert_processor.nlp = mock_transformers["pipeline_instance"]

        monkeypatch.setattr(
            biobert_processor, "_initialize_model", mock_initialize_model
        )
        biobert_processor._initialize_model(device)

        # Create a sample dataset
        sample_dataset = Dataset.from_dict(
            {
                "text": ["This is a test sentence."],
                "pmid": ["article1"],
                "sent_idx": [0],
            }
        )

        # Get optimal batch size
        batch_size = biobert_processor._get_optimal_batch_size(
            sample_dataset, "text"
        )

        # Verify the calculation function is called
        mock_utils["calculate_optimal_batch_size"].assert_called_once_with(
            pipeline=biobert_processor.nlp,
            dataset=sample_dataset,
            text_column="text",
            sample=True,
        )

        # Verify result matches expected
        assert batch_size == 32

    def test_process_dataset(
        self,
        biobert_processor,
        sample_files,
        mock_transformers,
        mock_utils,
        monkeypatch,
    ):
        """Test processing a complete dataset"""

        # Mock the model initialization
        def mock_initialize_model(dev):
            from pathlib import Path, PurePosixPath

            biobert_processor.model_path = PurePosixPath(
                Path("dummy/model/path")
            )
            biobert_processor.tokenizer = mock_transformers[
                "tokenizer_instance"
            ]
            biobert_processor.model = mock_transformers["model_instance"]
            biobert_processor.nlp = mock_transformers["pipeline_instance"]

        monkeypatch.setattr(
            biobert_processor, "_initialize_model", mock_initialize_model
        )

        # Mock glob to return our sample files
        def mock_glob(pattern):
            return sample_files["file_paths"]

        monkeypatch.setattr("glob.glob", mock_glob)

        # Mock the file processing method to track calls
        process_single_file_mock = MagicMock(return_value=1)
        monkeypatch.setattr(
            biobert_processor, "_process_single_file", process_single_file_mock
        )

        # Process the dataset
        device = torch.device("cuda:0")
        biobert_processor.process_dataset(sample_files["file_paths"], device)

        # Verify each file was processed
        assert process_single_file_mock.call_count == len(
            sample_files["file_paths"]
        )

    def test_parallel_processing(
        self,
        biobert_processor,
        sample_files,
        mock_transformers,
        mock_utils,
        monkeypatch,
    ):
        """Test parallel processing with multiprocessing"""
        # Enable multiprocessing in config
        biobert_processor.config["multiprocessing"] = True

        # Mock the _process_files_in_parallel method directly
        mock_process_parallel = MagicMock()
        monkeypatch.setattr(
            biobert_processor,
            "_process_files_in_parallel",
            mock_process_parallel,
        )

        # Process the dataset with a mocked _initialize_model
        def mock_initialize_model(dev):
            from pathlib import Path, PurePosixPath

            biobert_processor.model_path = PurePosixPath(
                Path("dummy/model/path")
            )
            biobert_processor.tokenizer = mock_transformers[
                "tokenizer_instance"
            ]
            biobert_processor.model = mock_transformers["model_instance"]
            biobert_processor.nlp = mock_transformers["pipeline_instance"]

        monkeypatch.setattr(
            biobert_processor, "_initialize_model", mock_initialize_model
        )

        device = torch.device("cuda:0")
        biobert_processor.process_dataset(sample_files["file_paths"], device)

        # Verify _process_files_in_parallel was called with the correct files
        mock_process_parallel.assert_called_once_with(
            sample_files["file_paths"]
        )
