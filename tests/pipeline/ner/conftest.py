import pytest
import torch
import json
from unittest.mock import patch, MagicMock, PropertyMock
from datasets import Dataset


@pytest.fixture
def mock_cuda_setup():
    """Setup mocks for CUDA and related functionality"""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.empty_cache") as mock_empty_cache,
    ):
        yield {"empty_cache": mock_empty_cache}


@pytest.fixture
def mock_transformers():
    """Mock all transformer components"""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch(
            "transformers.AutoModelForTokenClassification.from_pretrained"
        ) as mock_model,
        patch("transformers.pipeline") as mock_pipeline,
    ):
        # Configure model mock
        model_instance = MagicMock()
        model_instance.__class__.__module__ = "torch.nn.modules.module"
        model_instance.__class__.__name__ = "PreTrainedModel"
        model_instance.device = torch.device("cuda:0")
        model_instance.eval = MagicMock(return_value=model_instance)
        type(model_instance).hf_device_map = PropertyMock(return_value=None)
        mock_model.return_value = model_instance

        # Configure pipeline mock
        pipeline_instance = MagicMock()
        pipeline_instance.return_value = [{}]
        mock_pipeline.return_value = pipeline_instance

        # Configure tokenizer mock
        tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = tokenizer_instance

        yield {
            "tokenizer": mock_tokenizer,
            "model": mock_model,
            "pipeline": mock_pipeline,
            "model_instance": model_instance,
            "pipeline_instance": pipeline_instance,
        }


@pytest.fixture
def mock_utils():
    """Mock utility functions"""
    with (
        patch("easyner.pipeline.ner.utils.get_device_int") as mock_get_device,
        patch(
            "easyner.pipeline.ner.utils.calculate_optimal_batch_size",
            create=True,
        ) as mock_calc,
        patch("easyner.io.utils.extract_batch_index") as mock_extract_index,
        patch(
            "easyner.io.utils._remove_all_files_from_dir"
        ) as mock_remove_files,
    ):
        mock_get_device.return_value = 0
        mock_calc.return_value = 32
        mock_extract_index.return_value = 1
        yield {
            "get_device_int": mock_get_device,
            "calculate_optimal_batch_size": mock_calc,
            "extract_batch_index": mock_extract_index,
            "remove_files": mock_remove_files,
        }


@pytest.fixture
def mock_io():
    """Mock I/O operations"""
    with (
        patch("easyner.io.handlers.JsonHandler") as mock_json_handler,
        patch("easyner.util.append_to_json_file") as mock_append_json,
        patch(
            "easyner.io.converters.articles_to_datasets.convert_articles_to_dataset"
        ) as mock_convert_to_dataset,
        patch(
            "easyner.io.converters.articles_to_datasets.convert_dataset_to_dict"
        ) as mock_convert_to_dict,
    ):
        # Configure mock JSON handler
        handler_instance = MagicMock()
        handler_instance.read.return_value = {
            "article1": {"sentences": [{"text": "Test sentence"}]}
        }
        mock_json_handler.return_value = handler_instance

        # Configure dataset conversion
        sample_dataset = Dataset.from_dict(
            {
                "text": ["This is a test sentence."],
                "pmid": ["article1"],
                "sent_idx": [0],
            }
        )
        mock_convert_to_dataset.return_value = sample_dataset
        mock_convert_to_dict.return_value = {
            "article1": {
                "sentences": [{"text": "Test sentence", "entities": []}]
            }
        }

        yield {
            "json_handler": mock_json_handler,
            "append_json": mock_append_json,
            "convert_to_dataset": mock_convert_to_dataset,
            "convert_to_dict": mock_convert_to_dict,
        }


@pytest.fixture
def sample_config():
    """Sample configuration for NER processors"""
    return {
        "model_type": "biobert_finetuned",
        "model_folder": "/dummy/model/folder",
        "model_name": "dummy_model",
        "model_max_length": 192,
        "input_path": "/dummy/input/path/",
        "output_path": "/dummy/output/path",
        "output_file_prefix": "test_output",
        "clear_old_results": True,
        "multiprocessing": False,
        "batch_size": 16,
    }


@pytest.fixture
def sample_files(tmp_path):
    """Create sample files for testing"""
    # Create test directories
    input_dir = tmp_path / "test_input"
    input_dir.mkdir(exist_ok=True)
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)

    # Create sample files
    file_paths = []
    for i in range(3):
        sample_data = {
            f"article{i}": {
                "sentences": [
                    {"text": f"This is test sentence {i}."},
                    {"text": f"This is another test sentence {i}."},
                ]
            }
        }
        file_path = input_dir / f"batch-{i}.json"
        with open(file_path, "w") as f:
            json.dump(sample_data, f)
        file_paths.append(str(file_path))

    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "file_paths": file_paths,
    }
