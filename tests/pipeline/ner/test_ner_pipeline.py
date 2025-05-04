import pytest
from unittest.mock import patch, MagicMock


from easyner.pipeline.ner.ner_main import NERPipeline, run_ner_module


@pytest.fixture
def mock_processor_factory():
    """Mock the NERProcessorFactory"""
    with patch(
        "easyner.pipeline.ner.ner_main.NERProcessorFactory"
    ) as factory_mock:
        processor_mock = MagicMock()
        factory_mock.create_processor.return_value = processor_mock
        yield {"factory": factory_mock, "processor": processor_mock}


@pytest.fixture
def mock_glob_and_filesystem():
    """Mock glob and filesystem operations"""

    def patch_glob(monkeypatch, sample_files):
        def mock_glob(pattern):
            return sample_files["file_paths"]

        monkeypatch.setattr("easyner.pipeline.ner.ner_main.glob", mock_glob)

    def patch_os(monkeypatch):
        monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)

    return {"patch_glob": patch_glob, "patch_os": patch_os}


class TestNERPipeline:
    """Test suite for the NERPipeline class."""

    def test_pipeline_initialization(
        self, sample_config, mock_processor_factory
    ):
        """Test that NERPipeline initializes correctly"""
        pipeline = NERPipeline(sample_config, cpu_limit=4)

        # Verify CPU limit is set
        assert pipeline.config["cpu_limit"] == 4

        # Verify processor factory was called with config
        mock_processor_factory[
            "factory"
        ].create_processor.assert_called_once_with(sample_config)

    def test_input_file_sorting(self, sample_config, monkeypatch):
        """Test that input files are properly sorted by numeric indices"""

        # Mock the glob function to return files with numeric indices
        def mock_glob(pattern):
            return ["batch-10.json", "batch-2.json", "batch-1.json"]

        monkeypatch.setattr("easyner.pipeline.ner.ner_main.glob", mock_glob)
        monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)

        with patch("torch.device", return_value="mocked_device"):
            pipeline = NERPipeline(sample_config)
            files = pipeline._get_filtered_input_files()

            # Verify files are sorted by numeric index, not lexicographically
            assert files == [
                "batch-1.json",
                "batch-2.json",
                "batch-10.json",
            ]

    def test_input_file_sorting_error_handling(
        self, sample_config, monkeypatch
    ):
        """Test that sorting handles files without the expected numeric format"""

        # Mock the glob function to return files without numeric indices
        def mock_glob(pattern):
            return ["file1.json", "file2.json", "another-file.json"]

        monkeypatch.setattr("easyner.pipeline.ner.ner_main.glob", mock_glob)
        monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)

        with patch("torch.device", return_value="mocked_device"):
            pipeline = NERPipeline(sample_config)
            # Should fall back to lexicographical sorting without error
            files = pipeline._get_filtered_input_files()

            # Files should be sorted lexicographically
            assert files == [
                "another-file.json",
                "file1.json",
                "file2.json",
            ]

    def test_empty_input_directory(
        self, sample_config, monkeypatch, mock_processor_factory
    ):
        """Test handling of an empty input directory"""

        # Mock the glob function to return no files
        def mock_glob(pattern):
            return []

        monkeypatch.setattr("easyner.pipeline.ner.ner_main.glob", mock_glob)
        monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)

        with patch("torch.device", return_value="mocked_device"):
            pipeline = NERPipeline(sample_config)

            # Should return empty list without error
            files = pipeline._get_filtered_input_files()
            assert files == []

            # Running the pipeline with no files should not crash
            pipeline.run()
            # No processing should be done when no files exist
            assert not pipeline.processor.process_dataset.called

    def test_pipeline_run(
        self,
        sample_config,
        mock_processor_factory,
        mock_glob_and_filesystem,
        monkeypatch,
    ):
        """Test the complete pipeline run method"""
        # Apply mocks
        mock_glob_and_filesystem["patch_glob"](
            monkeypatch, {"file_paths": ["batch-1.json", "batch-2.json"]}
        )
        mock_glob_and_filesystem["patch_os"](monkeypatch)

        # Mock torch.device
        with patch("torch.device", return_value="mocked_device"):
            # Create and run pipeline
            pipeline = NERPipeline(sample_config)
            pipeline.run()

            # Verify processor was called correctly
            mock_processor_factory[
                "processor"
            ].process_dataset.assert_called_once()
            args, kwargs = mock_processor_factory[
                "processor"
            ].process_dataset.call_args
            assert len(args[0]) == 2  # Should have 2 files
            assert kwargs.get("device") == "mocked_device"

    def test_run_ner_module(self, sample_config, monkeypatch):
        """Test the legacy run_ner_module function"""
        # Mock the NERPipeline
        mock_pipeline = MagicMock()
        monkeypatch.setattr(
            "easyner.pipeline.ner.ner_main.NERPipeline",
            lambda config, cpu_limit: mock_pipeline,
        )

        # Call the legacy function
        run_ner_module(sample_config, 2)

        # Verify pipeline was created and run
        mock_pipeline.run.assert_called_once()
