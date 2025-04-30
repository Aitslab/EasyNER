import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, call
from datasets import Dataset

from easyner.pipeline.ner.utils import (
    get_device_int,
    calculate_optimal_batch_size,
)


class TestNERUtils:
    """Test suite for NER utility functions."""

    @pytest.mark.parametrize(
        "device_input,expected_output",
        [
            (torch.device("cuda:0"), 0),
            (torch.device("cuda:1"), 1),
            (torch.device("cpu"), -1),
            ("cuda:0", 0),
            ("cuda:1", 1),
            ("cpu", -1),
            (0, 0),
            (1, 1),
            (-1, -1),
            ("0", 0),
            ("1", 1),
            ("invalid", -1),  # Should default to CPU
            (None, -1),  # Should default to CPU
        ],
    )
    def test_get_device_int(self, device_input, expected_output):
        """Test the get_device_int function with various device inputs."""
        result = get_device_int(device_input)
        assert (
            result == expected_output
        ), f"Expected {expected_output} for input {device_input}, but got {result}"

    def test_calculate_optimal_batch_size_cpu(self):
        """Test that CPU devices return a fixed batch size."""
        pipeline = MagicMock()
        pipeline.device = torch.device("cpu")

        dataset = Dataset.from_dict(
            {"text": ["Sample text 1", "Sample text 2"]}
        )

        with patch("builtins.print"):  # Suppress print outputs
            result = calculate_optimal_batch_size(pipeline, dataset)
            assert (
                result == 32
            ), "CPU devices should return a fixed batch size of 32"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.reset_peak_memory_stats")
    @patch("torch.cuda.max_memory_allocated")
    def test_calculate_optimal_batch_size_cuda_simple(
        self, mock_memory, mock_reset, mock_empty, mock_cuda_available
    ):
        """Test optimal batch size calculation with CUDA - simplified version."""
        # Setup mocks
        pipeline = MagicMock()
        pipeline.device = torch.device("cuda:0")

        # Configure memory usage (well below limits)
        mock_memory.return_value = 4 * (1024**3)  # 4GB

        # Mock device properties
        mock_device = MagicMock()
        mock_device.total_memory = 16 * (1024**3)  # 16GB total memory

        # We need many time values because the function calls time.time() multiple times
        time_values = []
        for i in range(100):  # Generate enough values for all calls
            time_values.append(i)

        # Call function
        with patch(
            "torch.cuda.get_device_properties", return_value=mock_device
        ):
            with patch("time.time", side_effect=time_values):
                with patch("builtins.print"):  # Suppress print outputs
                    # Create small dataset for quick testing
                    dataset = Dataset.from_dict(
                        {"text": ["Sample text 1", "Sample text 2"]}
                    )

                    # Pass a small max_batch_size to limit the number of iterations
                    result = calculate_optimal_batch_size(
                        pipeline, dataset, max_batch_size=32
                    )

                    # Just verify we get a reasonable batch size back
                    assert result > 0, "Should return a positive batch size"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.reset_peak_memory_stats")
    def test_calculate_optimal_batch_size_oom(
        self, mock_reset, mock_empty, mock_cuda_available
    ):
        """Test batch size calculation handles out-of-memory errors gracefully."""
        # Setup mocks
        pipeline = MagicMock()
        pipeline.device = torch.device("cuda:0")

        # Simulate OOM on second batch size
        def side_effect(*args, **kwargs):
            batch_size = kwargs.get("batch_size", 16)
            if batch_size >= 32:
                raise RuntimeError("CUDA out of memory")
            return [{}] * 10

        pipeline.side_effect = side_effect

        dataset = Dataset.from_dict(
            {"text": ["Sample text " * 10 for _ in range(10)]}
        )

        # Call the function with print suppressed
        with patch("builtins.print"):  # Suppress print outputs
            with patch(
                "time.time", side_effect=range(100)
            ):  # Mock time with many values
                result = calculate_optimal_batch_size(
                    pipeline, dataset, sample=True, max_batch_size=64
                )

                # Should fallback to half the failing batch size or start_batch
                assert result == 2, f"Expected 2 but got {result}"

    def test_optimal_batch_size_verbose_output(self):
        """Test that batch size calculation outputs informative progress messages."""
        # Test for CUDA device, since CPU devices exit early before printing
        pipeline = MagicMock()
        pipeline.device = torch.device(
            "cuda:0"
        )  # Use CUDA device to trigger prints

        dataset = Dataset.from_dict(
            {"text": ["Sample text 1", "Sample text 2"]}
        )

        # Collect print calls
        print_values = []

        def my_print(*args, **kwargs):
            print_values.append(args[0] if args else "")

        # Setup necessary mocks for CUDA
        with (
            patch("builtins.print", side_effect=my_print),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch(
                "torch.cuda.max_memory_allocated", return_value=1 * (1024**3)
            ),  # 1GB
            patch(
                "time.time", side_effect=range(100)
            ),  # Mock time with many values
            patch(
                "torch.cuda.get_device_properties",
                return_value=MagicMock(total_memory=16 * (1024**3)),
            ),
        ):
            # Call the function
            calculate_optimal_batch_size(pipeline, dataset, max_batch_size=32)

            # Check that "Finding optimal batch size..." was printed
            assert any(
                "Finding optimal batch size..." in msg for msg in print_values
            ), "Expected 'Finding optimal batch size...' message"
