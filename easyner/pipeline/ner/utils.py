from typing import Union, Optional, List, Dict, Any
import torch
from datasets import Dataset
import time
import numpy as np
import random
import warnings  #


def get_device_int(device: Union[int, torch.device]) -> int:
    """Convert device specifier to integer for pipeline"""
    # Convert device input to integer format expected by pipeline
    pipeline_device_int = -1  # Default to CPU
    if isinstance(device, torch.device):
        if device.type == "cuda":
            pipeline_device_int = (
                device.index if device.index is not None else 0
            )
        else:  # 'cpu'
            pipeline_device_int = -1
    elif isinstance(device, str):
        if "cuda" in device:
            try:
                # Extract index from "cuda:X"
                pipeline_device_int = int(device.split(":")[-1])
            except (ValueError, IndexError):
                pipeline_device_int = (
                    0  # Default to cuda:0 if format is unexpected
                )
        elif "cpu" in device:
            pipeline_device_int = -1
        else:  # Assume it might be an integer string
            try:
                pipeline_device_int = int(device)
            except ValueError:
                print(
                    f"Warning: Could not parse device string '{device}', defaulting to CPU (-1)."
                )
                pipeline_device_int = -1
    elif isinstance(device, int):
        pipeline_device_int = device  # Already an int
    else:
        print(
            f"Warning: Unknown device type '{type(device)}', defaulting to CPU (-1)."
        )
        pipeline_device_int = -1
    return pipeline_device_int


def get_available_gpus() -> List[int]:
    """
    Detect available GPUs on the system.

    Returns:
    --------
    List[int]: List of available GPU device IDs
    """
    if not torch.cuda.is_available():
        return []

    # Do a more thorough check to ensure all GPUs are detected
    try:
        # Force CUDA initialization to detect all GPUs
        torch.cuda.init()

        # Get actual device count
        num_devices = torch.cuda.device_count()

        # First try a simple detection approach
        if num_devices > 0:
            # Just return all device IDs without individual testing
            # This is more reliable in some environments where detailed probing can fail
            gpu_ids = list(range(num_devices))
            print(f"Detected {len(gpu_ids)} GPU(s) using device count method")
            return gpu_ids

        # Fall back to thorough checking only if the simple approach returns 0
        # Double-check by trying to query each device
        available_gpus = []
        for i in range(num_devices):
            try:
                # Check if we can access the device properties
                props = torch.cuda.get_device_properties(i)
                # Create a small test tensor to verify the device works
                test_tensor = torch.zeros(1, device=f"cuda:{i}")
                del test_tensor
                # Add this GPU to available list
                available_gpus.append(i)
                print(
                    f"GPU {i} ({props.name}) is available with {props.total_memory / (1024**3):.1f} GB memory"
                )
            except Exception as e:
                print(f"GPU {i} exists but is not usable: {e}")

        print(
            f"Found {len(available_gpus)} usable GPU(s) out of {num_devices} detected"
        )
        return available_gpus
    except Exception as e:
        print(f"Error detecting available GPUs: {e}")
        # Fall back to simple count as last resort
        try:
            count = torch.cuda.device_count()
            return list(range(count))
        except:
            print("Could not determine GPU count. Assuming no GPUs available.")
            return []


def calculate_optimal_batch_size(
    pipeline,
    dataset: Dataset,
    text_column: str = "text",
    sample: bool = False,
    max_batch_size: Optional[int] = None,
    start_batch=2,
    min_improvement_percentage=2,
):
    """
    Dynamically determine optimal batch size based on GPU memory and processing speed.

    Args:
        pipeline: HuggingFace pipeline object
        sample_text: Representative text sample
        device: Device to use (cuda:0, etc)
        max_batch_size: Upper limit to try
        start_batch: Starting batch size

    Returns:
        int: Optimal batch size
    """
    # Suppress the specific UserWarning from transformers.pipelines.base during this function
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You seem to be using the pipelines sequentially on GPU.",
            category=UserWarning,
            module="transformers.pipelines.base",
        )

        CPU_BATCH_SIZE = 32

        # Force testing of a minimum range of batch sizes before considering plateaus
        MIN_BATCH_SIZE_TO_TEST = 512
        device = pipeline.device
        # Skip optimization on CPU
        if device.type == "cpu":
            return CPU_BATCH_SIZE

        if isinstance(device, int) and device < 0:
            return CPU_BATCH_SIZE

        test_input_iterable = (
            None  # Variable to hold the correct input for the pipeline
        )

        if sample:  # Use actual text samples for testing
            indices = random.sample(
                range(len(dataset)), min(2000, len(dataset))
            )
            test_batch_dataset = dataset.select(indices)
            test_input_iterable = test_batch_dataset[
                text_column
            ]  # Use the column slice
            test_batch_len = len(
                test_batch_dataset
            )  # Need length for throughput calculation
        else:  # Use a sample text of average length for memory estimation
            text_lengths = [
                len(t) for t in dataset[text_column]
            ]  # Access column directly
            avg_length = sum(text_lengths) / len(text_lengths)
            sample_idx = min(
                range(len(dataset)),
                key=lambda i: abs(
                    len(dataset[text_column][i]) - avg_length
                ),  # Access column directly
            )
            sample_text = dataset[text_column][
                sample_idx
            ]  # Access column directly

            # Create a representative batch with varying lengths
            short_text = sample_text[: len(sample_text) // 2]
            long_text = sample_text * 2

            test_batch_list = []
            for _ in range(
                10
            ):  # Create a larger, more representative test batch
                test_batch_list.extend(
                    [short_text] * 33 + [sample_text] * 34 + [long_text] * 33
                )
            test_input_iterable = test_batch_list  # Use the list directly
            test_batch_len = len(
                test_batch_list
            )  # Need length for throughput calculation

        batch_sizes = []
        throughputs = []
        memory_usages = []

        # Start with smallest batch size and increase
        batch_size = start_batch

        print("Finding optimal batch size...")

        # Warm up CUDA before measurements
        try:
            # Use the prepared iterable input, remove input_column
            pipeline(test_input_iterable, batch_size=start_batch)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"Error during warm-up: {e}")
            pass

        while max_batch_size is None or batch_size <= max_batch_size:
            # Clear cache between tests
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            try:
                # Test process time
                # Run each batch size test 3 times and take average
                times = []
                for _ in range(3):
                    start_time = time.time()
                    # Use the prepared iterable input, remove input_column
                    pipeline(test_input_iterable, batch_size=batch_size)
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                elapsed = sum(times) / len(times)  # Average time

                # Calculate throughput (samples/second)
                # Use the stored length
                throughput = test_batch_len / elapsed

                # Get memory stats
                memory_used = torch.cuda.max_memory_allocated() / (
                    1024**3
                )  # GB

                print(
                    f"  Batch size {batch_size}: {throughput:.1f} samples/sec, {memory_used:.2f}GB"
                )

                batch_sizes.append(batch_size)
                throughputs.append(throughput)
                memory_usages.append(memory_used)

                # Add memory threshold check

                # Get the correct device index
                device_idx = 0
                if isinstance(device, torch.device) and device.type == "cuda":
                    device_idx = (
                        device.index if device.index is not None else 0
                    )
                elif isinstance(device, str) and "cuda:" in device:
                    try:
                        device_idx = int(device.split(":")[-1])
                    except (ValueError, IndexError):
                        device_idx = 0
                elif isinstance(device, int) and device >= 0:
                    device_idx = device

                total_memory = torch.cuda.get_device_properties(
                    device_idx
                ).total_memory

                memory_utilization = (
                    memory_used * (1024**3)
                ) / total_memory  # More precise
                print(f"    Memory utilization: {memory_utilization:.1%}")

                # Stop if memory usage exceeds 80%
                if memory_utilization > 0.8:
                    # Ensure safe_batch_size is at least start_batch
                    safe_batch_size = max(batch_size // 2, start_batch)
                    print(
                        f"  ⚠️ Memory utilization ({memory_utilization:.1%}) exceeded 80% threshold"
                    )
                    print(
                        f"  SELECTING BATCH SIZE {safe_batch_size} (half of {batch_size}, min {start_batch}) for safety"
                    )
                    return safe_batch_size

                if len(throughputs) > 2:
                    # Calculate improvement percentage
                    # Don't trigger plateau detection until we've tested sufficiently large batches
                    if batch_size < MIN_BATCH_SIZE_TO_TEST:
                        pass  # Skip plateau detection for small batch sizes
                    else:
                        # Existing plateau detection logic
                        improvement = (
                            throughputs[-1] / throughputs[-2] - 1
                        ) * 100

                        # More robust plateau detection - require THREE consecutive below-threshold improvements
                        if (
                            improvement < min_improvement_percentage
                            and len(throughputs) >= 5
                        ):
                            # Check if we've had THREE consecutive below-threshold improvements
                            if len(throughputs) > 4:
                                previous_improvement1 = (
                                    throughputs[-2] / throughputs[-3] - 1
                                ) * 100
                                previous_improvement2 = (
                                    throughputs[-3] / throughputs[-4] - 1
                                ) * 100

                                if (
                                    previous_improvement1
                                    < min_improvement_percentage
                                    and previous_improvement2
                                    < min_improvement_percentage
                                ):
                                    # NOW we've confirmed a plateau with THREE consecutive suboptimal improvements
                                    max_throughput_idx = np.argmax(throughputs)
                                    best_batch_size = batch_sizes[
                                        max_throughput_idx
                                    ]
                                    print(
                                        f"  📊 Throughput plateaued: consecutive improvements {previous_improvement2:.1f}% → {previous_improvement1:.1f}% → {improvement:.1f}% below threshold ({min_improvement_percentage}%)"
                                    )
                                    print(
                                        f"  SELECTING BATCH SIZE {best_batch_size} (best throughput: {throughputs[max_throughput_idx]:.1f} samples/sec)"
                                    )
                                    return best_batch_size

                # Increase batch size exponentially
                batch_size *= 2

            # In the RuntimeError exception handler
            except RuntimeError as e:
                # Ensure safe_batch_size is at least start_batch
                safe_batch_size = max(batch_size // 2, start_batch)
                print(f"  🛑 Error at batch size {batch_size}: {str(e)}")
                print(
                    f"  SELECTING BATCH SIZE {safe_batch_size} (avoiding runtime error, min {start_batch})"
                )
                return safe_batch_size

        # If we reach max_batch_size without OOM, find best throughput/memory trade-off
        # using the "knee point" of throughput vs batch size
        throughputs_arr = np.array(throughputs)
        normalized_throughputs = throughputs_arr / throughputs_arr.max()

        # Simple heuristic: batch size where we get 90% of max throughput
        threshold = 0.9
        above_threshold = np.where(normalized_throughputs >= threshold)[0]
        if len(above_threshold) > 0:
            optimal_idx = above_threshold[
                0
            ]  # First batch size that gives good throughput
            max_throughput = throughputs_arr.max()
            threshold_value = max_throughput * threshold
            print(
                f"  📈 Selected smallest batch size reaching {threshold:.0%} of max throughput"
            )
            print(
                f"      Threshold: {threshold_value:.1f} samples/sec ({threshold:.0%} of {max_throughput:.1f})"
            )
            print(
                f"  SELECTING BATCH SIZE {batch_sizes[optimal_idx]} (efficiency threshold reached)"
            )
            return batch_sizes[optimal_idx]

        # If all else fails, return the batch size with highest throughput
        max_throughput_idx = np.argmax(throughputs)
        best_batch_size = batch_sizes[max_throughput_idx]
        print(f"  ✅ Tested up to maximum batch size {max_batch_size}")
        print(
            f"  SELECTING BATCH SIZE {best_batch_size} (highest throughput: {throughputs[max_throughput_idx]:.1f} samples/sec)"
        )
        return best_batch_size
