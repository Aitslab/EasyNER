from typing import Union
import torch


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
