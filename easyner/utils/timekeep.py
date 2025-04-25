import functools
from pathlib import Path
from contextlib import contextmanager
import time

# Constants
DEFAULT_TIMEKEEP_FILE = "timekeep.txt"


class TimingManager:
    """Manages timing measurement for the pipeline."""

    def __init__(self, enabled=False, output_file=None):
        self.enabled = enabled
        self.timings = {}
        self.start_time = time.time() if enabled else None

        # Set up output file if timing is enabled
        if enabled and output_file:
            self.output_file = Path(output_file)
            self.output_file.parent.mkdir(exist_ok=True, parents=True)

            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(f"start_time at: {self.start_time}\n")
        else:
            self.output_file = None

    def record_timing(self, module_name, elapsed):
        """Record timing for a module."""
        self.timings[module_name] = elapsed

        if self.output_file:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(f"{module_name.capitalize()} time: {elapsed}\n")

                # Special case for NER module
                if module_name == "ner" and self.start_time:
                    current_time = time.time()
                    f.write(
                        f"Total time till NER: {current_time - self.start_time}\n"
                    )

    @contextmanager
    def measure(self, module_name):
        """Context manager to measure execution time of a block."""
        if not self.enabled:
            yield
            return

        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            elapsed = end - start
            self.record_timing(module_name, elapsed)

    def finalize(self):
        """Write final timing information."""
        if not self.enabled or not self.output_file or not self.start_time:
            return

        end_time = time.time()
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(f"end_time at: {end_time}\n")
            f.write(f"Total runtime: {end_time - self.start_time}\n")


def timed_execution(method):
    """Decorator for timing module execution."""

    @functools.wraps(method)
    def timed(self, *args, **kwargs):
        timer = kwargs.get("timer")

        # If no timer or timing disabled, just run the method
        if not timer or not timer.enabled:
            return method(self, *args, **kwargs)

        with timer.measure(self.name):
            return method(self, *args, **kwargs)

    return timed
