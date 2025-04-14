import time
import logging
import tabulate
import statistics

# Get logger for this module
logger = logging.getLogger("easyner.pipeline.splitter.stats")


class StatsManager:
    """
    Manages statistics for the splitter pipeline.
    Centralizes statistics collection and reporting.
    """

    def __init__(self):
        """Initialize a statistics manager"""
        self.processing_times = []
        self.batch_sizes = []
        self.total_articles = 0
        self.total_batches = 0
        self.start_time = time.time()

    def update_stats(self, num_articles, processing_time, batch_size=None):
        """
        Update statistics with a new batch.

        Args:
            num_articles: Number of articles processed
            processing_time: Time taken to process the batch
            batch_size: Size of the batch
        """
        self.total_articles += num_articles
        self.total_batches += 1
        self.processing_times.append(processing_time)
        if batch_size:
            self.batch_sizes.append(batch_size)

    def get_elapsed_time(self):
        """
        Get elapsed time since start.

        Returns:
            float: Elapsed time in seconds
        """
        return time.time() - self.start_time

    def format_elapsed_time(self, seconds=None):
        """
        Format elapsed time as a human-readable string.

        Args:
            seconds: Time in seconds to format (uses elapsed time if None)

        Returns:
            str: Formatted time string
        """
        if seconds is None:
            seconds = self.get_elapsed_time()

        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"

    def get_summary(self):
        """
        Get a summary of statistics.

        Returns:
            dict: Statistics summary
        """
        elapsed = self.get_elapsed_time()
        speed = self.total_articles / elapsed if elapsed > 0 else 0

        summary = {
            "elapsed_time": elapsed,
            "formatted_time": self.format_elapsed_time(elapsed),
            "total_articles": self.total_articles,
            "total_batches": self.total_batches,
            "speed": speed,
        }

        # Add batch size stats if available
        if self.batch_sizes:
            summary.update(
                {
                    "avg_batch_size": statistics.mean(self.batch_sizes),
                    "min_batch_size": min(self.batch_sizes),
                    "max_batch_size": max(self.batch_sizes),
                }
            )

        # Add processing time stats if available
        if self.processing_times:
            summary.update(
                {
                    "avg_batch_time": statistics.mean(self.processing_times),
                    "min_batch_time": min(self.processing_times),
                    "max_batch_time": max(self.processing_times),
                    "median_batch_time": statistics.median(self.processing_times),
                }
            )

            # Add percentiles if we have enough data points
            if len(self.processing_times) >= 10:
                summary.update(
                    {
                        "p90_batch_time": statistics.quantiles(self.processing_times, n=10)[8],
                        "p95_batch_time": statistics.quantiles(self.processing_times, n=20)[18],
                    }
                )

        return summary

    def format_summary(self, worker_stats=None):
        """
        Get a formatted summary for display.

        Args:
            worker_stats: Optional worker statistics dictionary including 'peak_memory_mb'

        Returns:
            str: Formatted summary table
        """
        summary = self.get_summary()

        # Create basic summary data
        summary_data = [
            ["Total runtime", summary["formatted_time"]],
            ["Batches processed", f"{summary['total_batches']:,}"],
            ["Articles processed", f"{summary['total_articles']:,}"],
            ["Processing speed", f"{summary['speed']:.1f} articles/second"],
        ]

        # Add batch size info if available
        if "avg_batch_size" in summary:
            summary_data.append(
                [
                    "Average batch size",
                    f"{summary['avg_batch_size']:.1f} articles",
                ]
            )

        # Add processing time info if available
        if "avg_batch_time" in summary:
            summary_data.append(
                [
                    "Average batch time",
                    f"{summary['avg_batch_time']:.1f} seconds",
                ]
            )
            summary_data.append(
                [
                    "Median batch time",
                    f"{summary['median_batch_time']:.1f} seconds",
                ]
            )

        # Format as a nice table
        summary_table = tabulate.tabulate(
            summary_data, headers=["Metric", "Value"], tablefmt="simple"
        )

        result = f"\n===== PROCESSING SUMMARY =====\n{summary_table}\n============================="

        # Add worker stats if provided
        if worker_stats and len(worker_stats) > 0:
            worker_data = []
            # <-- Add 'Peak Mem (MiB)' to headers -->
            headers = [
                "Worker",
                "Batches",
                "Articles",
                "Time",
                "Art/s",
                "Peak Mem (MiB)",
            ]
            for worker_id, stats in sorted(
                worker_stats.items()
            ):  # Sort by worker_id for consistent order
                peak_mem = stats.get("peak_memory_mb", 0.0)  # Get peak memory
                worker_data.append(
                    [
                        f"Worker {worker_id}",
                        stats.get("batches", 0),
                        f"{stats.get('articles', 0):,}",  # Format articles with comma
                        f"{stats.get('time', 0):.1f}s",
                        (
                            f"{stats.get('articles', 0) / stats.get('time', 1):.1f}"
                            if stats.get("time", 0) > 0
                            else "N/A"
                        ),
                        f"{peak_mem:.1f}",  # Add formatted peak memory
                    ]
                )

            worker_table = tabulate.tabulate(
                worker_data,
                headers=headers,  # Use updated headers
                tablefmt="simple",
                floatfmt=".1f",  # Ensure floats are formatted nicely
            )

            result += (
                f"\n\n==== WORKER PERFORMANCE ====\n{worker_table}\n============================"
            )

        return result
