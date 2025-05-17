"""PubMed Bulk Downloader.

This module provides functionality to download PubMed XML files in bulk.

TODO: Doensn't yet properly handle config start and end values with consistent opened
closes ranges AND/OR null values.
"""

import asyncio
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from io import TextIOWrapper
from pathlib import Path
from typing import Any, ClassVar, Optional

import aiohttp
from pydantic import Field, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass
from tqdm import tqdm, trange

PUBMED_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/"
PUBMED_BASELINE_URL = f"{PUBMED_BASE_URL}baseline/"
PUBMED_UPDATE_URL = f"{PUBMED_BASE_URL}updatefiles/"

# Global set to track currently downloading files for cleanup on interruption
_downloading_files: set[Path] = set()


@asynccontextmanager
async def track_downloading_file(file_path: Path):
    """Context manager to track files being downloaded.

    Allows for cleanup of partial downloads on program termination.

    Args:
        file_path: Path of the file being downloaded

    Yields:
        None

    """
    try:
        _downloading_files.add(file_path)
        yield
    finally:
        _downloading_files.remove(file_path)


def cleanup_partial_downloads() -> None:
    """Remove any partially downloaded files when the program is interrupted."""
    if _downloading_files:
        print(f"\nCleaning up {len(_downloading_files)} incomplete downloads...")
        for file_path in _downloading_files:
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"Removed incomplete file: {file_path}")
                except Exception as e:
                    print(f"Failed to remove incomplete file {file_path}: {e}")


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful termination."""

    def signal_handler(sig, frame):
        print("\nInterrupt received, cleaning up and terminating...")
        cleanup_partial_downloads()
        print("Cleanup complete. Exiting.")
        sys.exit(1)

    # Register SIGINT (Ctrl+C) handler
    signal.signal(signal.SIGINT, signal_handler)
    # Register SIGTERM handler if on Unix-like systems
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)


@dataclass
class DownloaderConfig:
    """Configuration for downloading PubMed XML files.

    Attributes:
        baseline (int): The baseline year (0-99).
        n_start (int): The starting file number for baseline downloads.
        n_end (Optional[int]): The ending file number for baseline downloads.
        download_updates (bool): Whether to download nightly update files.
        u_start (Optional[int]): The starting file number for update downloads.
        u_end (Optional[int]): The ending file number for update downloads.
        save_path (str): The directory path to save downloaded files.
        skip_download (bool): Whether to skip the download process.
        skip_existing (bool): Whether to skip downloading files that already exist.

    Methods:
        from_config_dict(config: dict[str, Any]) -> "DownloaderConfig":
            Creates a DownloaderConfig instance from a configuration dictionary.

    """

    baseline: int = Field(..., ge=0, le=99)
    n_start: int = Field(0, ge=0)
    n_end: int | None = Field(None)
    download_updates: bool = Field(False)
    u_start: int | None = Field(None)
    u_end: int | None = Field(None)
    save_path: str = Field("data/tmp/pubmed/")
    skip_download: bool = Field(False)
    skip_existing: bool = Field(True)
    max_connections: int = Field(
        20,
        ge=1,
        description="Maximum number of parallel connections",
    )
    chunk_size: int = Field(
        1024 * 1024,
        ge=1024,
        description="Chunk size in bytes for streaming downloads",
    )

    # Default path values - useful for creating from dictionary
    DEFAULT_SAVE_PATH: ClassVar[str] = "data/tmp/pubmed/"

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> "DownloaderConfig":
        """Create a DownloaderConfig instance from a configuration dictionary.

        This class method is deprecated and only maintained for backwards compatibility.
        Please use the direct parameter initialization approach in
        run_pubmed_download and run_pubmed_updates_download functions instead.

        Args:
            config: The configuration dictionary from config.json

        Returns:
            A validated DownloaderConfig instance

        """
        # Issue a deprecation warning
        import warnings

        warnings.warn(
            "DownloaderConfig.from_config_dict is deprecated. Use direct parameter "
            "initialization in the run_pubmed_download or run_pubmed_updates_download "
            "functions instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Basic parameters
        params = {
            "baseline": config.get("baseline"),
            "save_path": (
                config.get("raw_download_path", "")
                if config.get("raw_download_path")
                and len(config.get("raw_download_path", "")) > 0
                else cls.DEFAULT_SAVE_PATH
            ),
            "download_updates": config.get("download_updates", False),
            "skip_download": config.get("skip_download", False),
        }

        # Handle baseline file range parameters
        if config.get("file_start") is not None:
            params["n_start"] = config["file_start"]
        if config.get("file_end") is not None:
            params["n_end"] = config["file_end"]

        # Handle update parameters
        if config.get("update_start") is not None:
            params["u_start"] = config["update_start"]
        if config.get("update_end") is not None:
            params["u_end"] = config["update_end"]

        # Create and return validated instance
        return cls(**params)

    # Helper methods for validation
    @classmethod
    def _check_required_for_updates(
        cls,
        value: Optional[int],
        field_name: Optional[str],
        download_updates: bool,
    ) -> Optional[int]:
        """Check if a parameter is required for updates and present."""
        if download_updates and value is None:
            if field_name == "u_start":
                print(f"Warning: {field_name} not provided, using default value of 1")
                return 1
            else:
                field_name_str = "parameter" if field_name is None else field_name
                error_msg = (
                    f"When download_updates is True, {field_name_str} must be provided"
                )
                raise ValueError(error_msg)
        return value

    @classmethod
    def _check_value_bounds(
        cls,
        value: int,
        field_name: Optional[str],
        min_val: Optional[int],
        max_val: Optional[int],
    ) -> int:
        """Check if a value is within bounds and adjust if necessary."""
        if min_val is not None and value < min_val and field_name == "u_start":
            print(
                f"Warning: u_start ({value}) is less than minimum available "
                f"update file ({min_val}). Using minimum available.",
            )
            return min_val
        elif max_val is not None and value > max_val and field_name == "u_end":
            print(
                f"Warning: u_end ({value}) exceeds maximum available update file "
                f"({max_val}). Using maximum available.",
            )
            return max_val
        return value

    @classmethod
    def _check_start_end_order(
        cls,
        end_val: int,
        start_val: Optional[int],
        field_name: Optional[str],
    ) -> int:
        """Check that start value is less than end value."""
        if field_name == "u_end" and start_val is not None and end_val < start_val:
            error_msg = f"u_end ({end_val}) cannot be less than u_start ({start_val})."
            raise ValueError(error_msg)
        return end_val

    # Updated to use field_validator with the newer syntax and proper type annotations
    @field_validator("u_start", "u_end", mode="after")
    @classmethod
    def validate_update_numbers(
        cls,
        v: Optional[int],
        info: ValidationInfo,
    ) -> Optional[int]:
        """Validate update file numbers.

        Args:
            v: The value to validate
            info: ValidationInfo containing field info and data

        Returns:
            The validated value, possibly adjusted within bounds

        Raises:
            ValueError: If the value is invalid or missing when required

        """
        # Check if required for updates
        v = cls._check_required_for_updates(
            v,
            info.field_name,
            info.data.get("download_updates", False),
        )

        # If download_updates is True, validate against server file listings
        if info.data.get("download_updates") and v is not None:
            baseline = info.data.get("baseline")
            formatted_baseline = f"{baseline:02d}"

            # Get min and max update file numbers from server
            min_file, max_file = cls._get_min_max_update_file_numbers(
                formatted_baseline,
            )

            # Check bounds and adjust if necessary
            v = cls._check_value_bounds(v, info.field_name, min_file, max_file)

            # Check that u_start < u_end if both are provided
            v = cls._check_start_end_order(
                v,
                info.data.get("u_start"),
                info.field_name,
            )

        return v

    @field_validator("baseline", mode="after")
    @classmethod
    def _validate_baseline_year(cls, v: int) -> int:
        """Validate the baseline year.

        Args:
            v: The baseline year value

        Returns:
            The validated baseline year

        """
        # Check if baseline is in future (warning only)
        current_year_short = time.localtime().tm_year % 100
        if v > current_year_short:
            print(
                f"Warning: Baseline year {v} ({2000+v}) is later than current year "
                f"({2000+current_year_short}). Ensure this is intended if downloading "
                f"baseline files.",
            )
        return v

    @field_validator("n_end", mode="after")
    @classmethod
    def _validate_n_end(cls, v: Optional[int], info: ValidationInfo) -> int:
        """Validate the n_end value.

        Args:
            v: The n_end value
            info: ValidationInfo containing field info and data

        Returns:
            The validated n_end value, possibly derived from server

        Raises:
            ValueError: If n_end is invalid or cannot be determined

        """
        # If n_end is not provided, determine from server
        baseline = info.data.get("baseline")
        formatted_baseline = f"{baseline:02d}"

        if v is None:
            max_file_num = cls._get_max_baseline_file_number(formatted_baseline)
            if max_file_num is not None:
                v = max_file_num
                print(f"n_end not provided, using determined max from server: {v}")
            else:
                error_msg = (
                    "Could not determine max file number for baseline "
                    f"{formatted_baseline}. "
                    "Please specify n_end manually or check server availability."
                )
                raise ValueError(error_msg)
        else:
            # If n_end is provided, check that it doesn't exceed server max
            max_file_num = cls._get_max_baseline_file_number(formatted_baseline)
            if max_file_num is not None and v > max_file_num:
                print(
                    f"Provided n_end ({v}) exceeds server maximum ({max_file_num}). "
                    f"Using server maximum.",
                )
                v = max_file_num

        # Ensure n_start is not greater than n_end
        n_start = info.data.get("n_start", 0)
        if n_start > v:
            error_msg = f"n_start ({n_start}) cannot be greater than n_end ({v})."
            raise ValueError(error_msg)

        return v

    @classmethod
    def _fetch_directory_listing(cls, listing_url: str) -> Optional[str]:
        """Fetch directory listing from the given URL.

        Args:
            listing_url: URL to fetch directory listing from

        Returns:
            Directory listing content as string, or None on error

        """
        try:
            with urllib.request.urlopen(listing_url, timeout=30) as response:
                if response.getcode() == 200:
                    return response.read().decode("utf-8")
                else:
                    print(
                        f"Error fetching directory listing from {listing_url}: "
                        f"HTTP {response.getcode()}",
                    )
                    return None
        except urllib.error.HTTPError as e:
            print(
                f"Error fetching directory listing (HTTPError) from {listing_url}: "
                f"{e.code} {e.reason}",
            )
            return None
        except urllib.error.URLError as e:
            print(
                f"Error fetching directory listing (URLError) from {listing_url}: "
                f"{e.reason}",
            )
            return None
        except Exception as e:
            print(
                f"An unexpected error occurred while fetching directory listing from "
                f"{listing_url}: {e}",
            )
            return None

    @classmethod
    def _get_min_max_file_numbers(
        cls,
        formatted_baseline: str,
        listing_url: str,
        context_type: str,
    ) -> tuple[Optional[int], Optional[int]]:
        """Get minimum and maximum file numbers from the specified server URL.

        Args:
            formatted_baseline: The two-digit baseline year (e.g., "23" for 2023)
            listing_url: The URL to fetch the directory listing from
            context_type: Type of files being retrieved (e.g., "baseline" or "update")

        Returns:
            A tuple containing (min_file_number, max_file_number) if files found,
            otherwise (None, None)

        """
        html_content = cls._fetch_directory_listing(listing_url)
        if not html_content:
            return None, None

        pattern = re.compile(rf"pubmed{formatted_baseline}n(\d+)\.xml\.gz")
        numbers = []
        for line in html_content.splitlines():
            match = pattern.search(line)
            if match:
                numbers.append(int(match.group(1)))

        if numbers:
            return min(numbers), max(numbers)
        else:
            print(
                f"Warning: No {context_type} files found for baseline "
                f"{formatted_baseline} matching the pattern.",
            )
            return None, None

    @classmethod
    def _get_max_baseline_file_number(cls, formatted_baseline: str) -> Optional[int]:
        """Get maximum baseline file number."""
        _, max_num = cls._get_min_max_file_numbers(
            formatted_baseline,
            PUBMED_BASELINE_URL,
            "baseline",
        )
        return max_num

    @classmethod
    def _get_min_max_update_file_numbers(
        cls,
        formatted_baseline: str,
    ) -> tuple[Optional[int], Optional[int]]:
        """Get minimum and maximum update file numbers from the server."""
        return cls._get_min_max_file_numbers(
            formatted_baseline,
            PUBMED_UPDATE_URL,
            "update",
        )

    @field_validator("save_path", mode="after")
    @classmethod
    def _validate_and_resolve_path(cls, v: str) -> str:
        """Ensure the save path is properly resolved to an absolute path.

        Args:
            v: The save path value

        Returns:
            Absolute path string

        """
        # If it's already an absolute path, just return it
        path = Path(v)
        if path.is_absolute():
            return str(path)

        # Import here to avoid circular imports
        from easyner.infrastructure.paths import PROJECT_ROOT

        # Resolve relative to PROJECT_ROOT
        resolved_path = PROJECT_ROOT / path
        return str(resolved_path)


class PubMedDownloader:
    """Downloader for PubMed XML files.

    This class handles downloading baseline and nightly update files from PubMed.
    It manages the download process, including error handling and logging.
    """

    def __init__(self, config: DownloaderConfig) -> None:
        """Initialize the PubMed downloader with configuration.

        Args:
            config: A validated DownloaderConfig instance

        """
        # Initialize attributes directly from config
        self.baseline = config.baseline
        self.formatted_baseline = f"{self.baseline:02d}"
        self.save_path = Path(config.save_path)
        self.n_start: int = config.n_start
        self.n_end: int = config.n_end
        self.download_updates = config.download_updates
        self.skip_existing = config.skip_existing
        self.max_connections = config.max_connections
        self.chunk_size = config.chunk_size

        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Handle update file parameters with proper typing
        if self.download_updates:
            # According to DownloaderConfig validation, when download_updates is True:
            # - u_start will be at least 1 (defaulted if None)
            # - u_end must be provided and valid
            self.u_start: int = 1 if config.u_start is None else config.u_start
            self.u_end: int = config.u_end if config.u_end is not None else 0

            # Safety check for unexpected None values
            # (should never happen after validation)
            if self.u_end == 0:
                msg = (
                    "u_end must be specified when download_updates is True. "
                    "Please check the configuration."
                )
                raise ValueError(msg)
        else:
            # When not downloading updates, use placeholder values
            self.u_start: int = 0
            self.u_end: int = 0

        # Store the maximum baseline file number for potential use
        self.n_max = None

    async def _download_file_async(
        self,
        url: str,
        file_path: Path,
        error_log_file: TextIOWrapper,
        file_id_for_log: str,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Download a single file asynchronously from the specified URL.

        Args:
            url: URL to download from
            file_path: Path where to save the downloaded file
            error_log_file: Open file for error logging
            file_id_for_log: Identifier for this file in the log
            semaphore: Semaphore to limit concurrent connections

        Returns:
            None

        """
        # Check if file already exists and has content
        if self.skip_existing and file_path.exists() and file_path.stat().st_size > 0:
            return

        # Use semaphore to limit concurrent connections
        async with semaphore:
            # Use our context manager to track the file being downloaded
            async with track_downloading_file(file_path):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                # Create parent directory if it doesn't exist
                                file_path.parent.mkdir(parents=True, exist_ok=True)

                                # Write file in chunks to avoid memory issues
                                with open(file_path, "wb") as f:
                                    downloaded = 0
                                    while True:
                                        chunk = await response.content.read(
                                            self.chunk_size,
                                        )
                                        if not chunk:
                                            break
                                        f.write(chunk)
                                        downloaded += len(chunk)

                                # Verify file was created successfully
                                if not (
                                    file_path.exists() and file_path.stat().st_size > 0
                                ):
                                    msg = f"ERROR: File not created or empty after download: {file_path}"
                                    error_log_file.write(f"{file_id_for_log}\t{msg}\n")
                                    error_log_file.flush()  # Ensure error is written immediately
                            else:
                                msg = f"ERROR: HTTP {response.status} for {file_id_for_log} from {url}"
                                error_log_file.write(f"{file_id_for_log}\t{msg}\n")
                                error_log_file.flush()
                except Exception as e:
                    msg = f"ERROR downloading {file_id_for_log} from {url}: {str(e)}"
                    error_log_file.write(f"{file_id_for_log}\t{str(e)}\n")
                    error_log_file.flush()
                    # Remove partial download if it exists
                    if file_path.exists():
                        try:
                            file_path.unlink()
                            error_log_file.write(
                                f"{file_id_for_log}\tRemoved incomplete download\n",
                            )
                            error_log_file.flush()
                        except Exception as clean_error:
                            error_log_file.write(
                                f"{file_id_for_log}\tFailed to remove incomplete download: {str(clean_error)}\n",
                            )
                            error_log_file.flush()

    def _prepare_download_tasks(
        self,
        start_idx: int,
        end_idx: int,
        is_update: bool,
        error_log_file: TextIOWrapper,
    ) -> list[dict[str, Any]]:
        """Prepare download tasks for either baseline or update files.

        Args:
            start_idx: Starting file index
            end_idx: Ending file index
            is_update: Whether these are update files (True) or baseline files (False)
            error_log_file: Open file for error logging

        Returns:
            List of file info dictionaries containing URL, path, and log info

        """
        download_tasks = []
        base_url = PUBMED_UPDATE_URL if is_update else PUBMED_BASELINE_URL
        task_type = "update" if is_update else "baseline"

        for i in range(start_idx, end_idx + 1):
            file_name = f"pubmed{self.formatted_baseline}n{i:04d}.xml.gz"
            url = f"{base_url}{file_name}"
            file_path = self.save_path / file_name
            file_id = f"{task_type}_{self.formatted_baseline}n{i:04d}"

            # Skip if file already exists and skip_existing is True
            if (
                self.skip_existing
                and file_path.exists()
                and file_path.stat().st_size > 0
            ):
                continue

            download_tasks.append(
                {
                    "url": url,
                    "file_path": file_path,
                    "file_id": file_id,
                },
            )

        return download_tasks

    async def _download_files_async(
        self,
        download_tasks: list[dict[str, Any]],
        error_log_file: TextIOWrapper,
    ) -> None:
        """Download multiple files asynchronously.

        Args:
            download_tasks: List of dictionaries with download task information
            error_log_file: Open file for error logging

        Returns:
            None

        """
        # Create a semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(self.max_connections)

        # Create download tasks
        tasks = []
        for task in download_tasks:
            tasks.append(
                self._download_file_async(
                    task["url"],
                    task["file_path"],
                    error_log_file,
                    task["file_id"],
                    semaphore,
                ),
            )

        # Use tqdm to show progress
        total_tasks = len(tasks)
        if total_tasks == 0:
            print("No files to download (all files exist or no files in range)")
            return

        print(
            f"Downloading {total_tasks} files with max {self.max_connections} concurrent connections",
        )

        # Progress tracking with tqdm
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Downloading files",
            unit="file",
        ):
            await future

    async def _download_baseline_files_async(
        self,
        error_log_file: TextIOWrapper,
    ) -> None:
        """Download baseline files asynchronously within the specified range.

        Args:
            error_log_file: Open file for error logging

        Returns:
            None

        """
        print(
            f"Downloading baseline files n{self.n_start:04d} to n{self.n_end:04d} "
            f"for baseline {self.formatted_baseline}",
        )

        # Prepare download tasks
        download_tasks = self._prepare_download_tasks(
            self.n_start,
            self.n_end,
            False,  # Not update files
            error_log_file,
        )

        # Download files asynchronously
        await self._download_files_async(download_tasks, error_log_file)

    async def _download_update_files_async(self, error_log_file: TextIOWrapper) -> None:
        """Download update files asynchronously within the specified range.

        Args:
            error_log_file: Open file for error logging

        Returns:
            None

        """
        print(
            f"Downloading Nightly Update Files n{self.u_start:04d} to "
            f"n{self.u_end:04d} for baseline {self.formatted_baseline}",
        )

        # Prepare download tasks
        download_tasks = self._prepare_download_tasks(
            self.u_start,
            self.u_end,
            True,  # These are update files
            error_log_file,
        )

        # Download files asynchronously
        await self._download_files_async(download_tasks, error_log_file)

    def execute_download(self) -> None:
        """Execute the download process for PubMed files.

        Downloads baseline files or update files based on the configuration.
        Records any errors in a log file.

        Returns:
            None

        """
        # Set up signal handlers for graceful termination
        setup_signal_handlers()

        print(f"PubMed Downloader initialized for baseline {self.baseline}.")
        print(f"Target save path: {self.save_path.resolve()}")
        print(f"Using max {self.max_connections} concurrent connections")
        print("Press Ctrl+C to interrupt (incomplete downloads will be cleaned up)")

        error_log_path = self.save_path / "download_errors.txt"
        with open(error_log_path, "w", encoding="utf8") as error_log_file:
            try:
                # Create and run the event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                if self.download_updates:
                    print("Running in update files mode.")
                    loop.run_until_complete(
                        self._download_update_files_async(error_log_file),
                    )
                else:
                    print("Running in baseline files mode.")
                    loop.run_until_complete(
                        self._download_baseline_files_async(error_log_file),
                    )
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected, cleaning up...")
                cleanup_partial_downloads()
                print("Interrupted. Incomplete downloads have been cleaned up.")
                error_log_file.write("Download process interrupted by user\n")
                error_log_file.flush()
                return
            except Exception as e:
                print(f"\nUnexpected error during download: {str(e)}")
                cleanup_partial_downloads()
                error_log_file.write(f"Download process failed with error: {str(e)}\n")
                error_log_file.flush()
                raise

        print("PubMed download process finished.")
        # Check if error log has content other than being empty
        if error_log_path.stat().st_size > 0:
            print(
                "Some errors occurred during download. "
                f"Please check: {error_log_path.resolve()}",
            )
        else:
            error_log_path.unlink()  # Remove empty error log
            print("No download errors recorded.")

    # Keep these legacy methods for backwards compatibility
    def _download_individual_file(
        self,
        url: str,
        file_path: Path,
        error_log_file: TextIOWrapper,
        file_id_for_log: str,
    ) -> None:
        """Legacy synchronous download method.

        This method is kept for backwards compatibility.
        """
        # Check if file already exists and has content
        if self.skip_existing and file_path.exists() and file_path.stat().st_size > 0:
            print(f"Skipping existing file: {file_path.name}")
            return

        try:
            urllib.request.urlretrieve(url, filename=file_path)
            if not (file_path.exists() and file_path.stat().st_size > 0):
                msg = f"ERROR: File not created or empty after download: {file_path}"
                error_log_file.write(f"{file_id_for_log}\t{msg}\n")
        except Exception as e:
            msg = f"ERROR downloading {file_id_for_log} from {url}: {str(e)}"
            error_log_file.write(f"{file_id_for_log}\t{str(e)}\n")

    def _download_baseline_files(self, error_log_file: TextIOWrapper) -> None:
        """Legacy method for synchronous baseline downloads."""
        print(
            "Using legacy synchronous download method. Consider using the newer async implementation.",
        )
        print(
            f"Downloading baseline files n{self.n_start:04d} to n{self.n_end:04d} "
            f"for baseline {self.formatted_baseline}",
        )
        for i in trange(
            self.n_start,
            self.n_end + 1,
            desc=f"Baseline {self.formatted_baseline}",
        ):
            file_name = f"pubmed{self.formatted_baseline}n{i:04d}.xml.gz"
            url = f"{PUBMED_BASELINE_URL}{file_name}"
            file_path = self.save_path / file_name
            self._download_individual_file(
                url,
                file_path,
                error_log_file,
                f"baseline_{self.formatted_baseline}n{i:04d}",
            )
            if i % 10 == 0:  # Adjusted sleep frequency
                time.sleep(0.05)  # Reduced sleep time

    def _download_update_files(self, error_log_file: TextIOWrapper) -> None:
        """Legacy method for synchronous update downloads."""
        print(
            "Using legacy synchronous download method. Consider using the newer async implementation.",
        )
        print(
            f"Downloading Nightly Update Files n{self.u_start:04d} to "
            f"n{self.u_end:04d} for baseline {self.formatted_baseline}",
        )
        for i in trange(
            self.u_start,
            self.u_end + 1,
            desc=f"Updates {self.formatted_baseline}",
        ):
            # Update files also use baseline in name
            file_name = f"pubmed{self.formatted_baseline}n{i:04d}.xml.gz"
            url = f"{PUBMED_UPDATE_URL}{file_name}"
            file_path = self.save_path / file_name
            self._download_individual_file(
                url,
                file_path,
                error_log_file,
                f"update_{self.formatted_baseline}n{i:04d}",
            )
            if i % 10 == 0:
                time.sleep(0.05)


def run_pubmed_download(config: dict) -> None:
    """Run the PubMed baseline download process based on the provided configuration.

    Args:
        config: A dictionary containing baseline download configuration parameters

    """
    # Create a configuration for baseline downloads
    params = {
        "baseline": config.get("baseline"),
        "save_path": config.get(
            "raw_download_path",
            DownloaderConfig.DEFAULT_SAVE_PATH,
        ),
        "n_start": config.get("file_start", 0),
        "n_end": config.get("file_end"),
        "download_updates": False,  # Explicitly set this to False for baseline downloads
        "skip_download": config.get("skip_download", False),
        "skip_existing": config.get("skip_existing", True),
    }

    # Create and validate the config
    downloader_config = DownloaderConfig(**params)

    # Check if we should skip download
    if not downloader_config.skip_download:
        print("Downloading PubMed baseline files...")
        try:
            # Initialize and run downloader
            downloader = PubMedDownloader(downloader_config)
            downloader.execute_download()
            print("PubMed baseline download complete.")
        except Exception as e:
            print(f"Error during PubMed baseline download: {str(e)}")
    else:
        print("PubMed baseline download step skipped based on configuration.")


def run_pubmed_updates_download(config: dict) -> None:
    """Run the PubMed updates download process based on the provided configuration.

    Args:
        config: A dictionary containing update download configuration parameters

    """
    # Create a configuration for update downloads
    params = {
        "baseline": config.get("baseline"),
        "save_path": config.get(
            "raw_download_path",
            DownloaderConfig.DEFAULT_SAVE_PATH,
        ),
        "download_updates": True,  # Explicitly set this to True for update downloads
        "u_start": config.get("update_start"),
        "u_end": config.get("update_end"),
        "skip_download": config.get("skip_download", False),
        "skip_existing": config.get("skip_existing", True),
    }

    # Create and validate the config
    downloader_config = DownloaderConfig(**params)

    # Check if we should skip download
    if not downloader_config.skip_download:
        print("Downloading PubMed update files...")
        try:
            # Initialize and run downloader
            downloader = PubMedDownloader(downloader_config)
            downloader.execute_download()
            print("PubMed update download complete.")
        except Exception as e:
            print(f"Error during PubMed update download: {str(e)}")
    else:
        print("PubMed update download step skipped based on configuration.")


if __name__ == "__main__":
    try:
        # When run as a standalone script, use the config from the config module
        import sys

        from easyner.config import config_manager

        # Display help message if requested
        if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
            print("PubMed Bulk Downloader")
            print("=====================")
            print("Usage:")
            print(
                "  python -m easyner.pipeline.pubmed.pubmed_bulk_downloader [OPTIONS]",
            )
            print("\nOptions:")
            print(
                "  --update-files    Download PubMed update files instead of baseline",
            )
            print("  --force-download  Force redownload of existing files")
            print("  -h, --help        Show this help message and exit")
            print("\nConfiguration:")
            print("  This script uses either the pubmed_bulk_downloader or")
            print("  pubmed_bulk_updates_downloader sections from config.json.")
            sys.exit(0)

        # Load the config if it isn't already loaded
        config = config_manager.get_config()

        # Check command line arguments to determine mode
        # Use --update-files flag to download update files instead of baseline
        update_mode = "--update-files" in sys.argv
        # Use --force-download flag to force redownload of existing files
        force_download = "--force-download" in sys.argv

        if update_mode:
            # Use the pubmed_bulk_updates_downloader section
            if "pubmed_bulk_updates_downloader" in config:
                print("Using updates configuration from config.json")
                updates_config = config["pubmed_bulk_updates_downloader"]

                # Check that required configuration options exist
                required_keys = [
                    "raw_download_path",
                    "baseline",
                    "update_start",
                    "update_end",
                ]
                missing_keys = [
                    key for key in required_keys if key not in updates_config
                ]

                if missing_keys:
                    msg = (
                        f"Missing required keys in pubmed_bulk_updates_downloader: "
                        f"{', '.join(missing_keys)}"
                    )
                    raise ValueError(msg)

                # Update the skip_existing parameter based on force_download flag
                if force_download:
                    updates_config["skip_existing"] = False
                run_pubmed_updates_download(updates_config)
            else:
                msg = (
                    "pubmed_bulk_updates_downloader section not found in config.json.\n"
                    "Please add this section to your config file."
                )
                raise ValueError(msg)
        else:
            # Use the pubmed_bulk_downloader section (default mode)
            if "pubmed_bulk_downloader" in config:
                print("Using baseline configuration from config.json")
                downloader_config = config["pubmed_bulk_downloader"]

                # Check that required configuration options exist
                required_keys = [
                    "raw_download_path",
                    "baseline",
                    "file_start",
                    "file_end",
                ]
                missing_keys = [
                    key for key in required_keys if key not in downloader_config
                ]

                if missing_keys:
                    msg = (
                        f"Missing required keys in pubmed_bulk_downloader: "
                        f"{', '.join(missing_keys)}"
                    )
                    raise ValueError(msg)

                # Update the skip_existing parameter based on force_download flag
                if force_download:
                    downloader_config["skip_existing"] = False
                run_pubmed_download(downloader_config)
            else:
                msg = (
                    "pubmed_bulk_downloader section not found in config.json.\n"
                    "Please add this section to your config file."
                )
                raise ValueError(msg)
    except KeyboardInterrupt:
        print("\nProgram interrupted. Cleaning up...")
        cleanup_partial_downloads()
        print("Exiting.")
        sys.exit(1)
