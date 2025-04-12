import logging
from .splitter_runner import run_splitter

# Configure logging
logger = logging.getLogger("easyner.pipeline.splitter")
logger.setLevel(logging.INFO)

# Create console handler with a higher log level if not already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(console_handler)

__all__ = ["run_splitter"]
