"""Development utilities for PubMed processing.

WARNING: These modules are for development purposes only.
They are not intended for use in production code and may change
or be removed without notice.
"""

import warnings

warnings.warn(
    "The modules in easyner.pipeline.pubmed._dev_utils are for development "
    "purposes only and should not be used in production code. "
    "They may change or be removed without notice.",
    DeprecationWarning,
    stacklevel=2,
)
