from pathlib import Path


def _resolve_path(path_str: str) -> str:
    """Resolve a path against the project root if it's relative.

    Args:
        path_str: Path string that might be relative

    Returns:
        Resolved absolute path as string

    """
    path = Path(path_str)
    if path.is_absolute():
        return str(path)

    # Import here to avoid circular imports
    from easyner.infrastructure.paths import PROJECT_ROOT

    # Resolve relative to PROJECT_ROOT
    resolved_path = PROJECT_ROOT / path
    return str(resolved_path)
