from easyner.core.module import registry


def register_module(func):
    """Decorator to register a module factory function."""
    registry.register(func)
    return func
