from abc import ABC, abstractmethod
from typing import Dict, Any


class ModuleInterface(ABC):
    """Abstract base class that all EasyNer modules must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name identifier of this module."""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate that the provided configuration is acceptable."""
        pass

    @abstractmethod
    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute module processing with the given configuration."""
        pass


class ModuleRegistryError(Exception):
    """Error raised when there are issues with module registration."""

    pass


class ModuleRegistry:
    """Central registry of all available modules."""

    def __init__(self):
        self._modules = {}

    def register(self, module_factory):
        """Register a module factory function."""
        module = module_factory()
        if not isinstance(module, ModuleInterface):
            raise ModuleRegistryError(
                f"Module {module.__class__.__name__} does not implement ModuleInterface"
            )
        if module.name in self._modules:
            raise ModuleRegistryError(
                f"Module with name '{module.name}' already registered"
            )
        self._modules[module.name] = module_factory
        return module_factory  # Return for decorator support

    def get_module(self, name):
        """Get a new instance of a module by name."""
        if name not in self._modules:
            raise ModuleRegistryError(
                f"No module registered with name '{name}'"
            )
        return self._modules[name]()

    def get_all_modules(self):
        """Get all registered module instances."""
        return {name: factory() for name, factory in self._modules.items()}


# Create a global registry instance
registry = ModuleRegistry()
