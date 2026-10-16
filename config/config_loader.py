import yaml
from pathlib import Path
from typing import Dict, Any, Union
import os

class ConfigLoader:
    """YAML loader with !include support and recursive dictionary merging."""

    def __init__(self, base_path: Union[str, Path] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.loaded_files = set()  # to detect circular includes

    def load(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML file, processing any !include tags."""
        file_path = Path(file_path)
        print(f"Loading file: {file_path} (base_path: {self.base_path})")
        if not file_path.is_absolute():
            file_path = self.base_path / file_path

        if file_path in self.loaded_files:
            raise ValueError(f"Circular include detected: {file_path}")
        self.loaded_files.add(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.load(f, Loader=self._get_loader())

        self.loaded_files.remove(file_path)
        return data

    def _get_loader(self):
        """Return a custom YAML loader that handles !include tags."""
        class Loader(yaml.SafeLoader):
            pass

        def include_constructor(loader, node):
            """Handle !include filename."""
            filename = loader.construct_scalar(node)
            return self.load(filename)

        Loader.add_constructor('!include', include_constructor)
        return Loader


def merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    - Lists are replaced (not merged).
    - Dictionaries are merged recursively.
    """
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(main_file: Union[str, Path], base_path: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Load a main config file, resolve !includes, and return the merged dictionary.
    """
    loader = ConfigLoader(base_path)
    return loader.load(main_file)