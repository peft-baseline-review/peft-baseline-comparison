# config/__init__.py - UPDATED VERSION
"""
Configuration factory for loading different config types with improved detection
"""

from pathlib import Path
from typing import Union, Any, Dict, Optional
import yaml


def __getattr__(name):
    """Defer peft-dependent imports until actually requested."""
    if name in ("PEFTConfig", "ImportanceMethod"):
        from .peft import PEFTConfig, ImportanceMethod
        return {"PEFTConfig": PEFTConfig, "ImportanceMethod": ImportanceMethod}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def detect_config_type(data: Dict[str, Any]) -> Optional[str]:
    """
    Smart configuration type detection
    """
    keys = set(data.keys())
    
    # 1. Check for unique identifiers
    if 'baseline_methods' in keys:
        return "baseline"
    
    if 'search_space' in keys and 'optimization' in keys:
        return "sweep"
    
    # 2. Check for required sections
    experiment_sections = {'task', 'model', 'peft', 'training'}
    if experiment_sections.issubset(keys):
        return "experiment"
    
    # 3. Check for file name patterns (fallback)
    # This should be handled by the caller if file path is available
    return None

def load_config(config_path: Union[str, Path], config_type: Optional[str] = None) -> Any:
    """
    Load configuration based on type
    
    Args:
        config_path: Path to configuration file
        config_type: Optional, force config type ('experiment', 'baseline', 'sweep')
    
    Returns:
        Appropriate configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Determine config type
    if config_type is None:
        config_type = detect_config_type(data)
        
        # If still None, try filename-based detection
        if config_type is None:
            filename = config_path.stem.lower()
            if 'baseline' in filename:
                config_type = 'baseline'
            elif 'sweep' in filename:
                config_type = 'sweep'
            elif any(keyword in filename for keyword in ['experiment', 'sentiment', 'news', 'lora', 'progressive']):
                config_type = 'experiment'
            else:
                # Default to experiment config (most common)
                config_type = 'experiment'
    
    print(f"📋 Loading {config_path.name} as {config_type} configuration")
    
    # Load based on type
    if config_type == "experiment":
        from .experiment import ExperimentConfig
        return ExperimentConfig(**data)
    
    elif config_type == "baseline":
        from .baseline_config import BaselineRunnerConfig
        return BaselineRunnerConfig(**data)
    
    elif config_type == "sweep":
        from .sweep import SweepConfig  # We need to create this!
        return SweepConfig(**data)
    
    else:
        raise ValueError(f"Unknown config type: {config_type}")