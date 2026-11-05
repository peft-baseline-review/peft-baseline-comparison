# config/sweep.py
"""
Configuration for Bayesian Optimization Sweeps
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum

class OptimizationType(str, Enum):
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    MULTI_FIDELITY = "multi_fidelity"

class Objective(BaseModel):
    """Objective definition for multi-objective optimization"""
    name: str
    weight: float = 1.0
    direction: str = "maximize"  # "maximize" or "minimize"
    
    @field_validator('direction')
    def validate_direction(cls, v):
        if v not in ["maximize", "minimize"]:
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {v}")
        return v

class SearchSpace(BaseModel):
    """Search space definition"""
    
    # PEFT parameters
    methods: Optional[List[str]] = None
    rank_range: Tuple[int, int] = (4, 64)
    alpha_range: Tuple[int, int] = (4, 128)
    dropout_range: Tuple[float, float] = (0.0, 0.3)
    
    # Training parameters
    learning_rate_range: Tuple[float, float] = (1e-5, 5e-4)
    num_epochs_range: Tuple[int, int] = (3, 15)
    batch_sizes: Optional[List[int]] = None
    
    # Progressive PEFT specific
    base_method_options: Optional[List[str]] = None
    initial_rank_range: Tuple[int, int] = (4, 16)
    final_rank_range: Tuple[int, int] = (16, 64)
    rank_schedule_options: Optional[List[str]] = None
    allocation_strategy_options: Optional[List[str]] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Set defaults if None
        if self.methods is None:
            self.methods = ["lora", "dora", "adalora", "prefix", "prompt", "ia3", "progressive"]
        
        if self.batch_sizes is None:
            self.batch_sizes = [8, 16, 32, 64]
        
        if self.base_method_options is None:
            self.base_method_options = ["lora", "dora"]
        
        if self.rank_schedule_options is None:
            self.rank_schedule_options = ["linear", "adaptive", "step"]
        
        if self.allocation_strategy_options is None:
            self.allocation_strategy_options = ["threshold", "top_k", "efficient"]

class SweepConfig(BaseModel):
    """Configuration for Bayesian optimization sweeps"""
    
    base_config: str  # Path to base experiment configuration
    search_space: SearchSpace
    optimization: Dict[str, Any] = Field(
        default_factory=lambda: {
            "optimization_type": "single_objective",
            "n_trials": 30,
            "study_name": "bopet_study",
        }
    )
    
    # Optional sections with defaults
    sampler: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "tpe",
            "seed": 42
        }
    )
    
    pruner: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "type": "median",
            "n_startup_trials": 5,
            "n_warmup_steps": 10
        }
    )
    
    parallel: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,
            "n_jobs": 4,
            "backend": "joblib"
        }
    )
    
    output: Dict[str, Any] = Field(
        default_factory=lambda: {
            "save_results": True,
            "save_pareto_front": True,
            "results_dir": "sweep_results"
        }
    )
    
    @field_validator('base_config')
    def validate_base_config(cls, v):
        """Validate base config path exists or is relative"""
        from pathlib import Path
        
        # Try relative to configs directory
        possible_paths = [
            Path(v),
            Path("configs") / v,
            Path("configs/experiments") / v,
            Path("configs/experiments") / f"{v}.yaml",
            Path("configs/experiments") / f"{v}_flat.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        print(f"⚠️ Warning: Base config '{v}' not found in standard locations")
        return v
    
    @classmethod
    def from_yaml(cls, path):
        """Load configuration from YAML file"""
        import yaml
        from pathlib import Path
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)