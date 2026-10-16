from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import json

import yaml

# torch is only needed to pick a default device and validate it.
# Imported lazily inside the validator / default factory so that
# simply importing config.base does not require torch.


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

class TaskType(str, Enum):
    SENTIMENT = "sentiment"
    NEWS = "news"
    CUSTOM = "custom"

class ModelArchitecture(str, Enum):
    BERT = "bert"
    ROBERTA = "roberta"
    XLM_ROBERTA = "xlm-roberta"
    LLAMA = "llama"
    GPT2 = "gpt2"

class PEFTMethod(str, Enum):
    LORA = "lora"
    DORA = "dora"
    ADALORA = "adalora"
    QLORA = "qlora"
    PREFIX = "prefix"
    PROMPT = "prompt"
    IA3 = "ia3"
    PROGRESSIVE = "progressive"
    FULL = "full"
    BITFIT = "bitfit"
    VANILLA = "vanilla"
    DY_LORA = "dy_lora"      # Dynamic LoRA
    SA_LORA = "sa_lora"      # Structure-Aware LoRA

class RankSchedule(str, Enum):
    LINEAR = "linear"
    ADAPTIVE = "adaptive"
    EXPONENTIAL = "exponential"
    FIXED = "fixed"
    STEP = "step"

class AllocationStrategy(str, Enum):
    THRESHOLD = "threshold"
    TOP_K = "top_k"
    PROPORTIONAL = "proportional"
    EFFICIENT = "efficient"

    # New values for progressive PEFT
    BUDGET = "budget"
    STRUCTURE = "structure"

class BaseConfig(BaseModel):
    """Base configuration with common settings"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    # Experiment metadata
    experiment_id: str = Field(default_factory=lambda: f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    experiment_name: str = "bopet_research"
    study_id: Optional[str] = None
    hypothesis: Optional[str] = None
    description: str = ""
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    device: str = Field(default_factory=_default_device)
    
    # MLflow tracking
    use_mlflow: bool = True
    mlflow_experiment_name: str = "peft_research"
    mlflow_tracking_uri: Optional[str] = None
    # Enhanced MLflow tracking
    mlflow_log_frequency: int = Field(
        default=100,
        description="Log metrics every N batches"
    )
    mlflow_log_system_metrics: bool = Field(
        default=True,
        description="Enable system metrics logging (CPU/GPU)"
    )
    mlflow_register_models: bool = Field(
        default=True,
        description="Register models in MLflow Model Registry"
    )
    mlflow_model_staging: str = Field(
        default="Staging",
        description="Stage for registered models (Staging/Production/Archived)"
    )
    
    # Logging
    log_level: str = "INFO"
    save_artifacts: bool = True
    save_checkpoints: bool = True
    
    @field_validator('device')
    def validate_device(cls, v):
        if v == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("⚠️ CUDA requested but not available. Falling back to CPU.")
                    return "cpu"
            except ImportError:
                return "cpu"
        return v
    
    def to_yaml(self, path: Path):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(exclude_none=True), f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'BaseConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)