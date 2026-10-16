# config/model.py
from .base import BaseConfig, ModelArchitecture
from typing import Dict, List, Any, Optional
from pydantic import Field, field_validator
import torch

class ModelConfig(BaseConfig):
    """Model configuration with architecture-aware defaults"""
    name: str
    architecture: ModelArchitecture
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    vocab_size: int
    
    # Training settings
    pretrained: bool = True
    freeze_base: bool = False
    torch_dtype: str = "float32"
    
    # Target modules for different PEFT methods
    target_modules: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "lora": ["query", "value"],
            "dora": ["query", "value"],
            "adalora": ["query", "value", "key"],
            "prefix": ["key", "value"],
            "ia3": ["key", "value", "output.dense"],
        }
    )

    target_policy: Optional[str] = None
    
    @field_validator('torch_dtype')
    def validate_torch_dtype(cls, v):
        valid_dtypes = ["float32", "float16", "bfloat16"]
        if v not in valid_dtypes:
            raise ValueError(f"torch_dtype must be one of {valid_dtypes}")
        return v
    
    @property
    def get_dtype(self):
        return getattr(torch, self.torch_dtype)
    
    def get_target_modules(self, method: str) -> List[str]:
        """Get target modules for specific PEFT method"""
        return self.target_modules.get(method, ["query", "value"])