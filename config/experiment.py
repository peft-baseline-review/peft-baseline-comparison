# config/experiment.py - UPDATED VERSION with model_dump
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from pydantic import Field, model_validator
import yaml
from .base import BaseConfig
from .task import TaskConfig
from .model import ModelConfig
from .peft import PEFTConfig
from .training import TrainingConfig
from .evaluation import EvaluationConfig

class ExperimentConfig(BaseConfig):
    """Complete experiment configuration with YAML-compatible nested structure"""
    
    # Component configurations
    task: TaskConfig
    model: ModelConfig
    peft: PEFTConfig
    training: TrainingConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    # Experiment design
    replication_count: int = 3
    randomize_order: bool = True
    counterbalance: bool = False
    
    # Resource management
    max_gpu_memory_gb: Optional[float] = None
    max_system_memory_gb: Optional[float] = None
    timeout_hours: int = 24
    
    # Output paths
    output_dir: str = "experiments"
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"

    mlflow_tags: Optional[Dict[str, str]] = None


    
    @model_validator(mode='before')
    @classmethod
    def flatten_nested_config(cls, data: Any) -> Any:
        """Flatten nested YAML structure into flat Pydantic structure"""
        if isinstance(data, dict):
            # Create a copy to avoid modifying the original
            data = data.copy()
            
            # If we have an 'experiment' section, flatten it
            if 'experiment' in data:
                experiment_data = data.pop('experiment')
                if isinstance(experiment_data, dict):
                    # Merge experiment data into top level
                    data.update(experiment_data)
            
            # Handle nested configurations
            nested_sections = ['task', 'model', 'peft', 'training', 'evaluation']
            for section in nested_sections:
                if section in data and isinstance(data[section], dict):
                    # This will be processed by the respective Pydantic model
                    pass
            
            # Handle experiment_design, resources, output sections if present
            if 'experiment_design' in data:
                design = data.pop('experiment_design')
                if isinstance(design, dict):
                    data.update({k: v for k, v in design.items() if k not in data})
            
            if 'resources' in data:
                resources = data.pop('resources')
                if isinstance(resources, dict):
                    data.update({k: v for k, v in resources.items() if k not in data})
            
            if 'output' in data:
                output = data.pop('output')
                if isinstance(output, dict):
                    data.update({k: v for k, v in output.items() if k not in data})
        
        return data
    
    def get_experiment_path(self) -> Path:
        """Get structured experiment path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(
            self.output_dir,
            self.experiment_name,
            self.task.task_type.value,
            self.model.architecture.value,
            self.peft.method.value,
            timestamp
        )
    
    def to_mlflow_params(self) -> Dict[str, Any]:
        """Convert config to MLflow parameters format - ENHANCED VERSION"""
        params = {}
        
        # Core experiment metadata
        params.update({
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "study_id": self.study_id or "N/A",
            "task": self.task.task_type.value,
            "model": self.model.name,
            "architecture": self.model.architecture.value,
            "peft_method": self.peft.method.value,
            "learning_rate": self.training.learning_rate,
            "batch_size": self.training.effective_batch_size,
            "num_epochs": self.training.num_epochs,
            "seed": self.seed,
            "num_labels": self.task.num_labels,
            "max_length": self.task.max_length,
            "freeze_base": self.model.freeze_base,
            "use_amp": self.training.use_amp,
            "use_class_weights": self.task.use_class_weights,
        })
        
        # Conditional PEFT parameters (only for methods that use them)
        if self.peft.method.value not in ["full", "vanilla"]:
            if self.peft.rank is not None:
                params["peft_rank"] = self.peft.rank
            if self.peft.alpha is not None:
                params["peft_alpha"] = self.peft.alpha
            if self.peft.dropout is not None and self.peft.dropout > 0:
                params["peft_dropout"] = self.peft.dropout
            if self.peft.bias and self.peft.bias != "none":
                params["peft_bias"] = self.peft.bias
        
        # Progressive PEPT specific parameters
        if self.peft.method.value == "progressive":
            params.update({
                "progressive_base_method": self.peft.base_method,
                "progressive_initial_rank": self.peft.initial_rank,
                "progressive_final_rank": self.peft.final_rank,
                "progressive_allocation_strategy": self.peft.allocation_strategy,
                "progressive_importance_method": self.peft.importance_method,
            })
        
        return params
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from YAML file with nested structure support"""
        import yaml
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Apply flattening via the validator
        return cls(**data)
    
    # FIXED: Add method to save config as YAML
    def to_yaml(self, path: Path):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(exclude_none=True), f, default_flow_style=False)