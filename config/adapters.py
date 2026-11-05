# config/adapters.py - UPDATED VERSION
"""
Configuration adapter system for backward compatibility
Converts legacy nested YAML structure to flat Pydantic structure - FIXED
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

# Import Pydantic models
from .experiment import ExperimentConfig
from .task import TaskConfig
from .model import ModelConfig
from .peft import PEFTConfig
from .training import TrainingConfig
from .base import TaskType as BaseTaskType

# Import PEFT TaskType
try:
    from peft import TaskType as PEFTTaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PEFTTaskType = None

logger = logging.getLogger(__name__)


class TaskTypeMapper:
    """Map between custom TaskType and PEFT TaskType"""
    
    @staticmethod
    def to_peft_task_type(custom_type: str) -> str:
        """Convert custom task type to PEFT TaskType"""
        mapping = {
            'sentiment': 'SEQ_CLS',
            'news': 'SEQ_CLS',
            'custom': 'SEQ_CLS',
            'SEQ_CLS': 'SEQ_CLS',
            'SEQ_2_SEQ_LM': 'SEQ_2_SEQ_LM',
            'CAUSAL_LM': 'CAUSAL_LM',
            'TOKEN_CLS': 'TOKEN_CLS',
            'QUESTION_ANS': 'QUESTION_ANS',
            'FEATURE_EXTRACTION': 'FEATURE_EXTRACTION'
        }
        return mapping.get(custom_type, 'SEQ_CLS')
    
    @staticmethod
    def to_base_task_type(peft_type: str) -> str:
        """Convert PEFT TaskType to custom task type"""
        reverse_mapping = {
            'SEQ_CLS': 'sentiment',
            'SEQ_2_SEQ_LM': 'custom',
            'CAUSAL_LM': 'custom',
            'TOKEN_CLS': 'custom',
            'QUESTION_ANS': 'custom',
            'FEATURE_EXTRACTION': 'custom'
        }
        return reverse_mapping.get(peft_type, 'custom')


class LegacyYAMLAdapter:
    """
    Adapter for converting legacy nested YAML structure to flat Pydantic structure
    
    Legacy Structure (nested):
        experiment: {...}
        task: {...}
        model: {...}
        peft: {...}
        training: {...}
        evaluation: {...}
    
    Flat Structure (Pydantic):
        experiment_id: ...
        experiment_name: ...
        task: TaskConfig(...)
        model: ModelConfig(...)
        peft: PEFTConfig(...)
        training: TrainingConfig(...)
    """
    
    @staticmethod
    def adapt_experiment_config(yaml_path: Path) -> ExperimentConfig:
        """Load and adapt a legacy experiment configuration"""
        
        with open(yaml_path, 'r') as f:
            raw_data = yaml.safe_load(f)
        
        logger.info(f"Adapting legacy YAML: {yaml_path}")
        
        # Extract sections from legacy structure
        experiment_data = raw_data.get('experiment', {})
        task_data = raw_data.get('task', {})
        model_data = raw_data.get('model', {})
        peft_data = raw_data.get('peft', {})
        training_data = raw_data.get('training', {})
        evaluation_data = raw_data.get('evaluation', {})
        
        # Fix TaskType mismatch
        if 'task_type' in task_data:
            # Convert custom task type to PEFT TaskType for TaskConfig
            custom_task_type = task_data['task_type']
            peft_task_type = TaskTypeMapper.to_peft_task_type(custom_task_type)
            
            # Update task_data for TaskConfig (needs PEFT TaskType)
            task_data['task_type'] = peft_task_type
            
            # Also set custom task type in experiment metadata for reference
            if 'task_type' not in experiment_data:
                experiment_data['task_type'] = custom_task_type
        
        # Merge additional fields into experiment_data
        if evaluation_data:
            experiment_data['evaluation_config'] = evaluation_data
        
        # Check for experiment_design, resources, output sections
        for section in ['experiment_design', 'resources', 'output']:
            if section in raw_data:
                experiment_data[f'{section}_config'] = raw_data[section]
        
        # Create Pydantic objects - FIXED: Use ** unpacking
        task_config = TaskConfig(**task_data) if task_data else TaskConfig(task_type="sentiment", num_labels=2, label_names=[], data_dir="", train_file="", test_file="")
        model_config = ModelConfig(**model_data) if model_data else ModelConfig(name="bert", architecture="bert", hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512, vocab_size=30522)
        peft_config = PEFTConfig(**peft_data) if peft_data else PEFTConfig(method="lora")
        training_config = TrainingConfig(**training_data) if training_data else TrainingConfig()
        
        # Merge all data for ExperimentConfig
        config_data = {**experiment_data}
        
        # Create ExperimentConfig - FIXED: Use model_dump for Pydantic v2 compatibility
        try:
            experiment_config = ExperimentConfig(
                **config_data,
                task=task_config,
                model=model_config,
                peft=peft_config,
                training=training_config
            )
            
            logger.info(f"✅ Successfully adapted: {experiment_config.experiment_name}")
            return experiment_config
            
        except Exception as e:
            logger.error(f"Failed to create ExperimentConfig: {e}")
            raise
    
    @staticmethod
    def adapt_baseline_config(yaml_path: Path) -> Dict[str, Any]:
        """Load and adapt a baseline comparison configuration"""
        
        with open(yaml_path, 'r') as f:
            raw_data = yaml.safe_load(f)
        
        logger.info(f"Adapting baseline config: {yaml_path}")
        
        # Baseline config has different structure
        adapted_data = {
            'base_config': raw_data.get('base_config'),
            'baseline_methods': raw_data.get('baseline_methods', []),
            'experiment_design': raw_data.get('experiment_design', {}),
            'statistical_analysis': raw_data.get('statistical_analysis', {})
        }
        
        return adapted_data
    
    @staticmethod
    def save_flat_config(config: ExperimentConfig, output_path: Path):
        """Save Pydantic config to flat YAML format - FIXED for Pydantic v2"""
        
        # Convert to dictionary - FIXED: Use model_dump instead of dict()
        data = config.model_dump(exclude_none=True)
        
        # Save as YAML
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"Saved flat config to: {output_path}")


# Global adapter instance
adapter = LegacyYAMLAdapter()


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Main entry point for loading experiment configs (backward compatible)"""
    path = Path(config_path)
    
    # Check if it's a baseline config
    with open(path, 'r') as f:
        first_line = f.readline().strip()
        if 'baseline' in first_line.lower() or 'baseline' in path.name:
            raise ValueError(f"{path} is a baseline config, use load_baseline_config() instead")
    
    # Load as experiment config
    return adapter.adapt_experiment_config(path)


def load_baseline_config(config_path: str) -> Dict[str, Any]:
    """Load baseline comparison configuration"""
    path = Path(config_path)
    return adapter.adapt_baseline_config(path)