# config/task.py - UPDATED VERSION with Pydantic v2 validators
from .base import BaseConfig, TaskType
from typing import List, Optional
from pydantic import Field, field_validator, ValidationInfo
from enum import Enum

class DelimiterType(str, Enum):
    COMMA = ","
    SEMICOLON = ";"
    TAB = "\t"
    PIPE = "|"

class TaskConfig(BaseConfig):
    """Task-specific configuration with dataset format options"""
    task_type: TaskType
    num_labels: int
    label_names: List[str]
    
    # Data paths
    data_dir: str
    train_file: str
    val_file: Optional[str] = None
    test_file: str
    
    # Dataset format
    delimiter: DelimiterType = DelimiterType.COMMA
    text_column: str = "text"
    label_column: str = "label"
    
    # Text preprocessing
    use_cleaned_text: bool = False  # For news dataset with cleaned_text column
    use_translit_text: bool = False  # For news dataset with translit_text column
    
    # Data processing
    max_length: int = 128
    batch_size: int = 32
    
    # Dataset properties
    use_class_weights: bool = True
    imbalanced: bool = False
    val_split: float = 0.1
    
    # Performance
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    @field_validator('text_column')
    @classmethod
    def validate_text_column(cls, v: str, info: ValidationInfo) -> str:
        """Set appropriate text column based on task type"""
        data = info.data
        if data.get('use_cleaned_text'):
            return "cleaned_text"
        elif data.get('use_translit_text'):
            return "translit_text"
        return v
    
    @property
    def is_classification(self) -> bool:
        return self.task_type in [TaskType.SENTIMENT, TaskType.NEWS]
    
    @property
    def delimiter_str(self) -> str:
        """Get delimiter as string"""
        return self.delimiter.value