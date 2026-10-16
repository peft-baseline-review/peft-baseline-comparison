"""
Enhanced Training Configuration with v2.2 Scientific Rigour
Includes all fields for comprehensive training control
"""

from .base import BaseConfig
from typing import Any, Dict, List, Optional
from pydantic import Field, field_validator, ValidationInfo
from enum import Enum


class LRScheduler(str, Enum):
    """Learning rate scheduler types"""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    COSINE_WARMUP = "cosine_warmup"
    LINEAR_WARMUP = "linear_warmup"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    STEP = "step"
    EXPONENTIAL = "exponential"


class OptimizerType(str, Enum):
    """Optimizer types"""
    ADAMW = "AdamW"
    ADAM = "Adam"
    SGD = "SGD"
    ADAMW_8BIT = "AdamW8bit"  # For 8-bit optimization
    LION = "Lion"  # New optimizer
    ADAFACTOR = "Adafactor"  # For memory efficiency


class TrainingConfig(BaseConfig):
    """
    Comprehensive training configuration with scientific rigour v2.2
    
    Includes all fields needed for reproducible, efficient training
    with platform-aware optimizations.
    """
    
    # Basic training parameters
    num_epochs: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=2e-5,
        ge=1e-7,
        le=1.0,
        description="Initial learning rate"
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Weight decay (L2 regularization)"
    )
    
    # Batch configuration
    batch_size_base: int = Field(
        default=16,
        ge=1,
        le=1024,
        description="Base batch size per GPU"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of steps to accumulate gradients"
    )
    effective_batch_size: Optional[int] = Field(
        default=None,
        description="Effective batch size (auto-calculated if None)"
    )
    
    # Optimizer configuration
    optimizer: OptimizerType = Field(
        default=OptimizerType.ADAMW,
        description="Optimizer type"
    )
    betas: List[float] = Field(
        default_factory=lambda: [0.9, 0.999],
        min_items=2,
        max_items=2,
        description="Adam/AdamW beta parameters"
    )
    eps: float = Field(
        default=1e-8,
        ge=1e-12,
        le=1e-4,
        description="Adam/AdamW epsilon parameter"
    )
    
    # Gradient handling
    clip_grad: bool = Field(
        default=True,
        description="Enable gradient clipping"
    )
    gradient_clip_value: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=10.0,
        description="Gradient clipping value (None for adaptive)"
    )
    max_grad_norm: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Maximum gradient norm for clipping"
    )
    gradient_clip_algorithm: str = Field(
        default="norm",
        pattern="^(norm|value|agc)$",  # norm, value, adaptive gradient clipping
        description="Gradient clipping algorithm"
    )
    
    # Learning rate scheduling
    lr_scheduler: LRScheduler = Field(
        default=LRScheduler.LINEAR,
        description="Learning rate scheduler type"
    )
    warmup_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Warmup proportion of total training steps"
    )
    warmup_steps: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of warmup steps (overrides warmup_ratio if set)"
    )
    lr_decay_factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Learning rate decay factor for step scheduler"
    )
    lr_decay_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Steps between learning rate decays"
    )
    
    # Mixed precision training
    use_amp: bool = Field(
        default=True,
        description="Use Automatic Mixed Precision"
    )
    amp_enabled: bool = Field(
        default=True,
        description="AMP master switch (can be disabled by platform optimizations)"
    )
    amp_dtype: str = Field(
        default="float16",
        pattern="^(float16|bfloat16|float32)$",
        description="AMP compute dtype"
    )
    amp_scale_window: int = Field(
        default=1000,
        ge=10,
        description="Window for dynamic loss scaling"
    )

    use_torch_compile: bool = Field(
        default=False,
        description="Enable torch.compile for model acceleration (experimental)"
    )
    
    # Early stopping and checkpointing
    early_stopping: bool = Field(
        default=True,
        description="Enable early stopping"
    )
    patience: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Early stopping patience (epochs)"
    )
    monitor_metric: str = Field(
        default="val_accuracy",
        description="Metric to monitor for early stopping/checkpointing"
    )
    monitor_mode: str = Field(
        default="max",
        pattern="^(min|max)$",
        description="Whether to minimize or maximize the monitored metric"
    )
    min_delta: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Minimum change in monitored metric to qualify as improvement"
    )
    
    # Checkpoint configuration
    save_frequency: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Save checkpoint every N epochs"
    )
    save_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of best checkpoints to keep"
    )
    save_best_only: bool = Field(
        default=False,
        description="Only save when monitored metric improves"
    )
    checkpoint_metric: str = Field(
        default="val_accuracy",
        description="Metric to use for checkpoint selection"
    )
    
    # Validation configuration
    val_check_interval: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Validation check interval (1.0 = every epoch)"
    )
    val_frequency: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Run validation every N epochs"
    )
    
    # Reproducibility
    deterministic: bool = Field(
        default=True,
        description="Enable deterministic algorithms for reproducibility"
    )
    benchmark: bool = Field(
        default=False,
        description="Enable cuDNN benchmarking (trade-off: reproducibility vs speed)"
    )
    cudnn_deterministic: bool = Field(
        default=True,
        description="Use deterministic cuDNN algorithms"
    )
    seed_workers: bool = Field(
        default=True,
        description="Seed workers for DataLoader reproducibility"
    )
    worker_init_fn_seed: Optional[int] = Field(
        default=None,
        description="Specific seed for worker initialization"
    )
    
    # Advanced training options
    label_smoothing: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Label smoothing epsilon"
    )
    stochastic_weight_avg: bool = Field(
        default=False,
        description="Enable Stochastic Weight Averaging"
    )
    swa_lrs: Optional[float] = Field(
        default=None,
        ge=1e-6,
        le=1e-2,
        description="Learning rate for SWA"
    )
    swa_start_epoch: int = Field(
        default=10,
        ge=1,
        description="Epoch to start SWA"
    )
    swa_annealing_epochs: int = Field(
        default=10,
        ge=1,
        description="Number of annealing epochs for SWA"
    )
    
    # Gradient checkpointing (memory optimization)
    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing (trade-off: memory vs speed)"
    )
    gradient_checkpointing_every: int = Field(
        default=1,
        ge=1,
        description="Apply gradient checkpointing every N layers"
    )
    
    # Progressive PEPT specific (if applicable)
    progressive_pept: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Progressive PEPT specific training parameters"
    )

    @field_validator('effective_batch_size', mode='before')
    @classmethod
    def calculate_effective_batch_size(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        """Calculate effective batch size if not provided"""
        if v is None:
            data = info.data
            batch_size_base = data.get('batch_size_base')
            grad_accum = data.get('gradient_accumulation_steps')
            if batch_size_base is not None and grad_accum is not None:
                return batch_size_base * grad_accum
        return v

    @field_validator('warmup_steps')
    @classmethod
    def validate_warmup_steps(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        """Validate warmup steps configuration"""
        if v is not None and v < 0:
            raise ValueError("warmup_steps must be >= 0")
        return v

    @field_validator('betas')
    @classmethod
    def validate_betas(cls, v: List[float]) -> List[float]:
        """Validate beta parameters"""
        if len(v) != 2:
            raise ValueError("betas must have exactly 2 values")
        if not (0.0 <= v[0] < 1.0 and 0.0 <= v[1] < 1.0):
            raise ValueError("betas must be between 0.0 and 1.0")
        return v

    @field_validator('lr_decay_steps')
    @classmethod
    def validate_lr_decay_steps(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        """Validate learning rate decay steps"""
        if v is not None:
            num_epochs = info.data.get('num_epochs')
            if num_epochs is not None and v > num_epochs:
                raise ValueError("lr_decay_steps cannot exceed num_epochs")
        return v

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        extra = "ignore"  # Ignore extra fields to prevent validation errors