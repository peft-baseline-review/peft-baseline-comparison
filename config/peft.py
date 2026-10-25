# config/peft.py - UPDATED WITH PROGRESSIVE FIELDS and v2 validators
from .base import AllocationStrategy, BaseConfig, PEFTMethod, RankSchedule
from typing import Optional, List, Dict, Any
from pydantic import Field, field_validator, ValidationInfo
from peft import TaskType as PEFTTaskType
from enum import Enum

class ImportanceMethod(str, Enum):
    GRADIENT_NORM = "gradient_norm"
    PARAMETER_CHANGE = "parameter_change"
    LOSS_CONTRIBUTION = "loss_contribution"
    FISHER_INFORMATION = "fisher_information"

class PEFTConfig(BaseConfig):
    """Unified PEFT configuration with progressive fields"""
    method: PEFTMethod
    task_type: str = "SEQ_CLS"
    
    # Common parameters
    rank: Optional[int] = 8
    alpha: Optional[int] = 16
    dropout: Optional[float] = 0.1  # Changed from float to Optional[float]
    bias: Optional[str] = "none"  
    target_modules: Optional[List[str]] = None
    use_rank_budget: bool = False   # Disable adaptive rank scaling by default
    optimize_hyperparams: bool = True

    # --- Progressive Rank Thermostat v2.0 ---
    rank_thermostat: Optional[bool] = False
    thermostat_beta: Optional[float] = 0.0
    thermostat_temperature: Optional[float] = 1.0

    temperature: float = Field(default=1.0, ge=0.1, le=10.0, description="Softmax temperature for rank allocation")
    importance_smoothing: float = Field(default=0.99, ge=0.5, le=1.0, description="EMA factor for importance smoothing")

    ema_beta: float = Field(default=0.9, ge=0.0, le=1.0, description="Per‑parameter EMA factor")

    # Method-specific flags
    use_dora: bool = False
    lora_plus: bool = False
    total_steps: int = Field(default=10000, ge=100, le=100000)
    gradient_checkpointing: bool = False

    # AdaLora specific
    adalora_init_rank: Optional[int] = None
    adalora_target_rank: Optional[int] = None
    
    # QLoRA specific
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype_str: str = "float32"
    quantization_enabled: bool = False
    
    modules_to_not_convert: Optional[List[str]] = Field(
        default=None,
        description="List of module names to exclude from quantization (e.g., ['classifier'])"
    )

    # --- SaLoRA specific (pruning parameters) ---
    prune_frequency: Optional[int] = Field(default=None, ge=50, description="Steps between pruning operations for SaLoRA")
    prune_ratio: Optional[float] = Field(default=None, ge=0.05, le=0.8, description="Fraction of ranks to prune each time for SaLoRA")
    prune_criterion: str = Field(default="magnitude", description="Pruning criterion: magnitude, gradient, or random")
    
    # Progressive PEFT - basic
    base_method: str = "lora"
    initial_rank: int = 8
    final_rank: int = 16
    min_rank: int = Field(default=2, ge=1, le=128)
    max_rank: int = Field(default=32, ge=1, le=256)
    rank_schedule: RankSchedule = RankSchedule.ADAPTIVE
    allocation_strategy: AllocationStrategy = AllocationStrategy.THRESHOLD
    
    # Progressive PEFT - training phases
    warmup_epochs: int = 1
    cooldown_epochs: int = 1
    reset_optimizer_on_rank_change: bool = False
    
    # Progressive PEFT - importance estimation (NEW FIELDS)
    importance_method: ImportanceMethod = ImportanceMethod.GRADIENT_NORM
    importance_update_frequency: int = 25
    importance_memory_size: int = 500
    importance_threshold: float = 0.05
    
    # Progressive PEFT - allocation (NEW FIELDS)
    allocation_update_frequency: int = 50
    sparsity_target: float = 0.7
    regularization_strength: float = 0.001
    
    # Progressive PEFT - monitoring (NEW FIELDS)
    track_importance_distribution: bool = True
    save_importance_maps: bool = False
    log_frequency: int = 100
    
    # Progressive PEFT - performance (NEW FIELDS)
    use_memory_efficient_importance: bool = True
    batch_importance_computation: bool = True
    #importance_smoothing: float = 0.9
    gradient_clip_norm: float = 1.0
    
    # Virtual tokens for prefix/prompt tuning
    num_virtual_tokens: Optional[int] = None

    @field_validator('alpha')
    @classmethod
    def validate_alpha(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        """Validate alpha relative to rank"""
        if v is not None:
            rank = info.data.get('rank')
            if rank is not None and v < rank:
                print(f"⚠️ Alpha ({v}) is less than rank ({rank}). Consider setting alpha >= rank")
        return v

    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Validate task_type against PEFT's TaskType"""
        valid_types = [t.value for t in PEFTTaskType]
        if v not in valid_types:
            raise ValueError(f"task_type must be one of {valid_types}")
        return v

    @field_validator('importance_update_frequency', 'allocation_update_frequency')
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that frequency values are positive"""
        if v <= 0:
            raise ValueError(f"Frequency must be positive, got {v}")
        return v

    @field_validator('sparsity_target')
    @classmethod
    def validate_sparsity_target(cls, v: float) -> float:
        """Validate sparsity_target is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError(f"sparsity_target must be between 0 and 1, got {v}")
        return v

    @field_validator('importance_smoothing')
    @classmethod
    def validate_smoothing(cls, v: float) -> float:
        """Validate importance_smoothing is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError(f"importance_smoothing must be between 0 and 1, got {v}")
        return v