from .base import BaseConfig
from typing import List, Dict, Any, Optional, Union
from pydantic import Field, field_validator
from enum import Enum


class EvaluationConfig(BaseConfig):
    """
    Production-ready evaluation configuration with v2.2 enhancements.
    
    This version includes all fields needed for the enhanced evaluator
    while maintaining backward compatibility.
    """
    
    # Core metrics
    metrics: List[str] = Field(
        default_factory=lambda: ["accuracy", "f1_weighted", "f1_macro"],
        description="Metrics to compute"
    )

    #energy metrics
    measure_power: bool = Field(
        default=False,
        description="If True, attempt real power measurement via NVML (requires GPU and pynvml). Falls back to theoretical on failure."
    )
    
    # Confidence intervals
    compute_ci: bool = Field(default=False)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.999)
    bootstrap_samples: int = Field(default=1000, ge=100, le=10000)

    #energy parameters
    carbon_intensity_kg_per_kwh: float = 0.233          # global average (kg CO₂/kWh)
    energy_per_flop_pj: float = 20.0                    # picojoules per FLOP (arithmetic only)
    memory_overhead_factor: float = 2.5                  # accounts for DRAM energy
    enable_power_measurement: bool = True                # attempt NVML on Linux
    
    # Statistical analysis (NEW FIELDS for v2.2)
    min_samples_for_analysis: int = Field(
        default=10,
        ge=5,
        description="Minimum samples for statistical analysis"
    )
    multiple_comparison_correction: str = Field(
        default="holm",
        description="Method for multiple comparison correction"
    )
    power_analysis_alpha: float = Field(
        default=0.05,
        ge=0.001,
        le=0.1,
        description="Alpha level for power analysis"
    )
    power_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="Threshold for adequate statistical power"
    )
    
    # Statistical tests
    statistical_tests: List[str] = Field(
        default_factory=lambda: ["t_test", "anova"],
        description="Statistical tests to perform"
    )
    
    # Effect sizes
    effect_size_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "small": 0.2,
            "medium": 0.5,
            "large": 0.8
        },
        description="Effect size thresholds"
    )
    
    # Composite score
    composite_alpha: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Weight for efficiency in composite score"
    )
    
    # Inference measurement (NEW FIELDS for v2.2)
    inference_batch_size: int = Field(
        default=1,
        ge=1,
        le=128,
        description="Batch size for inference measurements"
    )
    num_inference_runs: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of inference runs"
    )
    measure_energy: bool = Field(
        default=False,
        description="Measure energy consumption"
    )
    measure_memory: bool = Field(
        default=True,
        description="Measure memory usage"
    )
    
    # Reporting
    export_formats: List[str] = Field(
        default_factory=lambda: ["json", "md", "txt"],
        description="Formats for exporting reports"
    )
    
    @field_validator('multiple_comparison_correction')
    def validate_correction_method(cls, v):
        valid_methods = ['none', 'bonferroni', 'holm', 'benjamini_hochberg', 'holm_bonferroni']
        if v.lower() not in valid_methods:
            raise ValueError(f"Multiple comparison correction must be one of: {valid_methods}")
        return v.lower()
    
    class Config:
        extra = "ignore"  # Important: Ignores extra fields without error