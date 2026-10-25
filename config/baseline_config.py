#!/usr/bin/env python
"""
Scientific Baseline Comparison Runner - PRODUCTION OPTIMIZED v3.2
Fixed for Pydantic v2.10+ compatibility and Windows parallel execution
"""
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent   # goes from config/ to project root

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import concurrent.futures
from dataclasses import dataclass, field
import warnings
import platform
import gc
import psutil
import signal
from functools import partial

import mlflow
import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

# Pydantic imports
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional

from .config_loader import load_config, merge_config

# Try to import torch here to avoid issues
try:
    import torch
except ImportError:
    torch = None

# ============================================================================
# CONFIGURATION CLASSES (Fixed for Pydantic v2.10+)
# ============================================================================

class BaselineScientificRole(str, Enum):
    """Scientific roles for baseline methods"""
    POSITIVE_CONTROL = "positive_control"
    NEGATIVE_CONTROL = "negative_control"
    EXPERIMENTAL = "experimental"
    EXPERIMENTAL_NOVEL = "experimental_novel"
    EFFICIENCY_BASELINE = "efficiency_baseline"
    QUANTIZATION_BASELINE = "quantization_baseline"
    ABLATION_STUDY = "ablation_study"

class BaselineMethodConfig(BaseModel):
    """Configuration for a single baseline method"""
    name: str
    description: str
    peft_method: str
    scientific_role: BaselineScientificRole = BaselineScientificRole.EXPERIMENTAL
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 1  # 1=highest, 5=lowest
    expected_compute_hours: float = Field(default=2.0, ge=0.1, le=24.0)
    
    @field_validator('name')
    def name_must_be_valid(cls, v):
        """Validate method name format"""
        if not v.replace('_', '').isalnum():
            raise ValueError('Method name must be alphanumeric with underscores')
        return v

class StatisticalAnalysisConfig(BaseModel):
    """Configuration for statistical analysis - FIXED for Pydantic v2.10"""
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)
    bootstrap_samples: int = Field(default=1000, ge=100, le=10000)
    statistical_tests: List[str] = Field(
        default_factory=lambda: ["anova", "tukey_hsd", "effect_size", "bayesian"]
    )
    
    # FIXED: Changed 'regex' to 'pattern' for Pydantic v2.10+
    multiple_comparison_correction: str = Field(
        default="holm",
        pattern="^(holm|bonferroni|fdr_bh|none)$"
    )
    
    effect_size_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"small": 0.2, "medium": 0.5, "large": 0.8}
    )
    min_samples_for_analysis: int = Field(default=2, ge=2)

class BaselineRunnerConfig(BaseModel):
    """Main configuration for baseline runner"""
    base_config: str
    baseline_methods: List[BaselineMethodConfig]
    replication_count: int = Field(default=3, ge=1, le=10)
    randomize_order: bool = True
    counterbalance: bool = False
    output_dir: str = "results/baseline_comparisons"
    
    # Resource management
    max_workers: int = Field(default=4, ge=1, le=16)
    timeout_hours_per_experiment: float = Field(default=4.0, ge=0.5, le=24.0)
    max_memory_gb: Optional[float] = Field(default=None, ge=1.0, le=128.0)
    
    # Statistical analysis
    statistical_analysis: StatisticalAnalysisConfig = Field(default_factory=StatisticalAnalysisConfig)
    
    # Platform-specific settings
    platform_optimizations: Dict[str, Any] = Field(
        default_factory=lambda: {
            "windows": {"num_workers": 0, "pin_memory": False},
            "linux": {"num_workers": 4, "pin_memory": True},
            "darwin": {"num_workers": 2, "pin_memory": False}
        }
    )
    
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }
    
    @field_validator('output_dir')
    def output_dir_must_be_absolute(cls, v):
        if not os.path.isabs(v):
            # Use the module-level PROJECT_ROOT (defined at top of file)
            return str(PROJECT_ROOT / v)
        return v
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'BaselineRunnerConfig':
        path = Path(path)
        print(f"Loading baseline config from: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        base_dir = path.parent
        print(f"Base directory: {base_dir}")
        # Load the main config with !include support
        data = load_config(path.name, base_path=base_dir)
        print(f"Loaded baseline data keys: {list(data.keys())}")

        # If the config has a 'base_config' field, load and merge it
        if 'base_config' in data:
            base_config_path = data['base_config']
            print(f"Base config relative path: {base_config_path}")
            base_config_abs = PROJECT_ROOT / base_config_path
            print(f"Base config absolute path: {base_config_abs}")
            base_data = load_config(str(base_config_abs), base_path=base_config_abs.parent)
            print(f"Loaded base data keys: {list(base_data.keys())}")
            # Merge base_data into data (base_config remains in data)
            data = merge_config(base_data, data)
            # Remove the base_config key to avoid confusion
            #data.pop('base_config', None)

        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            data = self.dict(exclude_defaults=True)
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BaselineMethod:
    """Runtime representation of a baseline method"""
    name: str
    description: str
    peft_method: str
    config_overrides: Dict[str, Any]
    scientific_role: str
    priority: int = 1
    expected_compute_hours: float = 2.0

@dataclass
class ExperimentResult:
    """Structured result from a single experiment"""
    method_name: str
    seed: int
    accuracy: float
    f1_weighted: float
    trainable_percentage: float
    trainable_params: int
    composite_score: float
    training_time_hours: float
    peak_memory_gb: float
    success: bool
    error_message: Optional[str] = None
    config_path: Optional[str] = None
    experiment_dir: Optional[str] = None
    predictions_path: Optional[str] = None   # path to saved predictions

    latency_mean_ms: float = 0.0
    peak_memory_mb: float = 0.0
    energy_per_sample_joules: float = 0.0


# ============================================================================
# TIMEOUT HANDLING - Windows Compatible
# ============================================================================

class TimeoutException(Exception):
    """Exception raised when a task times out"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout (Unix only)"""
    raise TimeoutException("Experiment timed out")

# Alternative for Windows
if platform.system() == "Windows":
    def timeout_handler(signum, frame):
        """Dummy handler for Windows"""
        pass


# ============================================================================
# STANDALONE FUNCTION FOR PARALLEL EXECUTION (Windows-compatible)
# ============================================================================

def _run_experiment(
    method: BaselineMethod,
    seed: int,
    base_config_data: dict,
    config: BaselineRunnerConfig,
    output_dir: Path
) -> ExperimentResult:
    """
    Standalone function to run a single experiment in a separate process.
    This avoids pickling issues with bound methods on Windows.
    """
    # Recreate runner inside each process
    from experiment_runner import ScientificExperimentRunner
    
    # Create method-specific configuration
    import copy
    method_config = copy.deepcopy(base_config_data)
    
    # Apply overrides
    for key_path, value in method.config_overrides.items():
        keys = key_path.split('.')
        current = method_config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    # Set seed and experiment name
    method_config['seed'] = seed
    if 'experiment_name' not in method_config:
        method_config['experiment_name'] = f"{method.name}_seed_{seed}"
    
    # Create experiment directory
    method_dir = output_dir / method.name / f"seed_{seed}"
    method_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = method_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(method_config, f, default_flow_style=False)
    
    start_time = datetime.now()
    timeout_hours = config.timeout_hours_per_experiment
    
    try:
        # Set up timeout (Unix only)
        if platform.system() != "Windows" and hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_hours * 3600))
        
        # Monitor memory before start
        start_memory = psutil.virtual_memory().used / 1e9
        
        # Run experiment
        runner = ScientificExperimentRunner(str(config_path))
        results = runner.run()
        
        # Calculate metrics
        training_time = (datetime.now() - start_time).total_seconds() / 3600
        peak_memory = (
            results["results"]
            .get("inference_metrics", {})
            .get("peak_memory_mb", 0.0) / 1024
        )
        
        # Log memory difference
        end_memory = psutil.virtual_memory().used / 1e9
        memory_diff = end_memory - start_memory
        logger.debug(f"Memory change during experiment: {memory_diff:.2f} GB")

        predictions_file = method_dir / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump({
                "labels": results["results"]["predictions"]["labels"],
                "predictions": results["results"]["predictions"]["predictions"]
            }, f)
        
        latency = results["results"]["inference_metrics"].get("latency_mean_ms", 0.0)
        memory = results["results"]["inference_metrics"].get("peak_memory_mb", 0.0)
        
        energy = results["results"]["inference_metrics"].get("energy_per_sample_joules", 0.0)

        result = ExperimentResult(
            method_name=method.name,
            seed=seed,
            accuracy=results["results"]["model_metrics"].get("accuracy", 0),
            f1_weighted=results["results"]["model_metrics"].get("f1_weighted", 0),
            trainable_percentage=results["results"]["efficiency_metrics"].get("trainable_percentage", 100),
            trainable_params=results["results"]["efficiency_metrics"].get("trainable_params", 0),
            composite_score=(
                results["results"]
                .get("experiment_report", {})
                .get("overall_scores", {})
                .get("composite_score", 0)
            ),
            training_time_hours=training_time,
            peak_memory_gb=peak_memory,
            success=True,
            config_path=str(config_path),
            experiment_dir=str(runner.experiment_dir),
            predictions_path=str(predictions_file),
            latency_mean_ms=latency,
            peak_memory_mb=memory,
            energy_per_sample_joules=energy,
        )
        
        # Cleanup
        del runner
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear timeout
        if platform.system() != "Windows" and hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        return result
        
    except TimeoutException:
        logger.error(f"⏰ Timeout: {method.name} (seed={seed}) exceeded {timeout_hours} hours")
        return ExperimentResult(
            method_name=method.name,
            seed=seed,
            accuracy=0.0,
            f1_weighted=0.0,
            trainable_percentage=0.0,
            trainable_params=0,
            composite_score=-100.0,
            training_time_hours=timeout_hours,
            peak_memory_gb=0.0,
            success=False,
            error_message=f"Timeout after {timeout_hours} hours"
        )
        
    except Exception as e:
        logger.error(f"❌ {method.name} (seed={seed}) failed: {e}")
        return ExperimentResult(
            method_name=method.name,
            seed=seed,
            accuracy=0.0,
            f1_weighted=0.0,
            trainable_percentage=0.0,
            trainable_params=0,
            composite_score=-100.0,
            training_time_hours=(datetime.now() - start_time).total_seconds() / 3600,
            peak_memory_gb=0.0,
            success=False,
            error_message=str(e)
        )


# ============================================================================
# MAIN RUNNER CLASS
# ============================================================================

class ScientificBaselineRunner:
    """
    Production-optimized baseline comparison runner
    Now with Windows compatibility and Pydantic v2.10+ fixes
    """
    
    def __init__(
        self,
        baseline_config: BaselineRunnerConfig,
        random_seeds: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
        enable_mlflow: bool = True,
        enable_monitoring: bool = True
    ):
        """Initialize with enhanced configuration"""
        self.config = baseline_config
        
        # Platform detection
        self.platform = platform.system().lower()
        self.is_windows = self.platform == "windows"
        
        self.platform_optimizations = baseline_config.platform_optimizations.get(
            self.platform, 
            baseline_config.platform_optimizations.get("linux", {})
        )
        
        # Resolve project root
        try:
            self.project_root = Path(__file__).resolve().parent.parent
        except Exception:
            self.project_root = Path.cwd()
        
        # Build output directory with timestamp
        raw_output = output_dir if output_dir else baseline_config.output_dir
        output_path = Path(raw_output)
        if not output_path.is_absolute():
            output_path = self.project_root / output_path
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = (output_path / timestamp).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds
        if random_seeds is None:
            self.random_seeds = list(range(42, 42 + baseline_config.replication_count))
        else:
            self.random_seeds = random_seeds[:baseline_config.replication_count]
        
        # Load base configuration with fallback
        self.base_config_data = self._load_base_config_with_fallback()
        
        # Convert to runtime methods with priority sorting
        self.baseline_methods = self._initialize_methods_with_priority()
        
        # Initialize monitoring
        self.enable_monitoring = enable_monitoring
        self.monitoring_data = []
        
        # Initialize MLflow
        self.mlflow_enabled = enable_mlflow and self._setup_mlflow()
        
        # Results storage
        self.results: Dict[str, List[ExperimentResult]] = {}
        self.comparison_results = {}
        self.failed_experiments = []
        
        # Resource tracking
        self.total_compute_hours_estimated = sum(
            m.expected_compute_hours * baseline_config.replication_count 
            for m in self.baseline_methods
        )
        
        logger.info("=" * 80)
        logger.info(f"🔬 SCIENTIFIC BASELINE RUNNER v3.2 (Platform: {self.platform})")
        logger.info("=" * 80)
        logger.info(f"📁 Output: {self.output_dir}")
        logger.info(f"🧪 Methods: {len(self.baseline_methods)} (priority sorted)")
        logger.info(f"🔁 Replications: {baseline_config.replication_count}")
        logger.info(f"⏱️ Estimated compute: {self.total_compute_hours_estimated:.1f} hours")
        logger.info(f"📊 Statistical tests: {baseline_config.statistical_analysis.statistical_tests}")
        logger.info(f"💻 Platform optimizations: {self.platform_optimizations}")
        
        # Save configuration
        self._save_configuration()
    
    def _load_base_config_with_fallback(self) -> Dict[str, Any]:
        """Load base configuration and apply platform optimizations from YAML."""
        from .config_loader import load_config

        base_config_path = Path(self.config.base_config)
        if not base_config_path.is_absolute():
            base_config_path = self.project_root / base_config_path

        if not base_config_path.exists():
            # Search in common locations
            search_paths = [
                Path("configs/experiments") / base_config_path.name,
                Path("configs/experiments") / f"{base_config_path.stem}.yaml",
                base_config_path.with_suffix('.yaml'),
                Path.cwd() / "configs" / base_config_path.name,
            ]
            for path in search_paths:
                abs_path = self.project_root / path if not path.is_absolute() else path
                if abs_path.exists():
                    base_config_path = abs_path
                    logger.info(f"📁 Found base config at: {base_config_path}")
                    break
            else:
                raise FileNotFoundError(
                    f"Base config not found: {self.config.base_config}\n"
                    f"Searched in: {[str(p) for p in search_paths]}"
                )

        # Load with custom loader to resolve !include tags
        config_data = load_config(str(base_config_path), base_path=base_config_path.parent)

        # Apply platform-specific optimizations from YAML (safe dictionary access)
        if 'task' in config_data:
            config_data['task']['num_workers'] = self.platform_optimizations.get('num_workers', 0)
            config_data['task']['pin_memory'] = self.platform_optimizations.get('pin_memory', False)
        if 'training' in config_data:
            config_data['training']['use_amp'] = self.platform_optimizations.get('use_amp', True)
            config_data['training']['gradient_accumulation_steps'] = self.platform_optimizations.get('gradient_accumulation_steps', 1)

        logger.info(f"Applied platform optimizations: {self.platform_optimizations}")
        return config_data
    
    def _initialize_methods_with_priority(self) -> List[BaselineMethod]:
        """Initialize methods sorted by priority (highest first)"""
        methods = []
        for method_config in self.config.baseline_methods:
            method = BaselineMethod(
                name=method_config.name,
                description=method_config.description,
                peft_method=method_config.peft_method,
                config_overrides=method_config.config_overrides,
                scientific_role=method_config.scientific_role.value,
                priority=method_config.priority,
                expected_compute_hours=method_config.expected_compute_hours
            )
            methods.append(method)
        
        # Sort by priority (highest first), then by expected compute time (shortest first)
        methods.sort(key=lambda m: (m.priority, m.expected_compute_hours))
        
        logger.info(f"📋 Method priority order:")
        for i, method in enumerate(methods, 1):
            logger.info(f"  {i:2d}. {method.name} (priority={method.priority}, est={method.expected_compute_hours}h)")
        
        return methods
    
    def _load_predictions(self, path: str):
        """Load labels and predictions from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return np.array(data['labels']), np.array(data['predictions'])
    
    def _setup_mlflow(self) -> bool:
        """Setup MLflow with error handling"""
        try:
            mlflow.set_experiment("baseline_comparisons")
            mlflow.start_run(run_name=f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            mlflow.log_params({
                "replication_count": self.config.replication_count,
                "methods_count": len(self.baseline_methods),
                "platform": self.platform,
                "random_seeds": str(self.random_seeds)
            })
            
            logger.info("📊 MLflow initialized")
            return True
        except Exception as e:
            logger.warning(f"⚠️ MLflow initialization failed: {e}")
            return False
    
    def _save_configuration(self):
        """Save complete configuration for reproducibility"""
        config_save_path = self.output_dir / "runner_configuration.json"
        
        config_dict = {
            "baseline_config": self.config.dict(),
            "random_seeds": self.random_seeds,
            "platform": self.platform,
            "platform_optimizations": self.platform_optimizations,
            "timestamp": datetime.now().isoformat(),
            "git_info": self._get_git_info(),
            "environment": self._get_environment_info()
        }
        
        with open(config_save_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"💾 Configuration saved: {config_save_path}")
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git information for reproducibility"""
        try:
            import subprocess
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            commit_hash = result.stdout.strip()
            result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=True)
            has_uncommitted = len(result.stdout.strip()) > 0
            branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
            return {
                "commit_hash": commit_hash,
                "has_uncommitted_changes": has_uncommitted,
                "branch": branch
            }
        except Exception:
            return {"error": "Git not available"}
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information"""
        try:
            import torch
            torch_info = {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            }
        except ImportError:
            torch_info = {"error": "Torch not available"}
        
        return {
            "python_version": sys.version,
            **torch_info,
            "cpu_count": os.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / 1e9,
            "platform": platform.platform()
        }
    
    def _is_dominated(self, a, b):
        """Return True if a is dominated by b (b is better or equal in all objectives)."""
        return all(a[obj] <= b[obj] for obj in a.keys()) and any(a[obj] < b[obj] for obj in a.keys())

    def _pareto_frontier(self, points):
        """Return list of indices of non‑dominated points."""
        n = len(points)
        is_dominated = [False] * n
        for i in range(n):
            for j in range(n):
                if i != j and self._is_dominated(points[i], points[j]):
                    is_dominated[i] = True
                    break
        return [i for i in range(n) if not is_dominated[i]]

    def _hypervolume_4d(self, points, ref_point=(0,0,0,0), seed=42):
        """
        Approximate hypervolume via Monte Carlo sampling.
        Seeded for reproducibility: identical frontier inputs produce
        identical HV output on every run.
        """
        import random
        rng = random.Random(seed)
        samples = 100000
        ndims = len(ref_point)
        count = 0
        for _ in range(samples):
            p = [rng.uniform(0, 1) for _ in range(ndims)]
            dominated = False
            for fp in points:
                if all(p[d] <= fp[d] for d in range(ndims)):
                    dominated = True
                    break
            if dominated:
                count += 1
        return count / samples  # volume of dominated region
    
    def calculate_statistical_power(self, pilot_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Calculate statistical power with pilot data or defaults.
        """
        try:
            from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
            
            if pilot_data and 'effect_size' in pilot_data:
                effect_size_d = float(pilot_data['effect_size'])
                effect_size_f = effect_size_d / 2.0
            else:
                effect_size_d = 0.5
                effect_size_f = effect_size_d / 2.0
            
            alpha = 0.05
            power_target = 0.8
            n_groups = len(self.baseline_methods)
            n_per_group = self.config.replication_count
            
            power_analysis = TTestIndPower()
            achieved_power = power_analysis.power(
                effect_size=effect_size_d,
                nobs1=n_per_group,
                alpha=alpha,
                ratio=1.0
            )
            required_n = power_analysis.solve_power(
                effect_size=effect_size_d,
                power=power_target,
                alpha=alpha,
                ratio=1.0
            )
            
            if n_groups >= 2:
                try:
                    total_nobs = n_per_group * n_groups
                    anova_power_val = FTestAnovaPower().power(
                        effect_size=effect_size_f,
                        nobs=total_nobs,
                        alpha=alpha,
                        k_groups=n_groups
                    )
                except Exception as e:
                    logger.warning(f"ANOVA power calculation failed: {e}. Using t‑test power as fallback.")
                    anova_power_val = achieved_power
            else:
                anova_power_val = float('nan')
                        
            power_results = {
                "achieved_power": float(achieved_power),
                "anova_power": float(anova_power_val),
                "effect_size_d": float(effect_size_d),
                "effect_size_f": float(effect_size_f),
                "alpha": alpha,
                "target_power": power_target,
                "current_sample_per_group": n_per_group,
                "required_sample_per_group": int(np.ceil(required_n)),
                "adequate_power": bool(achieved_power >= power_target),
                "recommendation": (
                    "Increase replication count" if achieved_power < power_target else "Adequate"
                )
            }
            
            power_path = self.output_dir / "statistical_power_analysis.json"
            with open(power_path, 'w') as f:
                json.dump(power_results, f, indent=2, default=str)
            
            logger.info("📊 Statistical Power Analysis:")
            logger.info(f"  Achieved power: {achieved_power:.3f} ({'✅ Adequate' if achieved_power >= 0.8 else '⚠️ Low'})")
            logger.info(f"  Required replications: {int(np.ceil(required_n))}")
            logger.info(f"  Current replications: {n_per_group}")
            
            return power_results
        except ImportError:
            logger.warning("statsmodels not available for power analysis")
            return {"error": "statsmodels not available"}
        except Exception as e:
            logger.warning(f"Power analysis failed: {e}")
            return {"error": str(e)}
    
    def run_comparison(
        self, 
        parallel: bool = True,
        adaptive_scheduling: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete baseline comparison with adaptive resource management
        """
        logger.info("🚀 Starting systematic baseline comparison...")
        
        # Calculate statistical power
        power_analysis = self.calculate_statistical_power()
        
        # Adaptive scheduling based on available resources
        if adaptive_scheduling:
            max_workers = self._calculate_adaptive_workers()
        else:
            max_workers = self.config.max_workers
        
        logger.info(f"🔧 Adaptive scheduling: {max_workers} workers")
        
        # Run experiments
        if parallel and max_workers > 1:
            results = self._run_parallel(max_workers)
        else:
            results = self._run_sequential()
        
        # Store results
        self.results = results
        
        # Perform comprehensive statistical analysis
        self._perform_statistical_analysis()
        
        # Generate reports and visualizations
        self._generate_reports()
        
        # Log to MLflow if enabled
        if self.mlflow_enabled:
            self._log_to_mlflow(power_analysis)
        
        logger.info("✅ Baseline comparison completed successfully!")
        
        return {
            "results": self.results,
            "comparison": self.comparison_results,
            "power_analysis": power_analysis,
            "summary": self._generate_summary()
        }
    
    def _calculate_adaptive_workers(self) -> int:
        """Calculate optimal number of workers based on available resources"""
        try:
            cpu_count = os.cpu_count() or 4
            memory_gb = psutil.virtual_memory().total / 1e9
            
            if memory_gb < 16:
                workers = 1
            elif memory_gb < 32:
                workers = min(2, cpu_count // 2)
            elif memory_gb < 64:
                workers = min(4, cpu_count // 2)
            else:
                workers = min(self.config.max_workers, cpu_count - 2)
            
            if self.is_windows:
                workers = max(1, workers // 2)
            
            logger.info(f"💻 Resource check: CPU={cpu_count}, RAM={memory_gb:.1f}GB -> {workers} workers")
            return workers
        except Exception:
            return min(2, self.config.max_workers)
    
    def _run_parallel(self, max_workers: int) -> Dict[str, List[ExperimentResult]]:
        """Run experiments in parallel using standalone function."""
        all_results = {}
        tasks = [(method, seed) for method in self.baseline_methods for seed in self.random_seeds]
        
        # Prepare partial function with fixed arguments
        experiment_func = partial(
            _run_experiment,
            base_config_data=self.base_config_data,
            config=self.config,
            output_dir=self.output_dir
        )
        
        logger.info(f"🔀 Running {len(tasks)} experiments with {max_workers} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for method, seed in tasks:
                future = executor.submit(experiment_func, method, seed)
                future_to_task[future] = (method.name, seed)
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                method_name, seed = future_to_task[future]
                try:
                    result = future.result(timeout=300)
                    if method_name not in all_results:
                        all_results[method_name] = []
                    all_results[method_name].append(result)
                    completed += 1
                    status = "✅" if result.success else "❌"
                    logger.info(
                        f"  {status} [{completed}/{len(tasks)}] {method_name} (seed={seed}): "
                        f"Acc={result.accuracy:.4f}, Time={result.training_time_hours:.2f}h"
                    )
                except concurrent.futures.TimeoutError:
                    logger.error(f"⏰ Result timeout: {method_name} (seed={seed})")
                except Exception as e:
                    logger.error(f"❌ Failed to get result for {method_name}: {e}")
        
        return all_results
    
    def _run_sequential(self) -> Dict[str, List[ExperimentResult]]:
        """Run experiments sequentially (safer for Windows)."""
        all_results = {}
        total_experiments = len(self.baseline_methods) * len(self.random_seeds)
        completed = 0
        
        for method in self.baseline_methods:
            method_results = []
            for seed in self.random_seeds:
                result = self._run_single_experiment_local(method, seed)
                method_results.append(result)
                completed += 1
                status = "✅" if result.success else "❌"
                logger.info(
                    f"  {status} [{completed}/{total_experiments}] {method.name} (seed={seed}): "
                    f"Acc={result.accuracy:.4f}, Time={result.training_time_hours:.2f}h"
                )
            all_results[method.name] = method_results
        
        return all_results
    
    def _run_single_experiment_local(self, method: BaselineMethod, seed: int) -> ExperimentResult:
        """
        Local version for sequential runs (uses the same process).
        """
        # Delegate to standalone function with same process
        return _run_experiment(method, seed, self.base_config_data, self.config, self.output_dir)
    
    def _perform_statistical_analysis(self):
        """Perform comprehensive statistical analysis"""
        analysis_results = {}
        performance_data = {}
        
        for method_name, results in self.results.items():
            successful = [r for r in results if r.success]
            if successful:
                accuracies = [r.accuracy for r in successful]
                f1s = [r.f1_weighted for r in successful]
                latencies = [r.latency_mean_ms for r in successful]
                energies = [r.energy_per_sample_joules for r in successful]
                params_m = [r.trainable_params / 1e6 for r in successful]   # millions

                # Compute means and CIs (using t‑distribution, fallback to mean if n<2)
                def mean_ci(vals):
                    if len(vals) >= 2:
                        m = np.mean(vals)
                        sem = stats.sem(vals)
                        ci = stats.t.interval(0.95, len(vals)-1, loc=m, scale=sem)
                        return m, (ci[1] - ci[0]) / 2
                    else:
                        m = np.mean(vals) if vals else 0
                        return m, 0

                acc_mean, acc_ci = mean_ci(accuracies)
                f1_mean, f1_ci = mean_ci(f1s)
                lat_mean, lat_ci = mean_ci(latencies)
                eng_mean, eng_ci = mean_ci(energies)
                param_mean, _ = mean_ci(params_m)   # no CI for params (they are fixed per method)

                performance_data[method_name] = {
                    "accuracy_mean": float(acc_mean),
                    "accuracy_ci_width": float(acc_ci),
                    "f1_mean": float(f1_mean),
                    "f1_ci_width": float(f1_ci),
                    "latency_mean_ms": float(lat_mean),
                    "latency_ci_width": float(lat_ci),
                    "energy_mean_j": float(eng_mean),
                    "energy_ci_width": float(eng_ci),
                    "params_millions": float(param_mean),
                    "efficiency_mean": float(np.mean([r.trainable_percentage for r in successful])),
                    "composite_mean": float(np.mean([r.composite_score for r in successful])),
                    "n_success": len(successful),
                    "success_rate": len(successful) / len(results)
                }
        
        analysis_results["performance_summary"] = performance_data
        self.comparison_results = analysis_results 

        # ------------------------------------------------------------
        # Pairwise statistical tests
        # ------------------------------------------------------------
        method_names = list(self.results.keys())
        if len(method_names) < 2:
            logger.info("Less than 2 methods, skipping pairwise tests.")
            return

        # Build seed‑indexed accuracy dictionaries for proper pairing
        method_acc_by_seed = {}
        for method_name, results in self.results.items():
            acc_by_seed = {}
            for r in results:
                if r.success:
                    acc_by_seed[r.seed] = r.accuracy
            method_acc_by_seed[method_name] = acc_by_seed

        # Wilcoxon signed‑rank test (seed‑aligned)
        wilcoxon_results = []
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                m1, m2 = method_names[i], method_names[j]
                common_seeds = sorted(set(method_acc_by_seed[m1].keys()) & set(method_acc_by_seed[m2].keys()))
                if len(common_seeds) >= 2:
                    vals1 = [method_acc_by_seed[m1][s] for s in common_seeds]
                    vals2 = [method_acc_by_seed[m2][s] for s in common_seeds]
                    stat, p = wilcoxon(vals1, vals2, alternative='two-sided')
                    wilcoxon_results.append({
                        'method1': m1,
                        'method2': m2,
                        'statistic': float(stat),
                        'p_value': float(p),
                        'test': 'wilcoxon'
                    })
                else:
                    logger.warning(f"Cannot perform Wilcoxon for {m1} vs {m2}: insufficient common successful seeds.")

        # McNemar's test (unchanged, already seed‑aware)
        mcnemar_results = []
        for i in range(len(method_names)):
            for j in range(i+1, len(method_names)):
                m1, m2 = method_names[i], method_names[j]
                p_values = []
                for seed in self.random_seeds:
                    res1 = next((r for r in self.results[m1] if r.seed == seed and r.success and r.predictions_path), None)
                    res2 = next((r for r in self.results[m2] if r.seed == seed and r.success and r.predictions_path), None)
                    if res1 is None or res2 is None:
                        continue
                    y_true1, y_pred1 = self._load_predictions(res1.predictions_path)
                    y_true2, y_pred2 = self._load_predictions(res2.predictions_path)
                    if not np.array_equal(y_true1, y_true2):
                        logger.warning(f"Label mismatch between {m1} and {m2} for seed {seed}")
                        continue
                    a = np.sum((y_pred1 == y_true1) & (y_pred2 == y_true2))
                    b = np.sum((y_pred1 == y_true1) & (y_pred2 != y_true2))
                    c = np.sum((y_pred1 != y_true1) & (y_pred2 == y_true2))
                    d = np.sum((y_pred1 != y_true1) & (y_pred2 != y_true2))
                    table = [[a, b], [c, d]]
                    if b + c == 0:
                        # Perfect agreement: no evidence of difference.
                        # Do not call mcnemar() — the continuity-corrected chi2 divides
                        # by zero and produces a spurious 0.0 (or NaN) that corrupts
                        # the median.
                        p = 1.0
                    else:
                        result = mcnemar(table, exact=False, correction=True)
                        p = float(result.pvalue)
                    p_values.append(p)
                if p_values:
                    median_p = float(np.median(p_values))
                    mcnemar_results.append({
                        'method1': m1,
                        'method2': m2,
                        'p_value_median': median_p,
                        'p_values_per_seed': p_values,
                        'test': 'mcnemar',
                        'n_seeds_with_discordance': sum(1 for p in p_values if p < 1.0),
                    })

        # Combine p-values for Holm correction (rest unchanged)
        all_pairs = wilcoxon_results + mcnemar_results
        if all_pairs:
            pvals = [res['p_value'] for res in wilcoxon_results] + [res['p_value_median'] for res in mcnemar_results]
            reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='holm')
            for i, res in enumerate(wilcoxon_results):
                res['p_value_corrected'] = pvals_corrected[i]
                res['significant'] = bool(reject[i])
            offset = len(wilcoxon_results)
            for i, res in enumerate(mcnemar_results):
                res['p_value_corrected'] = pvals_corrected[offset + i]
                res['significant'] = bool(reject[offset + i])

        # Store in comparison_results
        self.comparison_results['pairwise_tests'] = {
            'wilcoxon': wilcoxon_results,
            'mcnemar': mcnemar_results
        }

        # ------------------------------------------------------------
        # Pareto Frontier Analysis (true multi‑objective)
        # ------------------------------------------------------------
        if len(method_names) >= 2:
            # Collect raw objectives for each method (mean across seeds)
            objectives_raw = {}
            for name in method_names:
                # Get mean values from successful runs
                acc = np.mean([r.accuracy for r in self.results[name] if r.success])
                params = np.mean([r.trainable_params for r in self.results[name] if r.success])
                latency = np.mean([r.latency_mean_ms for r in self.results[name] if r.success])
                memory = np.mean([r.peak_memory_mb for r in self.results[name] if r.success])

                # Compute efficiencies (higher is better)
                param_eff = 1.0 / params if params > 0 else 0.0
                speed = 1.0 / latency if latency > 0 else 0.0
                mem_eff = 1.0 / memory if memory > 0 else 0.0

                objectives_raw[name] = {
                    'accuracy': acc,
                    'param_efficiency': param_eff,
                    'speed': speed,
                    'memory_efficiency': mem_eff
                }

            # Normalize each objective to [0,1] across methods
            objectives_norm = {}
            obj_names = ['accuracy', 'param_efficiency', 'speed', 'memory_efficiency']
            for obj in obj_names:
                vals = [objectives_raw[name][obj] for name in method_names]
                min_val, max_val = min(vals), max(vals)
                if max_val > min_val:
                    for name in method_names:
                        if name not in objectives_norm:
                            objectives_norm[name] = {}
                        objectives_norm[name][obj] = (objectives_raw[name][obj] - min_val) / (max_val - min_val)
                else:
                    # All values equal
                    for name in method_names:
                        if name not in objectives_norm:
                            objectives_norm[name] = {}
                        objectives_norm[name][obj] = 0.5

            # Build list of normalized vectors
            vectors = [list(objectives_norm[name].values()) for name in method_names]

            # Pareto dominance test
            n = len(method_names)
            is_dominated = [False] * n
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i is dominated by j
                    if all(vectors[i][k] <= vectors[j][k] for k in range(4)) and any(vectors[i][k] < vectors[j][k] for k in range(4)):
                        is_dominated[i] = True
                        break

            frontier_indices = [i for i in range(n) if not is_dominated[i]]
            frontier_methods = [method_names[i] for i in frontier_indices]

            # Approximate hypervolume using Monte Carlo (seeded for
            # determinism — matches _hypervolume_4d's convention).
            import random
            rng = random.Random(42)
            samples = 100000
            count = 0
            for _ in range(samples):
                p = [rng.uniform(0, 1) for _ in range(4)]
                dominated = False
                for idx in frontier_indices:
                    fp = vectors[idx]
                    if all(p[k] <= fp[k] for k in range(4)):
                        dominated = True
                        break
                if dominated:
                    count += 1
            hypervolume = count / samples

            # Store results
            self.comparison_results['pareto_frontier'] = {
                'frontier_methods': frontier_methods,
                'hypervolume': hypervolume,
                'normalized_objectives': objectives_norm,
                'raw_objectives': objectives_raw
            }

        # Save analysis to file (existing code)
        analysis_path = self.output_dir / "statistical_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(self.comparison_results, f, indent=2, default=str)
        logger.info(f"📊 Statistical analysis saved: {analysis_path}")
    
    def _generate_reports(self):
        """Generate comprehensive reports""" 

        summary_data = []
        for method_name, data in self.comparison_results.get("performance_summary", {}).items():
            summary_data.append({
                "Method": method_name,
                "Accuracy": f"{data['accuracy_mean']:.4f} ± {data['accuracy_ci_width']:.4f}",
                "F1 (weighted)": f"{data['f1_mean']:.4f} ± {data['f1_ci_width']:.4f}",
                "Latency (ms)": f"{data['latency_mean_ms']:.2f} ± {data['latency_ci_width']:.2f}",
                "Energy (J)": f"{data['energy_mean_j']:.3f} ± {data['energy_ci_width']:.3f}",
                "Params (M)": f"{data['params_millions']:.2f}",
                "Efficiency (%)": f"{data['efficiency_mean']:.2f}",
                "Composite": f"{data['composite_mean']:.4f}",
                "N": data['n_success'],
                "Success Rate": f"{data['success_rate']*100:.0f}%"
            })
        df_summary = pd.DataFrame(summary_data)

        csv_path = self.output_dir / "baseline_summary.csv"
        df_summary.to_csv(csv_path, index=False)
        
        raw_path = self.output_dir / "raw_results.json"
        with open(raw_path, 'w') as f:
            raw_results = {}
            for method_name, results in self.results.items():
                raw_results[method_name] = [r.__dict__ for r in results]
            json.dump(raw_results, f, indent=2, default=str)
        
        self._generate_markdown_report(df_summary)
        self._create_visualizations()
        
        logger.info(f"📄 Reports saved to {self.output_dir}")
    
    def _generate_markdown_report(self, df_summary: pd.DataFrame):
        """Generate comprehensive markdown report with full performance table."""
        total_experiments = sum(len(results) for results in self.results.values())
        successful_experiments = sum(
            sum(1 for r in results if r.success) 
            for results in self.results.values()
        )
        overall_success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0

        # Base report
        report = f"""# Scientific Baseline Comparison Report

    ## Overview
    - **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - **Platform**: {self.platform}
    - **Methods**: {len(self.baseline_methods)}
    - **Replications**: {self.config.replication_count}
    - **Total experiments**: {total_experiments}
    - **Successful experiments**: {successful_experiments}
    - **Overall success rate**: {overall_success_rate*100:.1f}%

    ## Performance Summary

    {df_summary.to_markdown(index=False)}
    """

        # Pairwise Statistical Comparisons (if available)
        if 'pairwise_tests' in self.comparison_results:
            report += "\n## Pairwise Statistical Comparisons\n\n"
            pt = self.comparison_results['pairwise_tests']

            if pt.get('wilcoxon'):
                report += "### Wilcoxon Signed‑Rank Test (Accuracy)\n\n"
                report += "| Method 1 | Method 2 | Statistic | p‑value | Corrected p | Significant |\n"
                report += "|----------|----------|-----------|---------|-------------|-------------|\n"
                for res in pt['wilcoxon']:
                    sig = "✅" if res['significant'] else "❌"
                    report += f"| {res['method1']} | {res['method2']} | {res['statistic']:.3f} | {res['p_value']:.4f} | {res['p_value_corrected']:.4f} | {sig} |\n"

            if pt.get('mcnemar'):
                report += "\n### McNemar's Test (Prediction Agreement)\n\n"
                report += "| Method 1 | Method 2 | Median p‑value | Corrected p | Significant |\n"
                report += "|----------|----------|----------------|-------------|-------------|\n"
                for res in pt['mcnemar']:
                    sig = "✅" if res['significant'] else "❌"
                    report += f"| {res['method1']} | {res['method2']} | {res['p_value_median']:.4f} | {res['p_value_corrected']:.4f} | {sig} |\n"

        # Pareto Frontier Analysis (if available)
        if 'pareto_frontier' in self.comparison_results:
            pf = self.comparison_results['pareto_frontier']
            report += "\n## Pareto Frontier Analysis\n\n"
            report += f"- **Hypervolume** (4‑dimensional): {pf['hypervolume']:.4f}\n"
            report += "- **Pareto‑optimal methods**: " + ", ".join(pf['frontier_methods']) + "\n\n"
            report += "Normalized objectives (higher is better):\n\n"
            report += "| Method | Accuracy | Param Efficiency | Speed | Memory Efficiency |\n"
            report += "|--------|----------|------------------|-------|-------------------|\n"
            norm = pf['normalized_objectives']
            for name in norm:
                n = norm[name]
                report += f"| {name} | {n['accuracy']:.3f} | {n['param_efficiency']:.3f} | {n['speed']:.3f} | {n['memory_efficiency']:.3f} |\n"

        # Experimental Design
        report += """
    ## Experimental Design

    ### Baseline Methods
    """
        for method in self.baseline_methods:
            report += f"""
    #### {method.name}
    - **Scientific Role**: {method.scientific_role}
    - **Description**: {method.description}
    - **PEFT Method**: {method.peft_method}
    """

        report += f"""
    ### Configuration Details
    - **Random seeds**: {self.random_seeds}
    - **Output directory**: {self.output_dir}
    - **Statistical tests**: {', '.join(self.config.statistical_analysis.statistical_tests)}

    ---

    *Generated by Scientific Baseline Comparison Runner v3.2*
    *Platform: {self.platform}*
    *Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
    """
        with open(self.output_dir / "baseline_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _create_visualizations(self):
        """Create a comprehensive visualization suite for baseline comparison (Q1‑grade)."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import gaussian_kde
            import pandas as pd

            # Set publication‑ready style
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_context("paper", font_scale=1.2)
            sns.set_palette("husl")

            # Prepare data: seed‑indexed accuracy and efficiency
            method_names = list(self.results.keys())
            if len(method_names) < 2:
                logger.warning("Less than two methods, skipping most visualizations.")
                return

            # Build seed‑indexed dictionaries for accurate pairing
            acc_by_seed = {}
            eff_by_seed = {}
            comp_by_seed = {}
            for m in method_names:
                acc_by_seed[m] = {r.seed: r.accuracy for r in self.results[m] if r.success}
                eff_by_seed[m] = {r.seed: r.trainable_percentage for r in self.results[m] if r.success}
                comp_by_seed[m] = {r.seed: r.composite_score for r in self.results[m] if r.success}

            # Compute per‑method means and stds for summary plots
            acc_means = [np.mean(list(acc_by_seed[m].values())) for m in method_names]
            acc_stds = [np.std(list(acc_by_seed[m].values())) if len(acc_by_seed[m]) > 1 else 0 for m in method_names]
            eff_means = [np.mean(list(eff_by_seed[m].values())) for m in method_names]
            comp_means = [np.mean(list(comp_by_seed[m].values())) for m in method_names]
            comp_stds = [np.std(list(comp_by_seed[m].values())) if len(comp_by_seed[m]) > 1 else 0 for m in method_names]

            # -------------------------------
            # 1. Accuracy Bar Plot with 95% CI
            # -------------------------------
            fig, ax = plt.subplots(figsize=(10, 6))
            ci_half = []
            for m in method_names:
                vals = list(acc_by_seed[m].values())
                if len(vals) >= 2:
                    ci = stats.t.interval(0.95, len(vals)-1, loc=np.mean(vals), scale=stats.sem(vals))
                    ci_half.append((ci[1] - ci[0]) / 2)
                else:
                    ci_half.append(0)
            x_pos = np.arange(len(method_names))
            bars = ax.bar(x_pos, acc_means, yerr=ci_half, capsize=5, alpha=0.7,
                        color='steelblue', ecolor='black', error_kw={'linewidth': 1.5})
            ax.set_xlabel('Method')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Comparison with 95% Confidence Intervals')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / "accuracy_barplot.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "accuracy_barplot.pdf", bbox_inches='tight')
            plt.close()

            # -------------------------------
            # 2. Boxplot of Accuracies with Individual Points
            # -------------------------------
            fig, ax = plt.subplots(figsize=(10, 6))
            data_to_plot = [list(acc_by_seed[m].values()) for m in method_names]
            bp = ax.boxplot(data_to_plot, labels=method_names, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', linewidth=1.5),
                            medianprops=dict(color='red', linewidth=2),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5))
            # Overlay individual points with jitter
            for i, data in enumerate(data_to_plot):
                x = np.random.normal(i+1, 0.05, size=len(data))
                ax.plot(x, data, 'o', color='black', alpha=0.6, markersize=5)
            ax.set_xlabel('Method')
            ax.set_ylabel('Accuracy')
            ax.set_title('Distribution of Accuracy Across Seeds')
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / "accuracy_boxplot.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "accuracy_boxplot.pdf", bbox_inches='tight')
            plt.close()

            # -------------------------------
            # 3. Paired Difference Plot (vs first method)
            # -------------------------------
            baseline = method_names[0]  # you may change to a specific method, e.g., "lora"
            if baseline in acc_by_seed:
                fig, ax = plt.subplots(figsize=(10, 6))
                diff_data = []
                labels = []
                baseline_vals = acc_by_seed[baseline]
                for m in method_names[1:]:
                    common_seeds = sorted(set(baseline_vals.keys()) & set(acc_by_seed[m].keys()))
                    if len(common_seeds) >= 2:
                        diffs = [acc_by_seed[m][s] - baseline_vals[s] for s in common_seeds]
                        diff_data.append(diffs)
                        labels.append(m)
                if diff_data:
                    bp = ax.boxplot(diff_data, labels=labels, patch_artist=True,
                                    boxprops=dict(facecolor='lightgreen', linewidth=1.5),
                                    medianprops=dict(color='red', linewidth=2))
                    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
                    ax.set_xlabel('Method')
                    ax.set_ylabel(f'Accuracy Difference vs {baseline}')
                    ax.set_title(f'Paired Accuracy Difference from {baseline}')
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "paired_differences.png", dpi=300, bbox_inches='tight')
                    plt.savefig(self.output_dir / "paired_differences.pdf", bbox_inches='tight')
                    plt.close()

            # -------------------------------
            # 4. P‑Value Heatmap with Significance Stars
            # -------------------------------
            if 'pairwise_tests' in self.comparison_results:
                pt = self.comparison_results['pairwise_tests']
                # Build matrix of corrected p-values
                pval_matrix = pd.DataFrame(index=method_names, columns=method_names, dtype=float)
                for res in pt.get('wilcoxon', []):
                    m1, m2 = res['method1'], res['method2']
                    p = res['p_value_corrected']
                    pval_matrix.loc[m1, m2] = p
                    pval_matrix.loc[m2, m1] = p
                for res in pt.get('mcnemar', []):
                    m1, m2 = res['method1'], res['method2']
                    p = res['p_value_corrected']
                    pval_matrix.loc[m1, m2] = p
                    pval_matrix.loc[m2, m1] = p
                # Fill diagonal with 1.0
                for m in method_names:
                    pval_matrix.loc[m, m] = 1.0

                # Drop rows/cols with all NaNs (if any method missing all comparisons)
                pval_matrix = pval_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)

                if not pval_matrix.empty and not pval_matrix.isnull().all().all():
                    # Function to add significance stars
                    def star(p):
                        if p < 0.001:
                            return '***'
                        elif p < 0.01:
                            return '**'
                        elif p < 0.05:
                            return '*'
                        else:
                            return 'n.s.'

                    # Create annotation matrix with p‑value and star
                    annot_matrix = pval_matrix.applymap(lambda p: f"{p:.3f}\n{star(p)}" if not pd.isna(p) else "")
                    fig, ax = plt.subplots(figsize=(9, 7))
                    sns.heatmap(pval_matrix, annot=annot_matrix, fmt='', cmap='Reds_r',
                                cbar_kws={'label': 'Corrected p-value'},
                                square=True, linewidths=0.5, linecolor='gray',
                                mask=(pval_matrix > 0.05),  # only show significant ones? remove if you want all
                                ax=ax)
                    ax.set_title('Pairwise Statistical Significance (Holm-corrected p-values)')
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "pvalue_heatmap.png", dpi=300, bbox_inches='tight')
                    plt.savefig(self.output_dir / "pvalue_heatmap.pdf", bbox_inches='tight')
                    plt.close()

            # -------------------------------
            # 5. Efficiency Scatter (Accuracy vs Trainable %)
            # -------------------------------
            fig, ax = plt.subplots(figsize=(10, 6))
            for m in method_names:
                x = np.mean(list(eff_by_seed[m].values()))
                y = np.mean(list(acc_by_seed[m].values()))
                ax.scatter(x, y, s=100, label=m, alpha=0.7, edgecolors='black', linewidth=1)
                ax.text(x, y, m, fontsize=9, ha='center', va='bottom')
                # Add error bars (mean ± std)
                xerr = np.std(list(eff_by_seed[m].values())) if len(eff_by_seed[m]) > 1 else 0
                yerr = np.std(list(acc_by_seed[m].values())) if len(acc_by_seed[m]) > 1 else 0
                ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', ecolor='gray', capsize=3, alpha=0.5)
            ax.set_xlabel('Trainable Parameters (%)')
            ax.set_ylabel('Accuracy')
            ax.set_title('Efficiency vs. Performance')
            ax.grid(True, alpha=0.3)
            # Add legend (might be crowded; you can remove if labels are on points)
            # ax.legend(loc='best', fontsize=8)
            plt.tight_layout()
            plt.savefig(self.output_dir / "efficiency_scatter.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "efficiency_scatter.pdf", bbox_inches='tight')
            plt.close()

            # -------------------------------
            # 6. Composite Score Horizontal Bar Plot
            # -------------------------------
            fig, ax = plt.subplots(figsize=(8, 6))
            y_pos = np.arange(len(method_names))
            ax.barh(y_pos, comp_means, xerr=comp_stds, capsize=3, color='coral', alpha=0.8,
                    error_kw={'linewidth': 1.5, 'ecolor': 'black'})
            ax.set_yticks(y_pos)
            ax.set_yticklabels(method_names)
            ax.set_xlabel('Composite Score')
            ax.set_title('Composite Score Comparison')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(self.output_dir / "composite_barplot.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "composite_barplot.pdf", bbox_inches='tight')
            plt.close()

            # -------------------------------
            # 7. Parallel Coordinates Plot (Pareto Frontier)
            # -------------------------------
            if 'pareto_frontier' in self.comparison_results:
                pf = self.comparison_results['pareto_frontier']
                norm = pf['normalized_objectives']
                df = pd.DataFrame.from_dict(norm, orient='index')
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Method'}, inplace=True)
                # Mark frontier methods
                df['Pareto'] = df['Method'].apply(lambda x: 'Frontier' if x in pf['frontier_methods'] else 'Dominated')

                fig, ax = plt.subplots(figsize=(10, 6))
                parallel_coordinates(df, class_column='Pareto', cols=['accuracy', 'param_efficiency', 'speed', 'memory_efficiency'],
                                    color=['red', 'gray'], ax=ax, linewidth=2, alpha=0.7)
                ax.set_xticklabels(['Accuracy', 'Param Efficiency', 'Speed', 'Memory Efficiency'])
                ax.set_ylabel('Normalized Value (higher is better)')
                ax.set_title('Pareto Frontier – Parallel Coordinates')
                ax.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(self.output_dir / "pareto_parallel.png", dpi=300, bbox_inches='tight')
                plt.savefig(self.output_dir / "pareto_parallel.pdf", bbox_inches='tight')
                plt.close()

            # -------------------------------
            # 8. Inference Time Comparison (Latency)
            # -------------------------------
            fig, ax = plt.subplots(figsize=(10, 6))
            latency_means = []
            latency_cis = []
            for m in method_names:
                vals = [r.latency_mean_ms for r in self.results[m] if r.success]
                if len(vals) >= 2:
                    mean = np.mean(vals)
                    sem = stats.sem(vals)
                    ci = stats.t.interval(0.95, len(vals)-1, loc=mean, scale=sem)
                    ci_half = (ci[1] - ci[0]) / 2
                else:
                    mean = np.mean(vals) if vals else 0
                    ci_half = 0
                latency_means.append(mean)
                latency_cis.append(ci_half)

            x_pos = np.arange(len(method_names))
            bars = ax.bar(x_pos, latency_means, yerr=latency_cis, capsize=5, alpha=0.7,
                        color='teal', ecolor='black', error_kw={'linewidth': 1.5})
            ax.set_xlabel('Method')
            ax.set_ylabel('Inference Time (ms)')
            ax.set_title('Inference Latency Comparison with 95% Confidence Intervals')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / "latency_barplot.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "latency_barplot.pdf", bbox_inches='tight')
            plt.close()

            # -------------------------------
            # 8. Energy vs. Accuracy Scatter
            # -------------------------------
            fig, ax = plt.subplots(figsize=(10, 6))
            for m in method_names:
                # Collect energy and accuracy for each seed
                x_vals = [r.energy_per_sample_joules for r in self.results[m] if r.success]
                y_vals = [r.accuracy for r in self.results[m] if r.success]
                if x_vals and y_vals:
                    ax.scatter(x_vals, y_vals, s=60, label=m, alpha=0.7, edgecolors='black', linewidth=1)
                    # Optionally add mean point
                    ax.scatter(np.mean(x_vals), np.mean(y_vals), s=150, marker='D', edgecolors='black', linewidth=2, color='none')
            ax.set_xlabel('Energy per Sample (J)')
            ax.set_ylabel('Accuracy')
            ax.set_title('Energy Efficiency vs. Performance')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "energy_scatter.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / "energy_scatter.pdf", bbox_inches='tight')
            plt.close()

            logger.info(f"📊 All visualizations saved to {self.output_dir}")

        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    def _log_to_mlflow(self, power_analysis: Dict[str, Any]):
        """Log results to MLflow"""
        if not self.mlflow_enabled:
            return
        try:
            mlflow.log_metric("statistical_power", power_analysis.get("achieved_power", 0))
            for method_name, results in self.results.items():
                success_results = [r for r in results if r.success]
                if success_results:
                    mean_acc = np.mean([r.accuracy for r in success_results])
                    mlflow.log_metric(f"{method_name}_accuracy", mean_acc)
                    mlflow.log_metric(f"{method_name}_composite_score", np.mean([r.composite_score for r in success_results]))
            mlflow.log_artifact(str(self.output_dir))
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_experiments = sum(len(r) for r in self.results.values())
        successful = sum(sum(1 for r in results if r.success) for results in self.results.values())
        
        training_times = []
        for results in self.results.values():
            for r in results:
                if r.success:
                    training_times.append(r.training_time_hours)
        avg_time = np.mean(training_times) if training_times else 0
        
        return {
            "total_experiments": total_experiments,
            "successful_experiments": successful,
            "success_rate": successful / total_experiments if total_experiments > 0 else 0,
            "average_training_time_hours": avg_time,
            "output_dir": str(self.output_dir),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# SETUP LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baseline_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_baseline_comparison_from_config(
    config_path: str,
    output_dir: Optional[str] = None,
    parallel: bool = True,
    seeds: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run baseline comparison from config file
    """
    config = BaselineRunnerConfig.from_yaml(config_path)
    runner = ScientificBaselineRunner(
        baseline_config=config,
        random_seeds=seeds,
        output_dir=output_dir,
        enable_mlflow=True,
        enable_monitoring=True
    )
    results = runner.run_comparison(parallel=parallel)
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run scientific baseline comparison")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially (no parallel)")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seeds")
    
    args = parser.parse_args()
    
    seed_list = [int(s.strip()) for s in args.seeds.split(',')]
    
    try:
        results = run_baseline_comparison_from_config(
            config_path=args.config,
            output_dir=args.output_dir,
            parallel=not args.sequential,
            seeds=seed_list
        )
        print(f"\n{'='*80}")
        print(f"✅ BASELINE COMPARISON COMPLETED!")
        print(f"{'='*80}")
        print(f"📊 Success rate: {results['summary']['success_rate']*100:.1f}%")
        print(f"📁 Results saved to: {results['summary'].get('output_dir', 'N/A')}")
        print(f"📄 Reports available in the output directory")
    except Exception as e:
        print(f"\n❌ BASELINE COMPARISON FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)